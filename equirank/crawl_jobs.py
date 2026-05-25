# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Pont entre le serveur FastAPI (equirank/server.py) et le crawler FFE
(ffe_crawler/core/runner.py).

On gère ici en RAM :
  • startJob(...)              lance un worker thread, renvoie un job_id
  • getJobSnapshot(job_id, cur) renvoie l'état + nouveaux events
  • getJobDatasetCsv(job_id)   sérialise le dataset final en CSV
  • writeJobToData(job_id)     écrit dans data/dataset_brut_v2.csv
                               puis chaîne adapt_dataset_v2 + train.py
                               et recharge le predictor en mémoire
  • fetchFilterOptions(url)    parse les 4 dropdowns du calendrier FFE
                               (région / département / discipline /
                               championnat) pour les exposer côté UI
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import time
import uuid
import threading
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


_ROOT = Path(__file__).resolve().parent.parent          # PFE/

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ffe_crawler.core.runner          import runFfeChain
from ffe_crawler.core.ffe_chain       import SUPPORTED_DISCIPLINES, DEFAULT_ENTRY_URL
from ffe_crawler.core.dataset_builder import buildDatasetBrutV2, DATASET_COLUMNS


# ──────────────────────────────────────────────
# État des jobs
# ──────────────────────────────────────────────
@dataclass
class _Job:
    job_id:        str
    state:         str = 'pending'           # pending|running|done|error
    events:        list[dict] = field(default_factory=list)
    error:         str = ''
    dataset_rows:  list[dict] = field(default_factory=list)
    total_rows:    int = 0
    total_urls:    int = 0
    started_at:    float = field(default_factory=time.time)
    finished_at:   float | None = None
    params:        dict[str, Any] = field(default_factory=dict)


_jobs:      dict[str, _Job] = {}
_jobs_lock: threading.Lock  = threading.Lock()


# ──────────────────────────────────────────────
# Parsing des dropdowns du calendrier FFE
# ──────────────────────────────────────────────
# Labels visibles que Telemat emploie en tête de chaque dropdown — sert
# à détecter quel dropdown on lit (l'ordre des dropdowns peut bouger).
_DROPDOWN_LABELS = {
    'region':      ('toutes régions',),
    'departement': ('tous départements',),
    'discipline':  ('toutes disciplines',),
    'championnat': ('championnats',),       # "Championnats : tous concours"
}


_USER_AGENT_REAL = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0.0.0 Safari/537.36'
)


def _fallbackOptions() -> dict[str, Any]:
    """
    Réponse "dégradée" quand FFE Telemat ne répond pas : on expose au
    moins les disciplines hardcodées et l'URL d'entrée par défaut.
    Le frontend reçoit un objet bien formé et n'est pas obligé d'afficher
    une erreur — l'utilisateur peut tout de même lancer un crawl global.
    """
    return {
        'source_url':  DEFAULT_ENTRY_URL,
        'default_url': DEFAULT_ENTRY_URL,
        'disciplines': list(SUPPORTED_DISCIPLINES),
        'degraded':    True,
        'filters': {
            'region':      {'options': [{'label': 'Toutes régions',      'url': DEFAULT_ENTRY_URL}], 'selected_label': ''},
            'departement': {'options': [{'label': 'Tous départements',   'url': DEFAULT_ENTRY_URL}], 'selected_label': ''},
            'discipline':  {'options': [{'label': d, 'url': DEFAULT_ENTRY_URL} for d in SUPPORTED_DISCIPLINES],
                            'selected_label': 'Toutes disciplines'},
            'championnat': {'options': [{'label': 'Championnats : tous concours', 'url': DEFAULT_ENTRY_URL}], 'selected_label': ''},
        },
    }


def fetchFilterOptions(url: str | None = None) -> dict[str, Any]:
    """
    Parse les 4 dropdowns du calendrier FFE Telemat et renvoie pour
    chacun : son label "all" actuel + la liste des options
    (label, url absolue).

    Si `url` est None, on part de DEFAULT_ENTRY_URL — pratique pour le
    premier appel. Quand l'utilisateur choisit une option, le frontend
    relance fetchFilterOptions(NEW_URL) pour cascader les choix (par
    ex. après avoir choisi une région, la liste des départements se
    restreint à cette région).

    Si FFE Telemat est injoignable / rate-limit / contenu inattendu,
    on retombe sur `_fallbackOptions()` (dégradé mais utilisable) plutôt
    que de lever une exception. Le frontend n'est pas obligé de bloquer.
    """
    target = url or DEFAULT_ENTRY_URL

    # ── Fetch avec retry sur 429 et timeout (max 3 tentatives) ──
    headers = {
        'User-Agent':      _USER_AGENT_REAL,
        'Accept-Language': 'fr-FR,fr;q=0.9',
    }
    # Proxy optionnel (Tor / proxy rotatif) — partage la même conf que
    # le crawler principal. Sans ça, /api/crawl/filters partait toujours
    # depuis l'IP locale et tombait sur 429 même quand le reste du
    # crawler passait par Tor.
    from ffe_crawler.engine.crawler import _get_proxy_dict
    proxies = _get_proxy_dict()
    if proxies:
        print(f'[filters] proxy actif : {next(iter(proxies.values()))}', flush = True)
    resp = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(target, headers = headers, proxies = proxies, timeout = 15)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                time.sleep(min(wait, 10))
                continue
            resp.raise_for_status()
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5)
            continue

    if resp is None or resp.status_code != 200:
        # FFE injoignable après 3 tries → fallback dégradé. On log mais
        # on ne lève pas : le frontend a au moins les disciplines.
        msg = f'FFE Telemat inaccessible ({last_exc or resp.status_code if resp else "no response"}) — fallback dégradé.'
        print(f'[filters] {msg}', flush = True)
        return _fallbackOptions()

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Le dropdown qui nous intéresse est dans le 3e form (deb/fin/walias).
    # On reste tolérant si la structure change : on cherche tous les
    # dropdown-menu de la page, puis on identifie chacun par le label
    # de son premier item.
    out: dict[str, dict] = {
        'region':      {'options': [], 'selected_label': ''},
        'departement': {'options': [], 'selected_label': ''},
        'discipline':  {'options': [], 'selected_label': ''},
        'championnat': {'options': [], 'selected_label': ''},
    }

    for menu in soup.find_all('div', class_ = 'dropdown-menu'):
        items = menu.find_all('a', class_ = 'dropdown-item')
        if not items:
            continue
        firstText = items[0].get_text(' ', strip = True).lower()

        kind: str | None = None
        for k, prefixes in _DROPDOWN_LABELS.items():
            if any(firstText.startswith(p) for p in prefixes):
                kind = k
                break
        if kind is None:
            continue

        out[kind]['options'] = [
            {
                'label': a.get_text(' ', strip = True),
                'url':   urljoin(target, a.get('href', '')),
            }
            for a in items
        ]

    # Bouton actif de chaque dropdown (= valeur "selected") — on essaie
    # de récupérer le texte du <button.dropdown-toggle> qui précède le
    # menu, sinon on retombe sur la 1re option "Toutes …".
    for kind in out.keys():
        if out[kind]['options']:
            out[kind].setdefault('selected_label', out[kind]['options'][0]['label'])

    # Garde-fou : si aucun dropdown n'a été trouvé (FFE a changé sa
    # structure HTML, ou la page nous renvoie une modal de login),
    # on retombe sur le fallback dégradé pour que l'UI reste utilisable.
    if not any(out[k]['options'] for k in out):
        print('[filters] aucun dropdown FFE détecté — fallback dégradé.', flush = True)
        return _fallbackOptions()

    return {
        'source_url':    target,
        'default_url':   DEFAULT_ENTRY_URL,
        'disciplines':   list(SUPPORTED_DISCIPLINES),
        'degraded':      False,
        'filters':       out,
    }


# ──────────────────────────────────────────────
# Lifecycle des jobs
# ──────────────────────────────────────────────
def listDisciplines() -> list[str]:
    return list(SUPPORTED_DISCIPLINES)


def startJob(
    deb:        str,
    fin:        str,
    discipline: str,
    username:   str | None = None,
    password:   str | None = None,
    entry_url:  str | None = None,
) -> str:
    if discipline not in SUPPORTED_DISCIPLINES:
        raise ValueError(
            f"Discipline {discipline!r} non supportée — voir /api/crawl/disciplines."
        )

    # Garde-fou : ce crawler sert à entraîner — on rejette les dates
    # futures (concours sans résultats). Pour les dates à venir, l'utilisateur
    # doit passer par /crawl-predict qui ne touche pas au modèle.
    today = time.strftime('%Y-%m-%d')
    if fin > today:
        raise ValueError(
            f"date de fin ({fin}) postérieure à aujourd'hui — utiliser /crawl-predict "
            f"pour prédire les résultats des concours à venir."
        )

    job = _Job(
        job_id = uuid.uuid4().hex,
        state  = 'running',
        params = {
            'deb':        deb,
            'fin':        fin,
            'discipline': discipline,
            'with_auth':  bool(username and password),
            'entry_url':  entry_url or DEFAULT_ENTRY_URL,
        },
    )
    with _jobs_lock:
        _jobs[job.job_id] = job

    thread = threading.Thread(
        target = _runWorker,
        args   = (job.job_id, deb, fin, discipline, username, password, entry_url),
        daemon = True,
    )
    thread.start()
    return job.job_id


def _appendEvent(job_id: str, evt: dict) -> None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is not None:
            j.events.append(evt)


def _runWorker(
    job_id:     str,
    deb:        str,
    fin:        str,
    discipline: str,
    username:   str | None,
    password:   str | None,
    entry_url:  str | None,
) -> None:
    def publish(evt: dict, _sid: str) -> None:
        _appendEvent(job_id, evt)

    try:
        result = runFfeChain(
            deb        = deb,
            fin        = fin,
            discipline = discipline,
            username   = username or None,
            password   = password or None,
            publish    = publish,
            entry_url  = entry_url or None,
        )
        rows = buildDatasetBrutV2(step_rows = result.step_rows)

        _appendEvent(job_id, {
            'type':  'ffe_done',
            'rows':  len(rows),
            'total': result.total_rows,
        })

        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is not None:
                j.state         = 'done'
                j.dataset_rows  = rows
                j.total_rows    = result.total_rows
                j.total_urls    = result.total_urls
                j.finished_at   = time.time()

    except Exception as exc:
        import traceback
        msg = f'{exc}'
        _appendEvent(job_id, {
            'type':    'error',
            'message': msg,
            'trace':   traceback.format_exc(),
        })
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is not None:
                j.state       = 'error'
                j.error       = msg
                j.finished_at = time.time()


def getJobSnapshot(job_id: str, cursor: int = 0) -> dict[str, Any]:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            return {'state': 'unknown', 'cursor': cursor, 'events': []}
        new_events = j.events[cursor:]
        return {
            'job_id':       j.job_id,
            'state':        j.state,
            'cursor':       len(j.events),
            'events':       new_events,
            'error':        j.error,
            'dataset_rows': len(j.dataset_rows),
            'total_rows':   j.total_rows,
            'total_urls':   j.total_urls,
            'params':       j.params,
        }


def getJobDatasetCsv(job_id: str) -> bytes | None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None or j.state != 'done':
            return None
        rows = list(j.dataset_rows)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames = list(DATASET_COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    return ('﻿' + buf.getvalue()).encode('utf-8')


def writeJobToData(job_id: str) -> Path | None:
    """Écrit le dataset brut_v2 dans data/ sans toucher au modèle."""
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None or j.state != 'done':
            return None
        rows = list(j.dataset_rows)

    target = _ROOT / 'data' / 'dataset_brut_v2.csv'
    target.parent.mkdir(parents = True, exist_ok = True)
    with target.open('w', encoding = 'utf-8-sig', newline = '') as fp:
        writer = csv.DictWriter(fp, fieldnames = list(DATASET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return target


# ──────────────────────────────────────────────
# Pipeline post-crawl complet : dataset → adapt → train → reload
# ──────────────────────────────────────────────
@dataclass
class _PipelineJob:
    pipeline_id: str
    state:       str = 'running'              # running|done|error
    logs:        list[str] = field(default_factory=list)
    error:       str = ''
    artifacts:   dict[str, Any] = field(default_factory=dict)


_pipelines:      dict[str, _PipelineJob] = {}
_pipelines_lock: threading.Lock          = threading.Lock()


def _pipelineLog(pid: str, line: str) -> None:
    with _pipelines_lock:
        p = _pipelines.get(pid)
        if p is not None:
            p.logs.append(f'[{time.strftime("%H:%M:%S")}] {line}')


def startTrainingPipeline(job_id: str) -> str:
    """
    Enchaîne, en arrière-plan : écriture data/dataset_brut_v2.csv →
    adapt_dataset_v2 → train.py → reload du predictor en mémoire.

    Renvoie un pipeline_id à poller via getPipelineSnapshot().
    """
    # Garde-fou : le crawl doit être terminé avec un dataset non vide
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            raise ValueError(f'Job {job_id} introuvable')
        if j.state != 'done':
            raise ValueError(f'Job {job_id} pas encore terminé (state={j.state})')
        if not j.dataset_rows:
            raise ValueError(f'Job {job_id} : dataset vide, rien à entraîner')

    pid = uuid.uuid4().hex[:12]
    with _pipelines_lock:
        _pipelines[pid] = _PipelineJob(pipeline_id = pid)

    threading.Thread(
        target = _runPipelineWorker,
        args   = (pid, job_id),
        daemon = True,
    ).start()
    return pid


def _runPipelineWorker(pid: str, job_id: str) -> None:
    try:
        # 1. Écriture dataset_brut_v2
        _pipelineLog(pid, 'Écriture data/dataset_brut_v2.csv …')
        target = writeJobToData(job_id)
        if target is None:
            raise RuntimeError('writeJobToData a renvoyé None — job inconnu/inachevé.')
        _pipelineLog(pid, f'  OK → {target}')

        # 2. adapt_dataset_v2 (en subprocess pour ne pas pourrir l'import-state)
        _pipelineLog(pid, 'Lancement adapt_dataset_v2.py …')
        adapt = _runScript(['adapt_dataset_v2.py',
                            '--input',  str(target),
                            '--output', str(_ROOT / 'data' / 'dataset.csv')])
        _pipelineLog(pid, _truncTail(adapt, 1200))
        if adapt['returncode'] != 0:
            raise RuntimeError('adapt_dataset_v2 a échoué — voir log.')

        # 3. train.py — LightGBM (défaut du predictor)
        _pipelineLog(pid, 'Lancement train.py --model LightGBM --no-eda --no-cv …')
        train = _runScript(['train.py',
                            '--data',  str(_ROOT / 'data' / 'dataset.csv'),
                            '--model', 'LightGBM',
                            '--no-eda',
                            '--no-cv'])
        _pipelineLog(pid, _truncTail(train, 1500))
        if train['returncode'] != 0:
            raise RuntimeError('train.py a échoué — voir log.')

        # 4. Recharger le predictor en mémoire pour que la page /
        # voie le nouveau modèle sans avoir à redémarrer uvicorn.
        _pipelineLog(pid, 'Reload du predictor en mémoire …')
        try:
            from . import predictor as _pred
            _pred._default_predictor = None  # le prochain appel le recrée
            _pipelineLog(pid, '  OK — le nouveau modèle sera servi au prochain /api/predict')
        except Exception as exc:  # pragma: no cover — best effort
            _pipelineLog(pid, f'  ⚠ reload échoué : {exc}')

        # 5. Snapshot automatique : on archive le dataset + les modèles
        # qui viennent d'être produits, pour que l'utilisateur puisse
        # plus tard revenir à cet état ou comparer plusieurs versions.
        try:
            from . import snapshots as _snaps
            params = _jobs[job_id].params if job_id in _jobs else {}
            label_bits = []
            if params.get('discipline'): label_bits.append(params['discipline'])
            if params.get('deb') and params.get('fin'):
                label_bits.append(f"{params['deb']}→{params['fin']}")
            label = 'Crawl ' + ' · '.join(label_bits) if label_bits else f'Crawl {job_id[:8]}'
            snap = _snaps.archiveCurrentDataset(
                label       = label,
                job_id      = job_id,
                description = f'Snapshot automatique après pipeline d\'entraînement (pid={pid}).',
            )
            _pipelineLog(pid, f'📦 Snapshot archivé : {snap.snapshot_id} ({snap.label})')
        except Exception as exc:
            _pipelineLog(pid, f'  ⚠ archivage snapshot échoué : {exc}')

        with _pipelines_lock:
            p = _pipelines.get(pid)
            if p is not None:
                p.state = 'done'
                p.artifacts = {
                    'dataset_brut_v2': str(target),
                    'dataset_csv':     str(_ROOT / 'data' / 'dataset.csv'),
                    'models_dir':      str(_ROOT / 'models'),
                }

    except Exception as exc:
        _pipelineLog(pid, f'✗ {exc}')
        with _pipelines_lock:
            p = _pipelines.get(pid)
            if p is not None:
                p.state = 'error'
                p.error = str(exc)


def _runScript(argv: list[str]) -> dict[str, Any]:
    """
    Lance un script Python via le venv courant (sys.executable). On
    fusionne stdout+stderr car les scripts PFE impriment en mode
    "trace" sur les deux flux.
    """
    full_cmd = [sys.executable, *[a if Path(a).is_absolute() or not a.endswith('.py')
                                   else str(_ROOT / a) for a in argv]]
    proc = subprocess.run(
        full_cmd,
        cwd            = str(_ROOT),
        capture_output = True,
        text           = True,
        encoding       = 'utf-8',
        errors         = 'replace',
    )
    return {
        'returncode': proc.returncode,
        'stdout':     proc.stdout or '',
        'stderr':     proc.stderr or '',
    }


def _truncTail(result: dict, max_chars: int) -> str:
    """Concatène stdout+stderr en gardant les `max_chars` derniers caractères."""
    blob = (result.get('stdout', '') + '\n' + result.get('stderr', '')).strip()
    if len(blob) <= max_chars:
        return blob
    return '… (sortie tronquée) …\n' + blob[-max_chars:]


def getPipelineSnapshot(pid: str) -> dict[str, Any]:
    with _pipelines_lock:
        p = _pipelines.get(pid)
        if p is None:
            return {'state': 'unknown', 'logs': [], 'error': ''}
        return {
            'pipeline_id': p.pipeline_id,
            'state':       p.state,
            'logs':        list(p.logs),
            'error':       p.error,
            'artifacts':   dict(p.artifacts),
        }
