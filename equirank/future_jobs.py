# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Gestion des crawls FUTURS et stockage des prédictions associées.

Différences avec equirank/crawl_jobs.py :
  • on n'a PAS d'authentification (pas de s5)
  • on n'écrit PAS dataset_brut_v2.csv et on NE relance PAS l'entraînement
  • dès que le crawl finit, on prédit chaque engagement via
    `EquirankPredictor.predictBatch` et on stocke le résultat en RAM
  • on garde un index par concours (s2_Numéro) pour la page /predictions

Le résultat est navigable depuis :
  /predictions                  → liste des concours crawlés
  /predictions/{concours_id}    → détail d'un concours (table des partants)
  /api/predictions/search?q=... → recherche cheval / cavalier / n° concours
"""

from __future__ import annotations

import csv
import io
import sys
import time
import uuid
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Any

_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ffe_crawler.core.runner          import runFfeChainFuture
from ffe_crawler.core.ffe_chain       import SUPPORTED_DISCIPLINES, DEFAULT_ENTRY_URL
from ffe_crawler.core.dataset_builder import buildDatasetFuture, DATASET_COLUMNS

from .predictor          import getDefaultPredictor
from .                   import prediction_history


# Colonnes additionnelles dans le CSV de prédictions (ordre stable)
PREDICTION_EXTRA_COLUMNS: tuple[str, ...] = (
    'classement_predit', 'proba', 'verdict', 'is_cold_start',
    'model_name', 'note',
    'cheval_courses', 'cheval_victoires',
    'jockey_courses', 'jockey_victoires',
    'duo_courses',    'duo_victoires',
    'forme',
)


def _addClassementPredit(predictions: list[dict]) -> None:
    """
    Calcule le rang prédit de chaque participant DANS SON ÉPREUVE
    (concours_id + épreuve). 1 = meilleure proba. Les cold-start
    sont rangés derrière les prédictions "fortes". Mute la liste en
    place pour ajouter la clé `classement_predit` à chaque row.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in predictions:
        key = (str(r.get('s2_Numéro', '')), str(r.get('s3_Épreuve', '')))
        groups[key].append(r)

    for rows in groups.values():
        # Tri stable par (cold-start asc, proba desc). On considère les
        # rows sans proba (string vide) comme proba=-1 pour finir en fond.
        def _sortKey(r: dict) -> tuple[int, float]:
            isCold = 1 if r.get('is_cold_start') else 0
            try:
                p = float(r.get('proba')) if r.get('proba') != '' else -1.0
            except (TypeError, ValueError):
                p = -1.0
            return (isCold, -p)
        rows.sort(key = _sortKey)
        for rank, r in enumerate(rows, start = 1):
            r['classement_predit'] = rank


@dataclass
class _FutureJob:
    job_id:        str
    state:         str = 'pending'             # pending|running|predicting|done|error
    events:        list[dict] = field(default_factory=list)
    error:         str = ''
    raw_rows:      list[dict] = field(default_factory=list)   # dataset_brut_v2 style
    predictions:   list[dict] = field(default_factory=list)   # raw_rows + proba/verdict
    by_concours:   dict[str, list[dict]] = field(default_factory=dict)
    total_rows:    int = 0
    total_urls:    int = 0
    started_at:    float = field(default_factory=time.time)
    finished_at:   float | None = None
    params:        dict[str, Any] = field(default_factory=dict)


_jobs:      dict[str, _FutureJob] = {}
_jobs_lock: threading.Lock        = threading.Lock()


# ──────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────
def startFutureJob(
    deb:        str,
    fin:        str,
    discipline: str,
    entry_url:  str | None = None,
    username:   str | None = None,
    password:   str | None = None,
) -> str:
    if discipline not in SUPPORTED_DISCIPLINES:
        raise ValueError(
            f"Discipline {discipline!r} non supportée — voir /api/crawl/disciplines."
        )

    # Garde-fou inverse de crawl_jobs.startJob : on n'accepte que des
    # dates strictement futures (concours pas encore disputés). Si
    # l'utilisateur veut entraîner sur du passé, il a /crawl pour ça.
    today = time.strftime('%Y-%m-%d')
    if deb <= today:
        raise ValueError(
            f"date de début ({deb}) ≤ aujourd'hui ({today}) — "
            f"utiliser /crawl pour les concours déjà disputés."
        )

    job = _FutureJob(
        job_id = uuid.uuid4().hex,
        state  = 'running',
        params = {
            'deb':        deb,
            'fin':        fin,
            'discipline': discipline,
            'entry_url':  entry_url or DEFAULT_ENTRY_URL,
            'with_auth':  bool(username and password),
        },
    )
    with _jobs_lock:
        _jobs[job.job_id] = job

    threading.Thread(
        target = _runWorker,
        args   = (job.job_id, deb, fin, discipline, entry_url, username, password),
        daemon = True,
    ).start()
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
    entry_url:  str | None,
    username:   str | None,
    password:   str | None,
) -> None:
    def publish(evt: dict, _sid: str) -> None:
        _appendEvent(job_id, evt)

    try:
        result = runFfeChainFuture(
            deb        = deb,
            fin        = fin,
            discipline = discipline,
            username   = username or None,
            password   = password or None,
            publish    = publish,
            entry_url  = entry_url or None,
        )
        rows = buildDatasetFuture(step_rows = result.step_rows)
        _appendEvent(job_id, {
            'type':  'ffe_done',
            'rows':  len(rows),
            'total': result.total_rows,
        })

        # ── Étape prédiction ──────────────────────────────────────
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is not None:
                j.state      = 'predicting'
                j.raw_rows   = rows
                j.total_rows = result.total_rows
                j.total_urls = result.total_urls

        _appendEvent(job_id, {
            'type':    'predicting',
            'message': f'Prédiction de {len(rows)} engagement(s)…',
        })

        predictions = getDefaultPredictor().predictBatch(rows)

        # Classement prédit : pour chaque épreuve, on trie par proba
        # décroissante (les cold-start traités comme proba 0) et on pose
        # `classement_predit` (1 = favori). Sert à la fois pour l'UI
        # (médailles top 3) et pour le CSV téléchargeable.
        _addClassementPredit(predictions)

        # Index par concours pour la page liste / détail
        byConc: dict[str, list[dict]] = defaultdict(list)
        for r in predictions:
            byConc[str(r.get('s2_Numéro', ''))].append(r)

        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is not None:
                j.state       = 'done'
                j.predictions = predictions
                j.by_concours = dict(byConc)
                j.finished_at = time.time()

        _appendEvent(job_id, {
            'type':       'predictions_done',
            'count':      len(predictions),
            'concours':   len(byConc),
        })

        # Auto-save dans l'historique : évite à l'utilisateur d'avoir à
        # exporter le CSV puis le réimporter manuellement dans /history.
        # On écrit un fichier dédié `predict_<job>_<ts>.csv` qui apparaît
        # dans la grille de /history sans toucher au principal
        # predictions.csv (réservé aux prédictions unitaires).
        try:
            saved = prediction_history.saveBatch(
                predictions  = predictions,
                source_label = job_id[:8],
            )
            _appendEvent(job_id, {
                'type':     'history_saved',
                'filename': saved.get('filename', ''),
                'n_rows':   saved.get('n_rows', 0),
                'error':    saved.get('error', ''),
            })
        except Exception as exc:
            # Best-effort : on ne casse pas le job si l'écriture échoue.
            _appendEvent(job_id, {
                'type':    'history_saved',
                'error':   f'auto-save échoué : {exc}',
            })

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


# ──────────────────────────────────────────────
# Snapshot pour la page /crawl-future (polling)
# ──────────────────────────────────────────────
def getFutureSnapshot(job_id: str, cursor: int = 0) -> dict[str, Any]:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            return {'state': 'unknown', 'cursor': cursor, 'events': []}
        return {
            'job_id':       j.job_id,
            'state':        j.state,
            'cursor':       len(j.events),
            'events':       j.events[cursor:],
            'error':        j.error,
            'rows':         len(j.predictions),
            'concours':     len(j.by_concours),
            'total_rows':   j.total_rows,
            'total_urls':   j.total_urls,
            'params':       j.params,
        }


# ──────────────────────────────────────────────
# API "Prédictions" — liste / détail / recherche / download
# ──────────────────────────────────────────────
def listJobs() -> list[dict[str, Any]]:
    """Tous les jobs futurs en mémoire (terminés ou en cours)."""
    with _jobs_lock:
        return [
            {
                'job_id':       j.job_id,
                'state':        j.state,
                'started_at':   j.started_at,
                'finished_at':  j.finished_at,
                'params':       j.params,
                'rows':         len(j.predictions),
                'concours':     len(j.by_concours),
            }
            for j in _jobs.values()
        ]


def listConcours(job_id: str | None = None) -> list[dict[str, Any]]:
    """
    Tous les concours crawlés (tous jobs confondus si job_id=None).
    Pour chaque concours : numéro, discipline majoritaire, nb engagements,
    proba moyenne, job_id source.
    """
    with _jobs_lock:
        jobs = [_jobs[job_id]] if (job_id and job_id in _jobs) else list(_jobs.values())

    out: list[dict[str, Any]] = []
    for j in jobs:
        if j.state != 'done':
            continue
        for conc_num, rows in j.by_concours.items():
            if not conc_num or conc_num == 'NONE':
                continue
            disciplines = [r.get('s3_Discipline', '') for r in rows]
            disc_mode = max(set(disciplines), key = disciplines.count) if disciplines else ''
            probas = [r['proba'] for r in rows if isinstance(r.get('proba'), int)]
            avgProba = (sum(probas) / len(probas)) if probas else None
            out.append({
                'concours_id': conc_num,
                'job_id':      j.job_id,
                'discipline':  disc_mode,
                'engagements': len(rows),
                'avg_proba':   round(avgProba, 1) if avgProba is not None else None,
                'epreuves':    len(set(r.get('s3_Épreuve', '') for r in rows)),
            })

    # Tri par numéro de concours (string mais souvent numérique)
    out.sort(key = lambda x: x['concours_id'])
    return out


def getConcoursDetail(concours_id: str) -> dict[str, Any] | None:
    """
    Tous les engagements (= participants) d'un concours futur, avec leurs
    prédictions. Cherche dans tous les jobs — si plusieurs jobs ont
    crawlé le même concours, on renvoie la version la plus récente.
    """
    with _jobs_lock:
        candidates = [
            (j.finished_at or 0, j)
            for j in _jobs.values()
            if j.state == 'done' and concours_id in j.by_concours
        ]
        if not candidates:
            return None
        candidates.sort(reverse = True)
        j = candidates[0][1]
        rows = list(j.by_concours[concours_id])

    if not rows:
        return None

    disciplines = sorted({r.get('s3_Discipline', '') for r in rows if r.get('s3_Discipline')})
    epreuves    = sorted({r.get('s3_Épreuve', '')    for r in rows if r.get('s3_Épreuve')})
    return {
        'concours_id': concours_id,
        'job_id':      j.job_id,
        'disciplines': disciplines,
        'epreuves':    epreuves,
        'engagements': rows,
    }


def search(query: str, limit: int = 100) -> list[dict[str, Any]]:
    """
    Recherche sur cheval / cavalier / numéro de concours dans toutes les
    prédictions en mémoire. Une ligne par engagement par concours (jamais
    agrégé) — c'est ce que demande l'UX : on veut voir le même cheval
    apparaître plusieurs fois s'il est engagé sur plusieurs concours.
    """
    q = (query or '').strip().lower()
    if not q:
        return []

    out: list[dict[str, Any]] = []
    with _jobs_lock:
        jobs = list(_jobs.values())

    for j in jobs:
        if j.state != 'done':
            continue
        for r in j.predictions:
            hay = ' '.join([
                str(r.get('s2_Numéro',   '')),
                str(r.get('s4_Équidé',   '')),
                str(r.get('s4_Cavalier', '')),
                str(r.get('s4_Club engageur', '')),
            ]).lower()
            if q in hay:
                row = dict(r)
                row['job_id'] = j.job_id
                out.append(row)
                if len(out) >= limit:
                    return out
    return out


def downloadPredictionsCsv(job_id: str) -> bytes | None:
    """Sérialise les prédictions d'un job en CSV (utf-8-sig)."""
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None or j.state != 'done':
            return None
        rows = list(j.predictions)

    return _rowsToCsvBytes(rows)


def downloadAllPredictionsCsv() -> bytes | None:
    """Concatène tous les jobs terminés en un seul CSV téléchargeable."""
    with _jobs_lock:
        rows: list[dict] = []
        for j in _jobs.values():
            if j.state == 'done':
                rows.extend(j.predictions)
    return _rowsToCsvBytes(rows)


def downloadConcoursCsv(concours_id: str) -> bytes | None:
    """Toutes les prédictions d'un concours (tous jobs confondus)."""
    rows: list[dict] = []
    with _jobs_lock:
        for j in _jobs.values():
            if j.state != 'done':
                continue
            rows.extend(j.by_concours.get(concours_id, []))
    return _rowsToCsvBytes(rows)


def _rowsToCsvBytes(rows: list[dict]) -> bytes | None:
    if not rows:
        return None
    columns = list(DATASET_COLUMNS) + list(PREDICTION_EXTRA_COLUMNS)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames = columns, extrasaction = 'ignore')
    writer.writeheader()
    writer.writerows(rows)
    return ('﻿' + buf.getvalue()).encode('utf-8')
