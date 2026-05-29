# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Historique des prédictions unitaires (page /) — stockage CSV multi-fichier.

Layout disque :

    data/history/
    ├── predictions.csv           ← ajouts automatiques via /api/predict
    ├── imported_2026-05-25.csv   ← ton CSV importé via l'UI
    └── n_importe_quoi.csv        ← n'importe quel CSV que tu déposes ici

La page /history fusionne TOUS les `*.csv` du dossier (les uns appendés
automatiquement par l'app, les autres déposés à la main par l'utilisateur).
Pratique : tu peux drop un CSV directement dans `data/history/` sans
passer par l'UI — il apparaîtra au prochain refresh de /history.

Colonnes reconnues (insensible casse, séparateur auto-détecté) :
    cheval / horse / equide / équidé        (obligatoire)
    cavalier / jockey / rider               (obligatoire)
    discipline                              (obligatoire)
    proba / probability / probabilité       (0-100 ou 0-1, défaut 0)
    verdict                                  (favori / outsider+ / outsider / cold-start)
    ts / date / timestamp                    (ISO, défaut now)
    model_name                               (défaut '')
    is_cold_start                            (true/false, défaut false)
    snapshot                                 (défaut '')
    id                                       (défaut auto-généré)

Cap : 5000 entrées au total dans `predictions.csv` (le fichier auto-append).
Les CSV importés ne sont JAMAIS tronqués automatiquement — l'utilisateur
les supprime à la main quand il veut.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing  import Any, Iterable


_ROOT         = Path(__file__).resolve().parent.parent     # PFE/
_DATA_DIR     = _ROOT / 'data'
_HISTORY_DIR  = _DATA_DIR / 'history'
_MAIN_FILE    = _HISTORY_DIR / 'predictions.csv'

# Migration : l'ancien fichier JSONL à la racine de data/
_LEGACY_FILE  = _DATA_DIR / 'prediction_history.jsonl'

# Colonnes canoniques pour le fichier principal (ordre stable).
# `epreuve` est ajoutée pour porter le détail épreuve (s3_Épreuve dans
# les CSV du crawler futur) quand il est disponible — c'est elle qui
# détermine le vrai classement (un cheval est rang 1 dans son épreuve,
# pas dans toute la discipline).
_COLUMNS = (
    'ts', 'cheval', 'cavalier', 'discipline', 'epreuve', 'hauteur', 'niveau',
    'classement_estime',
    'proba', 'verdict', 'is_cold_start',
    'model_name', 'snapshot', 'id',
)

_MAX_MAIN_ENTRIES = 5000
_lock             = threading.Lock()


# ──────────────────────────────────────────────
# Bootstrap : crée le dossier + migre l'ancien JSONL si présent
# ──────────────────────────────────────────────
def _ensureLayout() -> None:
    _HISTORY_DIR.mkdir(parents = True, exist_ok = True)
    if not _MAIN_FILE.exists():
        _writeMainFromRows([])

    # Migration silencieuse de l'ancien format JSONL
    if _LEGACY_FILE.exists():
        try:
            rows: list[dict] = []
            for raw in _LEGACY_FILE.read_text(encoding = 'utf-8').splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except Exception:
                    continue
            if rows:
                existing = _readMain()
                _writeMainFromRows(rows + existing)   # rows JSONL en premier (plus anciens)
            _LEGACY_FILE.unlink()
        except Exception:
            pass


def _writeMainFromRows(rows: list[dict]) -> None:
    """Réécrit `predictions.csv` (utf-8-sig, séparateur virgule)."""
    _MAIN_FILE.parent.mkdir(parents = True, exist_ok = True)
    with _MAIN_FILE.open('w', encoding = 'utf-8-sig', newline = '') as fp:
        writer = csv.DictWriter(fp, fieldnames = list(_COLUMNS), extrasaction = 'ignore')
        writer.writeheader()
        writer.writerows(rows)


def _readMain() -> list[dict]:
    if not _MAIN_FILE.exists():
        return []
    try:
        return _readCsvFile(_MAIN_FILE)
    except Exception:
        return []


# ──────────────────────────────────────────────
# Parsing tolérant d'un CSV (fichier déposé par l'utilisateur)
# ──────────────────────────────────────────────
def _readCsvFile(path: Path) -> list[dict]:
    """
    Lit un CSV (ou TSV, ou ;-séparé) et normalise vers le schéma _COLUMNS.
    Les colonnes inconnues sont ignorées. Les lignes sans cheval/cavalier/
    discipline sont sautées.
    """
    raw_bytes = path.read_bytes()
    text = ''
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            text = raw_bytes.decode(enc); break
        except UnicodeDecodeError:
            continue
    if not text:
        return []

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters = ',;\t|')
    except csv.Error:
        dialect = csv.excel  # type: ignore[assignment]

    reader = csv.DictReader(io.StringIO(text), dialect = dialect)
    if not reader.fieldnames:
        return []

    norm = {h.strip().lower(): h for h in reader.fieldnames}

    def _col(*aliases: str) -> str | None:
        for a in aliases:
            if a in norm:
                return norm[a]
        return None

    # Alias étendus : on accepte les colonnes du dataset_brut_v2 / crawl
    # futur (s4_Équidé, s4_Cavalier, s3_Discipline) en plus des noms
    # canoniques — ça permet de déposer directement un CSV exporté de
    # /crawl-future dans data/history/ sans le transformer.
    colCheval     = _col('cheval', 'horse', 'equide', 'équidé',
                         's4_équidé', 's4_equide', 's4_cheval')
    colCavalier   = _col('cavalier', 'jockey', 'rider',
                         's4_cavalier', 's4_jockey')
    colDiscipline = _col('discipline',
                         's3_discipline')
    colEpreuve    = _col('epreuve', 'épreuve', 's3_épreuve', 's3_epreuve')
    colHauteur    = _col('hauteur', 'hauteur_cm', 'distance', 'hauteur_centimetres')
    colNiveau     = _col('niveau', 'niveau_epreuve', 's3_niveau')
    colRank       = _col('classement_estime', 'classement_predit', 'rank')
    colProba      = _col('proba', 'probability', 'probabilité', 'probabilite')
    colVerdict    = _col('verdict')
    colTs         = _col('ts', 'date', 'timestamp', 'time')
    colModel      = _col('model_name', 'model', 'modele', 'modèle')
    colCold       = _col('is_cold_start', 'cold_start', 'cold-start', 'coldstart')
    colSnap       = _col('snapshot')
    colId         = _col('id')

    if not (colCheval and colCavalier and colDiscipline):
        return []

    out: list[dict] = []
    for row in reader:
        cheval     = (row.get(colCheval)     or '').strip()
        cavalier   = (row.get(colCavalier)   or '').strip()
        discipline = (row.get(colDiscipline) or '').strip()
        # Cavalier devient optionnel — certaines épreuves n'en ont pas
        # (équipe non nominative, attelage, …). Cheval + discipline
        # restent obligatoires pour pouvoir matcher dans le modèle.
        if not cheval or not discipline:
            continue

        proba = 0
        if colProba and row.get(colProba):
            raw = str(row[colProba]).strip().rstrip('%').replace(',', '.')
            try:
                v = float(raw)
                proba = int(round(v * 100)) if 0 <= v <= 1 else int(round(v))
                proba = max(0, min(100, proba))
            except ValueError:
                proba = 0

        is_cold = False
        if colCold and row.get(colCold):
            v = str(row[colCold]).strip().lower()
            is_cold = v in ('true', '1', 'yes', 'oui', 'cold-start', 'cold')

        verdict = (row.get(colVerdict, '') if colVerdict else '').strip().lower()
        if not verdict:
            verdict = ('cold-start' if is_cold else
                       'favori' if proba >= 60 else
                       'outsider+' if proba >= 35 else
                       'outsider')

        ts        = (row.get(colTs, '')    if colTs    else '').strip() or time.strftime('%Y-%m-%dT%H:%M:%S')
        model_nm  = (row.get(colModel, '') if colModel else '').strip()
        snapshot  = (row.get(colSnap,  '') if colSnap  else '').strip()
        entry_id  = (row.get(colId, '')    if colId    else '').strip() or uuid.uuid4().hex[:12]
        epreuve   = (row.get(colEpreuve, '') if colEpreuve else '').strip()

        hauteur = ''
        if colHauteur and row.get(colHauteur):
            raw = str(row[colHauteur]).strip().lower().replace('cm', '').replace(',', '.').strip()
            try:
                v = float(raw)
                if 0 < v <= 200:
                    hauteur = str(int(round(v)))
            except ValueError:
                hauteur = ''

        classement_estime = 0
        if colRank and row.get(colRank):
            try:
                classement_estime = int(float(str(row[colRank]).strip()))
                if classement_estime < 1:
                    classement_estime = 0
            except ValueError:
                classement_estime = 0

        niveau = (row.get(colNiveau, '') if colNiveau else '').strip()

        out.append({
            'id':                entry_id,
            'ts':                ts,
            'cheval':            cheval,
            'cavalier':          cavalier,
            'discipline':        discipline,
            'epreuve':           epreuve,
            'hauteur':           hauteur,
            'niveau':            niveau,
            'classement_estime': classement_estime,
            'proba':             proba,
            'verdict':           verdict,
            'model_name':        model_nm,
            'is_cold_start':     is_cold,
            'snapshot':          snapshot,
        })
    return out


# ──────────────────────────────────────────────
# API publique
# ──────────────────────────────────────────────
def addEntry(
    cheval:            str,
    cavalier:          str,
    discipline:        str,
    proba:             int,
    verdict:           str,
    model_name:        str,
    is_cold_start:     bool,
    snapshot:          str = '',
    epreuve:           str = '',
    hauteur:           str = '',
    niveau:            str = '',
    classement_estime: int = 0,
) -> dict[str, Any]:
    """
    Append au fichier principal `data/history/predictions.csv`.

    Déduplication : si une entrée avec la MÊME combinaison
    (cheval, cavalier, discipline, hauteur, niveau) existe déjà dans
    predictions.csv, on la remplace (mise à jour de la proba et de la
    date) plutôt que de créer un doublon. Évite que l'historique se
    pollue de la même prédiction faite 10 fois pour vérifier la valeur.
    """
    entry = {
        'id':                uuid.uuid4().hex[:12],
        'ts':                time.strftime('%Y-%m-%dT%H:%M:%S'),
        'cheval':            cheval,
        'cavalier':          cavalier,
        'discipline':        discipline,
        'epreuve':           epreuve,
        'hauteur':           str(hauteur) if hauteur not in (None, '') else '',
        'niveau':            str(niveau)  if niveau  not in (None, '') else '',
        'classement_estime': int(classement_estime) if classement_estime else 0,
        'proba':             int(proba),
        'verdict':           verdict,
        'model_name':        model_name,
        'is_cold_start':     bool(is_cold_start),
        'snapshot':          snapshot,
    }

    with _lock:
        _ensureLayout()

        # Clé de dédup — normalisée (insensible casse + strip).
        def _comboKey(d: dict) -> tuple:
            return (
                (str(d.get('cheval', '')) or '').strip().upper(),
                (str(d.get('cavalier', '')) or '').strip().upper(),
                (str(d.get('discipline', '')) or '').strip(),
                str(d.get('hauteur', '') or '').strip(),
                str(d.get('niveau', '') or '').strip(),
            )
        newKey = _comboKey(entry)

        existing = _readMain()
        # On garde toutes les entrées dont la clé combo diffère —
        # remplace donc l'éventuelle entrée existante avec le même duo.
        kept = [r for r in existing if _comboKey(r) != newKey]
        kept.append(entry)
        _writeMainFromRows(kept)

        # Tronque si on dépasse le cap
        if len(kept) > _MAX_MAIN_ENTRIES:
            _truncateMainIfNeeded()
    return entry


def saveBatch(
    predictions: list[dict],
    source_label: str = '',
) -> dict[str, Any]:
    """
    Auto-sauvegarde d'un lot de prédictions (typiquement produit par
    /crawl-predict) dans un fichier CSV dédié de `data/history/`.

    On NE touche PAS au fichier principal `predictions.csv` (réservé
    aux prédictions unitaires faites depuis la page d'accueil) — on
    écrit dans un fichier séparé `predict_<label>_<ts>.csv`. Ça permet
    à l'utilisateur de retrouver ses crawls predict dans /history sans
    avoir à exporter/importer manuellement le CSV.

    `predictions` : list[dict] au format `future_jobs._FutureJob.predictions`
                    (clés `s4_Équidé`, `s4_Cavalier`, `s3_Discipline`,
                    `s3_Épreuve`, `proba`, `verdict`, `is_cold_start`,
                    `model_name`, `classement_predit`).
    `source_label`: ex. job_id court — utilisé pour nommer le fichier.

    Renvoie : {filename, n_rows, error?}
    """
    if not predictions:
        return {'filename': '', 'n_rows': 0, 'error': 'aucune prédiction'}

    _ensureLayout()

    import re as _re
    label = _re.sub(r'[^a-zA-Z0-9_-]+', '_', (source_label or '').strip())[:24]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name  = f'predict_{label}_{stamp}.csv' if label else f'predict_{stamp}.csv'
    target = _HISTORY_DIR / name

    # Normalisation au schéma `_COLUMNS` pour que /history et la page
    # de détail lisent le fichier exactement comme predictions.csv.
    rows_out: list[dict] = []
    for p in predictions:
        cheval     = str(p.get('s4_Équidé')   or p.get('cheval')     or '').strip()
        cavalier   = str(p.get('s4_Cavalier') or p.get('cavalier')   or '').strip()
        discipline = str(p.get('s3_Discipline') or p.get('discipline') or '').strip()
        epreuve    = str(p.get('s3_Épreuve')  or p.get('epreuve')    or '').strip()
        if not cheval or not discipline:
            continue

        try:
            proba_raw = p.get('proba')
            proba = int(round(float(proba_raw))) if proba_raw not in (None, '') else 0
        except (TypeError, ValueError):
            proba = 0
        proba = max(0, min(100, proba))

        try:
            rank = int(p.get('classement_predit') or 0)
        except (TypeError, ValueError):
            rank = 0

        is_cold = bool(p.get('is_cold_start'))
        verdict = str(p.get('verdict') or
                      ('cold-start' if is_cold else
                       'favori'     if proba >= 60 else
                       'outsider+'  if proba >= 35 else
                       'outsider')).strip()

        rows_out.append({
            'ts':                time.strftime('%Y-%m-%dT%H:%M:%S'),
            'cheval':            cheval,
            'cavalier':          cavalier,
            'discipline':        discipline,
            'epreuve':           epreuve,
            'hauteur':           '',
            'classement_estime': rank,
            'proba':             proba,
            'verdict':           verdict,
            'is_cold_start':     is_cold,
            'model_name':        str(p.get('model_name') or ''),
            'snapshot':          f'crawl-predict:{label}' if label else 'crawl-predict',
            'id':                uuid.uuid4().hex[:12],
        })

    if not rows_out:
        return {'filename': '', 'n_rows': 0, 'error': 'aucune ligne exploitable'}

    with _lock:
        with target.open('w', encoding = 'utf-8-sig', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames = list(_COLUMNS))
            writer.writeheader()
            writer.writerows(rows_out)

    return {'filename': target.name, 'n_rows': len(rows_out)}


def _truncateMainIfNeeded() -> None:
    rows = _readMain()
    if len(rows) <= _MAX_MAIN_ENTRIES:
        return
    # Tri par ts ASC, on garde les plus récents
    rows.sort(key = lambda r: r.get('ts', ''))
    _writeMainFromRows(rows[-_MAX_MAIN_ENTRIES:])


def _allFiles() -> list[Path]:
    """Tous les *.csv du dossier history."""
    if not _HISTORY_DIR.exists():
        return []
    return sorted(_HISTORY_DIR.glob('*.csv'))


def _allRows() -> list[dict]:
    """Fusion de tous les CSV du dossier — dédup par id."""
    _ensureLayout()
    seen: set[str] = set()
    out: list[dict] = []
    for path in _allFiles():
        for entry in _readCsvFile(path):
            eid = str(entry.get('id') or '')
            if eid and eid in seen:
                continue
            if eid:
                seen.add(eid)
            entry['_source'] = path.name
            out.append(entry)
    return out


def listEntries(limit: int = 200, query: str = '') -> list[dict[str, Any]]:
    """
    Plus récents en premier. Optionnellement filtre par sous-chaîne
    (insensible casse) sur cheval/cavalier/discipline.
    """
    rows = _allRows()
    rows.sort(key = lambda r: r.get('ts', ''), reverse = True)

    q = (query or '').strip().lower()
    if q:
        rows = [r for r in rows if q in ' '.join([
            str(r.get('cheval', '')), str(r.get('cavalier', '')),
            str(r.get('discipline', '')),
        ]).lower()]

    return rows[:limit]


def deleteEntry(entry_id: str) -> bool:
    """
    Supprime une entrée par id, uniquement dans `predictions.csv` (le
    fichier auto-append). Les CSV importés par l'utilisateur ne sont
    JAMAIS modifiés — il les édite/supprime à la main s'il veut.
    """
    with _lock:
        rows = _readMain()
        kept = [r for r in rows if r.get('id') != entry_id]
        if len(kept) == len(rows):
            return False
        _writeMainFromRows(kept)
        return True


def clearAll() -> int:
    """
    Vide `predictions.csv`. AVANT le vidage, on archive le contenu
    actuel dans un fichier daté `predictions_archive_<ts>.csv` qui
    apparaîtra dans la grille /history. Comme ça l'utilisateur ne
    perd JAMAIS de données — il peut récupérer en restaurant le
    fichier archive si nécessaire.
    Les CSV importés par l'utilisateur ne sont pas touchés.
    """
    with _lock:
        rows = _readMain()
        n = len(rows)
        if n > 0:
            # Archive horodatée — visible dans /history comme un import
            ts = time.strftime('%Y-%m-%d_%H%M%S')
            archive = _HISTORY_DIR / f'predictions_archive_{ts}.csv'
            try:
                with archive.open('w', encoding = 'utf-8-sig', newline = '') as fp:
                    writer = csv.DictWriter(fp, fieldnames = list(_COLUMNS), extrasaction = 'ignore')
                    writer.writeheader()
                    writer.writerows(rows)
            except Exception:
                pass     # best effort — si on ne peut pas archiver, on continue
        _writeMainFromRows([])
    return n


def exportCsv() -> bytes:
    """Sérialise TOUT l'historique (auto + importés) en un seul CSV."""
    rows = listEntries(limit = 100_000)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames = list(_COLUMNS), extrasaction = 'ignore')
    writer.writeheader()
    writer.writerows(rows)
    return ('﻿' + buf.getvalue()).encode('utf-8')


def importCsv(content: bytes, filename: str = '') -> dict[str, Any]:
    """
    Sauvegarde le CSV uploadé tel quel dans `data/history/` (le nom
    par défaut est dérivé du nom d'origine, sinon timestamp).
    Le fichier est lu au prochain listEntries (zero parsing au moment
    de l'upload — on garde la source originale intacte).

    Renvoie : {filename, n_rows, error?}
    """
    _ensureLayout()

    # Choix du nom de fichier final — sanitize et évite l'écrasement
    import re as _re
    safe = _re.sub(r'[^a-zA-Z0-9._-]+', '_', (filename or '').strip()) or 'import'
    if not safe.lower().endswith('.csv'):
        safe = safe + '.csv'
    if safe == 'predictions.csv':           # protection : ne pas écraser le principal
        safe = 'imported_' + time.strftime('%Y%m%d_%H%M%S') + '.csv'

    target = _HISTORY_DIR / safe
    if target.exists():
        target = _HISTORY_DIR / (target.stem + '_' + time.strftime('%H%M%S') + '.csv')

    target.write_bytes(content)

    # Lecture immédiate pour valider + compter
    try:
        parsed = _readCsvFile(target)
    except Exception as exc:
        return {'filename': target.name, 'n_rows': 0, 'error': str(exc)}

    if not parsed:
        # Le fichier est là mais on n'a rien pu en sortir — on prévient
        # l'utilisateur, mais on garde le fichier (il peut l'éditer).
        return {
            'filename': target.name, 'n_rows': 0,
            'error': 'Aucune ligne reconnue — il faut au moins les colonnes cheval, cavalier, discipline.',
        }

    return {'filename': target.name, 'n_rows': len(parsed)}


def listFiles() -> list[dict[str, Any]]:
    """
    Métadonnées de chaque fichier CSV du dossier — utile pour l'UI.
    Inclut une `avg_proba` calculée sur ses entrées (utile pour la
    grille de cards de /history).
    """
    _ensureLayout()
    out: list[dict[str, Any]] = []
    for path in _allFiles():
        try:
            stat = path.stat()
            rows = _readCsvFile(path)
            probas = [int(r['proba']) for r in rows
                      if isinstance(r.get('proba'), (int, float)) and not r.get('is_cold_start')]
            avg = round(sum(probas) / len(probas), 1) if probas else None
            disciplines = sorted({str(r.get('discipline', '')).strip()
                                  for r in rows if r.get('discipline')})
            out.append({
                'name':         path.name,
                'is_main':      path.name == _MAIN_FILE.name,
                'n_rows':       len(rows),
                'n_cold':       sum(1 for r in rows if r.get('is_cold_start')),
                'avg_proba':    avg,
                'disciplines':  disciplines,
                'size_bytes':   stat.st_size,
                'modified':     time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(stat.st_mtime)),
            })
        except Exception:
            continue
    # Tri : principal en premier, puis les plus récents
    out.sort(key = lambda f: (not f['is_main'], -1 * _parseModified(f['modified'])))
    return out


def _parseModified(iso: str) -> float:
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0


def listEntriesFromFile(name: str, query: str = '', sort: str = 'proba') -> list[dict[str, Any]]:
    """
    Entrées d'UN SEUL fichier CSV — pas de fusion. Trié par défaut par
    proba descendante (mode classement, cold-start relégué derrière).
    """
    target = _HISTORY_DIR / name
    if not target.exists() or target.parent != _HISTORY_DIR:
        return []
    try:
        rows = _readCsvFile(target)
    except Exception:
        return []

    q = (query or '').strip().lower()
    if q:
        rows = [r for r in rows if q in ' '.join([
            str(r.get('cheval', '')), str(r.get('cavalier', '')),
            str(r.get('discipline', '')),
        ]).lower()]

    if sort == 'ts':
        rows.sort(key = lambda r: r.get('ts', ''), reverse = True)
    else:
        rows.sort(key = lambda r: (
            1 if r.get('is_cold_start') else 0,
            -(r.get('proba') if isinstance(r.get('proba'), (int, float)) else -1),
        ))

    # On annote le rang dans le sous-ensemble retourné
    for i, r in enumerate(rows, start = 1):
        r['rank'] = i
    return rows


def deleteFile(name: str) -> bool:
    """Supprime un fichier CSV du dossier history (refuse predictions.csv)."""
    if name == _MAIN_FILE.name:
        raise ValueError("predictions.csv (fichier principal) — utiliser clearAll() à la place.")
    target = _HISTORY_DIR / name
    if not target.exists() or target.parent != _HISTORY_DIR:
        return False
    target.unlink()
    return True


# Iterable compat — pas exposé directement mais utile si besoin
def _iterAllEntries() -> Iterable[dict]:
    yield from _allRows()
