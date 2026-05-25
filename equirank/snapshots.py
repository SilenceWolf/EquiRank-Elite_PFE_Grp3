# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Gestion des snapshots de datasets — permet à l'utilisateur de :
  • conserver le dataset/modèle "par défaut" (livré avec le PFE)
  • archiver chaque dataset crawlé + entraîné en un snapshot nommé
  • basculer l'application sur un autre snapshot (les prédictions
    et l'autocomplete suivent le dataset actif)
  • revenir au snapshot par défaut à tout moment

Layout disque (sous `data/snapshots/`) :

    data/snapshots/
    ├── default/
    │   ├── dataset_brut_v2.csv      (copie du fichier livré au 1er run)
    │   ├── dataset.csv              (version adaptée)
    │   ├── models/
    │   │   ├── LightGBM.joblib
    │   │   ├── encoders.joblib
    │   │   └── feature_cols.joblib
    │   └── metadata.json
    └── crawl_2026-05-24_ab12cd34/    (un dossier par crawl archivé)
        └── … même structure

Le dataset *actif* est celui présent dans `data/` et `models/` à la racine
du PFE. Activer un snapshot = copier ses fichiers à la racine, puis
demander au predictor de se recharger.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib     import Path
from typing      import Any


_ROOT          = Path(__file__).resolve().parent.parent           # PFE/
_DATA_DIR      = _ROOT / 'data'
_MODELS_DIR    = _ROOT / 'models'
_SNAPSHOTS_DIR = _DATA_DIR / 'snapshots'

DEFAULT_ID = 'default'

# Fichier qui mémorise quel snapshot est actuellement actif. Stocké à
# la racine de snapshots/ (pas dans le snapshot lui-même) pour que la
# config survive aux activations.
_ACTIVE_MARKER = _SNAPSHOTS_DIR / 'ACTIVE'


# ──────────────────────────────────────────────
# Dataclass + utilitaires
# ──────────────────────────────────────────────
@dataclass
class Snapshot:
    snapshot_id:  str
    label:        str
    created_at:   str
    n_rows:       int = 0
    accuracy:     float = 0.0
    model_name:   str = ''
    description:  str = ''
    is_default:   bool = False
    is_active:    bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'snapshot_id': self.snapshot_id,
            'label':       self.label,
            'created_at':  self.created_at,
            'n_rows':      int(self.n_rows),
            'accuracy':    float(self.accuracy),
            'model_name':  self.model_name,
            'description': self.description,
            'is_default':  bool(self.is_default),
            'is_active':   bool(self.is_active),
        }


def _snapshotDir(snapshot_id: str) -> Path:
    return _SNAPSHOTS_DIR / snapshot_id


def _safeReadJson(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding = 'utf-8'))
    except Exception:
        pass
    return {}


def _writeJson(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(obj, indent = 2, ensure_ascii = False), encoding = 'utf-8')


def _getActiveId() -> str:
    if _ACTIVE_MARKER.exists():
        try:
            v = _ACTIVE_MARKER.read_text(encoding = 'utf-8').strip()
            if v:
                return v
        except Exception:
            pass
    return DEFAULT_ID


def _setActiveId(snapshot_id: str) -> None:
    _SNAPSHOTS_DIR.mkdir(parents = True, exist_ok = True)
    _ACTIVE_MARKER.write_text(snapshot_id, encoding = 'utf-8')


# ──────────────────────────────────────────────
# Snapshot "default" — capturé au 1er démarrage
# ──────────────────────────────────────────────
def ensureDefaultSnapshot() -> None:
    """
    À appeler au démarrage du serveur. Si `data/snapshots/default/`
    n'existe pas, on copie le contenu actuel de `data/` + `models/`
    pour figer l'état d'origine — sert de "point de restauration".
    """
    defaultDir = _snapshotDir(DEFAULT_ID)
    if defaultDir.exists():
        return

    defaultDir.mkdir(parents = True, exist_ok = True)

    # Copie des CSV
    for fname in ('dataset_brut_v2.csv', 'dataset.csv'):
        src = _DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, defaultDir / fname)

    # Copie des artefacts modèle
    modelsDest = defaultDir / 'models'
    modelsDest.mkdir(exist_ok = True)
    for fname in ('LightGBM.joblib','XGBoost.joblib','RandomForest.joblib',
                  'DecisionTree.joblib','LogisticRegression.joblib',
                  'encoders.joblib','feature_cols.joblib'):
        src = _MODELS_DIR / fname
        if src.exists():
            shutil.copy2(src, modelsDest / fname)

    # Métriques de base (si dispo)
    accuracy   = 0.0
    modelName  = ''
    for name in ('LightGBM','XGBoost','RandomForest','DecisionTree','LogisticRegression'):
        metricsPath = _ROOT / 'outputs' / 'models' / f'metrics_{name}.json'
        if metricsPath.exists():
            metrics = _safeReadJson(metricsPath)
            accuracy  = float(metrics.get('accuracy', 0.0) or 0.0)
            modelName = name
            break

    # Compte les lignes du dataset.csv
    n_rows = 0
    dsPath = defaultDir / 'dataset.csv'
    if dsPath.exists():
        try:
            with dsPath.open('r', encoding = 'utf-8', errors = 'ignore') as fp:
                n_rows = max(0, sum(1 for _ in fp) - 1)
        except Exception:
            n_rows = 0

    _writeJson(defaultDir / 'metadata.json', {
        'snapshot_id': DEFAULT_ID,
        'label':       'Dataset par défaut (livré avec le PFE)',
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_rows':      n_rows,
        'accuracy':    accuracy,
        'model_name':  modelName,
        'description': 'Snapshot initial — restauration garantie.',
        'is_default':  True,
    })

    # Le snapshot default devient actif par défaut s'il n'y en a pas
    if not _ACTIVE_MARKER.exists():
        _setActiveId(DEFAULT_ID)


# ──────────────────────────────────────────────
# Archivage d'un crawl + entraînement
# ──────────────────────────────────────────────
def archiveCurrentDataset(
    label:       str,
    job_id:      str | None = None,
    description: str = '',
) -> Snapshot:
    """
    Snapshot du contenu actuel de `data/` + `models/`. Appelé après
    chaque pipeline d'entraînement réussi pour permettre à l'utilisateur
    de revenir à cet état plus tard.
    """
    ensureDefaultSnapshot()

    stamp = time.strftime('%Y-%m-%d_%H%M%S')
    suffix = (job_id or '')[:8]
    snapshot_id = f'crawl_{stamp}' + (f'_{suffix}' if suffix else '')

    target = _snapshotDir(snapshot_id)
    target.mkdir(parents = True, exist_ok = True)

    # Copie des CSV courants
    for fname in ('dataset_brut_v2.csv', 'dataset.csv'):
        src = _DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, target / fname)

    # Copie des modèles actuels
    modelsDest = target / 'models'
    modelsDest.mkdir(exist_ok = True)
    for src in _MODELS_DIR.glob('*.joblib'):
        shutil.copy2(src, modelsDest / src.name)

    # Métriques fraîches
    accuracy  = 0.0
    modelName = ''
    for name in ('LightGBM','XGBoost','RandomForest','DecisionTree','LogisticRegression'):
        metricsPath = _ROOT / 'outputs' / 'models' / f'metrics_{name}.json'
        if metricsPath.exists():
            metrics = _safeReadJson(metricsPath)
            accuracy  = float(metrics.get('accuracy', 0.0) or 0.0)
            modelName = name
            break

    # Compte les rows de dataset.csv
    n_rows = 0
    dsPath = target / 'dataset.csv'
    if dsPath.exists():
        try:
            with dsPath.open('r', encoding = 'utf-8', errors = 'ignore') as fp:
                n_rows = max(0, sum(1 for _ in fp) - 1)
        except Exception:
            n_rows = 0

    _writeJson(target / 'metadata.json', {
        'snapshot_id': snapshot_id,
        'label':       label,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_rows':      n_rows,
        'accuracy':    accuracy,
        'model_name':  modelName,
        'description': description,
        'is_default':  False,
    })

    return _loadSnapshot(snapshot_id, active = (_getActiveId() == snapshot_id))


# ──────────────────────────────────────────────
# Liste / activation / suppression
# ──────────────────────────────────────────────
def _loadSnapshot(snapshot_id: str, active: bool) -> Snapshot:
    meta = _safeReadJson(_snapshotDir(snapshot_id) / 'metadata.json')
    return Snapshot(
        snapshot_id = snapshot_id,
        label       = meta.get('label', snapshot_id),
        created_at  = meta.get('created_at', ''),
        n_rows      = int(meta.get('n_rows', 0) or 0),
        accuracy    = float(meta.get('accuracy', 0.0) or 0.0),
        model_name  = meta.get('model_name', ''),
        description = meta.get('description', ''),
        is_default  = bool(meta.get('is_default', snapshot_id == DEFAULT_ID)),
        is_active   = active,
    )


def listSnapshots() -> list[Snapshot]:
    """Liste tous les snapshots disponibles, default en tête."""
    ensureDefaultSnapshot()
    activeId = _getActiveId()
    out: list[Snapshot] = []
    if not _SNAPSHOTS_DIR.exists():
        return out

    for child in _SNAPSHOTS_DIR.iterdir():
        if not child.is_dir():
            continue
        sid = child.name
        out.append(_loadSnapshot(sid, active = (sid == activeId)))

    # Default tjs en tête, puis par date décroissante
    out.sort(key = lambda s: (not s.is_default, -1 * _parseTimestamp(s.created_at)))
    return out


def _parseTimestamp(ts: str) -> float:
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def activateSnapshot(snapshot_id: str) -> Snapshot:
    """
    Copie les fichiers du snapshot vers `data/` et `models/`, marque le
    snapshot comme actif, et reset le predictor en mémoire pour qu'il
    recharge au prochain appel.
    """
    src = _snapshotDir(snapshot_id)
    if not src.exists():
        raise FileNotFoundError(f'Snapshot {snapshot_id} introuvable.')

    # 1. CSV
    for fname in ('dataset_brut_v2.csv', 'dataset.csv'):
        srcFile = src / fname
        if srcFile.exists():
            shutil.copy2(srcFile, _DATA_DIR / fname)

    # 2. Modèles
    srcModels = src / 'models'
    if srcModels.exists():
        # On supprime les .joblib actuels pour éviter qu'un fichier
        # absent du snapshot ne reste actif (et fasse choisir le "mauvais"
        # modèle dans _MODEL_PRIORITY côté predictor).
        for existing in _MODELS_DIR.glob('*.joblib'):
            existing.unlink()
        _MODELS_DIR.mkdir(exist_ok = True)
        for srcFile in srcModels.glob('*.joblib'):
            shutil.copy2(srcFile, _MODELS_DIR / srcFile.name)

    _setActiveId(snapshot_id)

    # 3. Reload predictor en mémoire (lazy : la prochaine prédiction
    # recrée l'instance). Import tardif pour éviter une dépendance
    # circulaire au boot.
    try:
        from . import predictor as _pred
        _pred._default_predictor = None
    except Exception:
        warnings.warn('Reload predictor a échoué — restart manuel d\'uvicorn requis.')

    return _loadSnapshot(snapshot_id, active = True)


def restoreDefault() -> Snapshot:
    """Raccourci : activate le snapshot 'default'."""
    return activateSnapshot(DEFAULT_ID)


def deleteSnapshot(snapshot_id: str) -> None:
    """Supprime un snapshot. Le default n'est jamais supprimable."""
    if snapshot_id == DEFAULT_ID:
        raise ValueError("Le snapshot 'default' ne peut pas être supprimé.")
    if _getActiveId() == snapshot_id:
        raise ValueError(
            f"Snapshot {snapshot_id} actuellement actif — bascule d'abord "
            f"sur un autre snapshot avant de le supprimer."
        )
    target = _snapshotDir(snapshot_id)
    if target.exists():
        shutil.rmtree(target, ignore_errors = True)


def getActiveSnapshot() -> Snapshot | None:
    ensureDefaultSnapshot()
    sid = _getActiveId()
    if not _snapshotDir(sid).exists():
        return None
    return _loadSnapshot(sid, active = True)
