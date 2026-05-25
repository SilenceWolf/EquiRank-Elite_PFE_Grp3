# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Reconstitue un dataset au format `dataset_brut_v2.csv` à partir des
data_rows captés par chaque étape de la chaîne FFE.

Format cible (ordre des colonnes immuable, modèle PFE) :
    s2_Numéro,
    s3_Discipline, s3_Épreuve,
    s4_Clt., s4_Cavalier, s4_Club engageur, s4_Équidé,
    s4_Pts qualif. Chpt, s4_Équipe,
    s5_Robe, s5_Sexe, s5_Taille, s5_Père, s5_Mère

Logique de JOIN :
  • s4 (engagements) est la table centrale — une ligne par cheval engagé
  • s5 ⋈ s4    sur (s5.source_url == s4.Équidé_url)    → détails équidé
  • s4 ⋈ s3    sur (s4.source_url == s3.href)          → discipline/épreuve
  • s3 ⋈ s2    sur (s3.source_url == s2.href)          → numéro du concours
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing  import Iterable


# Colonnes finales du dataset, dans l'ordre exact attendu par les
# scripts PFE (preprocessing_v2, model_training, etc.).
DATASET_COLUMNS: tuple[str, ...] = (
    's2_Numéro',
    's3_Discipline',
    's3_Épreuve',
    's4_Clt.',
    's4_Cavalier',
    's4_Club engageur',
    's4_Équidé',
    's4_Pts qualif. Chpt',
    's4_Équipe',
    's5_Robe',
    's5_Sexe',
    's5_Taille',
    's5_Père',
    's5_Mère',
)


def _normUrl(u: object) -> str:
    """Normalise une URL pour usage en clé de join — strip + str."""
    if u is None:
        return ''
    return str(u).strip()


def _pickValue(row: dict, *candidates: str) -> str:
    """
    Premier candidate présent et non vide dans la row. Évite d'avoir à
    gérer manuellement les variantes de noms de colonnes (ex : `Numéro`
    vs `Numero` selon que la page a un caractère accentué dans le <th>).
    Retourne 'NONE' si rien trouvé — c'est le sentinel utilisé dans le
    dataset_brut_v2 de référence pour les valeurs manquantes.
    """
    for key in candidates:
        val = row.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s and s.upper() != 'NONE':
            return s
    return 'NONE'


def _indexBy(rows: Iterable[dict], key: str) -> dict[str, dict]:
    """
    Crée un index {valeur_clé → row} sur la colonne `key`. En cas de
    doublons (s4 a parfois deux fois le même Équidé_url avec des
    données différentes), on garde la dernière vue : c'est généralement
    la row la plus complète après l'enrichissement par url_filters.
    """
    out: dict[str, dict] = {}
    for r in rows:
        k = _normUrl(r.get(key))
        if not k:
            continue
        out[k] = r
    return out


def buildDatasetBrutV2(
    step_rows:    dict[str, list[dict]],
    output_path:  Path | str | None = None,
) -> list[dict]:
    """
    Construit la liste de rows au format dataset_brut_v2 et, si
    `output_path` est fourni, la sérialise en CSV (UTF-8 avec BOM
    pour compatibilité Excel — c'est ce qu'utilise le crawler).

    Args:
        step_rows   : dict step_id → liste de data_rows captées
        output_path : chemin du CSV à écrire (créé si absent)

    Returns:
        La liste de dicts au format final (utile pour les tests ou
        pour brancher directement sur pandas sans repasser par disk).
    """

    s2Rows = step_rows.get('s2', [])
    s3Rows = step_rows.get('s3', [])
    s4Rows = step_rows.get('s4', [])
    s5Rows = step_rows.get('s5', [])

    # Indexes pour les lookups O(1)
    s2ByHref      = _indexBy(s2Rows, 'href')
    s3ByHref      = _indexBy(s3Rows, 'href')
    s5BySourceUrl = _indexBy(s5Rows, 'source_url')

    finalRows: list[dict] = []

    for s4 in s4Rows:
        # Le lien équidé est la clé qui fait le pont vers la fiche s5
        equidé_url   = _normUrl(s4.get('Équidé_url'))
        s5           = s5BySourceUrl.get(equidé_url, {})

        # Pour remonter à s3 (Discipline/Épreuve), on lookup l'URL parent
        s4_sourceUrl = _normUrl(s4.get('source_url'))
        s3           = s3ByHref.get(s4_sourceUrl, {})

        # Idem pour s2 (Numéro du concours)
        s3_sourceUrl = _normUrl(s3.get('source_url'))
        s2           = s2ByHref.get(s3_sourceUrl, {})

        finalRows.append({
            's2_Numéro':          _pickValue(s2, 'Numéro', 'Numero'),
            's3_Discipline':      _pickValue(s3, 'Discipline'),
            's3_Épreuve':         _pickValue(s3, 'Épreuve', 'Epreuve'),
            's4_Clt.':            _pickValue(s4, 'Clt.', 'Clt'),
            's4_Cavalier':        _pickValue(s4, 'Cavalier'),
            's4_Club engageur':   _pickValue(s4, 'Club engageur'),
            's4_Équidé':          _pickValue(s4, 'Équidé', 'Equide'),
            's4_Pts qualif. Chpt':_pickValue(s4, 'Pts qualif. Chpt'),
            's4_Équipe':          _pickValue(s4, 'Équipe', 'Equipe'),
            's5_Robe':            _pickValue(s5, 'Robe'),
            's5_Sexe':            _pickValue(s5, 'Sexe'),
            's5_Taille':          _pickValue(s5, 'Taille'),
            's5_Père':            _pickValue(s5, 'Père', 'Pere'),
            's5_Mère':            _pickValue(s5, 'Mère', 'Mere'),
        })

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents = True, exist_ok = True)
        # utf-8-sig : on garde le BOM comme le reste de la chaîne crawler
        # pour qu'Excel ouvre directement sans corrompre les accents.
        with path.open('w', encoding = 'utf-8-sig', newline = '') as fp:
            writer = csv.DictWriter(fp, fieldnames = list(DATASET_COLUMNS))
            writer.writeheader()
            writer.writerows(finalRows)

    return finalRows


# ──────────────────────────────────────────────
# Builder pour les concours FUTURS (sans s5, sans Clt./Pts)
# ──────────────────────────────────────────────
def buildDatasetFuture(
    step_rows:   dict[str, list[dict]],
    output_path: Path | str | None = None,
) -> list[dict]:
    """
    Variante "concours futurs" du builder : on n'a pas s5 (fiche équidé
    authentifiée) ni les colonnes de résultat (Clt., Pts qualif. Chpt).
    Mêmes joins que buildDatasetBrutV2 mais s5_* sont remplis avec
    'NONE' et Clt./Pts qualif aussi.

    Logique de JOIN identique :
      • s4 (engagements) reste la table centrale
      • s4 ⋈ s3 sur (s4.source_url == s3.href)
      • s3 ⋈ s2 sur (s3.source_url == s2.href)

    Returns: liste de dicts au format DATASET_COLUMNS (avec 'NONE' pour
             les colonnes manquantes — prêt à être chaîné à un prédicteur).
    """
    s2Rows = step_rows.get('s2', [])
    s3Rows = step_rows.get('s3', [])
    s4Rows = step_rows.get('s4', [])

    s2ByHref = _indexBy(s2Rows, 'href')
    s3ByHref = _indexBy(s3Rows, 'href')

    finalRows: list[dict] = []
    for s4 in s4Rows:
        s4_sourceUrl = _normUrl(s4.get('source_url'))
        s3           = s3ByHref.get(s4_sourceUrl, {})
        s3_sourceUrl = _normUrl(s3.get('source_url'))
        s2           = s2ByHref.get(s3_sourceUrl, {})

        finalRows.append({
            's2_Numéro':           _pickValue(s2, 'Numéro', 'Numero'),
            's3_Discipline':       _pickValue(s3, 'Discipline'),
            's3_Épreuve':          _pickValue(s3, 'Épreuve', 'Epreuve'),
            's4_Clt.':             'NONE',                                # futur → inconnu
            's4_Cavalier':         _pickValue(s4, 'Cavalier'),
            's4_Club engageur':    _pickValue(s4, 'Club engageur'),
            's4_Équidé':           _pickValue(s4, 'Équidé', 'Equide'),
            's4_Pts qualif. Chpt': 'NONE',                                # futur → inconnu
            's4_Équipe':           _pickValue(s4, 'Équipe', 'Equipe'),
            's5_Robe':             'NONE',
            's5_Sexe':             'NONE',
            's5_Taille':           'NONE',
            's5_Père':             'NONE',
            's5_Mère':             'NONE',
        })

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents = True, exist_ok = True)
        with path.open('w', encoding = 'utf-8-sig', newline = '') as fp:
            writer = csv.DictWriter(fp, fieldnames = list(DATASET_COLUMNS))
            writer.writeheader()
            writer.writerows(finalRows)

    return finalRows
