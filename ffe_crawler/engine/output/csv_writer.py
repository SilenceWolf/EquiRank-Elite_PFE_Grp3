# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

import csv
import re
from pathlib import Path


# Colonnes prioritaires dans le CSV — dans cet ordre
# Les colonnes internes (step_id, extractor, tag, extra) sont exclues
# "text" a été retiré : les rows utilisent maintenant des colonnes
# nommées d'après data-label/th/class, pas un "text" fourre-tout.
CSV_COLUMNS = [
    'href',
    'image',
    'source_url',
]

# Colonnes à exclure du CSV (internes au crawler)
_EXCLUDED = {'step_id', 'extractor', 'type', 'tag', 'extra'}

# Sentinels de remplissage quand une colonne est présente dans certaines
# rows mais pas dans d'autres. Numérique = 0, textuel = 'NONE'.
# Pour les colonnes "classement / points" : vide = vraiment 0 point
# (cas légitime), pas une anomalie. Les -1 sont injectés en amont
# (crawler.py `_force_ranking_value`) UNIQUEMENT sur les valeurs
# textuelles anormales (Cavalier inconnu, Niveau trop élevé, etc.)
# et les éliminés. La distinction est claire : 0 = pas de point ;
# -1 = anomalie / pas de résultat exploitable.
_NUMERIC_FILL = '0'
_TEXT_FILL    = 'NONE'

_NUM_RE = re.compile(r'^-?\d+(?:\.\d+)?$')


def _is_numeric(value: str) -> bool:
    return bool(_NUM_RE.match(value.strip())) if value else False


def _detect_numeric_columns(rows: list[dict], cols: list[str]) -> set[str]:
    """Une colonne est considérée numérique si TOUTES ses valeurs
    non-vides parsent en nombre. On utilise ça pour choisir le bon
    sentinel de remplissage (-1 vs NONE)."""
    numeric: set[str] = set()
    for col in cols:
        values = [str(r.get(col, '')).strip() for r in rows]
        non_empty = [v for v in values if v]
        if non_empty and all(_is_numeric(v) for v in non_empty):
            numeric.add(col)
    return numeric


class CsvWriter:
    """
    Écrit une liste de lignes (dicts) dans un fichier CSV.
    N'inclut que les colonnes utiles, exclut les colonnes internes.

    Remplit les cellules manquantes avec -1 (colonnes numériques) ou
    'NONE' (colonnes textuelles) pour que les consommateurs externes
    distinguent "donnée inexistante" d'un vrai vide.
    """

    def write(self, rows: list[dict], dest: Path) -> None:
        if not rows:
            return

        # Colonnes dynamiques : celles des données qui ne sont pas déjà listées
        extra_cols = []
        for row in rows:
            for key in row:
                if key not in CSV_COLUMNS and key not in extra_cols and key not in _EXCLUDED:
                    extra_cols.append(key)

        all_cols = [c for c in CSV_COLUMNS if any(c in r for r in rows)] + extra_cols
        numeric_cols = _detect_numeric_columns(rows, all_cols)

        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                out_row = {}
                for col in all_cols:
                    v = row.get(col, '')
                    if v is None or str(v).strip() == '':
                        out_row[col] = _NUMERIC_FILL if col in numeric_cols else _TEXT_FILL
                    else:
                        out_row[col] = v
                writer.writerow(out_row)
