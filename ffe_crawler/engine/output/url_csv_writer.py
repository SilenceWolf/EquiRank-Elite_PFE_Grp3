# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# CSV dédié aux URLs extraites. Contient le texte associé + le lien.

import csv
from pathlib import Path


class UrlCsvWriter:
    """
    Écrit un CSV contenant uniquement les URLs extraites.
    Colonnes : text, href, source_url.
    """

    def write(
        self,
        rows:        list[dict],
        url_columns: list[str],
        dest:        Path,
    ) -> None:
        if not rows:
            return

        cols = ['text', 'href', 'source_url']

        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                href = row.get('href', '')
                if not href:
                    continue
                writer.writerow({col: row.get(col, '') for col in cols})
