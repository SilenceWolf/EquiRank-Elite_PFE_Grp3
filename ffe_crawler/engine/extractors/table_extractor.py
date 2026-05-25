# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

import json
from bs4 import Tag
from .base_extractor import BaseCrawlerExtractor


class TableExtractor(BaseCrawlerExtractor):
    """
    Extrait les données tabulaires : <table> classiques et listes de lignes
    structurées (ul/li, dl/dt+dd).
    Chaque ligne du tableau devient une ligne dans le CSV avec
    ses colonnes sérialisées en JSON dans 'content'.
    """

    def can_handle(self, element: Tag) -> bool:
        return bool(
            element.find('table')
            or element.name == 'table'
        )

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []

        tables = (
            [element] if element.name == 'table'
            else element.find_all('table')
        )

        for table_idx, table in enumerate(tables):
            headers  = _extract_headers(table)
            data_rows = _extract_rows(table)

            for row_idx, row_cells in enumerate(data_rows):
                # On associe les headers si possible, sinon colonnes numérotées
                if headers and len(headers) == len(row_cells):
                    row_dict = dict(zip(headers, row_cells))
                elif headers:
                    row_dict = {h: row_cells[i] if i < len(row_cells) else ''
                                for i, h in enumerate(headers)}
                else:
                    row_dict = {f'col_{i}': v for i, v in enumerate(row_cells)}

                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'table_row',
                    'tag':        'table',
                    'content':    json.dumps(row_dict, ensure_ascii=False),
                    'extra':      f'table={table_idx} row={row_idx}',
                })

        return rows


def _extract_headers(table: Tag) -> list[str]:
    """Cherche les <th> dans le thead ou la première ligne."""
    thead = table.find('thead')
    source = thead if thead else table
    headers = [th.get_text(strip=True) for th in source.find_all('th')]
    return headers if headers else []


def _extract_rows(table: Tag) -> list[list[str]]:
    tbody = table.find('tbody') or table
    rows: list[list[str]] = []
    for tr in tbody.find_all('tr'):
        cells = [td.get_text(separator=' ', strip=True) for td in tr.find_all(['td', 'th'])]
        # On ignore les lignes vides et les lignes de headers déjà traitées
        if cells and any(c for c in cells):
            rows.append(cells)
    return rows
