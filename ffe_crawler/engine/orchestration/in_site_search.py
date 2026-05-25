# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# InSiteSearchRunner : alimente le formulaire de recherche d'un site avec
# les valeurs d'une colonne d'un CSV produit par une étape précédente.
#
# Cas d'usage type : on a extrait une liste de noms de chevaux en step 1,
# on veut faire une recherche pour chacun sur un second site en step 2.
# Le cache in-memory évite les requêtes doublons quand la même valeur
# revient plusieurs fois (généralise le horse_cache de l'exemple IFCE).

from __future__ import annotations

import csv
from pathlib  import Path
from typing   import Any, Callable
from urllib.parse import urlencode, urlparse, urlunparse

from log     import CrawlerLogger
from crawler import Crawler

from session.step import CrawlStep, InSiteSearchSpec


PublishFn = Callable[[dict[str, Any], str], None]


class InSiteSearchRunner:
    """
    Pour chaque valeur d'une colonne source, exécute une recherche sur
    un site cible et crawle la page de résultats.
    """

    def __init__(
        self,
        crawler: Crawler | None = None,
        publish: PublishFn | None = None,
    ) -> None:
        self._crawler = crawler or Crawler()
        self._publish = publish or (lambda evt, sid: None)
        self._log     = CrawlerLogger.get_instance()
        self._cache: dict[str, list[dict]] = {}   # clé=valeur → rows

    def run(
        self,
        spec:       InSiteSearchSpec,
        step:       CrawlStep,
        session_id: str,
    ) -> list[dict]:
        """
        Prend les valeurs (de spec.values ou en les lisant depuis
        spec.source_csv), fait une recherche par valeur sur le site
        cible et retourne la liste cumulée des rows extraites.
        """
        # Mode 1 : valeurs fournies directement par le front
        if spec.values:
            values = self._dedupe(spec.values)
            self._log.info(
                f'In-site search : {len(values)} valeur(s) uniques (fournies)'
            )
        # Mode 2 : relecture depuis un CSV sur disque (replay)
        elif spec.source_csv and spec.source_column:
            values = self._load_column(Path(spec.source_csv), spec.source_column)
            self._log.info(
                f'In-site search : {len(values)} valeur(s) uniques depuis '
                f'{Path(spec.source_csv).name}.{spec.source_column}'
            )
        else:
            self._log.error('In-site search : ni values ni source_csv fournis')
            return []

        all_rows: list[dict] = []

        for value in values:
            if not value:
                continue

            if value in self._cache:
                rows = self._cache[value]
            else:
                target_url = self._build_search_url(spec, value)
                try:
                    result = self._crawler.crawl(target_url, step.data_selector or 'body')
                    rows = list(result.rows)
                except Exception as exc:
                    self._log.warn(f'  échec recherche "{value}" : {exc}')
                    rows = []
                self._cache[value] = rows

            for row in rows:
                row['source_query'] = value
                row['step_id']      = step.step_id
                all_rows.append(row)
                self._publish({
                    'type':    'row',
                    'step_id': step.step_id,
                    'row':     row,
                }, session_id)

        return all_rows

    # ── Helpers ────────────────────────────────────────────────────────────

    def _dedupe(self, raw_values: list[str]) -> list[str]:
        """Déduplique en préservant l'ordre et en virant les vides."""
        seen: set[str] = set()
        out:  list[str] = []
        for v in raw_values:
            s = (v or '').strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _load_column(self, csv_path: Path, column: str) -> list[str]:
        """Charge une colonne d'un CSV en dédupliquant les valeurs."""
        if not csv_path.exists():
            self._log.error(f'CSV source introuvable : {csv_path}')
            return []

        values: list[str] = []
        seen:   set[str]  = set()
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get(column) or '').strip()
                if raw and raw not in seen:
                    seen.add(raw)
                    values.append(raw)
        return values

    def _build_search_url(self, spec: InSiteSearchSpec, value: str) -> str:
        """
        Construit l'URL de recherche. On ne gère pour l'instant que le cas
        GET : on ajoute (ou remplace) le query param spec.field_name.
        Le cas POST nécessiterait playwright pour submit le form — TODO.
        """
        parsed = urlparse(spec.search_url)
        # On reconstruit le query string en ajoutant/remplaçant le champ
        from urllib.parse import parse_qsl
        existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
        existing[spec.field_name] = value
        new_query = urlencode(existing)
        return urlunparse(parsed._replace(query=new_query))
