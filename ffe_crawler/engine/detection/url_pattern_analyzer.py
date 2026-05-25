# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# URLPatternAnalyzer : analyse un lot d'URLs et les regroupe par
# similarité de pattern (même domaine + même template de path).
#
# Pas de fetch, pas de DOM — c'est purement syntaxique et donc gratuit.
# Complémentaire à StructureComparator qui, lui, fait un fetch pour
# vérifier la similarité DOM.
#
# Usage côté frontend : après une étape qui produit N liens, on appelle
# /crawl/analyze-urls pour savoir si ces liens ont une structure
# d'adresse similaire → si oui, on propose "crawler tous en répétition
# avec la même config".

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing      import Any
from urllib.parse import urlparse


# Segments "variables" typiques d'une URL : on les remplace par un
# placeholder dans le template pour pouvoir grouper par pattern.
_NUMERIC_ID    = re.compile(r'^\d+$')
_HEX_ID        = re.compile(r'^[0-9a-f]{8,}$', re.I)
_UUID          = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
_SLUG_WITH_ID  = re.compile(r'^[\w-]{8,}-\d+$')   # ex: "my-product-12345"


@dataclass
class URLPatternGroup:
    """Un cluster d'URLs partageant le même template."""
    pattern: str               # template lisible, ex: "https://ex.com/item/:id"
    urls:    list[str]         # URLs réelles dans ce groupe
    count:   int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class URLPatternResult:
    """Résultat complet de l'analyse."""
    groups:               list[URLPatternGroup] = field(default_factory=list)
    dominant_group_idx:   int   = 0
    similarity:           float = 0.0    # 0..1, proportion dans le groupe dominant
    total_urls:           int   = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'groups':             [g.to_dict() for g in self.groups],
            'dominant_group_idx': self.dominant_group_idx,
            'similarity':         self.similarity,
            'total_urls':         self.total_urls,
        }


class URLPatternAnalyzer:
    """
    Analyse un lot d'URLs et produit un clustering par pattern.

    Pas d'état — on peut appeler analyze() plusieurs fois sans soucis.
    """

    def analyze(self, urls: list[str]) -> URLPatternResult:
        if not urls:
            return URLPatternResult(total_urls=0)

        # Calcule un template par URL, puis groupe par template
        groups: dict[str, list[str]] = defaultdict(list)
        for raw in urls:
            tpl = self._templatize(raw)
            if tpl:
                groups[tpl].append(raw)

        # Transforme en URLPatternGroup, tri par taille décroissante
        group_list = [
            URLPatternGroup(pattern=tpl, urls=list(us), count=len(us))
            for tpl, us in groups.items()
        ]
        group_list.sort(key=lambda g: g.count, reverse=True)

        total = sum(g.count for g in group_list)
        if total == 0:
            return URLPatternResult(total_urls=0)

        similarity = group_list[0].count / total

        return URLPatternResult(
            groups             = group_list,
            dominant_group_idx = 0,
            similarity         = round(similarity, 3),
            total_urls         = total,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _templatize(self, url: str) -> str:
        """
        Réduit une URL à un template :
          'https://ex.com/product/12345?lang=fr' → 'https://ex.com/product/:id?lang=:val'
          'https://ex.com/user/abc123/profile'   → 'https://ex.com/user/:id/profile'

        Seuls le domaine et la structure des segments sont conservés.
        Les IDs (numériques, hex, UUID) deviennent ':id'. Les autres
        segments restent tels quels — on préserve la distinction
        /products/ vs /categories/.
        """
        if not url:
            return ''
        try:
            parsed = urlparse(url)
        except ValueError:
            return ''
        if not parsed.netloc:
            return ''

        # Path : segment par segment, on templatise les IDs
        segments = [s for s in parsed.path.split('/') if s]
        tpl_segments: list[str] = []
        for seg in segments:
            if _UUID.match(seg) or _HEX_ID.match(seg) or _NUMERIC_ID.match(seg) or _SLUG_WITH_ID.match(seg):
                tpl_segments.append(':id')
            else:
                tpl_segments.append(seg)
        tpl_path = '/' + '/'.join(tpl_segments) if tpl_segments else '/'

        # Query : on garde les clés mais on efface les valeurs
        tpl_query = ''
        if parsed.query:
            keys = sorted({k for k, _ in _parse_qs(parsed.query)})
            tpl_query = '?' + '&'.join(f'{k}=:val' for k in keys)

        return f'{parsed.scheme}://{parsed.netloc}{tpl_path}{tpl_query}'


def _parse_qs(query: str) -> list[tuple[str, str]]:
    """Parse trivial de query string sans décoder — suffisant pour grouper."""
    pairs: list[tuple[str, str]] = []
    for chunk in query.split('&'):
        if not chunk:
            continue
        if '=' in chunk:
            k, v = chunk.split('=', 1)
        else:
            k, v = chunk, ''
        pairs.append((k, v))
    return pairs
