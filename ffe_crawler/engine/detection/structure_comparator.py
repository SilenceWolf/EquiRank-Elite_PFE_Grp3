# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# StructureComparator : compare la structure DOM de N URLs pour décider
# si elles peuvent être crawlées avec la même config. Si la similarité
# est élevée (>0.85 par défaut), le front propose "une seule config pour
# toutes". Sinon, il affiche les clusters séparément.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing      import Any

from bs4 import BeautifulSoup

from log     import CrawlerLogger
from crawler import Crawler

from .dom_fingerprint import fingerprint


_SAMPLE_LIMIT = 5   # on ne fetche pas plus de 5 pages — coût réseau


@dataclass
class ComparisonResult:
    """
    Résultat d'une comparaison de structure.

    similarity : 1.0 = toutes identiques, 0.0 = toutes différentes.
    Concrètement : 1 - (nb_clusters / nb_échantillons).

    clusters : groupes d'URLs partageant la même empreinte.
    dominant_cluster_idx : index du cluster le plus gros (= la structure
    "majoritaire" de l'échantillon).
    """

    similarity:           float
    clusters:             list[list[str]] = field(default_factory=list)
    dominant_cluster_idx: int             = 0
    sampled_count:        int             = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructureComparator:
    """
    Compare la structure DOM de plusieurs URLs en fingerprintant un
    échantillon de pages.
    """

    def __init__(self, crawler: Crawler | None = None) -> None:
        self._crawler = crawler or Crawler()
        self._log     = CrawlerLogger.get_instance()

    def compare(self, urls: list[str]) -> ComparisonResult:
        if not urls:
            return ComparisonResult(similarity=1.0, clusters=[], dominant_cluster_idx=0, sampled_count=0)

        sample = urls[:_SAMPLE_LIMIT]
        self._log.info(f'Comparaison de structure sur {len(sample)} URL(s)')

        fingerprints: dict[str, list[str]] = defaultdict(list)
        for u in sample:
            try:
                html = self._crawler._fetch(u)
                soup = BeautifulSoup(html, 'html.parser')
                fp   = fingerprint(soup)
                fingerprints[fp].append(u)
            except Exception as exc:
                self._log.warn(f'  échec fingerprint sur {u} : {exc}')
                # Clé unique → considéré comme son propre cluster
                fingerprints[f'__error_{id(u)}__'].append(u)

        # Les clusters sont les valeurs du dict, triés par taille décroissante
        cluster_list = sorted(fingerprints.values(), key=len, reverse=True)

        # Calcul de similarité : 1 - (distinct / total)
        similarity = 1.0 - (len(cluster_list) / len(sample))

        dominant_idx = 0    # après tri, le cluster 0 est toujours le plus gros

        result = ComparisonResult(
            similarity           = round(similarity, 3),
            clusters             = cluster_list,
            dominant_cluster_idx = dominant_idx,
            sampled_count        = len(sample),
        )

        self._log.success(
            f'  → similarité {result.similarity:.2f}, '
            f'{len(cluster_list)} cluster(s) détecté(s)'
        )
        return result
