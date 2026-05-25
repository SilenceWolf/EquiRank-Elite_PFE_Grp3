# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Dataclasses décrivant une étape de crawl. Volontairement pures :
# elles servent de "contrat" entre le front (zod), l'API (pydantic) et
# l'orchestration Python. Aucune logique métier ici — juste des données
# sérialisables en JSON.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib     import Path
from typing      import Any


@dataclass
class InSiteSearchSpec:
    """
    Décrit une recherche in-site alimentée par les valeurs d'une colonne
    d'une étape précédente.

    Ex : on a extrait une liste de noms de chevaux en étape 1, on veut
    faire une recherche pour chacun sur un second site en étape 2.

    Deux modes d'alimentation — on prend le premier non-vide :
      1. `values` : liste explicite fournie par le front (cas par défaut
         quand l'utilisateur clique "crawler depuis la colonne X" dans
         la LiveDataTable — on a déjà les valeurs en mémoire)
      2. `source_csv` + `source_column` : lire depuis un CSV sur disque
         (cas replay de session — les valeurs ne sont pas sérialisées)
    """

    search_url:    str                  # URL de la page qui héberge le formulaire
    field_name:    str                  # attribut name du champ à remplir
    form_selector: str = ''             # sélecteur CSS du <form> cible (optionnel pour GET)
    values:        list[str] = field(default_factory=list)
    source_csv:    str = ''             # chemin du CSV source (fallback)
    source_column: str = ''             # colonne à lire dans source_csv
    method:        str = 'GET'          # 'GET' ou 'POST'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'InSiteSearchSpec':
        # Tolérant aux clés manquantes et aux clés inconnues
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


@dataclass
class CrawlStep:
    """
    Une unité de travail dans une chaîne de crawl.

    entry_urls : URLs à crawler pour cette étape. Si vide et parent_step_id
                 non nul, le ChainRunner les remplira depuis les URLs
                 exportées de l'étape parente.

    link_selector / data_selector : sélecteurs CSS pour cibler ce qu'on
                 veut extraire. link_selector est optionnel (utile
                 uniquement si on veut alimenter l'étape suivante).

    recursive_depth : si > 0, le StepRunner instancie un RecursiveExtractor
                 au lieu des DEFAULT_EXTRACTORS, pour capturer des données
                 imbriquées avec des noms de colonnes dynamiques.

    url_export_columns : clés des rows qu'on considère comme des URLs
                 à router vers urls.csv plutôt que data.csv.
    """

    step_id:               str
    entry_urls:            list[str]      = field(default_factory=list)
    parent_step_id:        str | None     = None
    link_selector:         str            = ''
    data_selector:         str            = ''
    recursive_depth:       int            = 0
    capture_headers_from:  str            = 'auto'   # 'th' | 'label' | 'auto'
    field_values:          dict[str, str] = field(default_factory=dict)
    in_site_search:        InSiteSearchSpec | None = None
    url_export_columns:    list[str]      = field(default_factory=list)
    force_dynamic:         bool           = False
    requires_auth:         bool           = False
    auto_uniformize:       bool           = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict sérialise déjà récursivement la dataclass imbriquée
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'CrawlStep':
        # Copie défensive pour ne pas muter l'input
        data = dict(d)

        # Reconstruction de la sous-dataclass si présente
        raw_search = data.get('in_site_search')
        if raw_search is not None and not isinstance(raw_search, InSiteSearchSpec):
            data['in_site_search'] = InSiteSearchSpec.from_dict(raw_search)

        # Ignore silencieusement les clés inconnues pour être tolérant
        # aux évolutions de schéma côté front
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered   = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
