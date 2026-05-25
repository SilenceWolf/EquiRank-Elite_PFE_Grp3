# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

from abc import ABC, abstractmethod
from bs4 import Tag


class BaseCrawlerExtractor(ABC):
    """
    Contrat commun pour tous les extracteurs.

    can_handle() permet au Crawler de demander à chaque extracteur
    s'il est pertinent AVANT d'appeler extract(). C'est plus propre
    que de tout essayer aveuglément et de gérer des listes vides partout.
    """

    @abstractmethod
    def can_handle(self, element: Tag) -> bool:
        """Retourne True si cet extracteur a quelque chose à faire sur cet élément."""
        ...

    @abstractmethod
    def extract(self, element: Tag, url: str) -> list[dict]:
        """
        Extrait les données de l'élément et retourne une liste de lignes.
        Chaque ligne est un dict avec des clés homogènes :
          - source_url   : URL d'origine
          - extractor    : nom de l'extracteur
          - type         : nature de la donnée (text, link, row, event, field, media)
          - tag          : balise HTML de l'élément source
          - content      : valeur principale (texte, href, src...)
          - extra        : données complémentaires sérialisées en str (title, alt, etc.)
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
