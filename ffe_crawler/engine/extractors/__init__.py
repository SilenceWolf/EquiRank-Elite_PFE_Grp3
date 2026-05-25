# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

from .base_extractor   import BaseCrawlerExtractor
from .text_extractor   import TextExtractor
from .link_extractor   import LinkExtractor
from .table_extractor  import TableExtractor
from .calendar_extractor import CalendarExtractor
from .form_extractor   import FormExtractor
from .media_extractor  import MediaExtractor

# Ordre d'application : du plus spécialisé au plus générique.
# TextExtractor passe en dernier — il est le filet de sécurité qui
# capture tout ce que les autres n'ont pas traité.
DEFAULT_EXTRACTORS: list[BaseCrawlerExtractor] = [
    CalendarExtractor(),
    FormExtractor(),
    TableExtractor(),
    MediaExtractor(),
    LinkExtractor(),
    TextExtractor(),
]

__all__ = [
    'BaseCrawlerExtractor',
    'TextExtractor',
    'LinkExtractor',
    'TableExtractor',
    'CalendarExtractor',
    'FormExtractor',
    'MediaExtractor',
    'DEFAULT_EXTRACTORS',
]
