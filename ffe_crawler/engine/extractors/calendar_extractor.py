# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

import re
from bs4 import Tag
from .base_extractor import BaseCrawlerExtractor


# Patterns caractéristiques des calendriers web les plus répandus
# FullCalendar, Google Calendar, dhtmlxScheduler, etc.
_CALENDAR_CLASS_PATTERNS = re.compile(
    r'(fc-|cal-|calendar|event|rdp-|flatpickr|datepicker|scheduler|dhx)',
    re.IGNORECASE,
)

_DATE_ATTRS = ('data-date', 'data-day', 'data-timestamp', 'datetime', 'data-start', 'data-event')


class CalendarExtractor(BaseCrawlerExtractor):
    """
    Extrait les événements et dates depuis les interfaces calendrier.
    Reconnaît les patterns de classe CSS des bibliothèques calendrier courantes
    et les attributs data-* portant des informations de date.

    Note : pour un calendrier dynamique (qui charge les événements en JS),
    il faut utiliser le mode playwright dans le Crawler afin que les
    événements soient effectivement rendus dans le DOM avant l'extraction.
    """

    def can_handle(self, element: Tag) -> bool:
        # On cherche des indicateurs calendrier dans l'élément ou ses descendants
        return (
            _has_calendar_classes(element)
            or bool(element.find(attrs={attr: True for attr in _DATE_ATTRS}))
            or bool(element.find('time'))
        )

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []

        # Événements FullCalendar (fc-event, fc-title, etc.)
        for event_el in element.find_all(class_=re.compile(r'fc-event|event-item|rdp-day', re.I)):
            rows.append(_make_event_row(event_el, url, self.name, source='class-pattern'))

        # Éléments avec attributs de date explicites
        for attr in _DATE_ATTRS:
            for el in element.find_all(attrs={attr: True}):
                if el in [r.get('_el') for r in rows]:  # éviter les doublons
                    continue
                row = _make_event_row(el, url, self.name, source=f'attr:{attr}')
                row['extra'] = f'{attr}={el[attr]} | ' + row.get('extra', '')
                rows.append(row)

        # Balises <time> isolées
        for time_el in element.find_all('time'):
            rows.append({
                'source_url': url,
                'extractor':  self.name,
                'type':       'calendar_time',
                'tag':        'time',
                'content':    time_el.get_text(strip=True),
                'extra':      f'datetime={time_el.get("datetime", "")}',
            })

        return rows


def _has_calendar_classes(el: Tag) -> bool:
    classes = el.get('class', [])
    if isinstance(classes, str):
        classes = classes.split()
    return any(_CALENDAR_CLASS_PATTERNS.search(c) for c in classes)


def _make_event_row(el: Tag, url: str, extractor_name: str, source: str) -> dict:
    text = el.get_text(separator=' ', strip=True)

    # Titre de l'événement : souvent dans un élément enfant avec "title" dans la classe
    title_el = el.find(class_=re.compile(r'title|name|summary', re.I))
    title    = title_el.get_text(strip=True) if title_el else text

    # Date : cherche l'attribut de date le plus proche
    date_val = ''
    for attr in _DATE_ATTRS:
        if el.get(attr):
            date_val = el[attr]
            break

    return {
        'source_url': url,
        'extractor':  extractor_name,
        'type':       'calendar_event',
        'tag':        el.name,
        'content':    title,
        'extra':      f'date={date_val} | full_text={text[:120]} | source={source}',
    }
