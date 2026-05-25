# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

from urllib.parse import urljoin
from bs4 import Tag
from .base_extractor import BaseCrawlerExtractor


class LinkExtractor(BaseCrawlerExtractor):
    """
    Extrait tous les liens cliquables : <a href>, boutons avec data-href,
    et éléments avec un attribut onclick contenant une URL.
    On résout les URLs relatives par rapport à l'URL d'origine.
    """

    def can_handle(self, element: Tag) -> bool:
        return bool(
            element.find('a', href=True)
            or element.find(attrs={'data-href': True})
            or element.get('href')
        )

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []

        # <a href="...">
        for anchor in element.find_all('a', href=True):
            href = anchor['href'].strip()
            if not href or href.startswith('#') and len(href) == 1:
                continue

            absolute_href = urljoin(url, href)

            rows.append({
                'source_url': url,
                'extractor':  self.name,
                'type':       'link',
                'tag':        'a',
                'content':    absolute_href,
                'extra':      _link_extra(anchor),
            })

        # Éléments avec data-href (pattern courant dans les frameworks JS)
        for el in element.find_all(attrs={'data-href': True}):
            data_href = el['data-href'].strip()
            if not data_href:
                continue
            rows.append({
                'source_url': url,
                'extractor':  self.name,
                'type':       'link',
                'tag':        el.name,
                'content':    urljoin(url, data_href),
                'extra':      f'text={el.get_text(strip=True)} | via=data-href',
            })

        # L'élément racine lui-même a un href (ex : <div class="card" href="...">)
        if element.get('href'):
            rows.append({
                'source_url': url,
                'extractor':  self.name,
                'type':       'link',
                'tag':        element.name,
                'content':    urljoin(url, element['href']),
                'extra':      f'text={element.get_text(strip=True)}',
            })

        return rows


def _link_extra(anchor: Tag) -> str:
    parts = []
    text = anchor.get_text(strip=True)
    if text:
        parts.append(f'text={text}')
    if anchor.get('title'):
        parts.append(f'title={anchor["title"]}')
    if anchor.get('rel'):
        rel = ' '.join(anchor['rel']) if isinstance(anchor['rel'], list) else anchor['rel']
        parts.append(f'rel={rel}')
    if anchor.get('target'):
        parts.append(f'target={anchor["target"]}')
    return ' | '.join(parts)
