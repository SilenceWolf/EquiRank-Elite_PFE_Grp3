# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

from urllib.parse import urljoin
from bs4 import Tag
from .base_extractor import BaseCrawlerExtractor


class MediaExtractor(BaseCrawlerExtractor):
    """
    Extrait les médias : images, vidéos, audios, iframes, et sources lazily loaded.
    Les URLs sont résolues en absolues. Le téléchargement effectif se fait
    dans OutputManager si l'option --download-media est activée.
    """

    def can_handle(self, element: Tag) -> bool:
        return bool(element.find(['img', 'video', 'audio', 'iframe', 'source', 'picture']))

    def extract(self, element: Tag, url: str) -> list[dict]:
        rows: list[dict] = []

        # Images — on gère aussi le lazy loading (data-src, data-lazy)
        for img in element.find_all('img'):
            src = (
                img.get('src')
                or img.get('data-src')
                or img.get('data-lazy')
                or img.get('data-original')
                or ''
            ).strip()

            if not src or src.startswith('data:'):
                continue

            rows.append({
                'source_url': url,
                'extractor':  self.name,
                'type':       'image',
                'tag':        'img',
                'content':    urljoin(url, src),
                'extra': (
                    f'alt={img.get("alt", "")} | '
                    f'title={img.get("title", "")} | '
                    f'width={img.get("width", "")} | '
                    f'height={img.get("height", "")}'
                ),
            })

        # <picture> avec sources multiples
        for picture in element.find_all('picture'):
            for source in picture.find_all('source'):
                srcset = source.get('srcset', '').split(',')
                for s in srcset:
                    src = s.strip().split(' ')[0]
                    if src:
                        rows.append({
                            'source_url': url,
                            'extractor':  self.name,
                            'type':       'image_srcset',
                            'tag':        'source',
                            'content':    urljoin(url, src),
                            'extra':      f'media={source.get("media", "")} | type={source.get("type", "")}',
                        })

        # Vidéos
        for video in element.find_all('video'):
            src = video.get('src', '').strip()
            poster = video.get('poster', '').strip()
            sources = [s.get('src', '').strip() for s in video.find_all('source') if s.get('src')]

            all_srcs = ([src] if src else []) + sources
            for s in all_srcs:
                if s:
                    rows.append({
                        'source_url': url,
                        'extractor':  self.name,
                        'type':       'video',
                        'tag':        'video',
                        'content':    urljoin(url, s),
                        'extra':      f'poster={urljoin(url, poster) if poster else ""}',
                    })

        # Audios
        for audio in element.find_all('audio'):
            src = audio.get('src', '').strip()
            sources = [s.get('src', '').strip() for s in audio.find_all('source') if s.get('src')]
            for s in ([src] if src else []) + sources:
                if s:
                    rows.append({
                        'source_url': url,
                        'extractor':  self.name,
                        'type':       'audio',
                        'tag':        'audio',
                        'content':    urljoin(url, s),
                        'extra':      '',
                    })

        # Iframes (contenu embarqué — YouTube, cartes, etc.)
        for iframe in element.find_all('iframe'):
            src = iframe.get('src', '').strip()
            if src:
                rows.append({
                    'source_url': url,
                    'extractor':  self.name,
                    'type':       'iframe',
                    'tag':        'iframe',
                    'content':    urljoin(url, src),
                    'extra': (
                        f'title={iframe.get("title", "")} | '
                        f'width={iframe.get("width", "")} | '
                        f'height={iframe.get("height", "")}'
                    ),
                })

        return rows
