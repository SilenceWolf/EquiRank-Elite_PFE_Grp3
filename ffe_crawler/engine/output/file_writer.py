# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from .csv_writer import CsvWriter

if TYPE_CHECKING:
    from ..crawler import CrawlResult

# Racine des résultats — toujours à côté du dossier outils/
CRAWLRESULT_ROOT = Path(__file__).resolve().parents[3] / 'crawlresult'


class FileWriter:
    """Gère l'écriture des fichiers bruts (HTML, médias) dans crawlresult/."""

    def save_raw_html(self, html: str, dest_dir: Path, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        raw_dir  = dest_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest_file = raw_dir / f'{url_hash}.html'
        dest_file.write_text(html, encoding='utf-8')
        return dest_file

    def download_media(self, media_url: str, dest_dir: Path) -> Path | None:
        """Télécharge un fichier média et le sauvegarde dans dest_dir/media/."""
        media_dir = dest_dir / 'media'
        media_dir.mkdir(parents=True, exist_ok=True)

        # Nom de fichier depuis l'URL — on sanitize les caractères interdits
        filename  = re.sub(r'[^\w.\-]', '_', Path(media_url.split('?')[0]).name)
        dest_file = media_dir / filename

        try:
            try:
                from ..crawler import _get_proxy_dict
                _proxies = _get_proxy_dict()
            except Exception:
                _proxies = None
            resp = requests.get(media_url, stream=True, proxies=_proxies, timeout=15)
            resp.raise_for_status()
            with dest_file.open('wb') as f:
                shutil.copyfileobj(resp.raw, f)
            return dest_file
        except Exception:
            return None


class OutputManager:
    """
    Point d'entrée unique pour sauvegarder les résultats d'une session de crawl.
    Crée la structure crawlresult/{slug}/{timestamp}/ et y écrit tout.
    """

    def __init__(self) -> None:
        self._csv    = CsvWriter()
        self._writer = FileWriter()

    def save(
        self,
        results:         list['CrawlResult'],
        query_slug:      str,
        save_raw:        bool = False,
        download_media:  bool = False,
    ) -> Path:
        """
        Sauvegarde tous les résultats dans le dossier dédié.
        Retourne le chemin du dossier de session créé.
        """
        timestamp   = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        session_dir = CRAWLRESULT_ROOT / _slugify(query_slug) / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)

        # Agrège toutes les lignes de tous les résultats dans un seul CSV
        all_rows: list[dict] = []
        for result in results:
            all_rows.extend(result.rows)

        csv_path = session_dir / 'data.csv'
        self._csv.write(all_rows, csv_path)

        # HTML brut si demandé
        if save_raw:
            for result in results:
                if result.raw_html:
                    self._writer.save_raw_html(result.raw_html, session_dir, result.url)

        # Téléchargement des médias si demandé
        if download_media:
            media_urls = {
                row['content']
                for row in all_rows
                if row.get('type') in ('image', 'video', 'audio', 'image_srcset')
                and row.get('content', '').startswith('http')
            }
            for media_url in media_urls:
                self._writer.download_media(media_url, session_dir)

        return session_dir


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://', '', text)
    text = re.sub(r'[^\w\-]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')[:60]
