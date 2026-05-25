# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Stockage fichier pour les sessions de crawl. Volontairement simple :
# un fichier JSON par session dans crawlresult/sessions/. Pas de DB car
# l'usage est mono-utilisateur et on veut garder les sessions inspectables
# à la main depuis un éditeur de texte.

from __future__ import annotations

from pathlib import Path

from output.file_writer import CRAWLRESULT_ROOT


class SessionStore:
    """
    Manage les fichiers JSON de sessions sur disque.
    Le dossier racine est créé paresseusement à la première écriture.
    """

    def __init__(self, root: Path | None = None) -> None:
        # Par défaut : crawlresult/sessions/ — à côté des résultats de crawl
        self._root = root or (CRAWLRESULT_ROOT / 'sessions')
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, chain_id: str) -> Path:
        """Retourne le chemin du fichier JSON d'une session donnée."""
        # On sanitize le chain_id pour éviter tout path traversal
        safe_id = ''.join(c for c in chain_id if c.isalnum() or c in '-_')
        if not safe_id:
            raise ValueError(f'chain_id invalide : {chain_id!r}')
        return self._root / f'{safe_id}.json'

    def all_paths(self) -> list[Path]:
        """Liste tous les fichiers de session, triés par nom."""
        if not self._root.exists():
            return []
        return sorted(self._root.glob('*.json'))

    def session_dir(self, chain_id: str) -> Path:
        """
        Retourne le dossier où les CSV d'une session sont écrits.
        Distinct du path_for() qui cible le JSON de définition.
        """
        safe_id = ''.join(c for c in chain_id if c.isalnum() or c in '-_')
        path = CRAWLRESULT_ROOT / 'runs' / safe_id
        path.mkdir(parents=True, exist_ok=True)
        return path
