# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Enveloppe fine autour d'une CrawlChain pour la persistance JSON.
# Délègue tout le stockage au SessionStore — la séparation garantit qu'on
# peut changer de backend (fichier → base → Redis) sans toucher à l'API.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing      import Any

from log import CrawlerLogger

from .chain           import CrawlChain
from storage.session_store import SessionStore


@dataclass
class SessionListItem:
    """Entrée légère pour lister les sessions sans charger tout le JSON."""
    chain_id:    str
    name:        str
    created_at:  str
    steps_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'chain_id':    self.chain_id,
            'name':        self.name,
            'created_at':  self.created_at,
            'steps_count': self.steps_count,
        }


class CrawlSession:
    """
    API de haut niveau pour save/load/list/delete les chaînes de crawl.
    Ne contient aucune logique d'exécution — c'est le ChainRunner qui joue
    la chain une fois chargée.
    """

    def __init__(self, store: SessionStore | None = None) -> None:
        self._store = store or SessionStore()
        self._log   = CrawlerLogger.get_instance()

    def save(self, chain: CrawlChain) -> str:
        """Persiste la chain sur disque et retourne son chain_id."""
        path = self._store.path_for(chain.chain_id)
        path.write_text(
            json.dumps(chain.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        self._log.success(
            f'Session sauvegardée : {chain.name} ({len(chain.steps)} étape(s))'
        )
        return chain.chain_id

    def load(self, chain_id: str) -> CrawlChain:
        """Charge une chain depuis son id — lève FileNotFoundError si absente."""
        path = self._store.path_for(chain_id)
        if not path.exists():
            raise FileNotFoundError(f'Session introuvable : {chain_id}')
        data = json.loads(path.read_text(encoding='utf-8'))
        return CrawlChain.from_dict(data)

    def list(self) -> list[SessionListItem]:
        """
        Liste toutes les sessions stockées. On parse chaque JSON pour
        récupérer name/created_at/steps_count — c'est linéaire en nombre
        de sessions, acceptable pour un usage mono-utilisateur.
        """
        items: list[SessionListItem] = []
        for path in self._store.all_paths():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                items.append(SessionListItem(
                    chain_id    = data.get('chain_id', path.stem),
                    name        = data.get('name', path.stem),
                    created_at  = data.get('created_at', ''),
                    steps_count = len(data.get('steps', [])),
                ))
            except (json.JSONDecodeError, OSError) as exc:
                # On skip les fichiers corrompus mais on log pour trace
                self._log.warn(f'Session illisible {path.name} : {exc}')
        # Plus récent en premier (ordre décroissant de date)
        items.sort(key=lambda it: it.created_at, reverse=True)
        return items

    def delete(self, chain_id: str) -> bool:
        path = self._store.path_for(chain_id)
        if not path.exists():
            return False
        path.unlink()
        self._log.info(f'Session supprimée : {chain_id}')
        return True
