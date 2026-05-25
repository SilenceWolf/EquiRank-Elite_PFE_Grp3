# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# File d'attente FIFO pour les runs (steps + chains). Un seul worker
# consomme la queue à la fois, ce qui évite de surcharger les sites
# cibles ET garantit que le user voit l'avancement séquentiellement.
#
# Manipulation : add / remove / reorder via API REST. Le run en cours
# (running_item) n'est pas dans la queue (déjà sorti).

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing      import Any, Callable, Optional


@dataclass
class QueueItem:
    """Un travail en attente — soit un step seul, soit une chain."""
    item_id:    str
    kind:       str                    # 'step' ou 'chain'
    name:       str                    # nom humain pour l'affichage
    payload:    dict[str, Any]         # spec sérialisée (CrawlStep/CrawlChain)
    enqueued_at: float = 0.0


class RunQueue:
    """File FIFO + worker single-threaded. Singleton."""

    _instance: Optional['RunQueue'] = None

    def __init__(self) -> None:
        self._items:  list[QueueItem]      = []
        self._lock:   threading.Lock       = threading.Lock()
        self._cond:   threading.Condition  = threading.Condition(self._lock)
        self._running: Optional[QueueItem] = None
        self._worker_thread: Optional[threading.Thread] = None
        # Callback exécuté pour chaque item (set par le sidecar au démarrage).
        # Reçoit le QueueItem et bloque jusqu'à la fin du run.
        self._handler: Optional[Callable[[QueueItem], None]] = None

    @classmethod
    def get(cls) -> 'RunQueue':
        if cls._instance is None:
            cls._instance = RunQueue()
        return cls._instance

    # ── Lifecycle ────────────────────────────────────────────────────

    def set_handler(self, handler: Callable[[QueueItem], None]) -> None:
        """Branche le handler qui exécute un item. Démarre le worker
        si pas déjà fait."""
        self._handler = handler
        if self._worker_thread is None:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name='run-queue-worker',
            )
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._items:
                    self._cond.wait()
                item = self._items.pop(0)
                self._running = item
            try:
                if self._handler is not None:
                    self._handler(item)
            except Exception:
                # Le handler doit logguer ses propres erreurs ; ici on
                # protège juste le thread worker.
                pass
            finally:
                with self._lock:
                    self._running = None

    # ── API publique ─────────────────────────────────────────────────

    def enqueue(self, kind: str, name: str, payload: dict) -> str:
        import time
        item = QueueItem(
            item_id     = uuid.uuid4().hex[:12],
            kind        = kind,
            name        = name,
            payload     = payload,
            enqueued_at = time.time(),
        )
        with self._cond:
            self._items.append(item)
            self._cond.notify()
        return item.item_id

    def remove(self, item_id: str) -> bool:
        with self._lock:
            for i, it in enumerate(self._items):
                if it.item_id == item_id:
                    del self._items[i]
                    return True
        return False

    def move(self, item_id: str, new_index: int) -> bool:
        with self._lock:
            idx = next((i for i, it in enumerate(self._items)
                        if it.item_id == item_id), -1)
            if idx == -1:
                return False
            item = self._items.pop(idx)
            new_index = max(0, min(new_index, len(self._items)))
            self._items.insert(new_index, item)
            return True

    def snapshot(self) -> dict:
        """
        Vue lecture seule pour le front : items en attente + en cours.

        Expose `session_id` et `chain_id` (extraits du payload) — sans
        ces champs, un nouveau tab ne peut pas resync sur le crawl en
        cours (RunStateGuard a besoin de session_id pour reconnecter
        la WS et chain_id pour recharger la chain dans le wizard).
        """
        def _meta(it: 'QueueItem') -> dict:
            payload = it.payload or {}
            return {
                'item_id':     it.item_id,
                'kind':        it.kind,
                'name':        it.name,
                'enqueued_at': it.enqueued_at,
                'session_id':  payload.get('session_id'),
                'chain_id':    payload.get('chain_id'),
            }
        with self._lock:
            return {
                'running': _meta(self._running) if self._running else None,
                'items':   [_meta(it) for it in self._items],
            }
