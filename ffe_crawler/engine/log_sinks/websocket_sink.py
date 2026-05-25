# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# WebSocketLogSink : pont entre CrawlerLogger et l'EventBus.
#
# Le current session_id est porté par un contextvar pour que les tâches
# asyncio des runners puissent transmettre leur identité au logger sans
# avoir à passer session_id partout dans l'API.
#
# Le setup_log_streaming() de api/dependencies.py monkey-patche
# CrawlerLogger._write_plain au runtime pour appeler publish_log sur
# chaque ligne — scope limité au processus sidecar.

from __future__ import annotations

import contextvars
from datetime import datetime


class WebSocketLogSink:
    """
    Sink sans état (tout est static/classmethods) : garde un contextvar
    pour le session_id courant et publie chaque ligne de log dans
    l'EventBus de ce session_id.
    """

    _current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        'crawl_session_id', default=None,
    )

    @classmethod
    def set_session(cls, session_id: str | None) -> None:
        """Positionne le session_id courant pour la tâche en cours."""
        cls._current_session.set(session_id)

    @classmethod
    def get_session(cls) -> str | None:
        return cls._current_session.get()

    @classmethod
    def publish_log(cls, line: str, level: str = 'INFO') -> None:
        """
        Publie une ligne de log sur l'EventBus de la session courante.
        Si aucune session n'est active (contextvar = None), on ignore
        silencieusement — le log file reste écrit par le comportement
        d'origine de CrawlerLogger.
        """
        session_id = cls._current_session.get()
        if session_id is None:
            return

        # Import local pour éviter la dépendance circulaire avec api/events
        from api.events import EventBus

        # Parse le level depuis la ligne si elle est au format "[LEVEL] ts ctx msg"
        parsed_level = level
        if line.startswith('['):
            end = line.find(']')
            if end != -1:
                parsed_level = line[1:end].strip() or level

        EventBus.publish({
            'type':      'log',
            'level':     parsed_level,
            'message':   line,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }, session_id)
