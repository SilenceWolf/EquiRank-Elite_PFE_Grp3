# ============================================================
# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

# Stockage en mémoire des sessions HTTP authentifiées : permet à
# l'utilisateur de se connecter UNE FOIS sur un site (ex. FFE
# Telemat), de récupérer un `auth_id` opaque, et de le passer
# ensuite à chaque appel de step pour que les requêtes héritent
# des cookies de session déjà posés.
#
# Pas de persistance disque : les credentials et cookies vivent
# uniquement en RAM du sidecar. Au redémarrage du process, tout
# est perdu — l'utilisateur doit se re-loguer. C'est intentionnel
# pour des raisons de sécurité (pas de mot de passe sur disque).

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class AuthSession:
    """Une session HTTP authentifiée associée à un site."""
    auth_id:    str
    login_url:  str
    username:   str
    cookies:    list[dict]            # cookies au format Playwright (réutilisable des 2 côtés)
    created_at: float                  # timestamp epoch
    last_used:  float                  # mis à jour à chaque accès
    raw_cookies_dict: dict[str, str] = field(default_factory=dict)  # name → value pour requests


class _AuthStore:
    """Singleton thread-safe : map auth_id → AuthSession."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, AuthSession] = {}

    def store(
        self,
        login_url: str,
        username:  str,
        session:   requests.Session,
    ) -> str:
        """
        Encapsule les cookies d'une session `requests` post-login dans
        un AuthSession identifié par un UUID. Retourne l'auth_id.
        """
        import time
        from .playwright_pool import cookies_from_requests_session as _to_pw_cookies

        auth_id = uuid.uuid4().hex
        cookies_pw   = _to_pw_cookies(session)
        cookies_dict = {c.name: c.value for c in session.cookies}
        now = time.time()
        with self._lock:
            self._sessions[auth_id] = AuthSession(
                auth_id    = auth_id,
                login_url  = login_url,
                username   = username,
                cookies    = cookies_pw,
                created_at = now,
                last_used  = now,
                raw_cookies_dict = cookies_dict,
            )
        return auth_id

    def get(self, auth_id: str) -> Optional[AuthSession]:
        """Retourne la session ou None si auth_id inconnu/expiré."""
        if not auth_id:
            return None
        with self._lock:
            sess = self._sessions.get(auth_id)
            if sess is not None:
                import time
                sess.last_used = time.time()
            return sess

    def revoke(self, auth_id: str) -> bool:
        """Supprime une session (logout côté store)."""
        with self._lock:
            return self._sessions.pop(auth_id, None) is not None


# Singleton module-level
_store_instance = _AuthStore()


def get_store() -> _AuthStore:
    return _store_instance
