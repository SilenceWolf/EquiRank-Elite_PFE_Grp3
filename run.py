# ============================================================
# © 2026 PFE EquiRank Elite — Tous droits réservés
# Équipe : Louis Guillory · Karlotta Martin · Mathéo Isidoro
#         · Mattéo D'Andrea · Melvin Bandeira · Yohan Cana
# Licence : PROPRIETARY — voir fichier LICENSE
# ============================================================

"""
Launcher EquiRank Elite — alternative à `uvicorn --reload`.

Lance le serveur en mode "production locale" : pas de file-watcher,
Ctrl+C tue le process Python proprement et tout de suite. Idéal quand
tu lances un crawl long et que tu veux pouvoir l'interrompre sans
laisser un sub-process orphelin.

Usage :
    python run.py                  # port 8000 par défaut
    python run.py --port 8001
    python run.py --host 0.0.0.0   # exposé sur le réseau local
    python run.py --reload         # active le reloader (dev pur)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description = 'EquiRank Elite — serveur web')
    parser.add_argument('--host',   default = '127.0.0.1', help = 'Hôte (défaut: 127.0.0.1)')
    parser.add_argument('--port',   default = 8000, type = int, help = 'Port (défaut: 8000)')
    parser.add_argument('--reload', action = 'store_true',
                        help = 'Active le reloader (dev pur — rend Ctrl+C plus lent à fermer)')
    args = parser.parse_args()

    # SIGINT propre : si l'utilisateur fait Ctrl+C deux fois, on force
    # quand même la fin (au cas où uvicorn serait bloqué).
    _ctrl_c_count = {'n': 0}
    def _sigint_handler(_sig, _frame):
        _ctrl_c_count['n'] += 1
        if _ctrl_c_count['n'] >= 2:
            print('\n⛔ Deuxième Ctrl+C — force exit immédiat.', flush = True)
            os._exit(130)
        # Premier Ctrl+C : on laisse uvicorn faire son shutdown naturel
        # (le lifespan dans server.py s'occupe de Playwright).
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        import uvicorn
    except ImportError:
        print('uvicorn non installé — pip install "uvicorn[standard]"', file = sys.stderr)
        sys.exit(1)

    print(f'🐴 EquiRank Elite — http://{args.host}:{args.port}/', flush = True)
    if args.reload:
        print('   (mode reload — Ctrl+C peut être un peu lent à fermer)', flush = True)
    else:
        print('   (mode standard — Ctrl+C ferme tout de suite)', flush = True)

    try:
        uvicorn.run(
            'equirank.server:app',
            host       = args.host,
            port       = args.port,
            reload     = args.reload,
            log_level  = 'info',
        )
    except KeyboardInterrupt:
        # Le shutdown a été géré par le lifespan de server.py
        print('Bye.', flush = True)


if __name__ == '__main__':
    main()
