# © 2026 Louis Guillory (Silence Wolf) — Tous droits réservés
# Licence : PROPRIETARY — voir fichier LICENSE

# Miroir Python du logger TypeScript dans outils/log/.
# Même philosophie : Singleton, niveaux filtrables, sortie console colorée
# et fichier texte en parallèle. On garde la même API pour ne pas dérouter
# quelqu'un qui connaît déjà le logger TS.

from __future__ import annotations

import sys
import io
from datetime import datetime

# Windows : le terminal peut être en cp1252 — on force UTF-8 pour les emojis et ANSI
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf-8-sig'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf-8-sig'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from enum import IntEnum
from pathlib import Path
from typing import TextIO


class LogLevel(IntEnum):
    DEBUG   = 0
    INFO    = 1
    SUCCESS = 2
    WARN    = 3
    ERROR   = 4
    FATAL   = 5


# Métadonnées d'affichage pour chaque niveau
_LEVEL_META: dict[LogLevel, dict] = {
    LogLevel.DEBUG:   {'label': 'DEBUG',   'icon': '🔍', 'color': '\033[36m'},
    LogLevel.INFO:    {'label': 'INFO',    'icon': 'ℹ️ ', 'color': '\033[34m'},
    LogLevel.SUCCESS: {'label': 'SUCCESS', 'icon': '✅', 'color': '\033[32m'},
    LogLevel.WARN:    {'label': 'WARN',    'icon': '⚠️ ', 'color': '\033[33m'},
    LogLevel.ERROR:   {'label': 'ERROR',   'icon': '❌', 'color': '\033[31m'},
    LogLevel.FATAL:   {'label': 'FATAL',   'icon': '💀', 'color': '\033[35m'},
}

_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_CYAN   = '\033[36m'
_MAGENTA = '\033[35m'
_DIM    = '\033[2m'


class CrawlerLogger:
    """
    Singleton logger — une seule instance par processus.
    Usage : logger = CrawlerLogger.get_instance(log_file=Path(...))
    """

    _instance: CrawlerLogger | None = None

    def __init__(
        self,
        context:   str        = 'crawler',
        min_level: LogLevel   = LogLevel.DEBUG,
        log_file:  Path | None = None,
    ) -> None:
        self._context   = context
        self._min_level = min_level
        self._file_handle: TextIO | None = None

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = log_file.open('a', encoding='utf-8')

    # ── Singleton ──────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        context:   str        = 'crawler',
        min_level: LogLevel   = LogLevel.DEBUG,
        log_file:  Path | None = None,
    ) -> 'CrawlerLogger':
        if cls._instance is None:
            cls._instance = cls(context, min_level, log_file)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Réinitialise le singleton — utile pour les tests ou changement de session."""
        if cls._instance and cls._instance._file_handle:
            cls._instance._file_handle.close()
        cls._instance = None

    # ── API publique ───────────────────────────────────────────────────────

    def debug(self,   msg: str) -> None: self._log(LogLevel.DEBUG,   msg)
    def info(self,    msg: str) -> None: self._log(LogLevel.INFO,    msg)
    def success(self, msg: str) -> None: self._log(LogLevel.SUCCESS, msg)
    def warn(self,    msg: str) -> None: self._log(LogLevel.WARN,    msg)
    def error(self,   msg: str) -> None: self._log(LogLevel.ERROR,   msg)
    def fatal(self,   msg: str) -> None: self._log(LogLevel.FATAL,   msg)

    def separator(self, char: str = '─', width: int = 60) -> None:
        line = char * width
        print(f'{_DIM}{line}{_RESET}')
        self._write_plain(line)

    def box(self, title: str, lines: list[str]) -> None:
        width   = max(len(title), max((len(l) for l in lines), default=0)) + 4
        border  = '─' * width
        content = [f'│ {l.ljust(width - 2)} │' for l in lines]

        print(f'\n{_CYAN}┌{border}┐{_RESET}')
        print(f'{_CYAN}│{_BOLD} {title.center(width)} {_RESET}{_CYAN}│{_RESET}')
        print(f'{_CYAN}├{border}┤{_RESET}')
        for line in content:
            print(f'{_CYAN}{line}{_RESET}')
        print(f'{_CYAN}└{border}┘{_RESET}\n')

        self._write_plain(f'[BOX] {title}: {" | ".join(lines)}')

    def banner(self, title: str) -> None:
        width  = max(len(title) + 4, 40)
        border = '═' * width
        print(f'\n{_MAGENTA}╔{border}╗{_RESET}')
        print(f'{_MAGENTA}║{_BOLD}{title.center(width)}{_RESET}{_MAGENTA}║{_RESET}')
        print(f'{_MAGENTA}╚{border}╝{_RESET}\n')
        self._write_plain(f'=== {title} ===')

    # ── Internals ──────────────────────────────────────────────────────────

    def _log(self, level: LogLevel, msg: str) -> None:
        if level < self._min_level:
            return

        meta  = _LEVEL_META[level]
        now   = datetime.now().strftime('%H:%M:%S')
        label = meta['label'].ljust(7)
        color = meta['color']
        icon  = meta['icon']

        console_line = (
            f"{color}{_BOLD}[{label}]{_RESET} "
            f"{icon}  "
            f"{_DIM}{now}{_RESET} "
            f"{_CYAN}[{self._context}]{_RESET} "
            f"{msg}"
        )

        stream = sys.stderr if level >= LogLevel.WARN else sys.stdout
        print(console_line, file=stream)

        iso_now = datetime.now().isoformat(timespec='seconds')
        self._write_plain(f'[{meta["label"]}] {iso_now} [{self._context}] {msg}')

    def _write_plain(self, line: str) -> None:
        if self._file_handle:
            self._file_handle.write(line + '\n')
            self._file_handle.flush()

    def __del__(self) -> None:
        if self._file_handle:
            self._file_handle.close()
