"""The shared `logger`: stdlib logging printing `<time> - <LEVEL> - <message>` to stdout.

Colour is used on a terminal that supports it. The minimum level is INFO unless
RELAYD_LOG_LEVEL is set, and can be changed at run time with `set_level()`.
"""

import logging
import os
import sys
from datetime import datetime

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_COLORS = {"DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[35m"}


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    return any(t in term for t in ("color", "ansi", "xterm")) or os.environ.get("COLORTERM", "").lower() in (
        "truecolor",
        "24bit",
        "yes",
    )


class _Formatter(logging.Formatter):
    def format(self, record):
        stamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = f"{record.levelname:<8}"
        if _supports_color():
            level = f"{_COLORS.get(record.levelname, '')}{level}\033[0m"
        text = f"{stamp} - {level} - {record.getMessage()}"
        # an unencodable character must not lose the line
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        return text.encode(encoding, errors="replace").decode(encoding)


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever `sys.stdout` is at the time of each record (it may be redirected)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def set_level(level: str) -> None:
    """Set the minimum level printed. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    name = str(level).upper()
    if name not in _LEVELS:
        raise ValueError(f"Unknown log level '{level}'. Use one of: {', '.join(_LEVELS)}")
    logger.setLevel(name)


def apply_user_level() -> None:
    """Level from RELAYD_LOG_LEVEL if set, else from the ``log_level`` user setting, else INFO."""
    from relay_d.utils import user_settings

    level = (os.environ.get("RELAYD_LOG_LEVEL") or user_settings.get("log_level", "INFO")).upper()
    logger.setLevel(level if level in _LEVELS else "INFO")


logger = logging.getLogger("relay_d")
if not any(getattr(h, "relayd", False) for h in logger.handlers):  # importing twice must not double every line
    _handler = _StdoutHandler()
    _handler.relayd = True
    _handler.setFormatter(_Formatter())
    logger.addHandler(_handler)
    logger.propagate = False
    apply_user_level()
