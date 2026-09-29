import sys
import os
from datetime import datetime
import inspect


# Minimum level printed by every ColoredLogger instance (existing and
# future) — checked live in _log(), not cached per-instance, so changing it
# via set_level() takes effect everywhere immediately. Defaults to INFO
# (suppressing DEBUG); override for a single run with RELAYD_LOG_LEVEL=DEBUG.
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_min_level = _LEVELS.get(os.environ.get("RELAYD_LOG_LEVEL", "INFO").upper(), 20)


def set_level(level: str) -> None:
    """Set the minimum level printed by every ColoredLogger instance
    (existing and future). One of: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    global _min_level
    name = str(level).upper()
    if name not in _LEVELS:
        raise ValueError(f"Unknown log level '{level}'. Use one of: {', '.join(_LEVELS)}")
    _min_level = _LEVELS[name]


def get_level() -> str:
    """Return the current minimum level printed, as its name."""
    return next(name for name, val in _LEVELS.items() if val == _min_level)


class ColoredLogger:
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset color
    }

    def __init__(self, name=None):
        if name is None:
            # Get the name of the calling module
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get("__name__", "unknown")
        self.name = name

        # Check if we're in a terminal that supports colors
        self.use_colors = self._supports_color()

    def _supports_color(self):
        """Check if the terminal supports ANSI colors"""
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check for common color-supporting terminals
        term = os.environ.get("TERM", "").lower()
        colorterm = os.environ.get("COLORTERM", "").lower()

        return (
            "color" in term
            or "ansi" in term
            or "xterm" in term
            or colorterm in ("truecolor", "24bit", "yes")
        )

    def _log(self, level, message):
        if _LEVELS.get(level, 20) < _min_level:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            formatted_message = f"{timestamp} - {color}{level:<8}{reset} - {message}"
        else:
            formatted_message = f"{timestamp} - {level:<8} - {message}"

        try:
            print(formatted_message, file=sys.stdout)
        except UnicodeEncodeError:
            safe_message = formatted_message.encode("utf-8", errors="replace").decode(
                "utf-8"
            )
            print(safe_message, file=sys.stdout)
        sys.stdout.flush()

    def debug(self, message):
        self._log("DEBUG", str(message))

    def info(self, message):
        self._log("INFO", str(message))

    def warning(self, message):
        self._log("WARNING", str(message))

    def warn(self, message):  # Alias for warning
        self.warning(message)

    def error(self, message):
        self._log("ERROR", str(message))

    def critical(self, message):
        self._log("CRITICAL", str(message))


# Create a default logger instance that can be imported directly
logger = ColoredLogger()


# Also provide a function to create named loggers if needed
def get_logger(name):
    return ColoredLogger(name)
