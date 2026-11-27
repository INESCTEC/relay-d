"""How the app start-up and live changes use the user settings. Pure functions (no Qt, no ROS), so they are tested."""

import os
from typing import Mapping

_PAGES = ("start", "data", "review", "train", "import", "settings")


def startup_page(settings: Mapping) -> str:
    """Name of the page shown at start: one of the menu pages, else the help page."""
    name = settings.get("startup_page")
    return name if name in _PAGES else "help"


def needs_config(page: str, config_loaded: bool) -> bool:
    """The Record page ("start") cannot be used without a loaded config: the app opens the Config page instead."""
    return page == "start" and not config_loaded


def live_changes(settings: Mapping, theme: str, spin_ms: int) -> dict:
    """What a saved settings dict changes in the running window: ``theme`` and/or ``spin_ms`` (only if different)."""
    changes = {}
    if settings.get("theme", theme) != theme:
        changes["theme"] = settings["theme"]
    if int(settings.get("ros_spin_ms", spin_ms)) != spin_ms:
        changes["spin_ms"] = int(settings["ros_spin_ms"])
    return changes


def theme_stylesheet(qss_path: str) -> str:
    """The QSS of a theme file, with ``@ICONS@`` set to the absolute icons/ folder next to it (Qt resolves
    ``url()`` against the working directory, not the stylesheet)."""
    with open(qss_path, "r") as f:
        qss = f.read()
    icons = os.path.join(os.path.dirname(os.path.abspath(qss_path)), "icons").replace(os.sep, "/")
    return qss.replace("@ICONS@", icons)
