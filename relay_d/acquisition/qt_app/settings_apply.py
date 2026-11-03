"""How the app start-up and live changes use the user settings. Pure functions (no Qt, no ROS), so they are tested."""

from typing import Mapping, Optional

_PAGES = ("start", "data", "review", "train", "import", "settings")


def ros_domain_to_set(settings: Mapping, environ: Mapping) -> Optional[str]:
    """Value for ROS_DOMAIN_ID, or None to leave it alone (-1 in the settings, or already set in the environment)."""
    domain = int(settings.get("ros_domain_id", -1))
    return str(domain) if domain >= 0 and "ROS_DOMAIN_ID" not in environ else None


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
