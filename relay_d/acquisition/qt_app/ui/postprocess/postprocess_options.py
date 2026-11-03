"""Conversion-format options, kept apart from the acquisition config.

The acquisition config says what to record and how streams/actions are built.
Choices that only matter when converting to a particular format (train/val
split fractions, the LeRobot task text, ...) live in an optional
``postprocess_config.yaml``::

    postprocess:
      robomimic:
        splits: {train_frac: 0.8, val_frac: 0.2}
      lerobot:
        task: "lift the box"
        robot_type: iiwa
        state_streams: [eef_translation, robot_joint_pos]

Every key not set here takes its default. Removed keys (here or in the
acquisition config) are refused with what to do instead. No Qt or ROS needed.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from relay_d.utils import user_settings

OPTIONS_FILENAME = "postprocess_config.yaml"

# Converted output says which format it is, like recording session folders end in _Recording.
FORMAT_SUFFIX = {"robomimic": "_Robomimic", "lerobot": "_Lerobot"}


def dataset_output_paths(save_folder: str, fmt: str, now: datetime) -> Tuple[str, str]:
    """``(output folder, dataset path)`` of one Post-process conversion, e.g.
    ``<save>/2026-11-02_15-10-00_Robomimic`` and ``.../dataset_1793630857_Robomimic.h5``.
    A LeRobot dataset is a folder, so it has no extension."""
    suffix = FORMAT_SUFFIX[fmt]
    output_dir = os.path.join(save_folder, now.strftime("%Y-%m-%d_%H-%M-%S") + suffix)
    extension = "" if fmt == "lerobot" else ".h5"
    return output_dir, os.path.join(output_dir, f"dataset_{int(now.timestamp())}{suffix}{extension}")


# Removed keys -> why and what to do. A config that still has one is refused, never silently ignored.
_SPLITS_MOVED = "move it to postprocess_config.yaml under robomimic.splits"
_TEST_SPLIT = "the test split was removed (robomimic never trains or validates on it); delete it"
REMOVED_ACQUISITION_KEYS = {
    "metadata": "robomimic env metadata is no longer written (env_args is a fixed stub); "
                "put recording_frequency at the top level of config and delete 'metadata'",
    "training_model": "choose the policy on the Train tab (or relay-d-train --policy) and delete it",
    "train_frac": _SPLITS_MOVED,
    "val_frac": _SPLITS_MOVED,
    "test_frac": _TEST_SPLIT,
    "lerobot": "move it to postprocess_config.yaml under 'lerobot'",
}
REMOVED_OPTIONS = {
    "metadata": "robomimic env metadata is no longer written (env_args is a fixed stub); delete it",
    "training_model": "choose the policy on the Train tab (or relay-d-train --policy) and delete it",
}


def removed_key_errors(acq_config: Optional[Dict]) -> List[str]:
    """One error per removed key still in the acquisition config's ``config`` section."""
    return [f"'{k}' was removed: {why}." for k, why in REMOVED_ACQUISITION_KEYS.items() if k in (acq_config or {})]


class PostprocessOptions:
    """Per-format conversion options (optionally loaded from a YAML file)."""

    def __init__(self, data: Optional[Dict] = None, source: Optional[str] = None):
        self._data: Dict = data or {}
        self.source = source

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str) -> Tuple[Optional["PostprocessOptions"], List[str]]:
        """Parse and validate ``path``. Returns ``(options, problems)``; options is
        None when the file is unusable (``problems`` explains why)."""
        if not path or not os.path.isfile(path):
            return None, [f"File not found: {path}"]
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            return None, [f"Could not parse YAML file: {e}"]
        if not isinstance(raw, dict) or not isinstance(raw.get("postprocess"), dict):
            return None, [f"The file has no top-level 'postprocess' section."]
        section = raw["postprocess"]
        problems: List[str] = []
        for fmt, value in section.items():
            if not isinstance(value, dict):
                problems.append(f"'postprocess.{fmt}' must be a mapping.")
        if problems:
            return None, problems
        robomimic = section.get("robomimic", {})
        problems = [f"'robomimic.{k}' was removed: {why}." for k, why in REMOVED_OPTIONS.items() if k in robomimic]
        if "test_frac" in (robomimic.get("splits") or {}):
            problems.append(f"'robomimic.splits.test_frac' was removed: {_TEST_SPLIT}.")
        if problems:
            return None, problems
        known = {"robomimic", "lerobot"}
        warnings = [
            f"warning: unknown format section 'postprocess.{k}' (known: {sorted(known)})"
            for k in section
            if k not in known
        ]
        return cls(section, source=path), warnings

    @staticmethod
    def find_sibling(config_path: Optional[str]) -> Optional[str]:
        """``postprocess_config.yaml`` next to the given acquisition config, if present."""
        if not config_path:
            return None
        candidate = os.path.join(os.path.dirname(os.path.abspath(config_path)), OPTIONS_FILENAME)
        if os.path.isfile(candidate) and os.path.abspath(candidate) != os.path.abspath(config_path):
            return candidate
        return None

    # ------------------------------------------------------------------ lookups

    def _section(self, fmt: str) -> Dict:
        value = self._data.get(fmt)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _default_splits() -> Dict[str, float]:
        """Defaults from the user settings (train/val; any rest also trains)."""
        return {"train_frac": float(user_settings.get("split_train")), "val_frac": float(user_settings.get("split_val"))}

    def splits(self) -> Dict[str, float]:
        """train/val fractions for the robomimic ``mask`` group."""
        out = self._default_splits()
        file_splits = self._section("robomimic").get("splits")
        if isinstance(file_splits, dict):
            out.update({k: file_splits[k] for k in out if k in file_splits})
        return out

    def lerobot(self) -> Dict[str, Any]:
        """LeRobot options: ``task``, ``robot_type``, ``state_streams``."""
        return {k: v for k, v in self._section("lerobot").items() if v is not None}
