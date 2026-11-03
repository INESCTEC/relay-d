"""Conversion-format options, kept apart from the acquisition config.

The acquisition config says what to record and how streams/actions are built.
Choices that only matter when converting to a particular format (robomimic env
metadata, the robomimic training-config model, train/val/test split fractions,
the LeRobot task text, ...) live in an optional ``postprocess_config.yaml``::

    postprocess:
      robomimic:
        training_model: bc_rnn_gmm
        splits: {train_frac: 0.8, val_frac: 0.2, test_frac: 0.0}
        metadata: {env_name: Lift, env_version: 1.5.1, type: 1, env_kwargs: {...}}
      lerobot:
        task: "lift the box"
        robot_type: iiwa
        state_streams: [eef_translation, robot_joint_pos]

Lookup order for every key: this file, then the legacy location inside the
acquisition config (with a one-time deprecation warning), then a default.
No Qt or ROS needed.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger

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


_warned_legacy: Set[str] = set()


def _warn_legacy(key: str, new_location: str) -> None:
    if key in _warned_legacy:
        return
    _warned_legacy.add(key)
    logger.warning(
        f"'{key}' in the acquisition config is deprecated; move it to "
        f"{OPTIONS_FILENAME} under '{new_location}'."
    )


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

    def robomimic_metadata(self, acq_config: Optional[Dict] = None) -> Dict:
        """Environment metadata embedded as robomimic ``env_args``."""
        meta = self._section("robomimic").get("metadata")
        if isinstance(meta, dict) and meta:
            return meta
        legacy = (acq_config or {}).get("metadata")
        if isinstance(legacy, dict) and legacy:
            _warn_legacy("metadata", "robomimic.metadata")
            return legacy
        return {}

    def training_model(self, acq_config: Optional[Dict] = None) -> str:
        model = self._section("robomimic").get("training_model")
        if model:
            return str(model)
        legacy = (acq_config or {}).get("training_model")
        if legacy:
            _warn_legacy("training_model", "robomimic.training_model")
            return str(legacy)
        default = user_settings.get("postprocess_default_training_model")  # the Settings table owns the default
        logger.info(f"No 'training_model' set; using the default '{default}'.")
        return default

    @staticmethod
    def _default_splits() -> Dict[str, float]:
        """Defaults from the user settings (train/val; the rest is eval)."""
        train, val = float(user_settings.get("split_train")), float(user_settings.get("split_val"))
        return {"train_frac": train, "val_frac": val, "test_frac": round(max(0.0, 1.0 - train - val), 6)}

    def splits(self, acq_config: Optional[Dict] = None) -> Dict[str, float]:
        """train/val/test fractions for the robomimic ``mask`` group."""
        out = self._default_splits()
        file_splits = self._section("robomimic").get("splits")
        file_splits = file_splits if isinstance(file_splits, dict) else {}
        for key in list(out):
            if key in file_splits:
                out[key] = file_splits[key]
            elif acq_config and key in acq_config:
                _warn_legacy(key, "robomimic.splits")
                out[key] = acq_config[key]
        return out

    def lerobot(self, acq_config: Optional[Dict] = None) -> Dict[str, Any]:
        """LeRobot options: ``task``, ``robot_type``, ``state_streams``."""
        legacy = (acq_config or {}).get("lerobot")
        merged: Dict[str, Any] = {}
        if isinstance(legacy, dict) and legacy:
            _warn_legacy("lerobot", "lerobot")
            merged.update(legacy)
        merged.update({k: v for k, v in self._section("lerobot").items() if v is not None})
        return merged
