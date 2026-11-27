"""Robomimic training-config templates and observation modality classification."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from relay_d.utils.config_loader import explicit_modalities

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def robomimic_template_path(model: str = "bc_rnn_gmm", templates_dir: Optional[str] = None) -> str:
    """Path of the bundled robomimic training-config JSON template for ``model``."""
    return os.path.join(templates_dir or _TEMPLATES_DIR, f"robomimic_{model}_config.json")


class RobomimicConfigHandler:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.json_config: Dict = json.load(f)

    def get_config(self) -> Dict:
        return self.json_config

    def set_observation_modalities(
        self,
        low_dim: Iterable[str],
        rgb: Iterable[str],
        depth: Iterable[str],
        scan: Iterable[str] = (),
    ) -> None:
        """Populate ``observation.modalities.obs.{low_dim,rgb,depth,scan}``."""
        obs = (
            self.json_config.setdefault("observation", {})
            .setdefault("modalities", {})
            .setdefault("obs", {})
        )
        obs["low_dim"] = list(low_dim)
        obs["rgb"] = list(rgb)
        obs["depth"] = list(depth)
        obs["scan"] = list(scan)


def classify_observations(
    data_config: Dict,
    shape_cache: Dict[str, Tuple[Tuple[int, ...], Any]],
) -> Dict[str, list]:
    """Classify every key in ``shape_cache`` into Robomimic's low_dim/rgb/depth/scan modalities.

    Priority:
      1. Explicit ``modality: rgb|depth|scan`` on the acquisition YAML ``input`` entry.
      2. Real recorded dtype == uint8 -> ``rgb``.
      3. Otherwise -> ``low_dim``.
    """
    explicit = explicit_modalities(data_config)

    low_dim, rgb, depth, scan = [], [], [], []
    buckets = {"low_dim": low_dim, "rgb": rgb, "depth": depth, "scan": scan}

    for key in sorted(shape_cache.keys()):
        modality = explicit.get(key)
        if modality is None:
            _, dtype = shape_cache[key]
            modality = "rgb" if dtype == np.uint8 else "low_dim"
        buckets[modality].append(key)

    return {"low_dim": low_dim, "rgb": rgb, "depth": depth, "scan": scan}
