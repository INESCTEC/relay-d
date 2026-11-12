import json
from typing import Any, Dict, Iterable, Tuple

import numpy as np


class RobomimicConfigHandler:
    def __init__(self, config_path):
        self.config_path = config_path
        self.json_config = self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            return json.load(f)

    def get_config(self) -> Dict:
        return self.json_config

    def set_observation_modalities(
        self,
        low_dim: Iterable[str],
        rgb: Iterable[str],
        depth: Iterable[str],
        scan: Iterable[str] = (),
    ):
        """Populate observation.modalities.obs.{low_dim,rgb,depth,scan}."""
        obs = (
            self.json_config.setdefault("observation", {})
            .setdefault("modalities", {})
            .setdefault("obs", {})
        )
        obs["low_dim"] = list(low_dim)
        obs["rgb"] = list(rgb)
        obs["depth"] = list(depth)
        obs["scan"] = list(scan)


def explicit_modalities(data_config: Dict) -> Dict[str, str]:
    """Output (observation) key -> "rgb"|"depth"|"scan" for camera / point-cloud outputs whose input
    sets ``modality``. ``output: agentview_depth: camera_depth/images`` takes the modality of the input
    recorded into the group ``camera_depth`` (the input's name)."""
    by_group = {
        name: spec.get("modality")
        for name, spec in (data_config.get("input") or {}).items()
        if isinstance(spec, dict)
    }
    modalities: Dict[str, str] = {}
    for key, path in (data_config.get("output") or {}).items():
        if key in ("streams", "prefix_path") or not isinstance(path, str):
            continue
        modality = by_group.get(path.split("/")[0])
        if modality in ("rgb", "depth", "scan"):
            modalities[key] = modality
    return modalities


def classify_observations(
    data_config: Dict,
    shape_cache: Dict[str, Tuple[Tuple[int, ...], Any]],
) -> Dict[str, list]:
    """
    Classify every key in shape_cache (the observations that actually
    are in the converted dataset) into Robomimic's low_dim/rgb/depth/scan
    obs modalities.

    Priority:
      1. An explicit `modality: rgb|depth|scan` on the data_config.yaml
         `input` entry that produced this key (authoritative — dtype alone
         can't distinguish depth/scan from other float data).
      2. Real recorded dtype == uint8 -> rgb (the common case: camera
         inputs that never bothered setting `modality`).
      3. Otherwise -> low_dim.
    """
    explicit_modality = explicit_modalities(data_config)

    low_dim, rgb, depth, scan = [], [], [], []
    buckets = {"low_dim": low_dim, "rgb": rgb, "depth": depth, "scan": scan}

    for key in sorted(shape_cache.keys()):
        modality = explicit_modality.get(key)
        if modality is None:
            _, dtype = shape_cache[key]
            modality = "rgb" if dtype == np.uint8 else "low_dim"
        buckets[modality].append(key)

    return {"low_dim": low_dim, "rgb": rgb, "depth": depth, "scan": scan}
