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

    def set_train_dir(self, train_dir: str):
        self.json_config.setdefault("train", {})["output_dir"] = train_dir

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

    def save(self, output_path: str = None):
        with open(output_path or self.config_path, "w") as f:
            json.dump(self.json_config, f, indent=4)


def classify_observations(
    data_config: Dict,
    shape_cache: Dict[str, Tuple[Tuple[int, ...], Any]],
) -> Dict[str, list]:
    """
    Classify every key in shape_cache (the observations that actually
    survived the last conversion) into Robomimic's low_dim/rgb/depth/scan
    obs modalities.

    Priority:
      1. An explicit `modality: rgb|depth|scan` on the data_config.yaml
         `input` entry that produced this key (authoritative — dtype alone
         can't distinguish depth/scan from other float data).
      2. Real recorded dtype == uint8 -> rgb (the common case: camera
         inputs that never bothered setting `modality`).
      3. Otherwise -> low_dim.
    """
    explicit_modality: Dict[str, str] = {}
    for input_name, spec in data_config.get("input", {}).items():
        if not isinstance(spec, dict):
            continue
        modality = spec.get("modality")
        if modality in ("rgb", "depth", "scan"):
            key = spec.get("output_map") or input_name
            explicit_modality[key] = modality

    low_dim, rgb, depth, scan = [], [], [], []
    buckets = {"low_dim": low_dim, "rgb": rgb, "depth": depth, "scan": scan}

    for key in sorted(shape_cache.keys()):
        modality = explicit_modality.get(key)
        if modality is None:
            _, dtype = shape_cache[key]
            modality = "rgb" if dtype == np.uint8 else "low_dim"
        buckets[modality].append(key)

    return {"low_dim": low_dim, "rgb": rgb, "depth": depth, "scan": scan}
