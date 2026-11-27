"""
config_loader.py
----------------
Shared YAML-loading helper for relay_d.dispatch. Some upstream config-export
tooling produces "double-encoded" YAML files: the parsed top-level value is
itself a string containing YAML text that must be parsed a second time to
reach the real mapping. This transparently handles both cases.
"""

from __future__ import annotations

from typing import Dict

import yaml


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        parsed = yaml.safe_load(f)

    if isinstance(parsed, str):
        parsed = yaml.safe_load(parsed)

    if not isinstance(parsed, dict):
        raise TypeError(
            f"Config file '{path}' did not parse to a mapping after up to "
            f"two YAML-decode passes (got {type(parsed).__name__})."
        )

    return parsed


def output_bridge_map(data_config: Dict) -> Dict[str, str]:
    """Flat ``{logical_key: recorded_path}`` bridge-map entries from ``config.output``
    (top-level string entries other than ``streams`` and ``prefix_path``)."""
    output_section = (data_config or {}).get("output") or {}
    return {
        key: value
        for key, value in output_section.items()
        if key not in ("streams", "prefix_path") and isinstance(value, str)
    }


def explicit_modalities(data_config: Dict) -> Dict[str, str]:
    """Output (observation) key -> ``'rgb' | 'depth' | 'scan'`` for camera / point-cloud outputs
    whose input sets ``modality`` (e.g. ``agentview_depth: camera_depth/images`` takes the modality
    of input ``camera_depth``)."""
    by_group = {
        name: spec.get("modality")
        for name, spec in ((data_config or {}).get("input") or {}).items()
        if isinstance(spec, dict)
    }
    modalities: Dict[str, str] = {}
    for key, path in output_bridge_map(data_config).items():
        modality = by_group.get(path.split("/")[0])
        if modality in ("rgb", "depth", "scan"):
            modalities[key] = modality
    return modalities
