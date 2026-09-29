"""
config_loader.py
----------------
Shared YAML-loading helper for relay_d.dispatch. Some upstream config-export
tooling produces "double-encoded" YAML files: the parsed top-level value is
itself a string containing YAML text that must be parsed a second time to
reach the real mapping. This transparently handles both cases.
"""

from __future__ import annotations

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
