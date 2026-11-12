"""Checks of an acquisition config dict, shared by the Config builder, the Config page and the offline loader (no Qt, no ROS).

The builder treats ``error`` problems as blocking. The Config page shows them all as a warning when a file loads; the recorder
refuses the removed keys that changed the recorded data (``RECORDING_CHANGING``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from relay_d.acquisition.qt_app.ui.config.builder.draft import spec_inputs

ERROR, WARNING = "error", "warning"
_MODALITIES = ("rgb", "depth", "scan")


@dataclass(frozen=True)
class Problem:
    level: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.message}"


def stream_width(specs: List[str], inputs: Dict[str, dict]) -> Optional[int]:
    """Number of values one stream row has, or None if it cannot be known from the config (a ``[:]`` slice without joint_names)."""
    width = 0
    for spec in specs:
        if re.fullmatch(r"zero:\d+", spec.strip()):  # zero:N is N zeros
            width += int(spec.strip()[5:])
        elif re.sub(r"^(sin|cos)\((.*)\)$", r"\2", spec.strip()).endswith("[:]"):  # sin(a.position[:]) is as wide as its slice
            names = (inputs.get(next(iter(spec_inputs(spec)), ""), {}) or {}).get("joint_names")
            if not names:
                return None
            width += len(names)
        else:
            width += 1
    return width


def _numbers(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_config(data: Dict[str, Any]) -> List[Problem]:
    problems: List[Problem] = []
    add = lambda level, msg: problems.append(Problem(level, msg))  # noqa: E731
    config = (data or {}).get("config")
    if not isinstance(config, dict):
        return [Problem(ERROR, "the file has no 'config:' section")]

    for key, why in removed_keys(config):
        add(ERROR, why)
    freq = config.get("recording_frequency")
    if freq is None:
        add(ERROR, "recording_frequency is required (the rate in Hz every input is sampled at)")
    elif not _numbers(freq) or freq <= 0:
        add(ERROR, f"recording_frequency must be a positive number, got {freq!r}")
    if config.get("sync_mode") not in (None, "soft", "strict"):
        add(ERROR, f"sync_mode must be 'soft' or 'strict', got {config.get('sync_mode')!r}")

    # ---- inputs
    inputs = {n: v for n, v in (config.get("input") or {}).items() if isinstance(v, dict)}
    for name, raw in inputs.items():
        is_tf = "parent_frame" in raw and "child_frame" in raw
        if not is_tf and not raw.get("topic"):
            add(WARNING, f"input '{name}' has neither a topic nor parent_frame + child_frame: it is ignored")
        if str(raw.get("modality") or "rgb") not in _MODALITIES:
            add(WARNING, f"input '{name}': modality '{raw.get('modality')}' is not one of {_MODALITIES}")
        if name == "streams":
            add(ERROR, "an input named 'streams' would be recorded into the group that holds the output.streams data: rename it")

    # ---- streams
    output = config.get("output") or {}
    streams = output.get("streams") or {}
    for name, value in streams.items():
        value_dict = value if isinstance(value, dict) else {"specs": value}
        specs = [str(s) for s in (value_dict.get("specs") or [])]
        if not specs:
            add(ERROR, f"stream '{name}' has no specs")
            continue
        for inp in sorted({i for s in specs for i in spec_inputs(s)} - set(inputs)):
            add(ERROR, f"stream '{name}' uses the input '{inp}', which is not defined under input:")
        fmt = value_dict.get("rotation_format", "quat")
        if fmt not in ("quat", "rpy"):
            add(ERROR, f"stream '{name}': rotation_format must be 'quat' or 'rpy', got {fmt!r}")
        elif fmt == "rpy" and (len(specs) != 4 or any(s.rstrip().endswith("[:]") for s in specs)):
            add(ERROR, f"stream '{name}': rotation_format rpy needs exactly 4 scalar quaternion specs (x, y, z, w), has {len(specs)}")
        if msg := normalization_error(name, value):
            add(ERROR, msg)

    # ---- camera / point cloud entries
    for key, value in output.items():
        if key in ("streams", "prefix_path"):
            continue
        if not isinstance(value, str) or "/" not in value:
            add(WARNING, f"output entry '{key}' should look like '<input>/images' or '<input>/points'")
            continue
        group = value.split("/", 1)[0]
        if group not in inputs:
            add(WARNING, f"output '{key}' points at '{group}', which is not an input")

    # ---- actions
    actions = config.get("actions_to_extract") or []
    if not actions:
        add(ERROR, "actions_to_extract is empty: name the stream(s) the policy outputs as its action")
    for name in actions:
        if name not in streams:
            add(ERROR, f"actions_to_extract: '{name}' is not a stream under output.streams")
    for name in config.get("delta_actions") or []:
        if name not in actions:
            add(ERROR, f"delta_actions: '{name}' is not listed in actions_to_extract")
    return problems


_DEAD = "it was never used"
# (section, key) -> why it was removed and what to do. An old config that still has one gets an error, never silent ignoring.
_REMOVED = {
    ("input", "output_map"): "inputs are always recorded under their own name (output_map made recording, conversion and "
                             "dispatch disagree). Delete it and rename the input to the group name you want",
    ("input", "fill_default"): "it wrote a made-up value before the first real message. Delete it: with fill_gaps the "
                               "first real message fills the leading samples",
    ("input", "transfer_rate"): f"{_DEAD}. Delete it: recording_frequency sets the rate of every input",
    ("output", "prefix_path"): f"{_DEAD}. Delete it",
    ("config", "data"): f"{_DEAD}. Delete it",
    ("metadata", "recording_frequency"): "the rate has one place. Delete it and set config.recording_frequency instead",
    ("metadata", "transfer_rate"): f"{_DEAD}. Delete it: config.recording_frequency sets the rate",
}
RECORDING_CHANGING = ("output_map", "fill_default")  # the recorder refuses these (they changed the recorded data)


def removed_keys(config: Dict[str, Any]) -> List[tuple]:
    """``(key, message)`` for every removed key still in ``config`` (the ``config:`` section)."""
    sections = {"config": config, "output": config.get("output"), "metadata": config.get("metadata")}
    found = []
    for (section, key), why in _REMOVED.items():
        if section == "input":
            for name, raw in (config.get("input") or {}).items():
                if isinstance(raw, dict) and key in raw:
                    found.append((key, f"input '{name}': '{key}' was removed: {why}."))
        elif isinstance(sections[section], dict) and key in sections[section]:
            label = key if section in ("config", "output") else f"{section}.{key}"
            found.append((key, f"'{label}' was removed: {why}."))
    return found


def normalization_error(name: str, value: Any) -> Optional[str]:
    """The error for a stream that still has the removed ``normalize`` / ``normalized_range`` / ``limits`` keys, else None."""
    if isinstance(value, dict) and any(k in value for k in ("normalize", "normalized_range", "limits")):
        return (f"Relay-D no longer normalizes: robomimic and LeRobot normalize with the dataset's stats. "
                f"Remove normalize / normalized_range / limits from stream '{name}'.")
    return None


def stream_build_errors(data: Dict[str, Any]) -> List[str]:
    """Spec syntax errors, from the same ``StreamBuilder`` the recorder and the Config page use."""
    from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser
    from relay_d.acquisition.qt_app.utils.stream_builder import StreamBuilder

    parser = YamlParser()
    if not parser.load_from_yaml_string(yaml.safe_dump(data, sort_keys=False)):
        return ["the config could not be parsed"]
    return list(StreamBuilder(parser).get_build_errors())


def has_errors(problems: List[Problem]) -> bool:
    return any(p.level == ERROR for p in problems)
