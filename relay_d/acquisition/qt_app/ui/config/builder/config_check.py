"""Checks of an acquisition config dict, shared by the Config builder, the Config page and the offline loader (no Qt, no ROS).

The builder treats ``error`` problems as blocking. The load paths (Config page insert / save, ``load_offline_config``) show them
all as warnings: hand-written configs that worked before must keep loading.
"""

from __future__ import annotations

import math
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
        if spec.rstrip().endswith("[:]"):
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

    freq = config.get("recording_frequency")
    if freq is None:
        add(WARNING, "recording_frequency is not set (30 Hz is used; LeRobot conversion needs a frequency in the config)")
    elif not _numbers(freq) or freq <= 0:
        add(ERROR, f"recording_frequency must be a positive number, got {freq!r}")
    if config.get("sync_mode") not in (None, "soft", "strict"):
        add(ERROR, f"sync_mode must be 'soft' or 'strict', got {config.get('sync_mode')!r}")

    # ---- inputs
    inputs = {n: v for n, v in (config.get("input") or {}).items() if isinstance(v, dict)}
    groups: Dict[str, str] = {}
    for name, raw in inputs.items():
        is_tf = "parent_frame" in raw and "child_frame" in raw
        if not is_tf and not raw.get("topic"):
            add(WARNING, f"input '{name}' has neither a topic nor parent_frame + child_frame: it is ignored")
        if str(raw.get("modality") or "rgb").split()[0] not in _MODALITIES:
            add(WARNING, f"input '{name}': modality '{raw.get('modality')}' is not one of {_MODALITIES}")
        group = raw.get("output_map") or name
        if group in groups:
            add(ERROR, f"inputs '{groups[group]}' and '{name}' are both recorded into the group '{group}': give one an output_map")
        groups[group] = name
        if group == "streams":
            add(ERROR, f"input '{name}' would be recorded into the group 'streams', which holds the output.streams data")
        if raw.get("fill_default") is not None and not raw.get("fill_gaps"):
            add(WARNING, f"input '{name}': fill_default has no effect without fill_gaps: true")

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
        width = stream_width(specs, inputs)
        fmt = value_dict.get("rotation_format", "quat")
        if fmt not in ("quat", "rpy"):
            add(ERROR, f"stream '{name}': rotation_format must be 'quat' or 'rpy', got {fmt!r}")
        elif fmt == "rpy" and (len(specs) != 4 or any(s.rstrip().endswith("[:]") for s in specs)):
            add(ERROR, f"stream '{name}': rotation_format rpy needs exactly 4 scalar quaternion specs (x, y, z, w), has {len(specs)}")
        if value_dict.get("normalize"):
            limits = value_dict.get("limits")
            columns = 3 if fmt == "rpy" else width
            if not limits:
                add(ERROR, f"stream '{name}': normalize is on but there are no limits")
                continue
            bad = [lim for lim in limits if not (isinstance(lim, dict) and _numbers(lim.get("lower_lim")) and _numbers(lim.get("upper_lim")))]
            if bad:
                add(ERROR, f"stream '{name}': every limit needs numeric lower_lim and upper_lim (found {bad[0]!r})")
            elif any(lim["lower_lim"] >= lim["upper_lim"] for lim in limits):
                add(ERROR, f"stream '{name}': lower_lim must be smaller than upper_lim")
            elif columns is not None and len(limits) not in (1, columns):
                add(ERROR, f"stream '{name}': {len(limits)} limits for {columns} values (use one per value, or one for all)")

    # ---- camera / point cloud entries
    for key, value in output.items():
        if key in ("streams", "prefix_path"):
            continue
        if not isinstance(value, str) or "/" not in value:
            add(WARNING, f"output entry '{key}' should look like '<input>/images' or '<input>/points'")
            continue
        group = value.split("/", 1)[0]
        if group not in groups:
            if group in inputs:
                add(ERROR, f"output '{key}' points at '{group}', but that input is recorded as '{inputs[group].get('output_map')}' (its output_map)")
            else:
                add(WARNING, f"output '{key}' points at '{group}', which is not an input")

    # ---- actions
    known = set(streams) | {k for k in output if k not in ("streams", "prefix_path")}
    actions = config.get("actions_to_extract") or []
    for name in actions:
        if name not in known:
            add(ERROR, f"actions_to_extract: '{name}' is not a stream or output entry")
    for name in config.get("delta_actions") or []:
        if name not in actions:
            add(ERROR, f"delta_actions: '{name}' is not listed in actions_to_extract")
        elif isinstance(streams.get(name), dict) and streams[name].get("normalize"):
            add(ERROR, f"delta_actions: '{name}' is normalized; the dispatcher cannot undo a normalized delta (use a raw stream for the action)")
    return problems


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


def as_warnings(problems: List[Problem]) -> List[str]:
    """For the load paths: every problem as a ``warning: ...`` line (they never reject a file that loaded before)."""
    return [f"warning: {p.message}" for p in problems]
