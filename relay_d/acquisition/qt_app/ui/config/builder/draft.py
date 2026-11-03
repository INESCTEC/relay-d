"""The acquisition config as an editable draft (no Qt, no ROS).

``ConfigDraft`` holds what the Config builder edits and writes the same ``{"config": {...}}`` dict a hand-written
``acquisition_config.yaml`` has. ``from_config`` reads such a dict back, so an existing file can be edited in the builder;
keys the builder does not know (``metadata``, ``prefix_path``, ...) are kept and written back unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

# The first word of an input name picks the live view of the Record page (container_manager.py): see _PREFIXES.
_PREFIXES = {
    "image": "image", "depth": "image", "pointcloud": "pointcloud", "joint_state": "joint",
    "pose": "pose", "transform": "tf", "twist": "tool", "wrench": "tool", "scalar": "gripper", "tf": "tf",
}
_KNOWN_TOP = ("dataset_name", "recording_frequency", "max_buffer_mb", "sync_tolerance_sec", "sync_mode",
              "input", "output", "actions_to_extract", "delta_actions")
_KNOWN_TOPIC_INPUT = ("topic", "output_map", "modality", "joint_names", "fill_gaps", "fill_default", "transfer_rate")
_KNOWN_TF_INPUT = ("parent_frame", "child_frame", "tf_topic", "tf_static_topic", "tf_wait_timeout",
                   "output_map", "fill_gaps", "fill_default", "transfer_rate")
_STREAM_DICT_KEYS = ("specs", "normalize", "normalized_range", "limits", "rotation_format")


@dataclass
class InputDraft:
    name: str
    kind: str = "topic"  # "topic" | "tf"
    topic: str = ""
    msg_type: str = ""  # only to remember what the live graph said; not written to the file
    output_map: str = ""  # "" = the input name
    modality: str = ""  # "" = not written (rgb is the default); "depth" | "scan"
    joint_names: List[str] = field(default_factory=list)
    fill_gaps: bool = False
    fill_default: Any = None
    transfer_rate: Optional[float] = None
    parent_frame: str = ""
    child_frame: str = ""
    tf_topic: str = ""  # "" = /tf
    tf_static_topic: str = ""  # "" = /tf_static
    tf_wait_timeout: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def group(self) -> str:
        """Name of the HDF5 group this input is recorded into."""
        return self.output_map or self.name

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.kind == "tf":
            out.update(parent_frame=self.parent_frame, child_frame=self.child_frame)
            for key in ("tf_topic", "tf_static_topic"):
                if getattr(self, key):
                    out[key] = getattr(self, key)
            if self.tf_wait_timeout is not None:
                out["tf_wait_timeout"] = self.tf_wait_timeout
        else:
            out["topic"] = self.topic
            if self.modality:
                out["modality"] = self.modality
            if self.joint_names:
                out["joint_names"] = list(self.joint_names)
        if self.output_map:
            out["output_map"] = self.output_map
        if self.transfer_rate is not None:
            out["transfer_rate"] = self.transfer_rate
        if self.fill_gaps:
            out["fill_gaps"] = True
            if self.fill_default is not None:
                out["fill_default"] = self.fill_default
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "InputDraft":
        is_tf = "parent_frame" in raw and "child_frame" in raw
        known = _KNOWN_TF_INPUT if is_tf else _KNOWN_TOPIC_INPUT
        return cls(
            name=name, kind="tf" if is_tf else "topic", topic=raw.get("topic", ""),
            output_map=raw.get("output_map", ""), modality=raw.get("modality", ""),
            joint_names=list(raw.get("joint_names") or []), fill_gaps=bool(raw.get("fill_gaps", False)),
            fill_default=raw.get("fill_default"), transfer_rate=raw.get("transfer_rate"),
            parent_frame=raw.get("parent_frame", ""), child_frame=raw.get("child_frame", ""),
            tf_topic=raw.get("tf_topic", ""), tf_static_topic=raw.get("tf_static_topic", ""),
            tf_wait_timeout=raw.get("tf_wait_timeout"),
            extra={k: v for k, v in raw.items() if k not in known},
        )


@dataclass
class StreamDraft:
    """One entry of ``output.streams``: an observation (or action) vector built from message fields."""

    name: str
    specs: List[str] = field(default_factory=list)
    normalize: bool = False
    normalized_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    limits: List[Dict[str, float]] = field(default_factory=list)
    rotation_format: str = "quat"  # "rpy": 4 quaternion specs become roll, pitch, yaw
    explicit_dict: bool = False  # written as a dict even if nothing needs it (keeps a loaded file's shape)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_value(self) -> Any:
        if not (self.normalize or self.rotation_format != "quat" or self.explicit_dict or self.extra):
            return list(self.specs)
        out: Dict[str, Any] = {"specs": list(self.specs)}
        if self.normalize:
            out["normalize"] = True
            out["normalized_range"] = list(self.normalized_range)
            out["limits"] = [dict(lim) for lim in self.limits]
        if self.rotation_format != "quat":
            out["rotation_format"] = self.rotation_format
        out.update(self.extra)
        return out

    @classmethod
    def from_value(cls, name: str, value: Any) -> "StreamDraft":
        if not isinstance(value, dict):
            return cls(name=name, specs=[str(s) for s in (value or [])])
        return cls(
            name=name, specs=[str(s) for s in value.get("specs", [])], normalize=bool(value.get("normalize", False)),
            normalized_range=list(value.get("normalized_range", [-1.0, 1.0])),
            limits=[dict(lim) for lim in (value.get("limits") or []) if isinstance(lim, dict)],
            rotation_format=str(value.get("rotation_format", "quat")), explicit_dict=True,
            extra={k: v for k, v in value.items() if k not in _STREAM_DICT_KEYS},
        )


@dataclass
class ConfigDraft:
    dataset_name: str = "demo"
    recording_frequency: Optional[float] = 30
    max_buffer_mb: Optional[float] = None
    sync_tolerance_sec: Optional[float] = None
    sync_mode: str = ""  # "" = not written ("soft" is the default)
    inputs: Dict[str, InputDraft] = field(default_factory=dict)
    streams: Dict[str, StreamDraft] = field(default_factory=dict)
    cameras: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # obs key -> (input name, "images" | "points")
    actions: List[str] = field(default_factory=list)  # actions_to_extract: stream names, in order
    delta_actions: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # unknown keys of `config`
    output_extra: Dict[str, Any] = field(default_factory=dict)  # unknown keys of `config.output` (e.g. prefix_path)
    raw_bridges: Dict[str, Any] = field(default_factory=dict)  # bridge entries that point at no known input: kept as they were

    # ------------------------------------------------------------------ editing helpers

    def unique_name(self, wanted: str) -> str:
        """``wanted`` made a valid, unused input name (letters, digits, underscore)."""
        base = re.sub(r"\W+", "_", wanted).strip("_").lower() or "input"
        name, n = base, 2
        while name in self.inputs:
            name, n = f"{base}_{n}", n + 1
        return name

    def add_input(self, draft: InputDraft) -> InputDraft:
        self.inputs[draft.name] = draft
        return draft

    def remove_input(self, name: str) -> None:
        """Drop an input and everything built on it (streams whose specs use it, camera entries)."""
        self.inputs.pop(name, None)
        for key in [k for k, (inp, _) in self.cameras.items() if inp == name]:
            del self.cameras[key]
        for stream in [s for s in self.streams.values() if any(spec_inputs(sp) & {name} for sp in s.specs)]:
            self.remove_stream(stream.name)

    def remove_stream(self, name: str) -> None:
        self.streams.pop(name, None)
        self.actions = [a for a in self.actions if a != name]
        self.delta_actions = [a for a in self.delta_actions if a != name]

    # ------------------------------------------------------------------ file <-> draft

    def to_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {"dataset_name": self.dataset_name}
        for key in ("recording_frequency", "max_buffer_mb", "sync_tolerance_sec"):
            if getattr(self, key) is not None:
                config[key] = getattr(self, key)
        if self.sync_mode:
            config["sync_mode"] = self.sync_mode
        config["input"] = {name: inp.to_dict() for name, inp in self.inputs.items()}
        output: Dict[str, Any] = dict(self.raw_bridges)
        for key, (inp_name, dataset) in self.cameras.items():
            inp = self.inputs.get(inp_name)
            output[key] = f"{inp.group if inp else inp_name}/{dataset}"
        if self.streams:
            output["streams"] = {name: s.to_value() for name, s in self.streams.items()}
        output.update(self.output_extra)
        config["output"] = output
        if self.actions:
            config["actions_to_extract"] = list(self.actions)
        if self.delta_actions:
            config["delta_actions"] = list(self.delta_actions)
        config.update(self.extra)
        return {"config": config}

    def to_yaml(self) -> str:
        """The config as file text: keys keep their order (streams and actions are order-sensitive), with a header comment."""
        return "# Written by the Relay-D Config builder (Config page > Build from ROS).\n" + yaml.safe_dump(
            self.to_config(), sort_keys=False, default_flow_style=False
        )

    @classmethod
    def from_config(cls, data: Dict[str, Any]) -> "ConfigDraft":
        config = (data or {}).get("config") or {}
        draft = cls(
            dataset_name=config.get("dataset_name", "demo"), recording_frequency=config.get("recording_frequency"),
            max_buffer_mb=config.get("max_buffer_mb"), sync_tolerance_sec=config.get("sync_tolerance_sec"),
            sync_mode=config.get("sync_mode", ""), actions=list(config.get("actions_to_extract") or []),
            delta_actions=list(config.get("delta_actions") or []),
            extra={k: v for k, v in config.items() if k not in _KNOWN_TOP},
        )
        for name, raw in (config.get("input") or {}).items():
            if isinstance(raw, dict):
                draft.inputs[name] = InputDraft.from_dict(name, raw)
        by_group = {inp.group: inp.name for inp in draft.inputs.values()}
        for key, value in (config.get("output") or {}).items():
            if key == "streams":
                draft.streams = {n: StreamDraft.from_value(n, v) for n, v in (value or {}).items()}
            elif isinstance(value, str) and "/" in value and value.split("/", 1)[0] in by_group \
                    and value.split("/", 1)[1] in ("images", "points"):
                draft.cameras[key] = (by_group[value.split("/", 1)[0]], value.split("/", 1)[1])
            elif key == "prefix_path":
                draft.output_extra[key] = value
            else:
                draft.raw_bridges[key] = value
        return draft


def spec_inputs(spec: str) -> set:
    """Input names a stream spec refers to: ``a.x - b.x`` -> {a, b}; ``sin(a.position[:])`` -> {a}."""
    names = set()
    for operand in re.split(r"\s[+\-*/]\s", spec):
        operand = re.sub(r"^(sin|cos)\((.*)\)$", r"\2", operand.strip())
        if "." in operand:
            names.add(operand.split(".", 1)[0])
    return names


def suggest_input_name(kind: str, topic: str) -> str:
    """A name that starts with the prefix the Record page needs for this kind of input, e.g. ``image_camera_color``."""
    prefix = _PREFIXES.get(kind, "tool")
    skip = {prefix, "image", "raw", "rect", "data", "msg", "states"} if kind != "joint_state" else {prefix, "msg"}
    words = [w for w in re.split(r"[\W_]+", topic.lower()) if w and w not in skip]
    rest = "_".join(words[-2:])
    return f"{prefix}_{rest}" if rest else prefix
