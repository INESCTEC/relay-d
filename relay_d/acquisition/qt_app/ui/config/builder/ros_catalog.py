"""What the live ROS 2 graph offers the Config builder: topics, TF frames and the numeric fields of a message type.

No Qt and no rclpy import at module level: the node, the TF buffer and the message loader are passed in, so tests use fakes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import yaml

# message type -> kind of input (the kind picks the input-name prefix, see draft.suggest_input_name)
KINDS = {
    "sensor_msgs/msg/Image": "image",
    "sensor_msgs/msg/CompressedImage": "image",
    "sensor_msgs/msg/PointCloud2": "pointcloud",
    "sensor_msgs/msg/JointState": "joint_state",
    "geometry_msgs/msg/Pose": "pose",
    "geometry_msgs/msg/PoseStamped": "pose",
    "geometry_msgs/msg/TransformStamped": "transform",
    "geometry_msgs/msg/Twist": "twist",
    "geometry_msgs/msg/TwistStamped": "twist",
    "geometry_msgs/msg/Wrench": "wrench",
    "geometry_msgs/msg/WrenchStamped": "wrench",
    "tf2_msgs/msg/TFMessage": "tf",
}
_SCALAR = re.compile(r"std_msgs/msg/(Float|Int|UInt)\d+(MultiArray)?$")
_HIDDEN = ("/rosout", "/parameter_events", "/robot_description", "/clock")
_NUMERIC = re.compile(r"^(float|double|u?int\d+|byte|octet|char)$")
TF_FIELDS = [f"translation.{a}" for a in "xyz"] + [f"rotation.{a}" for a in "xyzw"]  # a TF input adds the ``transform.`` prefix itself


@dataclass(frozen=True)
class TopicInfo:
    name: str
    type: str  # "pkg/msg/Name"
    kind: str  # one of KINDS.values(), "scalar" or "other"


def classify(type_name: str) -> str:
    return KINDS.get(type_name) or ("scalar" if _SCALAR.match(type_name) else "other")


def guess_modality(topic: str) -> str:
    """``depth`` if the topic name says so, else ``rgb`` (the encoding is only known once a frame arrives)."""
    return "depth" if "depth" in topic.lower() else "rgb"


def list_topics(node) -> List[TopicInfo]:
    """Topics of the live graph (``node.get_topic_names_and_types()``), sorted by name. A topic with several types gets one row per type."""
    rows = [
        TopicInfo(name, t, classify(t))
        for name, types in node.get_topic_names_and_types()
        if name not in _HIDDEN
        for t in types
    ]
    return sorted(rows, key=lambda r: (r.name, r.type))


def tf_frames(buffer) -> List[str]:
    """Every frame the TF buffer knows (any two connected frames can be a parent/child pair), sorted."""
    frames = yaml.safe_load(buffer.all_frames_as_yaml()) or {}
    # root frames only appear as a "parent", never as a key
    return sorted(set(frames) | {v["parent"] for v in frames.values() if isinstance(v, dict) and v.get("parent")})


def default_loader(type_name: str):
    from rosidl_runtime_py.utilities import get_message

    return get_message(type_name)


def message_fields(type_name: str, loader: Callable[[str], type] = default_loader) -> List[Tuple[str, bool]]:
    """Numeric leaves of a message as ``(path, is_array)``, e.g. ``("position", True)``, ``("pose.position.x", False)``.

    Strings, booleans, headers and arrays of messages are left out (they cannot be a stream); covariance matrices are too, they only add noise.
    """
    out: List[Tuple[str, bool]] = []

    def walk(cls, prefix: str):
        for field, ftype in cls.get_fields_and_field_types().items():
            if field == "header" or field.endswith("covariance"):
                continue
            path = prefix + field
            array = re.fullmatch(r"(?:sequence<(.+?)(?:, \d+)?>|(.+?)\[\d*\])", ftype)
            base = array.group(1) or array.group(2) if array else ftype
            if _NUMERIC.match(base):
                out.append((path, bool(array)))
            elif "/" in base and not array:
                walk(loader(base if "/msg/" in base else base.replace("/", "/msg/", 1)), path + ".")

    walk(loader(type_name), "")
    return out


def spec_for(input_name: str, path: str, is_array: bool, index: Optional[int] = None) -> str:
    """``joint_1.position[:]``, ``joint_1.position[2]`` (one element of an array) or ``tf_eef.translation.x``."""
    if is_array:
        return f"{input_name}.{path}[{':' if index is None else index}]"
    return f"{input_name}.{path}"


def joint_names(msg) -> List[str]:
    """Joint names from a received JointState (what ``joint_names:`` of the input should list)."""
    return list(getattr(msg, "name", []) or [])
