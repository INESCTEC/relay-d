"""Reconstructs ROS2 messages from a recorded demo's HDF5 layout, for playback.

Pure h5py/numpy/ROS-message-construction logic - no PyQt5. Mirrors, in reverse,
the type handling in DataRecorder._process_message_for_recording /
_save_data_type_to_h5_safe, and reuses the same group-classification approach
as demo_validator.py / review_page.py (probe which dataset keys exist).
"""

import json
from dataclasses import dataclass, field
from enum import Enum

import h5py
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, JointState
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import String, Float32, Float64, Int32, Bool

from relay_d.utils.coloring_logger import logger

# Groups under data/<demo_name>/ that are not "raw input" containers.
_NON_INPUT_GROUPS = {"streams", "metadata", "gripper_states"}

_SIMPLE_MSG_CLASSES = {
    "string": String,
    "float32": Float32,
    "float64": Float64,
    "int32": Int32,
    "bool": Bool,
}

_cv_bridge = CvBridge()


class ChannelKind(Enum):
    JOINT_STATE = "joint_state"
    POSE = "pose"
    TWIST = "twist"
    TRANSFORM = "transform"
    IMAGE = "image"
    COMPRESSED_IMAGE = "compressed_image"
    SIMPLE_STD_MSG = "simple_std_msg"
    POINTCLOUD_UNSUPPORTED = "pointcloud_unsupported"
    GENERIC_UNSUPPORTED = "generic_unsupported"


_UNSUPPORTED_KINDS = {ChannelKind.POINTCLOUD_UNSUPPORTED, ChannelKind.GENERIC_UNSUPPORTED}

_KIND_LABELS = {
    ChannelKind.JOINT_STATE: "joint_state",
    ChannelKind.POSE: "pose",
    ChannelKind.TWIST: "twist",
    ChannelKind.TRANSFORM: "transform",
    ChannelKind.IMAGE: "image",
    ChannelKind.COMPRESSED_IMAGE: "compressed_image",
    ChannelKind.SIMPLE_STD_MSG: "simple_value",
}

_SKIP_REASONS = {
    ChannelKind.POINTCLOUD_UNSUPPORTED: "point data not stored at record time",
    ChannelKind.GENERIC_UNSUPPORTED: "unsupported message type",
}

# Channel kinds that map onto a single, directly-publishable ROS message type
# (JOINT_STATE is combined across channels by the reader, TRANSFORM goes out
# via a TransformBroadcaster - neither has its own topic/publisher).
_DIRECT_MSG_CLASSES = {
    ChannelKind.POSE: PoseStamped,
    ChannelKind.TWIST: Twist,
    ChannelKind.IMAGE: Image,
    ChannelKind.COMPRESSED_IMAGE: CompressedImage,
}


def _load_json_attr(grp, key):
    raw = grp.attrs.get(key)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _first_dataset_len(grp, keys):
    for key in keys:
        if key in grp:
            return grp[key].shape[0]
    return 0


@dataclass
class PlaybackChannel:
    """A single recorded input, ready to reconstruct ROS messages frame-by-frame."""

    name: str
    kind: ChannelKind
    group: object = None
    num_frames: int = 0
    joint_names: list = field(default_factory=list)
    parent_frames: list = field(default_factory=list)
    child_frames: list = field(default_factory=list)
    image_format: str = "jpeg"
    simple_type: str = None
    frame_id: str = None

    def _clamp(self, frame_index):
        if self.num_frames <= 0:
            return 0
        return max(0, min(frame_index, self.num_frames - 1))

    def msg_class(self):
        """The ROS message class to publish this channel as, or None if this
        channel has no publisher of its own (JOINT_STATE is combined by the
        reader; TRANSFORM goes out via a TransformBroadcaster; unsupported
        kinds are never published).
        """
        if self.kind == ChannelKind.SIMPLE_STD_MSG:
            return _SIMPLE_MSG_CLASSES.get(self.simple_type)
        return _DIRECT_MSG_CLASSES.get(self.kind)

    def build(self, frame_index, stamp):
        """Return a ROS message for this channel at frame_index, or None if unsupported."""
        idx = self._clamp(frame_index)
        try:
            if self.kind == ChannelKind.JOINT_STATE:
                return self._build_joint_state(idx, stamp)
            if self.kind == ChannelKind.POSE:
                return self._build_pose(idx, stamp)
            if self.kind == ChannelKind.TWIST:
                return self._build_twist(idx)
            if self.kind == ChannelKind.TRANSFORM:
                return self._build_transform(idx, stamp)
            if self.kind == ChannelKind.IMAGE:
                return self._build_image(idx, stamp)
            if self.kind == ChannelKind.COMPRESSED_IMAGE:
                return self._build_compressed_image(idx, stamp)
            if self.kind == ChannelKind.SIMPLE_STD_MSG:
                return self._build_simple(idx)
        except Exception as e:
            logger.error(f"Error reconstructing '{self.name}' frame {idx}: {e}")
        return None

    def _build_joint_state(self, idx, stamp):
        grp = self.group
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id or ""

        n = 0
        if "joint_positions" in grp:
            msg.position = [float(v) for v in grp["joint_positions"][idx]]
            n = len(msg.position)
        if "joint_velocities" in grp:
            msg.velocity = [float(v) for v in grp["joint_velocities"][idx]]
            n = n or len(msg.velocity)
        if "joint_efforts" in grp:
            msg.effort = [float(v) for v in grp["joint_efforts"][idx]]
            n = n or len(msg.effort)

        if self.joint_names:
            msg.name = list(self.joint_names)
        else:
            msg.name = [f"{self.name}_joint_{i}" for i in range(n)]
        return msg

    def _build_pose(self, idx, stamp):
        grp = self.group
        msg = PoseStamped()
        msg.header.stamp = stamp
        # Per-sample frame_id is not persisted at record time for pose data - the
        # channel name is the closest available identifier.
        msg.header.frame_id = self.frame_id or self.name

        if "positions" in grp:
            p = grp["positions"][idx]
            msg.pose.position.x = float(p[0])
            msg.pose.position.y = float(p[1])
            msg.pose.position.z = float(p[2])
        if "orientations" in grp:
            o = grp["orientations"][idx]
            msg.pose.orientation.x = float(o[0])
            msg.pose.orientation.y = float(o[1])
            msg.pose.orientation.z = float(o[2])
            msg.pose.orientation.w = float(o[3])
        else:
            msg.pose.orientation.w = 1.0
        return msg

    def _build_twist(self, idx):
        grp = self.group
        msg = Twist()
        if "linear_velocities" in grp:
            lv = grp["linear_velocities"][idx]
            msg.linear.x, msg.linear.y, msg.linear.z = float(lv[0]), float(lv[1]), float(lv[2])
        if "angular_velocities" in grp:
            av = grp["angular_velocities"][idx]
            msg.angular.x, msg.angular.y, msg.angular.z = float(av[0]), float(av[1]), float(av[2])
        return msg

    def _build_transform(self, idx, stamp):
        grp = self.group
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.parent_frames[idx] if idx < len(self.parent_frames) else self.name
        )
        msg.child_frame_id = (
            self.child_frames[idx] if idx < len(self.child_frames) else f"{self.name}_child"
        )
        t = grp["translations"][idx]
        r = grp["rotations"][idx]
        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])
        msg.transform.rotation.x = float(r[0])
        msg.transform.rotation.y = float(r[1])
        msg.transform.rotation.z = float(r[2])
        msg.transform.rotation.w = float(r[3])
        return msg

    def _build_image(self, idx, stamp):
        # Recorded images were always converted to bgr8 at record time (see
        # DataRecorder._process_image_message), regardless of the original
        # topic's encoding - reconstruct as bgr8, not the stored (original)
        # image_encoding attr.
        arr = np.asarray(self.group["images"][idx])
        msg = _cv_bridge.cv2_to_imgmsg(arr, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id or self.name
        return msg

    def _build_compressed_image(self, idx, stamp):
        raw = self.group["compressed_data"][idx]
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id or self.name
        msg.format = self.image_format
        msg.data = bytes(np.asarray(raw, dtype=np.uint8))
        return msg

    def _build_simple(self, idx):
        raw = self.group["data"][idx]
        raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        item = json.loads(raw)
        msg_cls = _SIMPLE_MSG_CLASSES[self.simple_type]
        msg = msg_cls()
        msg.data = item.get("data")
        return msg


class DemoPlaybackReader:
    """Opens a recorded demo .h5 file (independent of any other open handle) and
    exposes its inputs as PlaybackChannels ready for frame-by-frame reconstruction.
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._h5 = h5py.File(h5_path, "r")

        if "data" not in self._h5 or not list(self._h5["data"].keys()):
            self._h5.close()
            raise ValueError(f"No data/<demo_name> group found in {h5_path}")

        demo_name = list(self._h5["data"].keys())[0]
        self.demo_group = self._h5["data"][demo_name]

        if "timestamps" not in self.demo_group or self.demo_group["timestamps"].shape[0] == 0:
            self._h5.close()
            raise ValueError(f"Demo '{demo_name}' has no timestamps to play back")

        self.timestamps_nsec = self.demo_group["timestamps"][:]
        self.num_frames = int(self.timestamps_nsec.shape[0])
        self.channels = self._scan_channels()

    @property
    def duration_sec(self) -> float:
        if self.num_frames < 2:
            return 0.0
        return float(self.timestamps_nsec[-1] - self.timestamps_nsec[0]) / 1e9

    def elapsed_sec(self, frame_index: int) -> float:
        if self.num_frames == 0:
            return 0.0
        idx = max(0, min(frame_index, self.num_frames - 1))
        return float(self.timestamps_nsec[idx] - self.timestamps_nsec[0]) / 1e9

    def _scan_channels(self):
        channels = []
        for name, item in self.demo_group.items():
            if name in _NON_INPUT_GROUPS or not isinstance(item, h5py.Group):
                continue
            channels.append(self._classify_group(name, item))
        return channels

    def _classify_group(self, name, grp):
        if grp.attrs.get("data_type") == "pointcloud":
            return PlaybackChannel(name=name, kind=ChannelKind.POINTCLOUD_UNSUPPORTED)

        if any(k in grp for k in ("joint_positions", "joint_velocities", "joint_efforts")):
            joint_names = []
            if "joint_names" in grp.attrs:
                try:
                    joint_names = json.loads(grp.attrs["joint_names"])
                except (TypeError, ValueError):
                    joint_names = []
            n = _first_dataset_len(
                grp, ("joint_positions", "joint_velocities", "joint_efforts")
            )
            return PlaybackChannel(
                name=name,
                kind=ChannelKind.JOINT_STATE,
                group=grp,
                num_frames=n,
                joint_names=joint_names,
            )

        if "positions" in grp or "orientations" in grp:
            n = _first_dataset_len(grp, ("positions", "orientations"))
            return PlaybackChannel(name=name, kind=ChannelKind.POSE, group=grp, num_frames=n)

        if "linear_velocities" in grp or "angular_velocities" in grp:
            n = _first_dataset_len(grp, ("linear_velocities", "angular_velocities"))
            return PlaybackChannel(name=name, kind=ChannelKind.TWIST, group=grp, num_frames=n)

        if "translations" in grp and "rotations" in grp:
            n = _first_dataset_len(grp, ("translations",))
            return PlaybackChannel(
                name=name,
                kind=ChannelKind.TRANSFORM,
                group=grp,
                num_frames=n,
                parent_frames=_load_json_attr(grp, "parent_frames"),
                child_frames=_load_json_attr(grp, "child_frames"),
            )

        if "images" in grp:
            n = _first_dataset_len(grp, ("images",))
            return PlaybackChannel(name=name, kind=ChannelKind.IMAGE, group=grp, num_frames=n)

        if "compressed_data" in grp:
            n = _first_dataset_len(grp, ("compressed_data",))
            return PlaybackChannel(
                name=name,
                kind=ChannelKind.COMPRESSED_IMAGE,
                group=grp,
                num_frames=n,
                image_format=grp.attrs.get("format", "jpeg"),
            )

        if "data" in grp and grp["data"].shape[0] > 0:
            try:
                raw = grp["data"][0]
                raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                first = json.loads(raw)
            except (TypeError, ValueError):
                first = {}
            msg_type = first.get("type")
            if msg_type in _SIMPLE_MSG_CLASSES:
                return PlaybackChannel(
                    name=name,
                    kind=ChannelKind.SIMPLE_STD_MSG,
                    group=grp,
                    num_frames=grp["data"].shape[0],
                    simple_type=msg_type,
                )
            return PlaybackChannel(name=name, kind=ChannelKind.GENERIC_UNSUPPORTED)

        return PlaybackChannel(name=name, kind=ChannelKind.GENERIC_UNSUPPORTED)

    def build_combined_joint_state(self, frame_index, stamp):
        """Union every JOINT_STATE channel into one message - robot_state_publisher
        needs a single authoritative joint_states stream, not several partial ones.
        """
        msg = JointState()
        msg.header.stamp = stamp
        names, positions, velocities, efforts = [], [], [], []
        for ch in self.channels:
            if ch.kind != ChannelKind.JOINT_STATE:
                continue
            sub = ch.build(frame_index, stamp)
            if sub is None:
                continue
            names.extend(sub.name)
            positions.extend(sub.position)
            velocities.extend(sub.velocity)
            efforts.extend(sub.effort)
        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        return msg

    def describe(self) -> str:
        supported = [c for c in self.channels if c.kind not in _UNSUPPORTED_KINDS]
        skipped = [c for c in self.channels if c.kind in _UNSUPPORTED_KINDS]

        parts = []
        if supported:
            counts = {}
            for c in supported:
                counts[c.kind] = counts.get(c.kind, 0) + 1
            parts.append(
                ", ".join(f"{n} {_KIND_LABELS[k]}" for k, n in counts.items())
            )
        else:
            parts.append("nothing playable")

        if skipped:
            skip_desc = ", ".join(
                f"{c.name} ({_SKIP_REASONS[c.kind]})" for c in skipped
            )
            parts.append(f"skipped: {skip_desc}")

        return "; ".join(parts)

    def has_playable_channels(self) -> bool:
        return any(c.kind not in _UNSUPPORTED_KINDS for c in self.channels)

    def close(self):
        try:
            self._h5.close()
        except Exception:
            pass
