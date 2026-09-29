"""
obs_builder.py
--------------
Builds the observation dict for model inference by subscribing to ROS2 topics
and performing TF lookups, as configured in obs_config.yaml.

Config format (obs_config.yaml):
  config:
    sync:
      threshold: 0.05   # seconds — max allowed timestamp gap between an
                        # observation's live (non-static) inputs. Static
                        # inputs are always stamped with "now" and never
                        # cause a desync on their own. Omit this section to
                        # disable the check entirely.

    input:
      <name>:                        # topic input — type auto-discovered
        topic: /some/topic
        joint_names: [j1, j2, ...]   # optional: filter + order JointState joints
      <name>:                        # TF input
        parent_frame: base
        child_frame: tool0
        tf_topic: /isaac/tf          # optional, default /tf. Read this TF
                                     # tree from a topic other than the
                                     # global /tf (e.g. a namespaced
                                     # robot_state_publisher). A custom
                                     # tf_topic that can't be subscribed to,
                                     # or never produces this transform,
                                     # raises TfTopicSubscriptionError —
                                     # it never silently falls back to /tf.
        # tf_static_topic: /tf_static  # optional, default shown
      <name>:                        # static (constant) input — no topic needed
        static: [1.0, 0.0]           # list or scalar; exposed as .data field
                                     # use <name>.data[:] or <name>.data in specs

    observations:
      # Plain list form — no normalization:
      <obs_key>:
        - <input>.<field>            # scalar
        - <input>.<field>[:]         # expand iterable (JointState.position etc.)
        - sin(<input>.<field>[:])    # element-wise trig
        - zero:N                     # N literal zeros

      # Dict form — with optional normalization:
      <obs_key>:
        specs:
          - <input>.<field>[:]
        normalize: true
        normalized_range: [-1.0, 1.0]
        limits:
          - {lower_lim: -3.14, upper_lim: 3.14}   # per-element; one entry → broadcast

      # Arithmetic between two scalar specs (spaces around the operator required):
      <obs_key>:
        - <input_a>.<field> - <input_b>.<field>   # + - * / all supported

      # Dict form — quaternion (x,y,z,w) converted to roll/pitch/yaw:
      <obs_key>:
        specs: [<in>.<q>.x, <in>.<q>.y, <in>.<q>.z, <in>.<q>.w]
        rotation_format: rpy   # 4 specs in -> 3 values out, applied BEFORE
                                # normalization (limits takes 3 entries).
                                # Works for ANY quaternion-bearing input, not
                                # just TF (unlike <tf_input>.euler.roll).

      # Passthrough — camera / organized pointcloud obs (no specs, no
      # normalization/rotation-format: the array is used as-is, exactly as
      # decoded from the live message). `source` must name an `input:`
      # entry whose topic resolves to sensor_msgs/Image or an ORGANIZED
      # sensor_msgs/PointCloud2 (height > 1) — mirrors the acquisition-side
      # bridge-map (`output: <obs_key>: <input_name>/images`) that exposes
      # the same data as obs/next_obs in the training dataset. Requires
      # `--backend robomimic` at inference time — the custom backend's
      # policy networks have no vision encoder and will raise a clear error
      # if handed one of these.
      <obs_key>:
        source: <input_name>
"""

import operator
import re
import threading
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import Buffer, TransformListener

from rosidl_runtime_py.utilities import get_message
from relay_d.utils.config_loader import load_yaml_config
from relay_d.utils.ros_media_codec import decode_image, decode_organized_pointcloud
from relay_d.utils.tf_topic import get_or_create_tf_buffer


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass
class _NormConfig:
    """Per-stream normalization parameters.

    limits: list of (lower_lim, upper_lim) tuples — one per element, or a single
            entry that is broadcast to every element in the stream.
    """
    limits: List[Tuple[float, float]]
    norm_min: float = -1.0
    norm_max: float =  1.0


def _apply_normalization(arr: np.ndarray, nc: _NormConfig) -> np.ndarray:
    """Linearly map each element from its real-world range to normalized_range.

    Symmetric inverse of action_dispatcher._apply_limits.
    """
    out = np.empty_like(arr)
    for i, v in enumerate(arr):
        lo, hi = nc.limits[min(i, len(nc.limits) - 1)]
        t = float(v - lo) / float(hi - lo) if hi != lo else 0.0
        out[i] = t * (nc.norm_max - nc.norm_min) + nc.norm_min
    return out


def _get_specs(stream_def) -> List[str]:
    """Return the spec list from either a plain list or a dict stream definition."""
    return stream_def if isinstance(stream_def, list) else stream_def.get("specs", [])


# ---------------------------------------------------------------------------
# Spec parsing helpers
# ---------------------------------------------------------------------------

_TRIG_OPS = {"sin": np.sin, "cos": np.cos}

_BIN_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

# "<spec> <op> <spec>" — whitespace around the operator is REQUIRED, which is
# what keeps this from matching ordinary dotted field paths (they never
# contain whitespace).
_BINOP_RE = re.compile(r"^(.+?)\s+([+\-*/])\s+(.+)$")


def _strip_trig(spec: str):
    """Strip a sin(...)/cos(...) wrapper that encloses the WHOLE spec.

    Returns (math_op | None, inner_spec).

    Paren-balanced on purpose: "sin(a.x - b.x)" IS a wrapper (its opening
    paren closes on the last char), but "sin(a.x) - cos(b.x)" is NOT (the
    first paren closes early) and must fall through to arithmetic parsing
    instead of being corrupted by a naive startswith/endswith check.
    """
    for name, fn in _TRIG_OPS.items():
        prefix = name + "("
        if spec.startswith(prefix) and spec.endswith(")"):
            depth = 0
            for i, ch in enumerate(spec[len(name):], start=len(name)):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        if i == len(spec) - 1:
                            return fn, spec[len(prefix):-1].strip()
                        break   # closes early -> not an enclosing wrapper
    return None, spec


def _read_scalar(getter, msg, math_op) -> float:
    """Extract exactly one float from `msg` via `getter`, applying optional trig.

    np.asarray(...).reshape(-1)[0]: robust to numpy>=2.0, which raises on
    float() of a 1-element ndarray (e.g. a static input's `.data` field,
    always stored as an ndarray).
    """
    v = float(np.asarray(getter(msg)).reshape(-1)[0])
    return float(math_op(v)) if math_op is not None else v


# ---------------------------------------------------------------------------
# Input synchronization
# ---------------------------------------------------------------------------

def _stamp_to_seconds(stamp) -> float:
    """builtin_interfaces/Time (sec, nanosec) -> float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def _inputs_in_sync(names, timestamps, threshold: Optional[float]) -> bool:
    """True if the given inputs' timestamps all fall within `threshold` seconds
    of each other.

    Always True when threshold is None (sync check disabled) or fewer than
    two of `names` have a recorded timestamp (nothing to compare).
    """
    if threshold is None:
        return True
    ts = [timestamps[n] for n in names if n in timestamps]
    if len(ts) < 2:
        return True
    return (max(ts) - min(ts)) <= threshold


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_to_euler(x: float, y: float, z: float, w: float):
    """Quaternion (x,y,z,w) -> (roll, pitch, yaw) radians, ZYX convention."""
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# TF enriched wrapper
# ---------------------------------------------------------------------------

def _make_enriched_tf(tf_stamped) -> types.SimpleNamespace:
    """
    Wrap a TransformStamped so that attrgetter works for all supported TF specs:
      translation.x/y/z
      rotation.x/y/z/w
      euler.roll/pitch/yaw         — ZYX radians
      rot6d.i0..i5                 — first two columns of R, column-major
      rot_matrix.r00..r22          — full 3x3, row-major
    """
    t = tf_stamped.transform.translation
    r = tf_stamped.transform.rotation
    qx, qy, qz, qw = r.x, r.y, r.z, r.w

    roll, pitch, yaw = _quat_to_euler(qx, qy, qz, qw)
    euler = types.SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw)

    R = _quat_to_rot_matrix(qx, qy, qz, qw)
    rot_matrix = types.SimpleNamespace(
        r00=R[0, 0], r01=R[0, 1], r02=R[0, 2],
        r10=R[1, 0], r11=R[1, 1], r12=R[1, 2],
        r20=R[2, 0], r21=R[2, 1], r22=R[2, 2],
    )
    rot6d_vals = list(R[:, 0]) + list(R[:, 1])
    rot6d = types.SimpleNamespace(**{f"i{i}": float(v) for i, v in enumerate(rot6d_vals)})

    return types.SimpleNamespace(
        transform=types.SimpleNamespace(
            translation=t,
            rotation=r,
            euler=euler,
            rot_matrix=rot_matrix,
            rot6d=rot6d,
        )
    )


# ---------------------------------------------------------------------------
# JointState filtered view
# ---------------------------------------------------------------------------

class _JointStateView:
    """
    Wraps a JointState message and presents filtered, ordered arrays based on
    a configured joint_names list.  attrgetter("position")(view) yields the
    filtered/ordered list instead of the full robot joint array.
    """

    def __init__(self, msg, joint_names: List[str]):
        all_names = list(msg.name)
        indices = [all_names.index(n) for n in joint_names if n in all_names]
        self.name     = [joint_names[j] for j in range(len(indices))]
        self.position = [float(msg.position[i]) for i in indices] if msg.position else []
        self.velocity = [float(msg.velocity[i]) for i in indices] if msg.velocity else []
        self.effort   = [float(msg.effort[i])   for i in indices] if msg.effort   else []


# ---------------------------------------------------------------------------
# Extractor tuple: (input_name, payload, is_slice, math_op)
#
#   normal   : (input_name,  attrgetter,    is_slice, math_op | None)
#   zero-fill: ("__zero__",  zero_count,    False,    None)
#   binary op: ("__binop__", binop_payload, False,    outer_math_op | None)
#
# binop_payload = (raw_spec, l_input, l_getter, l_math_op,
#                  op_func, r_input, r_getter, r_math_op)
# ---------------------------------------------------------------------------
_Entry = Tuple[str, Any, bool, Any]


# ---------------------------------------------------------------------------
# ObsBuilder
# ---------------------------------------------------------------------------

class ObsBuilder(Node):
    """
    ROS2 node that maintains the latest observation dict.

    Subscriptions and TF lookups are driven entirely by obs_config.yaml.
    Call get_obs() from the inference loop to get the latest snapshot.
    """

    def __init__(self, obs_config_path: str):
        super().__init__("obs_builder")

        cfg = load_yaml_config(obs_config_path)["config"]

        self._input_cfg: Dict[str, dict]      = cfg.get("input", {})
        self._obs_cfg:   Dict[str, List[str]] = cfg.get("observations", {})

        sync_cfg = cfg.get("sync", {}) or {}
        self._sync_threshold: Optional[float] = sync_cfg.get("threshold")

        self._lock = threading.Lock()
        self._latest_inputs: Dict[str, Any]        = {}   # raw messages / enriched TF
        self._input_timestamps: Dict[str, float]   = {}   # per-input timestamp, seconds
        self._latest:        Dict[str, np.ndarray] = {}   # evaluated obs (sample-and-hold)

        self._inputs_received: set        = set()  # inputs that have yielded ≥1 real message
        self._tf_warned: set              = set()  # throttle: warn once per failing TF input
        self._joint_mismatch_warned: set  = set()  # throttle: warn once per joint filter input
        self._desync_warned: set          = set()  # throttle: warn once per desynced obs
        self._passthrough_decode_warned: set = set()  # throttle: warn once per bad passthrough source

        self._cv_bridge = CvBridge()  # only used for passthrough (camera) obs

        # Classify inputs
        self._tf_inputs: set                      = set()
        self._static_inputs: set                  = set()
        self._joint_filters: Dict[str, List[str]] = {}
        for name, spec in self._input_cfg.items():
            if "parent_frame" in spec and "child_frame" in spec:
                self._tf_inputs.add(name)
            if "joint_names" in spec:
                self._joint_filters[name] = spec["joint_names"]
            if "static" in spec:
                self._static_inputs.add(name)

        # Pre-populate static inputs — always available, never overwritten by callbacks
        for name in self._static_inputs:
            val = self._input_cfg[name]["static"]
            if isinstance(val, (int, float)):
                arr = np.array([float(val)], dtype=np.float32)
            else:
                arr = np.array(val, dtype=np.float32)
            self._latest_inputs[name] = types.SimpleNamespace(data=arr)

        # Build per-obs extractor lists and normalization configs
        self._extractors:      Dict[str, List[_Entry]]          = {}
        self._norm_params:     Dict[str, Optional[_NormConfig]] = {}
        self._rotation_format: Dict[str, Optional[str]]         = {}
        self._obs_input_names: Dict[str, set]                   = {}
        self._passthrough_obs: Dict[str, str]                   = {}  # obs_name -> input_name
        self._build_extractors()
        # Inputs whose live messages must be decoded (Image/organized
        # PointCloud2) into a plain array instead of stored as a raw ROS
        # message, because at least one obs references them via `source:`.
        self._passthrough_sources: set = set(self._passthrough_obs.values())

        # NOTE: _latest intentionally starts empty. It is populated only when real
        # data arrives via callbacks, so _wait_for_obs() correctly blocks until
        # actual sensor data has been received (not pre-filled zeros).

        # TF — one Buffer per distinct (tf_topic, tf_static_topic) pair
        # referenced by config.input, keyed the same way as the acquisition
        # side (relay_d.utils.tf_topic). Inputs that don't set tf_topic /
        # tf_static_topic default to the standard /tf, /tf_static pair,
        # which uses the stock tf2_ros.TransformListener below. A TF input
        # that DOES configure a custom pair goes through
        # get_or_create_tf_buffer, which raises TfTopicSubscriptionError
        # (uncaught here — crashes node startup) if that custom topic can't
        # be subscribed to. This is deliberate: a broken custom tf_topic
        # must never silently fall back to reading the default /tf tree.
        self._tf_buffers: Dict[tuple, Buffer] = {}
        self._tf_listeners: dict = {}
        self._DEFAULT_TF_KEY = ("/tf", "/tf_static")
        self._tf_input_buffer: Dict[str, Buffer] = {}

        if self._tf_inputs:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._tf_buffers[self._DEFAULT_TF_KEY] = self._tf_buffer
            self._tf_listeners[self._DEFAULT_TF_KEY] = self._tf_listener

            for name in self._tf_inputs:
                spec = self._input_cfg[name]
                tf_topic = spec.get("tf_topic", "/tf")
                tf_static_topic = spec.get("tf_static_topic", "/tf_static")
                self._tf_input_buffer[name] = get_or_create_tf_buffer(
                    self._tf_buffers,
                    self._tf_listeners,
                    self,
                    tf_topic,
                    tf_static_topic,
                    self._DEFAULT_TF_KEY,
                )

        # Topic subscriptions (msg type auto-discovered from topic graph)
        self._pending_topics: List[Tuple[str, str]] = []
        self._retry_timer = None
        self._setup_subscriptions()

        if self._tf_inputs:
            self.create_timer(0.02, self._tf_callback)   # 50 Hz TF poll

        self.get_logger().info("ObsBuilder ready.")

    # ------------------------------------------------------------------
    # Extractor building
    # ------------------------------------------------------------------

    def _build_extractors(self) -> None:
        for obs_name, stream_def in self._obs_cfg.items():
            # Passthrough obs: {source: <input_name>} — no specs, no
            # normalization/rotation-format. Deliberately kept OUT of
            # self._extractors/_norm_params/_rotation_format entirely (those
            # all assume a flat float row); handled by its own pass in
            # _topic_callback/_evaluate_all instead.
            if isinstance(stream_def, dict) and "source" in stream_def:
                input_name = stream_def["source"]
                if input_name not in self._input_cfg:
                    raise ValueError(
                        f"obs '{obs_name}': source '{input_name}' is not a "
                        f"configured input. Check config.input in obs_config.yaml."
                    )
                self._passthrough_obs[obs_name] = input_name
                continue

            # Parse normalization config (dict form only)
            if isinstance(stream_def, list):
                self._norm_params[obs_name] = None
                rot_fmt = None
            else:
                if stream_def.get("normalize", False):
                    norm_min, norm_max = stream_def.get("normalized_range", [-1.0, 1.0])
                    limits = [
                        (float(l["lower_lim"]), float(l["upper_lim"]))
                        for l in stream_def.get("limits", [])
                    ]
                    if not limits:
                        raise ValueError(
                            f"obs '{obs_name}': normalize=true but 'limits' is "
                            f"empty or missing in obs_config.yaml"
                        )
                    self._norm_params[obs_name] = _NormConfig(limits, norm_min, norm_max)
                else:
                    self._norm_params[obs_name] = None

                # rotation_format (dict form only)
                rot_fmt = stream_def.get("rotation_format")
                if rot_fmt is not None:
                    rot_fmt = str(rot_fmt).strip().lower()
                    if rot_fmt != "rpy":
                        raise ValueError(
                            f"obs '{obs_name}': unsupported rotation_format "
                            f"'{stream_def['rotation_format']}' — only 'rpy' "
                            f"(quaternion x,y,z,w -> roll,pitch,yaw) is supported."
                        )
            self._rotation_format[obs_name] = rot_fmt

            specs = _get_specs(stream_def)
            entries: List[_Entry] = []
            for raw_spec in specs:
                spec = raw_spec.strip()

                # sin() / cos() wrapper — only when it encloses the whole spec,
                # so "sin(a.x) - cos(b.x)" falls through to arithmetic parsing.
                math_op, spec = _strip_trig(spec)

                # zero:N sentinel
                if spec.startswith("zero:"):
                    entries.append(("__zero__", int(spec[5:]), False, None))
                    continue

                # "<spec> <op> <spec>" arithmetic between two scalar specs
                m = _BINOP_RE.match(spec)
                if m is not None:
                    entries.append(self._parse_binop(obs_name, raw_spec, m, math_op))
                    continue

                entries.append(self._parse_atomic(obs_name, raw_spec, spec, math_op))

            if rot_fmt == "rpy":
                self._validate_rpy(obs_name, specs, entries)

            self._extractors[obs_name] = entries

            # Real underlying inputs this obs depends on (used for the sync
            # check). __binop__ entries expand to BOTH of their operand inputs.
            real_inputs: set = set()
            for inp_name, payload, _, _ in entries:
                if inp_name == "__zero__":
                    continue
                if inp_name == "__binop__":
                    real_inputs.update((payload[1], payload[5]))
                else:
                    real_inputs.add(inp_name)
            self._obs_input_names[obs_name] = {
                n for n in real_inputs if n not in self._static_inputs
            }

    def _parse_atomic(self, obs_name: str, raw_spec: str, spec: str, math_op) -> _Entry:
        """Parse one atomic '<input>.<field_path>' or '<input>.<field_path>[:]'.

        `spec` must already have any sin()/cos() wrapper stripped (the caller
        passes the resulting op in as `math_op`); `raw_spec` is kept only for
        error messages.
        """
        if "." not in spec:
            raise ValueError(
                f"obs '{obs_name}' spec '{raw_spec}': '{spec}' is not a valid "
                f"field spec — expected '<input_name>.<field_path>'."
            )
        dot = spec.index(".")
        input_name = spec[:dot]
        if input_name not in self._input_cfg:
            raise ValueError(
                f"obs '{obs_name}' spec '{raw_spec}' references unknown "
                f"input '{input_name}'. Check config.input in obs_config.yaml."
            )
        raw_field  = spec[dot + 1:]
        is_slice   = raw_field.endswith("[:]")
        field_path = raw_field[:-3] if is_slice else raw_field

        if any(c in field_path for c in "+-*/"):
            raise ValueError(
                f"obs '{obs_name}' spec '{raw_spec}': unexpected operator inside "
                f"field path '{field_path}'. Arithmetic between two specs needs "
                f"spaces around the operator, e.g. 'a.x - b.x'."
            )

        if input_name in self._tf_inputs:
            field_path = "transform." + field_path

        return (input_name, operator.attrgetter(field_path), is_slice, math_op)

    def _parse_binop(self, obs_name: str, raw_spec: str, m, outer_op) -> _Entry:
        """Parse '<spec> <op> <spec>' into a single-value __binop__ entry."""
        op_func = _BIN_OPS[m.group(2)]

        sides = []
        for side_txt, label in ((m.group(1).strip(), "left"),
                                 (m.group(3).strip(), "right")):
            if _BINOP_RE.match(side_txt):
                raise ValueError(
                    f"obs '{obs_name}' spec '{raw_spec}': only one arithmetic "
                    f"operation per spec is supported (chained expression on the "
                    f"{label} side: '{side_txt}')."
                )
            side_op, side_spec = _strip_trig(side_txt)
            inp, getter, is_slice, side_op = self._parse_atomic(
                obs_name, raw_spec, side_spec, side_op
            )
            if is_slice:
                raise ValueError(
                    f"obs '{obs_name}' spec '{raw_spec}': slice specs ('[:]') "
                    f"cannot be arithmetic operands — arithmetic is scalar-only."
                )
            sides.append((inp, getter, side_op))

        (l_inp, l_get, l_op), (r_inp, r_get, r_op) = sides
        payload = (raw_spec.strip(), l_inp, l_get, l_op, op_func, r_inp, r_get, r_op)
        return ("__binop__", payload, False, outer_op)

    def _validate_rpy(self, obs_name: str, specs: List[str],
                       entries: List[_Entry]) -> None:
        """rotation_format: rpy needs exactly 4 plain scalar specs (x, y, z, w)."""
        if len(entries) != 4:
            raise ValueError(
                f"obs '{obs_name}': rotation_format: rpy requires exactly 4 specs "
                f"(quaternion x, y, z, w) — got {len(entries)}."
            )
        for spec_txt, (inp_name, _, is_slice, math_op) in zip(specs, entries):
            if inp_name in ("__zero__", "__binop__") or is_slice or math_op is not None:
                raise ValueError(
                    f"obs '{obs_name}': rotation_format: rpy specs must be plain "
                    f"scalar field paths — '{spec_txt}' is not. zero:N, '[:]' "
                    f"slices, arithmetic and sin()/cos() are not allowed here."
                )

        suffixes = [s.strip().rsplit(".", 1)[-1].lower() for s in specs]
        if suffixes != ["x", "y", "z", "w"]:
            self.get_logger().warn(
                f"obs '{obs_name}': rotation_format: rpy expects the 4 specs in "
                f"quaternion x, y, z, w order — got {suffixes}. Verify the order "
                f"(w LAST, not first); a wrong order silently yields wrong angles."
            )

        nc = self._norm_params.get(obs_name)
        if nc is not None and len(nc.limits) not in (1, 3):
            raise ValueError(
                f"obs '{obs_name}': rotation_format: rpy produces 3 values "
                f"(roll, pitch, yaw) but 'limits' has {len(nc.limits)} entries — "
                f"provide 3 (one per axis) or 1 (broadcast to all)."
            )

    def _compute_initial_obs(self) -> Dict[str, np.ndarray]:
        """Pre-fill zeros for observations whose element counts are known at parse time.

        Normalization is NOT applied to the zero-filled arrays — they will be
        replaced with real (normalized) values on the first data callback.
        """
        initial = {}
        for obs_name, entries in self._extractors.items():
            size = 0
            known = True
            for inp_name, payload, is_slice, _ in entries:
                if inp_name == "__zero__":
                    size += payload
                elif is_slice:
                    jnames = self._joint_filters.get(inp_name)
                    static_ns = self._latest_inputs.get(inp_name)
                    if jnames is not None:
                        size += len(jnames)
                    elif static_ns is not None:
                        size += len(static_ns.data)
                    else:
                        known = False
                        break
                else:
                    size += 1   # scalar spec always contributes 1 element
            if known:
                if self._rotation_format.get(obs_name) == "rpy":
                    size = 3   # 4 quaternion components collapse to (r, p, y)
                initial[obs_name] = np.zeros(size, dtype=np.float32)
        return initial

    # ------------------------------------------------------------------
    # Subscription setup with auto type discovery
    # ------------------------------------------------------------------

    def _setup_subscriptions(self) -> None:
        topic_map = {name: tlist for name, tlist in self.get_topic_names_and_types()}
        for input_name, spec in self._input_cfg.items():
            if input_name in self._tf_inputs or input_name in self._static_inputs:
                continue
            topic = spec["topic"]
            if topic in topic_map and topic_map[topic]:
                self._subscribe(input_name, topic, get_message(topic_map[topic][0]))
            else:
                self.get_logger().warn(
                    f"Topic '{topic}' not yet advertised — will retry every 1 s."
                )
                self._pending_topics.append((input_name, topic))

        if self._pending_topics:
            self._retry_timer = self.create_timer(1.0, self._retry_subscriptions)

    def _subscribe(self, input_name: str, topic: str, msg_class) -> None:
        self.create_subscription(
            msg_class,
            topic,
            lambda msg, n=input_name: self._topic_callback(msg, n),
            10,
        )
        self.get_logger().info(
            f"Subscribed to '{topic}' ({msg_class.__name__}) as input '{input_name}'"
        )

    def _retry_subscriptions(self) -> None:
        topic_map = {name: tlist for name, tlist in self.get_topic_names_and_types()}
        remaining = []
        for input_name, topic in self._pending_topics:
            if topic in topic_map and topic_map[topic]:
                self._subscribe(input_name, topic, get_message(topic_map[topic][0]))
            else:
                remaining.append((input_name, topic))
        self._pending_topics = remaining
        if not remaining and self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _decode_passthrough(self, input_name: str, msg: Any) -> Optional[np.ndarray]:
        """Decode a camera/pointcloud message for a passthrough obs into a
        plain numpy array (no flatten/normalize) via the shared codec module,
        so a live obs matches what a trained model saw during postprocessing
        (relay_d/acquisition/qt_app/ui/postprocess/yaml_driven_converter.py's
        bridge-map obs). Returns None (and warns once) on anything it can't
        decode: a message type other than Image/PointCloud2, or an
        unorganized (height<=1) cloud.
        """
        if isinstance(msg, Image):
            is_depth = self._input_cfg[input_name].get("modality") == "depth"
            decoded = decode_image(msg, self._cv_bridge, is_depth=is_depth)
        elif isinstance(msg, PointCloud2):
            decoded = decode_organized_pointcloud(msg)
        else:
            decoded = None
            if input_name not in self._passthrough_decode_warned:
                self._passthrough_decode_warned.add(input_name)
                self.get_logger().error(
                    f"Input '{input_name}' is a {type(msg).__name__}, not an "
                    f"Image or PointCloud2 — every obs with 'source: "
                    f"{input_name}' will never produce data."
                )
            return decoded

        if decoded is None and input_name not in self._passthrough_decode_warned:
            self._passthrough_decode_warned.add(input_name)
            self.get_logger().error(
                f"Input '{input_name}': decode failed (for PointCloud2 this "
                f"means it's unorganized/height<=1 — not supported as an obs; "
                f"see the 'output.streams' comment block in config_template.yaml)."
            )
        return decoded

    def _topic_callback(self, msg: Any, input_name: str) -> None:
        # Capture timestamp before any wrapping (e.g. _JointStateView) hides
        # the original message's header.
        if hasattr(msg, "header"):
            ts = _stamp_to_seconds(msg.header.stamp)
        else:
            ts = _stamp_to_seconds(self.get_clock().now().to_msg())

        if input_name in self._passthrough_sources:
            decoded = self._decode_passthrough(input_name, msg)
            if decoded is None:
                return   # sample-and-hold: leave any previous value untouched
            with self._lock:
                self._latest_inputs[input_name] = decoded
                self._input_timestamps[input_name] = ts
            self._inputs_received.add(input_name)
            self._evaluate_all()
            return

        jnames = self._joint_filters.get(input_name)
        if jnames is not None:
            if hasattr(msg, "name") and input_name not in self._joint_mismatch_warned:
                msg_names = list(msg.name)
                missing = [n for n in jnames if n not in msg_names]
                if missing:
                    self._joint_mismatch_warned.add(input_name)
                    self.get_logger().warn(
                        f"[joint_names] Input '{input_name}': configured joints not found "
                        f"in message.\n"
                        f"  Configured : {jnames}\n"
                        f"  Published  : {msg_names}\n"
                        f"  Missing    : {missing}\n"
                        f"  Update 'joint_names' in obs_config.yaml to match exactly."
                    )
            msg = _JointStateView(msg, jnames)
        with self._lock:
            self._latest_inputs[input_name] = msg
            self._input_timestamps[input_name] = ts
        self._inputs_received.add(input_name)
        self._evaluate_all()

    def _tf_callback(self) -> None:
        updates = {}
        ts_updates = {}
        for name in self._tf_inputs:
            spec = self._input_cfg[name]
            try:
                tf = self._tf_input_buffer[name].lookup_transform(
                    spec["parent_frame"],
                    spec["child_frame"],
                    rclpy.time.Time(),
                )
                updates[name] = _make_enriched_tf(tf)
                ts_updates[name] = _stamp_to_seconds(tf.header.stamp)
                self._inputs_received.add(name)
            except Exception as exc:
                if name not in self._tf_warned:
                    self._tf_warned.add(name)
                    self.get_logger().warn(
                        f"[TF] Lookup failed for '{name}' "
                        f"({spec['parent_frame']} → {spec['child_frame']}): {exc}"
                    )
        if updates:
            with self._lock:
                self._latest_inputs.update(updates)
                self._input_timestamps.update(ts_updates)
            self._evaluate_all()

    # ------------------------------------------------------------------
    # Stream evaluation
    # ------------------------------------------------------------------

    def _evaluate_all(self) -> None:
        now = _stamp_to_seconds(self.get_clock().now().to_msg())
        with self._lock:
            snapshot = dict(self._latest_inputs)
            ts_snapshot = dict(self._input_timestamps)
        for name in self._static_inputs:
            ts_snapshot[name] = now

        new_vals: Dict[str, np.ndarray] = {}

        for obs_name, entries in self._extractors.items():
            input_names = self._obs_input_names[obs_name]
            if not _inputs_in_sync(input_names, ts_snapshot, self._sync_threshold):
                self._warn_desync(obs_name, input_names, ts_snapshot)
                continue   # sample-and-hold: leave self._latest[obs_name] untouched

            row: List[float] = []
            ok = True

            for inp_name, payload, is_slice, math_op in entries:
                if inp_name == "__zero__":
                    row.extend([0.0] * payload)
                    continue

                if inp_name == "__binop__":
                    (spec_txt, l_inp, l_get, l_op,
                     op_func, r_inp, r_get, r_op) = payload
                    l_raw = snapshot.get(l_inp)
                    r_raw = snapshot.get(r_inp)
                    if l_raw is None or r_raw is None:
                        ok = False
                        break
                    try:
                        v = float(op_func(_read_scalar(l_get, l_raw, l_op),
                                           _read_scalar(r_get, r_raw, r_op)))
                        row.append(float(math_op(v)) if math_op is not None else v)
                    except Exception as exc:
                        self.get_logger().warn(
                            f"Extraction failed for obs '{obs_name}' spec "
                            f"'{spec_txt}': {exc}"
                        )
                        ok = False
                        break
                    continue

                raw = snapshot.get(inp_name)
                if raw is None:
                    ok = False
                    break

                try:
                    val = payload(raw)   # payload is operator.attrgetter result
                    if is_slice:
                        vals = list(val) if hasattr(val, "__iter__") else [val]
                        if math_op is not None:
                            row.extend(float(math_op(v)) for v in vals)
                        else:
                            row.extend(float(v) for v in vals)
                    else:
                        row.append(_read_scalar(payload, raw, math_op))
                except Exception as exc:
                    self.get_logger().warn(
                        f"Extraction failed for obs '{obs_name}' input '{inp_name}': {exc}"
                    )
                    ok = False
                    break

            if ok:
                if self._rotation_format.get(obs_name) == "rpy":
                    # 4 raw quaternion floats -> (roll, pitch, yaw), BEFORE
                    # normalization, so `limits` describes the Euler ranges.
                    row = [float(v) for v in _quat_to_euler(*row)]
                arr = np.array(row, dtype=np.float32)
                nc = self._norm_params.get(obs_name)
                if nc is not None:
                    arr = _apply_normalization(arr, nc)
                new_vals[obs_name] = arr

        # Passthrough obs (camera/organized-pointcloud): the decoded array is
        # used as-is — no flatten, no float32 cast, no normalization/rotation.
        for obs_name, input_name in self._passthrough_obs.items():
            if not _inputs_in_sync({input_name}, ts_snapshot, self._sync_threshold):
                self._warn_desync(obs_name, {input_name}, ts_snapshot)
                continue
            arr = snapshot.get(input_name)
            if arr is not None:
                new_vals[obs_name] = arr

        if new_vals:
            with self._lock:
                self._latest.update(new_vals)

    def _warn_desync(self, obs_name: str, names: set, timestamps: Dict[str, float]) -> None:
        if obs_name in self._desync_warned:
            return
        self._desync_warned.add(obs_name)
        ts = {n: timestamps[n] for n in names if n in timestamps}
        gap = (max(ts.values()) - min(ts.values())) if len(ts) >= 2 else 0.0
        self.get_logger().warn(
            f"[sync] obs '{obs_name}': inputs {ts} out of sync "
            f"(gap={gap*1000:.1f}ms > threshold={self._sync_threshold*1000:.1f}ms). "
            f"Holding last value. (further desyncs on this obs won't be logged)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Return a thread-safe snapshot of the latest observations."""
        with self._lock:
            return {k: v.copy() for k, v in self._latest.items()}

    def validate_inputs(self, timeout: float = 10.0) -> Tuple[bool, dict]:
        """Check that every configured input source is reachable.

        Polls the ROS graph for topic inputs and calls lookup_transform for TF
        inputs, retrying until timeout. Must be called after the executor is
        spinning (so TF callbacks and topic graph queries are live).

        Returns (all_ok, report) where report[input_name] is a dict with:
          ok: bool, kind: str, detail: str, error: str (on failure only)
        """
        deadline = time.monotonic() + timeout
        report: dict = {}

        # Static inputs — always available
        for name in self._static_inputs:
            report[name] = {
                "ok": True,
                "kind": "static",
                "detail": str(self._input_cfg[name]["static"]),
            }

        # Topic inputs — probe ROS graph with retries
        topic_inputs = {
            name: self._input_cfg[name]["topic"]
            for name in self._input_cfg
            if name not in self._tf_inputs and name not in self._static_inputs
        }
        pending_topics = set(topic_inputs)

        # TF inputs — probe via lookup_transform with retries
        pending_tf = set(self._tf_inputs)

        while (pending_topics or pending_tf) and time.monotonic() < deadline:
            if pending_topics:
                topic_map = {n: tlist for n, tlist in self.get_topic_names_and_types()}
                resolved = set()
                for input_name in list(pending_topics):
                    topic = topic_inputs[input_name]
                    if topic in topic_map and topic_map[topic]:
                        msg_type = topic_map[topic][0]
                        detail = f"topic={topic}  type={msg_type}"
                        if input_name in self._joint_filters:
                            detail += f"  joint_names={self._joint_filters[input_name]}"
                        report[input_name] = {
                            "ok": True,
                            "kind": "topic",
                            "topic": topic,
                            "msg_type": msg_type,
                            "detail": detail,
                        }
                        resolved.add(input_name)
                pending_topics -= resolved

            if pending_tf:
                resolved = set()
                for name in list(pending_tf):
                    spec = self._input_cfg[name]
                    parent, child = spec["parent_frame"], spec["child_frame"]
                    tf_topic = spec.get("tf_topic", "/tf")
                    try:
                        self._tf_input_buffer[name].lookup_transform(
                            parent, child, rclpy.time.Time()
                        )
                        report[name] = {
                            "ok": True,
                            "kind": "tf",
                            "parent": parent,
                            "child": child,
                            "tf_topic": tf_topic,
                            "detail": f"{parent} → {child} (tf_topic={tf_topic})",
                        }
                        resolved.add(name)
                    except Exception:
                        pass
                pending_tf -= resolved

            if pending_topics or pending_tf:
                time.sleep(0.2)

        # Mark remaining as failed with diagnostic hints
        for input_name in pending_topics:
            topic = topic_inputs[input_name]
            try:
                available = sorted(n for n, _ in self.get_topic_names_and_types())
                avail_str = ", ".join(available) if available else "(none advertised)"
            except Exception:
                avail_str = "(could not query)"
            report[input_name] = {
                "ok": False,
                "kind": "topic",
                "topic": topic,
                "detail": "",
                "error": f"Topic '{topic}' not found in ROS graph after {timeout:.0f}s",
                "hint": f"Available topics: {avail_str}",
            }

        for name in pending_tf:
            spec = self._input_cfg[name]
            parent, child = spec["parent_frame"], spec["child_frame"]
            tf_topic = spec.get("tf_topic", "/tf")
            try:
                frames_str = self._tf_input_buffer[name].all_frames_as_string()
            except Exception:
                frames_str = "(tf buffer not ready)"
            report[name] = {
                "ok": False,
                "kind": "tf",
                "parent": parent,
                "child": child,
                "tf_topic": tf_topic,
                "detail": f"{parent} → {child} (tf_topic={tf_topic})",
                "error": (
                    f"TF transform '{parent}' → '{child}' not available on "
                    f"tf_topic '{tf_topic}' after {timeout:.0f}s"
                ),
                "hint": f"Available TF frames:\n{frames_str}",
            }

        all_ok = all(v["ok"] for v in report.values())
        return all_ok, report
