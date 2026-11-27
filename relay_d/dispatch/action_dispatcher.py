"""
action_dispatcher.py
--------------------
Routes slices of the model's action array to ROS2 topics or action servers,
driven entirely by action_config.yaml.

  action: false (or omitted) -> node.create_publisher  -> publish()
  action: true               -> ActionClient           -> send_goal_async()

The model already outputs real units (robomimic/LeRobot un-normalize with the
checkpoint's stats). An entry's `limits` only clamps each value to
[lower_lim, upper_lim] before its transform; there is no de-normalization.

dry_run=True: no publisher or action client is created; every message and goal
is still built and checked, then logged ("[DRY RUN] ...") instead of sent.

send_stop() (inference stops) publishes each topic entry's `stop:` values and
cancels every goal it sent that is still active.

`max_step` (with `joint_names`) limits each command to the robot's measured position (JointState on `state_topic`)
+- max_step, like LeRobot's max_relative_target: no command can jump further than that in one tick, whatever `limits`
allows. Without a fresh measured position nothing is sent.

`trajectory: true` entries publish trajectory_msgs/JointTrajectory (to a ros2_control joint_trajectory_controller,
which interpolates at its own rate, e.g. over RTDE or FRI): the policy's whole chunk when it has one
(dispatch_trajectory), else one point per tick. Their stop holds the measured position.
"""

from contextlib import nullcontext
import re
import threading
import time

import numpy as np

from action_msgs.msg import GoalStatus
from rclpy.node import Node
from rclpy.action import ActionClient
from typing import Any, Dict, Optional

from rosidl_runtime_py.utilities import get_action as _rosidl_get_action, get_message
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from relay_d.utils.config_loader import load_yaml_config

_GOAL_DONE = (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED)

# Removed config: an old action_config that still has one fails at startup instead of being ignored.
_REMOVED_KEYS = {
    "command": "shell commands are no longer run; use a topic entry (topic + msg_type + mapping) or an action entry (action: true)",
    "dedupe": "topic entries publish every tick; delete the key",
}
_REMOVED_TRANSFORMS = {
    "scale": "use limits for clamping, or record the action in the units the robot takes",
    "clip": "use limits for clamping",
    "vacuum": "use a threshold transform with on_value / off_value and a topic or action entry",
    "magnetic": "use a threshold transform with on_value / off_value and a topic or action entry",
}

def get_action(action_type_str: str):
    """Dynamically load a ROS2 action class from a string like 'move_arm_skill_msgs/action/MoveArmSkill'."""
    parts = action_type_str.split("/")
    if len(parts) != 3 or parts[1] != "action":
        raise ValueError(
            f"Invalid action_type format: '{action_type_str}'. "
            f"Expected '<pkg>/action/<Name>'"
        )
    return _rosidl_get_action(action_type_str)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler (roll, pitch, yaw) radians, ZYX convention -> quaternion [qx, qy, qz, qw]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,   # qx
        cr * sp * cy + sr * cp * sy,   # qy
        cr * cp * sy - sr * sp * cy,   # qz
        cr * cp * cy + sr * sp * sy,   # qw
    ], dtype=np.float32)


def _parse_part(part: str):
    """
    Split "points[0][1]" into ("points", [0, 1]).
    Plain "position" returns ("position", []).
    """
    match = re.match(r'^(\w+)((?:\[\d+\])*)$', part)
    if not match:
        raise ValueError(f"Cannot parse path segment: '{part}'")
    name = match.group(1)
    indices = [int(i) for i in re.findall(r'\[(\d+)\]', match.group(2))]
    return name, indices


def _walk(obj: Any, path: str):
    """Follow a dot path such as "move_point[0].transform.translation.x" to its last segment.

    Returns (parent, last_name, last_indices): the field itself is ``getattr(parent, last_name)[i]...``.
    """
    parts = path.split(".")
    for part in parts[:-1]:
        name, indices = _parse_part(part)
        obj = getattr(obj, name)
        for idx in indices:
            obj = obj[idx]
    name, indices = _parse_part(parts[-1])
    return obj, name, indices


def _assign(obj: Any, path: str, make) -> None:
    """Set the field at ``path`` to ``make(current_value)``, supporting array indexing."""
    parent, name, indices = _walk(obj, path)
    if indices:
        target = getattr(parent, name)
        for idx in indices[:-1]:
            target = target[idx]
        target[indices[-1]] = make(target[indices[-1]])
    else:
        setattr(parent, name, make(getattr(parent, name)))


def _set_nested(obj: Any, path: str, value: float) -> None:
    """Set one scalar at a dot path, cast to the type of the field it overwrites."""
    _assign(obj, path, lambda current: _coerce(current, value))


def _coerce(current: Any, value: float) -> Any:
    """Cast a model output to the type of the field it overwrites (rclpy rejects a float for an int/bool field)."""
    if isinstance(current, bool):
        return float(value) > 0.5  # bool(-0.9) would be True
    if isinstance(current, int):
        return int(round(float(value)))
    return float(value)


def _set_nested_array(obj: Any, path: str, values) -> None:
    """Assign a whole list at a dot path, for float64[] fields (e.g. a JointTrajectoryPoint's `positions`)."""
    values = [float(v) for v in values]
    _assign(obj, path, lambda _current: values)


def _get_nested(obj: Any, path: str) -> Any:
    """Read the value at a dot path, supporting array indexing."""
    parent, name, indices = _walk(obj, path)
    obj = getattr(parent, name)
    for idx in indices:
        obj = obj[idx]
    return obj


def _check_success(msg: Any, fb_cfg: dict, default_field: str = "reached") -> bool:
    """
    Generic success check for a `feedback:` block: read `success_field`
    (default "reached") off `msg` and compare it to `success_value`
    (default True). Lets a config decide between e.g. `reached: true` and
    `skill_status: "REACHED"` without any code change.
    """
    field = fb_cfg.get("success_field", default_field)
    expected = fb_cfg.get("success_value", True)
    try:
        return _get_nested(msg, field) == expected
    except Exception as e:
        logger.warning(
            f"[ActionDispatcher] feedback success_field '{field}' not found "
            f"on {type(msg).__name__}: {e}"
        )
        return False


def _compile_mapping(mapping: dict):
    """Pre-parse dot/index field paths once at startup so _fill_msg() never runs regexes per tick."""
    compiled = []
    for field_path, slice_idx in mapping.items():
        parts = field_path.split(".")
        parent_steps = tuple(_parse_part(p) for p in parts[:-1])
        last_name, last_indices = _parse_part(parts[-1])
        compiled.append((field_path, slice_idx, isinstance(slice_idx, list), parent_steps, last_name, tuple(last_indices)))
    return compiled


def _compile_limits(limits: list, width: int):
    """Precompute vectorized (lo, hi) float32 bounds (-inf/+inf for null entries)."""
    lo = np.full(width, -np.inf, dtype=np.float32)
    hi = np.full(width, np.inf, dtype=np.float32)
    for i in range(width):
        lim = limits[min(i, len(limits) - 1)]
        if lim is not None:
            lo[i] = float(lim["lower_lim"])
            hi[i] = float(lim["upper_lim"])
    return lo, hi


def _fill_msg(msg: Any, cfg: dict, slice_values: np.ndarray, node: Node, compiled_mapping=None) -> bool:
    """Fill header (if present) and mapped fields into a ROS message.

    Returns False if any mapped field could not be set: the message would carry default
    values (zero velocity, zero quaternion) and must not be sent to the robot.
    """
    if hasattr(msg, "header"):  # keep the header: goal_init may have set its frame_id
        msg.header.stamp = node.get_clock().now().to_msg()
        if "frame_id" in cfg:
            msg.header.frame_id = cfg["frame_id"]

    ok = True
    if compiled_mapping is not None:
        for field_path, slice_idx, is_list, parent_steps, last_name, last_indices in compiled_mapping:
            try:
                obj = msg
                for p_name, p_indices in parent_steps:
                    obj = getattr(obj, p_name)
                    for idx in p_indices:
                        obj = obj[idx]
                if is_list:
                    vals = [float(v) for v in slice_values[slice_idx]]
                    if last_indices:
                        target = getattr(obj, last_name)
                        for idx in last_indices[:-1]:
                            target = target[idx]
                        target[last_indices[-1]] = vals
                    else:
                        setattr(obj, last_name, vals)
                else:
                    val = slice_values[slice_idx]
                    if last_indices:
                        target = getattr(obj, last_name)
                        for idx in last_indices[:-1]:
                            target = target[idx]
                        target[last_indices[-1]] = _coerce(target[last_indices[-1]], val)
                    else:
                        setattr(obj, last_name, _coerce(getattr(obj, last_name), val))
            except Exception as e:
                ok = False
                logger.warning(f"[ActionDispatcher] Failed to set {field_path}: {e}")
        return ok

    for field_path, slice_idx in cfg["mapping"].items():
        try:
            if isinstance(slice_idx, list):
                _set_nested_array(msg, field_path, slice_values[slice_idx])
            else:
                _set_nested(msg, field_path, slice_values[slice_idx])
        except Exception as e:
            ok = False
            logger.warning(
                f"[ActionDispatcher] Failed to set {field_path}: {e}"
            )
    return ok


# ---------------------------------------------------------------------------
# ActionDispatcher
# ---------------------------------------------------------------------------

class ActionDispatcher:
    max_state_age: Optional[float] = None  # set by InferenceSession from --max-obs-age; None -> infer_max_obs_age_s

    def __init__(self, node: Node, action_config_path: str, dry_run: bool = False):
        self._node = node
        self._dry_run = dry_run
        if dry_run:
            logger.warning("[ActionDispatcher] DRY RUN: nothing is published or sent to the robot, commands are only logged.")

        config = load_yaml_config(action_config_path)

        self._action_configs = config["actions"]
        self._publishers: Dict[str, Any] = {}
        self._action_clients: Dict[str, ActionClient] = {}
        self._goal_classes: Dict[str, Any] = {}
        self._msg_classes: Dict[str, Any] = {}
        self._compiled_limits: Dict[str, Any] = {}
        self._compiled_mappings: Dict[str, Any] = {}
        self._last_sent: Dict[str, str] = {}
        self._stopped = True  # nothing commanded yet, so there is nothing to stop (see send_stop)
        self._goals = []  # (name, accepted goal handle; None in a dry run) since the last stop: send_stop cancels them
        self._goals_lock = threading.Lock()  # protects _stopped, _goals, _last_sent across executor + control threads
        self._not_ready_warned = set()  # action entries whose server was not ready (warned once)
        self.period = 0.05  # seconds between ticks (set by InferenceSession.run): a trajectory's time step
        self._joint_states: Dict[str, Any] = {}  # state_topic -> ({joint name: position}, monotonic arrival time)
        self._state_lock = threading.Lock()  # joint states arrive on the executor thread
        self._clip_logged: Dict[str, float] = {}  # entry -> last time a max_step clip was logged

        # feedback: -- opt-in ack-wait for topic entries (see _wait_for_ack)
        self._ack_latest: Dict[str, Any] = {}
        self._ack_events: Dict[str, threading.Event] = {}
        self._ack_lock = threading.Lock()

        for name, cfg in self._action_configs.items():
            for key, why in _REMOVED_KEYS.items():
                if key in cfg:
                    raise ValueError(f"[ActionDispatcher] '{name}': '{key}' was removed: {why}.")
            removed = (cfg.get("transform") or {}).get("type")
            if removed in _REMOVED_TRANSFORMS:
                raise ValueError(f"[ActionDispatcher] '{name}': transform '{removed}' was removed: "
                                 f"{_REMOVED_TRANSFORMS[removed]}.")
            if cfg.get("denormalize") or cfg.get("apply_limits") or "normalized_range" in cfg:
                raise ValueError(f"[ActionDispatcher] '{name}': the model already outputs real units; remove "
                                 f"denormalize / normalized_range (limits now only clamp).")
            if "limits" in cfg and len(cfg["limits"]) not in (1, len(cfg["indices"])):
                raise ValueError(f"[ActionDispatcher] '{name}': 'limits' needs 1 entry or one per index "
                                 f"({len(cfg['indices'])}), got {len(cfg['limits'])}.")
            for lim in cfg.get("limits") or []:  # a swapped or non-finite pair would corrupt every output
                if lim is not None and (not isinstance(lim, dict) or lim.get("lower_lim") is None
                                        or lim.get("upper_lim") is None
                                        or not np.isfinite(float(lim["lower_lim"]))
                                        or not np.isfinite(float(lim["upper_lim"]))
                                        or float(lim["lower_lim"]) > float(lim["upper_lim"])):
                    raise ValueError(f"[ActionDispatcher] '{name}': each 'limits' entry needs "
                                     f"lower_lim <= upper_lim, got {lim!r}.")
            if "limits" in cfg:
                self._compiled_limits[name] = _compile_limits(cfg["limits"], len(cfg["indices"]))
            if "mapping" in cfg and not cfg.get("trajectory"):
                self._compiled_mappings[name] = _compile_mapping(cfg["mapping"])
            if "joint_names" in cfg or "max_step" in cfg or cfg.get("trajectory"):
                self._check_joint_entry(name, cfg)
            if cfg.get("trajectory"):
                self._msg_classes[name] = JointTrajectory
                if not dry_run:
                    self._publishers[name] = node.create_publisher(JointTrajectory, cfg["topic"], 10)
                logger.info(f"[ActionDispatcher] '{name}' -> TRAJECTORY {cfg['topic']} ({len(cfg['joint_names'])} joints)")
            elif cfg.get("action", False):
                if "action_type" not in cfg:
                    raise ValueError(
                        f"[ActionDispatcher] '{name}' has action: true "
                        f"but is missing 'action_type'."
                    )
                if "feedback" in cfg and "success_field" not in cfg["feedback"]:
                    raise ValueError(
                        f"[ActionDispatcher] '{name}' has an action "
                        f"'feedback' block but is missing required "
                        f"'success_field' (there's no universal default — "
                        f"Result/Feedback shapes vary per action type)."
                    )
                action_class = get_action(cfg["action_type"])
                self._check_mapping(name, cfg, action_class.Goal)
                self._goal_classes[name] = action_class.Goal
                if not dry_run:
                    self._action_clients[name] = ActionClient(
                        node, action_class, cfg["topic"]
                    )
                logger.info(
                    f"[ActionDispatcher] '{name}' -> ACTION {cfg['topic']} "
                    f"({cfg['action_type']})"
                )
                if "feedback" in cfg:
                    fb = cfg["feedback"]
                    logger.info(
                        f"[ActionDispatcher] '{name}' -> waiting for "
                        f"{fb.get('source', 'result')} before advancing, "
                        f"timeout={fb.get('timeout', 10.0)}s"
                    )

            else:                                    # topic publisher
                if "msg_type" not in cfg:
                    raise ValueError(
                        f"[ActionDispatcher] '{name}' is missing 'msg_type'."
                    )
                msg_class = get_message(cfg["msg_type"])
                self._check_mapping(name, cfg, msg_class)
                self._msg_classes[name] = msg_class
                if not dry_run:
                    self._publishers[name] = node.create_publisher(
                        msg_class, cfg["topic"], 10
                    )
                logger.info(
                    f"[ActionDispatcher] '{name}' -> TOPIC {cfg['topic']} "
                    f"({cfg['msg_type']})"
                )

                if "feedback" in cfg:
                    fb = cfg["feedback"]
                    for key in ("ack_topic", "ack_msg_type"):
                        if key not in fb:
                            raise ValueError(
                                f"[ActionDispatcher] '{name}' has a 'feedback' "
                                f"block but is missing '{key}'."
                            )
                    ack_class = get_message(fb["ack_msg_type"])
                    self._ack_events[name] = threading.Event()
                    node.create_subscription(
                        ack_class, fb["ack_topic"],
                        lambda msg, n=name: self._on_ack(msg, n), 10,
                    )
                    logger.info(
                        f"[ActionDispatcher] '{name}' -> waiting for ack on "
                        f"{fb['ack_topic']} ({fb['ack_msg_type']}) before "
                        f"advancing, timeout={fb.get('timeout', 10.0)}s"
                    )

        for name, cfg in self._action_configs.items():
            if "stop" in cfg:
                self._check_stop(name, cfg)
            elif not cfg.get("action", False) and not cfg.get("trajectory"):  # goals are cancelled, trajectories hold
                logger.warning(f"[ActionDispatcher] '{name}' has no 'stop:': when inference stops, its last command "
                               f"stays active on the robot.")
            if cfg.get("action", False) and "feedback" not in cfg:
                logger.warning(f"[ActionDispatcher] '{name}' sends a new goal every tick, each one preempting the last "
                               f"(a joint_trajectory_controller crashed under that at 20 Hz): for continuous control use a "
                               f"trajectory: true entry; keep actions for discrete moves (with feedback).")
            if "limits" not in cfg and "max_step" not in cfg:
                logger.warning(f"[ActionDispatcher] '{name}' has no 'limits': the model's outputs are sent unclamped.")

        state_topics = {cfg.get("state_topic", "/joint_states") for cfg in self._action_configs.values()
                        if "max_step" in cfg or cfg.get("trajectory")}
        for topic in sorted(state_topics):
            if node is not None:  # read-only, also in a dry run (the commands are computed from it)
                node.create_subscription(JointState, topic, lambda msg, t=topic: self._on_joint_state(msg, t), 10)

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def dispatch(self, action: np.ndarray, trajectories: bool = True) -> None:
        """Send one action to every entry. ``trajectories=False``: skip the trajectory entries (their chunk was sent
        with dispatch_trajectory)."""
        if not np.isfinite(action).all():
            logger.error(f"[ActionDispatcher] Non-finite action, not dispatching: {action}")
            self.send_stop()
            return

        # Phase 1: compute every output. If one config entry is broken (bad slice, unknown transform)
        # this raises before anything was sent, instead of moving the arm and skipping the gripper.
        compiled_limits = getattr(self, "_compiled_limits", None)
        outputs = []
        for name, cfg in self._action_configs.items():
            if cfg.get("trajectory") and not trajectories:
                continue
            slice_values = action[cfg["indices"]]

            # 1. Safety clamp to [lower_lim, upper_lim], in real units
            if "limits" in cfg:
                if compiled_limits is not None and name in compiled_limits:
                    lo, hi = compiled_limits[name]
                    slice_values = np.clip(slice_values.astype(np.float32, copy=False), lo, hi)
                else:
                    slice_values = self._apply_limits(slice_values, cfg["limits"])
            # 1b. No jump beyond max_step from the measured position
            if "max_step" in cfg:
                limited = self._step_limited(name, cfg, slice_values[np.newaxis])
                if limited is None:
                    self.send_stop()
                    return
                slice_values = limited[0]

            # 2. Optional further transform (threshold, euler_to_quat)
            if "transform" in cfg:
                transformed = self._apply_transform(slice_values, cfg["transform"])
            else:
                transformed = slice_values
            if not np.isfinite(transformed).all():  # e.g. a NaN in `limits`
                logger.error(f"[ActionDispatcher] '{name}' output is not finite after its transform, not dispatching: {transformed}")
                self.send_stop()
                return
            outputs.append((name, cfg, transformed))

        # Phase 2: send.
        with self._goals_lock:
            self._stopped = False
        for name, cfg, transformed in outputs:
            if cfg.get("trajectory"):  # no chunk from the policy: one point, reached by the next tick
                self._publish_trajectory(name, cfg, transformed[np.newaxis])
            # ROS2 action client path
            elif cfg.get("action", False):
                key = str(transformed)
                with self._goals_lock:
                    if self._last_sent.get(name) == key:
                        continue
                    # Remember it before sending: a rejection (on the executor thread) may arrive before
                    # _dispatch_action returns, and its pop must win so the goal is retried.
                    self._last_sent[name] = key
                if not self._dispatch_action(name, cfg, transformed):
                    self._forget_sent(name, key)

            # Publisher path: sent every tick (controllers with a watchdog need a stream).
            # A message that cannot be filled (e.g. -1 into a uint8 field) must not leave the other
            # entries moving on their own: stop everything instead.
            elif not self._dispatch_topic(name, cfg, transformed):
                self.send_stop()
                return

    def send_stop(self) -> None:
        """Send each entry's ``stop:`` values once, e.g. a zero velocity: called when inference stops sending (inputs
        stale, model error, Ctrl+C, close), because a controller without a watchdog keeps the last command.

        Nothing happens again until the next dispatch(). Never raises: every entry is tried, failures are logged.
        """
        with self._goals_lock:
            if self._stopped:
                return
            self._stopped = True
            goals, self._goals = self._goals, []
            self._last_sent.clear()  # the cancelled goals must be sent again when inference resumes
        for name, goal_handle in goals:
            self._cancel_goal(name, goal_handle)
        for name, cfg in self._action_configs.items():
            if cfg.get("trajectory"):
                self._hold(name, cfg)
                continue
            if "stop" not in cfg:
                continue
            try:
                msg = self._stop_message(name, cfg, np.asarray(cfg["stop"], dtype=np.float64))
                if msg is None:
                    raise ValueError("a mapped field could not be set")
                if self._dry_run:
                    logger.warning(f"[DRY RUN] '{name}' stop {cfg['stop']} -> {msg}")
                    continue
                self._publishers[name].publish(msg)
                logger.warning(f"[ActionDispatcher] '{name}': sent stop {cfg['stop']}")
            except Exception as e:
                logger.error(f"[ActionDispatcher] '{name}': stop could not be sent: {e}")

    @property
    def has_trajectories(self) -> bool:
        return any(cfg.get("trajectory") for cfg in self._action_configs.values())

    def dispatch_trajectory(self, chunk: np.ndarray) -> None:
        """Send a chunk of actions (n steps x action size) to the trajectory entries: one JointTrajectory each, its
        points ``period`` apart, starting now. Every point is clamped to ``limits`` and to +- max_step from the
        previous one (the first from the measured position)."""
        chunk = np.atleast_2d(np.asarray(chunk, dtype=np.float64))
        if not np.isfinite(chunk).all():
            logger.error("[ActionDispatcher] Non-finite action chunk, not dispatching.")
            self.send_stop()
            return
        compiled_limits = getattr(self, "_compiled_limits", None)
        outputs = []
        for name, cfg in self._action_configs.items():
            if not cfg.get("trajectory"):
                continue
            rows = chunk[:, cfg["indices"]]
            if "limits" in cfg:
                if compiled_limits is not None and name in compiled_limits:
                    lo, hi = compiled_limits[name]
                    rows = np.clip(rows.astype(np.float32, copy=False), lo, hi)
                else:
                    rows = np.stack([self._apply_limits(r, cfg["limits"]) for r in rows])
            if "max_step" in cfg:
                rows = self._step_limited(name, cfg, rows)
                if rows is None:
                    self.send_stop()
                    return
            outputs.append((name, cfg, rows))
        with self._goals_lock:
            self._stopped = False
        for name, cfg, rows in outputs:
            self._publish_trajectory(name, cfg, rows)

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------

    def _check_joint_entry(self, name: str, cfg: dict) -> None:
        """``joint_names`` / ``max_step`` / ``trajectory`` must fit the entry: a wrong one would limit or drive the
        wrong joint."""
        names = cfg.get("joint_names")
        if not isinstance(names, list) or len(names) != len(cfg["indices"]):
            raise ValueError(f"[ActionDispatcher] '{name}': 'joint_names' needs one joint name per index "
                             f"({len(cfg['indices'])}) with max_step or trajectory, got {names!r}.")
        if cfg.get("trajectory"):
            extra = [k for k in ("action", "mapping", "goal_init", "transform", "feedback", "stop") if cfg.get(k)]
            if extra or cfg.get("msg_type", "trajectory_msgs/msg/JointTrajectory") != "trajectory_msgs/msg/JointTrajectory":
                raise ValueError(f"[ActionDispatcher] '{name}': a trajectory entry publishes JointTrajectory positions "
                                 f"itself (its stop holds the measured position); remove {extra or ['msg_type']}.")
            if any(not isinstance(n, str) for n in names):
                raise ValueError(f"[ActionDispatcher] '{name}': every 'joint_names' entry of a trajectory is a joint name.")
        if "max_step" in cfg:
            steps = cfg["max_step"] if isinstance(cfg["max_step"], list) else [cfg["max_step"]]
            if len(steps) not in (1, len(cfg["indices"])) or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and v > 0 for v in steps):
                raise ValueError(f"[ActionDispatcher] '{name}': 'max_step' needs 1 positive number or one per index "
                                 f"({len(cfg['indices'])}), got {cfg['max_step']!r}.")

    def _on_joint_state(self, msg: Any, topic: str) -> None:
        with self._state_lock:
            positions = dict(self._joint_states.get(topic, ({}, 0.0))[0])  # drivers may split joints over messages
            positions.update(zip(msg.name, msg.position))
            self._joint_states[topic] = (positions, time.monotonic())

    def _measured(self, name: str, cfg: dict) -> Optional[np.ndarray]:
        """The entry's measured joint positions (NaN for a null joint name), or None if they are missing or older than
        the stale-input limit (infer_max_obs_age_s): commands must not be limited against an old position."""
        topic = cfg.get("state_topic", "/joint_states")
        with self._state_lock:
            positions, stamp = self._joint_states.get(topic, ({}, 0.0))
        max_age = (
            float(self.max_state_age)
            if self.max_state_age is not None
            else float(user_settings.get("infer_max_obs_age_s") or 1.0)
        )
        missing = [j for j in cfg["joint_names"] if j is not None and j not in positions]
        if not positions or missing or time.monotonic() - stamp > max_age:
            why = f"no {missing} in" if positions and missing else "no fresh"
            logger.error(f"[ActionDispatcher] '{name}': {why} {topic} (max_step / hold need the measured position): "
                         f"not sending.")
            return None
        return np.array([np.nan if j is None else positions[j] for j in cfg["joint_names"]], dtype=np.float64)

    def _step_limited(self, name: str, cfg: dict, rows: np.ndarray) -> Optional[np.ndarray]:
        """``rows`` (n x joints) with each row within +- max_step of the previous one, the first of the measured
        position; None (send nothing) without a fresh measured position."""
        prev = self._measured(name, cfg)
        if prev is None:
            return None
        step = np.broadcast_to(np.asarray(cfg["max_step"], dtype=np.float64), prev.shape)
        out = np.array(rows, dtype=np.float64)
        for i in range(len(out)):
            free = np.isnan(prev)  # a null joint name: not limited
            out[i] = np.where(free, out[i], np.clip(out[i], prev - step, prev + step))
            prev = np.where(free, np.nan, out[i])
        asked = float(np.max(np.abs(rows - out))) if len(out) else 0.0
        if asked > 1e-6 and time.monotonic() - self._clip_logged.get(name, 0.0) > 1.0:
            self._clip_logged[name] = time.monotonic()
            logger.warning(f"[ActionDispatcher] '{name}': command cut by max_step {cfg['max_step']} "
                           f"(the model asked {asked:.3f} further).")
        return out

    def _trajectory_message(self, cfg: dict, rows: np.ndarray) -> Any:
        msg = JointTrajectory()  # header.stamp 0: start now, replacing the trajectory being executed
        msg.joint_names = list(cfg["joint_names"])
        period_ns = float(self.period) * 1e9
        for k, row in enumerate(rows):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in row]
            sec, nanosec = divmod(int(round((k + 1) * period_ns)), 1_000_000_000)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            msg.points.append(point)
        return msg

    def _publish_trajectory(self, name: str, cfg: dict, rows: np.ndarray) -> None:
        msg = self._trajectory_message(cfg, rows)
        if self._dry_run:
            logger.info(f"[DRY RUN] '{name}' -> trajectory of {len(rows)} point(s), first {np.round(rows[0], 4).tolist()}, "
                        f"last {np.round(rows[-1], 4).tolist()} over {len(rows) * self.period:.2f} s")
            return
        self._publishers[name].publish(msg)

    def _hold(self, name: str, cfg: dict) -> None:
        """A trajectory entry's stop: one point at the measured position (the controller holds it); without a fresh
        measured position an empty trajectory, which a joint_trajectory_controller takes as stop the current one.
        Never raises."""
        try:
            present = self._measured(name, cfg)
            if present is None:
                msg = JointTrajectory()
                msg.joint_names = list(cfg["joint_names"])
            else:
                msg = self._trajectory_message(cfg, present[np.newaxis])
            if self._dry_run:
                logger.warning(f"[DRY RUN] '{name}' stop: hold {None if present is None else present.tolist()}")
                return
            self._publishers[name].publish(msg)
            logger.warning(f"[ActionDispatcher] '{name}': stop, holding "
                           f"{'the measured position' if present is not None else '(empty trajectory)'}")
        except Exception as e:
            logger.error(f"[ActionDispatcher] '{name}': stop could not be sent: {e}")


    def _track_goal(self, name: str, goal_handle: Any) -> None:
        """Remember a sent goal for send_stop; one accepted after the stop is cancelled at once."""
        with self._goals_lock:
            if not self._stopped:
                self._goals = [g for g in self._goals if g[1] is None or g[1].status not in _GOAL_DONE]
                if goal_handle is not None or not any(n == name and h is None for n, h in self._goals):
                    self._goals.append((name, goal_handle))
                return
        self._cancel_goal(name, goal_handle)

    def _cancel_goal(self, name: str, goal_handle: Any) -> None:
        """Ask the server to cancel the goal if still active. Not awaited: the stop path must not block. Never raises."""
        try:
            if goal_handle is None:
                logger.warning(f"[DRY RUN] '{name}' cancel the sent goal")
            elif goal_handle.status not in _GOAL_DONE:
                goal_handle.cancel_goal_async()
                logger.warning(f"[ActionDispatcher] '{name}': cancelling the active goal")
        except Exception as e:
            logger.error(f"[ActionDispatcher] '{name}': goal could not be cancelled: {e}")

    def _compiled_mapping_for(self, name: str):
        cm = getattr(self, "_compiled_mappings", None)
        return cm.get(name) if cm is not None else None

    def _stop_message(self, name: str, cfg: dict, values: np.ndarray) -> Any:
        """The topic message carrying ``stop:`` values (no limits, no transform), or None if the mapping does not fit."""
        msg = self._msg_classes[name]()
        if "goal_init" in cfg:
            self._init_goal(msg, cfg["goal_init"], name)
        return msg if _fill_msg(msg, cfg, values, self._node, self._compiled_mapping_for(name)) else None

    def _check_stop(self, name: str, cfg: dict) -> None:
        """A broken ``stop:`` must stop startup: it is only needed at the moment something already went wrong."""
        stop = cfg["stop"]
        if cfg.get("action", False):
            raise ValueError(f"[ActionDispatcher] '{name}': 'stop' is for topic entries; "
                             f"an action entry's still-active goal is cancelled when inference stops.")
        if not isinstance(stop, list) or not stop or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in stop):
            raise ValueError(f"[ActionDispatcher] '{name}': 'stop' must be a list of numbers, got {stop!r}.")
        if self._stop_message(name, cfg, np.asarray(stop, dtype=np.float64)) is None:
            raise ValueError(f"[ActionDispatcher] '{name}': 'stop' does not fit 'mapping' (see the warnings above).")

    def _check_mapping(self, name: str, cfg: dict, msg_class: Any) -> None:
        """Fill one message of ``msg_class`` with dummy values, so a broken ``mapping`` / ``goal_init``
        stops startup. Found at the first tick it would send the other entries (the arm) without this one."""
        values = np.zeros(len(cfg["indices"]), dtype=np.float32)
        if "limits" in cfg:
            values = self._apply_limits(values, cfg["limits"])
        if "transform" in cfg:
            values = self._apply_transform(values, cfg["transform"])
        msg = msg_class()
        if "goal_init" in cfg:
            self._init_goal(msg, cfg["goal_init"], name)
        if not _fill_msg(msg, cfg, values, self._node):
            raise ValueError(f"[ActionDispatcher] '{name}': 'mapping' does not fit {msg_class.__name__} (see the warnings above).")

    def _dispatch_topic(self, name: str, cfg: dict, slice_values: np.ndarray) -> bool:
        msg = self._msg_classes[name]()
        if "goal_init" in cfg:
            self._init_goal(msg, cfg["goal_init"], name)
        compiled = self._compiled_mapping_for(name)
        filled = _fill_msg(msg, cfg, slice_values, self._node, compiled) if compiled is not None else _fill_msg(msg, cfg, slice_values, self._node)
        if not filled:
            logger.error(f"[ActionDispatcher] '{name}' not published: a mapped field could not be set.")
            return False
        if self._dry_run:  # nothing published, so no ack to wait for
            logger.info(f"[DRY RUN] '{name}' -> {msg}")
            return True
        logger.debug(
            f"[ActionDispatcher] Publishing '{name}' -> {cfg['topic']}:\n  {msg}"
        )
        if "feedback" in cfg:
            self._ack_events[name].clear()  # before publishing: an ack that comes back at once must not be wiped
        self._publishers[name].publish(msg)

        if "feedback" in cfg:
            self._wait_for_ack(name, cfg["feedback"], getattr(getattr(msg, "header", None), "stamp", None))
        return True

    def _on_ack(self, msg: Any, name: str) -> None:
        lock = getattr(self, "_ack_lock", None)
        with (lock if lock is not None else nullcontext()):
            self._ack_latest[name] = msg
            self._ack_events[name].set()

    def _wait_for_ack(self, name: str, fb_cfg: dict, sent_stamp: Any) -> None:
        """
        Block until an ack matching `sent_stamp` arrives on the configured
        ack topic, or `fb_cfg['timeout']` seconds elapse.

        Matching on `correlation_field` (default "command_stamp", echoed
        back by the driver) discards a stale ack left over from a previous
        command instead of mistaking it for this one. Set
        `correlation_field: null` in config to disable this check entirely
        (accept whichever ack arrives next) for an ack message with no such
        field.

        Success/failure is decided by `success_field`/`success_value` (see
        _check_success) rather than a hardcoded field/value, so a config can
        match e.g. `reached: true` or `skill_status: "REACHED"` — or
        anything else the ack message happens to carry.
        """
        timeout = fb_cfg.get("timeout", 10.0)
        correlation_field = fb_cfg.get("correlation_field", "command_stamp")
        deadline = time.monotonic() + timeout
        ev = self._ack_events[name]
        lock = getattr(self, "_ack_lock", None)

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if ev.wait(timeout=min(0.05, remaining)):
                with (lock if lock is not None else nullcontext()):
                    ev.clear()  # before reading: an ack landing in between sets it again instead of being lost
                    ack = self._ack_latest.get(name)
                if ack is None:
                    continue
                if correlation_field and sent_stamp is not None:  # no header: nothing to correlate on
                    try:
                        stamp = _get_nested(ack, correlation_field)
                        if (stamp.sec, stamp.nanosec) != (sent_stamp.sec, sent_stamp.nanosec):
                            continue  # stale ack from an earlier command
                    except Exception as e:
                        logger.warning(
                            f"[ActionDispatcher] '{name}' correlation_field "
                            f"'{correlation_field}' not found on ack: {e}"
                        )
                if not _check_success(ack, fb_cfg):
                    logger.warning(f"[ActionDispatcher] '{name}' did not report success.")
                return
        self._handle_timeout(name, fb_cfg, timeout)

    def _handle_timeout(self, name: str, fb_cfg: dict, timeout: float) -> None:
        """
        Shared timeout policy for both topic-ack and action feedback/result
        waits. `on_timeout: raise` raises a RuntimeError out of dispatch():
        step() sends the stop and re-raises, run() lets it through and ends.
        The default "warn" logs and lets dispatch() proceed.
        """
        msg = (
            f"[ActionDispatcher] '{name}' timed out waiting for feedback "
            f"after {timeout}s"
        )
        if fb_cfg.get("on_timeout", "warn") == "raise":
            raise RuntimeError(msg)
        logger.warning(msg + "; proceeding anyway.")

    def _dispatch_action(self, name: str, cfg: dict, slice_values: np.ndarray) -> bool:
        if not self._dry_run:
            client = self._action_clients[name]
            if not client.server_is_ready():
                if name not in self._not_ready_warned:  # retried every tick: warn once until it is ready
                    self._not_ready_warned.add(name)
                    logger.warning(f"[ActionDispatcher] Action server '{cfg['topic']}' not ready, skipping.")
                return False
            self._not_ready_warned.discard(name)

        goal = self._goal_classes[name]()

        # 1. Pre-populate static structure (arrays, frame_ids, namespaces)
        if "goal_init" in cfg:
            self._init_goal(goal, cfg["goal_init"], name)

        # 2. Fill model output values into mapped fields
        compiled = self._compiled_mapping_for(name)
        filled = _fill_msg(goal, cfg, slice_values, self._node, compiled) if compiled is not None else _fill_msg(goal, cfg, slice_values, self._node)
        if not filled:
            logger.error(f"[ActionDispatcher] '{name}' goal not sent: a mapped field could not be set.")
            return False
        if self._dry_run:  # no goal sent, so no result or feedback to wait for
            logger.info(f"[DRY RUN] '{name}' -> {goal}")
            self._track_goal(name, None)
            return True

        logger.debug(
            f"[ActionDispatcher] Sending goal for '{name}':\n{goal}"
        )

        fb_cfg = cfg.get("feedback")
        if fb_cfg is None:
            future = client.send_goal_async(goal)
            with self._goals_lock:
                key = self._last_sent.get(name)  # this goal's: a late rejection must not forget a newer one
            future.add_done_callback(
                lambda f, n=name, k=key: self._goal_response_callback(f, n, k)
            )
            return True

        self._dispatch_action_and_wait(name, client, goal, fb_cfg)
        return True

    def _dispatch_action_and_wait(
        self, name: str, client: ActionClient, goal: Any, fb_cfg: dict
    ) -> None:
        """
        Block dispatch() until this goal's completion signal arrives (or
        `fb_cfg['timeout']` elapses), per the opt-in `feedback:` block on an
        `action: true` entry.

        fb_cfg['source']:
          "result"   (default) — wait for the action's terminal Result via
                     get_result_async(), the correct completion signal for a
                     well-behaved action server (succeeded/aborted/canceled).
          "feedback" — instead watch the continuous Feedback stream and
                     succeed as soon as one Feedback message matches
                     success_field/success_value (for a server that only
                     ever signals completion through feedback ticks).

        Uses a locally-scoped Event/holder per call — unlike the persistent
        topic-ack subscription, each send_goal_async() here gets its own
        dedicated future/goal-handle chain, so there's no stale-data/
        correlation problem across calls to solve.
        """
        source = fb_cfg.get("source", "result")
        ev = threading.Event()
        holder: Dict[str, Any] = {}
        with self._goals_lock:
            key = self._last_sent.get(name)  # this goal's: a rejection after a timeout must not forget a newer one

        def _on_feedback(feedback_msg):
            try:
                if source == "feedback" and _check_success(feedback_msg.feedback, fb_cfg):
                    holder["msg"] = feedback_msg.feedback
                    ev.set()
            except Exception as e:
                holder["error"] = e
                ev.set()

        def _on_goal_response(f):
            try:
                goal_handle = f.result()
                if not goal_handle.accepted:
                    logger.warning(f"[ActionDispatcher] Goal rejected by '{name}'")
                    self._forget_sent(name, key)  # not sent after all: retry it next tick
                    holder["rejected"] = True
                    ev.set()
                    return
                self._track_goal(name, goal_handle)
                if source == "result":
                    result_future = goal_handle.get_result_async()

                    def _on_result(rf):
                        try:
                            holder["msg"] = rf.result().result  # GetResult.Response.result
                        except Exception as e:
                            holder["error"] = e
                        finally:
                            ev.set()

                    result_future.add_done_callback(_on_result)
            except Exception as e:
                self._forget_sent(name, key)
                holder["error"] = e
                ev.set()

        future = client.send_goal_async(
            goal, feedback_callback=_on_feedback if source == "feedback" else None,
        )
        future.add_done_callback(_on_goal_response)

        timeout = fb_cfg.get("timeout", 10.0)
        if not ev.wait(timeout=timeout):
            self._handle_timeout(name, fb_cfg, timeout)
            return
        if "error" in holder:
            logger.warning(f"[ActionDispatcher] '{name}' action future failed: {holder['error']}")
            return
        if holder.get("rejected"):
            return
        if source == "result" and not _check_success(holder.get("msg"), fb_cfg):
            logger.warning(f"[ActionDispatcher] '{name}' action did not report success.")

    def _init_goal(self, goal: Any, init_cfg: dict, name: str) -> None:
        """
        Pre-populate goal fields before mapping.
        Handles string fields, static lists, and array pre-allocation.

        Extend ARRAY_TYPES below when you add new action types that
        require array pre-allocation.
        """
        from trajectory_msgs.msg import JointTrajectoryPoint
        from geometry_msgs.msg import TransformStamped

        # Registry: goal field name -> ROS message type to instantiate
        ARRAY_TYPES = {
            "move_point":       TransformStamped,
            "points":           JointTrajectoryPoint,
            "joint_trajectory": JointTrajectoryPoint,
            "transforms":       TransformStamped,   # geometry_msgs/TransformStamped[] (ArmCommand.transforms)
        }

        for field_path, value in init_cfg.items():
            try:
                parts = field_path.split(".")
                first_name, first_indices = _parse_part(parts[0])

                # "move_point: 1" — pre-allocate array at top level.
                # type(value) is int (not isinstance) so that bool config
                # values (e.g. "use_delta: true") fall through to the plain
                # scalar-set branch below instead of being misread as an
                # array pre-allocation count — bool is an int subclass in
                # Python, so isinstance(True, int) is True.
                if len(parts) == 1 and not first_indices and type(value) is int:
                    if first_name not in ARRAY_TYPES:
                        raise ValueError(
                            f"Unknown array field '{first_name}'. "
                            f"Add it to ARRAY_TYPES in _init_goal."
                        )
                    array = getattr(goal, first_name)
                    for _ in range(value):
                        array.append(ARRAY_TYPES[first_name]())
                    continue

                # "trajectory.points: 1" — pre-allocate a nested array (control_msgs/FollowJointTrajectory)
                if len(parts) > 1 and type(value) is int:
                    parent, last, last_indices = _walk(goal, field_path)
                    if not last_indices and last in ARRAY_TYPES and isinstance(getattr(parent, last), list):
                        for _ in range(value):
                            getattr(parent, last).append(ARRAY_TYPES[last]())
                        continue

                _assign(goal, field_path, lambda _current, v=value: v)  # all other cases: set the value at the path

            except Exception as e:
                # Raise, never skip: a message missing a goal_init field (e.g. use_delta) moves the robot wrongly.
                raise ValueError(f"[ActionDispatcher] goal_init failed for '{name}.{field_path}': {e}") from e

    @staticmethod
    def _apply_limits(slice_values: np.ndarray, limits: list) -> np.ndarray:
        """Clamp each value to its [lower_lim, upper_lim] (real units).

        limits: list of {lower_lim, upper_lim} dicts, one per index, or one entry broadcast to all.
        A null entry leaves that index unchanged.
        """
        result = slice_values.astype(np.float32)
        for i in range(len(result)):
            lim = limits[min(i, len(limits) - 1)]
            if lim is not None:
                result[i] = np.clip(result[i], float(lim["lower_lim"]), float(lim["upper_lim"]))
        return result

    def _apply_transform(self, slice_values: np.ndarray, transform_cfg: dict):
        values = slice_values.copy()
        t = transform_cfg["type"]

        if t == "threshold":
            on = values > transform_cfg.get("threshold", 0.0)
            return np.where(on, transform_cfg.get("on_value", 1.0), transform_cfg.get("off_value", 0.0)).astype(np.float32)

        elif t == "euler_to_quat":
            start = int(transform_cfg.get("euler_start", 3))
            roll, pitch, yaw = float(values[start]), float(values[start + 1]), float(values[start + 2])
            quat = _euler_to_quat(roll, pitch, yaw)
            return np.concatenate([values[:start], quat, values[start + 3:]])

        else:
            raise ValueError(f"[ActionDispatcher] Unknown transform type: '{t}'")

    def _goal_response_callback(self, future: Any, name: str, key: Optional[str] = None) -> None:
        try:
            goal_handle = future.result()
        except Exception as e:
            logger.warning(f"[ActionDispatcher] Goal response failed for '{name}': {e}")
            self._forget_sent(name, key)
            return
        if not goal_handle.accepted:
            logger.warning(
                f"[ActionDispatcher] Goal rejected by '{name}'"
            )
            self._forget_sent(name, key)  # not sent after all: retry it next tick
            return
        self._track_goal(name, goal_handle)

    def _forget_sent(self, name: str, key: Optional[str]) -> None:
        """Let a rejected goal be sent again, unless a newer goal (another key) was sent meanwhile."""
        with self._goals_lock:
            if key is None or self._last_sent.get(name) == key:
                self._last_sent.pop(name, None)
