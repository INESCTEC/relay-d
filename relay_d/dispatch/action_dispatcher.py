"""
action_dispatcher.py
--------------------
Routes slices of the model's action array to ROS2 topics or action servers,
driven entirely by action_config.yaml.

  action: false (or omitted) -> node.create_publisher  -> publish()
  action: true               -> ActionClient           -> send_goal_async()
"""

import re
import threading
import time

import numpy as np
import rclpy
import subprocess
import shlex

from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Header
from typing import Dict, Any

from rosidl_runtime_py.utilities import get_message
from relay_d.utils.coloring_logger import logger
from relay_d.utils.config_loader import load_yaml_config

def get_action(action_type_str: str):
    """
    Dynamically load a ROS2 action class from a string like
    'move_arm_skill_msgs/action/MoveArmSkill'
    """
    parts = action_type_str.split("/")
    if len(parts) != 3 or parts[1] != "action":
        raise ValueError(
            f"Invalid action_type format: '{action_type_str}'. "
            f"Expected '<pkg>/action/<Name>'"
        )
    pkg, _, name = parts
    module = __import__(f"{pkg}.action", fromlist=[name])
    return getattr(module, name)

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


def _set_nested(obj: Any, path: str, value: float) -> None:
    """
    Set a value at a dot-separated path, supporting array indexing.

    Examples:
        "data"                               -> obj.data = value
        "pose.position.x"                    -> obj.pose.position.x = value
        "move_point[0].transform.translation.x" -> obj.move_point[0].transform.translation.x = value
    """
    parts = path.split(".")

    for part in parts[:-1]:
        name, indices = _parse_part(part)
        obj = getattr(obj, name)
        for idx in indices:
            obj = obj[idx]

    last_name, last_indices = _parse_part(parts[-1])
    if last_indices:
        target = getattr(obj, last_name)
        for idx in last_indices[:-1]:
            target = target[idx]
        target[last_indices[-1]] = float(value)
    else:
        setattr(obj, last_name, float(value))


def _set_nested_array(obj: Any, path: str, values) -> None:
    """
    Like _set_nested, but assigns an entire list to the final field instead
    of a single scalar — for float64[]-style array fields (e.g. a
    JointTrajectoryPoint's `positions`) that must be set in one shot rather
    than index-by-index.
    """
    parts = path.split(".")

    for part in parts[:-1]:
        name, indices = _parse_part(part)
        obj = getattr(obj, name)
        for idx in indices:
            obj = obj[idx]

    last_name, last_indices = _parse_part(parts[-1])
    values = [float(v) for v in values]
    if last_indices:
        target = getattr(obj, last_name)
        for idx in last_indices[:-1]:
            target = target[idx]
        target[last_indices[-1]] = values
    else:
        setattr(obj, last_name, values)


def _get_nested(obj: Any, path: str) -> Any:
    """
    Read a value at a dot-separated path, supporting array indexing.
    Mirrors _set_nested, but reads instead of writes.
    """
    for part in path.split("."):
        name, indices = _parse_part(part)
        obj = getattr(obj, name)
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
        logger.warn(
            f"[ActionDispatcher] feedback success_field '{field}' not found "
            f"on {type(msg).__name__}: {e}"
        )
        return False


def _fill_msg(msg: Any, cfg: dict, slice_values: np.ndarray, node: Node) -> None:
    """Fill header (if present) and mapped fields into a ROS message."""
    if hasattr(msg, "header"):
        msg.header = Header()
        msg.header.stamp = node.get_clock().now().to_msg()
        if "frame_id" in cfg:
            msg.header.frame_id = cfg["frame_id"]

    for field_path, slice_idx in cfg["mapping"].items():
        try:
            if isinstance(slice_idx, list):
                _set_nested_array(msg, field_path, slice_values[slice_idx])
            else:
                _set_nested(msg, field_path, slice_values[slice_idx])
        except Exception as e:
            logger.warn(
                f"[ActionDispatcher] Failed to set {field_path}: {e}"
            )


# ---------------------------------------------------------------------------
# ActionDispatcher
# ---------------------------------------------------------------------------

class ActionDispatcher:

    def __init__(self, node: Node, action_config_path: str):
        self._node = node

        config = load_yaml_config(action_config_path)

        self._action_configs = config["actions"]
        self._publishers: Dict[str, Any] = {}
        self._action_clients: Dict[str, ActionClient] = {}
        self._goal_classes: Dict[str, Any] = {}
        self._msg_classes: Dict[str, Any] = {}
        self._last_sent: Dict[str, str] = {}

        # feedback: -- opt-in ack-wait for topic entries (see _wait_for_ack)
        self._ack_latest: Dict[str, Any] = {}
        self._ack_events: Dict[str, threading.Event] = {}

        # __init__: change if -> elif
        for name, cfg in self._action_configs.items():
            is_action = cfg.get("action", False)
            is_command = "command" in cfg

            if is_command:
                logger.info(
                    f"[ActionDispatcher] '{name}' -> COMMAND '{cfg['command'][:50]}...'"
                )

            elif is_action:                          # <-- elif, not if
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
                self._goal_classes[name] = action_class.Goal
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
                self._msg_classes[name] = msg_class
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

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def dispatch(self, action: np.ndarray) -> None:
        for name, cfg in self._action_configs.items():

            slice_values = action[cfg["indices"]]

            # 1. Per-index denormalization: model output range -> real-world range
            if "limits" in cfg and cfg.get("denormalize", cfg.get("apply_limits", True)):
                slice_values = self._apply_limits(
                    slice_values,
                    cfg["limits"],
                    cfg.get("normalized_range", [-1.0, 1.0]),
                )

            # 2. Optional further transform (vacuum, magnetic, scale, etc.)
            if "transform" in cfg:
                transformed = self._apply_transform(slice_values, cfg["transform"])
            else:
                transformed = slice_values

            # Command template path — simplest, just shell out
            if "command" in cfg:
                cmd = self._render_command(cfg["command"], transformed)
                if self._last_sent.get(name) == cmd:
                    continue
                self._last_sent[name] = cmd
                self._dispatch_command(name, cfg, cmd)

            # ROS2 action client path
            elif cfg.get("action", False):
                key = str(transformed)
                if self._last_sent.get(name) == key:
                    continue
                self._last_sent[name] = key
                self._dispatch_action(name, cfg, transformed)

            # Publisher path
            else:
                key = str(transformed)
                if self._last_sent.get(name) == key:
                    continue
                self._last_sent[name] = key
                self._dispatch_topic(name, cfg, transformed)

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------

    def _dispatch_command(
        self, name: str, cfg: dict, cmd: str
    ) -> None:
        logger.debug(
            f"[ActionDispatcher] Running command for '{name}':\n  {cmd}"
        )
        try:
            subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warn(
                f"[ActionDispatcher] Command failed for '{name}': {e}\n  cmd: {cmd}"
            )

    def _dispatch_topic(self, name: str, cfg: dict, slice_values: np.ndarray) -> None:
        msg = self._msg_classes[name]()
        if "goal_init" in cfg:
            self._init_goal(msg, cfg["goal_init"], name)
        _fill_msg(msg, cfg, slice_values, self._node)
        logger.debug(
            f"[ActionDispatcher] Publishing '{name}' -> {cfg['topic']}:\n  {msg}"
        )
        self._publishers[name].publish(msg)

        if "feedback" in cfg:
            self._wait_for_ack(name, cfg["feedback"], msg.header.stamp)

    def _on_ack(self, msg: Any, name: str) -> None:
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
        ev.clear()

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if ev.wait(timeout=min(0.05, remaining)):
                ack = self._ack_latest.get(name)
                ev.clear()
                if ack is None:
                    continue
                if correlation_field:
                    try:
                        stamp = _get_nested(ack, correlation_field)
                        if (stamp.sec, stamp.nanosec) != (sent_stamp.sec, sent_stamp.nanosec):
                            continue  # stale ack from an earlier command
                    except Exception as e:
                        logger.warn(
                            f"[ActionDispatcher] '{name}' correlation_field "
                            f"'{correlation_field}' not found on ack: {e}"
                        )
                if not _check_success(ack, fb_cfg):
                    logger.warn(f"[ActionDispatcher] '{name}' did not report success.")
                return
        self._handle_timeout(name, fb_cfg, timeout)

    def _handle_timeout(self, name: str, fb_cfg: dict, timeout: float) -> None:
        """
        Shared timeout policy for both topic-ack and action feedback/result
        waits. `on_timeout: raise` propagates a RuntimeError out of
        dispatch() (uncaught by step()/run() today) for a hard-stop-on-
        failure setup; the default "warn" logs and lets dispatch() proceed.
        """
        msg = (
            f"[ActionDispatcher] '{name}' timed out waiting for feedback "
            f"after {timeout}s"
        )
        if fb_cfg.get("on_timeout", "warn") == "raise":
            raise RuntimeError(msg)
        logger.warn(msg + "; proceeding anyway.")

    def _dispatch_action(self, name: str, cfg: dict, slice_values: np.ndarray) -> None:
        client = self._action_clients[name]

        if not client.server_is_ready():
            logger.warn(
                f"[ActionDispatcher] Action server '{cfg['topic']}' not ready, skipping."
            )
            return

        goal = self._goal_classes[name]()

        # 1. Pre-populate static structure (arrays, frame_ids, namespaces)
        if "goal_init" in cfg:
            self._init_goal(goal, cfg["goal_init"], name)

        # 2. Fill model output values into mapped fields
        _fill_msg(goal, cfg, slice_values, self._node)

        logger.debug(
            f"[ActionDispatcher] Sending goal for '{name}':\n{goal}"
        )

        fb_cfg = cfg.get("feedback")
        if fb_cfg is None:
            future = client.send_goal_async(goal)
            future.add_done_callback(
                lambda f, n=name: self._goal_response_callback(f, n)
            )
            return

        self._dispatch_action_and_wait(name, client, goal, fb_cfg)

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
        if "success_field" not in fb_cfg:
            raise ValueError(
                f"[ActionDispatcher] '{name}' has an action 'feedback' "
                f"block but is missing required 'success_field' (there's "
                f"no universal default — Result/Feedback shapes vary per "
                f"action type)."
            )

        source = fb_cfg.get("source", "result")
        ev = threading.Event()
        holder: Dict[str, Any] = {}

        def _on_feedback(feedback_msg):
            if source == "feedback" and _check_success(feedback_msg.feedback, fb_cfg):
                holder["msg"] = feedback_msg.feedback
                ev.set()

        def _on_goal_response(f):
            goal_handle = f.result()
            if not goal_handle.accepted:
                logger.warn(f"[ActionDispatcher] Goal rejected by '{name}'")
                holder["rejected"] = True
                ev.set()
                return
            if source == "result":
                result_future = goal_handle.get_result_async()

                def _on_result(rf):
                    holder["msg"] = rf.result().result  # GetResult.Response.result
                    ev.set()

                result_future.add_done_callback(_on_result)

        future = client.send_goal_async(
            goal, feedback_callback=_on_feedback if source == "feedback" else None,
        )
        future.add_done_callback(_on_goal_response)

        timeout = fb_cfg.get("timeout", 10.0)
        if not ev.wait(timeout=timeout):
            self._handle_timeout(name, fb_cfg, timeout)
            return
        if holder.get("rejected"):
            return
        if source == "result" and not _check_success(holder.get("msg"), fb_cfg):
            logger.warn(f"[ActionDispatcher] '{name}' action did not report success.")

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

                # All other cases: navigate dot-path and set value
                obj = goal
                for part in parts[:-1]:
                    part_name, indices = _parse_part(part)
                    obj = getattr(obj, part_name)
                    for idx in indices:
                        obj = obj[idx]

                last_name, last_indices = _parse_part(parts[-1])
                if last_indices:
                    target = getattr(obj, last_name)
                    for idx in last_indices[:-1]:
                        target = target[idx]
                    target[last_indices[-1]] = value
                else:
                    setattr(obj, last_name, value)

            except Exception as e:
                logger.warn(
                    f"[ActionDispatcher] goal_init failed for "
                    f"'{name}.{field_path}': {e}"
                )

    def _render_command(self, template: str, values) -> str:
        """
        Replace {I[0]}, {I[1]}, ... in a command template with actual values.
        values can be np.ndarray or dict.
        """
        result = template
        if isinstance(values, dict):
            for idx, val in values.items():
                # Format ints as ints, floats as floats
                formatted = str(int(val)) if float(val).is_integer() else str(round(float(val), 6))
                result = result.replace(f"{{I[{idx}]}}", formatted)
        else:
            for idx, val in enumerate(values):
                formatted = str(int(val)) if float(val).is_integer() else str(round(float(val), 6))
                result = result.replace(f"{{I[{idx}]}}", formatted)

        if re.search(r'\{I\[\d+\]\}', result):
            logger.warn(
                f"[ActionDispatcher] Command template has unreplaced placeholders "
                f"(action slice has {len(values)} values): {result!r}"
            )
        return result

    def _apply_limits(
        self,
        slice_values: np.ndarray,
        limits: list,
        normalized_range: list,
    ) -> np.ndarray:
        """
        Per-index linear denormalization.

        Maps each element of slice_values from normalized_range to the
        [lower_lim, upper_lim] defined in limits.

        Broadcast: a single limits entry is applied to all indices.
        Use null for a specific entry to leave that index unchanged.

        normalized_range: [model_min, model_max]  (typically [-1.0, 1.0])
        limits: list of {lower_lim, upper_lim} dicts — one per index OR one entry → broadcast.
        """
        model_min, model_max = float(normalized_range[0]), float(normalized_range[1])
        span = model_max - model_min
        result = slice_values.copy().astype(np.float32)

        if span == 0.0:
            logger.warn(
                f"[ActionDispatcher] normalized_range has zero span "
                f"({model_min} == {model_max}); skipping denormalization."
            )
            return result

        for i in range(len(result)):
            lim = limits[min(i, len(limits) - 1)]
            if lim is None:
                continue
            lower = float(lim["lower_lim"])
            upper = float(lim["upper_lim"])
            norm = np.clip((result[i] - model_min) / span, 0.0, 1.0)
            result[i] = lower + norm * (upper - lower)

        return result

    def _apply_transform(self, slice_values: np.ndarray, transform_cfg: dict):
        values = slice_values.copy()
        t = transform_cfg["type"]

        if t == "scale":
            in_min  = transform_cfg["in_min"]
            in_max  = transform_cfg["in_max"]
            out_min = transform_cfg["out_min"]
            out_max = transform_cfg["out_max"]
            values  = (values - in_min) / (in_max - in_min)
            values  = values * (out_max - out_min) + out_min
            values  = np.clip(values, out_min, out_max)
            return values

        elif t == "threshold":
            return (values > transform_cfg.get("threshold", 0.0)).astype(np.float32)

        elif t == "clip":
            return np.clip(values, transform_cfg["out_min"], transform_cfg["out_max"])

        elif t == "vacuum":
            activate      = float(values[0]) > transform_cfg.get("threshold", 0.0)
            max_suction   = transform_cfg.get("max_suction", 255)
            power_limit   = transform_cfg.get("power_limit", 400)
            channel_value = max_suction if activate else 0
            return {0: float(channel_value), 1: float(channel_value), 2: float(power_limit)}

        elif t == "magnetic":
            activate = float(values[0]) > transform_cfg.get("threshold", 0.0)
            mode     = 1 if activate else 0
            strength = 0
            if activate and "strength_scale" in transform_cfg:
                sc       = transform_cfg["strength_scale"]
                norm     = (float(values[0]) - sc["in_min"]) / (sc["in_max"] - sc["in_min"])
                strength = int(np.clip(norm * (sc["out_max"] - sc["out_min"]) + sc["out_min"],
                                    sc["out_min"], sc["out_max"]))
            return {0: float(mode), 1: float(strength)}

        elif t == "euler_to_quat":
            start = int(transform_cfg.get("euler_start", 3))
            roll, pitch, yaw = float(values[start]), float(values[start + 1]), float(values[start + 2])
            quat = _euler_to_quat(roll, pitch, yaw)
            return np.concatenate([values[:start], quat, values[start + 3:]])

        else:
            raise ValueError(f"[ActionDispatcher] Unknown transform type: '{t}'")

    def _goal_response_callback(self, future: Any, name: str) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.warn(
                f"[ActionDispatcher] Goal rejected by '{name}'"
            )