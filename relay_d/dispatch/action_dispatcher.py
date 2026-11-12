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
"""

import re
import threading
import time

import numpy as np

from action_msgs.msg import GoalStatus
from rclpy.node import Node
from rclpy.action import ActionClient
from typing import Dict, Any

from rosidl_runtime_py.utilities import get_message
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


def _fill_msg(msg: Any, cfg: dict, slice_values: np.ndarray, node: Node) -> bool:
    """Fill header (if present) and mapped fields into a ROS message.

    Returns False if any mapped field could not be set: the message would carry default
    values (zero velocity, zero quaternion) and must not be sent to the robot.
    """
    if hasattr(msg, "header"):  # keep the header: goal_init may have set its frame_id
        msg.header.stamp = node.get_clock().now().to_msg()
        if "frame_id" in cfg:
            msg.header.frame_id = cfg["frame_id"]

    ok = True
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
        self._last_sent: Dict[str, str] = {}
        self._stopped = True  # nothing commanded yet, so there is nothing to stop (see send_stop)
        self._goals = []  # (name, accepted goal handle; None in a dry run) since the last stop: send_stop cancels them
        self._goals_lock = threading.Lock()  # goals are accepted on the executor thread

        # feedback: -- opt-in ack-wait for topic entries (see _wait_for_ack)
        self._ack_latest: Dict[str, Any] = {}
        self._ack_events: Dict[str, threading.Event] = {}

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
            if cfg.get("action", False):
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

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def dispatch(self, action: np.ndarray) -> None:
        if not np.isfinite(action).all():
            logger.error(f"[ActionDispatcher] Non-finite action, not dispatching: {action}")
            self.send_stop()
            return

        # Phase 1: compute every output. If one config entry is broken (bad slice, unknown transform)
        # this raises before anything was sent, instead of moving the arm and skipping the gripper.
        outputs = []
        for name, cfg in self._action_configs.items():
            slice_values = action[cfg["indices"]]

            # 1. Safety clamp to [lower_lim, upper_lim], in real units
            if "limits" in cfg:
                slice_values = self._apply_limits(slice_values, cfg["limits"])

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
        self._stopped = False
        for name, cfg, transformed in outputs:
            # ROS2 action client path
            if cfg.get("action", False):
                key = str(transformed)
                if self._last_sent.get(name) == key:
                    continue
                # Remember it before sending: a rejection (on the executor thread) may arrive before
                # _dispatch_action returns, and its pop must win so the goal is retried.
                self._last_sent[name] = key
                if not self._dispatch_action(name, cfg, transformed):
                    self._last_sent.pop(name, None)

            # Publisher path: sent every tick (controllers with a watchdog need a stream).
            else:
                self._dispatch_topic(name, cfg, transformed)

    def send_stop(self) -> None:
        """Send each entry's ``stop:`` values once, e.g. a zero velocity: called when inference stops sending (inputs
        stale, model error, Ctrl+C, close), because a controller without a watchdog keeps the last command.

        Nothing happens again until the next dispatch(). Never raises: every entry is tried, failures are logged.
        """
        if self._stopped:
            return
        with self._goals_lock:
            self._stopped = True
            goals, self._goals = self._goals, []
        for name, goal_handle in goals:
            self._cancel_goal(name, goal_handle)
        for name, cfg in self._action_configs.items():
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

    # -------------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------------

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

    def _stop_message(self, name: str, cfg: dict, values: np.ndarray) -> Any:
        """The topic message carrying ``stop:`` values (no limits, no transform), or None if the mapping does not fit."""
        msg = self._msg_classes[name]()
        if "goal_init" in cfg:
            self._init_goal(msg, cfg["goal_init"], name)
        return msg if _fill_msg(msg, cfg, values, self._node) else None

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
        if not _fill_msg(msg, cfg, slice_values, self._node):
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

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if ev.wait(timeout=min(0.05, remaining)):
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
                logger.warning(
                    f"[ActionDispatcher] Action server '{cfg['topic']}' not ready, skipping."
                )
                return False

        goal = self._goal_classes[name]()

        # 1. Pre-populate static structure (arrays, frame_ids, namespaces)
        if "goal_init" in cfg:
            self._init_goal(goal, cfg["goal_init"], name)

        # 2. Fill model output values into mapped fields
        if not _fill_msg(goal, cfg, slice_values, self._node):
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
            future.add_done_callback(
                lambda f, n=name: self._goal_response_callback(f, n)
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

        def _on_feedback(feedback_msg):
            if source == "feedback" and _check_success(feedback_msg.feedback, fb_cfg):
                holder["msg"] = feedback_msg.feedback
                ev.set()

        def _on_goal_response(f):
            goal_handle = f.result()
            if not goal_handle.accepted:
                logger.warning(f"[ActionDispatcher] Goal rejected by '{name}'")
                self._last_sent.pop(name, None)  # not sent after all: retry it next tick
                holder["rejected"] = True
                ev.set()
                return
            self._track_goal(name, goal_handle)
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

    def _goal_response_callback(self, future: Any, name: str) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.warning(
                f"[ActionDispatcher] Goal rejected by '{name}'"
            )
            self._last_sent.pop(name, None)  # not sent after all: retry it next tick
            return
        self._track_goal(name, goal_handle)
