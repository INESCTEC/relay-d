"""
api.py
------
Programmatic entry point into the LfD inference/dispatch system, for use
outside the `lfd-inference-node` CLI (notebooks, scripts, REPL, other nodes).

InferenceSession wires together the same three pieces inference_node.py's
CLI does — ObsBuilder, a policy runner (ModelRunner or RobomimicPolicyRunner),
and ActionDispatcher — and exposes them as plain method calls:

    from relay_d.dispatch import InferenceSession

    with InferenceSession(
        checkpoint="model.pth",
        obs_config="config/obs_config.yaml",
        action_config="config/action_config.yaml",
        device="cpu",
    ) as session:
        ok, report = session.validate()
        session.reset()
        action = session.step()          # one tick: obs -> predict -> dispatch

        # or run the full loop, same as the CLI:
        session.run(hz=20)

Lower-level building blocks (ModelRunner, ObsBuilder, ActionDispatcher,
load_policy, detect_model_type) are re-exported below for direct/advanced use.
"""

from __future__ import annotations

import time
import numpy as np

from collections import deque
from importlib import resources
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .model_runner import ModelRunner, detect_model_type, load_policy
from relay_d.utils.coloring_logger import logger

# ObsBuilder/ActionDispatcher/rclpy require a sourced ROS2 environment. Guard
# the import so that pure-Python pieces (ModelRunner, load_policy, ...) stay
# importable in environments without ROS2 (e.g. offline debug/analysis venvs).
try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .action_dispatcher import ActionDispatcher
    from .obs_builder import ObsBuilder

    _ROS_IMPORT_ERROR = None
except ImportError as _e:
    rclpy = None
    MultiThreadedExecutor = None
    ActionDispatcher = None
    ObsBuilder = None
    _ROS_IMPORT_ERROR = _e

__all__ = [
    "InferenceSession",
    "ModelRunner",
    "ObsBuilder",
    "ActionDispatcher",
    "DispatcherHelpers",
    "detect_model_type",
    "load_policy",
]

def _wait_for_obs(obs_node: ObsBuilder, obs_keys: List[str], timeout: float, poll: float = 0.1) -> bool:
    """Block until all obs_keys appear in get_obs(), or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(k in obs_node.get_obs() for k in obs_keys):
            return True
        time.sleep(poll)
    return False

class DispatcherHelpers:

    def __init__(self):
        logger.info("Initiated DispatcherHelper.")

    def save_action_template(self, output_dir) -> str:
        resource_target = resources.files("relay_d.dispatch").joinpath("config_templates/action_config_template.yaml")
        destination_path = Path(output_dir) / "action_config.yaml"

        # Read the bytes directly from the resource and write to the output directory
        destination_path.write_bytes(resource_target.read_bytes())

        return destination_path

    def save_inference_script(self, output_dir):
        """Copy the bundled inference script to the output directory."""
        resource_target = resources.files("relay_d.dispatch").joinpath("templates/inference_script_template.py")
        destination_path = Path(output_dir) / "inference_script.py"

        # Read the bytes directly from the resource and write to the output directory
        destination_path.write_bytes(resource_target.read_bytes())

        return destination_path

class InferenceSession:
    """Owns one ObsBuilder + policy + ActionDispatcher for a single rclpy process."""

    def __init__(
        self,
        checkpoint: str,
        obs_config: str,
        action_config: str,
        device: str = "cpu",
        backend: str = "custom",
        model_type: str = "auto",
        obs_keys: Optional[List[str]] = None,
        action_horizon: Optional[int] = None,
    ):
        if _ROS_IMPORT_ERROR is not None:
            raise ImportError(
                "InferenceSession requires a sourced ROS2 environment "
                "(rclpy, tf2_ros, rosidl_runtime_py, ...) which is not "
                "available in this Python environment."
            ) from _ROS_IMPORT_ERROR

        if not rclpy.ok():
            rclpy.init()

        self.obs_node = ObsBuilder(obs_config)

        if backend == "robomimic":
            from .robomimic_runner import RobomimicPolicyRunner
            self.model = RobomimicPolicyRunner(checkpoint, device=device)
        else:
            self.model = ModelRunner(
                checkpoint,
                device=device,
                obs_keys=obs_keys,
                model_type=model_type,
                action_horizon=action_horizon,
            )

        self.dispatcher = ActionDispatcher(self.obs_node, action_config)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.obs_node)
        self._spin_thread = None

        self.obs_node.get_logger().info("=== InferenceSession ready. ===")

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def start(self) -> None:
        """Spin the ObsBuilder node in a background thread so subscribers stay alive."""
        if self._spin_thread is None:
            import threading
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

    def close(self) -> None:
        self._executor.shutdown()
        self.obs_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> "InferenceSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------
    # Pre-flight
    # -------------------------------------------------------------------

    def validate(self, timeout: float = 10.0) -> Tuple[bool, dict]:
        """Check every configured input source is reachable, then wait for all
        required obs keys to populate with real data. Returns (ok, report)."""
        all_inputs_ok, input_report = self.obs_node.validate_inputs(timeout=timeout)
        if not all_inputs_ok:
            return False, {"inputs": input_report, "obs_keys_ready": False}

        required_keys = self.model.obs_keys
        ready = _wait_for_obs(self.obs_node, required_keys, timeout=timeout)
        return ready, {"inputs": input_report, "obs_keys_ready": ready}

    # -------------------------------------------------------------------
    # Single-tick interaction
    # -------------------------------------------------------------------

    def get_observation(self) -> Dict[str, np.ndarray]:
        return self.obs_node.get_obs()

    def predict(self, obs: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        if obs is None:
            obs = self.get_observation()
        return self.model.get_action(obs)

    def dispatch(self, action: np.ndarray) -> None:
        self.dispatcher.dispatch(action)

    def step(self) -> np.ndarray:
        """One full tick: get latest obs -> predict -> dispatch. Returns the action."""
        obs = self.get_observation()
        action = self.predict(obs)
        self.dispatch(action)
        return action

    def reset(self) -> None:
        self.model.reset()

    # -------------------------------------------------------------------
    # Full loop (same behavior as the inference_node.py CLI)
    # -------------------------------------------------------------------

    def run(self, hz: float = 0.0, max_steps: Optional[int] = None, validate_timeout: float = 10.0) -> None:
        self.start()

        self.obs_node.get_logger().info(
            f"[Pre-flight] Validating input sources (timeout={validate_timeout}s) ..."
        )
        ok, report = self.validate(timeout=validate_timeout)

        for inp_name, result in report["inputs"].items():
            kind = result["kind"]
            if result["ok"]:
                self.obs_node.get_logger().info(f"  [OK]   {inp_name} ({kind}): {result['detail']}")
            else:
                self.obs_node.get_logger().error(f"  [FAIL] {inp_name} ({kind}): {result['error']}")
                if "hint" in result:
                    self.obs_node.get_logger().error(f"         {result['hint']}")

        if not report["obs_keys_ready"] and all(r["ok"] for r in report["inputs"].values()):
            self.obs_node.get_logger().error(
                f"[Pre-flight] Timed out after {validate_timeout}s — not all observations are ready. "
                "Check obs_config.yaml field paths."
            )
        if not ok:
            raise RuntimeError("[Pre-flight] Validation failed. Fix obs_config.yaml and restart.")

        self.obs_node.get_logger().info("[Pre-flight] All inputs and observations validated. Starting inference.")

        # Start a fresh episode right before the rollout begins.
        self.reset()

        min_dt = (1.0 / hz) if hz > 0 else 0.0
        step_count = 0

        # Live expected-vs-actual tracking: the model's action is a per-tick
        # robot0_joint_pos delta, so "expected next joint pos" computed on tick N
        # is checked against the freshly-read robot0_joint_pos on tick N+1.
        pending_expected_joint_pos = None
        joint_tracking_errors: deque = deque(maxlen=20)

        try:
            while rclpy.ok():
                t0 = time.monotonic()

                obs = self.get_observation()
                logger.debug(
                    "[InferenceSession] obs: "
                    + ", ".join(f"{k}={np.round(obs[k], 4)}" for k in self.model.obs_keys if k in obs)
                )

                if pending_expected_joint_pos is not None and "robot0_joint_pos" in obs:
                    actual_joint_pos = obs["robot0_joint_pos"]
                    tracking_error = actual_joint_pos - pending_expected_joint_pos
                    joint_tracking_errors.append(np.abs(tracking_error))
                    logger.debug(
                        "[InferenceSession] joint tracking error (actual - expected): "
                        f"{tracking_error.round(5)}"
                    )

                try:
                    action = self.predict(obs)
                except Exception as e:
                    self.obs_node.get_logger().error(f"Model inference failed ({type(e).__name__}): {e}")
                    break

                logger.debug(f"[InferenceSession] raw action (full {action.shape[0]}-dim): {action.round(5)}")

                if "robot0_joint_pos" in obs and action.shape[0] >= 7:
                    pending_expected_joint_pos = obs["robot0_joint_pos"] + action[:7]

                self.dispatch(action)

                step_count += 1
                if step_count % 100 == 0:
                    elapsed = time.monotonic() - t0
                    self.obs_node.get_logger().info(
                        f"Step {step_count} | action={action.round(4)} | "
                        f"inference_time={elapsed*1000:.1f}ms"
                    )
                if step_count % 20 == 0 and joint_tracking_errors:
                    mean_abs_err = np.mean(joint_tracking_errors, axis=0)
                    self.obs_node.get_logger().info(
                        f"[InferenceSession] joint tracking MAE (last {len(joint_tracking_errors)} ticks): "
                        f"{mean_abs_err.round(5)}"
                    )

                if max_steps is not None and step_count >= max_steps:
                    break

                if min_dt > 0:
                    elapsed = time.monotonic() - t0
                    sleep_time = min_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.obs_node.get_logger().info("Shutting down inference session.")
