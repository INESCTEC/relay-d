"""
api.py
------
Programmatic entry point into the LfD inference/dispatch system, for use
outside the `lfd-inference-node` CLI (notebooks, scripts, REPL, other nodes).

InferenceSession wires together the same three pieces inference_node.py's
CLI does — ObsBuilder, a policy runner (RobomimicPolicyRunner or
LeRobotPolicyRunner) and ActionDispatcher — and exposes them as plain method calls:

    from relay_d.dispatch import InferenceSession

    with InferenceSession(
        checkpoint="model.pth",              # robomimic .pth, or a LeRobot pretrained_model folder
        obs_config="config/obs_config.yaml",
        action_config="config/action_config.yaml",
        device="cpu",                        # default: the infer_device setting
    ) as session:
        ok, report = session.validate()
        session.reset()
        action = session.step()          # one tick: obs -> predict -> dispatch

        # or run the full loop, same as the CLI (default rate: the policy's, else the infer_hz setting):
        session.run(hz=20)

The backend is chosen by `backend=` ("robomimic" / "lerobot"), by default from the
checkpoint itself ("auto", see detect_backend). ObsBuilder and ActionDispatcher are
re-exported below for direct/advanced use.
"""

from __future__ import annotations

import logging
import time
import numpy as np

from importlib import resources
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .lerobot_runner import resolve_checkpoint
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger

# ObsBuilder/ActionDispatcher/rclpy require a sourced ROS2 environment. Guard
# the import so that pure-Python pieces (detect_backend, save_dispatch_templates) stay
# importable in environments without ROS2.
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from .action_dispatcher import ActionDispatcher
    from .obs_builder import ObsBuilder

    _ROS_IMPORT_ERROR = None
except ImportError as _e:
    rclpy = None
    SingleThreadedExecutor = None
    ActionDispatcher = None
    ObsBuilder = None
    _ROS_IMPORT_ERROR = _e

__all__ = [
    "InferenceSession",
    "ObsBuilder",
    "ActionDispatcher",
    "save_dispatch_templates",
    "detect_backend",
]

BACKENDS = ("robomimic", "lerobot")


def detect_backend(checkpoint) -> str:
    """"lerobot" for a LeRobot ``pretrained_model`` folder (or a folder holding one, like ``checkpoints/last``),
    "robomimic" for a ``.pth`` / ``.pt`` / ``.ckpt`` file."""
    path = Path(checkpoint)
    if resolve_checkpoint(path) is not None:
        return "lerobot"
    if path.is_file() and path.suffix in (".pth", ".pt", ".ckpt"):
        return "robomimic"
    raise ValueError(
        f"cannot tell the backend of '{checkpoint}': expected a robomimic .pth file or a LeRobot "
        "pretrained_model folder (config.json + model.safetensors). Pass --backend explicitly if needed."
    )


def _wait_for_obs(obs_node: ObsBuilder, obs_keys: List[str], timeout: float, poll: float = 0.1) -> bool:
    """Block until all obs_keys appear in get_obs(), or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(k in obs_node.get_obs() for k in obs_keys):
            return True
        time.sleep(poll)
    return False

def save_dispatch_templates(output_dir) -> List[Path]:
    """Copy the action_config.yaml template and the inference script into ``output_dir`` (the Post-process page's
    action_dispatcher folder), so data prep writes the same config format that inference reads."""
    package = resources.files("relay_d.dispatch")
    written = []
    for source, name in (("config_templates/action_config_template.yaml", "action_config.yaml"),
                         ("templates/inference_script_template.py", "inference_script.py")):
        destination = Path(output_dir) / name
        destination.write_bytes(package.joinpath(source).read_bytes())
        written.append(destination)
    return written

class InferenceSession:
    """Owns one ObsBuilder + policy + ActionDispatcher for a single rclpy process."""

    def __init__(
        self,
        checkpoint: str,
        obs_config: str,
        action_config: str,
        device: Optional[str] = None,
        backend: Optional[str] = None,
        state_keys: Optional[List[str]] = None,
        task: Optional[str] = None,
        dry_run: bool = False,
    ):
        """``device``: torch device (default: the infer_device setting). ``backend``: "robomimic", "lerobot" or "auto" (default: the infer_backend setting, "auto" detects it from
        the checkpoint). ``state_keys``: LeRobot only, the observations that form observation.state, in order
        (default: read from the dataset the policy was trained on). ``task``: LeRobot only, the instruction a language-conditioned
        policy (SmolVLA) follows (default: the training dataset's task). ``dry_run``: log every command instead of
        publishing it or sending a goal (the robot does not move)."""
        if _ROS_IMPORT_ERROR is not None:
            raise ImportError(
                "InferenceSession requires a sourced ROS2 environment "
                "(rclpy, tf2_ros, rosidl_runtime_py, ...) which is not "
                "available in this Python environment."
            ) from _ROS_IMPORT_ERROR

        device = device or user_settings.get("infer_device")
        backend = backend or user_settings.get("infer_backend")
        if backend == "auto":
            backend = detect_backend(checkpoint)
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend '{backend}' (choose from {BACKENDS} or 'auto')")
        if not rclpy.ok():
            rclpy.init()

        self.obs_node = ObsBuilder(obs_config)

        if backend == "robomimic":
            from .robomimic_runner import RobomimicPolicyRunner
            self.model = RobomimicPolicyRunner(checkpoint, device=device)
        else:
            from .lerobot_runner import LeRobotPolicyRunner
            self.model = LeRobotPolicyRunner(
                checkpoint, device=device, state_keys=state_keys,
                fallback_state_keys=self.obs_node.low_dim_obs_names, task=task,
            )

        self.dispatcher = ActionDispatcher(self.obs_node, action_config, dry_run=dry_run)
        self._check_action_layout()
        self._was_stale = False  # after stale inputs the policy restarts: no queued actions or frames from before the gap

        # Single-threaded, as the recorder: rclpy's MultiThreadedExecutor could not keep up with 3 cameras at 20 Hz next to
        # the inference loop (inputs went stale, ~6 Hz of commands). The ack waits run on the inference thread, not in callbacks.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.obs_node)
        self._spin_thread = None

        self.obs_node.get_logger().info("=== InferenceSession ready. ===")

    def _check_action_layout(self) -> None:
        """action_config.yaml indices against the model's action size: an index past it stops startup, a column no
        entry sends is warned about (a wrong list would drive the wrong joint silently). Skipped if the runner cannot tell."""
        dim = getattr(self.model, "action_dim", None)
        if not dim:
            return
        used = set()
        for name, cfg in self.dispatcher._action_configs.items():
            bad = [i for i in cfg["indices"] if not -dim <= i < dim]
            if bad:
                raise ValueError(f"action_config.yaml '{name}': indices {bad} are outside the model's action, "
                                 f"which has {dim} values (indices 0..{dim - 1}).")
            used.update(i % dim for i in cfg["indices"])
        unused = sorted(set(range(dim)) - used)
        if unused:
            logger.warning(f"[InferenceSession] action columns {unused} of the model's {dim} are sent by no entry "
                           "in action_config.yaml: check its indices.")

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
        self.dispatcher.send_stop()  # once: nothing is sent if run() already stopped and nothing was dispatched since
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

    def _stale_inputs(self, max_obs_age: float) -> Dict[str, float]:
        """Inputs, and the observations the model uses, that have not refreshed within max_obs_age."""
        ages = self.obs_node.get_input_ages()
        obs_ages = self.obs_node.get_obs_ages()
        ages.update({f"obs:{k}": obs_ages[k] for k in self.model.obs_keys if k in obs_ages})
        return {k: round(v, 2) for k, v in ages.items() if v > max_obs_age}

    def step(self, max_obs_age: Optional[float] = None) -> Optional[np.ndarray]:
        """One full tick: get latest obs -> predict -> dispatch. Returns the action, or None
        (nothing dispatched) if an input is older than max_obs_age, like run()."""
        max_obs_age = user_settings.get("infer_max_obs_age_s") if max_obs_age is None else max_obs_age
        self._model_failed = False
        obs = self.get_observation()
        if logger.isEnabledFor(logging.DEBUG):  # vectors only: never format whole camera frames
            logger.debug(
                "[InferenceSession] obs: "
                + ", ".join(f"{k}={np.round(obs[k], 4)}" for k in self.model.obs_keys if k in obs and obs[k].ndim == 1)
            )
        stale = self._stale_inputs(max_obs_age)
        if stale:
            self.obs_node.get_logger().error(
                f"Stale inputs (> {max_obs_age}s without new data), not dispatching: {stale}",
                throttle_duration_sec=1.0,
            )
            self.dispatcher.send_stop()  # once per stale period
            self._was_stale = True
            return None
        self._reset_after_stale()
        try:
            self._model_failed = True  # until predict returns: run() ends on a model error, lets dispatch errors through
            action = self.predict(obs)
            self._model_failed = False
            self.dispatch(action)
        except BaseException:
            self.dispatcher.send_stop()  # an error must not leave the last command running
            raise
        return action

    def reset(self) -> None:
        self.model.reset()

    def _reset_after_stale(self) -> None:
        if self._was_stale:
            self.reset()
            self._was_stale = False

    # -------------------------------------------------------------------
    # Full loop (same behavior as the inference_node.py CLI)
    # -------------------------------------------------------------------

    def run(self, hz: Optional[float] = None, max_steps: Optional[int] = None,
            validate_timeout: Optional[float] = None, max_obs_age: Optional[float] = None) -> bool:
        """The inference loop until Ctrl+C, max_steps or a model error; always ends with the safe stop.
        ``hz``: control rate, 0 = as fast as possible (default: the rate the policy was trained at, else the infer_hz
        setting). None for the others = the user setting (infer_obs_wait_s, infer_max_obs_age_s).
        Returns False if it ended on a model error, True otherwise."""
        fps = getattr(self.model, "fps", None)
        # Each predicted action is one step of the training data: at another rate the robot moves faster or slower than shown.
        hz = (fps or user_settings.get("infer_hz")) if hz is None else hz
        validate_timeout = user_settings.get("infer_obs_wait_s") if validate_timeout is None else validate_timeout
        max_obs_age = user_settings.get("infer_max_obs_age_s") if max_obs_age is None else max_obs_age
        self.start()

        if fps and hz != fps:
            pace = ("as fast as the loop runs" if hz <= 0 else
                    f"{fps / hz:.2g}x slower" if hz < fps else f"{hz / fps:.2g}x faster")
            logger.warning(
                f"[InferenceSession] The policy was trained at {fps} Hz but runs at {hz or 'unthrottled'} Hz: each action "
                f"is one {1 / fps:.3g} s step of the demos, so the robot moves {pace} than shown. Use hz={fps}."
            )

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

        self.obs_node.get_logger().info("[Pre-flight] All inputs and observations validated. Starting inference "
                                        f"at {f'{hz} Hz' if hz > 0 else 'unthrottled rate'}.")

        # Start a fresh episode right before the rollout begins.
        self.reset()

        min_dt = (1.0 / hz) if hz > 0 else 0.0
        step_count = 0
        model_failed = False

        try:
            while rclpy.ok():
                t0 = time.monotonic()

                try:
                    action = self.step(max_obs_age)
                except Exception as e:
                    if not self._model_failed:
                        raise
                    self.obs_node.get_logger().error(f"Model inference failed ({type(e).__name__}): {e}")
                    model_failed = True
                    break
                if action is None:  # stale inputs
                    time.sleep(0.05)
                    continue

                logger.debug(f"[InferenceSession] raw action (full {action.shape[0]}-dim): {action.round(5)}")

                step_count += 1
                if step_count % 100 == 0:
                    elapsed = time.monotonic() - t0
                    self.obs_node.get_logger().info(
                        f"Step {step_count} | action={action.round(4)} | "
                        f"inference_time={elapsed*1000:.1f}ms"
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
        finally:  # every way out of the loop (model error, on_timeout: raise, Ctrl+C, max_steps) stops the robot
            self.dispatcher.send_stop()
        return not model_failed
