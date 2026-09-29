"""
robomimic_runner.py
--------------------
Inference backend that runs a checkpoint through robomimic's own
RolloutPolicy (FileUtils.policy_from_checkpoint) instead of the
from-scratch reimplementation in model_runner.py.

Exposes the same minimal interface inference_node.py relies on from
ModelRunner: get_action(obs_dict) -> np.ndarray, reset(), and an
.obs_keys list.

RolloutPolicy expects raw, un-normalized observations — it normalizes
internally using the obs_normalization_stats embedded in the checkpoint,
and denormalizes actions the same way. Do not enable `normalize: true` in
obs_config.yaml for any key this backend consumes, or it will be
normalized twice (same caveat already documented for robot0_joint_pos in
the custom model_runner.py path).

This is also the ONLY backend that can run camera/organized-pointcloud obs
(ObsBuilder's "source:" passthrough entries, obs_builder.py). get_action()
below already handles them correctly with no special-casing: `.astype(
np.float32)` and `np.stack(..., axis=0)` operate on arbitrary-shape arrays,
not just 1-D vectors, so an (H,W,C) image passes through unmodified (and
frame-history-stacks into (To,H,W,C), matching what robomimic's own
FrameStackWrapper produces for images in normal env-based rollouts).
RolloutPolicy then runs its checkpoint's own CNN encoder internally for any
key its shape_metadata marks as an image/depth/scan modality. The custom
model_runner.py backend has no such encoder and will raise a clear error if
handed one of these keys instead.

Modeled directly on robomimic's own reference rollout, run_trained_agent.py
(robomimic/scripts/run_trained_agent.py, also vendored at the repo root):
its rollout() calls policy.start_episode() once, before the episode's first
observation is used, then simply `act = policy(ob=obs)` once per step with a
single current-frame, unbatched, raw obs dict — no manual batching,
normalization, or frame-history stacking in the script itself (RolloutPolicy
handles the first two internally).

Frame-history stacking for multi-frame policies (e.g. Diffusion Policy,
observation_horizon > 1) is NOT done by the script or RolloutPolicy either —
in normal robomimic usage it happens one layer down, at the environment
level: FileUtils.env_from_checkpoint wraps the env in
robomimic.envs.wrappers.FrameStackWrapper whenever config.train.frame_stack
> 1 (kept in sync with config.algo.horizon.observation_horizon by
convention), so env.step()/env.reset() already return (To, D)-shaped obs per
key by the time the script sees them. Since this ROS2 node has no EnvBase
object for robomimic to wrap, _init_obs_history/_stack_obs_history below
reimplement that wrapper's exact algorithm (duplicate-fill on reset,
append + oldest-first stack on each step) directly around ObsBuilder's
single-frame output.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

try:
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
except ImportError as e:
    raise ImportError(
        "robomimic is required for --backend robomimic but is not installed "
        "in this environment. Do NOT `pip install robomimic` — PyPI's "
        "published package only goes up to 0.3.0 and is not compatible with "
        "checkpoints trained against a newer robomimic source checkout. "
        "Instead, install your robomimic source checkout in editable mode "
        "first (`pip install -e /path/to/your/robomimic/checkout`), then "
        "`pip install -e .[robomimic]` in whichever venv runs "
        "inference_node.py. See README.md for the exact sequence."
    ) from e

from relay_d.utils.coloring_logger import logger


def _safe_cfg_get(obj, attr, default=None):
    """getattr() that also tolerates robomimic's locked Config objects, which
    raise RuntimeError/KeyError (not AttributeError) for a missing key."""
    try:
        return getattr(obj, attr)
    except (AttributeError, KeyError, RuntimeError):
        return default


class RobomimicPolicyRunner:
    """Runs inference via robomimic's RolloutPolicy, loaded straight from a checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        torch_device = TorchUtils.get_torch_device(try_to_use_cuda=device.startswith("cuda"))
        self._policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=torch_device)
        self.obs_keys: List[str] = list(ckpt["shape_metadata"]["all_obs_keys"])

        # observation_horizon (To) drives DiffusionPolicyUNet._get_action_trajectory
        # directly; robomimic's own shipped configs keep config.train.frame_stack
        # equal to this by convention, so reading it here is equivalent.
        algo_config = _safe_cfg_get(self._policy.policy, "algo_config")
        horizon_cfg = _safe_cfg_get(algo_config, "horizon")
        obs_horizon = int(_safe_cfg_get(horizon_cfg, "observation_horizon", 0)) if horizon_cfg else 0

        if not obs_horizon:
            # BC-Transformer / BC-Transformer-GMM checkpoints don't set algo.horizon
            # at all -- their temporal context window is driven by
            # algo.transformer.context_length instead (robomimic/algo/bc.py:
            # BC_Transformer_GMM asserts the obs batch's temporal dim matches this
            # exactly), so fall back to it when transformer is the active encoder.
            transformer_cfg = _safe_cfg_get(algo_config, "transformer")
            if transformer_cfg and _safe_cfg_get(transformer_cfg, "enabled", False):
                obs_horizon = int(_safe_cfg_get(transformer_cfg, "context_length", 1))

        self._obs_horizon: int = obs_horizon or 1
        self._obs_history: Dict[str, deque] = {}

        logger.info(
            f"[RobomimicPolicyRunner] algo={ckpt.get('algo_name')}  device={device}  "
            f"obs_horizon={self._obs_horizon}  checkpoint={checkpoint_path}"
        )
        logger.info(f"[RobomimicPolicyRunner] obs_keys: {self.obs_keys}")

        self.reset()

    def reset(self) -> None:
        """Call start_episode() and clear the per-key observation-history buffers."""
        self._policy.start_episode()
        self._obs_history = {k: deque(maxlen=self._obs_horizon) for k in self.obs_keys}

    def _init_obs_history(self, frame: Dict[str, np.ndarray]) -> None:
        """Duplicate-fill each key's history buffer with the first frame.
        Mirrors FrameStackWrapper._get_initial_obs_history()."""
        for k in self.obs_keys:
            self._obs_history[k].extend([frame[k]] * self._obs_horizon)

    def _stack_obs_history(self) -> Dict[str, np.ndarray]:
        """Return the oldest-first (To, D) stack per key from the current
        history buffers. Mirrors FrameStackWrapper._get_stacked_obs_from_history()."""
        return {k: np.stack(self._obs_history[k], axis=0) for k in self.obs_keys}

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Run one forward pass through robomimic's RolloutPolicy. Returns the action array."""
        frame = {}
        for k in self.obs_keys:
            if k not in obs_dict:
                raise KeyError(
                    f"[RobomimicPolicyRunner] Missing obs key '{k}'. Expected: {self.obs_keys}"
                )
            frame[k] = obs_dict[k].astype(np.float32)

        if self._obs_horizon <= 1:
            stacked = frame
        else:
            is_first = not self._obs_history[self.obs_keys[0]]
            if is_first:
                self._init_obs_history(frame)
            else:
                for k in self.obs_keys:
                    self._obs_history[k].append(frame[k])
            stacked = self._stack_obs_history()

        return np.asarray(self._policy(stacked))
