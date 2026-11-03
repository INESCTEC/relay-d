"""
robomimic_runner.py
-------------------
Inference backend that runs a robomimic checkpoint through robomimic's own
RolloutPolicy (FileUtils.policy_from_checkpoint). Same interface as
LeRobotPolicyRunner: get_action(obs_dict) -> np.ndarray, reset(), and an
.obs_keys list.

- RolloutPolicy wants raw observations and normalizes internally with the
  statistics in the checkpoint: do not set `normalize: true` in obs_config.yaml
  for keys this backend uses, or they are normalized twice.
- Camera / organized point cloud observations (`source:` entries) reach the
  policy unchanged: RolloutPolicy runs the checkpoint's own encoder.
- Multi-frame policies (e.g. Diffusion Policy) get their frame history in
  robomimic from FrameStackWrapper around the environment. There is no
  environment here, so _init_obs_history / _stack_obs_history below repeat that
  wrapper's algorithm (duplicate-fill on reset, append and stack oldest-first)
  around ObsBuilder's single-frame output.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

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
