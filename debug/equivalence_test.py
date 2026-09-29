#!/usr/bin/env python3
"""
equivalence_test.py
--------------------
Strict equivalence test between this repo's own Diffusion Policy inference
code (relay_d/dispatch/model_runner.py) and robomimic's real reference implementation
— not "close in aggregate" like replay_parity.py's MAE-over-a-rollout report,
but a direct, per-tick check that both sides compute the *same function*:

  1. Observation vector: the exact normalized, flattened obs_cond tensor fed
     into the noise-prediction network. This repo's side calls its own real
     `_build_obs_vector()` (the actual production code path, not a
     reimplementation of its formula). Robomimic's side calls robomimic's
     real `ObsUtils.normalize_dict` + the real trained `obs_encoder` network
     via `TensorUtils.time_distributed`, mirroring exactly what
     `DiffusionPolicyUNet._get_action_trajectory` does internally.

  2. Action: diffusion sampling is stochastic (both implementations draw
     fresh random noise while denoising). Naively calling
     `torch.manual_seed(seed)` before each side is NOT enough to make them
     match: robomimic's real noise_pred_net keeps its externally-visible
     diffusion sample in (B, T, action_dim) layout, while this repo's keeps
     (B, action_dim, T) — a harmless internal convention difference (both
     already verified correct independently), but it means a naively-shared
     RNG stream gets reshaped into transposed layouts on each side, so the
     same random draw lands on a different (channel, timestep) pair in each
     — producing a small, non-shrinking, uniform-looking error that looks
     like a bug but isn't one.

     To get a genuine equivalence check, this script instead pre-generates a
     fixed sequence of (action_dim, pred_horizon) noise grids from one seed,
     then temporarily monkeypatches torch.randn/torch.randn_like while each
     side runs to hand out that EXACT sequence — reshaped directly for this
     repo's (B, action_dim, T) convention, transposed for robomimic's
     (B, T, action_dim) convention — so the same physical noise value lands
     on the same (channel, timestep) pair in both, regardless of which
     tensor layout each implementation happens to use internally.

Both sides are driven through the SAME sequence of ticks (like
replay_parity.py), so their receding-horizon replan cadence (replan every
action_horizon ticks, cached-pop otherwise) stays in lockstep automatically —
no manual queue manipulation needed.

Usage:
    debug/equivalence_test.py --demo demo_0 --num-samples 16
    debug/equivalence_test.py --demo demo_5 --num-samples 16 --seed 123

    # Against real live observations instead of an HDF5 demo (read-only —
    # only subscribes via ObsBuilder, never dispatches/publishes anything):
    debug/equivalence_test.py --live --num-samples 16
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from replay_parity import load_demo, OBS_KEYS, DATASET_PATH, CKPT_PATH  # noqa: E402

OBS_ATOL = 1e-5
ACTION_ATOL = 1e-4
DEFAULT_OBS_CONFIG = str(REPO_ROOT / "config" / "obs_config.yaml")


def _wait_for_obs(obs_node, obs_keys: list, timeout: float, poll: float = 0.1) -> bool:
    """Block until all obs_keys appear in get_obs(), or timeout expires.
    Mirrors inference_node.py's own pre-flight helper."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(k in obs_node.get_obs() for k in obs_keys):
            return True
        time.sleep(poll)
    return False


def capture_live_obs(obs_config_path: str, num_samples: int, hz: float, obs_wait: float) -> dict:
    """Capture `num_samples` consecutive real obs snapshots from the running
    robot/sim via ObsBuilder — the exact same class production code uses, so
    this exercises the real TF-lookup/joint-state/sync-threshold path, not a
    replay of recorded values. Strictly read-only: only a subscriber node is
    created, nothing is ever published to the robot."""
    import threading

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from relay_d.dispatch.obs_builder import ObsBuilder

    rclpy.init()
    obs_node = ObsBuilder(obs_config_path)
    executor = MultiThreadedExecutor()
    executor.add_node(obs_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        print(f"[live] Validating input sources (timeout={obs_wait}s) ...")
        all_ok, report = obs_node.validate_inputs(timeout=obs_wait)
        for name, result in report.items():
            status = "OK" if result["ok"] else "FAIL"
            detail = result.get("detail", result.get("error", ""))
            print(f"  [{status}] {name} ({result['kind']}): {detail}")
        if not all_ok:
            raise RuntimeError("[live] One or more inputs are not reachable — check obs_config.yaml.")

        print(f"[live] Waiting up to {obs_wait}s for all obs keys to populate ...")
        if not _wait_for_obs(obs_node, OBS_KEYS, timeout=obs_wait):
            raise RuntimeError("[live] Timed out waiting for observations — check obs_config.yaml field paths.")

        print(f"[live] Capturing {num_samples} samples at {hz} Hz ...")
        dt = 1.0 / hz if hz > 0 else 0.0
        frames = {k: [] for k in OBS_KEYS}
        for _ in range(num_samples):
            t0 = time.monotonic()
            snapshot = obs_node.get_obs()
            for k in OBS_KEYS:
                frames[k].append(np.array(snapshot[k], dtype=np.float32))
            if dt > 0:
                elapsed = time.monotonic() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        return {k: np.stack(v, axis=0) for k, v in frames.items()}
    finally:
        executor.shutdown()
        obs_node.destroy_node()
        rclpy.shutdown()


class _SharedNoise:
    """Pre-generates a fixed sequence of (action_dim, pred_horizon) noise
    grids from one seed, then replays that exact sequence into whichever
    implementation is currently running — transposed as needed so the same
    physical noise value lands on the same (channel, timestep) pair
    regardless of that implementation's internal tensor layout."""

    def __init__(self, seed: int, action_dim: int, pred_horizon: int, num_draws: int):
        gen = torch.Generator().manual_seed(seed)
        self._grids = [torch.randn(action_dim, pred_horizon, generator=gen) for _ in range(num_draws)]
        self._idx = 0
        self._transpose = False

    def reset(self, transpose: bool) -> None:
        self._idx = 0
        self._transpose = transpose

    def next(self, shape) -> torch.Tensor:
        if self._idx >= len(self._grids):
            raise RuntimeError(
                f"_SharedNoise exhausted after {self._idx} draws — increase num_draws."
            )
        grid = self._grids[self._idx]
        self._idx += 1
        if self._transpose:
            grid = grid.transpose(0, 1)
        return grid.reshape(shape)


@contextmanager
def _inject_noise(noise: "_SharedNoise"):
    """Temporarily replace torch.randn/torch.randn_like so every call inside
    the `with` block draws from `noise` instead of the real RNG."""
    real_randn = torch.randn
    real_randn_like = torch.randn_like

    def fake_randn(*args, **kwargs):
        shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
        return noise.next(shape)

    def fake_randn_like(tensor, **kwargs):
        return noise.next(tuple(tensor.shape))

    torch.randn = fake_randn
    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.randn = real_randn
        torch.randn_like = real_randn_like


def _robomimic_obs_cond(algo, ckpt, obs_dict_np: dict, device) -> np.ndarray:
    """Recompute robomimic's real obs_cond for one 2-frame-stacked obs_dict,
    calling robomimic's actual normalize_dict + trained obs_encoder — not a
    hand-derived reimplementation of either."""
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.tensor_utils as TensorUtils

    obs_dict = {
        k: torch.from_numpy(v).float().unsqueeze(0).to(device)  # (1, T, D)
        for k, v in obs_dict_np.items()
    }
    normalized = ObsUtils.normalize_dict(obs_dict, normalization_stats=ckpt["obs_normalization_stats"])

    nets = algo.ema.averaged_model if algo.ema is not None else algo.nets
    obs_features = TensorUtils.time_distributed(
        {"obs": normalized, "goal": None}, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
    )
    obs_cond = obs_features.flatten(start_dim=1)
    return obs_cond.detach().cpu().numpy()


def run(label: str, obs: dict, ckpt_path: str, robomimic_path: str, num_samples: int, seed: int) -> bool:
    if robomimic_path:
        sys.path.insert(0, robomimic_path)

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    from relay_d.dispatch.model_runner import ModelRunner

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device)
    algo = policy.policy

    runner = ModelRunner(ckpt_path, device="cpu", model_type="diffusion")
    runner.reset()
    policy.start_episode()

    action_dim = runner._policy._action_dim
    pred_horizon = runner._policy._pred_horizon
    num_draws = len(runner._policy._scheduler.timesteps) + 1  # 1 initial + up to 1 per denoise step

    obs_results = []
    action_results = []

    for i in range(num_samples):
        idx0, idx1 = (0, 0) if i == 0 else (i - 1, i)
        frame_i = {k: obs[k][i] for k in OBS_KEYS}
        stacked = {k: np.stack([obs[k][idx0], obs[k][idx1]], axis=0) for k in OBS_KEYS}

        noise = _SharedNoise(seed=seed + i, action_dim=action_dim, pred_horizon=pred_horizon, num_draws=num_draws)

        noise.reset(transpose=False)  # this repo's native (action_dim, T) layout
        with _inject_noise(noise):
            this_action = runner.get_action(frame_i)
        this_obs_cond = runner._policy.last_obs_cond  # None on cached-pop ticks (no replan this tick)

        noise.reset(transpose=True)  # robomimic's native (T, action_dim) layout
        with _inject_noise(noise):
            ref_action = np.squeeze(policy(stacked))
        ref_obs_cond = _robomimic_obs_cond(algo, ckpt, stacked, device) if this_obs_cond is not None else None

        action_diff = np.abs(this_action - ref_action)
        action_pass = np.allclose(this_action, ref_action, atol=ACTION_ATOL)
        action_results.append((i, action_pass, action_diff))

        if this_obs_cond is not None and ref_obs_cond is not None:
            obs_diff = np.abs(this_obs_cond - ref_obs_cond)
            obs_pass = np.allclose(this_obs_cond, ref_obs_cond, atol=OBS_ATOL)
            obs_results.append((i, obs_pass, obs_diff))

    print(f"\n{label}  samples={num_samples}  seed={seed}")
    print("=" * 70)
    print(f"OBSERVATION VECTOR equivalence (checked on {len(obs_results)} replan tick(s), "
          f"atol={OBS_ATOL}):")
    obs_all_pass = True
    for i, ok, diff in obs_results:
        obs_all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{i:03d}] {status}  max_abs_diff={diff.max():.2e}  mean_abs_diff={diff.mean():.2e}")
    if not obs_results:
        print("  (no replan ticks sampled — increase --num-samples past action_horizon)")

    print(f"\nACTION equivalence (checked on all {num_samples} tick(s), atol={ACTION_ATOL}, "
          f"noise-injected so both sides denoise from identical, layout-matched noise):")
    action_all_pass = True
    for i, ok, diff in action_results:
        action_all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{i:03d}] {status}  max_abs_diff={diff.max():.2e}  mean_abs_diff={diff.mean():.2e}  "
              f"per_dim={np.array2string(diff, precision=6, suppress_small=True)}")

    print("\n" + "=" * 70)
    overall = obs_all_pass and action_all_pass
    print(f"OVERALL: {'PASS ✅ — this-repo and robomimic compute the same thing' if overall else 'FAIL ❌ — see diffs above'}")
    print("=" * 70)
    return overall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--dataset", default=DATASET_PATH)
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--robomimic-path", default=None, help="path to a robomimic checkout to add to sys.path")
    ap.add_argument("--num-samples", type=int, default=16, help="number of consecutive ticks to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", action="store_true",
                     help="capture real observations from the running robot/sim via ObsBuilder "
                          "instead of reading an HDF5 demo. Read-only — never dispatches/publishes.")
    ap.add_argument("--obs-config", default=DEFAULT_OBS_CONFIG, help="only used with --live")
    ap.add_argument("--hz", type=float, default=20.0, help="live capture tick rate, only used with --live")
    ap.add_argument("--obs-wait", type=float, default=10.0,
                     help="seconds to wait for live observations to populate, only used with --live")
    args = ap.parse_args()

    if args.live:
        obs = capture_live_obs(args.obs_config, args.num_samples, args.hz, args.obs_wait)
        label = f"LIVE capture (obs_config={args.obs_config}, hz={args.hz})"
    else:
        obs, _gt_actions = load_demo(args.dataset, args.demo)
        label = f"Demo={args.demo}"

    ok = run(label, obs, args.ckpt, args.robomimic_path, args.num_samples, args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
