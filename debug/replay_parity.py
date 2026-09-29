#!/usr/bin/env python3
"""
replay_parity.py
----------------
Offline parity check for the "works in test_model.py, misbehaves live" bug class.

Replays a demo from validation_dataset.h5 through two independent inference
paths, using the exact same raw (un-normalized) observations read straight
from the HDF5 file for both:

  (a) this repo's scripts/model_runner.py ModelRunner (the code that actually
      runs on the robot),
  (b) robomimic's own RolloutPolicy via policy_from_checkpoint (the code
      test_model.py already validated as "correct").

and diffs both against each other and against the ground-truth recorded
actions. This isolates bugs in the obs-vector-building / normalization /
diffusion-inference code path from bugs in live sensor acquisition
(obs_builder.py), which is a separate, live-only failure mode not exercised
by this script.

Usage:
    debug/replay_parity.py --demo demo_0 --num-steps 40
    debug/replay_parity.py --demo demo_0 --skip-robomimic   # fast, this-repo only

The robomimic reference path needs a robomimic install; if unavailable in
this venv, point --robomimic-path at a robomimic checkout (added to
sys.path) or pass --skip-robomimic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = "/home/pascd/Downloads/dp_training/validation_dataset.h5"
CKPT_PATH = "/home/pascd/Downloads/dp_training/last.pth"

# Keys this checkpoint's shape_metadata declares (order doesn't matter here —
# each side re-orders per its own obs_keys / all_obs_keys).
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]


def load_demo(h5_path: str, demo_name: str):
    with h5py.File(h5_path, "r") as f:
        demo = f[f"data/{demo_name}"]
        obs = {k: np.array(demo[f"obs/{k}"]) for k in OBS_KEYS}
        actions = np.array(demo["actions"])
    return obs, actions


def run_this_repo(obs: dict, num_steps: int, ckpt_path: str) -> np.ndarray:
    """Feed single-frame raw obs through this repo's ModelRunner, one tick at
    a time — mirrors exactly how inference_node.py drives it live, letting
    ModelRunner's own internal obs ring-buffer (obs_horizon) and
    normalization do the work."""
    from relay_d.dispatch.model_runner import ModelRunner

    runner = ModelRunner(ckpt_path, device="cpu", model_type="diffusion")
    runner.reset()

    preds = []
    for i in range(num_steps):
        frame = {k: obs[k][i] for k in OBS_KEYS}
        preds.append(runner.get_action(frame))
    return np.array(preds)


def run_robomimic_reference(obs: dict, num_steps: int, ckpt_path: str, robomimic_path: str | None) -> np.ndarray:
    """Exactly mirrors test_model.py: manual 2-frame stack (duplicate-padded
    at t=0) fed through robomimic's RolloutPolicy, which owns its own
    receding-horizon action queue internally."""
    if robomimic_path:
        sys.path.insert(0, robomimic_path)

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    policy, _ckpt = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device)
    policy.start_episode()

    preds = []
    for i in range(num_steps):
        idx0, idx1 = (0, 0) if i == 0 else (i - 1, i)
        obs_dict = {k: np.stack([obs[k][idx0], obs[k][idx1]], axis=0) for k in OBS_KEYS}
        pred = np.squeeze(policy(obs_dict))
        preds.append(pred)
    return np.array(preds)


def summarize(name: str, pred: np.ndarray, gt: np.ndarray) -> None:
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    print(f"\n{name} per-dim MAE vs ground truth:")
    print(f"  joints 0-6: {np.array2string(mae[:7], precision=5, suppress_small=True)}")
    print(f"  gripper(7): {mae[7]:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--num-steps", type=int, default=40, help="0 = full demo length")
    ap.add_argument("--dataset", default=DATASET_PATH)
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--skip-robomimic", action="store_true", help="skip the robomimic reference run")
    ap.add_argument("--robomimic-path", default=None, help="path to a robomimic checkout to add to sys.path")
    ap.add_argument("--quiet", action="store_true", help="skip the per-step table, print summary only")
    args = ap.parse_args()

    obs, gt_actions = load_demo(args.dataset, args.demo)
    num_steps = args.num_steps if args.num_steps > 0 else gt_actions.shape[0]
    gt_actions = gt_actions[:num_steps]

    print(f"Demo={args.demo}  steps={num_steps}  obs_keys={OBS_KEYS}")

    print("\n--- running this-repo ModelRunner ---")
    this_repo_actions = run_this_repo(obs, num_steps, args.ckpt)

    ref_actions = None
    if not args.skip_robomimic:
        print("\n--- running robomimic reference (RolloutPolicy) ---")
        ref_actions = run_robomimic_reference(obs, num_steps, args.ckpt, args.robomimic_path)

    if not args.quiet:
        header = f"{'STEP':<5} | {'THIS-REPO (8D)':<50} | "
        header += f"{'ROBOMIMIC-REF (8D)':<50} | " if ref_actions is not None else ""
        header += "GROUND TRUTH (8D)"
        print("\n" + header)
        print("=" * len(header))
        fmt = {"float_kind": lambda x: f"{x:>6.3f}"}
        for i in range(num_steps):
            row = f"[{i:03d}] | {np.array2string(this_repo_actions[i], formatter=fmt):<50} | "
            if ref_actions is not None:
                row += f"{np.array2string(ref_actions[i], formatter=fmt):<50} | "
            row += np.array2string(gt_actions[i], formatter=fmt)
            print(row)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summarize("this-repo", this_repo_actions, gt_actions)

    if ref_actions is not None:
        summarize("robomimic-ref", ref_actions, gt_actions)

        diff = this_repo_actions - ref_actions
        mae_diff = np.mean(np.abs(diff), axis=0)
        print("\nthis-repo vs robomimic-ref per-dim MAE (should be small if the "
              "inference path is equivalent):")
        print(f"  joints 0-6: {np.array2string(mae_diff[:7], precision=5, suppress_small=True)}")
        print(f"  gripper(7): {mae_diff[7]:.5f}")

        # Both action streams are per-step deltas; a receding-horizon bug
        # (this repo replans every tick instead of every action_horizon
        # ticks like the validated reference) shows up as cumulative drift
        # over the rollout even if per-step MAE looks small.
        cum_this = np.cumsum(this_repo_actions[:, :7], axis=0)
        cum_ref = np.cumsum(ref_actions[:, :7], axis=0)
        cum_drift = np.abs(cum_this[-1] - cum_ref[-1])
        print(f"\nCumulative summed-delta drift over the rollout, this-repo vs "
              f"reference (joints 0-6, should stay small):")
        print(f"  {np.array2string(cum_drift, precision=5, suppress_small=True)}")


if __name__ == "__main__":
    main()
