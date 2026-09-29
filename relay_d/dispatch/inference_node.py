"""
inference_node.py
-----------------
CLI entry point. Thin wrapper around api.InferenceSession, which does the
actual work: grab latest obs -> run policy forward pass -> dispatch action.

Supported model types (auto-detected unless --model-type is set):
  bc_mlp | bc_rnn | bc_gmm | diffusion | act | torch_script

Usage (after `pip install -e .`):
    lfd-inference-node \
        --checkpoint path/to/model.pth \
        --obs-config path/to/obs_config.yaml \
        --action-config path/to/action_config.yaml \
        --device cpu

    # Explicit type + custom obs keys (required for TorchScript):
    lfd-inference-node \
        --checkpoint model.pt \
        --model-type torch_script \
        --obs-keys robot0_eef_pos,robot0_eef_quat \
        --obs-config obs_config.yaml \
        --action-config action_config.yaml

    # Use robomimic's own RolloutPolicy instead of the custom reimplementation:
    lfd-inference-node \
        --checkpoint model.pth \
        --backend robomimic \
        --obs-config obs_config.yaml \
        --action-config action_config.yaml \
        --device cpu
"""

import argparse

from .api import InferenceSession


def main():
    parser = argparse.ArgumentParser(description="LfD Inference Node")
    parser.add_argument("--checkpoint",    required=True,  help="Path to policy checkpoint (.pth or .pt)")
    parser.add_argument("--obs-config",    required=True,  help="Path to obs_config.yaml")
    parser.add_argument("--action-config", required=True,  help="Path to action_config.yaml")
    parser.add_argument("--device",        default="cuda",  help="Torch device: cpu | cuda")
    parser.add_argument("--hz",            type=float, default=0.0,
                        help="Max inference rate in Hz (0 = as fast as possible)")
    parser.add_argument("--model-type",    default="auto",
                        help="Model type: auto | bc_mlp | bc_rnn | bc_gmm | "
                             "diffusion | act | torch_script  (default: auto)")
    parser.add_argument("--obs-keys",      default=None,
                        help="Comma-separated obs keys in training order "
                             "(e.g. robot0_eef_pos,robot0_eef_quat). "
                             "Required for torch_script; overrides default for all types.")
    parser.add_argument("--obs-wait",      type=float, default=10.0,
                        help="Seconds to wait for all observations before starting inference (default: 10.0)")
    parser.add_argument("--action-horizon", type=int, default=None,
                        help="Override the diffusion policy's receding-horizon control window "
                             "(number of actions consumed per replan) instead of using the "
                             "value baked into the checkpoint's training config. Only used "
                             "with --backend custom --model-type diffusion (or auto-detected "
                             "diffusion). Must be within [1, pred_horizon - (obs_horizon - 1)].")
    parser.add_argument("--backend",       choices=["custom", "robomimic"], default="custom",
                        help="Inference backend: 'custom' (model_runner.py, no robomimic "
                             "dependency) or 'robomimic' (robomimic's own RolloutPolicy, "
                             "loaded via FileUtils.policy_from_checkpoint). --model-type and "
                             "--obs-keys are ignored for 'robomimic' (derived from the "
                             "checkpoint's own shape_metadata/algo_name). Default: custom")
    args = parser.parse_args()

    obs_keys = [k.strip() for k in args.obs_keys.split(",")] if args.obs_keys else None

    session = InferenceSession(
        checkpoint=args.checkpoint,
        obs_config=args.obs_config,
        action_config=args.action_config,
        device=args.device,
        backend=args.backend,
        model_type=args.model_type,
        obs_keys=obs_keys,
        action_horizon=args.action_horizon,
    )
    try:
        session.run(hz=args.hz, validate_timeout=args.obs_wait)
    finally:
        session.close()


if __name__ == "__main__":
    main()
