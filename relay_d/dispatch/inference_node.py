"""
inference_node.py
-----------------
CLI entry point. Thin wrapper around api.InferenceSession, which does the
actual work: grab latest obs -> run policy forward pass -> dispatch action.

Two backends, chosen from the checkpoint unless --backend says otherwise:
  robomimic  a robomimic checkpoint file (.pth), run with robomimic's own RolloutPolicy
  lerobot    a LeRobot pretrained_model folder (ACT, Diffusion, ...), run with LeRobot

Usage (after `pip install -e .`):
    lfd-inference-node \
        --checkpoint path/to/model.pth \
        --obs-config path/to/obs_config.yaml \
        --action-config path/to/action_config.yaml \
        --device cpu

    lfd-inference-node \
        --checkpoint path/to/run/output/checkpoints/last/pretrained_model \
        --obs-config path/to/obs_config.yaml \
        --action-config path/to/action_config.yaml
"""

import argparse
import sys

from relay_d.utils import user_settings

from .api import InferenceSession, detect_backend
from .reexec import reexec_if_needed


def main():
    parser = argparse.ArgumentParser(description="LfD Inference Node")
    parser.add_argument("--checkpoint",    required=True,
                        help="robomimic checkpoint (.pth, a pickle that runs code on load: only use checkpoints you trust) "
                             "or a LeRobot pretrained_model folder (e.g. .../checkpoints/last/pretrained_model)")
    parser.add_argument("--obs-config",    required=True,  help="Path to obs_config.yaml")
    parser.add_argument("--action-config", required=True,  help="Path to action_config.yaml")
    parser.add_argument("--device",        default=user_settings.get("infer_device"),  help="Torch device: cpu | cuda (default: the infer_device setting)")
    parser.add_argument("--hz",            type=float, default=None,
                        help="Control rate in Hz (0 = as fast as possible). Default: the rate the policy was trained at "
                             "(a LeRobot dataset's fps), else the infer_hz setting")
    parser.add_argument("--max-obs-age",   type=float, default=user_settings.get("infer_max_obs_age_s"),
                        help="Skip dispatching while any input is older than this many seconds (default: the infer_max_obs_age_s setting)")
    parser.add_argument("--obs-wait",      type=float, default=user_settings.get("infer_obs_wait_s"),
                        help="Seconds to wait for all observations before starting inference (default: the infer_obs_wait_s setting)")
    parser.add_argument("--backend",       choices=["auto", "robomimic", "lerobot"], default=user_settings.get("infer_backend"),
                        help="Inference backend. auto: robomimic for a .pth/.pt file, lerobot for a pretrained_model "
                             "folder (default: the infer_backend setting)")
    parser.add_argument("--state-keys",    default=None,
                        help="lerobot only: comma-separated observations that form observation.state, in order. "
                             "Default: read from the dataset the policy was trained on")
    parser.add_argument("--task",          default=None,
                        help="lerobot only: the instruction a language-conditioned policy (SmolVLA) follows. "
                             "Default: the task of the dataset the policy was trained on")
    args = parser.parse_args()

    try:  # the package of the backend may live in another environment (the one built by the Train tab)
        backend = args.backend if args.backend != "auto" else detect_backend(args.checkpoint)
        reexec_if_needed(backend, ["-m", "relay_d.dispatch.inference_node", *sys.argv[1:]])
    except ValueError:
        pass  # an unknown checkpoint: InferenceSession reports it

    state_keys = [k.strip() for k in args.state_keys.split(",")] if args.state_keys else None

    session = InferenceSession(
        checkpoint=args.checkpoint,
        obs_config=args.obs_config,
        action_config=args.action_config,
        device=args.device,
        backend=args.backend,
        state_keys=state_keys,
        task=args.task,
    )
    # Each predicted action is one step of the training data: at another rate the robot moves faster or slower than shown.
    hz = args.hz if args.hz is not None else (getattr(session.model, "fps", None) or user_settings.get("infer_hz"))
    try:
        session.run(hz=hz, validate_timeout=args.obs_wait, max_obs_age=args.max_obs_age)
    finally:
        session.close()


if __name__ == "__main__":
    main()
