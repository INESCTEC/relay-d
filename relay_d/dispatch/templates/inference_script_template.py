from relay_d.dispatch.api import InferenceSession
import argparse

# Default configuration file paths
ACTION_CONFIG="./action_config.yaml"
OBS_CONFIG="./obs_config.yaml"

def main():
  # Create argument parser for command-line interface
  parser = argparse.ArgumentParser(description="LfD Inference Node")

  # Required checkpoint path argument
  parser.add_argument("--checkpoint",
                      required=True,
                      help="Path to policy checkpoint (.pth or .pt)")

  # Observation configuration file path
  parser.add_argument("--obs-config",
                      help="Path to obs_config.yaml",
                      default=OBS_CONFIG)

  # Action configuration file path
  parser.add_argument("--action-config",
                      help="Path to action_config.yaml",
                      default=ACTION_CONFIG)

  # Torch device selection (CPU or CUDA)
  parser.add_argument("--device",
                      default="cuda",
                      help="Torch device: cpu | cuda")

  # Maximum inference rate in Hz (0 = unlimited)
  parser.add_argument("--hz",
                      type=float,
                      default=0.0,
                      help="Max inference rate in Hz (0 = as fast as possible)")

  # Wait time for all observations before inference starts
  parser.add_argument("--obs-wait",
                      type=float,
                      default=5.0,
                      help="Seconds to wait for all observations before starting inference (default: 10.0)")

  # Override action horizon for diffusion policy models
  parser.add_argument("--action-horizon",
                      type=int,
                      default=None,
                      help="Override the diffusion policy's receding-horizon control window "
                            "(number of actions consumed per replan) instead of using the "
                            "value baked into the checkpoint's training config. Only used "
                            "with --backend custom --model-type diffusion (or auto-detected "
                            "diffusion). Must be within [1, pred_horizon - (obs_horizon - 1)].")

  # Select inference backend implementation
  parser.add_argument("--backend",
                      choices=["custom", "robomimic"],
                      default="custom",
                      help="Inference backend: 'custom' (model_runner.py, no robomimic "
                            "dependency) or 'robomimic' (robomimic's own RolloutPolicy, "
                            "loaded via FileUtils.policy_from_checkpoint). --model-type and "
                            "--obs-keys are ignored for 'robomimic' (derived from the "
                            "checkpoint's own shape_metadata/algo_name). Default: custom")

  # Parse command-line arguments
  args = parser.parse_args()

  # Initialize inference session with parsed arguments
  session = InferenceSession(
          checkpoint=args.checkpoint,
          obs_config=args.obs_config,
          action_config=args.action_config,
          device=args.device,
          backend=args.backend,
          model_type=args.model_type,
          action_horizon=args.action_horizon,
      )

  # Start inference session
  session.run()

if __name__ == "__main__":
  main()
