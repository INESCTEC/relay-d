"""Run a trained policy on the robot with the configs next to this script.

Same options and defaults as `lfd-inference-node` (rate, stale-input limit, cleanup on Ctrl-C);
only --obs-config / --action-config default to the files in the current folder.
Example: python inference_script.py --checkpoint path/to/model.pth
"""

import sys

from relay_d.dispatch.inference_node import main

# Default configuration file paths
ACTION_CONFIG = "./action_config.yaml"
OBS_CONFIG = "./obs_config.yaml"

if __name__ == "__main__":
    for flag, path in (("--obs-config", OBS_CONFIG), ("--action-config", ACTION_CONFIG)):
        if not any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:]):
            sys.argv += [flag, path]
    main()
