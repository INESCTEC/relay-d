import os
from typing import Optional


class PostprocessConfigLoader:
    """Finds the robomimic training-config templates (``postprocess_configs/robomimic_<model>_config.json``)."""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprocess_configs")

    def get_output_train_config(self, format_name: str, model: str = "bc_rnn_gmm") -> str:
        """Path of the training-config template of ``model`` for ``format_name``."""
        return os.path.join(self.config_dir, f"{format_name}_{model}_config.json")
