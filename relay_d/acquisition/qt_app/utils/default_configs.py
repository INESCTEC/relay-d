import os
import shutil
from importlib import resources

from PyQt5.QtWidgets import QFileDialog

from relay_d.utils.coloring_logger import logger
from ..ui.settings.config_manager import config
from .custom_message_box import CustomMessageBox

TEMPLATES = ("config_template.yaml", "postprocess_config_template.yaml")


def generate_default_configs(parent=None):
    """Copy the default configs into a user-picked folder and remember it as the
    start folder for every config-opening dialog (``config_folder`` setting)."""
    folder = QFileDialog.getExistingDirectory(
        parent, "Select folder for the default configs", config.config_dialog_dir()
    )
    if not folder:
        return None

    try:
        src_dir = resources.files("relay_d.acquisition.qt_app.ui.config").joinpath("templates")
        for name in TEMPLATES:
            dst = os.path.join(folder, name)
            if os.path.exists(dst) and not CustomMessageBox.question(
                f"{name} already exists in this folder. Overwrite it?", parent=parent
            ):
                continue
            with resources.as_file(src_dir.joinpath(name)) as src:
                shutil.copyfile(src, dst)
    except Exception as e:
        logger.error(f"Failed to generate default configs: {e}")
        CustomMessageBox.error(f"Failed to generate default configs:\n{e}", parent=parent)
        return None

    config.save_settings({"config_folder": folder})
    logger.info(f"Default configs written to {folder}")
    CustomMessageBox.info(f"Default configs saved to:\n{folder}", parent=parent)
    return folder
