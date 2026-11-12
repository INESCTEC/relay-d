"""First start: recordings need a save folder. Asked before anything else can be used (GUI only; the API and CLIs keep their own defaults)."""

import os

from PyQt5.QtWidgets import QFileDialog

from relay_d.utils import user_settings
from .ui.settings.config_manager import config
from .utils.custom_message_box import CustomMessageBox


def ask_save_folder(parent=None, choose=QFileDialog.getExistingDirectory, quit_asked=CustomMessageBox.question) -> bool:
    """Ask for ``hdf5_save_folder`` until one is set. False if the user would rather quit than choose one."""
    while not user_settings.get("hdf5_save_folder"):
        folder = choose(parent, "Choose the folder where recordings are saved", os.path.expanduser("~"))
        if folder:
            config.save_settings({"hdf5_save_folder": folder})  # also tells the pages (Settings, Record) about it
        elif quit_asked("Relay-D needs a folder to save recordings. Quit?", title="Recordings folder", parent=parent):
            return False
    return True
