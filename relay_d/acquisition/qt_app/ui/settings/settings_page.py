import os
import json

from relay_d.utils.coloring_logger import logger
from ..settings.config_manager import config
from ...utils.custom_message_box import CustomMessageBox

class SettingsPage:
    def __init__(self, ui_instance):
        self.ui = ui_instance
        self.settings = {}
        self.settings_file = os.path.join(
            os.path.expanduser("~"), ".relayd_settings.json"
        )

        self.load_settings()
        self.setup_connections()

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    self.settings = json.load(f)
                config.settings.update(self.settings)
                self.save_settings()  # Ensure global config is updated
                logger.info("Settings loaded successfully")
            else:
                self.settings = self.get_default_settings()
                self.save_settings()
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self.settings = self.get_default_settings()

    def get_default_settings(self):
        return {
            "hdf5_save_folder": "",
            "hdf5_load_folder": "",
            "enable_compression": False,
            "freeze_containers": True,
            "theme": "light",
        }

    def save_settings(self):
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=4)

            config.save_settings(self.settings)  # Update the global config manager

            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def get_hdf5_save_folder(self):
        return self.settings.get("hdf5_save_folder", "")

    def get_hdf5_load_folder(self):
        return self.settings.get("hdf5_load_folder", "")

    def set_hdf5_save_folder(self, folder):
        self.settings["hdf5_save_folder"] = folder
        self.save_settings()

    def set_hdf5_load_folder(self, folder):
        self.settings["hdf5_load_folder"] = folder
        self.save_settings()

    def setup_connections(self):
        try:
            if hasattr(self.ui, "btn_save_settings"):
                self.ui.btn_save_settings.clicked.connect(self.save_settings_from_ui)

            if hasattr(self.ui, "btn_reset_settings"):
                self.ui.btn_reset_settings.clicked.connect(self.reset_to_default)

            if hasattr(self.ui, "btn_browse_save_folder"):
                self.ui.btn_browse_save_folder.clicked.connect(self.browse_save_folder)

            if hasattr(self.ui, "btn_browse_load_folder"):
                self.ui.btn_browse_load_folder.clicked.connect(self.browse_load_folder)

            self.apply_settings_to_ui()
            logger.info("Settings page connections established")

        except Exception as e:
            logger.error(f"Error setting up settings connections: {e}")

    def apply_settings_to_ui(self):
        try:
            if hasattr(self.ui, "lineEdit_hdf5_save_folder"):
                self.ui.lineEdit_hdf5_save_folder.setText(self.get_hdf5_save_folder())

            if hasattr(self.ui, "lineEdit_hdf5_load_folder"):
                self.ui.lineEdit_hdf5_load_folder.setText(self.get_hdf5_load_folder())

            if hasattr(self.ui, "checkBox_compression"):
                self.ui.checkBox_compression.setChecked(
                    self.settings.get("enable_compression", False)
                )

            if hasattr(self.ui, "checkBox_freezeContainer"):
                self.ui.checkBox_freezeContainer.setChecked(
                    self.settings.get("freeze_containers", False)
                )

            logger.info("Settings applied to UI")

        except Exception as e:
            logger.error(f"Error applying settings to UI: {e}")

    def browse_save_folder(self):
        from PyQt5.QtWidgets import QFileDialog

        current = ""
        if hasattr(self.ui, "lineEdit_hdf5_save_folder"):
            current = self.ui.lineEdit_hdf5_save_folder.text()

        folder = QFileDialog.getExistingDirectory(
            self.ui, "Select HDF5 Save Folder", current
        )
        if folder and hasattr(self.ui, "lineEdit_hdf5_save_folder"):
            self.ui.lineEdit_hdf5_save_folder.setText(folder)

    def browse_load_folder(self):
        from PyQt5.QtWidgets import QFileDialog

        current = ""
        if hasattr(self.ui, "lineEdit_hdf5_load_folder"):
            current = self.ui.lineEdit_hdf5_load_folder.text()

        folder = QFileDialog.getExistingDirectory(
            self.ui, "Select Default HDF5 Load Folder", current
        )
        if folder and hasattr(self.ui, "lineEdit_hdf5_load_folder"):
            self.ui.lineEdit_hdf5_load_folder.setText(folder)

    def save_settings_from_ui(self):

        try:
            if hasattr(self.ui, "lineEdit_hdf5_save_folder"):
                self.settings["hdf5_save_folder"] = (
                    self.ui.lineEdit_hdf5_save_folder.text()
                )

            if hasattr(self.ui, "lineEdit_hdf5_load_folder"):
                self.settings["hdf5_load_folder"] = (
                    self.ui.lineEdit_hdf5_load_folder.text()
                )

            if hasattr(self.ui, "checkBox_autoSave"):
                self.settings["auto_save"] = self.ui.checkBox_autoSave.isChecked()

            if hasattr(self.ui, "checkBox_compression"):
                self.settings["enable_compression"] = (
                    self.ui.checkBox_compression.isChecked()
                )

            if hasattr(self.ui, "checkBox_freezeContainer"):
                self.settings["freeze_containers"] = (
                    self.ui.checkBox_freezeContainer.isChecked()
                )

            self.save_settings()
            CustomMessageBox.info("Settings saved successfully!")
            logger.info("Settings updated from UI")

        except Exception as e:
            logger.error(f"Error saving settings from UI: {e}")
            CustomMessageBox.error(f"Failed to save settings: {e}")

    def reset_to_default(self):

        if CustomMessageBox.question("Are you sure you want to reset all settings to default?", title="Confirm Reset"):
            self.settings = self.get_default_settings()
            self.save_settings()
            self.apply_settings_to_ui()
            CustomMessageBox.info("Settings reset to default!")
            logger.info("Settings reset to default")