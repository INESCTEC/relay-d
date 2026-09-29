import os
import json
from PyQt5.QtCore import QObject, pyqtSignal


class ConfigManager(QObject):
    # This signal notifies other parts of the app when settings change
    settings_changed = pyqtSignal(dict)

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self.settings_file = os.path.join(
            os.path.expanduser("~"), ".relayd_settings.json"
        )
        self.settings = self.load_settings()
        self._initialized = True

    def load_settings(self):
        default_settings = {
            "hdf5_save_folder": "",
            "hdf5_load_folder": "",
            "enable_compression": False,
            "freeze_containers": True,
            "theme": "light",
        }
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    loaded = json.load(f)
                    default_settings.update(loaded)
                    return default_settings
            except Exception as e:
                print(f"Error loading settings file: {e}")
        return default_settings

    def save_settings(self, new_settings):
        self.settings.update(new_settings)
        with open(self.settings_file, "w") as f:
            json.dump(self.settings, f, indent=4)
        # Trigger the update for any listening files
        self.settings_changed.emit(self.settings)

    def get(self, key, default=None):
        return self.settings.get(key, default)


# Create a single instance to be imported elsewhere
config = ConfigManager()
