import os

from PyQt5.QtCore import QObject, pyqtSignal

from relay_d.utils import user_settings


class ConfigManager(QObject):
    """Qt wrapper around ``relay_d.utils.user_settings`` (the store the command-line tools read too).

    It adds the ``settings_changed`` signal: listeners get the full settings dict after every save.
    """

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
        self.settings = user_settings.reload()
        self._initialized = True

    @property
    def settings_file(self):
        return str(user_settings.settings_file())

    def save_settings(self, new_settings):
        self.settings = user_settings.save(new_settings)
        self.settings_changed.emit(self.settings)

    def get(self, key, default=None):
        return user_settings.get(key, default)  # one store: same value the command-line tools read

    def config_dialog_dir(self):
        """Start folder for config dialogs: config_folder, else hdf5_load_folder, else ''."""
        for key in ("config_folder", "hdf5_load_folder"):
            folder = user_settings.get(key)
            if folder and os.path.isdir(folder):
                return folder
        return ""


# Create a single instance to be imported elsewhere
config = ConfigManager()
