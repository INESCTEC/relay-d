"""Make `relay_d` and the sibling test modules importable without installing the package."""

import os
import sys

# Qt tests run headless
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Tests must not read or write the developer's real ~/.relayd_settings.json
import tempfile  # noqa: E402

os.environ["RELAYD_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(prefix="relayd_test_settings_"), "settings.json")

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
for path in (_ROOT, _TESTS):
    if path not in sys.path:
        sys.path.insert(0, path)

# Tests read templates such as config/*.yaml with repo-relative paths.
os.chdir(_ROOT)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole run (Qt singletons such as the settings manager are
    deleted if the application object is destroyed between tests)."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """A fresh settings file for this test (RELAYD_SETTINGS_FILE), with the in-process cache pointed at it."""
    from relay_d.utils import user_settings

    path = tmp_path / "settings.json"
    monkeypatch.setenv("RELAYD_SETTINGS_FILE", str(path))
    user_settings.reload()
    yield path
    monkeypatch.undo()
    reloaded = user_settings.reload()
    manager = sys.modules.get("relay_d.acquisition.qt_app.ui.settings.config_manager")
    if manager is not None:  # the GUI singleton keeps its own reference to the settings dict
        manager.config.settings = reloaded


# --- DataRecorder without ROS: rclpy and the message packages are replaced by stubs
import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

_STUBBED = [
    "rclpy", "rclpy.time", "rclpy.node", "sensor_msgs", "sensor_msgs.msg", "geometry_msgs",
    "geometry_msgs.msg", "std_msgs", "std_msgs.msg", "sensor_msgs_py", "relay_d.utils.ros_media_codec",
]
_MODULE = "relay_d.acquisition.qt_app.ui.record.data_recorder"
_ENCODING = "relay_d.acquisition.qt_app.ui.record.message_encoding"  # imports the message packages too


class _Node:
    def __init__(self, name):
        pass

    def get_clock(self):
        return MagicMock()


@pytest.fixture
def recorder_cls(monkeypatch):
    for name in _STUBBED:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    monkeypatch.setitem(sys.modules, "rclpy.node", node_mod)
    for name in (_MODULE, _ENCODING):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield importlib.import_module(_MODULE).DataRecorder
    for name in (_MODULE, _ENCODING):
        sys.modules.pop(name, None)
