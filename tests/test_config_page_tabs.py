"""The Config page keeps its file view and gains the builder tab; builder output goes through the same load path."""

import pytest

pytest.importorskip("PyQt5")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from PyQt5.QtWidgets import QMainWindow  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "relay_d/acquisition/qt_app"))  # mainWindow_ui.py does `import resources_rc`

from relay_d.acquisition.qt_app.ui.config import config_page as cp  # noqa: E402
from relay_d.acquisition.qt_app.ui.containers.mainWindow_ui import Ui_MainWindow  # noqa: E402


@pytest.fixture
def page(qapp, monkeypatch):
    boxes = []
    for name in ("info", "error", "warning"):
        monkeypatch.setattr(cp.CustomMessageBox, name, staticmethod(lambda *a, _n=name, **k: boxes.append((_n, a))))
    window, ui = QMainWindow(), Ui_MainWindow()
    ui.setupUi(window)
    p = cp.ConfigPage(ui)
    p.window, p.boxes = window, boxes
    return p


def test_tabs_and_load_config_path_emits_signal(page, tmp_path):
    assert [page.tabs.tabText(i) for i in range(2)] == ["Load / edit file", "Build from ROS"]
    assert page.ui.txt_config_display.parent() is not None and page.tabs.widget(0).isAncestorOf(page.ui.btn_insert_config)
    path = tmp_path / "c.yaml"
    path.write_text("config:\n  dataset_name: d\n  recording_frequency: 10\n  input: {}\n  output: {}\n")
    loaded = []
    page.signal_manager.config_loaded.connect(loaded.append)
    page.load_config_path(str(path))
    assert loaded == [str(path)] and page.yaml_parser.is_loaded()


def test_check_warnings_do_not_block_loading(page, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("config:\n  recording_frequency: 10\n  input: {}\n  output:\n    streams:\n      q: [a.x]\n")
    loaded = []
    page.signal_manager.config_loaded.connect(loaded.append)
    page.load_config_path(str(path))
    assert loaded and any(kind == "warning" and "not defined" in args[0] for kind, args in page.boxes)
