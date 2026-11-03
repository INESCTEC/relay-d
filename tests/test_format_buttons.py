"""Post-process page: the output-format buttons behave like a single-choice toggle."""

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QMainWindow  # noqa: E402

from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess import postprocess_page as pp  # noqa: E402


@pytest.fixture
def page(monkeypatch, qapp):
    # resources_rc (icons) is imported by the generated UI module via a top-level name
    import os
    import relay_d.acquisition.qt_app as qt_app

    monkeypatch.syspath_prepend(os.path.dirname(qt_app.__file__))
    from relay_d.acquisition.qt_app.ui.containers.mainWindow_ui import Ui_MainWindow

    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    p = pp.PostProcessPage(ui, YamlParser())
    p._window = window  # keep alive
    yield p, ui, qapp


def _state(page, ui):
    return (page.output_format or "-", ui.btn_format_robomimic.isChecked(), ui.btn_format_lerobot.isChecked())


def test_only_one_format_button_is_checked_at_a_time(page):
    p, ui, _ = page
    assert _state(p, ui) == ("robomimic", True, False)

    ui.btn_format_lerobot.click()
    assert _state(p, ui) == ("lerobot", False, True)

    ui.btn_format_robomimic.click()
    assert _state(p, ui) == ("robomimic", True, False)

    ui.btn_format_lerobot.click()
    ui.btn_format_lerobot.click()  # clicking the selected format again deselects it
    assert _state(p, ui) == ("-", False, False)

    ui.btn_format_robomimic.click()
    assert _state(p, ui) == ("robomimic", True, False)


def test_preview_follows_the_selected_format(page):
    p, ui, _ = page
    ui.btn_format_lerobot.click()
    assert "lerobot" in ui.txt_output_preview.toPlainText().lower()
    ui.btn_format_robomimic.click()
    assert "robomimic" in ui.txt_output_preview.toPlainText().lower()
    ui.btn_format_robomimic.click()  # deselect clears the preview
    assert ui.txt_output_preview.toPlainText() == ""
