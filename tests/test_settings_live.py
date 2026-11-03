"""A setting saved while the app runs is what the pages, recorder and API use next."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QWidget  # noqa: E402

from relay_d.acquisition.qt_app.ui.settings.config_manager import config  # noqa: E402
from relay_d.acquisition.qt_app.ui.settings.settings_page import SettingsPage  # noqa: E402
from relay_d.utils import user_settings  # noqa: E402


def test_one_store_and_other_pages_saves_show_in_settings(qapp, settings_file):
    page = SettingsPage(SimpleNamespace(page_settings=QWidget()))
    config.save_settings({"python_lerobot": "/venv/bin/python"})  # saved by the Train page, not by Settings
    assert config.get("python_lerobot") == user_settings.get("python_lerobot") == "/venv/bin/python"
    if "python_lerobot" in page.widgets:
        assert page.widgets["python_lerobot"][1]() == "/venv/bin/python"
    page.widgets["rec_default_frequency"][2](42)  # an unsaved edit elsewhere survives another page's save
    config.save_settings({"review_default_fixed_frame": "base_link"})
    assert page.widgets["rec_default_frequency"][1]() == 42
    assert page.widgets["review_default_fixed_frame"][1]() == "base_link"


def test_recorder_uses_the_current_defaults_at_each_start(recorder_cls, settings_file):
    rec = recorder_cls(SimpleNamespace(is_loaded=lambda: False, yaml_data=None))
    user_settings.save({"rec_default_frequency": 17, "rec_default_max_buffer_mb": 64})
    rec._apply_setting_defaults()  # what start_recording does first
    assert rec.recording_frequency == 17 and rec.max_buffer_bytes == 64 * 1024**2


def test_save_button_saves_the_values_on_screen(qapp, settings_file):
    page = SettingsPage(SimpleNamespace(page_settings=QWidget()))
    page.widgets["rec_default_frequency"][2](25)
    page.btn_save_settings.click()  # `clicked` passes `checked`; it must not be taken as the keys to save
    assert user_settings.get("rec_default_frequency") == 25
