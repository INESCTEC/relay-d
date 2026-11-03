"""Settings tab: built from the table, saves to the store, resets, reports restart-only changes."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QWidget  # noqa: E402

from relay_d.acquisition.qt_app.ui.settings import settings_page as sp  # noqa: E402
from relay_d.acquisition.qt_app.ui.settings.config_manager import config  # noqa: E402
from relay_d.utils import user_settings as us  # noqa: E402


@pytest.fixture
def page(qapp, settings_file, monkeypatch):
    config.settings = us.reload()
    asked = []
    monkeypatch.setattr(sp.CustomMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a) or True))
    monkeypatch.setattr(sp.CustomMessageBox, "error", staticmethod(lambda *a, **k: asked.append(("error", a))))
    p = sp.SettingsPage(SimpleNamespace(page_settings=QWidget()))
    p.path, p.asked = settings_file, asked
    return p


def test_every_setting_has_an_editor_and_shows_its_value(page):
    assert set(page.widgets) == {s.key for s in us.SETTINGS}
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == list(us.GROUPS)
    assert page.values_from_ui() == us.defaults()


def test_save_writes_the_file_emits_and_notes_restart_only_changes(page):
    seen = []
    config.settings_changed.connect(seen.append)
    page.widgets["rec_default_frequency"][2](45)
    page.widgets["theme"][2]("dark")
    page.widgets["ros_domain_id"][2](7)
    assert page.save_settings_from_ui() is True
    assert json.loads(page.path.read_text())["rec_default_frequency"] == 45
    assert us.get("theme") == "dark" and config.get("theme") == "dark"
    assert seen and seen[-1]["theme"] == "dark"
    assert "Restart" in page.label_status.text() and "ROS domain" in page.label_status.text()
    config.settings_changed.disconnect(seen.append)


def test_live_only_changes_do_not_ask_for_a_restart(page):
    page.widgets["theme"][2]("dark")
    page.save_settings_from_ui()
    assert "Restart" not in page.label_status.text()


def test_reset_one_tab_only_resets_that_tab(page):
    page.widgets["rec_default_frequency"][2](99)
    page.widgets["infer_hz"][2](33.0)
    page.save_settings_from_ui()
    page.reset_to_default("Recording")
    assert us.get("rec_default_frequency") == 30 and us.get("infer_hz") == 33.0
    page.reset_to_default(None)
    assert us.get("infer_hz") == 10.0


def test_load_settings_shows_saved_values_and_notifies(page):
    us.save({"log_level": "DEBUG", "hdf5_save_folder": "/data/rec"})
    seen = []
    config.settings_changed.connect(seen.append)
    page.load_settings()
    assert page.widgets["log_level"][1]() == "DEBUG" and page.widgets["hdf5_save_folder"][1]() == "/data/rec"
    assert page.get_hdf5_save_folder() == "/data/rec" and len(seen) == 1
    config.settings_changed.disconnect(seen.append)


def test_saving_unchanged_settings_keeps_the_file_free_of_defaults(page):
    page.save_settings_from_ui()
    assert not page.path.exists() or json.loads(page.path.read_text()) == {}
    page.widgets["rec_default_frequency"][2](45)
    page.save_settings_from_ui()
    assert json.loads(page.path.read_text()) == {"rec_default_frequency": 45}


def test_resetting_one_tab_neither_saves_nor_discards_edits_on_other_tabs(page):
    page.widgets["rec_default_frequency"][2](60)  # an edit on the Recording tab, not saved yet
    page.widgets["infer_hz"][2](33.0)
    page.save_settings_from_ui(keys=["infer_hz"])
    page.reset_to_default("Inference")
    assert us.get("infer_hz") == 10.0
    assert us.get("rec_default_frequency") == 30  # was saved by the old reset
    assert page.widgets["rec_default_frequency"][1]() == 60  # and the edit is still on screen
