import pytest

pytest.importorskip("PyQt5")
from relay_d.acquisition.qt_app.first_run import ask_save_folder  # noqa: E402
from relay_d.utils import user_settings  # noqa: E402


def test_asks_until_a_folder_is_chosen(settings_file, tmp_path):
    answers = iter(["", str(tmp_path)])  # cancel once (and decline to quit), then choose
    asked = []
    ok = ask_save_folder(None, choose=lambda *a: next(answers), quit_asked=lambda *a, **k: asked.append(a) or False)
    assert ok and len(asked) == 1 and user_settings.get("hdf5_save_folder") == str(tmp_path)


def test_quit_and_already_set(settings_file, tmp_path):
    assert not ask_save_folder(None, choose=lambda *a: "", quit_asked=lambda *a, **k: True)
    assert user_settings.get("hdf5_save_folder") == ""
    user_settings.save({"hdf5_save_folder": str(tmp_path)})
    assert ask_save_folder(None, choose=lambda *a: pytest.fail("must not ask"), quit_asked=lambda *a, **k: pytest.fail("no"))
