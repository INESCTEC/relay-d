"""Train tab: dataset detection, parameter form, start button (no real training, no ROS)."""

import stat
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QWidget  # noqa: E402

from relay_d.acquisition.qt_app.ui.train import train_page as tp  # noqa: E402
from relay_d.training import runs  # noqa: E402


def _settle(page, seconds=10):
    """Let the page's background jobs finish (the GUI thread only polls them)."""
    deadline = time.time() + seconds
    while page._busy and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()


@pytest.fixture
def page(qapp, tmp_path, monkeypatch):
    with h5py.File(tmp_path / "robomimic_dataset_1.h5", "w") as f:
        demo = f.create_group("data/demo_0")
        demo.attrs["num_samples"] = 6
        demo.create_dataset("obs/robot0_eef_pos", data=np.zeros((6, 3), np.float32))
        f.create_dataset("mask/train", data=[b"demo_0"])
        f.create_dataset("mask/valid", data=[b"demo_0"])
    monkeypatch.delenv("DISPLAY", raising=False)  # headless runs, no terminal window
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(tp.config, "save_settings", lambda *_: None)  # do not touch ~/.relayd_settings.json
    monkeypatch.setattr(tp.config, "get", lambda key, default=None: default)
    boxes = []
    for name in ("info", "error", "warning"):
        monkeypatch.setattr(tp.CustomMessageBox, name, staticmethod(lambda *a, _n=name, **k: boxes.append((_n, a))))
    ui = SimpleNamespace(page_train=QWidget())
    p = tp.TrainPage(ui)
    p.boxes, p.folder_path = boxes, tmp_path
    yield p
    p._timer.stop()


def test_loading_a_folder_fills_policy_and_parameters(page):
    page.load_folder(str(page.folder_path))
    assert page.combo_policy.itemText(0) == tp._FROM_FOLDER
    assert {"epochs", "batch_size", "learning_rate", "device"} <= set(page._param_widgets)
    assert page.btn_start.isEnabled() and "1 episodes" in page.label_summary.text()
    page.combo_policy.setCurrentText("diffusion_policy")  # another template: the form is rebuilt
    assert page._policy() == "diffusion_policy" and "epochs" in page._param_widgets


def test_empty_folder_disables_start(page, tmp_path_factory):
    page.load_folder(str(tmp_path_factory.mktemp("empty")))
    assert not page.btn_start.isEnabled() and "No robomimic file" in page.label_summary.text()


def test_start_launches_a_run_with_the_edited_values(page, tmp_path, monkeypatch):
    py = tmp_path / "fakepython"
    py.write_text("#!/bin/sh\necho ok\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    page.load_folder(str(page.folder_path))
    page.combo_policy.setCurrentText("bc_rnn_gmm")  # rebuilds the form, so set the values after
    page._param_widgets["epochs"][0].setValue(3)
    page.edit_python.setText(str(py))
    monkeypatch.setattr(tp.launcher, "check_environment", lambda *a, **k: {"ok": True, "torch": "2", "cuda": False})
    page.start_training()
    _settle(page)
    run = runs.list_runs(page.folder_path)[0]
    assert '"num_epochs": 3' in (run.path / "robomimic_config.json").read_text()
    deadline = time.time() + 10
    while runs.read_run(run.path).status != runs.FINISHED and time.time() < deadline:
        time.sleep(0.05)
    page.refresh_runs()
    assert page.table.item(0, 1).text() == runs.FINISHED
    assert page.boxes[0][0] == "info"  # "no terminal: running in the background"


def test_unusable_environment_blocks_start(page, monkeypatch):
    page.load_folder(str(page.folder_path))
    monkeypatch.setattr(tp.launcher, "check_environment",
                        lambda *a, **k: {"ok": False, "error": "'robomimic' is not installed for x", "cuda": False})
    page.start_training()
    _settle(page)
    assert page.boxes and page.boxes[0][0] == "error" and runs.list_runs(page.folder_path) == []


def test_slow_work_does_not_block_the_gui_thread(page, monkeypatch):
    page.load_folder(str(page.folder_path))
    started = {}
    monkeypatch.setattr(tp.launcher, "check_environment",
                        lambda *a, **k: time.sleep(0.4) or {"ok": True, "torch": "2", "cuda": False})
    t0 = time.time()
    page.check_environment(then=lambda ok: started.update(ok=ok))
    assert time.time() - t0 < 0.2  # returned at once; the import check runs in a thread
    assert page._busy == 1 and not page.btn_start.isEnabled() and not page.btn_stop.isEnabled()
    assert "Checking" in page.label_env.text()
    _settle(page)
    assert started == {"ok": True} and page.btn_start.isEnabled() and page.btn_stop.isEnabled()
    assert page.label_env.text().startswith("OK")


def test_an_error_in_background_work_is_reported_not_lost(page):
    seen = {}

    def boom():
        raise RuntimeError("disk gone")

    page._run_in_background(boom, lambda value, error: seen.update(value=value, error=str(error)))
    _settle(page)
    assert seen == {"value": None, "error": "disk gone"} and page._busy == 0


def test_the_run_that_starts_is_the_one_that_was_clicked(page, tmp_path, monkeypatch):
    import shutil

    shutil.copy(page.folder_path / "robomimic_dataset_1.h5", page.folder_path / "robomimic_dataset_2.h5")
    py = tmp_path / "fakepython"
    py.write_text("#!/bin/sh\necho ok\n")
    py.chmod(0o755)
    page.load_folder(str(page.folder_path))
    assert page.combo_dataset.count() == 2
    clicked = page._dataset().path.name
    page.edit_python.setText(str(py))
    monkeypatch.setattr(tp.launcher, "check_environment",
                        lambda *a, **k: time.sleep(0.3) or {"ok": True, "torch": "2", "cuda": False})
    page.start_training()
    assert not page.combo_dataset.isEnabled() and not page.edit_python.isEnabled()  # locked while it checks
    page.combo_dataset.setCurrentIndex(1 - page.combo_dataset.currentIndex())  # something changes it anyway
    _settle(page)
    meta = runs.list_runs(page.folder_path)[0].meta
    assert meta["dataset"].endswith(clicked)  # the dataset at the click, not the one selected afterwards
    assert page.combo_dataset.isEnabled() and page.edit_python.isEnabled() and page.btn_start.isEnabled()


def _lerobot_dataset(page, tmp_path):
    page.datasets = [type("D", (), {"format": launcher_format(), "summary": {}, "path": tmp_path, "base": tmp_path})()]
    page.combo_dataset.clear()
    page.combo_dataset.addItem("lerobot dataset")


def test_setup_button_builds_then_uses_the_environment(page, tmp_path, monkeypatch):
    from relay_d.training import envsetup

    monkeypatch.setenv("RELAYD_ENVS_DIR", str(tmp_path / "envs"))
    saved, checked = [], []
    monkeypatch.setattr(tp.config, "save_settings", lambda values: saved.append(values))
    monkeypatch.setattr(page, "check_environment", lambda *a, **k: checked.append(1))
    _lerobot_dataset(page, tmp_path)
    page._set_busy(0)
    assert page.btn_setup_env.isEnabled() and page.btn_setup_env.text() == "Set up lerobot environment"

    env = envsetup.default_env_dir("lerobot")
    started = []
    monkeypatch.setattr(envsetup, "start", lambda fmt, env_dir, terminal: started.append((fmt, env_dir)))
    monkeypatch.setattr(tp.CustomMessageBox, "question", staticmethod(lambda *a, **k: True))
    page.setup_environment()
    _settle(page)
    assert started == [("lerobot", env)] and "lerobot" in page._pending and not page.btn_setup_env.isEnabled()

    log, exit_file = envsetup._paths(env)
    exit_file.parent.mkdir(parents=True)
    envsetup.env_python(env).parent.mkdir(parents=True)
    envsetup.env_python(env).write_text("")
    exit_file.write_text("0\n")
    page._poll_setup()  # what the timer does
    assert saved == [{"python_lerobot": str(envsetup.env_python(env))}] and not page._pending and checked

    saved.clear()
    page._set_busy(0)
    assert page.btn_setup_env.text() == "Use lerobot environment"  # ready: the button just uses it, no rebuild
    page.setup_environment()
    assert saved == [{"python_lerobot": str(envsetup.env_python(env))}] and started == [("lerobot", env)]


def launcher_format():
    from relay_d.training import launcher

    return launcher.LEROBOT


def test_the_default_python_is_not_saved_as_a_setting(page, tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(tp.config, "save_settings", lambda values: saved.append(values))
    monkeypatch.setenv("RELAYD_ENVS_DIR", str(tmp_path / "envs"))
    _lerobot_dataset(page, tmp_path)
    page.edit_python.setText(tp.sys.executable)  # what the page shows when nothing is set and nothing is built
    page._save_python()
    page.edit_python.setText("/my/venv/bin/python")
    page._save_python()
    assert saved == [{"python_lerobot": ""}, {"python_lerobot": "/my/venv/bin/python"}]
