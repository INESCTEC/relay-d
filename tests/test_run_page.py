"""Run tab: command line, and Start / Stop on a stand-in process (no policy, no ROS)."""

import shutil
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QWidget  # noqa: E402

from relay_d.acquisition.qt_app.ui.run import run_page as rp  # noqa: E402

# Prints until SIGINT, then exits 0 like the node's Ctrl+C path.
STAND_IN = ("import time\ntry:\n    while True:\n        print('tick', flush=True)\n        time.sleep(0.05)\n"
            "except KeyboardInterrupt:\n    print('safe stop sent', flush=True)\n")


def _wait(cond, seconds=10):
    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    return cond()


@pytest.fixture
def page(qapp, tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(rp.config, "save_settings", saved.append)  # do not touch ~/.relayd_settings.json
    disp = tmp_path / "out" / "action_dispatcher"
    disp.mkdir(parents=True)
    ckpt = tmp_path / "model.pth"
    ckpt.write_text("")
    p = rp.RunPage(SimpleNamespace(page_run=QWidget()))
    p.saved, p.ckpt, p.disp = saved, str(ckpt), disp
    yield p
    p.shutdown()


def test_args_dry_run_and_hz():
    args = rp.build_args("/c.pth", "/d", "cpu", "", True)
    assert args[:2] == ["-m", "relay_d.dispatch.inference_node"]
    assert "--dry-run" in args and "--hz" not in args
    assert args[args.index("--obs-config") + 1] == "/d/obs_config.yaml"
    args = rp.build_args("/c.pth", "/d", "cuda", " 30 ", False)
    assert "--dry-run" not in args and args[args.index("--hz") + 1] == "30"


def test_start_needs_both_dispatch_files_and_postprocess_folder_maps_to_subfolder(page):
    page.edit_checkpoint.setText(page.ckpt)
    assert rp.dispatch_dir_for(str(page.disp.parent)) == str(page.disp)
    page.edit_dispatch.setText(str(page.disp))
    assert not page.btn_start.isEnabled() and "obs_config.yaml and action_config.yaml" in page.label_reason.text()
    (page.disp / "obs_config.yaml").write_text("")
    (page.disp / "action_config.yaml").write_text("")
    page._update_buttons()
    assert page.btn_start.isEnabled() and page.label_reason.text() == ""
    assert page.check_dry_run.isChecked()
    assert "--dry-run" in page.command()[1]


def test_start_then_stop_sends_sigint_and_exits_cleanly(page, monkeypatch):
    page.edit_checkpoint.setText(page.ckpt)
    monkeypatch.setattr(page, "command", lambda: (sys.executable, ["-c", STAND_IN]))
    page.start()
    assert page.saved and page.label_state.text() == "Running"
    assert not page.btn_start.isEnabled() and page.btn_stop.isEnabled()
    assert _wait(lambda: "tick" in page.log.toPlainText().splitlines())  # the child is up (its handler is set)
    page.stop()
    assert _wait(lambda: not page.is_running())
    assert page.proc.exitCode() == 0 and page.label_state.text() == "Stopped"
    assert "safe stop sent" in page.log.toPlainText() and not page.btn_stop.isEnabled()


GROUP_PROBE = ("import os, time\nprint('pid', os.getpid(), os.getpgid(0) == os.getpid(), flush=True)\n" + STAND_IN)


@pytest.mark.skipif(not shutil.which("setsid"), reason="no setsid (util-linux)")
def test_node_runs_in_its_own_process_group_and_stop_still_reaches_it(page, monkeypatch):
    page.edit_checkpoint.setText(page.ckpt)
    monkeypatch.setattr(page, "command", lambda: (sys.executable, ["-c", GROUP_PROBE]))
    page.start()
    assert _wait(lambda: "tick" in page.log.toPlainText().splitlines())
    # a terminal Ctrl+C reaches only the GUI (which stops the node once), and the PID is the node's, not setsid's
    assert f"pid {page.proc.processId()} True" in page.log.toPlainText()
    page.stop()
    assert _wait(lambda: not page.is_running())
    assert page.proc.exitCode() == 0 and "safe stop sent" in page.log.toPlainText()
