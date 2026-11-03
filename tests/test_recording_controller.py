"""RecordingController (start/pause/resume/stop + buttons) and ContainerManager's non-blocking wait.

Qt runs offscreen; the ROS-facing objects are mocks. Skipped when PyQt5 is not installed.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QPushButton  # noqa: E402

try:  # rclpy is not available without a sourced ROS 2 environment
    importlib.import_module("relay_d.utils.tf_topic")
except ImportError:
    fake = types.ModuleType("relay_d.utils.tf_topic")

    class TfTopicSubscriptionError(Exception):
        pass

    fake.TfTopicSubscriptionError = TfTopicSubscriptionError
    sys.modules["relay_d.utils.tf_topic"] = fake

from relay_d.acquisition.qt_app.ui.record import container_manager as cm_mod  # noqa: E402
from relay_d.acquisition.qt_app.ui.record import recording_controller as rc_mod  # noqa: E402
from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus  # noqa: E402
from relay_d.utils.tf_topic import TfTopicSubscriptionError  # noqa: E402


@pytest.fixture
def env(qapp, monkeypatch):
    ui = SimpleNamespace(
        btn_start_record=QPushButton("Start"),
        btn_pause_record=QPushButton("Pause"),
        btn_stop_record=QPushButton("Stop"),
    )
    yaml_parser = MagicMock()
    yaml_parser.is_loaded.return_value = True
    yaml_parser.get_dataset_name.return_value = "my_demo"
    recorder = MagicMock()
    recorder.start_recording.return_value = True
    recorder.pause_recording.return_value = True
    recorder.resume_recording.return_value = True
    recorder.stop_recording.return_value = True
    recorder.h5_file_path = "/data/my_demo_1.h5"
    recorder.retried_file = None
    subscribers = MagicMock()
    subscribers.create_subscribers_for_containers.return_value = {"a": True}
    containers = MagicMock()
    containers.containers = [object()]
    manager = MagicMock()
    validator = MagicMock()
    validator.validate.return_value = SimpleNamespace(
        overall_status=CheckStatus.PASS, summary_text=lambda: "summary"
    )
    box = MagicMock()
    monkeypatch.setattr(rc_mod, "CustomMessageBox", box)
    monkeypatch.setattr(rc_mod, "run_in_background", _run_now)  # the save thread, made synchronous
    ctrl = rc_mod.RecordingController(ui, yaml_parser, recorder, subscribers, containers, manager, validator)
    ctrl.connect_buttons()
    return SimpleNamespace(ui=ui, ctrl=ctrl, recorder=recorder, subscribers=subscribers,
                           manager=manager, validator=validator, box=box, yaml_parser=yaml_parser)


def _run_now(parent, work, on_done, busy=None):
    try:
        value, error = work(), None
    except Exception as e:
        value, error = None, e
    on_done(value, error)


def _states(ui):
    return (
        ui.btn_start_record.isEnabled(),
        ui.btn_pause_record.isEnabled(), ui.btn_pause_record.text(),
        ui.btn_stop_record.isEnabled(),
    )


def test_initial_and_config_loaded_button_states(env):
    assert _states(env.ui) == (False, False, "Pause", False)
    env.ctrl.set_config_loaded(True)
    assert _states(env.ui) == (True, False, "Pause", False)
    env.ctrl.set_config_loaded(False)
    assert _states(env.ui) == (False, False, "Pause", False)


def test_full_cycle_through_button_clicks(env):
    env.ctrl.set_config_loaded(True)
    env.ui.btn_start_record.click()
    assert env.ctrl.is_recording and not env.ctrl.is_paused
    assert env.recorder.start_recording.call_args[0][1].startswith("my_demo_")
    assert _states(env.ui) == (False, True, "Pause", True)

    env.ui.btn_pause_record.click()
    assert env.ctrl.is_paused and env.ctrl.is_recording
    env.recorder.pause_recording.assert_called_once()
    env.manager.set_status_for_all.assert_called_with("Paused")
    assert _states(env.ui) == (False, True, "Resume", True)

    env.ui.btn_pause_record.click()  # same slot, now resumes
    assert not env.ctrl.is_paused
    env.recorder.resume_recording.assert_called_once()
    env.recorder.pause_recording.assert_called_once()  # not called a second time
    assert _states(env.ui) == (False, True, "Pause", True)

    env.ui.btn_pause_record.click()
    env.ui.btn_stop_record.click()  # stop while paused resets the label
    assert not env.ctrl.is_recording and not env.ctrl.is_paused
    env.recorder.stop_recording.assert_called_once()
    env.subscribers.clear_all_subscribers.assert_called_once()
    env.validator.validate.assert_called_once_with("/data/my_demo_1.h5")
    assert env.ctrl.session_h5_files == ["/data/my_demo_1.h5"]
    assert _states(env.ui) == (True, False, "Pause", False)

    env.ui.btn_start_record.click()  # a second take works
    assert env.ctrl.is_recording
    assert env.recorder.start_recording.call_count == 2


def test_guards(env):
    assert env.ctrl.stop_recording() is False  # nothing to stop
    assert env.ctrl.pause_recording() is False
    env.yaml_parser.is_loaded.return_value = False
    assert env.ctrl.start_recording() is False
    env.recorder.start_recording.assert_not_called()
    assert "Load a config first" in env.box.warning.call_args.args[0]  # the user is told why nothing happens
    env.yaml_parser.is_loaded.return_value = True
    assert env.ctrl.start_recording() is True
    assert env.ctrl.start_recording() is False  # already recording
    assert env.recorder.start_recording.call_count == 1


def test_start_failure_shows_error_and_keeps_state(env):
    env.recorder.start_recording.return_value = False
    assert env.ctrl.start_recording() is False
    assert not env.ctrl.is_recording
    env.box.error.assert_called_once()
    assert env.box.error.call_args[1]["title"] == "Recording Error"


def test_tf_topic_error_is_reported_and_blocks_recording(env):
    env.subscribers.create_subscribers_for_containers.side_effect = TfTopicSubscriptionError("/bad/tf")
    assert env.ctrl.start_recording() is False
    assert not env.ctrl.is_recording
    env.recorder.start_recording.assert_not_called()
    assert env.box.error.call_args[1]["title"] == "TF Topic Error"
    assert "/bad/tf" in env.box.error.call_args[0][0]


def test_failed_validation_warns_and_validator_errors_do_not_break_stop(env):
    env.ctrl.start_recording()
    env.validator.validate.return_value = SimpleNamespace(
        overall_status=CheckStatus.FAIL, summary_text=lambda: "bad demo"
    )
    assert env.ctrl.stop_recording() is True
    env.box.warning.assert_called_once()
    assert "bad demo" in env.box.warning.call_args[0][0]

    env.ctrl.start_recording()
    env.validator.validate.side_effect = RuntimeError("boom")
    assert env.ctrl.stop_recording() is True
    assert len(env.ctrl.session_h5_files) == 2


def test_stop_failure_keeps_recording(env):
    env.ctrl.start_recording()
    env.recorder.stop_recording.return_value = False
    env.ctrl.stop_recording()
    assert env.ctrl.is_recording
    assert env.ctrl.session_h5_files == []


def test_save_runs_in_the_background_with_the_buttons_locked(env, monkeypatch):
    pending = []
    monkeypatch.setattr(rc_mod, "run_in_background", lambda parent, work, on_done, busy=None: pending.append((work, on_done)))
    env.ctrl.set_config_loaded(True)
    env.ctrl.start_recording()
    assert env.ctrl.stop_recording() is True
    assert _states(env.ui) == (False, False, "Pause", False)  # nothing to click while the demo is written
    assert env.ctrl.stop_recording() is False  # a second Stop does not start a second save
    work, on_done = pending.pop()
    on_done(work(), None)  # the thread finished
    assert not env.ctrl.is_recording and env.ctrl.session_h5_files == ["/data/my_demo_1.h5"]
    assert _states(env.ui) == (True, False, "Pause", False)


def test_a_crash_in_the_save_thread_is_a_failed_stop(env):
    env.ctrl.start_recording()
    env.recorder.stop_recording.side_effect = OSError("disk")
    env.ctrl.stop_recording()
    assert env.ctrl.session_h5_files == [] and not env.ctrl.is_saving


def test_remove_session_file(env):
    env.ctrl.session_h5_files.extend(["a.h5", "b.h5"])
    env.ctrl.remove_session_file("a.h5")
    env.ctrl.remove_session_file("missing.h5")
    assert env.ctrl.session_h5_files == ["b.h5"]


def test_pump_events_keeps_processing_events(qapp):
    fired = []
    QTimer.singleShot(10, lambda: fired.append(1))
    cm_mod.pump_events(80)
    assert fired == [1]  # the queued timer ran while "waiting"


def test_create_containers_is_not_reentrant_while_waiting(qapp):
    yaml_parser = MagicMock()
    yaml_parser.get_input_data.return_value = []
    manager = cm_mod.ContainerManager(MagicMock(), yaml_parser, MagicMock(), MagicMock(), MagicMock())
    results = []
    QTimer.singleShot(20, lambda: results.append(manager.create_containers()))  # a second click
    assert manager.create_containers() is True
    assert results == [False]
    assert manager.create_containers() is True  # usable again afterwards


def test_demo_saved_by_a_retry_at_start_joins_the_session(env):
    env.ctrl.set_config_loaded(True)
    env.recorder.retried_file = "/data/my_demo_0.h5"  # its save failed at Stop, the retry at Start worked
    env.ui.btn_start_record.click()
    env.ui.btn_stop_record.click()
    assert env.ctrl.session_h5_files == ["/data/my_demo_0.h5", "/data/my_demo_1.h5"]
