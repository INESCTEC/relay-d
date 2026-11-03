"""AppAPI session handling. Needs a sourced ROS 2 (rclpy): skipped without it, run by the "Tests (ROS 2 Jazzy)" CI job."""

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from relay_d.acquisition.api.api import AppAPI  # noqa: E402


def test_shutdown_saves_a_running_recording_before_converting():
    api = AppAPI.__new__(AppAPI)  # skip ROS setup
    calls = []
    api._data_recorder = SimpleNamespace(is_recording=True)
    api.stop_recording = lambda: calls.append("stop")
    api.convert_recording_to_dataset = lambda: calls.append("convert")
    api._spin_thread, api._rclpy_initialized = None, False
    api._topic_subscribers = SimpleNamespace(close=lambda: None)
    api.shutdown()
    assert calls == ["stop", "convert"]  # the demo was only in memory


def test_demo_saved_by_a_retry_at_start_joins_the_session():
    api = AppAPI.__new__(AppAPI)
    api._session_h5_files = []
    api._topic_subscribers = None
    api._data_recorder = SimpleNamespace(start_recording=lambda *a, **k: True, retried_file="/data/demo_0.h5")
    assert api.start_recording("demo_1")
    assert api._session_h5_files == ["/data/demo_0.h5"]  # was never converted


def test_shutdown_keeps_running_when_the_recording_could_not_be_saved():
    api = AppAPI.__new__(AppAPI)
    calls = []
    api._data_recorder = SimpleNamespace(is_recording=True, _unsaved=True, retry_save=lambda: False, h5_file_path="/full/disk.h5")
    api.stop_recording = lambda: calls.append("stop")
    api.convert_recording_to_dataset = lambda: calls.append("convert")
    api._spin_thread, api._rclpy_initialized = None, False
    api._topic_subscribers = SimpleNamespace(close=lambda: calls.append("close"))
    assert api.shutdown() is False
    assert calls == ["stop"]  # no convert, no close: the demo is still only in memory


@pytest.mark.parametrize("fmt, name", [("robomimic", "combined_dataset_Robomimic.h5"), ("lerobot", "combined_dataset_Lerobot")])
def test_session_dataset_is_named_after_the_format(fmt, name):
    api = AppAPI.__new__(AppAPI)
    seen = []
    api._session_h5_files = ["/rec/2026-11-02_15-00-00_Recording/d.h5"]
    api._data_recorder = SimpleNamespace(recorded_data_dir="/rec/2026-11-02_15-00-00_Recording")
    api._yaml_converter = SimpleNamespace(convert_multiple_files=lambda f, files, out: seen.append((f, out)) or True)
    assert api.convert_recording_to_dataset(fmt)
    assert seen == [(fmt, f"/rec/2026-11-02_15-00-00_Recording/{name}")]
