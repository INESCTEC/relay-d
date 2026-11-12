"""AppAPI session handling. Needs a sourced ROS 2 (rclpy): skipped without it, run by the "Tests (ROS 2 Jazzy)" CI job."""

import copy
import logging
import re
from types import SimpleNamespace

import pytest
import yaml

from helpers import DATA_CONFIG

pytest.importorskip("rclpy")
from relay_d.acquisition.api.api import AppAPI  # noqa: E402
from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser  # noqa: E402
from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus  # noqa: E402
from relay_d.utils.tf_topic import TfTopicSubscriptionError  # noqa: E402


def test_shutdown_saves_a_running_recording_before_converting():
    api = AppAPI.__new__(AppAPI)  # skip ROS setup
    calls = []
    api._data_recorder = SimpleNamespace(is_recording=True)
    api.stop_recording = lambda: calls.append("stop")
    api.convert_recording_to_dataset = lambda *a: calls.append("convert")
    api._rclpy_initialized = False
    api._topic_subscribers = SimpleNamespace(close=lambda: None)
    api.shutdown()
    assert calls == ["stop", "convert"]  # the demo was only in memory


def test_demo_saved_by_a_retry_at_start_joins_the_session():
    api = _recording_api()
    api._data_recorder.retried_file = "/data/demo_0.h5"
    assert api.start_recording("demo_1")
    assert api._session_h5_files == ["/data/demo_0.h5"]  # was never converted


def test_shutdown_keeps_running_when_the_recording_could_not_be_saved():
    api = AppAPI.__new__(AppAPI)
    calls = []
    api._data_recorder = SimpleNamespace(is_recording=True, _unsaved=True, retry_save=lambda: False, h5_file_path="/full/disk.h5")
    api.stop_recording = lambda: calls.append("stop")
    api.convert_recording_to_dataset = lambda *a: calls.append("convert")
    api._rclpy_initialized = False
    api._topic_subscribers = SimpleNamespace(close=lambda: calls.append("close"))
    assert api.shutdown() is False
    assert calls == ["stop"]  # no convert, no close: the demo is still only in memory


def test_api_subscribes_tf_inputs_on_their_configured_tf_topic():
    tf_input = {"name": "ee", "is_tf": True, "parent_frame": "base", "child_frame": "tool",
                "tf_topic": "/robot/tf", "tf_static_topic": "/robot/tf_static", "tf_wait_timeout": 7.0}
    api = _recording_api(inputs=[tf_input])
    assert api.initialize("cfg.yaml")
    assert api.start_recording()
    c = api._topic_subscribers.subscribed[0][0]
    assert (c.tf_topic, c.tf_static_topic, c.tf_wait_timeout) == ("/robot/tf", "/robot/tf_static", 7.0)  # was /tf


# --- recording: the same steps as the GUI's Start / Stop


class _Subscribers:
    """TopicSubscribers stand-in: records what AppAPI asks of it."""

    def __init__(self, unavailable=(), fail=None):
        self.subscribed, self.cleared, self.unavailable, self.fail = [], 0, list(unavailable), fail

    def unavailable_inputs(self, items):
        return self.unavailable

    def create_subscriber(self, c):  # the old initialize() subscribed here once
        self.subscribed.append([c])

    def create_subscribers_for_containers(self, containers):
        if self.fail:
            raise self.fail
        self.subscribed.append(list(containers))
        return {c.container_id: True for c in containers}

    def clear_all_subscribers(self):
        self.cleared += 1

    def start_spinning(self):
        pass


def _recording_api(inputs=({"name": "cam", "topic": "/cam"},), subscribers=None, parser=None):
    api = AppAPI.__new__(AppAPI)  # skip ROS setup
    api._session_h5_files, api._containers = [], []
    api._topic_subscribers = subscribers or _Subscribers()
    api._yaml_parser = parser or SimpleNamespace(
        load_yaml_file=lambda p: True, get_input_data=lambda: list(inputs), get_dataset_name=lambda: "stack",
        yaml_data={"config": {}}, get_streams=dict, clear=lambda: None,
    )
    api.start_spinning = lambda: None
    names = []
    api._data_recorder = SimpleNamespace(
        start_recording=lambda subs, name=None: names.append(name) or True, retried_file=None, names=names,
        is_recording=False, stop_recording=lambda: True, h5_file_path=None,
    )
    return api


def test_every_start_subscribes_every_input_and_names_the_demo_like_the_gui():
    api = _recording_api()
    assert api.initialize("cfg.yaml")
    assert api.start_recording() and api.start_recording()
    assert [[c.container_id for c in cs] for cs in api._topic_subscribers.subscribed] == [["cam"], ["cam"]]  # was once, at load
    assert all(re.fullmatch(r"stack_\d{13}", n) for n in api._data_recorder.names)  # was demo_0, demo_1


def test_start_is_refused_when_a_configured_tf_topic_cannot_be_read():
    api = _recording_api(subscribers=_Subscribers(fail=TfTopicSubscriptionError("/robot/tf")))
    assert api.initialize("cfg.yaml")
    assert api.start_recording() is False
    assert api._data_recorder.names == []  # as the GUI: no silent fallback to /tf


def test_stop_clears_the_subscribers_like_the_gui():
    api = _recording_api()
    api.initialize("cfg.yaml")
    api.start_recording()
    api.stop_recording()
    assert api._topic_subscribers.cleared == 1


def test_initialize_refuses_an_input_that_is_not_available():
    api = _recording_api(subscribers=_Subscribers(unavailable=["  cam: topic /cam not found"]))
    assert api.initialize("cfg.yaml") is False


def test_initialize_refuses_a_stream_spec_that_does_not_parse(tmp_path):
    config = copy.deepcopy(DATA_CONFIG)
    config["output"]["streams"]["robot_joint_pos"] = ["joint_1.position[:] + "]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"config": config}))
    api = _recording_api(parser=YamlParser())
    assert api.initialize(str(path)) is False
    assert not api._yaml_parser.is_loaded()


def test_the_after_recording_check_is_logged(caplog):
    api = _recording_api()
    api._data_recorder.h5_file_path = "/rec/stack_1.h5"
    api._validator = SimpleNamespace(validate=lambda p: SimpleNamespace(checks=[
        SimpleNamespace(status=CheckStatus.PASS, name="ok", message="fine"),
        SimpleNamespace(status=CheckStatus.WARN, name="rate", message="9.1 Hz"),
        SimpleNamespace(status=CheckStatus.FAIL, name="samples", message="no samples"),
    ]))
    relay_logger = logging.getLogger("relay_d")  # does not propagate to the root logger
    relay_logger.addHandler(caplog.handler)
    try:
        assert api.stop_recording()
    finally:
        relay_logger.removeHandler(caplog.handler)
    lines = [(r.levelname, r.getMessage()) for r in caplog.records if "Demo check" in r.getMessage()]
    assert lines == [("WARNING", "Demo check [WARN] rate: 9.1 Hz"), ("ERROR", "Demo check [FAIL] samples: no samples")]


def test_delete_demo_removes_the_file_and_drops_it_from_the_session(tmp_path):
    api = _recording_api()
    kept, rejected = tmp_path / "stack_1.h5", tmp_path / "stack_2.h5"
    kept.write_bytes(b"x")
    rejected.write_bytes(b"x")
    api._session_h5_files = [str(kept), str(rejected)]
    assert api.delete_demo(str(rejected))
    assert not rejected.exists() and kept.exists()
    assert api._session_h5_files == [str(kept)]  # shutdown() no longer converts a missing file
    assert api.delete_demo(str(rejected)) is False  # already gone: nothing changes
    assert api._session_h5_files == [str(kept)]
