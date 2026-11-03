"""DataRecorder's real sampling loop, stop/save path and image decoding, plus recording/inference stream parity."""

import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from helpers import joint_sample, make_recorder

_HDR = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id="")


def _real_message_classes(recorder_cls):
    """The encoder's message classes are MagicMocks under recorder_cls; isinstance() needs real classes."""
    module = sys.modules[recorder_cls.__module__.replace(".data_recorder", ".message_encoding")]
    for name in ("Image", "CompressedImage", "PointCloud2", "PoseStamped", "Twist",
                 "TransformStamped", "String", "Float32", "Int32", "Bool"):
        setattr(module, name, type(name, (), {}))
    module.JointState = type("JointState", (), {"name": [], "position": [], "velocity": [], "effort": [], "header": _HDR})
    module.Float64 = type("Float64", (), {"data": 0.0})
    return module


class _FakeSubscribers:
    """Feeds one dict of messages per tick to the loop, then stops it."""

    def __init__(self, rec, frames, classes):
        self.rec, self.frames, self.ticks = rec, frames, 0
        self.subscribers = dict.fromkeys(classes)
        self.subscriber_status = {cid: {"topic": "/" + cid} for cid in classes}
        self.message_type_cache = {"/" + cid: cls for cid, cls in classes.items()}
        self.tf_lookup_containers = {}

    def update_tf_lookups(self):
        pass

    def get_content(self):
        return dict.fromkeys(self.subscribers)

    def get_content_and_timestamps(self):
        frame = self.frames[min(self.ticks, len(self.frames) - 1)]
        self.ticks += 1
        if self.ticks >= len(self.frames):
            self.rec.stop_recording_flag = True
        return dict(frame), {}


def _loop_ready(recorder_cls, monkeypatch, parser=None):
    module = sys.modules[recorder_cls.__module__]
    get = module.user_settings.get
    monkeypatch.setattr(module.user_settings, "get", lambda k: 0 if k == "rec_warmup_s" else get(k))
    rec = recorder_cls(parser or SimpleNamespace(is_loaded=lambda: False, yaml_data=None))
    rec._clock = SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=1, nanosec=0)))
    rec.recording_frequency = 1000
    rec._last_valid_raw = {}
    rec.stop_recording_flag = False
    return rec


def test_a_failing_loop_sets_loop_failed(recorder_cls, monkeypatch):
    rec = _loop_ready(recorder_cls, monkeypatch)
    subs = _FakeSubscribers(rec, [{}], {})
    subs.get_content_and_timestamps = MagicMock(side_effect=RuntimeError("broken"))
    rec.topic_subscribers = subs
    rec._standard_recording_loop(subs)  # returns after 50 errors in a row
    assert subs.get_content_and_timestamps.call_count == 50
    assert rec.loop_failed is True


def _config_parser(config):
    return SimpleNamespace(is_loaded=lambda: True, yaml_data=config, get_current_config_path=lambda: "c.yaml",
                           get_metadata=lambda: {}, get_input_data=lambda: [], get_streams=lambda: {})


@pytest.mark.parametrize("recording_frequency", [None, 0])
def test_missing_or_zero_recording_frequency_is_refused_at_start(recorder_cls, monkeypatch, tmp_path, recording_frequency):
    config = {"config": {"input": {"arm": {"topic": "/arm"}}}}
    if recording_frequency is not None:
        config["config"]["recording_frequency"] = recording_frequency
    rec = _loop_ready(recorder_cls, monkeypatch, _config_parser(config))
    rec.output_directory = str(tmp_path)
    assert rec.start_recording(_FakeSubscribers(rec, [{}], {})) is False  # missing: was the 30 Hz setting
    assert rec.is_recording is False


@pytest.mark.parametrize("removed", [{"output_map": "robot"}, {"fill_gaps": True, "fill_default": 0.5}])
def test_removed_keys_that_changed_the_recorded_data_are_refused_at_start(recorder_cls, monkeypatch, tmp_path, removed):
    config = {"config": {"recording_frequency": 1000, "input": {"arm": {"topic": "/arm", **removed}}}}
    rec = _loop_ready(recorder_cls, monkeypatch, _config_parser(config))
    rec.output_directory = str(tmp_path)
    errors = []
    monkeypatch.setattr(sys.modules[recorder_cls.__module__].logger, "error", errors.append)
    assert rec.start_recording(_FakeSubscribers(rec, [{}], {}), "d") is False  # was True: groups renamed / made-up values
    assert rec.is_recording is False and rec.h5_file is None
    assert any("was removed" in e for e in errors)


def test_start_always_flushes_stale_data(recorder_cls, monkeypatch, tmp_path):
    """The GUI passes no flag: it must drop pre-start messages just like the API."""
    msgs = _real_message_classes(recorder_cls)
    config = {"config": {"recording_frequency": 1000, "input": {"arm": {"topic": "/arm"}}}}
    rec = _loop_ready(recorder_cls, monkeypatch, _config_parser(config))
    rec.output_directory = str(tmp_path)
    js = msgs.JointState()
    js.name, js.position = ["a"], [1.0]
    subs = _FakeSubscribers(rec, [{"arm": js}] * 3, {"arm": msgs.JointState})
    subs.flush_latest_data = MagicMock()
    assert rec.start_recording(subs, "d") is True
    rec.recording_thread.join(5)
    assert rec.stop_recording() is True
    subs.flush_latest_data.assert_called_once()  # was not called from the GUI path


def test_a_failed_close_keeps_the_demo_for_a_retry(recorder_cls, tmp_path):
    rec = make_recorder(recorder_cls, tmp_path, [1, 2], {"arm": [joint_sample()] * 2})
    real = rec.h5_file

    class _CloseFails:
        def __getattr__(self, name):
            return getattr(real, name)

        def close(self):
            raise OSError("disk full")

    rec.h5_file = _CloseFails()
    rec.is_recording = True
    rec.recording_metadata["start_time"] = 0.0
    assert rec.stop_recording() is False
    assert rec._unsaved is True  # was False: the next Start would have cleared the buffer
    real.close()
    rec.output_directory = str(tmp_path / "retry")
    assert rec.retry_save() is True and rec._unsaved is False
    with h5py.File(rec.retried_file) as f:
        assert f["data/d/arm/joint_positions"].shape == (2, 1)


@pytest.mark.parametrize("encoding, image", [
    ("16UC1", np.arange(12, dtype=np.uint16).reshape(3, 4)),
    ("8UC3", np.arange(36, dtype=np.uint8).reshape(3, 4, 3)),
])
def test_an_image_without_bgr8_conversion_is_kept_as_decoded(encoding, image, monkeypatch):
    pytest.importorskip("sensor_msgs_py")
    from relay_d.utils import ros_media_codec as codec

    warnings = []
    monkeypatch.setattr(codec.logger, "warning", warnings.append)
    msg = codec.cv2_to_imgmsg(image, encoding)
    msg.header.frame_id = f"cam_{encoding}"
    for _ in range(3):
        assert np.array_equal(codec.decode_image(msg), image)  # was None: nothing saved
    assert len(warnings) == 1  # once per camera, not once per frame


def test_record_page_ignores_config_changes_while_recording(qapp, monkeypatch):
    pytest.importorskip("rclpy")
    for name in ("PyQt5.QtMultimedia", "PyQt5.QtMultimediaWidgets"):  # not in every PyQt5 install; unused here
        try:
            __import__(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, MagicMock())
    from relay_d.acquisition.qt_app.ui.record.record_page import RecordPage

    page = RecordPage.__new__(RecordPage)
    page.recording = SimpleNamespace(is_recording=True)
    page.yaml_parser, page.topic_subscribers, page.topic_containers = MagicMock(), MagicMock(), MagicMock()
    page.container_manager, page.tf_update_timer = MagicMock(), MagicMock()
    page.label_no_config, page.ui = None, SimpleNamespace()
    page._on_config_saved("a.yaml")
    page._on_config_loaded("a.yaml")
    page._on_config_cleared()
    page.yaml_parser.load_yaml_file.assert_not_called()
    page.topic_subscribers.clear_all_subscribers.assert_not_called()
    page.container_manager.create_containers.assert_not_called()


def test_recorded_streams_match_the_live_obs(monkeypatch):
    pytest.importorskip("rclpy")
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CompressedImage, Image, JointState, PointCloud2

    from relay_d.acquisition.qt_app.utils import stream_builder
    from relay_d.acquisition.qt_app.utils.stream_builder import StreamBuilder
    # first imported under recorder_cls's stubbed sensor_msgs in a full run
    monkeypatch.setattr(stream_builder, "_UNSUPPORTED_STREAM_MSG_TYPES", (Image, CompressedImage, PointCloud2))
    from relay_d.dispatch.obs_builder import ObsBuilder
    from relay_d.utils.numerics import enriched_tf

    inputs = {"js": {"topic": "/js"}, "tf": {"parent_frame": "base", "child_frame": "ee"}}
    specs = ["js.position[:]", "cos(js.position[:])", "sin(tf.translation.x - tf.translation.y)",
             "sin(tf.translation.x) * cos(tf.translation.z)", "tf.translation.z / tf.translation.x", "zero:2",
             "tf.euler.yaw", "tf.rot6d.i1", "tf.rot_matrix.r02", "js.velocity"]
    js = JointState(name=["a", "b"], position=[0.1, -0.4], velocity=[0.7, 0.0])
    tf = TransformStamped()
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = 0.5, -0.2, 0.9
    q = np.array([0.1, 0.3, -0.2, 0.9])
    q /= np.linalg.norm(q)
    r = tf.transform.rotation
    r.x, r.y, r.z, r.w = (float(v) for v in q)

    sb = StreamBuilder(SimpleNamespace(yaml_data={"config": {"input": inputs}}, get_input_data=lambda: [],
                                       get_streams=lambda: {"s": specs}))
    assert sb.get_build_errors() == []
    sb.evaluate({"js": js, "tf": tf})
    recorded = sb.finalize()["s"][0]

    ob = ObsBuilder.__new__(ObsBuilder)  # skip ROS setup
    ob._obs_cfg, ob._input_cfg, ob._tf_inputs, ob._static_inputs = {"s": specs}, inputs, {"tf"}, set()
    ob._extractors, ob._rotation_format, ob._obs_input_names, ob._passthrough_obs = {}, {}, {}, {}
    ob._build_extractors()
    ob._lock, ob._sync_threshold, ob._latest, ob._obs_refreshed = threading.Lock(), None, {}, {}
    ob._latest_inputs, ob._input_timestamps = {"js": js, "tf": enriched_tf(tf)}, {}
    ob.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=0, nanosec=0)))
    ob._evaluate_all()
    live = ob._latest["s"]

    assert recorded.shape == live.shape == (13,)
    np.testing.assert_allclose(recorded, live, rtol=1e-6)
