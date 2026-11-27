"""Tests for StreamingH5Writer and DataRecorder's background HDF5 streaming path."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from relay_d.acquisition.qt_app.ui.record.h5_stream_writer import (
    StreamingH5Writer,
    is_streamable_sample,
)
from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus, DemoValidator


def _img(val: int, shape=(8, 10, 3), encoding="bgr8"):
    return {
        "type": "image",
        "encoding": encoding,
        "width": shape[1],
        "height": shape[0],
        "data": np.full(shape, val, dtype=np.uint8),
        "header": {"stamp": val, "frame_id": "cam"},
    }


def _cloud(val: float, shape=(4, 6, 3)):
    return {
        "type": "pointcloud",
        "organized": True,
        "width": shape[1],
        "height": shape[0],
        "fields": ["x", "y", "z"],
        "points": np.full(shape, val, dtype=np.float32),
        "header": {"stamp": int(val), "frame_id": "lidar"},
    }


def test_is_streamable_sample_classifies_heavy_arrays_only():
    assert is_streamable_sample(_img(1)) is True
    assert is_streamable_sample(_cloud(1.0)) is True
    assert is_streamable_sample({"type": "pointcloud", "organized": False, "points": None}) is False
    assert is_streamable_sample({"type": "joint_state", "positions": [1.0]}) is False
    assert is_streamable_sample(None) is False


def test_streaming_writer_matches_hdf5_schema_with_leading_nones_and_clouds(tmp_path):
    path = tmp_path / "demo.h5"
    with h5py.File(path, "w") as f:
        writer = StreamingH5Writer(f, "demo_0", max_queue_bytes=16 * 1024**2, flush_every=2)
        # 2 leading Nones back-filled with frame 0 -> total 4 image frames
        assert writer.enqueue("cam", _img(10), repeat=3) is True
        assert writer.enqueue("cam", _img(20), repeat=1) is True
        # Organized point cloud
        assert writer.enqueue("scan", _cloud(1.5), repeat=2) is True
        assert writer.enqueue("scan", _cloud(2.5), repeat=2) is True
        ok, streamed, err = writer.finish()
        assert ok is True and err is None
        assert streamed == {"cam", "scan"}
        assert writer.queued_bytes == 0

    with h5py.File(path, "r") as f:
        cam = f["data/demo_0/cam"]
        assert cam.attrs["image_width"] == 10
        assert cam.attrs["image_height"] == 8
        assert cam.attrs["image_encoding"] == "bgr8"
        imgs = cam["images"]
        assert imgs.shape == (4, 8, 10, 3)
        assert imgs.chunks == (1, 8, 10, 3)
        assert imgs.maxshape == (None, 8, 10, 3)
        assert np.all(imgs[0] == 10) and np.all(imgs[1] == 10) and np.all(imgs[2] == 10)
        assert np.all(imgs[3] == 20)

        scan = f["data/demo_0/scan"]
        assert bool(scan.attrs["organized"]) is True
        pts = scan["points"]
        assert pts.shape == (4, 4, 6, 3)
        assert np.allclose(pts[0], 1.5) and np.allclose(pts[3], 2.5)


def test_streaming_writer_handles_mid_demo_image_shape_change(tmp_path):
    path = tmp_path / "shape_change.h5"
    with h5py.File(path, "w") as f:
        writer = StreamingH5Writer(f, "d", max_queue_bytes=1024**2)
        writer.enqueue("cam", _img(1, shape=(4, 4, 3)))
        writer.enqueue("cam", _img(2, shape=(6, 6, 3)))
        ok, _, err = writer.finish()
        assert ok is True and err is None

    with h5py.File(path, "r") as f:
        assert f["data/d/cam/images"].shape == (1, 4, 4, 3)
        assert f["data/d/cam/image_1"].shape == (6, 6, 3)


def test_streaming_writer_reports_write_errors(tmp_path):
    path = tmp_path / "err.h5"
    f = h5py.File(path, "w")
    f.close()  # writing to a closed h5py.File triggers an error in the writer thread
    writer = StreamingH5Writer(f, "d", max_queue_bytes=1024**2)
    writer.enqueue("cam", _img(1))
    ok, _, err = writer.finish()
    assert ok is False and err is not None
    assert writer.error is not None


def test_data_recorder_streams_images_caches_repeated_frames_and_validates(recorder_cls, monkeypatch, tmp_path):
    """End-to-end DataRecorder start -> loop -> stop with streaming HDF5:
    - heavy images stream directly to HDF5 instead of staying in recorded_data RAM;
    - identical repeated Image objects are decoded only once;
    - leading None ticks are back-filled so len(images) == len(timestamps);
    - DemoValidator passes on the resulting HDF5 file."""
    rec_mod = sys.modules[recorder_cls.__module__]
    enc_mod = sys.modules[recorder_cls.__module__.replace(".data_recorder", ".message_encoding")]
    for name in ("Image", "CompressedImage", "PointCloud2", "JointState", "PoseStamped", "Twist",
                 "TransformStamped", "String", "Float32", "Float64", "Int32", "Bool"):
        setattr(enc_mod, name, type(name, (), {}))

    decode_calls = []

    def fake_decode(msg, is_depth=False):
        decode_calls.append(msg.tag)
        return np.full((4, 6, 3), msg.tag, dtype=np.uint8)

    monkeypatch.setattr(enc_mod, "decode_image", fake_decode)
    get = rec_mod.user_settings.get
    monkeypatch.setattr(rec_mod.user_settings, "get", lambda k: 0 if k == "rec_warmup_s" else get(k))

    def make_img_msg(tag, sec):
        m = enc_mod.Image()
        m.tag = tag
        m.encoding, m.width, m.height = "bgr8", 6, 4
        m.header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0), frame_id="cam")
        return m

    def make_js_msg(pos, sec):
        m = enc_mod.JointState()
        m.name, m.position, m.velocity, m.effort = ["j0"], [pos], [], []
        m.header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0), frame_id="")
        return m

    img1 = make_img_msg(11, 1)
    img2 = make_img_msg(22, 2)
    # 5 ticks: tick 0 has cam=None (leading None), ticks 1-2 repeat img1, ticks 3-4 repeat img2
    frames = [
        {"cam": None, "arm": make_js_msg(0.0, 1)},
        {"cam": img1, "arm": make_js_msg(0.1, 1)},
        {"cam": img1, "arm": make_js_msg(0.2, 2)},
        {"cam": img2, "arm": make_js_msg(0.3, 2)},
        {"cam": img2, "arm": make_js_msg(0.4, 3)},
    ]

    parser = SimpleNamespace(
        is_loaded=lambda: True,
        yaml_data={"config": {"recording_frequency": 50, "input": {"cam": {"topic": "/cam"}, "arm": {"topic": "/arm"}}}},
        get_current_config_path=lambda: "cfg.yaml",
        get_metadata=lambda: {},
        get_input_data=lambda: [],
        get_streams=lambda: {},
    )
    rec = recorder_cls(parser)
    tick_clock = {"n": 0}

    def fake_now():
        tick_clock["n"] += 1
        return SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=1, nanosec=tick_clock["n"] * 20_000_000))

    rec._clock = SimpleNamespace(now=fake_now)
    rec.output_directory = str(tmp_path)

    class Subs:
        def __init__(self):
            self.i = 0
            self.subscribers = {"cam": object(), "arm": object()}
            self.tf_lookup_containers = {}

        def flush_latest_data(self):
            pass

        def update_tf_lookups(self):
            pass

        def get_content(self):
            return {"cam": None, "arm": None}

        def get_content_and_timestamps(self):
            f = frames[min(self.i, len(frames) - 1)]
            self.i += 1
            if self.i >= len(frames):
                rec.stop_recording_flag = True
            return dict(f), {}

    assert rec.start_recording(Subs(), "demo_stream") is True
    rec.recording_thread.join(timeout=5.0)
    # Heavy images must NOT sit in recorded_data RAM while streaming
    assert "cam" not in rec.recorded_data
    assert rec.stop_recording() is True

    # Two unique image objects -> decoded only twice across 4 non-None ticks!
    assert decode_calls == [11, 22]

    with h5py.File(rec.h5_file_path, "r") as f:
        demo = f["data/demo_stream"]
        assert demo["timestamps"].shape == (5,)
        assert demo["arm/joint_positions"].shape == (5, 1)
        imgs = demo["cam/images"][:]
        assert imgs.shape == (5, 4, 6, 3)
        # Tick 0 (leading None) back-filled with img1 (11), ticks 1-2 = 11, ticks 3-4 = 22
        assert np.all(imgs[0] == 11) and np.all(imgs[1] == 11) and np.all(imgs[2] == 11)
        assert np.all(imgs[3] == 22) and np.all(imgs[4] == 22)

    validation = DemoValidator().validate(rec.h5_file_path)
    assert validation.overall_status != CheckStatus.FAIL, validation.summary_text()
