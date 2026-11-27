"""Tests for Workstreams 3 & 4: recording streaming safety, StreamBuilder caching, and DemoValidator checks."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from relay_d.acquisition.qt_app.ui.record.h5_stream_writer import StreamingH5Writer
from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus, DemoValidator


def test_streaming_writer_stall_aware_finish_succeeds_when_progressing(tmp_path, monkeypatch):
    """A slow disk that steadily writes items should finish cleanly even past stall_timeout."""
    h5_path = tmp_path / "slow.h5.partial"
    with h5py.File(h5_path, "w") as f:
        writer = StreamingH5Writer(f, "demo_0", max_queue_bytes=16 * 1024**2)
        orig_write = writer._write_item

        def slow_write(cid, sample, repeat):
            time.sleep(0.06)
            return orig_write(cid, sample, repeat)

        monkeypatch.setattr(writer, "_write_item", slow_write)
        for i in range(6):
            writer.enqueue("cam", {"type": "image", "data": np.full((8, 8, 3), i, dtype=np.uint8), "width": 8, "height": 8, "encoding": "bgr8"})

        # Total time ~0.36s > stall_timeout=0.2s, but each write finishes in 0.06s -> must succeed!
        ok, cids, err = writer.finish(timeout=2.0, stall_timeout=0.2)
        assert ok is True and err is None and cids == {"cam"}
        assert not writer.is_alive
        assert f["data/demo_0/cam/images"].shape == (6, 8, 8, 3)


def test_streaming_writer_stalled_finish_sets_abort_and_reports_alive(tmp_path, monkeypatch):
    """A truly hung disk write triggers TimeoutError without closing h5_file from another thread."""
    gate = threading.Event()
    h5_path = tmp_path / "hung.h5.partial"
    f = h5py.File(h5_path, "w")
    writer = StreamingH5Writer(f, "demo_0", max_queue_bytes=16 * 1024**2)
    monkeypatch.setattr(writer, "_write_item", lambda *a: gate.wait(timeout=5.0))
    writer.enqueue("cam", {"type": "image", "data": np.zeros((4, 4, 3), dtype=np.uint8), "width": 4, "height": 4, "encoding": "bgr8"})
    try:
        ok, _cids, err = writer.finish(timeout=0.25, stall_timeout=0.2)
        assert ok is False
        assert isinstance(err, TimeoutError)
        assert writer._abort_flag is True
        assert writer.is_alive is True
    finally:
        gate.set()
        writer._thread.join(timeout=2.0)
        f.close()


def test_stream_builder_caches_tf_and_joint_filter_once_per_tick(monkeypatch):
    from relay_d.acquisition.qt_app.utils import stream_builder as sb_mod

    tf_calls = []
    orig_enriched = sb_mod.enriched_tf

    def counting_enriched(msg):
        tf_calls.append(1)
        return orig_enriched(msg)

    monkeypatch.setattr(sb_mod, "enriched_tf", counting_enriched)
    monkeypatch.setattr(sb_mod, "_UNSUPPORTED_STREAM_MSG_TYPES", ())

    parser = SimpleNamespace(
        yaml_data={
            "config": {
                "input": {
                    "tf": {"parent_frame": "base", "child_frame": "ee"},
                    "js": {"topic": "/js", "joint_names": ["j1", "j2"]},
                }
            }
        },
        get_input_data=lambda: [{"name": "js", "joint_names": ["j1", "j2"]}],
        get_streams=lambda: {
            "pose": [
                "tf.translation.x",
                "tf.translation.y",
                "tf.translation.z",
                "tf.euler.roll",
                "tf.euler.pitch",
                "tf.euler.yaw",
            ],
            "joints": ["js.position[:]", "js.velocity[:]"],
        },
    )
    sb = sb_mod.StreamBuilder(parser)

    joint_calls = []
    orig_filter = sb._filter_joints

    def counting_filter(name, msg):
        joint_calls.append(name)
        return orig_filter(name, msg)

    monkeypatch.setattr(sb, "_filter_joints", counting_filter)

    tf_msg = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.1, y=0.2, z=0.3),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    js_msg = SimpleNamespace(header=None, name=["j1", "j2", "extra"], position=[1.0, 2.0, 9.0], velocity=[0.1, 0.2, 0.9], effort=[])

    sb.evaluate({"tf": tf_msg, "js": js_msg})
    assert len(tf_calls) == 1  # was 6 (once per spec!)
    assert len(joint_calls) == 1  # was 2 (once per spec!)
    out = sb.finalize()
    np.testing.assert_allclose(out["pose"][0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out["joints"][0], [1.0, 2.0, 0.1, 0.2], atol=1e-6)


def test_demo_validator_flags_nan_streams_and_checks_points_sample_count(tmp_path):
    p = tmp_path / "demo_nan.h5"
    n = 15
    with h5py.File(p, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("timestamps", data=np.arange(n, dtype=np.int64) * 50_000_000)
        mg = g.create_group("metadata")
        mg.attrs["duration"] = 0.75
        mg.attrs["recording_frequency"] = 20.0
        sg = g.create_group("streams")
        arr = np.zeros((n, 3), dtype=np.float32)
        arr[4, 1] = np.nan
        sg.create_dataset("state", data=arr)
        pc = g.create_group("lidar")
        pc.create_dataset("points", data=np.zeros((n - 2, 4, 4, 3), dtype=np.float32))

    res = DemoValidator().validate(str(p))
    by_name = {c.name: c for c in res.checks}
    assert by_name["finite_streams"].status == CheckStatus.FAIL
    assert "streams/state" in by_name["finite_streams"].message
    assert by_name["sample_count_consistency"].status == CheckStatus.FAIL
    assert "lidar/points" in by_name["sample_count_consistency"].message
    assert res.overall_status == CheckStatus.FAIL
