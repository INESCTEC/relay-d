"""Tests for lazy ObsBuilder decoding, unpadded imgmsg_to_cv2 fast path, and ContainerVisualizer optimizations."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_imgmsg_to_cv2_unpadded_and_padded_parity():
    from relay_d.utils import ros_media_codec as codec

    # Unpadded rgb8 -> bgr8 and rgb8 -> rgb8 (direct)
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    msg_unpadded = SimpleNamespace(
        encoding="rgb8", height=4, width=5, step=5 * 3, is_bigendian=0, data=rgb.tobytes()
    )
    np.testing.assert_array_equal(codec.imgmsg_to_cv2(msg_unpadded, "rgb8"), rgb)
    np.testing.assert_array_equal(codec.imgmsg_to_cv2(msg_unpadded, "bgr8"), rgb[..., ::-1])

    # Padded step > width * channels * itemsize
    padded = np.zeros((4, 5 * 3 + 7), dtype=np.uint8)
    padded[:, : 5 * 3] = rgb.reshape(4, 15)
    msg_padded = SimpleNamespace(
        encoding="rgb8", height=4, width=5, step=5 * 3 + 7, is_bigendian=0, data=padded.tobytes()
    )
    np.testing.assert_array_equal(codec.imgmsg_to_cv2(msg_padded, "rgb8"), rgb)
    np.testing.assert_array_equal(codec.imgmsg_to_cv2(msg_padded, "bgr8"), rgb[..., ::-1])

    # Big-endian 16UC1 depth passthrough
    depth = np.arange(12, dtype=">u2").reshape(3, 4)
    msg_be = SimpleNamespace(
        encoding="16UC1", height=3, width=4, step=4 * 2, is_bigendian=1, data=depth.tobytes()
    )
    out = codec.imgmsg_to_cv2(msg_be, "passthrough")
    assert out.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(out, depth.astype(np.uint16))

    # mono8 -> rgb8 single-step conversion for Qt display
    mono = np.full((3, 4), 77, dtype=np.uint8)
    msg_mono = SimpleNamespace(
        encoding="mono8", height=3, width=4, step=4, is_bigendian=0, data=mono.tobytes()
    )
    mono_rgb = codec.imgmsg_to_cv2(msg_mono, "rgb8")
    assert mono_rgb.shape == (3, 4, 3) and np.all(mono_rgb == 77)


def test_obs_builder_lazy_passthrough_decodes_once_per_unique_message(monkeypatch):
    pytest.importorskip("rclpy")
    from sensor_msgs.msg import Image, PointCloud2

    from relay_d.dispatch import obs_builder as ob_mod

    decodes = []

    def fake_decode(msg, is_depth=False):
        decodes.append(msg.header.frame_id)
        return np.full((4, 6, 3), len(decodes), dtype=np.uint8)

    monkeypatch.setattr(ob_mod, "decode_image", fake_decode)

    ob = ob_mod.ObsBuilder.__new__(ob_mod.ObsBuilder)
    ob._input_cfg = {"cam": {"topic": "/cam"}, "scan": {"topic": "/scan"}}
    ob._extractors = {}
    ob._rotation_format = {}
    ob._obs_input_names = {}
    ob._passthrough_obs = {"cam_obs": "cam", "scan_obs": "scan"}
    ob._passthrough_sources = {"cam", "scan"}
    ob._joint_filters = {}
    ob._used_inputs = None
    ob._lock = threading.Lock()
    ob._latest_inputs = {}
    ob._input_timestamps = {}
    ob._latest = {}
    ob._arrival = {}
    ob._obs_refreshed = {}
    ob._sync_threshold = None
    ob._desync_warned = set()
    ob._passthrough_decode_warned = set()
    ob._decoded_passthrough_cache = {}
    errors = []
    ob.get_logger = lambda: SimpleNamespace(error=errors.append, warn=lambda *a, **k: None)

    img1 = Image()
    img1.header.frame_id = "f1"
    img2 = Image()
    img2.header.frame_id = "f2"

    # Two frames arrive on the ROS executor thread: ZERO decodes must happen in _topic_callback!
    ob._topic_callback(img1, "cam")
    ob._topic_callback(img2, "cam")
    assert decodes == []

    # First get_obs() decodes ONLY the latest message (img2); img1 was skipped for free.
    snap1 = ob.get_obs()
    assert decodes == ["f2"]
    assert snap1["cam_obs"].shape == (4, 6, 3)
    assert not snap1["cam_obs"].flags.writeable

    # Repeated get_obs() with no new camera frame hits the cache and returns the same array zero-copy.
    snap2 = ob.get_obs()
    assert decodes == ["f2"]
    assert snap2["cam_obs"] is snap1["cam_obs"]

    # Unorganized PointCloud2 (height <= 1) is rejected in O(1) without touching _arrival.
    bad_pc = PointCloud2()
    bad_pc.height = 1
    bad_pc.width = 100
    ob._topic_callback(bad_pc, "scan")
    assert "scan" not in ob._arrival
    assert len(errors) == 1


def test_visualizer_skips_unchanged_messages_and_slices_pointcloud(qapp, monkeypatch):
    pytest.importorskip("rclpy")
    from sensor_msgs.msg import PointCloud2

    from relay_d.acquisition.qt_app.ui.record import container_visualizer as cv_mod

    vis = cv_mod.ContainerVisualizer()

    # PointCloud2 generator that fails if more than 6 points are consumed (proving islice(5)).
    consumed = {"n": 0}

    def exploding_read_points(msg, field_names=None, skip_nans=True):
        for i in range(1000):
            consumed["n"] += 1
            if i >= 6:
                raise AssertionError("read_points materialized more than 5 points!")
            yield (float(i), 0.0, 0.0)

    monkeypatch.setattr(cv_mod.point_cloud2, "read_points", exploding_read_points)

    pc = PointCloud2()
    pc.width, pc.height, pc.point_step = 1000, 1, 12
    text_box = MagicMock()
    container = SimpleNamespace(container_id="pc1", text_data_display=text_box)
    subs = SimpleNamespace(get_content=lambda cid: pc, get_message_count=lambda cid: 1)

    vis.update_container_display(container, subs)
    assert consumed["n"] == 5
    assert text_box.setText.call_count == 1

    # Calling again with the exact same message object must skip re-rendering completely.
    vis.update_container_display(container, subs)
    assert consumed["n"] == 5
    assert text_box.setText.call_count == 1
