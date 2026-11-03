"""DataRecorder._save_recorded_data without ROS: rclpy and the message packages are replaced by stubs."""

import os
import sys
import types

import h5py
import numpy as np
from helpers import image_sample, joint_sample, make_recorder


def test_leading_none_samples_are_padded_not_dropped(recorder_cls, tmp_path):
    rec = make_recorder(
        recorder_cls, tmp_path, [1, 2, 3, 4], {"cam": [None, None, image_sample(), image_sample()], "arm": [joint_sample()] * 4}
    )
    assert rec._save_recorded_data() is True
    demo = rec.h5_file["data/d"]
    assert demo["cam/images"].shape == (4, 2, 2, 3)  # same length as timestamps
    assert demo["arm/joint_positions"].shape == (4, 1)
    assert "joint_velocities" not in demo["arm"]  # all-empty series are not written


def test_failed_type_save_makes_the_whole_save_fail(recorder_cls, tmp_path):
    ragged = {"type": "pose", "position": [0.0, 0.0, 0.0], "orientation": [0.0]}  # cannot be stacked with the next
    rec = make_recorder(recorder_cls, tmp_path, [1, 2], {"pose": [ragged, {**ragged, "orientation": [0.0, 1.0]}]})
    assert rec._save_recorded_data() is False  # was reported as saved with an "error_data" note


def test_joint_series_mixed_lengths_saved_as_variable_length(recorder_cls, tmp_path):
    rec = make_recorder(recorder_cls, tmp_path, [1, 2], {})
    rec._save_joint_series(rec.h5_file, "joint_positions", [[1.0, 2.0], [3.0], []])
    assert [len(r) for r in rec.h5_file["joint_positions"][:]] == [2, 1, 0]  # one row per timestamp, empty ones too
    rec._save_joint_series(rec.h5_file, "joint_efforts", [[], []])
    assert "joint_efforts" not in rec.h5_file


def test_buffer_accounting_counts_every_array_of_a_sample(recorder_cls):
    module = sys.modules[recorder_cls.__module__]
    image = {"type": "image", "data": np.zeros((10, 10, 3), np.uint8)}
    cloud = {"type": "pointcloud", "points": np.zeros((64, 1024, 3), np.float32), "fields": ["x", "y", "z"]}
    assert module._sample_nbytes(image) == 300
    assert module._sample_nbytes(cloud) == 64 * 1024 * 3 * 4  # was 0: scans never reached max_buffer_mb
    assert module._sample_nbytes({"type": "pose", "position": [0.0, 0.0, 0.0]}) == 0


def test_stacked_write_matches_np_stack_with_one_chunk_per_frame(recorder_cls, tmp_path, monkeypatch):
    module = sys.modules[recorder_cls.__module__]
    frames = [np.full((4, 5, 3), i, np.uint8) for i in range(6)]
    with h5py.File(tmp_path / "s.h5", "w") as f:
        ds = module._write_stacked(f, "images", frames)  # default setting: gzip
        assert ds.compression == "gzip" and ds.chunks == (1, 4, 5, 3)
        assert np.array_equal(ds[:], np.stack(frames))
        monkeypatch.setattr(module.user_settings, "h5_compression", lambda: {})
        plain = module._write_stacked(f, "plain", frames)
        assert plain.compression is None and plain.chunks is None and np.array_equal(plain[:], np.stack(frames))


def test_changing_the_recordings_folder_starts_a_new_session_folder(recorder_cls, tmp_path):
    module = sys.modules[recorder_cls.__module__]
    rec = recorder_cls(types.SimpleNamespace(is_loaded=lambda: False, yaml_data=None))
    module.user_settings.save({"hdf5_save_folder": str(tmp_path / "a")})

    def demo(name):
        rec._create_h5_file(name)
        rec.h5_file.close()
        return rec.h5_file_path

    first, second = demo("d0"), demo("d1")
    assert os.path.dirname(first) == os.path.dirname(second)  # same folder setting: one session, as before
    assert first.startswith(str(tmp_path / "a"))
    for state in ({}, {"_unsaved": True}, {"is_recording": True}):  # however the recorder is busy, the setting decides
        module.user_settings.save({"hdf5_save_folder": str(tmp_path / f"b{len(state)}{''.join(state)}")})
        for key, value in state.items():
            setattr(rec, key, value)
        path = demo("d2")
        assert path.startswith(str(tmp_path / f"b{len(state)}{''.join(state)}")), path
    rec.output_directory = str(tmp_path / "explicit")  # AppAPI(output_dir=...) wins over the setting
    assert demo("d3").startswith(str(tmp_path / "explicit"))


def test_a_message_that_breaks_a_processor_is_skipped_not_raised(recorder_cls):
    """The one try/except in _process_message_for_recording covers every per-type helper."""
    module = sys.modules[recorder_cls.__module__.replace(".data_recorder", ".message_encoding")]
    for name in ("Image", "CompressedImage", "PointCloud2", "JointState", "PoseStamped", "Twist",
                 "TransformStamped", "String", "Float32", "Float64", "Int32", "Bool"):
        setattr(module, name, type(name, (), {}))  # the module's message classes are MagicMocks in this test
    rec = recorder_cls(types.SimpleNamespace(is_loaded=lambda: False, yaml_data=None))

    twist = module.Twist()
    assert rec._process_message_for_recording(twist, "cmd") is None  # no .linear / .angular: AttributeError inside
    twist.linear = types.SimpleNamespace(x=1.0, y=2.0, z=3.0)
    twist.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.5)
    assert rec._process_message_for_recording(twist, "cmd") == {
        "type": "twist", "linear": [1.0, 2.0, 3.0], "angular": [0.0, 0.0, 0.5]}


def test_the_inputs_modality_reaches_the_image_decoder(recorder_cls):
    module = sys.modules[recorder_cls.__module__.replace(".data_recorder", ".message_encoding")]
    for name in ("Image", "CompressedImage", "PointCloud2", "JointState"):
        setattr(module, name, type(name, (), {}))  # isinstance() needs real classes, the stubs are MagicMocks
    calls = []
    module.decode_image = lambda msg, is_depth: calls.append(is_depth) or np.zeros((1, 1), np.uint8)
    rec = recorder_cls(types.SimpleNamespace(is_loaded=lambda: False, yaml_data=None))
    rec._modalities = {"depth_cam": "depth", "rgb_cam": "rgb"}
    msg = module.Image()
    msg.encoding, msg.width, msg.height = "16UC1", 1, 1
    msg.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0, nanosec=0), frame_id="f")
    for cid in ("depth_cam", "rgb_cam", "unlisted"):
        assert rec._process_message_for_recording(msg, cid)["type"] == "image"
    assert calls == [True, False, False]


def test_empty_first_joint_row_is_zero_padded_not_a_failed_save(recorder_cls, tmp_path):
    rec = make_recorder(recorder_cls, tmp_path, [1, 2], {})
    rec._save_joint_series(rec.h5_file, "joint_velocities", [[], [1.0, 2.0]])  # driver sent no velocity at first
    assert rec.h5_file["joint_velocities"][:].tolist() == [[0.0, 0.0], [1.0, 2.0]]


def test_yaml_date_in_metadata_does_not_drop_the_later_keys(recorder_cls, tmp_path):
    import datetime

    rec = make_recorder(recorder_cls, tmp_path, [1], {"arm": [joint_sample()]})
    rec.recording_metadata.update({"session": {"date": datetime.date(2026, 10, 27)}, "total_samples": 1})
    assert rec._save_recorded_data() is True
    attrs = rec.h5_file["data/d/metadata"].attrs
    assert "2026-10-27" in attrs["session"] and attrs["total_samples"] == 1


def test_a_new_demo_never_overwrites_an_existing_file(recorder_cls, tmp_path):
    import pytest

    rec = recorder_cls(types.SimpleNamespace(is_loaded=lambda: False, yaml_data=None))
    rec.output_directory = str(tmp_path)
    rec._create_h5_file("demo_0")
    rec.h5_file.close()
    with pytest.raises(Exception):
        rec._create_h5_file("demo_0")  # e.g. the counter restarted in a reused session folder
    rec._create_h5_file("demo_0", mode="w")  # retry_save rewrites its own partial file
    rec.h5_file.close()
