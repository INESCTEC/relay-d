"""End-to-end: recorded-style HDF5 -> LeRobotDataset v3.0 through YAMLDrivenConverter."""

import json

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("av")

from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import (  # noqa: E402
    PostprocessConfigLoader,
)
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import (  # noqa: E402
    YAMLDrivenConverter,
)

from helpers import DATA_CONFIG, FPS, LENGTHS, write_recording  # noqa: E402


@pytest.fixture
def converter():
    return YAMLDrivenConverter(PostprocessConfigLoader(), data_config=json.loads(json.dumps(DATA_CONFIG)))


def test_convert_to_lerobot(tmp_path, converter):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    out = tmp_path / "lerobot_dataset"

    assert converter.convert_multiple_files("lerobot", [str(rec)], str(out))

    info = json.loads((out / "meta/info.json").read_text())
    assert info["fps"] == FPS
    assert info["total_episodes"] == len(LENGTHS)
    assert info["total_frames"] == sum(LENGTHS)
    feats = info["features"]
    # state = robot_joint_pos (4) + eef_translation (3), in declaration order
    assert feats["observation.state"]["shape"] == [7]
    assert feats["observation.state"]["names"][:4] == [f"robot_joint_pos.{i}" for i in range(4)]
    assert feats["observation.state"]["names"][4:] == [f"eef_translation.{i}" for i in range(3)]
    # action comes from actions_to_extract (robot_joint_pos)
    assert feats["action"]["shape"] == [4]
    assert feats["observation.images.agentview_image"]["shape"] == [24, 32, 3]

    tasks = pq.read_table(out / "meta/tasks.parquet")
    assert tasks["task"].to_pylist() == ["stack_cubes"]

    table = pq.read_table(out / "data/chunk-000/file-000.parquet")
    assert table.num_rows == sum(LENGTHS)
    # action column is the recorded robot_joint_pos stream one step ahead (the last step holds)
    with h5py.File(rec, "r") as f:
        streams = [f[f"data/demo_{i}/streams/robot_joint_pos"][:] for i in range(2)]
    expected = np.concatenate([np.concatenate([s[1:], s[-1:]]) for s in streams])
    got = np.array(table["action"].to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert (out / "videos/observation.images.agentview_image/chunk-000/file-000.mp4").exists()


def test_config_overrides(tmp_path, converter):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, [6])
    from relay_d.acquisition.qt_app.ui.postprocess.postprocess_options import PostprocessOptions

    converter.options = PostprocessOptions({"lerobot": {"task": "pick up the red cube", "state_streams": ["eef_translation"]}})
    out = tmp_path / "ds"
    assert converter.convert_multiple_files("lerobot", [str(rec)], str(out))
    info = json.loads((out / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [3]
    assert pq.read_table(out / "meta/tasks.parquet")["task"].to_pylist() == ["pick up the red cube"]


def test_missing_recording_frequency_fails(tmp_path, converter):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, [6])
    del converter.data_config["recording_frequency"]
    assert not converter.convert_multiple_files("lerobot", [str(rec)], str(tmp_path / "ds"))


def test_robomimic_path_still_works(tmp_path, converter):
    """The shared batch preparation must not change the robomimic output."""
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    out = tmp_path / "robomimic.h5"
    assert converter.convert_multiple_files("robomimic", [str(rec)], str(out))
    with h5py.File(out, "r") as f:
        assert f["data"].attrs["total"] == len(LENGTHS)
        assert set(f["data/demo_0"].keys()) == {"actions", "obs", "rewards", "dones"}
        assert f["data/demo_0/actions"].shape == (LENGTHS[0], 4)
        assert "agentview_image" in f["data/demo_0/obs"]


def test_missing_pyav_fails_fast_with_clear_message(tmp_path, converter, monkeypatch):
    import sys

    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    out = tmp_path / "ds"
    monkeypatch.setitem(sys.modules, "av", None)  # makes `import av` raise ImportError
    assert not converter.convert_multiple_files("lerobot", [str(rec)], str(out))
    assert "av" in converter.last_error and sys.executable in converter.last_error
    assert "pip install av" in converter.last_error
    assert not out.exists()  # failed before any output was created


def test_last_error_cleared_on_success(tmp_path, converter):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, [5])
    converter.last_error = "stale"
    assert converter.convert_multiple_files("lerobot", [str(rec)], str(tmp_path / "ds"))
    assert converter.last_error is None


def test_missing_fps_sets_reason(tmp_path, converter):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, [5])
    del converter.data_config["recording_frequency"]
    assert not converter.convert_multiple_files("lerobot", [str(rec)], str(tmp_path / "ds"))
    assert "recording_frequency" in converter.last_error


def _write_recording_no_camera(path, lengths):
    import h5py as _h5

    rng = np.random.default_rng(1)
    with _h5.File(path, "w") as f:
        data = f.create_group("data")
        for i, n in enumerate(lengths):
            streams = data.create_group(f"demo_{i}").create_group("streams")
            streams.create_dataset("robot_joint_pos", data=rng.normal(size=(n, 4)).astype(np.float32))
            streams.create_dataset("eef_translation", data=rng.normal(size=(n, 3)).astype(np.float32))


def test_recording_without_images_needs_no_pyav(tmp_path, converter, monkeypatch):
    """Config declares agentview_image but the recording has no camera: no PyAV needed."""
    import sys

    rec = tmp_path / "recording_0.h5"
    _write_recording_no_camera(rec, [8, 5])
    out = tmp_path / "ds"
    monkeypatch.setitem(sys.modules, "av", None)
    assert converter.convert_multiple_files("lerobot", [str(rec)], str(out))
    info = json.loads((out / "meta/info.json").read_text())
    assert info["video_path"] is None
    assert info["total_frames"] == 13
    assert not (out / "videos").exists()
    assert not any(k.startswith("observation.images") for k in info["features"])


def test_missing_pyarrow_fails_clearly_without_output(tmp_path, converter, monkeypatch):
    import sys

    rec = tmp_path / "recording_0.h5"
    write_recording(rec, [5])
    out = tmp_path / "ds"
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    assert not converter.convert_multiple_files("lerobot", [str(rec)], str(out))
    assert "pyarrow" in converter.last_error and "lerobot" in converter.last_error
    assert sys.executable in converter.last_error
    assert not out.exists()


def test_writer_module_imports_without_pyarrow():
    """The module must stay importable so check_dependencies() can explain the problem."""
    import importlib
    import sys

    saved = {k: v for k, v in sys.modules.items() if k.startswith("pyarrow") or k.endswith("lerobot_writer")}
    try:
        for k in list(saved):
            sys.modules.pop(k, None)
        sys.modules["pyarrow"] = None
        sys.modules["pyarrow.parquet"] = None
        mod = importlib.import_module("relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer")
        assert mod.pa is None
        with pytest.raises(mod.LeRobotDependencyError):
            mod.LeRobotDatasetWriter("/tmp/never_created_lerobot_ds", fps=10)
    finally:
        for k in ("pyarrow", "pyarrow.parquet"):
            sys.modules.pop(k, None)
        sys.modules.pop("relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer", None)
        sys.modules.update(saved)


def test_one_broken_file_is_skipped_not_the_whole_dataset_deleted(tmp_path, converter, monkeypatch):
    good, bad = tmp_path / "recording_0.h5", tmp_path / "recording_1.h5"
    write_recording(good, LENGTHS)
    write_recording(bad, [5])
    split = converter._lerobot_split_obs

    def split_or_fail(demo, *a):
        if demo["num_samples"] == 5:
            raise KeyError("damaged group")  # not an OSError: used to reach the writer, which deleted everything
        return split(demo, *a)

    monkeypatch.setattr(converter, "_lerobot_split_obs", split_or_fail)
    out = tmp_path / "lerobot_dataset"
    assert converter.convert_multiple_files("lerobot", [str(good), str(bad)], str(out))
    assert json.loads((out / "meta/info.json").read_text())["total_episodes"] == len(LENGTHS)


def test_missing_action_stream_fails_the_demo_instead_of_narrowing_the_action(tmp_path):
    cfg = json.loads(json.dumps(DATA_CONFIG))
    cfg["actions_to_extract"] = ["robot_joint_pos", "gripper"]  # gripper never recorded
    converter = YAMLDrivenConverter(PostprocessConfigLoader(), data_config=cfg)
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    assert not converter.convert_multiple_files("lerobot", [str(rec)], str(tmp_path / "out"))  # was 4-wide actions


def test_episode_recorded_slower_than_the_dataset_fps_is_skipped(tmp_path, converter):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:  # demo_0 kept up with recording_frequency (FPS), demo_1 only managed 70% of it
        f["data/demo_0"].create_dataset("timestamps", data=(np.arange(LENGTHS[0]) * 1e9 / FPS).astype(np.int64))
        f["data/demo_1"].create_dataset("timestamps", data=(np.arange(LENGTHS[1]) * 1e9 / (0.7 * FPS)).astype(np.int64))
    out = tmp_path / "out"
    assert converter.convert_multiple_files("lerobot", [str(rec)], str(out))
    info = json.loads((out / "meta/info.json").read_text())
    assert info["total_episodes"] == 1 and info["total_frames"] == LENGTHS[0]  # its time base would be 1.4x too fast
