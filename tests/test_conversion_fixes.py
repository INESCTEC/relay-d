"""Regression tests for the audited conversion bugs (robomimic + LeRobot paths)."""

import json

import h5py
import numpy as np
import pytest

pytest.importorskip("av")

from relay_d.acquisition.convert_cli import main as cli_main  # noqa: E402
from relay_d.acquisition.qt_app.ui.config.builder.config_check import ERROR, check_config  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess import lerobot_writer  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.robomimic_config_handler import classify_observations  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter  # noqa: E402

from helpers import DATA_CONFIG, LENGTHS, write_recording  # noqa: E402


def _cfg():
    return json.loads(json.dumps(DATA_CONFIG))


def _converter(cfg=None):
    return YAMLDrivenConverter(PostprocessConfigLoader(), data_config=cfg or _cfg())


def _robomimic(tmp_path, conv, rec):
    out = tmp_path / "out.h5"
    ok = conv.convert_multiple_files("robomimic", [str(rec)], str(out))
    return ok, out


def test_demo_missing_the_action_stream_is_skipped_not_given_zero_actions(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:
        del f["data/demo_1/streams/robot_joint_pos"]
    conv = _converter()
    ok, out = _robomimic(tmp_path, conv, rec)
    assert ok
    with h5py.File(out, "r") as f:
        assert f["data"].attrs["total"] == 1
        assert f["data/demo_0/actions"].shape[0] == LENGTHS[0]
    assert len(conv.skipped_demos) == 1 and conv.converted_demos == 1
    assert "Converted 1 of 2 demos (1 skipped" in conv.conversion_summary()


def test_demo_missing_a_camera_is_skipped_not_given_black_frames(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:
        del f["data/demo_1/camera_rgb"]
    ok, out = _robomimic(tmp_path, _converter(), rec)
    assert ok
    with h5py.File(out, "r") as f:
        assert f["data"].attrs["total"] == 1
        assert f["data/demo_0/obs/agentview_image"][:].any()


@pytest.mark.parametrize("actions", [[], None])
def test_no_actions_to_extract_fails_instead_of_zero_actions(tmp_path, actions):
    cfg = _cfg()
    if actions is None:
        del cfg["actions_to_extract"]
    else:
        cfg["actions_to_extract"] = actions
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    conv = _converter(cfg)
    ok, out = _robomimic(tmp_path, conv, rec)
    assert not ok and not out.exists()
    assert "actions_to_extract" in conv.last_error
    assert any(p.level == ERROR and "actions_to_extract" in p.message for p in check_config({"config": cfg}))


@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_stream_with_normalize_fails_the_conversion(tmp_path, fmt):
    cfg = _cfg()
    cfg["output"]["streams"]["robot_joint_pos"] = {"specs": ["joint_1.position[:]"], "normalize": True}
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    conv = _converter(cfg)
    out = tmp_path / ("out.h5" if fmt == "robomimic" else "lerobot_out")
    assert not conv.convert_multiple_files(fmt, [str(rec)], str(out))
    assert not out.exists()
    assert "Relay-D no longer normalizes" in conv.last_error and "'robot_joint_pos'" in conv.last_error


def _depth_config():
    cfg = _cfg()
    cfg["input"]["camera_depth"] = {"topic": "/depth", "modality": "depth"}
    cfg["output"]["agentview_depth"] = "camera_depth/images"
    return cfg


def test_modality_is_looked_up_through_the_output_key(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:
        for i, n in enumerate(LENGTHS):
            f[f"data/demo_{i}"].create_group("camera_depth").create_dataset(
                "images", data=np.ones((n, 6, 8), np.float32))
    cfg = _depth_config()
    conv = _converter(cfg)
    ok, out = _robomimic(tmp_path, conv, rec)
    assert ok
    classes = classify_observations(cfg, conv._shape_cache)
    assert classes["depth"] == ["agentview_depth"] and "agentview_depth" not in classes["low_dim"]
    with h5py.File(out, "r") as f:
        assert f["data/demo_0/obs/agentview_depth"].chunks == (1, 6, 8)


def test_robomimic_output_has_only_what_training_reads(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    ok, out = _robomimic(tmp_path, _converter(), rec)
    assert ok
    with h5py.File(out, "r") as f:
        assert set(f["data/demo_0"].keys()) == {"actions", "obs", "rewards", "dones"}  # no next_obs / states
        # robomimic copies env_args into the checkpoint (env_metadata): inference reads its rate from control_freq
        assert json.loads(f["data"].attrs["env_args"]) == {"env_name": "relay_d", "type": 1,
                                                           "env_kwargs": {"control_freq": float(DATA_CONFIG["recording_frequency"])}}
        assert set(f["mask"].keys()) <= {"train", "valid"}


@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_demo_without_streams_is_skipped(tmp_path, fmt):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:
        del f["data/demo_1/streams"]
    conv = _converter()
    out = tmp_path / ("out.h5" if fmt == "robomimic" else "lerobot_out")
    assert conv.convert_multiple_files(fmt, [str(rec)], str(out))
    assert conv.converted_demos == 1 and conv.skipped_demos == [f"{rec}::demo_1"]


def test_stream_shorter_than_the_demo_fails_the_demo(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:
        short = f["data/demo_1/streams/eef_translation"][:-2]
        del f["data/demo_1/streams/eef_translation"]
        f["data/demo_1/streams"].create_dataset("eef_translation", data=short)
    conv = _converter()
    ok, out = _robomimic(tmp_path, conv, rec)
    assert ok and conv.converted_demos == 1 and len(conv.skipped_demos) == 1


def test_skipped_demos_are_counted_for_lerobot_and_shown_by_the_cli(tmp_path, capsys):
    import yaml

    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:
        del f["data/demo_1/streams/robot_joint_pos"]
    cfg_path = tmp_path / "acq.yaml"
    cfg_path.write_text(yaml.safe_dump({"config": _cfg()}))
    assert cli_main(["--config", str(cfg_path), "--format", "lerobot", "--input", str(rec),
                     "--output", str(tmp_path / "ds")]) == 0
    assert "Converted 1 of 2 demos (1 skipped, see log)" in capsys.readouterr().out


def _two_camera_recording(path, lengths, seed):
    write_recording(path, lengths, seed=seed)
    with h5py.File(path, "a") as f:
        for i, n in enumerate(lengths):
            f[f"data/demo_{i}"].create_group("camera_wrist").create_dataset(
                "images", data=np.full((n, 24, 32, 3), 9, np.uint8))


def test_video_error_mid_episode_aborts_the_whole_lerobot_dataset(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg["input"]["camera_wrist"] = {"topic": "/wrist", "modality": "rgb"}
    cfg["output"]["wrist_image"] = "camera_wrist/images"
    a, b = tmp_path / "rec_0.h5", tmp_path / "rec_1.h5"
    _two_camera_recording(a, [6], 0)
    _two_camera_recording(b, [5], 1)

    real = lerobot_writer.LeRobotDatasetWriter._write_video
    calls = []

    def flaky(self, key, frames):
        calls.append(key)
        if len(calls) == 2:  # the 2nd camera of the 1st episode, after its rows were written
            raise OSError("encoder died")
        return real(self, key, frames)

    monkeypatch.setattr(lerobot_writer.LeRobotDatasetWriter, "_write_video", flaky)
    conv = _converter(cfg)
    out = tmp_path / "ds"
    assert not conv.convert_multiple_files("lerobot", [str(a), str(b)], str(out))
    assert "encoder died" in conv.last_error
    assert not out.exists()


def test_writer_cannot_finalize_after_a_failed_episode(tmp_path, monkeypatch):
    def broken(self, key, frames):
        raise OSError("disk full")

    monkeypatch.setattr(lerobot_writer.LeRobotDatasetWriter, "_write_video", broken)
    writer = lerobot_writer.LeRobotDatasetWriter(tmp_path / "ds", fps=10)
    frames = np.zeros((4, 8, 8, 3), np.uint8)
    with pytest.raises(lerobot_writer.LeRobotFatalError):
        writer.add_episode(np.zeros((4, 2)), np.zeros((4, 1)), {"observation.images.cam": frames}, task="t")
    with pytest.raises(lerobot_writer.LeRobotWriterError):
        writer.finalize()
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_demo_recorded_off_rate_is_skipped(tmp_path, fmt):
    from helpers import FPS

    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    with h5py.File(rec, "a") as f:  # demo_1 recorded at 70% of recording_frequency: a PC that could not keep up
        for name, rate in (("demo_0", FPS), ("demo_1", 0.7 * FPS)):
            f[f"data/{name}"].create_dataset("timestamps", data=(np.arange(f[f"data/{name}/streams/robot_joint_pos"].shape[0]) * 1e9 / rate).astype(np.int64))
    conv = _converter()
    out = tmp_path / ("out.h5" if fmt == "robomimic" else "lerobot_out")
    assert conv.convert_multiple_files(fmt, [str(rec)], str(out))
    assert conv.converted_demos == 1 and conv.skipped_demos == [f"{rec}::demo_1"]
