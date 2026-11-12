"""Each setting reaches the code that uses it, and the acquisition YAML still wins over it."""

import json
import logging
import types

import h5py
import numpy as np
import pytest

from relay_d.utils import user_settings as us
from helpers import image_sample, joint_sample, make_recorder


@pytest.fixture
def settings(settings_file):
    return us.save


def _yaml_parser(config=None):
    return types.SimpleNamespace(
        is_loaded=lambda: config is not None,
        yaml_data={"config": config} if config is not None else None,
        get_current_config_path=lambda: "cfg.yaml",
        get_input_data=lambda: [],
        get_streams=lambda: {},
        get_metadata=lambda: {},
    )


# ------------------------------------------------------------------ recorder


def test_recorder_takes_the_buffer_setting_and_the_frequency_only_from_the_yaml(recorder_cls, settings):
    settings({"rec_default_max_buffer_mb": 128, "rec_default_frequency": 12})  # an old settings file: the key is ignored
    rec = recorder_cls(_yaml_parser({}))
    assert rec.max_buffer_bytes == 128 * 1024**2
    assert rec.recording_frequency is None and (rec._sync_mode, rec._sync_tolerance) == ("soft", None)


def test_recorder_sync_tolerance_zero_means_off(recorder_cls, settings):
    assert recorder_cls(_yaml_parser())._sync_tolerance is None


def test_session_folder_name_follows_the_setting(recorder_cls, settings):
    settings({"session_folder_format": "run_%Y"})
    name = recorder_cls._session_folder_name()
    assert name.startswith("run_") and name.endswith("_Recording") and len(name) == 8 + len("_Recording")
    settings({"session_folder_format": ""})  # empty falls back to the default pattern
    name = recorder_cls._session_folder_name()
    assert len(name) == 19 + len("_Recording") and name.endswith("_Recording")


def test_compression_setting_reaches_the_hdf5_datasets(recorder_cls, settings, tmp_path):
    settings({"hdf5_compression": "lzf"})
    rec = make_recorder(recorder_cls, tmp_path, [1, 2], {"cam": [image_sample(), image_sample()], "arm": [joint_sample()] * 2})
    assert rec._save_recorded_data() is True
    assert rec.h5_file["data/d/cam/images"].compression == "lzf"
    rec.h5_file.close()
    settings({"hdf5_compression": "none", "hdf5_compression_level": 9})
    rec2 = make_recorder(recorder_cls, tmp_path / "..", [1, 2], {"cam": [image_sample(), image_sample()]})
    rec2.h5_file = h5py.File(tmp_path / "plain.h5", "w")
    assert rec2._save_recorded_data() is True
    assert rec2.h5_file["data/d/cam/images"].compression is None


# ------------------------------------------------------------------ logger, options, converter


def test_log_level_setting_and_environment_override(settings, monkeypatch):
    from relay_d.utils import coloring_logger as cl

    settings({"log_level": "DEBUG"})
    monkeypatch.delenv("RELAYD_LOG_LEVEL", raising=False)
    cl.apply_user_level()
    assert cl.logger.level == logging.DEBUG
    monkeypatch.setenv("RELAYD_LOG_LEVEL", "ERROR")
    cl.apply_user_level()
    assert cl.logger.level == logging.ERROR  # the environment wins
    monkeypatch.delenv("RELAYD_LOG_LEVEL")
    settings({"log_level": "INFO"})
    cl.apply_user_level()


def test_postprocess_defaults_come_from_settings_and_the_file_wins(settings, tmp_path):
    from relay_d.acquisition.qt_app.ui.postprocess.postprocess_options import PostprocessOptions

    settings({"split_train": 0.6, "split_val": 0.3})
    opts = PostprocessOptions()
    assert opts.splits() == {"train_frac": 0.6, "val_frac": 0.3}
    f = tmp_path / "postprocess_config.yaml"
    f.write_text("postprocess:\n  robomimic:\n    splits: {train_frac: 0.5, val_frac: 0.5}\n")
    loaded, _ = PostprocessOptions.load(str(f))
    assert loaded.splits()["train_frac"] == 0.5


def test_lerobot_encoder_settings_reach_the_dataset(settings, tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
    from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter

    settings({"lerobot_video_crf": 30, "lerobot_gop": 4, "lerobot_video_preset": "ultrafast", "lerobot_chunks_size": 7})
    cfg = {
        "dataset_name": "t", "recording_frequency": 10, "actions_to_extract": ["j"],
        "input": {"cam": {"topic": "/c", "modality": "rgb"}},
        "output": {"agentview_image": "cam/images", "streams": {"j": ["joint_1.position[:]"]}},
    }
    rec = tmp_path / "rec.h5"
    rng = np.random.default_rng(0)
    with h5py.File(rec, "w") as f:
        d = f.create_group("data/demo_0")
        d.create_group("streams").create_dataset("j", data=rng.normal(size=(8, 3)).astype(np.float32))
        d.create_group("cam").create_dataset("images", data=rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8))
    out = tmp_path / "ds"
    assert YAMLDrivenConverter(PostprocessConfigLoader(), data_config=cfg).convert_to_lerobot([str(rec)], str(out))
    info = json.loads((out / "meta" / "info.json").read_text())
    cam = next(v for k, v in info["features"].items() if v["dtype"] == "video")
    assert (cam["info"]["video.crf"], cam["info"]["video.g"], cam["info"]["video.preset"]) == (30, 4, "ultrafast")
    assert info["chunks_size"] == 7


# ------------------------------------------------------------------ training CLI


def test_train_cli_defaults_come_from_settings(settings, tmp_path, capsys):
    from relay_d.training.train_cli import main

    ds = tmp_path / "lerobot_dataset_1"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({
        "fps": 10, "total_frames": 10, "features": {"cam": {"dtype": "video"}, "action": {"dtype": "float32"}}}))
    settings({"python_lerobot": "/venvs/lerobot/bin/python", "train_default_device": "cpu",
              "train_lerobot_steps": 123, "train_lerobot_batch_size": 3})
    assert main(["--folder", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("/venvs/lerobot/bin/python -m lerobot")
    assert "--steps=123" in out and "--batch_size=3" in out and "--policy.device=cpu" in out
