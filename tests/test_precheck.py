"""Pre-check (YAMLDrivenConverter.precheck / relay-d-convert --check): same verdicts as the conversion, nothing written."""

import json

import h5py
import numpy as np
import pytest
import yaml

from relay_d.acquisition.convert_cli import main as cli_main
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_options import PostprocessOptions
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter

from helpers import DATA_CONFIG, FPS, write_recording


def _converter(cfg=None):
    return YAMLDrivenConverter(PostprocessConfigLoader(), data_config=cfg or json.loads(json.dumps(DATA_CONFIG)))


def _stamp(demo, rate, recorded_hz=None, config="/somewhere/config_peter.yaml"):
    n = demo["streams/eef_translation"].shape[0]
    demo.create_dataset("timestamps", data=(np.arange(n) * 1e9 / rate).astype(np.int64))
    if recorded_hz is not None:
        meta = demo.create_group("metadata")
        meta.attrs["recording_frequency"] = float(recorded_hz)
        meta.attrs["yaml_config"] = config


def _problem_recording(tmp_path):
    """demo_0 converts; 1 off-rate, 2 no action stream, 3 no camera, 4 a short stream."""
    rec = tmp_path / "rec.h5"
    write_recording(rec, [14, 9, 10, 11, 12])
    with h5py.File(rec, "a") as f:
        d = f["data"]
        for i in range(5):
            _stamp(d[f"demo_{i}"], 0.7 * FPS if i == 1 else FPS, recorded_hz=FPS)
        del d["demo_2/streams/robot_joint_pos"]
        del d["demo_3/camera_rgb"]
        short = d["demo_4/streams/eef_translation"][:-3]
        del d["demo_4/streams/eef_translation"]
        d["demo_4/streams"].create_dataset("eef_translation", data=short)
    return rec


@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_precheck_skips_exactly_what_conversion_skips(tmp_path, fmt):
    rec = _problem_recording(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    report = _converter().precheck(fmt, [str(rec)])
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing written
    assert report["error"] is None and report["total"] == 5 and report["converted"] == 1
    reasons = {d["demo"]: d["reason"] for d in report["demos"]}
    assert reasons["demo_0"] == ""
    assert "keep up" in reasons["demo_1"]  # metadata matches the config: a slow PC
    assert reasons["demo_2"] == "no robot_joint_pos (action stream)"
    assert reasons["demo_3"] == "no camera agentview_image"
    assert "eef_translation" in reasons["demo_4"] and "9 rows" in reasons["demo_4"]

    if fmt == "lerobot":
        pytest.importorskip("av")
    conv = _converter()
    out = tmp_path / ("out.h5" if fmt == "robomimic" else "lerobot_out")
    assert conv.convert_multiple_files(fmt, [str(rec)], str(out))
    assert {f"{d['file']}::{d['demo']}" for d in report["demos"] if not d["ok"]} == set(conv.skipped_demos)
    assert conv.converted_demos == report["converted"]


def test_rate_mismatch_names_the_config_the_demo_was_recorded_with(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, [14])
    with h5py.File(rec, "a") as f:
        _stamp(f["data/demo_0"], 2 * FPS, recorded_hz=2 * FPS)
    report = _converter().precheck("robomimic", [str(rec)])
    demo = report["demos"][0]
    assert demo["recorded_with"] == "config_peter.yaml" and demo["recorded_hz"] == 2 * FPS
    assert demo["rate_hz"] == pytest.approx(2 * FPS)
    assert not demo["ok"] and "keep up" not in demo["reason"]
    assert f"recorded at {2 * FPS} Hz with config_peter.yaml; the loaded config says {FPS} Hz" in demo["reason"]


@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_stream_in_no_recording_stops_the_conversion(tmp_path, fmt):
    cfg = json.loads(json.dumps(DATA_CONFIG))
    cfg["output"]["streams"]["robot0_gripper"] = ["gripper.data"]
    rec = tmp_path / "rec.h5"
    write_recording(rec, [14, 9])
    with h5py.File(rec, "a") as f:
        _stamp(f["data/demo_0"], FPS, recorded_hz=FPS)
    report = _converter(cfg).precheck(fmt, [str(rec)])
    conv = _converter(cfg)
    out = tmp_path / "out"
    assert not conv.convert_multiple_files(fmt, [str(rec)], str(out)) and not out.exists()
    assert report["error"] == conv.last_error and report["output"] == [] and report["converted"] == 0
    assert "robot0_gripper" in conv.last_error and "eef_translation, robot_joint_pos" in conv.last_error
    assert "config_peter.yaml" in conv.last_error


def test_output_layout_robomimic(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, [14, 9])
    conv = _converter()
    conv.options = PostprocessOptions({"robomimic": {"splits": {"train_frac": 0.9, "val_frac": 0.1}}})
    out = dict(map(tuple, conv.precheck("robomimic", [str(rec)])["output"]))
    assert out["data/demo_N/obs/robot_joint_pos"] == "(N, 4) float32"
    assert out["data/demo_N/obs/agentview_image"] == "(N, 24, 32, 3) uint8"
    assert out["data/demo_N/actions"] == "(N, 4) = robot_joint_pos[0:4]"
    assert out["mask/train, mask/valid"] == "2 / 0 demos (train_frac 0.9, val_frac 0.1)"
    assert out["env_args"] == f"control_freq {FPS}"


def test_output_layout_lerobot(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, [14, 9])
    out = dict(map(tuple, _converter().precheck("lerobot", [str(rec)])["output"]))
    assert out["observation.state"] == "(7) robot_joint_pos.0..3, eef_translation.0..2"
    assert out["observation.images.agentview_image"] == "video 3x24x32"
    assert out["action"] == "(4) = robot_joint_pos[0:4]"  # [:] without joint_names: the writer names them action_i
    assert out["fps"] == str(FPS) and out["task"] == "stack_cubes" and out["episodes"] == "2"


def test_cli_check_prints_the_report_and_writes_nothing(tmp_path, capsys):
    rec = _problem_recording(tmp_path)
    cfg = tmp_path / "acq.yaml"
    cfg.write_text(yaml.safe_dump({"config": DATA_CONFIG}))
    before = sorted(p.name for p in tmp_path.iterdir())
    assert cli_main(["--config", str(cfg), "--format", "robomimic", "--input", str(rec), "--check"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    text = capsys.readouterr().out
    assert "1 of 5 demos would be converted" in text and "no camera agentview_image" in text
    assert "data/demo_N/actions" in text

    bad = json.loads(json.dumps(DATA_CONFIG))
    bad["output"]["streams"]["robot0_gripper"] = ["gripper.data"]
    bad["input"]["gripper"] = {"topic": "/gripper"}
    cfg.write_text(yaml.safe_dump({"config": bad}))
    assert cli_main(["--config", str(cfg), "--format", "robomimic", "--input", str(rec), "--check"]) == 1
    assert "robot0_gripper" in capsys.readouterr().out
