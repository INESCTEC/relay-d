"""Offline (no live ROS) config loading for post-processing."""

import json

import pytest
import yaml

from relay_d.acquisition.qt_app.ui.postprocess.offline_config import load_offline_config
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter

pytest.importorskip("av")
from helpers import DATA_CONFIG, LENGTHS, write_recording  # noqa: E402


def _write_cfg(path, config):
    path.write_text(yaml.safe_dump({"config": config}))
    return str(path)


def test_valid_config_with_dead_topics_loads(tmp_path):
    cfg = json.loads(json.dumps(DATA_CONFIG))
    cfg["input"] = {"camera_rgb": {"topic": "/nonexistent/topic"}}
    parser, problems = load_offline_config(_write_cfg(tmp_path / "c.yaml", cfg))
    assert parser is not None and problems == []
    assert parser.yaml_data["config"]["recording_frequency"] == 10


def test_missing_file_and_missing_config_section(tmp_path):
    assert load_offline_config(str(tmp_path / "nope.yaml"))[0] is None
    (tmp_path / "bad.yaml").write_text("foo: 1\n")
    parser, problems = load_offline_config(str(tmp_path / "bad.yaml"))
    assert parser is None and "config" in problems[0]


def test_warnings_for_missing_output_and_fps(tmp_path):
    parser, problems = load_offline_config(_write_cfg(tmp_path / "c.yaml", {"dataset_name": "x"}))
    assert parser is not None
    assert any("no 'output'" in p for p in problems)
    assert any("recording_frequency" in p for p in problems)


def test_malformed_stream_spec_is_rejected(tmp_path):
    cfg = json.loads(json.dumps(DATA_CONFIG))
    cfg["output"]["streams"]["bad"] = ["a.x - b.y - c.z"]  # arithmetic allows exactly two operands
    parser, problems = load_offline_config(_write_cfg(tmp_path / "c.yaml", cfg))
    assert parser is None and problems and problems[0].startswith("output.streams")


def test_offline_config_drives_lerobot_conversion(tmp_path):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    parser, _ = load_offline_config(_write_cfg(tmp_path / "c.yaml", DATA_CONFIG))
    converter = YAMLDrivenConverter(PostprocessConfigLoader(), data_config=parser.yaml_data["config"])
    out = tmp_path / "ds"
    assert converter.convert_multiple_files("lerobot", [str(rec)], str(out))
    assert json.loads((out / "meta/info.json").read_text())["total_episodes"] == len(LENGTHS)
