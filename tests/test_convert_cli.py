"""Headless converter (relay_d.acquisition.convert_cli)."""

import json
import subprocess
import sys

import h5py
import pyarrow.parquet as pq
import pytest
import yaml

pytest.importorskip("av")

from relay_d.acquisition.convert_cli import collect_inputs, main  # noqa: E402
from helpers import DATA_CONFIG, LENGTHS, write_recording  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    rec_dir = tmp_path / "recordings" / "sub"
    rec_dir.mkdir(parents=True)
    write_recording(rec_dir / "rec_2.h5", [6])
    write_recording(rec_dir / "rec_10.h5", LENGTHS)
    cfg = tmp_path / "acquisition_config.yaml"
    cfg.write_text(yaml.safe_dump({"config": DATA_CONFIG}))
    return tmp_path, cfg


def test_collect_inputs_recursive_and_natural_order(workspace):
    tmp, _ = workspace
    files = collect_inputs([str(tmp / "recordings")])
    assert [f.rsplit("/", 1)[1] for f in files] == ["rec_2.h5", "rec_10.h5"]
    with pytest.raises(FileNotFoundError):
        collect_inputs([str(tmp / "missing")])


def test_lerobot_conversion(workspace, capsys):
    tmp, cfg = workspace
    out = tmp / "ds"
    code = main(["--config", str(cfg), "--format", "lerobot", "--input", str(tmp / "recordings"),
                 "--output", str(out), "--task", "stack the cubes"])
    assert code == 0
    info = json.loads((out / "meta/info.json").read_text())
    assert info["total_episodes"] == 1 + len(LENGTHS)
    assert pq.read_table(out / "meta/tasks.parquet")["task"].to_pylist() == ["stack the cubes"]
    assert (out / "acquisition_config.yaml").exists()


def test_robomimic_conversion_with_sibling_options(workspace):
    tmp, cfg = workspace
    (tmp / "postprocess_config.yaml").write_text(yaml.safe_dump({
        "postprocess": {"robomimic": {"metadata": {"env_name": "CliEnv", "env_version": "1", "type": 1, "env_kwargs": {}}}}
    }))
    out = tmp / "out" / "dataset.h5"
    assert main(["--config", str(cfg), "--format", "robomimic", "--input", str(tmp / "recordings"),
                 "--output", str(out)]) == 0
    with h5py.File(out, "r") as f:
        assert f["data"].attrs["total"] == 1 + len(LENGTHS)
        assert json.loads(f["data"].attrs["env_args"])["env_name"] == "CliEnv"
    assert (tmp / "out" / "dataset_acquisition_config.yaml").exists()
    assert (tmp / "out" / "dataset_postprocess_config.yaml").exists()


def test_usage_errors_exit_2(workspace, tmp_path, capsys):
    tmp, cfg = workspace
    base = ["--format", "lerobot", "--input", str(tmp / "recordings")]
    assert main(["--config", str(tmp / "nope.yaml"), *base, "--output", str(tmp / "o1")]) == 2
    (tmp / "bad.yaml").write_text("foo: 1\n")
    assert main(["--config", str(tmp / "bad.yaml"), *base, "--output", str(tmp / "o2")]) == 2
    assert main(["--config", str(cfg), "--format", "lerobot", "--input", str(tmp / "nodir"),
                 "--output", str(tmp / "o3")]) == 2
    (tmp / "empty").mkdir()
    assert main(["--config", str(cfg), "--format", "lerobot", "--input", str(tmp / "empty"),
                 "--output", str(tmp / "o4")]) == 2
    nonempty = tmp / "full"
    nonempty.mkdir()
    (nonempty / "x").write_text("x")
    assert main(["--config", str(cfg), *base, "--output", str(nonempty)]) == 2
    assert main(["--config", str(cfg), "--format", "robomimic", "--input", str(tmp / "recordings"),
                 "--output", str(tmp)]) == 2
    assert "error:" in capsys.readouterr().err


def test_failed_conversion_exits_1_with_reason(workspace, monkeypatch, capsys):
    tmp, cfg = workspace
    monkeypatch.setitem(sys.modules, "av", None)  # config has an image stream in rec files
    code = main(["--config", str(cfg), "--format", "lerobot", "--input", str(tmp / "recordings"),
                 "--output", str(tmp / "ds")])
    assert code == 1
    assert "av" in capsys.readouterr().err


def test_script_wrapper_help_runs_without_ros_or_qt(tmp_path):
    root = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
    result = subprocess.run(
        [sys.executable, "scripts/convert_dataset.py", "--help"],
        cwd=root, env={"PYTHONPATH": root, "PATH": ""}, capture_output=True, text=True,
    )
    assert result.returncode == 0 and "--format" in result.stdout


def test_robomimic_refuses_to_overwrite(workspace):
    tmp, cfg = workspace
    out = tmp / "out" / "dataset.h5"
    args = ["--config", str(cfg), "--format", "robomimic", "--input", str(tmp / "recordings"),
            "--output", str(out)]
    assert main(args) == 0
    before = out.read_bytes()
    assert main(args) == 2
    assert out.read_bytes() == before
