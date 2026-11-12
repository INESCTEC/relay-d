"""Headless converter (relay_d.acquisition.convert_cli)."""

import json
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
    assert (tmp / "acquisition_config.yaml").exists()  # above the dataset folder, where the launcher looks


def test_robomimic_conversion_with_sibling_options(workspace):
    tmp, cfg = workspace
    (tmp / "postprocess_config.yaml").write_text(yaml.safe_dump({
        "postprocess": {"robomimic": {"splits": {"train_frac": 1.0, "val_frac": 0.0}}}
    }))
    out = tmp / "out" / "dataset.h5"
    assert main(["--config", str(cfg), "--format", "robomimic", "--input", str(tmp / "recordings"),
                 "--output", str(out)]) == 0
    with h5py.File(out, "r") as f:
        assert f["data"].attrs["total"] == 1 + len(LENGTHS)
        assert list(f["mask"].keys()) == ["train"]  # the options file was used
    assert (tmp / "out" / "acquisition_config.yaml").exists()
    assert (tmp / "out" / "postprocess_config.yaml").exists()


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


def test_robomimic_refuses_to_overwrite(workspace):
    tmp, cfg = workspace
    out = tmp / "out" / "dataset.h5"
    args = ["--config", str(cfg), "--format", "robomimic", "--input", str(tmp / "recordings"),
            "--output", str(out)]
    assert main(args) == 0
    before = out.read_bytes()
    assert main(args) == 2
    assert out.read_bytes() == before


def test_removed_option_exits_2(workspace, capsys):
    tmp, cfg = workspace
    (tmp / "postprocess_config.yaml").write_text("postprocess:\n  robomimic:\n    training_model: bc_rnn_gmm\n")
    assert main(["--config", str(cfg), "--format", "robomimic", "--input", str(tmp / "recordings"),
                 "--output", str(tmp / "out" / "d.h5")]) == 2
    assert "robomimic.training_model" in capsys.readouterr().err
