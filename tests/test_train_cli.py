"""relay-d-train: arguments, dry run, headless start, list, stop."""

import stat

import h5py
import numpy as np
import pytest

from helpers import wait_until
from relay_d.training import runs
from relay_d.training.train_cli import EXIT_OK, EXIT_USAGE, main


@pytest.fixture
def folder(tmp_path):
    with h5py.File(tmp_path / "robomimic_dataset_1.h5", "w") as f:
        demo = f.create_group("data/demo_0")
        demo.attrs["num_samples"] = 4
        demo.create_dataset("obs/robot0_eef_pos", data=np.zeros((4, 3), np.float32))
        f.create_dataset("mask/train", data=[b"demo_0"])
        f.create_dataset("mask/valid", data=[b"demo_0"])
    return tmp_path


def test_dry_run_prints_the_command_and_writes_nothing(folder, capsys):
    assert main(["--folder", str(folder), "--epochs", "5", "--policy", "bc_rnn_gmm", "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "robomimic.scripts.train" in out and "--dataset" in out
    assert not (folder / "training_runs").exists()


@pytest.mark.parametrize("args, message", [
    (["--folder", "NOPE"], "not found"),
    (["--steps", "5"], "LeRobot options"),
    (["--epochs", "0", "--dry-run"], "greater than 0"),
    (["--format", "lerobot"], "no lerobot dataset"),
])
def test_usage_errors(folder, capsys, args, message):
    args = [str(folder / "NOPE") if a == "NOPE" else a for a in args]
    if "--folder" not in args:
        args = ["--folder", str(folder), *args]
    assert main(args) == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_start_list_and_stop_headless(folder, tmp_path, capsys):
    py = tmp_path / "fakepython"
    py.write_text("#!/bin/sh\nsleep 60\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    assert main(["--folder", str(folder), "--python", str(py), "--no-terminal"]) == EXIT_OK
    run = runs.list_runs(folder)[0]
    assert wait_until(lambda: runs.read_run(run.path).status == runs.RUNNING)
    capsys.readouterr()
    assert main(["--folder", str(folder), "--list"]) == EXIT_OK
    assert run.path.name in capsys.readouterr().out
    assert main(["--folder", str(folder), "--stop", run.path.name]) == EXIT_OK
    assert runs.read_run(run.path).status == runs.STOPPED
    assert main(["--folder", str(folder), "--stop", "nope"]) == EXIT_USAGE
