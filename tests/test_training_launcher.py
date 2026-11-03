"""Training launcher: dataset detection, commands for robomimic / LeRobot, detached runs (no GUI, no ROS)."""

import json
import os
import shlex
import stat

import h5py
import numpy as np
import pytest

from helpers import wait_until
from relay_d.training import launcher, runs
from relay_d.training.launcher import LEROBOT, ROBOMIMIC, TrainingError, TrainJob


def _robomimic_file(path, valid=True, frames=150):
    """Two demos; ``frames`` is the declared length of each (enough for robomimic's default batch size of 100)."""
    with h5py.File(path, "w") as f:
        for i in range(2):
            demo = f.create_group(f"data/demo_{i}")
            demo.attrs["num_samples"] = frames
            demo.create_dataset("obs/robot0_eef_pos", data=np.zeros((5, 3), np.float32))
            demo.create_dataset("obs/agentview_image", data=np.zeros((5, 4, 4, 3), np.uint8))
        f.create_dataset("mask/train", data=[b"demo_0"])
        if valid:
            f.create_dataset("mask/valid", data=[b"demo_1"])


def _lerobot_dir(path):
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(json.dumps({
        "fps": 10, "total_episodes": 3, "total_frames": 90, "robot_type": "arm",
        "features": {"action": {"dtype": "float32"}, "observation.images.cam": {"dtype": "video"}},
    }))


def _fake_python(tmp_path, body):
    exe = tmp_path / "fakepython"
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


@pytest.fixture
def folder(tmp_path):
    """A Post-process style output folder with one robomimic file and one LeRobot dataset."""
    _robomimic_file(tmp_path / "robomimic_dataset_1.h5")
    _lerobot_dir(tmp_path / "lerobot_dataset_1")
    with h5py.File(tmp_path / "raw_recording.h5", "w") as f:  # a raw recording: has no mask, not trainable
        f.create_group("data/demo_0")
    return tmp_path


def test_find_datasets_in_a_folder_or_directly(folder):
    found = launcher.find_datasets(folder)
    assert sorted(d.format for d in found) == [LEROBOT, ROBOMIMIC]
    rm = next(d for d in found if d.format == ROBOMIMIC)
    assert rm.summary["episodes"] == 2 and rm.summary["frames"] == 300 and rm.summary["train_frames"] == 150 and rm.summary["has_valid"]
    lr = next(d for d in found if d.format == LEROBOT)
    assert lr.summary["cameras"] == ["observation.images.cam"] and lr.summary["fps"] == 10
    assert launcher.find_datasets(folder / "lerobot_dataset_1")[0].format == LEROBOT
    assert launcher.find_datasets(folder / "robomimic_dataset_1.h5")[0].format == ROBOMIMIC
    assert launcher.find_datasets(folder / "raw_recording.h5") == []


def _robomimic(folder):
    return next(d for d in launcher.find_datasets(folder) if d.format == ROBOMIMIC)


def test_robomimic_command_and_config(folder, tmp_path):
    run_dir = folder / "training_runs" / "r1"
    job = TrainJob(_robomimic(folder), "py", "bc_rnn_gmm",
                   {"epochs": 7, "batch_size": 16, "learning_rate": 0.001, "device": "cpu"}, "--debug")
    argv = launcher.build_command(job, run_dir)
    cfg = json.loads((run_dir / "robomimic_config.json").read_text())
    assert cfg["train"]["data"].endswith("robomimic_dataset_1.h5")
    assert cfg["train"]["output_dir"] == str(run_dir / "output")
    assert (cfg["train"]["num_epochs"], cfg["train"]["batch_size"], cfg["train"]["cuda"]) == (7, 16, False)
    assert cfg["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] == 0.001
    assert cfg["experiment"]["name"] == "r1" and cfg["experiment"]["validate"] is True
    assert cfg["observation"]["modalities"]["obs"]["rgb"] == ["agentview_image"]  # uint8 -> rgb
    assert cfg["observation"]["modalities"]["obs"]["low_dim"] == ["robot0_eef_pos"]
    assert argv[:3] == ["py", "-m", "robomimic.scripts.train"] and argv[-1] == "--debug"
    assert argv[argv.index("--config") + 1] == str(run_dir / "robomimic_config.json")


def test_robomimic_without_validation_split_turns_validation_off(tmp_path):
    _robomimic_file(tmp_path / "d.h5", valid=False)
    ds = launcher.find_datasets(tmp_path)[0]
    launcher.build_command(TrainJob(ds, "py", "bc_rnn_gmm"), tmp_path / "run")
    assert json.loads((tmp_path / "run" / "robomimic_config.json").read_text())["experiment"]["validate"] is False
    assert json.loads((tmp_path / "run" / "robomimic_config.json").read_text())["train"]["hdf5_validation_filter_key"] is None


def test_robomimic_uses_the_folders_saved_config_when_no_policy_is_given(folder):
    saved = {"train": {"data": None, "num_epochs": 3}, "experiment": {}, "marker": "from-folder"}
    (folder / "robomimic_config.json").write_text(json.dumps(saved))
    run_dir = folder / "training_runs" / "r"
    launcher.build_command(TrainJob(_robomimic(folder), "py"), run_dir)
    cfg = json.loads((run_dir / "robomimic_config.json").read_text())
    assert cfg["marker"] == "from-folder" and cfg["train"]["num_epochs"] == 3
    assert launcher.default_params(_robomimic(folder))["epochs"] == 3


def test_lerobot_command(folder):
    ds = next(d for d in launcher.find_datasets(folder) if d.format == LEROBOT)
    argv = launcher.build_command(TrainJob(ds, "py", "diffusion", {"steps": 500, "device": "cpu"}, "--log_freq=50"),
                                  folder / "training_runs" / "r")
    assert argv[:3] == ["py", "-m", "lerobot.scripts.lerobot_train"]
    for flag in ("--policy.type=diffusion", "--steps=500", "--policy.device=cpu", "--batch_size=8",
                 "--policy.optimizer_lr=0.0001", "--policy.push_to_hub=false", f"--dataset.root={ds.path}"):
        assert flag in argv
    assert argv[-1] == "--log_freq=50" and "--wandb.enable=false" not in argv


@pytest.mark.parametrize("params, message", [
    ({"epochs": 0}, "greater than 0"),
    ({"epochs": "many"}, "must be int"),
    ({"steps": 5}, "unknown parameter"),  # a LeRobot parameter on a robomimic dataset
    ({"device": "tpu"}, "cuda"),
])
def test_bad_parameters_are_rejected(folder, params, message):
    with pytest.raises(TrainingError, match=message):
        launcher.build_command(TrainJob(_robomimic(folder), "py", "bc_rnn_gmm", params), folder / "r", write=False)


def test_unknown_policy_and_unbalanced_extra_args(folder):
    with pytest.raises(TrainingError, match="unknown robomimic policy"):
        launcher.build_command(TrainJob(_robomimic(folder), "py", "nope"), folder / "r", write=False)
    with pytest.raises(TrainingError, match="extra arguments"):
        launcher.build_command(TrainJob(_robomimic(folder), "py", "bc_rnn_gmm", {}, "--x 'oops"), folder / "r", write=False)


def test_headless_run_finishes_and_is_listed(folder, tmp_path):
    py = _fake_python(tmp_path, 'echo "trained with: $@"')
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm", {"epochs": 2}), terminal=False)
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.FINISHED)
    assert "trained with: -m robomimic.scripts.train" in (run_dir / "train.log").read_text()
    listed = runs.list_runs(folder)
    assert [r.path for r in listed] == [run_dir] and listed[0].meta["format"] == ROBOMIMIC
    assert runs.list_runs(folder / "robomimic_dataset_1.h5")[0].path == run_dir  # also found from the dataset path


def test_failed_run_is_reported_as_failed(folder, tmp_path):
    py = _fake_python(tmp_path, "echo boom; exit 3")
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"), terminal=False)
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.FAILED)


def test_running_run_can_be_stopped(folder, tmp_path):
    py = _fake_python(tmp_path, "sleep 60")
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"), terminal=False)
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.RUNNING)
    pid = int((run_dir / "pid").read_text())
    assert runs.stop(run_dir, grace=5) is True
    assert runs.read_run(run_dir).status == runs.STOPPED
    assert not os.path.exists(f"/proc/{pid}") or b"run.sh" not in open(f"/proc/{pid}/cmdline", "rb").read()


def test_start_validates_before_creating_anything(folder):
    with pytest.raises(TrainingError):
        launcher.start(TrainJob(_robomimic(folder), "py", "bc_rnn_gmm", {"epochs": -1}), terminal=False)
    assert not (folder / "training_runs").exists()


def test_check_environment(tmp_path):
    ok = _fake_python(tmp_path, """echo '{"package": true, "torch": "2.1", "cuda": false}'""")
    assert launcher.check_environment(ok, ROBOMIMIC)["ok"] is True
    missing = tmp_path / "missing"
    missing.mkdir()
    no_pkg = _fake_python(missing, """echo '{"package": false, "torch": "2.1", "cuda": false}'""")
    result = launcher.check_environment(no_pkg, LEROBOT)
    assert result["ok"] is False and "'lerobot' is not installed" in result["error"]
    broken = tmp_path / "broken"
    broken.mkdir()
    bad_import = _fake_python(broken, """echo '{"package": true, "torch": "2.1", "cuda": false, "import_error": "RuntimeError: operator torchvision::nms does not exist"}'""")
    result = launcher.check_environment(bad_import, ROBOMIMIC)
    assert result["ok"] is False and "cannot be imported: RuntimeError: operator torchvision::nms" in result["error"]
    assert launcher.check_environment(str(tmp_path / "does-not-exist"), ROBOMIMIC)["ok"] is False


def test_smolvla_is_fine_tuned_from_the_pretrained_base(folder):
    ds = next(d for d in launcher.find_datasets(folder) if d.format == LEROBOT)
    argv = launcher.build_command(TrainJob(ds, "py", "smolvla", {"device": "cuda"}), folder / "training_runs" / "s")
    assert "--policy.path=lerobot/smolvla_base" in argv and not any(a.startswith("--policy.type") for a in argv)
    assert "--policy.optimizer_lr=0.0001" in argv and "--policy.push_to_hub=false" in argv
    # the pretrained model knows its cameras as camera1..3: the dataset's are renamed to them, in order
    assert '--rename_map={"observation.images.cam": "observation.images.camera1"}' in argv
    ds.summary["cameras"] = [f"observation.images.c{i}" for i in range(4)]
    with pytest.raises(TrainingError, match="at most 3 cameras"):
        launcher.build_command(TrainJob(ds, "py", "smolvla"), folder / "training_runs" / "t")
    act = launcher.build_command(TrainJob(ds, "py", "act"), folder / "training_runs" / "a")
    assert "--policy.type=act" in act and not any(a.startswith("--policy.path") for a in act)  # from scratch, as before


def test_lerobot_dataset_without_camera_is_rejected(tmp_path):
    ds_dir = tmp_path / "state_only"
    (ds_dir / "meta").mkdir(parents=True)
    (ds_dir / "meta" / "info.json").write_text(json.dumps({"fps": 10, "features": {"action": {"dtype": "float32"}}}))
    ds = launcher.find_datasets(ds_dir)[0]
    with pytest.raises(TrainingError, match="no camera"):
        launcher.build_command(TrainJob(ds, "py", "act"), tmp_path / "r", write=False)


def _fake_terminals(tmp_path, monkeypatch, names_ok, names_failing):
    """Put fake terminal emulators first on PATH: the OK ones run the command after -e / --, the others fail."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in names_ok:
        (bin_dir / name).write_text('#!/bin/sh\nwhile [ "$1" != "-e" ] && [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n')
    for name in names_failing:
        (bin_dir / name).write_text("#!/bin/sh\nexit 1\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("DISPLAY", ":99")


def test_terminal_that_fails_is_skipped_for_the_next_one(folder, tmp_path, monkeypatch):
    _fake_terminals(tmp_path, monkeypatch, ["x-terminal-emulator"], ["gnome-terminal", "konsole", "xterm"])
    py = _fake_python(tmp_path, 'echo "trained"')
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"))
    assert json.loads((run_dir / "run.json").read_text())["terminal"] == "x-terminal-emulator"
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.FINISHED)


def test_run_falls_back_to_background_when_every_terminal_fails(folder, tmp_path, monkeypatch):
    _fake_terminals(tmp_path, monkeypatch, [], ["gnome-terminal", "x-terminal-emulator", "konsole", "xterm"])
    py = _fake_python(tmp_path, 'echo "trained"')
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"))
    assert json.loads((run_dir / "run.json").read_text())["terminal"] is None
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.FINISHED)


def test_the_config_files_of_the_selected_dataset_are_used(tmp_path):
    """relay-d-convert writes <output>_acquisition_config.yaml, so one folder can hold several datasets."""
    for name, model in (("a", "bc_rnn_gmm"), ("b", "diffusion_policy")):
        _robomimic_file(tmp_path / f"{name}.h5")
        (tmp_path / f"{name}_postprocess_config.yaml").write_text(f"postprocess:\n  robomimic:\n    training_model: {model}\n")
    by_name = {d.path.stem: d for d in launcher.find_datasets(tmp_path)}
    assert launcher._robomimic_policy(by_name["b"], None) == "diffusion_policy"  # not the alphabetically first file
    assert launcher._robomimic_policy(by_name["a"], None) == "bc_rnn_gmm"
    (tmp_path / "postprocess_config.yaml").write_text("postprocess:\n  robomimic:\n    training_model: bc_transformer_gmm\n")
    _robomimic_file(tmp_path / "c.h5")
    c = next(d for d in launcher.find_datasets(tmp_path) if d.path.stem == "c")
    assert launcher._robomimic_policy(c, None) == "bc_transformer_gmm"  # no c_* file: the plain one the GUI writes


def test_ctrl_c_in_the_training_window_ends_the_run_and_closes_the_window(folder, tmp_path):
    """Ctrl+C goes to the whole foreground group (training, tee and the run script). The script must record it and exit."""
    import signal

    py = _fake_python(tmp_path, "sleep 60")
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"), terminal=False)
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.RUNNING)
    pid = int((run_dir / "pid").read_text())
    os.killpg(os.getpgid(pid), signal.SIGINT)  # what the terminal does for Ctrl+C
    assert wait_until(lambda: not runs._alive(pid), seconds=5)  # the script exited by itself: no "Press Enter" wait
    assert (run_dir / "exit_code").read_text().strip() == "130"
    assert runs.read_run(run_dir).status == runs.STOPPED


def test_robomimic_batch_larger_than_the_training_data_is_refused_and_the_default_fits(tmp_path):
    _robomimic_file(tmp_path / "d.h5", frames=5)  # 5 training frames (mask/train = demo_0), 10 in total
    ds = launcher.find_datasets(tmp_path)[0]
    assert ds.summary["train_frames"] == 5 and ds.summary["frames"] == 10
    assert launcher.default_params(ds, "bc_rnn_gmm")["batch_size"] == 5  # the template's 100 would make no batch
    with pytest.raises(TrainingError, match="5 frames, fewer than the batch size 6"):  # an explicit choice that cannot work
        launcher.build_command(TrainJob(ds, "py", "bc_rnn_gmm", {"batch_size": 6}), tmp_path / "run")
    launcher.build_command(TrainJob(ds, "py", "bc_rnn_gmm", {}), tmp_path / "run2")  # nothing chosen: the template's 100 is capped
    assert json.loads((tmp_path / "run2" / "robomimic_config.json").read_text())["train"]["batch_size"] == 5
    launcher.build_command(TrainJob(ds, "py", "bc_rnn_gmm", {"batch_size": 5}), tmp_path / "run")  # exactly one batch: fine


def test_run_script_reports_a_failure_that_robomimic_swallows(tmp_path):
    """robomimic prints 'run failed with error' and exits 0: the run must still be recorded as failed."""
    import subprocess

    def run(body):
        d = tmp_path / body.replace(" ", "_")[:20]
        d.mkdir()
        script = d / "run.sh"
        command = f"bash -c {shlex.quote(body)}"
        script.write_text(launcher._RUN_SCRIPT.format(name="r", command=command, quoted=shlex.quote(command)))
        subprocess.run(["bash", str(script)], input="\n", capture_output=True, text=True, timeout=30)
        return (d / "exit_code").read_text().strip()

    assert run("echo all good") == "0"
    assert run("echo run failed with error: x") == "1"
    assert run("exit 3") == "3"
