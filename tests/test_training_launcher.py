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


def test_a_relative_folder_gives_absolute_paths(folder, monkeypatch):
    # run.sh cd's into the run folder before training: a relative dataset path would no longer resolve there
    monkeypatch.chdir(folder.parent)
    found = launcher.find_datasets(folder.name)
    assert found and all(d.path.is_absolute() for d in found)
    cmd = launcher.build_command(TrainJob(_robomimic(folder.name)), launcher.new_run_dir(found[0].base, ROBOMIMIC), write=False)
    assert all(os.path.isabs(a) for a in cmd[cmd.index("--config") + 1:cmd.index("--name")] if a != "--dataset")


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


def test_stopping_a_starting_run_waits_for_its_pid(folder, tmp_path):
    """Stop on a run whose terminal has not written ``pid`` yet must not report it stopped while it starts later."""
    import subprocess
    import threading

    run_dir = folder / "training_runs" / "r"
    run_dir.mkdir(parents=True)
    script = run_dir / "run.sh"
    script.write_text("echo $$ > pid\nsleep 60\n")
    assert runs.stop(run_dir, grace=0.3) is False  # no pid within grace: it may still start
    assert not (run_dir / "exit_code").exists()
    late = threading.Timer(0.5, lambda: subprocess.Popen(["bash", str(script)], cwd=run_dir, start_new_session=True))
    late.start()
    assert runs.stop(run_dir, grace=5) is True  # the pid appeared within grace: stopped as usual
    assert (run_dir / "exit_code").read_text().strip() == "143"


def test_a_reused_pid_is_not_the_run(tmp_path):
    (tmp_path / "pid").write_text(str(os.getpid()))  # after a reboot the pid belongs to some other process
    assert runs._status(tmp_path, {}) == runs.STOPPED


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


def test_the_config_file_next_to_the_dataset_is_used(tmp_path):
    _robomimic_file(tmp_path / "a.h5")
    ds = launcher.find_datasets(tmp_path)[0]
    (tmp_path / "other_acquisition_config.yaml").write_text("config: {}\n")  # not this dataset's
    assert launcher._config_file(ds, "acquisition_config.yaml") is None
    (tmp_path / "acquisition_config.yaml").write_text("config: {}\n")
    assert launcher._config_file(ds, "acquisition_config.yaml").name == "acquisition_config.yaml"


def test_robomimic_policy_is_chosen_on_the_train_tab_not_by_postprocess_options(folder):
    """robomimic.training_model was removed: a stray one in the folder changes nothing; no policy = the first template."""
    (folder / "postprocess_config.yaml").write_text("postprocess:\n  robomimic:\n    training_model: diffusion_policy\n")
    ds = _robomimic(folder)
    assert launcher._robomimic_policy(ds, None) == launcher.ROBOMIMIC_POLICIES[0]
    assert launcher._robomimic_policy(ds, "diffusion_policy") == "diffusion_policy"


def test_ctrl_c_in_the_training_window_ends_the_run_and_closes_the_window(folder, tmp_path):
    """Ctrl+C goes to the whole foreground group (training, tee and the run script). The script must record it and exit."""
    import signal

    py = _fake_python(tmp_path, "sleep 60")
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"), terminal=False)
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.RUNNING)
    pid = int((run_dir / "pid").read_text())
    os.killpg(os.getpgid(pid), signal.SIGINT)  # what the terminal does for Ctrl+C
    assert wait_until(lambda: not runs._alive(run_dir, pid), seconds=5)  # the script exited by itself: no "Press Enter" wait
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


def test_robomimic_validation_split_smaller_than_a_batch_turns_validation_off(tmp_path):
    _robomimic_file(tmp_path / "d.h5", frames=150)  # 150 train frames (demo_0), 150 valid (demo_1)
    ds = launcher.find_datasets(tmp_path)[0]
    launcher.build_command(TrainJob(ds, "py", "bc_rnn_gmm", {"batch_size": 100}), tmp_path / "ok")
    assert json.loads((tmp_path / "ok" / "robomimic_config.json").read_text())["experiment"]["validate"] is True
    ds.summary.update(train_frames=400, valid_frames=100)  # e.g. 5 demos x 100 frames, val_frac 0.2
    launcher.build_command(TrainJob(ds, "py", "bc_rnn_gmm", {"batch_size": 256}), tmp_path / "run")
    cfg = json.loads((tmp_path / "run" / "robomimic_config.json").read_text())
    assert cfg["experiment"]["validate"] is False  # was: StopIteration after epoch 1
    assert cfg["train"]["hdf5_validation_filter_key"] is None
    assert launcher.find_datasets(tmp_path)[0].summary["valid_frames"] == 150  # counted from mask/valid


@pytest.mark.parametrize("policy", launcher.ROBOMIMIC_POLICIES)
def test_robomimic_does_not_normalize_each_pixel(folder, policy):
    """robomimic's obs normalization is per element: each pixel divided by its own std (floor 0.001), so a near-constant
    background turns a 0.02 brightness change into ~20 std. robomimic's own templates leave it off."""
    run_dir = folder / "training_runs" / policy
    launcher.build_command(TrainJob(_robomimic(folder), "py", policy), run_dir)
    assert json.loads((run_dir / "robomimic_config.json").read_text())["train"]["hdf5_normalize_obs"] is False


def test_ctrl_c_keeps_the_trainings_last_output_in_the_log(folder, tmp_path):
    """tee is in the same foreground group: without -i it died on Ctrl+C and the training's last lines
    (its traceback or final checkpoint save) never reached train.log."""
    import signal

    py = _fake_python(tmp_path, "trap 'echo saved-final-checkpoint; exit 1' INT\nwhile :; do sleep 0.05; done")
    run_dir = launcher.start(TrainJob(_robomimic(folder), py, "bc_rnn_gmm"), terminal=False)
    assert wait_until(lambda: runs.read_run(run_dir).status == runs.RUNNING)
    pid = int((run_dir / "pid").read_text())
    os.killpg(os.getpgid(pid), signal.SIGINT)
    assert wait_until(lambda: not runs._alive(run_dir, pid), seconds=5)
    assert "saved-final-checkpoint" in (run_dir / "train.log").read_text()


def test_a_run_that_survives_stop_is_not_marked_stopped(folder, tmp_path, monkeypatch):
    """A process that outlives SIGTERM and SIGKILL (e.g. stuck in the kernel) must not read as stopped."""
    run_dir = folder / "training_runs" / "stuck"
    run_dir.mkdir(parents=True)
    (run_dir / "pid").write_text(f"{os.getpid()}\n")
    monkeypatch.setattr(runs, "_alive", lambda d, pid: True)  # survives every signal
    monkeypatch.setattr(runs.os, "getpgid", lambda pid: -1)
    monkeypatch.setattr(runs.os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(runs.os, "kill", lambda pid, sig: None)
    assert runs.stop(run_dir, grace=0.2) is False
    assert not (run_dir / "exit_code").exists()  # was "143": the run table showed "stopped" for a live run
    assert runs.read_run(run_dir).status == runs.RUNNING
