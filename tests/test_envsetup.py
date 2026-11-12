"""LeRobot / robomimic environment setup: the scripts it writes and how their state is read (nothing is installed here)."""

import os
import subprocess
import time

import pytest

from relay_d.training import envsetup, launcher


@pytest.mark.parametrize("fmt", ["lerobot", "robomimic"])
def test_scripts_quote_paths_skip_an_existing_env_and_are_valid_bash(tmp_path, fmt):
    env = tmp_path / "my envs" / fmt
    uv = envsetup.build_script(fmt, env, "/usr/bin/uv", "python3")
    plain = envsetup.build_script(fmt, env, None, "/usr/bin/python3")
    assert f"[ -x '{env}/bin/python' ]" in uv and "environment exists, updating it" in uv  # re-running must not recreate it
    assert "uv venv --python 3.12" in uv and f"uv pip install --python '{env}/bin/python'" in uv
    assert "/usr/bin/python3 -m venv" in plain and "-m pip install -U pip" in plain and "uv pip" not in plain
    assert envsetup.ENVS[fmt].check.split(";")[0].split(" ", 1)[1] in uv  # the final import check is in the script
    for script in (uv, plain):
        assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def test_package_lists_differ_per_format(tmp_path):
    lerobot = envsetup.build_script("lerobot", tmp_path / "l", "/usr/bin/uv", "python3")
    robomimic = envsetup.build_script("robomimic", tmp_path / "r", "/usr/bin/uv", "python3")
    v = envsetup.LEROBOT_VERSION  # pinned: launcher.py hard-codes this release's module and flags
    assert f"'lerobot[training]=={v}' 'lerobot[diffusion]=={v}' 'lerobot[smolvla]=={v}'" in lerobot and "robomimic" not in lerobot.replace("Relay-D", "")
    assert f"git+https://github.com/ARISE-Initiative/robomimic.git@{envsetup.ROBOMIMIC_COMMIT}" in robomimic
    assert "--no-deps" in robomimic and "diffusers==0.11.1" in robomimic and "torch torchvision" in robomimic
    assert "import robomimic.scripts.train" in robomimic
    assert "opencv-python-headless" in robomimic and "cv2" in robomimic  # inference decodes camera frames there


def test_existing_environment_is_not_recreated_by_the_script(tmp_path):
    """Run the real script with a fake `uv`: the second run must skip `uv venv` (the bug: it failed on an existing env)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    log = tmp_path / "calls"
    uv.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {log}\nif [ \"$1\" = venv ]; then mkdir -p \"${{@: -1}}/bin\" && "
                  f"printf '#!/usr/bin/env bash\\nexit 0\\n' > \"${{@: -1}}/bin/python\" && chmod +x \"${{@: -1}}/bin/python\"; fi\n")
    uv.chmod(0o755)
    env = tmp_path / "envs" / "lerobot"
    script = tmp_path / "setup.sh"
    script.write_text(envsetup.build_script("lerobot", env, str(uv), "python3"))
    for _ in range(2):
        out = subprocess.run(["bash", str(script)], input="\n", capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stdout + out.stderr
        assert envsetup.status(env) == "ready"
    calls = log.read_text().splitlines()
    assert sum(c.startswith("venv") for c in calls) == 1 and sum(c.startswith("pip install") for c in calls) == 2


def test_status_follows_the_files(tmp_path):
    env = tmp_path / "lerobot"
    log, exit_file = envsetup._paths(env)
    assert envsetup.status(env) == "missing"
    log.write_text("downloading torch\n")
    assert envsetup.status(env) == "running"
    old = time.time() - 2 * envsetup._STALE_SECONDS  # a log that stopped growing: its process is gone
    os.utime(log, (old, old))
    assert envsetup.status(env) == "failed"
    exit_file.write_text("0\n")
    assert envsetup.status(env) == "failed"  # exit 0 but no Python was built
    envsetup.env_python(env).parent.mkdir(parents=True)
    envsetup.env_python(env).write_text("")
    assert envsetup.status(env) == "ready"
    exit_file.write_text("1\n")
    assert envsetup.status(env) == "failed" and envsetup.log_tail(env, 1) == "downloading torch"


def test_a_silent_log_of_a_live_build_is_still_running_and_is_not_started_twice(tmp_path, monkeypatch):
    env = tmp_path / "lerobot"
    log, _ = envsetup._paths(env)
    log.write_text("downloading torch\n")
    old = time.time() - 2 * envsetup._STALE_SECONDS  # pip prints nothing into a pipe: the log stays silent
    os.utime(log, (old, old))
    envsetup._pid_file(env).write_text(str(os.getpid()))  # a live process
    assert envsetup.status(env) == "running"
    started = []
    monkeypatch.setattr(envsetup.launcher, "run_script", lambda *a: started.append(a))
    envsetup.start("lerobot", env)
    assert started == [] and log.read_text() == "downloading torch\n"  # left alone: a second build would corrupt the venv
    dead = subprocess.Popen(["true"])
    dead.wait()
    envsetup._pid_file(env).write_text(str(dead.pid))
    assert envsetup.status(env) == "failed"  # the script's process is gone
    envsetup.start("lerobot", env)
    assert len(started) == 1 and not envsetup._pid_file(env).exists()


def test_python_for_prefers_the_setting_then_the_built_environment(tmp_path, monkeypatch, settings_file):
    from relay_d.utils import user_settings

    monkeypatch.setenv("RELAYD_ENVS_DIR", str(tmp_path))
    assert envsetup.python_for("robomimic") == __import__("sys").executable  # nothing built, nothing set
    env = envsetup.default_env_dir("robomimic")
    envsetup.env_python(env).parent.mkdir(parents=True)
    envsetup.env_python(env).write_text("")
    envsetup._paths(env)[1].write_text("0\n")
    assert envsetup.python_for("robomimic") == str(env / "bin" / "python") and envsetup.ready_python("lerobot") is None
    user_settings.save({"python_robomimic": "/my/python"})
    assert envsetup.python_for("robomimic") == "/my/python"


def test_not_installed_message_tells_what_to_do(monkeypatch):
    class Done:
        stdout = '{"package": false, "torch": null, "torchvision": null, "cuda": false, "import_error": null}'

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: Done())
    for fmt in (launcher.LEROBOT, launcher.ROBOMIMIC):
        error = launcher.check_environment("/usr/bin/python3", fmt)["error"]
        assert f"is not installed for /usr/bin/python3" in error and f"'Set up {fmt} environment'" in error


def test_env_dir_with_shell_characters_is_printed_not_run(tmp_path):
    env = tmp_path / "a$(touch pwned)"
    line = next(x for x in envsetup.build_script("lerobot", env, None, "python3").splitlines() if x.startswith("echo \"Relay-D"))
    out = subprocess.run(["bash", "-c", line], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert str(env) in out.stdout and not (tmp_path / "pwned").exists()
