"""Dedicated Python environments for LeRobot and robomimic, built by one command (Train tab: "Set up ... environment").

Each environment lives in its own folder under ``~/.relayd/envs/``, so the Pythons you already use (and their torch) are not
touched. The work runs in a terminal window (or in the background without a display); its state is kept in files next to the
folder, so the GUI can pick it up later even if it was closed meanwhile. The script is safe to run again: it only creates the
environment when it is missing, then installs / updates into it. No Qt and no ROS here.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from relay_d.training import launcher

PYTHON_VERSION = "3.12"  # dispatch needs rclpy (ROS Jazzy: Python 3.12), so every environment is 3.12
_STALE_SECONDS = 900

# lerobot: pinned, because launcher.py hard-codes its training module and command-line flags.
LEROBOT_VERSION = "0.6.1"

# robomimic: the source checkout version the code is tested with (PyPI stops at 0.3.0; diffusion_policy needs master).
# Installed with --no-deps: its pins (egl_probe, which only robosuite environments import) are replaced by the list below.
ROBOMIMIC_COMMIT = "e10526b9a40c78b41f1e37e60041dc0ec0a5f60f"
_ROBOMIMIC_DEPS = ("numpy", "h5py", "psutil", "tqdm", "termcolor", "tensorboard", "tensorboardX", "imageio", "imageio-ffmpeg",
                   "matplotlib", "huggingface_hub==0.23.4", "transformers==4.41.2", "diffusers==0.11.1",
                   "opencv-python-headless>=4.8.0,<4.13")  # cv2: lfd-inference-node decodes camera frames in this environment


@dataclass(frozen=True)
class EnvSpec:
    folder: str
    installs: Tuple[Tuple[str, ...], ...]  # one ``pip install`` argument list per step, in order
    check: str  # python code that must run in the finished environment


ENVS = {
    "lerobot": EnvSpec(
        "lerobot",
        (tuple(f"lerobot[{extra}]=={LEROBOT_VERSION}" for extra in ("training", "diffusion", "smolvla")),),  # one extra per policy family
        "import lerobot, torch; print('lerobot ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())",
    ),
    "robomimic": EnvSpec(
        "robomimic",
        (("torch", "torchvision"),
         ("--no-deps", f"robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@{ROBOMIMIC_COMMIT}"),
         _ROBOMIMIC_DEPS),
        "import robomimic.scripts.train, torch, cv2; print('robomimic ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())",
    ),
}

_SCRIPT = """#!/usr/bin/env bash
# Written by Relay-D. Builds the {fmt} environment (safe to run again) and records how it ended.
rm -f {exit_file}
echo $$ > {pid_file}
trap 'interrupted=1' INT
{{
echo "Relay-D: {fmt} environment in" {env_dir}  # env_dir is shlex-quoted: never inside double quotes
{steps}
}} 2>&1 | tee {log}
code=${{PIPESTATUS[0]}}
if [ -n "$interrupted" ]; then echo 130 > {exit_file}; exit 0; fi
echo $code > {exit_file}
echo
echo "Setup exited with code $code. Press Enter to close this window."
read -r _ || true
"""


def default_env_dir(fmt: str = "lerobot") -> Path:
    return Path(os.environ.get("RELAYD_ENVS_DIR") or Path.home() / ".relayd" / "envs") / ENVS[fmt].folder


def env_python(env_dir: Path) -> Path:
    return Path(env_dir) / "bin" / "python"


def _paths(env_dir: Path):
    env_dir = Path(env_dir)
    return env_dir.with_name(env_dir.name + ".setup.log"), env_dir.with_name(env_dir.name + ".setup.exit")


def _pid_file(env_dir: Path) -> Path:
    return Path(env_dir).with_name(Path(env_dir).name + ".setup.pid")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a killed script nobody waited for stays a zombie: it does not run any more
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def build_script(fmt: str, env_dir: Path, uv: Optional[str], base_python: str) -> str:
    """The bash script that makes ``env_dir`` (if it is not there yet) and installs the packages of ``fmt`` into it."""
    q, spec = shlex.quote, ENVS[fmt]
    py = q(str(env_python(env_dir)))
    if uv:
        create = f"{q(uv)} venv --python {PYTHON_VERSION} {q(str(env_dir))}"
        install = lambda args: f"{q(uv)} pip install --python {py} " + " ".join(q(a) for a in args)  # noqa: E731
    else:
        create = f"{q(base_python)} -m venv {q(str(env_dir))} && {py} -m pip install -U pip"
        install = lambda args: f"{py} -m pip install " + " ".join(q(a) for a in args)  # noqa: E731
    check = f"{py} -c {q(spec.check)}"
    # An environment that exists is kept (re-running updates it): `uv venv` would refuse, and a rebuild would be slow.
    steps = [f"[ -x {py} ] && echo 'environment exists, updating it' || {{ echo {q('$ ' + create)}; {create} || exit 1; }}"]
    steps += [f"echo {q('$ ' + cmd)}\n{cmd} || exit 1" for cmd in [install(a) for a in spec.installs] + [check]]
    log, exit_file = _paths(env_dir)
    return _SCRIPT.format(fmt=fmt, env_dir=q(str(env_dir)), steps="\n".join(steps), log=q(str(log)), exit_file=q(str(exit_file)),
                          pid_file=q(str(_pid_file(env_dir))))


def status(env_dir: Path) -> str:
    """``missing`` (nothing started), ``running``, ``failed`` or ``ready`` (finished with code 0 and the Python exists)."""
    log, exit_file = _paths(env_dir)
    if not exit_file.exists():
        if not log.exists():
            return "missing"
        # pip prints no progress into a pipe, so a long download leaves the log silent: ask the script's process.
        # Without a pid file (the script has not started yet), a log silent for a quarter of an hour belongs to a killed one.
        try:
            return "running" if _alive(int(_pid_file(env_dir).read_text())) else "failed"
        except (OSError, ValueError):
            return "running" if time.time() - log.stat().st_mtime < _STALE_SECONDS else "failed"
    code = exit_file.read_text().strip()
    return "ready" if code == "0" and env_python(env_dir).exists() else "failed"


def ready_python(fmt: str) -> Optional[str]:
    """The Python of the environment built for ``fmt``, if it is ready."""
    env_dir = default_env_dir(fmt)
    return str(env_python(env_dir)) if status(env_dir) == "ready" else None


def python_for(fmt: str) -> str:
    """The Python that trains / runs ``fmt``: the ``python_<fmt>`` setting, else the environment built here, else this one."""
    from relay_d.utils import user_settings

    return user_settings.get(f"python_{fmt}") or ready_python(fmt) or sys.executable


def log_tail(env_dir: Path, lines: int = 12) -> str:
    log, _ = _paths(env_dir)
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def start(fmt: str, env_dir: Optional[Path] = None, terminal: bool = True) -> Path:
    """Start building (or updating) the environment of ``fmt``. The process belongs to its terminal (or its own session).
    A build that is still running is left alone: a second one in the same folder would corrupt it."""
    env_dir = Path(env_dir or default_env_dir(fmt))
    if status(env_dir) == "running":
        return env_dir
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    log, exit_file = _paths(env_dir)
    for f in (log, exit_file, _pid_file(env_dir)):
        f.unlink(missing_ok=True)
    log.touch()  # "running" from now on
    script = env_dir.with_name(env_dir.name + ".setup.sh")
    script.write_text(build_script(fmt, env_dir, shutil.which("uv"), sys.executable))
    launcher.run_script(script, f"Relay-D: {fmt} setup", terminal)
    return env_dir
