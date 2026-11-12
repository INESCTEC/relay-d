"""Follow and stop training runs started by ``relay_d.training.launcher``.

All state is on disk (``run.json``, ``pid``, ``exit_code`` in the run folder), so it is still there
after the GUI was closed and reopened.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from relay_d.training.launcher import RUNS_DIRNAME

STARTING, RUNNING, FINISHED, FAILED, STOPPED = "starting", "running", "finished", "failed", "stopped"
_STOP_CODES = {130, 143}  # Ctrl+C / SIGTERM
_START_GRACE_S = 30  # a terminal window can take a moment to open and write `pid`


@dataclass
class Run:
    path: Path
    meta: dict
    status: str

    @property
    def log(self) -> Path:
        return self.path / "train.log"


def _read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    """True if ``pid`` is still the run script (guards against a reused pid after a reboot)."""
    try:
        return b"run.sh" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def _status(run_dir: Path, meta: dict) -> str:
    code = _read_int(run_dir / "exit_code")
    if code is not None:
        return FINISHED if code == 0 else STOPPED if code in _STOP_CODES else FAILED
    pid = _read_int(run_dir / "pid")
    if pid is None:
        return STARTING if time.time() - meta.get("started", 0) < _START_GRACE_S else STOPPED
    return RUNNING if _alive(pid) else STOPPED  # no exit_code and the script is gone: it was killed


def read_run(run_dir) -> Run:
    run_dir = Path(run_dir)
    try:
        meta = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        meta = {"name": run_dir.name}
    return Run(run_dir, meta, _status(run_dir, meta))


def runs_root(path) -> Path:
    """The ``training_runs`` folder for ``path``: a Post-process output folder, or a dataset inside one."""
    path = Path(path)
    is_folder = path.is_dir() and not (path / "meta" / "info.json").exists()
    return (path if is_folder else path.parent) / RUNS_DIRNAME


def list_runs(path) -> List[Run]:
    """Runs recorded for ``path`` (see ``runs_root``), newest first."""
    root = runs_root(path)
    if not root.is_dir():
        return []
    runs = [read_run(d) for d in root.iterdir() if (d / "run.json").exists()]
    return sorted(runs, key=lambda r: r.meta.get("started", 0), reverse=True)


def stop(run_dir, grace: float = 10.0) -> bool:
    """Stop a running run (terminate, then kill after ``grace`` seconds). True if it is no longer running."""
    run_dir = Path(run_dir)
    pid = _read_int(run_dir / "pid")
    if pid is None or not _alive(pid):
        return True
    try:
        pgid = os.getpgid(pid)
        # Never signal our own group (a run started without its own session would take the caller down).
        send = (lambda sig: os.killpg(pgid, sig)) if pgid != os.getpgrp() else (lambda sig: os.kill(pid, sig))
        send(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _alive(pid):
            send(signal.SIGKILL)
            time.sleep(0.2)
    except ProcessLookupError:
        pass
    if not (run_dir / "exit_code").exists():
        (run_dir / "exit_code").write_text("143\n")  # so the run reads as "stopped", not "running"
    return not _alive(pid)
