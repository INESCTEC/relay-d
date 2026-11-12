"""Run the inference node in the Python environment that has the policy package (LeRobot / robomimic).

``lfd-inference-node`` starts in whichever Python launched it, which usually is not the one with LeRobot or robomimic. If the package
of the backend is missing here, the node starts again in the environment built by the Train tab (or the ``python_<backend>``
setting): same process, same arguments, ROS python packages and Relay-D passed on through ``PYTHONPATH``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, List, Mapping, Optional

GUARD = "RELAYD_REEXEC"  # set in the new process: never switch twice
_PACKAGE = {"lerobot": "lerobot", "robomimic": "robomimic"}


def _package_root(module: str, find_spec=importlib.util.find_spec) -> Optional[str]:
    """Folder that holds the top-level package ``module`` (its site-packages / source root), or None if it is not importable."""
    try:
        spec = find_spec(module)
    except (ImportError, ValueError):
        return None
    return str(Path(spec.origin).parent.parent) if spec and spec.origin else None


def reexec_if_needed(
    backend: str,
    argv: List[str],
    environ: Mapping = os.environ,
    current: str = sys.executable,
    python_for: Optional[Callable[[str], str]] = None,
    find_spec=importlib.util.find_spec,
    execve=os.execve,
    echo=lambda msg: print(msg, file=sys.stderr),
) -> bool:
    """Replace this process by ``<python> *argv`` when ``backend``'s package is not importable here and another Python is configured.

    Returns False when nothing was done (package present, already re-executed, no other Python); does not return after a switch.
    """
    package = _PACKAGE.get(backend)
    if package is None or environ.get(GUARD) or find_spec(package) is not None:
        return False
    if python_for is None:
        from relay_d.training.envsetup import python_for  # noqa: PLC0415 (only needed here)
    target = python_for(backend)
    if not target or os.path.abspath(target) == os.path.abspath(current) or not os.path.exists(target):
        return False
    paths = [p for p in (_package_root("rclpy", find_spec), _package_root("relay_d", find_spec), environ.get("PYTHONPATH")) if p]
    env = dict(environ, PYTHONPATH=os.pathsep.join(paths), **{GUARD: "1"})
    echo(f"'{package}' is not installed in {current}: running the node with {target}")
    execve(target, [target, *argv], env)
    return True
