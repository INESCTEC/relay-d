"""Launches/tears down the robot_state_publisher + rviz2 subprocess pair that
visualizes DataPlayer's republished /playback/* topics.

rviz2 has no practical Python embedding API for its render panel, so it is
launched as its own external OS process/window rather than inside the PyQt5
app.

Both nodes are launched via `ros2 run <pkg> <exe>` rather than invoking the
executable directly - robot_state_publisher's binary is not on PATH after
sourcing ROS (only reachable through `ros2 run` or its full lib/ path),
unlike rviz2 which happens to be on PATH. `ros2 run` itself forks a real
child process for the requested executable rather than exec'ing into it, so
terminating just the `ros2 run` wrapper leaves the actual node running as an
orphan - each process is therefore started in its own new session
(`start_new_session=True`) and torn down by signaling the whole process
group, which reliably takes down both the wrapper and its child.
"""

import os
import signal
import subprocess
import tempfile

from relay_d.utils.coloring_logger import logger

_RVIZ_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "playback_default.rviz")
_FIXED_FRAME_PLACEHOLDER = "__PLAYBACK_FIXED_FRAME__"

# robot_state_publisher's robot_description/joint_states use relative topic
# names, so a plain namespace remap moves them under /playback. TF is
# published on the absolute "/tf" regardless of namespace, so it needs an
# explicit topic remap on every process that touches it here.
_TF_REMAP_ARGS = ["-r", "/tf:=/playback/tf", "-r", "/tf_static:=/playback/tf_static"]

_REQUIRED_PACKAGES = ("robot_state_publisher", "rviz2")


class PlaybackEnvironmentError(Exception):
    pass


def _package_available(pkg: str) -> bool:
    try:
        result = subprocess.run(
            ["ros2", "pkg", "executables", pkg],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and pkg in result.stdout
    except Exception:
        return False


class PlaybackProcessManager:
    def __init__(self):
        self._rsp_proc = None
        self._rviz_proc = None
        self._rviz_config_path = None

    def is_running(self) -> bool:
        return (self._rsp_proc is not None and self._rsp_proc.poll() is None) or (
            self._rviz_proc is not None and self._rviz_proc.poll() is None
        )

    def start(self, urdf_xml: str, fixed_frame: str = "base_link"):
        if self.is_running():
            return

        missing = [pkg for pkg in _REQUIRED_PACKAGES if not _package_available(pkg)]
        if missing:
            pkgs = " ".join(f"ros-<distro>-{m.replace('_', '-')}" for m in missing)
            raise PlaybackEnvironmentError(
                f"Missing required ROS2 package(s): {', '.join(missing)}. "
                f"Install them (e.g. `sudo apt install {pkgs}`) and re-source ROS."
            )

        try:
            self._rsp_proc = subprocess.Popen(
                [
                    "ros2", "run", "robot_state_publisher", "robot_state_publisher",
                    "--ros-args",
                    "-p", f"robot_description:={urdf_xml}",
                    "-r", "__ns:=/playback",
                    *_TF_REMAP_ARGS,
                ],
                start_new_session=True,
            )

            self._rviz_config_path = self._build_rviz_config(fixed_frame)
            self._rviz_proc = subprocess.Popen(
                [
                    "ros2", "run", "rviz2", "rviz2",
                    "-d", self._rviz_config_path,
                    "--ros-args",
                    *_TF_REMAP_ARGS,
                ],
                start_new_session=True,
            )
        except Exception:
            self.stop()
            raise

    def _build_rviz_config(self, fixed_frame: str) -> str:
        with open(_RVIZ_TEMPLATE_PATH, "r") as f:
            content = f.read()
        content = content.replace(_FIXED_FRAME_PLACEHOLDER, fixed_frame or "base_link")

        fd, path = tempfile.mkstemp(prefix="lfd_playback_", suffix=".rviz")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def stop(self):
        self._terminate(self._rviz_proc, "rviz2")
        self._rviz_proc = None
        self._terminate(self._rsp_proc, "robot_state_publisher")
        self._rsp_proc = None

        if self._rviz_config_path and os.path.exists(self._rviz_config_path):
            try:
                os.remove(self._rviz_config_path)
            except OSError:
                pass
        self._rviz_config_path = None

    def _terminate(self, proc, label):
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            logger.warning(f"{label} did not exit in time, killing")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Error terminating {label}: {e}")
