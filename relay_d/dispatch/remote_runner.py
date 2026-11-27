"""Client proxy that runs a LeRobot or robomimic policy in an isolated worker subprocess.

Exposes the exact same interface as ``LeRobotPolicyRunner`` and ``RobomimicPolicyRunner``
(``obs_keys``, ``action_dim``, ``fps``, ``chunk_steps``, ``reset()``, ``get_action()``,
``get_action_chunk()``, plus ``close()``), so ``InferenceSession`` can swap it in transparently.

Why this exists:
- Eliminates GIL contention between multi-camera ``rclpy`` callbacks and PyTorch forward passes.
- Eliminates the ``reexec.py`` ``PYTHONPATH`` bridge between ROS 2 Jazzy's system Python
  (NumPy 1.26) and the LeRobot / robomimic venvs (NumPy 2.x + PyTorch).
- Transfers multi-MB camera frames via ``/dev/shm`` ``mmap`` slots that are immediately unlinked
  after the worker maps them, so zero shared memory leaks even on ``SIGKILL``.
"""

from __future__ import annotations

import atexit
import mmap
import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from relay_d.utils.coloring_logger import logger
from .policy_worker import recv_frame, send_frame


@dataclass
class _ClientSlot:
    mm: mmap.mmap
    view: np.ndarray
    shape: Tuple[int, ...]
    dtype_str: str
    nbytes: int

    def close(self) -> None:
        self.view = None  # type: ignore[assignment]
        try:
            self.mm.close()
        except Exception:
            pass


def _worker_python(backend: str, explicit_python: Optional[str] = None) -> str:
    if explicit_python:
        return explicit_python
    try:
        from relay_d.training.envsetup import python_for

        candidate = python_for(backend)
        if candidate and os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return sys.executable


def _worker_env() -> Dict[str, str]:
    """Environment for the ML worker: include Relay-D's package root, strip ``/opt/ros`` paths."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    relay_root = str(Path(__file__).resolve().parents[2])
    existing = [
        p
        for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and not os.path.normpath(p).startswith("/opt/ros/")
    ]
    if os.path.normpath(relay_root) not in {os.path.normpath(p) for p in existing}:
        existing.insert(0, relay_root)
    env["PYTHONPATH"] = os.pathsep.join(existing)
    return env


def _ipc_base_dir() -> Optional[str]:
    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None


class SubprocessPolicyRunner:
    """Runs a policy backend in a dedicated child Python process over Unix socket + ``/dev/shm`` mmap."""

    def __init__(
        self,
        backend: str,
        checkpoint: str,
        device: str = "cpu",
        state_keys: Optional[List[str]] = None,
        fallback_state_keys: Optional[List[str]] = None,
        task: Optional[str] = None,
        python_executable: Optional[str] = None,
        worker_module: str = "relay_d.dispatch.policy_worker",
        connect_timeout: float = 30.0,
    ):
        self._closed = False
        self._backend = backend
        self._proc: Optional[subprocess.Popen] = None
        self._server_sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._slots: Dict[str, _ClientSlot] = {}
        self._slot_counter = 0

        self._ipc_dir = tempfile.mkdtemp(prefix=f"relayd_ipc_{os.getpid()}_", dir=_ipc_base_dir())
        os.chmod(self._ipc_dir, 0o700)
        self._sock_path = os.path.join(self._ipc_dir, "worker.sock")

        atexit.register(self.close)

        try:
            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_sock.bind(self._sock_path)
            self._server_sock.listen(1)
            self._server_sock.setblocking(False)

            py = _worker_python(backend, python_executable)
            cmd = [py, "-m", worker_module, "--socket", self._sock_path]
            logger.info(f"[SubprocessPolicyRunner] Spawning {backend} worker with {py}")
            self._proc = subprocess.Popen(cmd, env=_worker_env())

            self._conn = self._accept_worker(timeout=connect_timeout)
            # Socket path is no longer needed once connected.
            self._server_sock.close()
            self._server_sock = None
            try:
                os.unlink(self._sock_path)
            except OSError:
                pass

            reply, _ = self._rpc(
                {
                    "cmd": "init",
                    "backend": backend,
                    "checkpoint": str(checkpoint),
                    "device": device,
                    "state_keys": state_keys,
                    "fallback_state_keys": fallback_state_keys,
                    "task": task,
                }
            )
            self.obs_keys: List[str] = list(reply.get("obs_keys", []))
            self.action_dim: Optional[int] = reply.get("action_dim")
            self.fps: Optional[float] = reply.get("fps")
            self.chunk_steps: int = int(reply.get("chunk_steps", 0) or 0)
        except BaseException:
            self.close()
            raise

    def _accept_worker(self, timeout: float) -> socket.socket:
        assert self._server_sock is not None and self._proc is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"[SubprocessPolicyRunner] {self._backend} worker did not connect within {timeout:.0f}s."
                )
            ready, _, _ = select.select([self._server_sock], [], [], min(0.2, remaining))
            if ready:
                conn, _ = self._server_sock.accept()
                conn.setblocking(True)
                return conn
            code = self._proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"[SubprocessPolicyRunner] {self._backend} worker exited early (code {code}) before connecting."
                )

    def _rpc(self, req: dict, payload: bytes = b"") -> Tuple[dict, bytes]:
        if self._closed or self._conn is None:
            raise RuntimeError("[SubprocessPolicyRunner] Policy worker is closed.")
        try:
            send_frame(self._conn, req, payload)
        except OSError as exc:
            raise RuntimeError(f"[SubprocessPolicyRunner] Failed sending '{req.get('cmd')}' to worker: {exc}") from exc

        # Poll readability alongside proc.poll() so a crashing worker or Ctrl+C surfaces promptly
        # while slow model loads (e.g. SmolVLA onto CUDA) never hit an artificial short timeout.
        while True:
            ready, _, _ = select.select([self._conn], [], [], 0.25)
            if ready:
                frame = recv_frame(self._conn)
                if frame is None:
                    code = self._proc.poll() if self._proc is not None else None
                    raise RuntimeError(
                        f"[SubprocessPolicyRunner] {self._backend} worker disconnected unexpectedly (exit code {code})."
                    )
                reply, rpayload = frame
                if not reply.get("ok", False):
                    err_type = reply.get("error_type", "RuntimeError")
                    err_msg = reply.get("error", "unknown worker error")
                    tb = reply.get("traceback", "")
                    if tb:
                        logger.debug(f"[SubprocessPolicyRunner] Worker traceback:\n{tb}")
                    raise RuntimeError(f"[{self._backend} worker {err_type}] {err_msg}")
                return reply, rpayload
            if self._proc is not None:
                code = self._proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"[SubprocessPolicyRunner] {self._backend} worker crashed (exit code {code}) during '{req.get('cmd')}'."
                    )

    def _sync_obs_to_shm(self, obs_dict: Dict[str, np.ndarray]) -> List[str]:
        if self.obs_keys:
            keys = [k for k in self.obs_keys if k in obs_dict]
        else:
            keys = list(obs_dict.keys())

        arrays: Dict[str, np.ndarray] = {}
        to_map: Dict[str, dict] = {}
        new_paths: List[str] = []

        for key in keys:
            arr = np.ascontiguousarray(obs_dict[key])
            arrays[key] = arr
            shape = tuple(arr.shape)
            dtype_str = arr.dtype.str
            nbytes = int(arr.nbytes)

            existing = self._slots.get(key)
            if (
                existing is not None
                and existing.shape == shape
                and existing.dtype_str == dtype_str
                and existing.nbytes == nbytes
            ):
                continue

            if existing is not None:
                existing.close()
                del self._slots[key]

            self._slot_counter += 1
            slot_path = os.path.join(self._ipc_dir, f"slot_{self._slot_counter}.bin")
            fd = os.open(slot_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, max(1, nbytes))
                mm = mmap.mmap(fd, max(1, nbytes), access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)

            if arr.size == 0:
                view = np.empty(shape, dtype=arr.dtype)
            else:
                view = np.frombuffer(mm, dtype=arr.dtype, count=arr.size).reshape(shape)
            self._slots[key] = _ClientSlot(mm=mm, view=view, shape=shape, dtype_str=dtype_str, nbytes=nbytes)
            to_map[key] = {"path": slot_path, "shape": list(shape), "dtype": dtype_str, "nbytes": nbytes}
            new_paths.append(slot_path)

        if to_map:
            try:
                self._rpc({"cmd": "map_shm", "slots": to_map})
            finally:
                # Immediately unlink slot paths once mapped (or on failure): both processes keep
                # their live mmap handles and the kernel reclaims the tmpfs RAM even on SIGKILL.
                for p in new_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        for key, arr in arrays.items():
            if arr.size > 0:
                np.copyto(self._slots[key].view, arr)
        return keys

    # ------------------------------------------------------------------
    # PolicyRunner duck-typed interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._rpc({"cmd": "reset"})

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        keys = self._sync_obs_to_shm(obs_dict)
        reply, payload = self._rpc({"cmd": "get_action", "keys": keys})
        return np.frombuffer(payload, dtype=np.dtype(reply["dtype"])).reshape(reply["shape"]).copy()

    def get_action_chunk(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        keys = self._sync_obs_to_shm(obs_dict)
        reply, payload = self._rpc({"cmd": "get_action_chunk", "keys": keys})
        return np.frombuffer(payload, dtype=np.dtype(reply["dtype"])).reshape(reply["shape"]).copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._conn is not None:
            try:
                if self._proc is not None and self._proc.poll() is None:
                    self._conn.settimeout(1.0)
                    send_frame(self._conn, {"cmd": "shutdown"})
                    recv_frame(self._conn)
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

        for slot in self._slots.values():
            slot.close()
        self._slots.clear()

        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)
            self._proc = None

        if self._ipc_dir and os.path.isdir(self._ipc_dir):
            shutil.rmtree(self._ipc_dir, ignore_errors=True)
