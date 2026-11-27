"""Isolated ML policy worker process (``python -m relay_d.dispatch.policy_worker``).

Runs ``LeRobotPolicyRunner`` or ``RobomimicPolicyRunner`` inside its own Python process and
environment (e.g. ``~/.relayd/envs/lerobot`` or ``~/.relayd/envs/robomimic``), completely
decoupled from ROS 2 (``rclpy``) and its GIL / NumPy ABI.

Heavy observation arrays (cameras, point clouds, states) arrive via zero-copy read-only ``mmap``
slots backed by ``/dev/shm`` files that the parent unlinks immediately after mapping, so the
kernel reclaims all shared RAM even on ``SIGKILL``. Small control commands and predicted actions
flow over a private Unix domain socket using length-prefixed JSON + raw byte trailers (no pickle).

Worker ``stdout``/``stderr`` remain inherited from the parent so PyTorch / LeRobot / robomimic
logs still appear live in the CLI and the Run tab.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import signal
import socket
import struct
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

_HEADER_STRUCT = struct.Struct("!II")


@dataclass
class _MappedSlot:
    mm: mmap.mmap
    view: np.ndarray
    nbytes: int

    def close(self) -> None:
        # Release the numpy view before closing mmap (BufferError otherwise).
        self.view = None  # type: ignore[assignment]
        try:
            self.mm.close()
        except Exception:
            pass


def send_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    """Send one length-prefixed ``(json_header, raw_payload)`` frame."""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = _HEADER_STRUCT.pack(len(header_bytes), len(payload))
    if payload:
        sock.sendall(prefix + header_bytes + payload)
    else:
        sock.sendall(prefix + header_bytes)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    if n == 0:
        return b""
    buf = bytearray(n)
    view = memoryview(buf)
    received = 0
    while received < n:
        chunk = sock.recv_into(view[received:], n - received)
        if chunk == 0:
            return None
        received += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket) -> Optional[Tuple[dict, bytes]]:
    """Receive one length-prefixed frame, or None on clean EOF."""
    prefix = _recv_exact(sock, _HEADER_STRUCT.size)
    if prefix is None:
        return None
    header_len, payload_len = _HEADER_STRUCT.unpack(prefix)
    header_bytes = _recv_exact(sock, header_len)
    if header_bytes is None:
        return None
    payload = _recv_exact(sock, payload_len)
    if payload is None:
        return None
    return json.loads(header_bytes.decode("utf-8")), payload


def _load_runner(req: dict):
    backend = req["backend"]
    checkpoint = req["checkpoint"]
    device = req.get("device", "cpu")
    if backend == "robomimic":
        from .robomimic_runner import RobomimicPolicyRunner

        return RobomimicPolicyRunner(checkpoint, device=device)
    if backend == "lerobot":
        from .lerobot_runner import LeRobotPolicyRunner

        return LeRobotPolicyRunner(
            checkpoint,
            device=device,
            state_keys=req.get("state_keys"),
            fallback_state_keys=req.get("fallback_state_keys"),
            task=req.get("task"),
        )
    raise ValueError(f"Unknown policy backend: {backend!r}")


def _map_slot(spec: dict) -> _MappedSlot:
    path = spec["path"]
    shape = tuple(int(d) for d in spec["shape"])
    dtype = np.dtype(spec["dtype"])
    nbytes = int(spec["nbytes"])
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, max(1, nbytes), access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if count == 0:
        view = np.empty(shape, dtype=dtype)
        view.setflags(write=False)
    else:
        view = np.frombuffer(mm, dtype=dtype, count=count).reshape(shape)
    return _MappedSlot(mm=mm, view=view, nbytes=nbytes)


def serve(socket_path: str) -> int:
    """Connect to the parent's Unix socket and serve policy RPC requests until shutdown or EOF."""
    # The parent ROS node handles SIGINT/SIGTERM, sends the robot's safe stop first,
    # and then sends a shutdown command (or closes the socket) to exit this worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    slots: Dict[str, _MappedSlot] = {}
    runner = None

    try:
        while True:
            frame = recv_frame(sock)
            if frame is None:
                break
            req, _payload = frame
            cmd = req.get("cmd")

            if cmd == "shutdown":
                send_frame(sock, {"ok": True})
                break

            try:
                if cmd == "init":
                    runner = _load_runner(req)
                    send_frame(
                        sock,
                        {
                            "ok": True,
                            "obs_keys": list(getattr(runner, "obs_keys", [])),
                            "action_dim": getattr(runner, "action_dim", None),
                            "fps": getattr(runner, "fps", None),
                            "chunk_steps": int(getattr(runner, "chunk_steps", 0) or 0),
                        },
                    )

                elif cmd == "map_shm":
                    for key, spec in req.get("slots", {}).items():
                        old = slots.pop(key, None)
                        if old is not None:
                            old.close()
                        slots[key] = _map_slot(spec)
                    send_frame(sock, {"ok": True})

                elif cmd == "reset":
                    if runner is None:
                        raise RuntimeError("Worker received 'reset' before 'init'")
                    runner.reset()
                    send_frame(sock, {"ok": True})

                elif cmd in ("get_action", "get_action_chunk"):
                    if runner is None:
                        raise RuntimeError(f"Worker received '{cmd}' before 'init'")
                    keys = req.get("keys")
                    if keys is None:
                        obs = {k: slot.view for k, slot in slots.items()}
                    else:
                        obs = {k: slots[k].view for k in keys if k in slots}
                    fn = runner.get_action if cmd == "get_action" else runner.get_action_chunk
                    result = np.ascontiguousarray(fn(obs), dtype=np.float32)
                    send_frame(
                        sock,
                        {"ok": True, "shape": list(result.shape), "dtype": result.dtype.str},
                        result.tobytes(),
                    )

                else:
                    raise ValueError(f"Unknown worker command: {cmd!r}")

            except Exception as exc:
                send_frame(
                    sock,
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        for slot in slots.values():
            slot.close()
        slots.clear()
        try:
            sock.close()
        except Exception:
            pass
    return 0


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Relay-D isolated ML policy worker")
    parser.add_argument("--socket", required=True, help="Path to parent's Unix domain socket")
    args = parser.parse_args(argv)
    sys.exit(serve(args.socket))


if __name__ == "__main__":
    main()
