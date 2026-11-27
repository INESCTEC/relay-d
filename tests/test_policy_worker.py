"""Integration tests for the isolated policy worker (policy_worker.py + remote_runner.py)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from relay_d.dispatch.remote_runner import SubprocessPolicyRunner, _worker_env


def _write_fake_worker_module(tmp_path: Path) -> str:
    """Write a small importable module that patches LeRobotPolicyRunner with a fast fake
    and invokes the real ``relay_d.dispatch.policy_worker.main()``."""
    mod_file = tmp_path / "fake_policy_worker_mod.py"
    mod_file.write_text(
        """
import os
import sys
import numpy as np
from relay_d.dispatch import lerobot_runner, policy_worker


class FakePolicy:
    def __init__(self, checkpoint, device="cpu", state_keys=None, fallback_state_keys=None, task=None):
        self.checkpoint = checkpoint
        self.device = device
        self.obs_keys = ["state", "cam_top", "cam_wrist"]
        self.action_dim = 4
        self.fps = 20.0
        self.chunk_steps = 3
        self.reset_calls = 0
        self.ros_imported = "rclpy" in sys.modules
        self.pythonpath = os.environ.get("PYTHONPATH", "")

    def reset(self):
        self.reset_calls += 1

    def get_action(self, obs_dict):
        if obs_dict["state"][0] < -99.0:
            raise ValueError("synthetic model error")
        return np.array([
            float(obs_dict["state"].sum()),
            float(obs_dict["cam_top"].mean()),
            float(obs_dict["cam_wrist"].shape[0]),
            float(self.reset_calls),
        ], dtype=np.float32)

    def get_action_chunk(self, obs_dict):
        base = self.get_action(obs_dict)
        return np.stack([base, base + 1.0, base + 2.0], axis=0).astype(np.float32)


lerobot_runner.LeRobotPolicyRunner = FakePolicy

if __name__ == "__main__":
    policy_worker.main()
"""
    )
    return "fake_policy_worker_mod"


def test_worker_env_strips_opt_ros_and_includes_repo_root(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy/lib/python3.12/site-packages:/custom/lib")
    env = _worker_env()
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert not any(p.startswith("/opt/ros/") for p in entries)
    assert "/custom/lib" in entries


def test_subprocess_policy_runner_end_to_end_shm_chunk_resize_and_cleanup(tmp_path, monkeypatch):
    worker_mod = _write_fake_worker_module(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    runner = SubprocessPolicyRunner(
        backend="lerobot",
        checkpoint="/fake/ckpt",
        device="cpu",
        python_executable=sys.executable,
        worker_module=worker_mod,
    )
    ipc_dir = runner._ipc_dir
    proc = runner._proc
    try:
        assert runner.obs_keys == ["state", "cam_top", "cam_wrist"]
        assert runner.action_dim == 4
        assert runner.fps == 20.0
        assert runner.chunk_steps == 3

        runner.reset()

        obs = {
            "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "cam_top": np.full((64, 80, 3), 120, dtype=np.uint8),
            "cam_wrist": np.full((48, 64, 3), 50, dtype=np.uint8),
        }
        action = runner.get_action(obs)
        assert action.dtype == np.float32 and action.shape == (4,)
        np.testing.assert_allclose(action, [6.0, 120.0, 48.0, 1.0])

        # Zero slot files (and on Linux memfd, zero IPC dir at all) so nothing leaks in /dev/shm on SIGKILL.
        assert not os.path.exists(ipc_dir) or [p for p in os.listdir(ipc_dir) if p.startswith("slot_")] == []

        # Second call reuses existing mmap slots with new pixel values.
        obs["cam_top"] = np.full((64, 80, 3), 200, dtype=np.uint8)
        chunk = runner.get_action_chunk(obs)
        assert chunk.shape == (3, 4)
        np.testing.assert_allclose(chunk[0], [6.0, 200.0, 48.0, 1.0])
        np.testing.assert_allclose(chunk[2], [8.0, 202.0, 50.0, 3.0])

        # Dynamic shape change reallocates cleanly and still leaves zero slot files on disk.
        obs["cam_wrist"] = np.full((96, 128, 3), 10, dtype=np.uint8)
        action2 = runner.get_action(obs)
        np.testing.assert_allclose(action2, [6.0, 200.0, 96.0, 1.0])
        assert not os.path.exists(ipc_dir) or [p for p in os.listdir(ipc_dir) if p.startswith("slot_")] == []

        # Errors inside the worker propagate as RuntimeError (so InferenceSession triggers safe stop).
        bad_obs = dict(obs, state=np.array([-100.0, 0.0, 0.0], dtype=np.float32))
        with pytest.raises(RuntimeError, match="synthetic model error"):
            runner.get_action(bad_obs)

        # Worker stays healthy for subsequent calls after a caught exception.
        action3 = runner.get_action(obs)
        np.testing.assert_allclose(action3, [6.0, 200.0, 96.0, 1.0])
    finally:
        runner.close()

    assert proc.poll() == 0
    assert not os.path.exists(ipc_dir)


def test_worker_exits_cleanly_on_parent_socket_eof(tmp_path, monkeypatch):
    worker_mod = _write_fake_worker_module(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    runner = SubprocessPolicyRunner(
        backend="lerobot",
        checkpoint="/fake/ckpt",
        device="cpu",
        python_executable=sys.executable,
        worker_module=worker_mod,
    )
    proc = runner._proc
    try:
        # Simulate abrupt parent crash by closing the socket without sending 'shutdown'.
        runner._conn.close()
        runner._conn = None
        deadline = time.monotonic() + 3.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.poll() == 0
    finally:
        runner.close()


def test_inference_node_defaults_to_isolated_worker_without_reexec(monkeypatch):
    from types import SimpleNamespace

    from relay_d.dispatch import inference_node

    reexec_calls = []
    isolate_flags = []

    def fake_session(**kwargs):
        isolate_flags.append(kwargs["isolate_policy"])
        return SimpleNamespace(run=lambda **k: True, close=lambda: None)

    monkeypatch.setattr(inference_node, "InferenceSession", fake_session)
    monkeypatch.setattr(inference_node, "reexec_if_needed", lambda *a: reexec_calls.append(a))
    monkeypatch.setattr(inference_node, "signals_run_the_safe_stop", lambda: None)

    base = ["lfd-inference-node", "--checkpoint", "m.pth", "--obs-config", "o.yaml",
            "--action-config", "a.yaml", "--backend", "robomimic"]
    monkeypatch.setattr(sys, "argv", base)
    inference_node.main()
    assert isolate_flags == [True]
    assert reexec_calls == []

    monkeypatch.setattr(sys, "argv", base + ["--in-process"])
    inference_node.main()
    assert isolate_flags == [True, False]
    assert len(reexec_calls) == 1

