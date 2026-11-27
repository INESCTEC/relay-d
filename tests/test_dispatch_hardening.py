"""Tests for Workstream 1: dispatch hardening, real-time safety, and speed optimizations."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_obs_builder_use_only_destroys_unused_subs_and_timers():
    pytest.importorskip("rclpy")
    from relay_d.dispatch.obs_builder import ObsBuilder

    ob = ObsBuilder.__new__(ObsBuilder)
    destroyed = []
    canceled = []
    unregistered = []

    ob._obs_input_names = {"q": {"js"}, "ee": {"tf"}}
    ob._passthrough_obs = {"front": "cam_front", "wrist": "cam_wrist"}
    ob._passthrough_sources = {"cam_front", "cam_wrist"}
    ob._tf_inputs = {"tf"}
    ob._extractors = {"q": [], "ee": []}
    ob._pending_topics = [("cam_wrist", "/wrist")]
    ob._used_inputs = None
    ob._lock = threading.Lock()
    ob._input_subs = {"js": "sub_js", "cam_front": "sub_front", "cam_wrist": "sub_wrist"}
    ob.destroy_subscription = destroyed.append
    ob._tf_timer = SimpleNamespace(cancel=lambda: canceled.append("tf"))
    ob._retry_timer = SimpleNamespace(cancel=lambda: canceled.append("retry"))
    ob._tf_listeners = {("/tf", "/tf_static"): SimpleNamespace(unregister=lambda: unregistered.append("tf"))}

    ob.use_only(["q"])

    assert set(destroyed) == {"sub_front", "sub_wrist"}
    assert list(ob._input_subs) == ["js"]
    assert "tf" in canceled and "retry" in canceled
    assert unregistered == ["tf"]
    assert ob._tf_timer is None and ob._retry_timer is None


def test_obs_builder_rejects_nan_tf_and_non_finite_low_dim():
    pytest.importorskip("rclpy")
    from relay_d.dispatch.obs_builder import ObsBuilder

    ob = ObsBuilder.__new__(ObsBuilder)
    ob._obs_cfg = {"q": ["js.x"], "ee": ["tf.translation.x"]}
    ob._input_cfg = {
        "js": {"topic": "/js"},
        "tf": {"parent_frame": "base", "child_frame": "ee"},
    }
    ob._tf_inputs = {"tf"}
    ob._extractors, ob._rotation_format, ob._obs_input_names, ob._passthrough_obs = {}, {}, {}, {}
    ob._build_extractors()
    ob._lock = threading.Lock()
    ob._sync_threshold = None
    ob._latest, ob._obs_refreshed, ob._input_timestamps, ob._arrival = {}, {}, {}, {}
    ob._latest_inputs = {}
    ob._tf_warned, ob._nan_obs_warned = set(), set()
    warnings = []
    ob.get_logger = lambda: SimpleNamespace(warn=warnings.append)

    # Good low-dim value first, then NaN and Inf -> must hold finite value 1.5
    ob._latest_inputs["js"] = SimpleNamespace(x=1.5)
    ob._evaluate_all()
    assert ob._latest["q"].tolist() == [1.5]

    ob._latest_inputs["js"] = SimpleNamespace(x=float("nan"))
    ob._evaluate_all()
    assert ob._latest["q"].tolist() == [1.5]

    ob._latest_inputs["js"] = SimpleNamespace(x=float("inf"))
    ob._evaluate_all()
    assert ob._latest["q"].tolist() == [1.5]
    assert any("NaN/Inf" in w for w in warnings)

    # NaN TF transform must not overwrite _latest_inputs or _arrival
    nan_tf = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=float("nan"), y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    ob._tf_input_buffer = {"tf": SimpleNamespace(lookup_transform=lambda *a: nan_tf)}
    ob._tf_callback()
    assert "tf" not in ob._latest_inputs and "tf" not in ob._arrival
    assert any("Non-finite transform" in w for w in warnings)


def test_action_dispatcher_duration_no_nanosec_overflow_and_max_state_age(tmp_path):
    pytest.importorskip("rclpy")
    import yaml
    from sensor_msgs.msg import JointState

    from relay_d.dispatch.action_dispatcher import ActionDispatcher

    cfg = {
        "actions": {
            "arm": {
                "indices": [0, 1],
                "trajectory": True,
                "topic": "/arm/traj",
                "joint_names": ["j1", "j2"],
                "max_step": 0.1,
            }
        }
    }
    path = tmp_path / "action_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    sent, subs = [], {}
    node = SimpleNamespace(
        create_publisher=lambda cls, topic, qos: SimpleNamespace(publish=sent.append),
        create_subscription=lambda cls, topic, cb, qos: subs.setdefault(topic, cb),
    )
    d = ActionDispatcher(node, str(path))

    # Period chosen so (k + 1) * period lands near an integer second boundary (floating-point rounding edge case)
    d.period = 1.0 / 3.0
    msg = d._trajectory_message(cfg["actions"]["arm"], np.zeros((15, 2)))
    for pt in msg.points:
        assert 0 <= pt.time_from_start.nanosec < 1_000_000_000
    assert (msg.points[2].time_from_start.sec, msg.points[2].time_from_start.nanosec) == (1, 0)
    assert (msg.points[5].time_from_start.sec, msg.points[5].time_from_start.nanosec) == (2, 0)

    # max_state_age override must take precedence over infer_max_obs_age_s
    js = JointState()
    js.name, js.position = ["j1", "j2"], [0.0, 0.0]
    subs["/joint_states"](js)
    positions, stamp = d._joint_states["/joint_states"]
    d._joint_states["/joint_states"] = (positions, stamp - 0.3)
    d.max_state_age = 0.1
    assert d._measured("arm", cfg["actions"]["arm"]) is None
    d.max_state_age = 1.0
    assert d._measured("arm", cfg["actions"]["arm"]) is not None


def test_action_dispatcher_failed_future_wakes_wait_immediately():
    pytest.importorskip("rclpy")
    from relay_d.dispatch.action_dispatcher import ActionDispatcher

    d = ActionDispatcher.__new__(ActionDispatcher)
    d._stopped, d._dry_run = False, False
    d._goals, d._goals_lock = [], threading.Lock()
    d._last_sent = {"grip": "[1.]"}

    class _ExplodingFuture:
        def add_done_callback(self, cb):
            cb(self)

        def result(self):
            raise RuntimeError("action server crashed")

    client = SimpleNamespace(send_goal_async=lambda goal, feedback_callback=None: _ExplodingFuture())
    t0 = time.monotonic()
    d._dispatch_action_and_wait("grip", client, object(), {"timeout": 5.0, "success_field": "ok"})
    assert time.monotonic() - t0 < 1.0  # woke immediately instead of hanging 5s
    assert "grip" not in d._last_sent  # cleared so next tick retries


def test_action_dispatcher_compiled_limits_and_mapping_parity(tmp_path):
    pytest.importorskip("rclpy")
    import yaml

    from relay_d.dispatch.action_dispatcher import ActionDispatcher

    cfg = {
        "actions": {
            "arm": {
                "indices": [0, 1, 2],
                "topic": "/arm",
                "msg_type": "geometry_msgs/msg/Vector3",
                "limits": [{"lower_lim": -1.0, "upper_lim": 1.0}, None, {"lower_lim": 0.0, "upper_lim": 2.0}],
                "mapping": {"x": 0, "y": 1, "z": 2},
            }
        }
    }
    path = tmp_path / "action_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    sent = []
    node = SimpleNamespace(create_publisher=lambda cls, topic, qos: SimpleNamespace(publish=sent.append))
    d = ActionDispatcher(node, str(path))
    assert "arm" in d._compiled_limits and "arm" in d._compiled_mappings
    d.dispatch(np.array([-5.0, 9.0, 5.0], dtype=np.float32))
    assert (sent[-1].x, sent[-1].y, sent[-1].z) == (-1.0, 9.0, 2.0)


def test_inference_session_close_joins_spin_thread_and_drift_free_pacing(monkeypatch):
    pytest.importorskip("rclpy")
    import relay_d.dispatch.api as api

    order = []
    s = api.InferenceSession.__new__(api.InferenceSession)
    s.dispatcher = SimpleNamespace(send_stop=lambda: order.append("stop"))
    s._executor = SimpleNamespace(shutdown=lambda: order.append("exec_shutdown"))
    s.obs_node = SimpleNamespace(destroy_node=lambda: order.append("destroy_node"))
    s._rclpy_initialized = False

    ready = threading.Event()

    def fake_spin():
        ready.set()
        time.sleep(0.05)
        order.append("spin_done")

    s._spin_thread = threading.Thread(target=fake_spin)
    s._spin_thread.start()
    ready.wait(timeout=1.0)
    s.close()
    assert order.index("spin_done") < order.index("destroy_node")

    # Drift-free pacing: 10 steps at 100 Hz (0.01s period) should complete near 0.09-0.12s without slow drift
    s2 = api.InferenceSession.__new__(api.InferenceSession)
    s2.dispatcher = SimpleNamespace(send_stop=lambda: None, dispatch=lambda a: None, max_state_age=None)
    s2.obs_node = SimpleNamespace(
        get_input_ages=lambda: {},
        get_obs_ages=lambda: {},
        get_logger=lambda: SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
    )
    s2.model = SimpleNamespace(obs_keys=[], reset=lambda: None)
    s2._was_stale = False
    s2.start = lambda: None
    s2.validate = lambda timeout: (True, {"inputs": {}, "obs_keys_ready": True})
    s2.get_observation = lambda: {}
    s2.predict = lambda obs: np.zeros(1, dtype=np.float32)
    monkeypatch.setattr(api.rclpy, "ok", lambda: True)
    t0 = time.monotonic()
    assert s2.run(hz=100.0, max_steps=10, validate_timeout=0.1, max_obs_age=0.25) is True
    elapsed = time.monotonic() - t0
    assert 0.07 <= elapsed <= 0.25
    assert s2.dispatcher.max_state_age == 0.25


def _write_hanging_worker_module(tmp_path: Path) -> str:
    mod_file = tmp_path / "hanging_policy_worker_mod.py"
    mod_file.write_text(
        """
import time
import numpy as np
from relay_d.dispatch import lerobot_runner, policy_worker


class HangingPolicy:
    def __init__(self, checkpoint, device="cpu", state_keys=None, fallback_state_keys=None, task=None):
        self.obs_keys = ["state"]
        self.action_dim = 2
        self.fps = 20.0
        self.chunk_steps = 0

    def reset(self):
        pass

    def get_action(self, obs_dict):
        if obs_dict["state"][0] > 50.0:
            time.sleep(30.0)
        return np.array([1.0, 2.0], dtype=np.float32)


lerobot_runner.LeRobotPolicyRunner = HangingPolicy

if __name__ == "__main__":
    policy_worker.main()
"""
    )
    return "hanging_policy_worker_mod"


def test_subprocess_policy_runner_step_timeout_kills_wedged_worker(tmp_path, monkeypatch):
    from relay_d.dispatch.remote_runner import SubprocessPolicyRunner

    mod = _write_hanging_worker_module(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    runner = SubprocessPolicyRunner(
        backend="lerobot",
        checkpoint="/fake/ckpt",
        device="cpu",
        python_executable=sys.executable,
        worker_module=mod,
        step_timeout=0.6,
    )
    proc = runner._proc
    try:
        # On Linux memfd, the IPC directory is already removed right after accept().
        if runner._use_memfd:
            assert not os.path.exists(runner._ipc_dir)

        act = runner.get_action({"state": np.array([1.0], dtype=np.float32)})
        np.testing.assert_allclose(act, [1.0, 2.0])

        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            runner.get_action({"state": np.array([100.0], dtype=np.float32)})
        assert time.monotonic() - t0 < 5.0
        assert proc.poll() is not None  # worker was terminated on timeout
    finally:
        runner.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux PR_SET_PDEATHSIG only")
def test_policy_worker_exits_when_parent_is_sigkilled(tmp_path):
    """Spawn a helper parent that starts SubprocessPolicyRunner, prints child PID, then gets SIGKILLed."""
    mod = _write_hanging_worker_module(tmp_path)
    helper = f"""
import sys, time
from relay_d.dispatch.remote_runner import SubprocessPolicyRunner
r = SubprocessPolicyRunner(
    backend="lerobot", checkpoint="/fake", device="cpu",
    python_executable=sys.executable, worker_module={mod!r},
)
print("WORKER_PID=" + str(r._proc.pid), flush=True)
time.sleep(60)
"""
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), repo_root, env.get("PYTHONPATH", "")])
    parent = subprocess.Popen([sys.executable, "-c", helper], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    worker_pid = None
    for line in parent.stdout:
        if line.startswith("WORKER_PID="):
            worker_pid = int(line.split("=", 1)[1].strip())
            break
    assert worker_pid is not None
    try:
        os.kill(parent.pid, 9)  # abrupt SIGKILL
        parent.wait(timeout=5.0)
        # Worker must die promptly via PR_SET_PDEATHSIG(SIGTERM) + socket EOF
        deadline = time.monotonic() + 3.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.05)
        assert not alive, f"worker pid {worker_pid} survived parent SIGKILL"
    finally:
        try:
            os.kill(worker_pid, 9)
        except ProcessLookupError:
            pass


def test_run_page_stop_ignores_zero_pid(qapp, monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QWidget

    from relay_d.acquisition.qt_app.ui.run import run_page as rp

    killed = []
    monkeypatch.setattr(rp.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    page = rp.RunPage(SimpleNamespace(page_run=QWidget()))
    monkeypatch.setattr(page, "is_running", lambda: True)
    monkeypatch.setattr(page.proc, "processId", lambda: 0)
    page.stop()
    assert killed == []  # must NEVER call os.kill(0, SIGTERM) and kill the GUI's own process group
