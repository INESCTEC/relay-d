"""InferenceSession stale-input handling and safe stop. Needs a sourced ROS 2 (rclpy): skipped without it, run by the "Tests (ROS 2 Jazzy)" CI job."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
from relay_d.dispatch.api import InferenceSession  # noqa: E402


def test_policy_is_reset_when_inputs_come_back_after_going_stale():
    s = InferenceSession.__new__(InferenceSession)  # skip ROS setup
    ages = {"cam": 5.0}
    resets = []
    s.obs_node = SimpleNamespace(get_input_ages=lambda: dict(ages), get_obs_ages=lambda: {},
                                 get_logger=lambda: SimpleNamespace(error=lambda *a, **k: None))
    s.model = SimpleNamespace(obs_keys=[], reset=lambda: resets.append(1))
    s._was_stale = False
    s.get_observation = lambda: {}
    s.predict = lambda obs: np.zeros(1)
    s.dispatch = lambda a: None
    s.dispatcher = SimpleNamespace(send_stop=lambda: None)
    assert s.step(max_obs_age=1.0) is None
    ages["cam"] = 0.0
    s.step(max_obs_age=1.0)
    s.step(max_obs_age=1.0)
    assert resets == [1]  # once, on the first fresh tick: no actions queued before the gap


def test_step_sends_the_stop_when_predict_raises():
    s = InferenceSession.__new__(InferenceSession)
    stops = []
    s.dispatcher = SimpleNamespace(send_stop=lambda: stops.append(1))
    s.obs_node = SimpleNamespace(get_input_ages=lambda: {}, get_obs_ages=lambda: {})
    s.model = SimpleNamespace(obs_keys=[], reset=lambda: None)
    s._was_stale = False
    s.get_observation = lambda: {}
    s.predict = lambda obs: (_ for _ in ()).throw(RuntimeError("model broke"))
    with pytest.raises(RuntimeError, match="model broke"):
        s.step(max_obs_age=1.0)
    assert stops == [1]


def test_run_lets_a_dispatch_error_through(monkeypatch):
    s = InferenceSession.__new__(InferenceSession)
    stops = []
    s.dispatcher = SimpleNamespace(send_stop=lambda: stops.append(1))
    s.obs_node = SimpleNamespace(get_input_ages=lambda: {}, get_obs_ages=lambda: {},
                                 get_logger=lambda: SimpleNamespace(info=print, error=print))
    s.model = SimpleNamespace(obs_keys=[], reset=lambda: None)
    s._was_stale = False
    s.start = lambda: None
    s.validate = lambda timeout: (True, {"inputs": {}, "obs_keys_ready": True})
    s.get_observation = lambda: {}
    s.predict = lambda obs: np.zeros(1)
    s.dispatch = lambda a: (_ for _ in ()).throw(RuntimeError("ack timeout"))  # feedback on_timeout: raise
    monkeypatch.setattr("relay_d.dispatch.api.rclpy.ok", lambda: True)
    with pytest.raises(RuntimeError, match="ack timeout"):  # only model errors end run() quietly
        s.run(validate_timeout=0.1, max_obs_age=1.0)
    assert stops


def test_joint_view_keeps_configured_order_and_skips_partial_arrays():
    from relay_d.dispatch.obs_builder import _JointStateView

    msg = SimpleNamespace(name=["j3", "j1", "j2"], position=[3.0, 1.0, 2.0], velocity=[0.1], effort=[])
    v = _JointStateView(msg, ["j1", "j2", "j3"])
    assert (v.name, v.position) == (["j1", "j2", "j3"], [1.0, 2.0, 3.0])
    assert v.velocity == [] and v.effort == []  # a 1-value velocity for 3 joints is not read


def test_message_missing_a_joint_keeps_the_last_good_value():
    import threading

    from relay_d.dispatch.obs_builder import ObsBuilder

    b = ObsBuilder.__new__(ObsBuilder)  # skip ROS setup
    b._passthrough_sources, b._joint_filters = set(), {"js": ["j1", "j2"]}
    b._joint_mismatch_warned, b._lock = set(), threading.Lock()
    b._latest_inputs, b._input_timestamps, b._arrival = {}, {}, {}
    b.get_logger = lambda: SimpleNamespace(warn=lambda m: None)
    stamp = SimpleNamespace(sec=1, nanosec=0)
    hdr = SimpleNamespace(stamp=stamp)
    b._topic_callback(SimpleNamespace(header=hdr, name=["j1", "j2"], position=[1.0, 2.0], velocity=[], effort=[]), "js")
    b._topic_callback(SimpleNamespace(header=hdr, name=["finger"], position=[0.5], velocity=[], effort=[]), "js")
    assert b._latest_inputs["js"].position == [1.0, 2.0]


def test_stale_inputs_send_the_stop_and_run_always_ends_with_one(monkeypatch):
    s = InferenceSession.__new__(InferenceSession)
    stops = []
    s.dispatcher = SimpleNamespace(send_stop=lambda: stops.append(1))
    s.obs_node = SimpleNamespace(get_input_ages=lambda: {"cam": 5.0}, get_obs_ages=lambda: {},
                                 get_logger=lambda: SimpleNamespace(error=lambda *a, **k: None))
    s.model = SimpleNamespace(obs_keys=[], reset=lambda: None)
    s._was_stale = False
    s.get_observation = lambda: {}
    assert s.step(max_obs_age=1.0) is None
    assert stops == [1]  # the dispatcher sends it once per stale period (send_stop is idempotent)

    stops.clear()
    s.obs_node = SimpleNamespace(get_input_ages=lambda: {}, get_obs_ages=lambda: {},
                                 get_logger=lambda: SimpleNamespace(info=print, error=print))
    s.start = lambda: None
    s.validate = lambda timeout: (True, {"inputs": {}, "obs_keys_ready": True})
    predicted = []
    s.predict = lambda obs: predicted.append(1) or (_ for _ in ()).throw(RuntimeError("model broke"))
    monkeypatch.setattr("relay_d.dispatch.api.rclpy.ok", lambda: True)  # no rclpy.init() here
    assert s.run(validate_timeout=0.1, max_obs_age=1.0) is False  # the node exits 1: the Run tab shows Failed
    assert predicted == [1] and stops  # (the real send_stop sends once) the loop broke on the model error: the robot is told to stop


@pytest.mark.parametrize("hz, warned", [(10.0, "3x slower"), (30.0, None), (60.0, "2x faster"), (0.0, "as fast as")])
def test_run_warns_when_the_rate_differs_from_the_policy(monkeypatch, hz, warned):
    import relay_d.dispatch.api as api

    s = InferenceSession.__new__(InferenceSession)
    warnings = []
    monkeypatch.setattr(api.logger, "warning", warnings.append)
    monkeypatch.setattr(api.rclpy, "ok", lambda: False)  # stop before the first tick
    s.model = SimpleNamespace(fps=30, obs_keys=[], reset=lambda: None)
    s.dispatcher = SimpleNamespace(send_stop=lambda: None)
    s.obs_node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))
    s.start = lambda: None
    s.validate = lambda timeout: (True, {"inputs": {}, "obs_keys_ready": True})
    s.run(hz=hz, validate_timeout=0.1, max_obs_age=1.0)
    rate_warnings = [w for w in warnings if "trained at 30 Hz" in w]
    if warned is None:
        assert rate_warnings == []
    else:
        assert len(rate_warnings) == 1 and warned in rate_warnings[0]


@pytest.mark.parametrize("key", [{"normalize": True}, {"normalized_range": [-1.0, 1.0]},
                                 {"limits": [{"lower_lim": 0.0, "upper_lim": 1.0}]}])
def test_obs_normalization_keys_are_rejected_at_startup(key):
    from relay_d.dispatch.obs_builder import ObsBuilder

    ob = ObsBuilder.__new__(ObsBuilder)  # skip ROS setup
    ob._obs_cfg, ob._input_cfg = {"q": {"specs": ["js.position[:]"], **key}}, {"js": {"topic": "/js"}}
    ob._tf_inputs, ob._static_inputs = set(), set()
    ob._extractors, ob._rotation_format, ob._obs_input_names, ob._passthrough_obs = {}, {}, {}, {}
    with pytest.raises(ValueError, match="no longer normalizes"):  # robomimic/LeRobot normalize with the checkpoint's stats
        ob._build_extractors()


def test_static_obs_input_is_rejected_at_startup(tmp_path):
    import yaml

    from relay_d.dispatch.obs_builder import ObsBuilder

    path = tmp_path / "obs_config.yaml"
    path.write_text(yaml.safe_dump({"config": {"input": {"grip": {"static": [1.0]}}, "observations": {"g": ["grip.data[:]"]}}}))
    with pytest.raises(ValueError, match="'grip'.*static.*removed"):  # the recorder never wrote one: no model was trained on it
        ObsBuilder(str(path))  # raises before the node is created: no rclpy.init needed


def _layout_session(action_dim, actions):
    s = InferenceSession.__new__(InferenceSession)  # skip ROS setup
    s.model = SimpleNamespace(action_dim=action_dim)
    s.dispatcher = SimpleNamespace(_action_configs=actions)
    return s


def test_action_index_past_the_model_output_stops_startup():
    s = _layout_session(7, {"arm": {"indices": [0, 1, 2, 3, 4, 5]}, "grip": {"indices": [7]}})
    with pytest.raises(ValueError, match="'grip'.*7.*7 values"):  # would drive nothing (or the wrong joint) silently
        s._check_action_layout()


def test_action_columns_no_entry_sends_are_warned_about(monkeypatch):
    import relay_d.dispatch.api as api

    warnings = []
    monkeypatch.setattr(api.logger, "warning", warnings.append)
    _layout_session(7, {"arm": {"indices": [0, 1, 2, 3, 4, 5]}})._check_action_layout()
    assert len(warnings) == 1 and "[6]" in warnings[0]
    warnings.clear()
    _layout_session(None, {"arm": {"indices": [99]}})._check_action_layout()  # the runner cannot tell: no check
    _layout_session(2, {"arm": {"indices": [0, 1]}})._check_action_layout()
    assert warnings == []


def _rate_session(monkeypatch, fps, infer_hz=7.0):
    import relay_d.dispatch.api as api

    s = InferenceSession.__new__(InferenceSession)
    infos = []
    monkeypatch.setattr(api.rclpy, "ok", lambda: False)  # stop before the first tick
    real_get = api.user_settings.get
    monkeypatch.setattr(api.user_settings, "get", lambda k: infer_hz if k == "infer_hz" else real_get(k))
    s.model = SimpleNamespace(fps=fps, obs_keys=[], reset=lambda: None)
    s.dispatcher = SimpleNamespace(send_stop=lambda: None)
    s.obs_node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=infos.append, error=lambda *a, **k: None))
    s.start = lambda: None
    s.validate = lambda timeout: (True, {"inputs": {}, "obs_keys_ready": True})
    return s, infos


@pytest.mark.parametrize("fps, hz, expected", [(30, None, "30 Hz"), (None, None, "7.0 Hz"), (30, 0.0, "unthrottled"),
                                               (30, 10.0, "10.0 Hz")])
def test_run_defaults_to_the_policy_rate_else_the_infer_hz_setting(monkeypatch, fps, hz, expected):
    s, infos = _rate_session(monkeypatch, fps)
    assert s.run(hz=hz, validate_timeout=0.1, max_obs_age=1.0) is True  # ended without a model error
    assert any("Starting inference" in m and expected in m for m in infos), infos


def test_device_defaults_to_the_infer_device_setting(monkeypatch):
    import relay_d.dispatch.api as api
    import relay_d.dispatch.lerobot_runner as lr

    devices = []
    real_get = api.user_settings.get
    monkeypatch.setattr(api.user_settings, "get", lambda k: "cuda:7" if k == "infer_device" else real_get(k))
    monkeypatch.setattr(api.rclpy, "ok", lambda: True)
    monkeypatch.setattr(api, "ObsBuilder", lambda cfg: SimpleNamespace(low_dim_obs_names=[],
                                                                       get_logger=lambda: SimpleNamespace(info=print)))
    monkeypatch.setattr(api, "ActionDispatcher", lambda *a, **k: SimpleNamespace(_action_configs={}))
    monkeypatch.setattr(api, "MultiThreadedExecutor", lambda: SimpleNamespace(add_node=lambda n: None))
    monkeypatch.setattr(lr, "LeRobotPolicyRunner", lambda ckpt, device, **k: devices.append(device) or SimpleNamespace())
    InferenceSession("m", "o.yaml", "a.yaml", backend="lerobot")
    InferenceSession("m", "o.yaml", "a.yaml", device="cpu", backend="lerobot")
    assert devices == ["cuda:7", "cpu"]  # the API runs where the CLI and the Run tab run


def test_node_passes_its_rate_through_and_exits_1_on_a_model_error(monkeypatch):
    import relay_d.dispatch.inference_node as node

    calls = []

    class FakeSession:
        def __init__(self, **kw):
            calls.append(("device", kw["device"]))

        def run(self, hz, validate_timeout, max_obs_age):
            calls.append(("hz", hz))
            return False

        def close(self):
            calls.append("close")

    monkeypatch.setattr(node, "InferenceSession", FakeSession)
    monkeypatch.setattr(node, "sigterm_runs_the_safe_stop", lambda: calls.append("sigterm"))
    monkeypatch.setattr(node.sys, "argv", ["node", "--checkpoint", "/nope", "--obs-config", "o", "--action-config", "a"])
    with pytest.raises(SystemExit) as e:
        node.main()
    assert e.value.code == 1
    assert ("hz", None) in calls and "sigterm" in calls and calls[-1] == "close"  # run() picks the rate; the stop still runs


SIGTERM_PROBE = """
import time, rclpy
from std_msgs.msg import Float64
from relay_d.dispatch.inference_node import sigterm_runs_the_safe_stop
rclpy.init()  # installs rclpy's own handlers, as InferenceSession does
sigterm_runs_the_safe_stop()
node = rclpy.create_node("sigterm_probe")
pub = node.create_publisher(Float64, "/sigterm_probe_stop", 1)
print("ready", flush=True)
try:
    while True:
        time.sleep(0.05)
finally:
    pub.publish(Float64(data=0.0))
    print("STOP PUBLISHED", flush=True)
"""


def test_sigterm_takes_the_safe_stop_path():
    import signal
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", SIGTERM_PROBE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout.readline().strip() == "ready"
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=20)
    assert "STOP PUBLISHED" in out, out  # rclpy's default SIGTERM handling exits without running finally
