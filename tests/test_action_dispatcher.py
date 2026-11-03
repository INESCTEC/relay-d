"""Action math and safe stop of ActionDispatcher. Needs a sourced ROS 2 (rclpy): skipped without it, run by the "Tests (ROS 2 Jazzy)" CI job."""

import numpy as np
import pytest

pytest.importorskip("rclpy")
from relay_d.dispatch.action_dispatcher import ActionDispatcher  # noqa: E402


@pytest.fixture
def d():
    return ActionDispatcher.__new__(ActionDispatcher)  # skip ROS setup: only pure methods are used


def test_limits_broadcast_clip_and_none(d):
    lims = [{"lower_lim": 0.0, "upper_lim": 10.0}]
    out = d._apply_limits(np.array([-1.0, 0.0, 1.0, 5.0]), lims, [-1.0, 1.0])
    assert out.tolist() == [0.0, 5.0, 10.0, 10.0]  # out-of-range input is clipped
    out = d._apply_limits(np.array([1.0, 0.3]), [None, {"lower_lim": 0, "upper_lim": 2}], [-1.0, 1.0])
    assert out[0] == 1.0  # None entry: unchanged
    assert d._apply_limits(np.array([0.5]), lims, [1.0, 1.0]).tolist() == [0.5]  # zero span: unchanged


def test_render_command(d):
    cmd = d._render_command("run {I[0]} {I[1]} {I[10]}", np.array([3.0, 0.1234567]))
    assert cmd == "run 3 0.123457 {I[10]}"  # {I[1]} must not eat {I[10]}


def test_dispatch_drops_non_finite_action(d):
    d._action_configs = {"x": {"indices": [0]}}
    d._last_sent = {}
    d.dispatch(np.array([np.nan]))  # would raise (no publishers set up) if it got past the guard
    d.dispatch(np.array([np.inf]))


def test_set_nested_casts_to_field_type():
    from types import SimpleNamespace

    from relay_d.dispatch.action_dispatcher import _set_nested

    msg = SimpleNamespace(count=0, flag=False, speed=0.0)
    _set_nested(msg, "count", 2.6)
    _set_nested(msg, "flag", 1.0)
    _set_nested(msg, "speed", 3)
    assert (msg.count, msg.flag, msg.speed) == (3, True, 3.0)  # rclpy rejects a float for an int/bool field


def test_fill_msg_reports_failed_mapping():
    from types import SimpleNamespace

    from relay_d.dispatch.action_dispatcher import _fill_msg

    msg = SimpleNamespace(x=0.0)
    values = np.array([1.0, 2.0])
    assert _fill_msg(msg, {"mapping": {"x": 1}}, values, None) is True
    assert msg.x == 2.0
    assert _fill_msg(msg, {"mapping": {"x": 0, "missing": 1}}, values, None) is False  # caller must not publish


def test_failed_send_is_not_remembered_so_it_is_retried(d):
    d._action_configs = {"grip": {"indices": [0], "action": True}}
    d._last_sent, d._procs = {}, {}
    sent = []
    results = iter([False, True, True])
    d._dispatch_action = lambda name, cfg, values: sent.append(name) or next(results)
    for _ in range(3):
        d.dispatch(np.array([1.0]))
    assert sent == ["grip", "grip"]  # 1st failed -> retried; 2nd succeeded -> same command is not re-sent


def test_dispatch_sends_nothing_if_one_entry_cannot_be_computed(d):
    d._action_configs = {
        "arm": {"indices": [0], "topic": "/arm"},
        "tool": {"indices": [0], "transform": {"type": "nope"}},
    }
    d._last_sent, d._procs = {}, {}
    d._dispatch_topic = lambda *a: pytest.fail("arm must not move while the tool entry is broken")
    with pytest.raises(ValueError):
        d.dispatch(np.array([1.0]))


def test_check_mapping_rejects_a_field_the_message_does_not_have(d):
    from types import SimpleNamespace

    msg_class = lambda: SimpleNamespace(x=0.0)  # noqa: E731
    d._node = None  # the message has no header, so the clock is never asked
    d._check_mapping("arm", {"indices": [0, 1], "mapping": {"x": 0}}, msg_class)
    with pytest.raises(ValueError):
        d._check_mapping("arm", {"indices": [0, 1], "mapping": {"x": 0, "typo": 1}}, msg_class)  # would move the arm, skip the rest


def test_dispatch_sends_nothing_if_a_transform_gives_nan(d):
    d._action_configs = {
        "arm": {"indices": [0], "topic": "/arm"},
        "tool": {"indices": [0], "transform": {"type": "scale", "in_min": 1, "in_max": 1, "out_min": 0, "out_max": 1}},
    }
    d._last_sent, d._procs = {}, {}
    d._dispatch_topic = lambda *a: pytest.fail("nothing may be sent when one output is NaN")
    d.dispatch(np.array([1.0]))


def test_bool_field_is_true_only_above_half():
    from types import SimpleNamespace

    from relay_d.dispatch.action_dispatcher import _set_nested

    msg = SimpleNamespace(flag=True)
    _set_nested(msg, "flag", -0.9)
    assert msg.flag is False  # bool(-0.9) is True: a raw [-1, 1] "open" would close the gripper


def test_check_mapping_rejects_a_goal_init_typo(d):
    class Msg:
        __slots__ = ("x", "use_delta")  # like a ROS message: an unknown field raises

        def __init__(self):
            self.x, self.use_delta = 0.0, False

    d._node = None
    with pytest.raises(ValueError):
        d._check_mapping("arm", {"indices": [0], "mapping": {"x": 0}, "goal_init": {"use_detla": True}}, Msg)


def test_ack_sent_back_at_once_is_not_lost(d, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    import relay_d.dispatch.action_dispatcher as ad

    stamp = SimpleNamespace(sec=1, nanosec=2)
    msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
    ack = SimpleNamespace(command_stamp=stamp, success=True)
    monkeypatch.setattr(ad, "_fill_msg", lambda *a: True)
    d._msg_classes, d._node = {"arm": lambda: msg}, None
    d._ack_events, d._ack_latest = {"arm": threading.Event()}, {}
    d._publishers = {"arm": SimpleNamespace(publish=lambda m: d._on_ack(ack, "arm"))}  # driver acks inside publish()
    cfg = {"topic": "/arm", "feedback": {"timeout": 0.5, "success_field": "success", "success_value": True}}
    t0 = time.monotonic()
    assert d._dispatch_topic("arm", cfg, np.array([0.0]))
    assert time.monotonic() - t0 < 0.4  # the ack was used, not waited out


def test_rejected_goal_is_retried_even_if_rejected_before_send_returns(d):
    from types import SimpleNamespace

    d._action_configs = {"grip": {"indices": [0], "action": True}}
    d._last_sent, d._procs = {}, {}
    sent = []

    def send_and_reject(name, cfg, values):  # the server answers before _dispatch_action returns
        sent.append(name)
        d._goal_response_callback(SimpleNamespace(result=lambda: SimpleNamespace(accepted=False)), name)
        return True

    d._dispatch_action = send_and_reject
    d.dispatch(np.array([1.0]))
    d.dispatch(np.array([1.0]))
    assert sent == ["grip", "grip"]  # was sent once, then skipped forever


def test_fill_msg_keeps_the_frame_set_by_goal_init():
    from types import SimpleNamespace

    from relay_d.dispatch.action_dispatcher import _fill_msg

    msg = SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id="base_link"), x=0.0)
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "now")))
    assert _fill_msg(msg, {"mapping": {"x": 0}}, np.array([1.0]), node)
    assert (msg.header.frame_id, msg.header.stamp) == ("base_link", "now")


def test_check_mapping_rejects_a_short_limits_list(d):
    from types import SimpleNamespace

    d._node = None
    lims = [{"lower_lim": 0.0, "upper_lim": 1.0}] * 3
    with pytest.raises(ValueError, match="limits"):  # joints 3-5 would reuse joint 2's range
        d._check_mapping("arm", {"indices": list(range(6)), "limits": lims, "mapping": {}}, lambda: SimpleNamespace())


def test_ack_arriving_while_a_stale_one_is_read_is_not_lost(d):
    import threading
    import time
    from types import SimpleNamespace

    sent, old = SimpleNamespace(sec=2, nanosec=0), SimpleNamespace(sec=1, nanosec=0)
    stale = SimpleNamespace(command_stamp=old, success=True)
    fresh = SimpleNamespace(command_stamp=sent, success=True)

    class Latest(dict):
        def get(self, key, default=None):  # the right ack lands just as the stale one is being read
            value = super().get(key, default)
            if value is stale:
                self[key] = fresh
                d._ack_events[key].set()
            return value

    d._ack_events, d._ack_latest = {"arm": threading.Event()}, Latest(arm=stale)
    d._ack_events["arm"].set()
    t0 = time.monotonic()
    d._wait_for_ack("arm", {"timeout": 0.5, "success_field": "success"}, sent)
    assert time.monotonic() - t0 < 0.4  # was: the fresh ack's signal was cleared and the wait timed out


@pytest.mark.parametrize("transform", [
    {"type": "scale", "in_min": 1, "in_max": 1, "out_min": 0, "out_max": 1},
    {"type": "magnetic", "strength_scale": {"in_min": 0.5, "in_max": 0.5, "out_min": 0, "out_max": 100}},
])
def test_zero_input_span_is_rejected_at_startup(transform):
    with pytest.raises(ValueError, match="in_min != in_max"):  # magnetic would raise ZeroDivisionError mid-task
        ActionDispatcher._check_transform("tool", transform)
    ActionDispatcher._check_transform("tool", {"type": "threshold"})
    ActionDispatcher._check_transform("tool", None)


def _stop_dispatcher(d, entries):
    from types import SimpleNamespace

    d._action_configs, d._last_sent, d._procs, d._stop_procs, d._node = entries, {}, {}, [], None
    d._msg_classes = {n: (lambda: SimpleNamespace(x=0.0, y=0.0)) for n, c in entries.items() if "topic" in c}
    published = []
    d._publishers = {n: SimpleNamespace(publish=lambda m, n=n: published.append((n, m.x, m.y))) for n in d._msg_classes}
    return published


def test_stop_is_sent_once_until_the_next_dispatch(d):
    published = _stop_dispatcher(d, {"base": {"indices": [0, 1], "topic": "/cmd_vel", "mapping": {"x": 0, "y": 1}, "stop": [0, 0]}})
    d._stopped = True
    d.send_stop()
    assert published == []  # nothing was commanded yet: nothing to stop
    d._dispatch_topic = lambda name, cfg, values: d._publishers[name].publish(type("M", (), {"x": values[0], "y": values[1]})) or True
    d.dispatch(np.array([0.4, 0.2]))
    d.send_stop()
    d.send_stop()  # e.g. run() then close(): one stop, not two
    assert published == [("base", 0.4, 0.2), ("base", 0.0, 0.0)]
    d.dispatch(np.array([0.4, 0.2]))
    d.send_stop()
    assert published[-1] == ("base", 0.0, 0.0) and len(published) == 4


def test_command_after_a_stop_is_not_skipped_as_a_repeat(d, monkeypatch):
    import relay_d.dispatch.action_dispatcher as ad

    started = []

    class Proc:
        def __init__(self, argv, **kw):
            self.argv, self.terminated = argv, False
            started.append(self)

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(ad.subprocess, "Popen", Proc)
    _stop_dispatcher(d, {"vel": {"indices": [0], "command": "vel {I[0]}", "stop": [0]}})
    d.dispatch(np.array([0.5]))
    motion = started[0]
    d.send_stop()
    assert motion.terminated and started[1].argv == ["vel", "0"]  # the old motion ends, the stop command runs
    d.close()
    assert not started[1].terminated  # close() must not kill the stop command
    d.dispatch(np.array([0.5]))
    assert started[-1].argv == ["vel", "0.5"]  # same as before the stop, but sent: the robot was stopped since


def test_one_failing_stop_does_not_block_the_others(d):
    from types import SimpleNamespace

    published = _stop_dispatcher(d, {
        "broken": {"indices": [0], "topic": "/a", "mapping": {"x": 0}, "stop": [0]},
        "base": {"indices": [0], "topic": "/b", "mapping": {"x": 0}, "stop": [0]},
    })
    d._publishers["broken"] = SimpleNamespace(publish=lambda m: 1 / 0)
    d._stopped = False
    d.send_stop()  # must not raise
    assert published == [("base", 0.0, 0.0)]


@pytest.mark.parametrize("cfg, match", [
    ({"indices": [0], "action": True, "stop": [0]}, "cancel"),
    ({"indices": [0, 1], "command": "go {I[0]} {I[1]}", "stop": [0]}, "too few"),
    ({"indices": [0], "command": "go {I[0]}", "stop": "zero"}, "list of numbers"),
    ({"indices": [0], "command": "go {I[0]}", "stop": [float("nan")]}, "list of numbers"),
    ({"indices": [0], "topic": "/a", "mapping": {"x": 0, "nope": 1}, "stop": [0, 0]}, "does not fit"),
])
def test_a_broken_stop_is_rejected_at_startup(d, cfg, match):
    _stop_dispatcher(d, {"e": cfg})
    with pytest.raises(ValueError, match=match):
        d._check_stop("e", cfg)
