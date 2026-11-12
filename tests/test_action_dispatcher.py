"""Action math and safe stop of ActionDispatcher. Needs a sourced ROS 2 (rclpy): skipped without it, run by the "Tests (ROS 2 Jazzy)" CI job."""

import threading

import numpy as np
import pytest

pytest.importorskip("rclpy")
from relay_d.dispatch.action_dispatcher import ActionDispatcher  # noqa: E402


@pytest.fixture
def d():
    d = ActionDispatcher.__new__(ActionDispatcher)  # skip ROS setup: only pure methods are used
    d._stopped, d._dry_run = True, False  # as __init__ sets them
    d._goals, d._goals_lock = [], threading.Lock()
    return d


def test_limits_clamp_broadcast_and_none(d):
    lims = [{"lower_lim": 0.0, "upper_lim": 2.0}]
    assert d._apply_limits(np.array([5.0, -1.0, 1.5]), lims).tolist() == [2.0, 0.0, 1.5]  # clamp, not rescale
    out = d._apply_limits(np.array([5.0, 5.0]), [None, {"lower_lim": 0, "upper_lim": 2}])
    assert out.tolist() == [5.0, 2.0]  # None entry: unchanged


def test_dispatch_drops_non_finite_action(d):
    d._action_configs = {"x": {"indices": [0]}}
    d._last_sent = {}
    d.dispatch(np.array([np.nan]))  # would raise (no publishers set up) if it got past the guard
    d.dispatch(np.array([np.inf]))
    published = _stop_dispatcher(d, {"base": {"indices": [0], "topic": "/cmd_vel", "mapping": {"x": 0}, "stop": [0]}})
    d._stopped = False
    d.dispatch(np.array([np.nan]))
    assert published == [("base", 0.0, 0.0)]  # a velocity controller must not keep the last command


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
    d._last_sent = {}
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
    d._last_sent = {}
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
        "tool": {"indices": [0], "limits": [{"lower_lim": float("nan"), "upper_lim": 1.0}]},  # e.g. a NaN limit
    }
    d._last_sent = {}
    d._dispatch_topic = lambda *a: pytest.fail("nothing may be sent when one output is NaN")
    d.dispatch(np.array([1.0]))
    d._action_configs["arm"].update(mapping={"x": 0}, stop=[0])
    published = _stop_dispatcher(d, d._action_configs)
    d._stopped = False
    d.dispatch(np.array([1.0]))
    assert published == [("arm", 0.0, 0.0)]  # the stop goes out instead


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
    d._last_sent = {}
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


@pytest.mark.parametrize("entry", [
    {"action": True, "action_type": "pkg/action/Name", "topic": "/arm", "mapping": {}},
    {"topic": "/arm", "msg_type": "std_msgs/msg/Float64MultiArray", "mapping": {}},
])
def test_short_limits_list_is_rejected_at_startup(tmp_path, entry):
    import yaml

    lims = [{"lower_lim": 0.0, "upper_lim": 1.0}] * 2
    path = tmp_path / "action_config.yaml"
    path.write_text(yaml.safe_dump({"actions": {"arm": {"indices": [0, 1, 2], "limits": lims, **entry}}}))
    with pytest.raises(ValueError, match="limits"):  # index 2 would reuse index 1's range
        ActionDispatcher(None, str(path))


@pytest.mark.parametrize("key", [{"denormalize": True}, {"apply_limits": True}, {"normalized_range": [-1.0, 1.0]}])
def test_old_denormalize_keys_are_rejected_at_startup(tmp_path, key):
    import yaml

    path = tmp_path / "action_config.yaml"
    entry = {"indices": [0], "topic": "/grip", "msg_type": "std_msgs/msg/Float64", "mapping": {"data": 0},
             "limits": [{"lower_lim": 0.0, "upper_lim": 2.0}], **key}
    path.write_text(yaml.safe_dump({"actions": {"grip": entry}}))
    with pytest.raises(ValueError, match="grip.*denormalize"):  # the model already outputs real units
        ActionDispatcher(None, str(path))


def _fake_node(sent):
    """A node whose publishers append every published message to ``sent``."""
    from types import SimpleNamespace

    return SimpleNamespace(create_publisher=lambda cls, topic, qos: SimpleNamespace(publish=sent.append))


def _load(tmp_path, actions, node=None, **kwargs):
    import yaml

    path = tmp_path / "action_config.yaml"
    path.write_text(yaml.safe_dump({"actions": actions}))
    return ActionDispatcher(node, str(path), **kwargs)


_GRIP = {"indices": [0], "topic": "/grip", "msg_type": "std_msgs/msg/Float64", "mapping": {"data": 0}}


def test_limits_clamp_the_dispatched_value(tmp_path):
    sent = []
    entry = {**_GRIP, "limits": [{"lower_lim": 0.0, "upper_lim": 2.0}],
             "denormalize": False}  # accepted and ignored: old configs that never de-normalized still start
    d = _load(tmp_path, {"grip": entry}, _fake_node(sent))
    d.dispatch(np.array([5.0]))
    assert [m.data for m in sent] == [2.0]  # clamped to upper_lim, not sent as 5


def test_topic_entry_is_published_every_tick(tmp_path):
    sent = []
    d = _load(tmp_path, {"grip": _GRIP}, _fake_node(sent))
    d.dispatch(np.array([1.0]))
    d.dispatch(np.array([1.0]))
    assert [m.data for m in sent] == [1.0, 1.0]  # a controller with a watchdog needs the stream


@pytest.mark.parametrize("extra, match", [
    ({"command": "go {I[0]}"}, "command.*removed.*topic entry.*action entry"),
    ({"dedupe": True}, "dedupe.*removed.*every tick"),
    ({"transform": {"type": "scale", "in_min": 0, "in_max": 1, "out_min": 0, "out_max": 2}}, "scale.*removed.*limits"),
    ({"transform": {"type": "clip", "out_min": 0, "out_max": 1}}, "clip.*removed.*limits"),
    ({"transform": {"type": "vacuum"}}, "vacuum.*removed.*threshold"),
    ({"transform": {"type": "magnetic"}}, "magnetic.*removed.*threshold"),
])
def test_removed_keys_are_rejected_at_startup(tmp_path, extra, match):
    with pytest.raises(ValueError, match="'grip'.*" + match):  # never silently ignored: it would change what the robot gets
        _load(tmp_path, {"grip": {**_GRIP, **extra}})


@pytest.mark.parametrize("transform, expected", [
    ({"type": "threshold", "threshold": 0.5}, [0.0, 0.0, 1.0]),  # default: 1 / 0, as before
    ({"type": "threshold", "threshold": 0.5, "on_value": 255, "off_value": 10}, [10.0, 10.0, 255.0]),
])
def test_threshold_on_and_off_values(d, transform, expected):
    out = d._apply_transform(np.array([0.2, 0.5, 0.9], dtype=np.float32), transform)
    assert out.dtype == np.float32 and out.tolist() == expected


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


def _stop_dispatcher(d, entries):
    from types import SimpleNamespace

    d._action_configs, d._last_sent, d._node = entries, {}, None
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
    ({"indices": [0, 1], "topic": "/a", "mapping": {"x": 0, "y": 1}, "stop": [0]}, "does not fit"),
    ({"indices": [0], "topic": "/a", "mapping": {"x": 0}, "stop": "zero"}, "list of numbers"),
    ({"indices": [0], "topic": "/a", "mapping": {"x": 0}, "stop": [float("nan")]}, "list of numbers"),
    ({"indices": [0], "topic": "/a", "mapping": {"x": 0, "nope": 1}, "stop": [0, 0]}, "does not fit"),
])
def test_a_broken_stop_is_rejected_at_startup(d, cfg, match):
    _stop_dispatcher(d, {"e": cfg})
    with pytest.raises(ValueError, match=match):
        d._check_stop("e", cfg)


def test_dry_run_creates_no_publisher_and_logs_dispatch_and_stop(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import relay_d.dispatch.action_dispatcher as ad

    logged = []
    monkeypatch.setattr(ad.logger, "info", logged.append)
    monkeypatch.setattr(ad.logger, "warning", logged.append)
    node = SimpleNamespace(create_publisher=lambda *a: pytest.fail("dry run must not create a publisher"))
    d = _load(tmp_path, {"grip": {**_GRIP, "stop": [0.25]}}, node, dry_run=True)
    d.dispatch(np.array([1.5]))
    d.send_stop()
    dry = [m for m in logged if "[DRY RUN]" in m]
    assert len(dry) == 2 and "'grip'" in dry[0] and "1.5" in dry[0] and "stop" in dry[1] and "0.25" in dry[1]


def test_dry_run_creates_no_action_client_and_does_not_wait_for_feedback(tmp_path, monkeypatch):
    import time

    import relay_d.dispatch.action_dispatcher as ad

    logged = []
    monkeypatch.setattr(ad.logger, "info", logged.append)
    monkeypatch.setattr(ad, "ActionClient", lambda *a: pytest.fail("dry run must not create an action client"))
    entry = {"indices": [0], "action": True, "topic": "/fib", "action_type": "example_interfaces/action/Fibonacci",
             "mapping": {"order": 0}, "feedback": {"success_field": "sequence", "timeout": 5.0, "on_timeout": "raise"}}
    with pytest.raises(ValueError, match="mapping"):  # messages are still built: a mapping error still stops startup
        _load(tmp_path, {"fib": {**entry, "mapping": {"nope": 0}}}, dry_run=True)
    d = _load(tmp_path, {"fib": entry}, dry_run=True)
    t0 = time.monotonic()
    d.dispatch(np.array([3.0]))  # no server answers in a dry run: no wait for its result
    assert time.monotonic() - t0 < 1.0
    assert any("[DRY RUN]" in m and "'fib'" in m and "order=3" in m for m in logged)


def _goal_handle(status=1):  # 1 = STATUS_ACCEPTED
    from types import SimpleNamespace

    h = SimpleNamespace(accepted=True, status=status, cancels=[])
    h.cancel_goal_async = lambda: h.cancels.append(1)
    return h


def _accept(d, handle, name="fib"):
    from types import SimpleNamespace

    d._goal_response_callback(SimpleNamespace(result=lambda: handle), name)


def test_stop_cancels_each_active_goal_once(d):
    _stop_dispatcher(d, {"fib": {"indices": [0], "action": True}})
    active, done = _goal_handle(), _goal_handle(status=4)  # 4 = STATUS_SUCCEEDED
    d._stopped = False  # a goal was sent
    _accept(d, active)
    _accept(d, done)
    d.send_stop()
    d.send_stop()
    assert active.cancels == [1] and done.cancels == []  # a finished goal has nothing to cancel
    late = _goal_handle()
    _accept(d, late)  # sent before the stop, accepted after it: still cancelled
    assert late.cancels == [1]


def test_stop_cancels_a_goal_sent_by_the_waiting_path(d):
    from types import SimpleNamespace

    _stop_dispatcher(d, {"fib": {"indices": [0], "action": True}})
    handle = _goal_handle()
    handle.get_result_async = lambda: SimpleNamespace(add_done_callback=lambda cb: None)  # never finishes
    future = SimpleNamespace(add_done_callback=lambda cb: cb(SimpleNamespace(result=lambda: handle)))
    client = SimpleNamespace(send_goal_async=lambda goal, feedback_callback=None: future)
    d._stopped = False
    d._dispatch_action_and_wait("fib", client, object(), {"timeout": 0.05})
    d.send_stop()
    assert handle.cancels == [1]


def test_dry_run_logs_the_cancel_of_a_sent_goal(tmp_path, monkeypatch):
    import relay_d.dispatch.action_dispatcher as ad

    logged = []
    monkeypatch.setattr(ad.logger, "info", logged.append)
    monkeypatch.setattr(ad.logger, "warning", logged.append)
    entry = {"indices": [0], "action": True, "topic": "/fib", "action_type": "example_interfaces/action/Fibonacci",
             "mapping": {"order": 0}}
    d = _load(tmp_path, {"fib": entry}, dry_run=True)
    d.send_stop()
    assert not [m for m in logged if "cancel" in m]  # nothing sent yet
    d.dispatch(np.array([3.0]))
    d.send_stop()
    assert len([m for m in logged if "[DRY RUN]" in m and "'fib'" in m and "cancel" in m]) == 1
