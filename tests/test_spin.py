"""TopicSubscribers' spin thread, shared by the GUI's Record page and AppAPI. Needs a sourced ROS 2 (rclpy)."""

import statistics
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from relay_d.acquisition.qt_app.ui.record.topic_subscribers import TopicSubscribers  # noqa: E402


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_the_spin_thread_keeps_up_with_four_topics_at_100_hz(ros):
    topics = [f"/spin_test_{i}" for i in range(4)]
    pub_node = rclpy.create_node("spin_test_pub")
    pubs = [pub_node.create_publisher(JointState, t, 10) for t in topics]

    def publish():
        for p in pubs:
            msg = JointState()
            msg.header.stamp = pub_node.get_clock().now().to_msg()
            p.publish(msg)

    pub_node.create_timer(0.01, publish)
    executor = SingleThreadedExecutor()
    executor.add_node(pub_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    subs = TopicSubscribers()
    try:
        subs.start_spinning()  # before subscribing, as in the GUI and AppAPI: new subscriptions must still be served
        containers = [SimpleNamespace(container_id=f"c{i}", topic_path=t, input_name=f"c{i}") for i, t in enumerate(topics)]
        assert all(subs.create_subscribers_for_containers(containers).values())
        time.sleep(1.0)
        counts0 = {c.container_id: subs.get_message_count(c.container_id) for c in containers}
        ages, t0 = [], time.time()
        while time.time() - t0 < 2.0:
            _, stamps = subs.get_content_and_timestamps()
            ages += [time.time() - s for s in stamps.values()]
            time.sleep(0.007)
        handled = sum(subs.get_message_count(c.container_id) - counts0[c.container_id] for c in containers) / (time.time() - t0)
        median_ms = statistics.median(ages) * 1000
        print(f"spin thread: {handled:.0f} msg/s handled of ~400, median age {median_ms:.1f} ms")
        assert median_ms < 30  # the GUI's 10 ms Qt-timer spin_once: ~98 ms
        assert handled > 300  # the Qt timer: ~78 msg/s
    finally:
        subs.close()
        executor.shutdown()
        pub_node.destroy_node()
    assert not subs._spin_thread.is_alive()


def test_the_record_page_spins_its_subscribers_in_their_own_thread_not_the_qt_timer(qapp, ros, monkeypatch, tmp_path):
    for name in ("PyQt5.QtMultimedia", "PyQt5.QtMultimediaWidgets"):  # not in every PyQt5 install; unused here
        try:
            __import__(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, MagicMock())
    from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser
    from relay_d.acquisition.qt_app.ui.record.record_page import RecordPage

    ros_nodes = []  # the app's 10 ms Qt-timer list (app.register_ros_node)
    ui = MagicMock()
    ui.register_ros_node = ros_nodes.append
    page = RecordPage(ui, YamlParser())
    try:
        page.container_manager = MagicMock()
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("config:\n  dataset_name: d\n  recording_frequency: 10\n  input: {}\n")
        page._on_config_loaded(str(cfg))
        assert page.topic_subscribers not in ros_nodes  # was registered: one callback per 10 ms tick
        assert page.topic_subscribers._spin_thread.is_alive()
    finally:
        page.tf_update_timer.stop()
        page.topic_subscribers.close()  # what MainWindow.closeEvent calls
    assert not page.topic_subscribers._spin_thread.is_alive()


def test_a_header_less_message_does_not_take_part_in_the_sync_check():
    """A wall-clock receive time next to robot-clock stamps made strict sync drop every tick."""
    from std_msgs.msg import Float64

    subs = TopicSubscribers.__new__(TopicSubscribers)  # skip ROS setup: only the callback is used
    subs.data_lock, subs.latest_data, subs.latest_timestamps, subs.subscriber_status = threading.Lock(), {}, {}, {}
    subs._create_callback("grip", "grip")(Float64(data=1.0))
    js = JointState()
    js.header.stamp.sec = 5
    subs._create_callback("arm", "arm")(js)
    assert subs.latest_data["grip"].data == 1.0
    assert subs.latest_timestamps == {"arm": 5.0}


def test_recorder_and_live_obs_read_joints_in_the_same_order_over_ros(ros, tmp_path):
    """End to end over a real topic: the robot publishes joints in another order than joint_names lists them."""
    import yaml

    from relay_d.acquisition.qt_app.utils.stream_builder import StreamBuilder
    from relay_d.dispatch.obs_builder import ObsBuilder

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow"]
    published = JointState(name=["elbow", "shoulder_lift", "shoulder_pan"], position=[3.0, 2.0, 1.0])
    cfg = tmp_path / "obs_config.yaml"
    cfg.write_text(yaml.safe_dump({"config": {
        "input": {"js": {"topic": "/parity_js", "joint_names": joint_names}},
        "observations": {"q": ["js.position[:]"]},
    }}))
    ob = ObsBuilder(str(cfg))
    pub_node = rclpy.create_node("parity_pub")
    pub = pub_node.create_publisher(JointState, "/parity_js", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(ob)
    executor.add_node(pub_node)
    try:
        deadline = time.monotonic() + 5.0
        while "q" not in ob.get_obs() and time.monotonic() < deadline:
            pub.publish(published)
            executor.spin_once(timeout_sec=0.05)
        live = ob.get_obs()["q"]
    finally:
        executor.shutdown()
        ob.destroy_node()
        pub_node.destroy_node()

    sb = StreamBuilder(SimpleNamespace(
        yaml_data={"config": {"input": {"js": {"topic": "/parity_js", "joint_names": joint_names}}}},
        get_input_data=lambda: [{"name": "js", "joint_names": joint_names}],
        get_streams=lambda: {"q": ["js.position[:]"]},
    ))
    sb.evaluate({"js": published})
    recorded = sb.finalize()["q"][0]
    np.testing.assert_allclose(live, recorded)  # was [1, 2, 3] live vs [3, 2, 1] in the dataset


def test_message_classes_are_cached_by_type_and_never_guessed_from_the_topic_name():
    from sensor_msgs.msg import JointState as JS
    from std_msgs.msg import Float64

    subs = TopicSubscribers.__new__(TopicSubscribers)  # skip ROS setup: only the lookup is used
    subs.message_type_cache = {}
    assert subs._get_message_class("sensor_msgs/msg/JointState", "/gripper") is JS
    # the same topic with another type in another config: was the cached JointState (no data would arrive)
    assert subs._get_message_class("std_msgs/msg/Float64", "/gripper") is Float64
    # a custom package that is not sourced: was guessed from the name ("gripper" -> Bool), which never receives anything
    assert subs._get_message_class("my_robot_msgs/msg/GripperState", "/gripper") is None
