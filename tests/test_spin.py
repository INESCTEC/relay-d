"""TopicSubscribers' spin thread, shared by the GUI's Record page and AppAPI. Needs a sourced ROS 2 (rclpy)."""

import statistics
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

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
