import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.duration import Duration as rclpy_duration
import rclpy.time

from relay_d.utils.coloring_logger import logger
from relay_d.utils.tf_topic import (
    TfTopicSubscriptionError,
    get_or_create_tf_buffer,
    wait_for_transform,
)
from threading import Lock, Thread
from rclpy.executors import SingleThreadedExecutor
import importlib
import time

from sensor_msgs.msg import Image, PointCloud2, JointState, CompressedImage
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32, Float64, Int32, Bool
from tf2_msgs.msg import TFMessage

try:
    from tf2_ros import TransformListener, Buffer
except ImportError:
    TransformListener = None
    Buffer = None
    logger.warning("tf2_ros not available")


class TopicSubscribers(Node):
    def __init__(self):
        super().__init__("topic_subscribers")
        self.subscribers = {}
        self.latest_data = {}
        self.latest_timestamps = {}
        self.data_lock = Lock()
        self._spin_lock = Lock()
        self.message_type_cache = {}
        self.subscriber_status = {}
        self.tf_lookup_containers = {}

        self.tf_buffer = None
        self.tf_listener = None
        self._tf_node = None
        self._tf_executor = None
        self._tf_spin_thread = None

        # (tf_topic, tf_static_topic) -> Buffer / listener. Seeded with the
        # default pair below once it's initialized; additional pairs are
        # created lazily by _get_or_create_tf_buffer() for TF inputs that
        # configure a custom tf_topic/tf_static_topic.
        self._tf_buffers = {}
        self._tf_listeners = {}
        self._DEFAULT_TF_KEY = ("/tf", "/tf_static")
        # How long a newly-created CUSTOM tf_topic gets to actually produce
        # the requested transform before _create_tf_lookup treats it as
        # failed (TfTopicSubscriptionError) instead of proceeding silently.
        # Overridable per-input via the YAML `tf_wait_timeout` key.
        self._CUSTOM_TF_WAIT_TIMEOUT = 10.0
        # How long _detect_topic_type polls the ROS graph for a topic before
        # giving up. 5s tolerates typical discovery latency right after node
        # startup without leaving the user waiting too long on a real miss.
        self._TOPIC_DISCOVERY_TIMEOUT = 5.0
        self._TOPIC_DISCOVERY_POLL = 0.3

        try:
            from tf2_ros import TransformListener, Buffer

            self.tf_buffer = Buffer(cache_time=rclpy_duration(seconds=10.0))

            # TF gets its own node + dedicated spin thread so that /tf (which
            # can arrive in much denser bursts than regular topics, especially
            # during rosbag replay) doesn't have to compete one-callback-at-a-time
            # with everything else on this node's manually-driven spin_once()
            # calls. Using a separate Node (rather than spin_thread=True on
            # `self`) avoids racing with the shared spin_once calls elsewhere
            # (data_recorder, AppAPI, the Qt spin timer) that also spin `self`.
            self._tf_node = rclpy.create_node(f"topic_subscribers_tf_listener_{id(self)}")
            self.tf_listener = TransformListener(self.tf_buffer, self._tf_node, spin_thread=False)
            self._tf_executor = SingleThreadedExecutor()
            self._tf_executor.add_node(self._tf_node)
            self._tf_spin_thread = Thread(target=self._tf_executor.spin, daemon=True)
            self._tf_spin_thread.start()

            self._tf_buffers[self._DEFAULT_TF_KEY] = self.tf_buffer
            self._tf_listeners[self._DEFAULT_TF_KEY] = self.tf_listener

            logger.info("TF2 buffer and listener initialized")
        except Exception as e:
            logger.warning(f"Could not initialize TF2: {e}")

        self.msg_type_mapping = {
            "sensor_msgs/Image": Image,
            "sensor_msgs/CompressedImage": CompressedImage,
            "sensor_msgs/PointCloud2": PointCloud2,
            "sensor_msgs/JointState": JointState,
            "geometry_msgs/Twist": Twist,
            "geometry_msgs/PoseStamped": PoseStamped,
            "geometry_msgs/TransformStamped": TransformStamped,
            "std_msgs/String": String,
            "std_msgs/Float32": Float32,
            "std_msgs/Float64": Float64,
            "std_msgs/Int32": Int32,
            "std_msgs/Bool": Bool,
            "tf2_msgs/TFMessage": TFMessage,
        }

        self._last_log_time = {}
        self._log_interval = 10.0
        logger.info("TopicSubscribers initialized")

    def _list_available_topics(self):
        try:
            topics = self.get_topic_names_and_types()
            logger.info(f"Available topics ({len(topics)}):")
            for topic, types in topics[:20]:
                logger.info(f"  {topic} -> {types[0] if types else 'unknown'}")
            if len(topics) > 20:
                logger.info(f"  ... and {len(topics) - 20} more")
        except Exception as e:
            logger.warning(f"Could not list topics: {e}")

    def _get_or_create_tf_buffer(self, tf_topic, tf_static_topic):
        """Return the Buffer feeding from (tf_topic, tf_static_topic), creating
        a new Buffer + listener pair on the shared TF node/executor if one for
        this exact topic pair doesn't exist yet.

        Delegates to relay_d.utils.tf_topic.get_or_create_tf_buffer: raises
        TfTopicSubscriptionError if creation fails for a non-default
        (explicitly configured) pair — a broken custom tf_topic must never
        silently fall back to the default /tf, /tf_static pair. Returns None
        only when the DEFAULT pair itself fails (unchanged legacy behavior).
        """
        return get_or_create_tf_buffer(
            self._tf_buffers,
            self._tf_listeners,
            self._tf_node,
            tf_topic,
            tf_static_topic,
            self._DEFAULT_TF_KEY,
        )

    def create_subscriber(self, container):
        try:
            container_id = getattr(container, "container_id", None)
            topic_path = getattr(container, "topic_path", None)
            input_name = getattr(container, "input_name", None)
            parent_frame = getattr(container, "parent_frame", None)
            child_frame = getattr(container, "child_frame", None)
            tf_topic = getattr(container, "tf_topic", None) or "/tf"
            tf_static_topic = getattr(container, "tf_static_topic", None) or "/tf_static"
            tf_wait_timeout = (
                getattr(container, "tf_wait_timeout", None)
                or self._CUSTOM_TF_WAIT_TIMEOUT
            )
            transfer_rate = getattr(container, "transfer_rate", 10)

            if not container_id:
                logger.error("Container missing container_id")
                return False

            is_tf_lookup = parent_frame and child_frame and not topic_path

            if is_tf_lookup:
                logger.info(
                    f"Creating TF lookup for {container_id}: {parent_frame} -> {child_frame} "
                    f"(tf_topic={tf_topic}, tf_static_topic={tf_static_topic})"
                )
                return self._create_tf_lookup(
                    container_id,
                    parent_frame,
                    child_frame,
                    transfer_rate,
                    tf_topic,
                    tf_static_topic,
                    tf_wait_timeout,
                )

            if not topic_path:
                logger.error(f"Container {container_id} missing topic_path")
                return False

            logger.info(f"Attempting subscriber for {container_id} -> {topic_path}")

            detected_type = self._detect_topic_type(topic_path)
            if not detected_type:
                logger.error(f"Could not detect message type for topic {topic_path}")
                return False

            msg_class = self._get_message_class(detected_type, topic_path)
            if not msg_class:
                logger.error(
                    f"Could not determine message class for {topic_path} (type: {detected_type})"
                )
                return False

            logger.info(f"Creating subscription: {topic_path} -> {msg_class.__name__}")

            callback = self._create_callback(container_id, input_name)
            qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
            subscriber = self.create_subscription(
                msg_class, topic_path, callback, qos_profile=qos
            )

            self.subscribers[container_id] = subscriber
            self.latest_data[container_id] = None
            self.subscriber_status[container_id] = {
                "topic": topic_path,
                "msg_type": str(msg_class),
                "created_at": time.time(),
                "message_count": 0,
                "last_message_time": None,
                "status": "active",
            }

            logger.info(f"Successfully created subscriber for {container_id}")
            return True

        except TfTopicSubscriptionError:
            # A deliberately-configured custom tf_topic couldn't be
            # subscribed to, or never produced data — must not be swallowed
            # into a generic "failed to create subscriber" soft-fail. Let it
            # propagate so the caller (e.g. start_recording) treats it as
            # fatal instead of silently continuing without this TF source.
            raise
        except Exception as e:
            logger.error(f"Failed to create subscriber: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _detect_topic_type(self, topic_path, timeout=None):
        deadline = time.monotonic() + (
            timeout if timeout is not None else self._TOPIC_DISCOVERY_TIMEOUT
        )
        last_topic_types = []
        while True:
            try:
                topic_types = self.get_topic_names_and_types()
                last_topic_types = topic_types
                for topic_name, types in topic_types:
                    if topic_name == topic_path and types:
                        logger.info(f"Detected type for {topic_path}: {types[0]}")
                        return types[0]
            except Exception as e:
                logger.debug(
                    f"Exception in _detect_topic_type for {topic_path}: {e}"
                )

            if time.monotonic() >= deadline:
                break
            time.sleep(self._TOPIC_DISCOVERY_POLL)

        logger.warning(f"Topic {topic_path} not found, available topics:")
        for topic_name, types in last_topic_types[:10]:
            logger.info(f"  {topic_name} -> {types[0] if types else 'unknown'}")
        if len(last_topic_types) > 10:
            logger.info(f"  ... and {len(last_topic_types) - 10} more")
        return None

    def is_topic_available(self, topic_path, timeout=None):
        """Side-effect-free check: does `topic_path` currently exist on the
        ROS graph (no subscription is created)? Used by config-load
        validation before any subscriber is actually created."""
        return self._detect_topic_type(topic_path, timeout=timeout) is not None

    def is_tf_available(
        self,
        parent_frame,
        child_frame,
        tf_topic="/tf",
        tf_static_topic="/tf_static",
        timeout=None,
    ):
        """Check whether `parent_frame -> child_frame` currently resolves on
        the given tf_topic/tf_static_topic pair, without registering a TF
        lookup container. Used by config-load validation. Note: for a
        non-default pair not seen before, this creates (and caches) the
        underlying buffer/listener as a side effect — the same one that
        would be created anyway the first time this pair is actually used."""
        try:
            buffer = self._get_or_create_tf_buffer(tf_topic, tf_static_topic)
        except TfTopicSubscriptionError:
            return False
        if not buffer:
            return False
        wait_timeout = (
            timeout if timeout is not None else self._CUSTOM_TF_WAIT_TIMEOUT
        )
        return wait_for_transform(
            buffer, parent_frame, child_frame, timeout=wait_timeout
        )

    def _get_message_class(self, topic_type_msg, topic_path):
        try:
            if topic_path in self.message_type_cache:
                return self.message_type_cache[topic_path]

            if topic_type_msg in self.msg_type_mapping:
                msg_class = self.msg_type_mapping[topic_type_msg]
                self.message_type_cache[topic_path] = msg_class
                return msg_class

            if topic_type_msg and "/" in topic_type_msg:
                parts = topic_type_msg.split("/")
                if len(parts) >= 2:
                    package = parts[0]
                    msg_name = parts[-1]
                    module_name = f"{package}.msg"
                    try:
                        module = importlib.import_module(module_name)
                        msg_class = getattr(module, msg_name)
                        self.message_type_cache[topic_path] = msg_class
                        return msg_class
                    except Exception as e:
                        logger.debug(f"Could not import {topic_type_msg}: {e}")
                        pass

            # Fallback: try to infer from topic name
            topic_lower = topic_path.lower()
            if "image" in topic_lower:
                if "compressed" in topic_lower:
                    msg_class = CompressedImage
                else:
                    msg_class = Image
            elif "joint" in topic_lower:
                msg_class = JointState
            elif "pointcloud" in topic_lower or "cloud" in topic_lower:
                msg_class = PointCloud2
            elif (
                "gripper" in topic_lower
                or "bool" in topic_lower
                or "open" in topic_lower
                or "close" in topic_lower
            ):
                msg_class = Bool
            elif "pose" in topic_lower:
                msg_class = PoseStamped
            elif "twist" in topic_lower or "cmd_vel" in topic_lower:
                msg_class = Twist
            else:
                return None

            if msg_class:
                self.message_type_cache[topic_path] = msg_class
                logger.info(f"Successfully inferred type for {topic_path}: {msg_class}")
                return msg_class

            logger.warning(
                f"Could not determine message type for {topic_path} (detected: {topic_type_msg})"
            )
            return None
        except Exception as e:
            logger.warning(f"Could not get message class for {topic_path}: {e}")
            return None

    def _log_throttled(self, message, key=None):
        try:
            current_time = time.time()
            if not key:
                key = message
            last_time = self._last_log_time.get(key, 0)
            if current_time - last_time >= self._log_interval:
                logger.info(message)
                self._last_log_time[key] = current_time
        except Exception:
            pass

    def _create_callback(self, container_id, input_name):
        def callback(msg):
            try:
                with self.data_lock:
                    self.latest_data[container_id] = msg
                    if hasattr(msg, "header"):
                        s = msg.header.stamp
                        self.latest_timestamps[container_id] = s.sec + s.nanosec * 1e-9
                    else:
                        self.latest_timestamps[container_id] = time.time()
                    if container_id in self.subscriber_status:
                        self.subscriber_status[container_id]["message_count"] += 1
                        self.subscriber_status[container_id]["last_message_time"] = (
                            time.time()
                        )
                        count = self.subscriber_status[container_id]["message_count"]
                        if count in [1, 10, 100, 1000]:
                            self._log_throttled(
                                f"Received {count} messages from {input_name} ({container_id})",
                                container_id,
                            )
            except Exception as e:
                self._log_throttled(
                    f"Callback error for {container_id}: {e}", container_id
                )

        return callback

    def update_tf_lookups(self):
        """Refresh all TF lookups. Call rate is controlled by the caller."""
        for container_id, tf_info in list(self.tf_lookup_containers.items()):
            self._lookup_tf_transform(
                container_id,
                tf_info["parent_frame"],
                tf_info["child_frame"],
                tf_info["buffer"],
            )

    def has_data(self, container_id):
        with self.data_lock:
            return (
                container_id in self.latest_data
                and self.latest_data[container_id] is not None
            )

    def stop_subscribing(self, container):
        try:
            if isinstance(container, str):
                container_id = container
            else:
                container_id = getattr(container, "container_id", None)

            if not container_id:
                return False

            if container_id in self.subscribers:
                sub = self.subscribers.pop(container_id)
                if sub != "tf":
                    try:
                        self.destroy_subscription(sub)
                    except Exception as e:
                        logger.warning(f"Could not destroy subscription for {container_id}: {e}")
                with self.data_lock:
                    self.latest_data.pop(container_id, None)
                    self.latest_timestamps.pop(container_id, None)
                if container_id in self.subscriber_status:
                    self.subscriber_status[container_id]["status"] = "stopped"
                if container_id in self.tf_lookup_containers:
                    del self.tf_lookup_containers[container_id]
                self._log_throttled(f"Stopped subscriber for {container_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop subscriber: {e}")
            return False

    def clear_all_subscribers(self):
        try:
            for container_id, sub in list(self.subscribers.items()):
                if sub != "tf":
                    try:
                        self.destroy_subscription(sub)
                    except Exception as e:
                        logger.warning(f"Could not destroy subscription for {container_id}: {e}")
            self.subscribers.clear()
            with self.data_lock:
                self.latest_data.clear()
                self.latest_timestamps.clear()
            self.tf_lookup_containers.clear()
            self.subscriber_status.clear()

            # Drop any custom-topic TF buffers/listeners created for the
            # previous config — otherwise their subscriptions pile up on
            # _tf_node across repeated load/record/clear cycles. The default
            # /tf, /tf_static pair is always kept.
            for key in list(self._tf_listeners.keys()):
                if key == self._DEFAULT_TF_KEY:
                    continue
                try:
                    self._tf_listeners[key].unregister()
                except Exception as e:
                    logger.warning(f"Could not unregister TF listener for {key}: {e}")
                del self._tf_listeners[key]
                self._tf_buffers.pop(key, None)

            logger.info("All subscribers destroyed and cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear subscribers: {e}")
            return False

    def close(self):
        """Stop the dedicated TF spin thread/executor and destroy its node."""
        try:
            if self._tf_executor is not None:
                self._tf_executor.shutdown()
            if self._tf_spin_thread is not None and self._tf_spin_thread.is_alive():
                self._tf_spin_thread.join(timeout=2.0)
            if self._tf_node is not None:
                self._tf_node.destroy_node()
            logger.info("TF listener spin thread stopped")
        except Exception as e:
            logger.warning(f"Error closing TF listener: {e}")

    def create_subscribers_for_containers(self, containers):
        results = {}
        for container in containers:
            container_id = getattr(container, "container_id", "unknown")
            topic_path = getattr(container, "topic_path", None)
            parent_frame = getattr(container, "parent_frame", None)
            child_frame = getattr(container, "child_frame", None)

            logger.info(
                f"Attempting: {container_id} | topic={topic_path} | TF={parent_frame}->{child_frame}"
            )
            results[container_id] = self.create_subscriber(container)

        successful = sum(results.values())
        failures = len(results) - successful

        logger.info(
            f"Created {successful}/{len(containers)} subscribers, {failures} failed"
        )
        return results

    def restart_subscriber(self, container):
        try:
            self.stop_subscribing(container)
            time.sleep(0.1)
            return self.create_subscriber(container)
        except Exception as e:
            logger.error(f"Failed to restart subscriber: {e}")
            return False

    def _create_tf_lookup(
        self,
        container_id,
        parent_frame,
        child_frame,
        transfer_rate=10,
        tf_topic="/tf",
        tf_static_topic="/tf_static",
        tf_wait_timeout=None,
    ):
        # Not wrapped in the try/except below: a broken custom tf_topic must
        # raise TfTopicSubscriptionError straight out of this method, not be
        # swallowed into a generic "failed" log line.
        buffer = self._get_or_create_tf_buffer(tf_topic, tf_static_topic)

        try:
            if not buffer:
                logger.error("TF2 not available for TF lookup")
                return False

            is_custom = (tf_topic, tf_static_topic) != self._DEFAULT_TF_KEY
            if is_custom:
                wait_timeout = tf_wait_timeout or self._CUSTOM_TF_WAIT_TIMEOUT
                logger.info(
                    f"Waiting up to {wait_timeout:.0f}s for "
                    f"{parent_frame} -> {child_frame} on custom tf_topic="
                    f"{tf_topic} (tf_static_topic={tf_static_topic})..."
                )
                if not wait_for_transform(
                    buffer, parent_frame, child_frame,
                    timeout=wait_timeout,
                ):
                    try:
                        available_frames = buffer.all_frames_as_string()
                    except Exception:
                        available_frames = "<unavailable>"
                    raise TfTopicSubscriptionError(
                        f"Custom tf_topic={tf_topic}/tf_static_topic="
                        f"{tf_static_topic} never produced transform "
                        f"{parent_frame} -> {child_frame} within "
                        f"{wait_timeout:.0f}s. Frames seen on this buffer:\n"
                        f"{available_frames}"
                    )

            self.tf_lookup_containers[container_id] = {
                "parent_frame": parent_frame,
                "child_frame": child_frame,
                "buffer": buffer,
                "tf_topic": tf_topic,
                "tf_static_topic": tf_static_topic,
            }

            self.subscribers[container_id] = "tf_lookup"
            self.latest_data[container_id] = None
            self.subscriber_status[container_id] = {
                "topic": f"TF:{parent_frame}->{child_frame} @ {tf_topic}",
                "msg_type": "tf_lookup",
                "created_at": time.time(),
                "message_count": 0,
                "last_message_time": None,
                "status": "active",
                "parent_frame": parent_frame,
                "child_frame": child_frame,
            }

            self._lookup_tf_transform(container_id, parent_frame, child_frame, buffer)
            return True
        except TfTopicSubscriptionError:
            raise
        except Exception as e:
            logger.error(f"Failed to create TF lookup: {e}")
            return False

    def _lookup_tf_transform(self, container_id, parent_frame, child_frame, buffer):
        try:
            if not buffer:
                logger.debug("tf buffer is None!")
                return

            try:
                logger.debug(
                    f"Attempting lookup {parent_frame} -> {child_frame}"
                )
                transform = buffer.lookup_transform(
                    parent_frame,
                    child_frame,
                    rclpy.time.Time(seconds=0.0),
                    timeout=rclpy_duration(seconds=0.0),
                )

                # Check for NaN or invalid transform
                t = transform.transform.translation
                r = transform.transform.rotation

                if not (t.x == t.x and t.y == t.y and t.z == t.z) or not (
                    r.x == r.x and r.y == r.y and r.z == r.z and r.w == r.w
                ):
                    msg_count = self.subscriber_status.get(container_id, {}).get(
                        "message_count", 0
                    )
                    if msg_count % 10 == 0:
                        logger.warning(
                            f"TF NaN detected for {parent_frame} -> {child_frame}"
                        )
                    return

                with self.data_lock:
                    self.latest_data[container_id] = transform
                    s = transform.header.stamp
                    self.latest_timestamps[container_id] = s.sec + s.nanosec * 1e-9
                    if container_id in self.subscriber_status:
                        status = self.subscriber_status[container_id]
                        status["message_count"] += 1
                        status["last_message_time"] = time.time()
                        count = status["message_count"]
                        if count == 1:
                            logger.info(
                                f"TF lookup successful: {parent_frame} -> {child_frame}"
                            )
                        elif count % 50 == 0:
                            self._log_throttled(
                                f"TF retrieved {count} transforms: {parent_frame} -> {child_frame}",
                                container_id,
                            )

            except Exception as e:
                msg_count = self.subscriber_status.get(container_id, {}).get(
                    "message_count", 0
                )
                if msg_count < 5:
                    self._log_throttled(
                        f"TF waiting: {parent_frame} -> {child_frame} ({str(e)[:60]})",
                        f"tf_{container_id}",
                    )
        except Exception as e:
            pass

    def lookup_tf_at_time(self, container_id, ros_time, timeout_sec=0.05):
        """Look up TF at a specific ROS timestamp.

        Used by the synchronized recording callback to retrieve the TF transform
        that corresponds to the same header.stamp as the synchronized topic messages.
        Falls back to None on any failure so the caller can use sample-and-hold.
        """
        tf_info = self.tf_lookup_containers.get(container_id)
        if not tf_info or not tf_info.get("buffer"):
            return None
        try:
            transform = tf_info["buffer"].lookup_transform(
                tf_info["parent_frame"],
                tf_info["child_frame"],
                ros_time,
                timeout=rclpy_duration(seconds=timeout_sec),
            )
            t = transform.transform.translation
            r = transform.transform.rotation
            if not (t.x == t.x and t.y == t.y and t.z == t.z and
                    r.x == r.x and r.y == r.y and r.z == r.z and r.w == r.w):
                return None
            return transform
        except Exception:
            return None

    def debug_subscribers(self):
        try:
            logger.info(f"Subscribers: {len(self.subscribers)}")
            for container_id, status in self.subscriber_status.items():
                logger.info(
                    f"  {container_id}: {status.get('topic', 'unknown')} | "
                    f"Status: {status.get('status')} | "
                    f"Messages: {status.get('message_count', 0)} | "
                    f"Data: {self.has_data(container_id)}"
                )
        except Exception as e:
            logger.error(f"Debug error: {e}")

    def get_content(self, container_id=None):
        try:
            with self.data_lock:
                if container_id:
                    return self.latest_data.get(container_id, None)
                else:
                    return dict(self.latest_data)
        except Exception as e:
            return None

    def get_timestamps(self):
        with self.data_lock:
            return dict(self.latest_timestamps)

    def get_content_and_timestamps(self):
        """Atomic snapshot of both latest_data and latest_timestamps under a single lock hold.

        Avoids the race condition where topic callbacks fire between a separate
        get_content() and get_timestamps() call, making topics appear fresher than TF.
        """
        with self.data_lock:
            return dict(self.latest_data), dict(self.latest_timestamps)

    def flush_latest_data(self):
        """Reset all latest_data entries to None and clear timestamps.

        Used by the API before starting a recording so that stale data accumulated
        since initialize() does not contaminate the first samples of the session.
        """
        with self.data_lock:
            for k in self.latest_data:
                self.latest_data[k] = None
            self.latest_timestamps.clear()
        logger.info("Flushed latest_data and timestamps before recording start")

    def get_container_data(self, container):
        container_id = getattr(container, "container_id", None)
        if container_id:
            return self.get_content(container_id)
        return None

    def get_topic_list(self):
        return [status.get("topic", "") for status in self.subscriber_status.values()]

    def get_all_data_summary(self):
        summary = {
            "total_subscribers": len(self.subscribers),
            "active_subscribers": sum(
                1
                for s in self.subscriber_status.values()
                if s.get("status") == "active"
            ),
            "containers_with_data": sum(
                1 for cid, d in self.latest_data.items() if d is not None
            ),
            "total_messages_received": sum(
                s.get("message_count", 0) for s in self.subscriber_status.values()
            ),
            "subscriber_details": [],
        }

        for container_id, status in self.subscriber_status.items():
            summary["subscriber_details"].append(
                {
                    "container_id": container_id,
                    "topic": status.get("topic", "unknown"),
                    "message_count": status.get("message_count", 0),
                    "has_data": self.has_data(container_id),
                    "status": status.get("status", "unknown"),
                }
            )

        return summary

    def get_message_count(self, container_id):
        return self.subscriber_status.get(container_id, {}).get("message_count", 0)

    def get_subscriber_status(self, container_id=None):
        if container_id:
            return self.subscriber_status.get(container_id, {})
        return dict(self.subscriber_status)

    def _get_message_class_from_cache(self, topic_path):
        return self.message_type_cache.get(topic_path)
