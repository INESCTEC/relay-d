"""
tf_topic.py
-----------
Shared support for reading a TF2 tree off a *non-default* topic pair.

tf2_ros.TransformListener hardcodes subscriptions to the absolute topics
'/tf' and '/tf_static' with no way to override them. `_TfTopicListener`
mirrors its subscription setup (same QoS defaults, same callback group, same
buffer feed) with the topic names parameterized, so a TF tree published
under e.g. '/robot_1/tf' can be read the same way as the default tree.

`get_or_create_tf_buffer` is the shared cache-or-create entry point used by
both the acquisition (topic_subscribers.py) and dispatch (obs_builder.py)
sides: given a dict of already-created buffers/listeners keyed by
`(tf_topic, tf_static_topic)`, it returns the existing buffer for that pair
or creates a new one.

Fail-loud contract: if creation fails for a pair OTHER than `default_key`,
`TfTopicSubscriptionError` is raised — a deliberately-configured custom TF
topic that can't be subscribed to must never silently fall back to reading
the default /tf, /tf_static tree instead. Failure on `default_key` itself
keeps the historical soft-fail (log + return None), since that path covers
setups that don't use TF at all or environments where tf2_ros itself isn't
available for unrelated reasons.
"""

import rclpy.time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as rclpy_duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from tf2_msgs.msg import TFMessage

from relay_d.utils.coloring_logger import logger

try:
    from tf2_ros import Buffer
except ImportError:
    Buffer = None


class TfTopicSubscriptionError(RuntimeError):
    """A configured custom tf_topic/tf_static_topic pair could not be
    subscribed to, or never produced the requested transform. Deliberately
    NOT swallowed by generic exception handlers along the call path — see
    module docstring."""


class _TfTopicListener:
    """Like tf2_ros.TransformListener, but the tf/tf_static topic names are
    configurable. tf2_ros.TransformListener hardcodes subscriptions to the
    absolute topics '/tf' and '/tf_static' with no way to override them, so
    this mirrors its subscription setup (same QoS defaults, same callback
    group, same buffer feed) with the topic names parameterized — used to
    read a TF tree published under a non-default topic (e.g. /robot_1/tf).
    """

    def __init__(self, buffer, node, tf_topic, tf_static_topic):
        self.buffer = buffer
        self.node = node
        # Reentrant so lookup_transform() can be called from within a TF
        # callback without deadlocking, same as upstream TransformListener.
        self.group = ReentrantCallbackGroup()

        qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.tf_sub = node.create_subscription(
            TFMessage, tf_topic, self._callback, qos, callback_group=self.group
        )
        self.tf_static_sub = node.create_subscription(
            TFMessage,
            tf_static_topic,
            self._static_callback,
            static_qos,
            callback_group=self.group,
        )

    def _callback(self, data):
        for transform in data.transforms:
            self.buffer.set_transform(transform, "default_authority")

    def _static_callback(self, data):
        for transform in data.transforms:
            self.buffer.set_transform_static(transform, "default_authority")

    def unregister(self):
        self.node.destroy_subscription(self.tf_sub)
        self.node.destroy_subscription(self.tf_static_sub)


def get_or_create_tf_buffer(
    buffers: dict,
    listeners: dict,
    node,
    tf_topic: str,
    tf_static_topic: str,
    default_key,
    cache_time: float = 10.0,
):
    """Return the Buffer feeding from (tf_topic, tf_static_topic), creating a
    new Buffer + _TfTopicListener pair on `node` if one for this exact topic
    pair doesn't exist yet in `buffers`/`listeners`.

    Raises TfTopicSubscriptionError if creation fails for any key other than
    `default_key` (see module docstring). Returns None if creation fails for
    `default_key` itself, or if tf2_ros/`node` aren't available at all.
    """
    key = (tf_topic, tf_static_topic)
    if key in buffers:
        return buffers[key]

    is_custom = key != default_key

    if Buffer is None or node is None:
        msg = f"TF2 not available — cannot create TF buffer for {tf_topic}, {tf_static_topic}"
        if is_custom:
            raise TfTopicSubscriptionError(msg)
        logger.error(msg)
        return None

    try:
        buf = Buffer(cache_time=rclpy_duration(seconds=cache_time))
        listener = _TfTopicListener(buf, node, tf_topic, tf_static_topic)
        buffers[key] = buf
        listeners[key] = listener
        logger.info(f"TF buffer created for topics {tf_topic}, {tf_static_topic}")
        return buf
    except Exception as e:
        msg = f"Could not create TF buffer for {tf_topic}/{tf_static_topic}: {e}"
        if is_custom:
            raise TfTopicSubscriptionError(msg) from e
        logger.warning(msg)
        return None


def wait_for_transform(buffer, parent_frame: str, child_frame: str,
                        timeout: float, poll: float = 0.2) -> bool:
    """Poll buffer.lookup_transform(parent_frame, child_frame) until it
    succeeds or `timeout` seconds elapse. Returns whether it ever succeeded.
    """
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            buffer.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
            return True
        except Exception:
            pass
        _time.sleep(poll)
    return False
