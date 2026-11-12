"""Subscription QoS that can actually connect to the publisher."""

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger


def subscription_qos(node, topic: str, depth: int = None) -> QoSProfile:
    """Reliable, unless a publisher on ``topic`` is best-effort.

    A reliable subscriber never connects to a best-effort publisher (typical for camera and
    lidar drivers), so the topic would look alive but deliver nothing. A best-effort subscriber
    connects to both kinds.
    """
    depth = depth or user_settings.get("qos_depth")  # setting: queue depth
    reliability = QoSReliabilityPolicy.RELIABLE
    try:
        infos = node.get_publishers_info_by_topic(topic)
        if any(i.qos_profile.reliability == QoSReliabilityPolicy.BEST_EFFORT for i in infos):
            reliability = QoSReliabilityPolicy.BEST_EFFORT
            logger.info(f"'{topic}': publisher is best-effort, subscribing best-effort")
    except Exception as e:
        logger.warning(f"'{topic}': could not read publisher QoS ({e}); subscribing reliable")
    return QoSProfile(depth=depth, reliability=reliability)
