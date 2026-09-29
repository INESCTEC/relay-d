"""Resolves a robot description (URDF XML string) from either a URDF/xacro
file or a live /robot_description topic, for feeding to a playback
robot_state_publisher.
"""

import os
import time

import rclpy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String

from relay_d.utils.coloring_logger import logger


class RobotDescriptionError(Exception):
    pass


def resolve_from_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        raise RobotDescriptionError(f"File not found: {path}")

    # Checked before ".urdf" so double-extension files (e.g. "robot.urdf.xacro")
    # go through the xacro processor rather than being read raw.
    if path.lower().endswith(".xacro"):
        try:
            import xacro
        except ImportError:
            raise RobotDescriptionError(
                "The 'xacro' Python package is not available. Install it via "
                "your ROS distro's package (e.g. `sudo apt install "
                "ros-<distro>-xacro`) and re-source ROS before retrying."
            )
        try:
            doc = xacro.process_file(path)
            return doc.toprettyxml(indent="  ")
        except Exception as e:
            raise RobotDescriptionError(f"Failed to process xacro file: {e}")

    try:
        with open(path, "r") as f:
            return f.read()
    except OSError as e:
        raise RobotDescriptionError(f"Failed to read file: {e}")


def resolve_from_topic(topic: str = "/robot_description", timeout_sec: float = 5.0) -> str:
    """Subscribe once to `topic` and return the first message received.

    Uses TRANSIENT_LOCAL durability to match robot_state_publisher's default
    latched QoS for robot_description, so a description published before this
    call started is still delivered.
    """
    node = None
    try:
        node = rclpy.create_node("robot_description_resolver")

        result = {"data": None}

        def _callback(msg):
            result["data"] = msg.data

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        sub = node.create_subscription(String, topic, _callback, qos_profile=qos)

        deadline = time.monotonic() + timeout_sec
        while result["data"] is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.destroy_subscription(sub)

        if result["data"] is None:
            raise RobotDescriptionError(
                f"No message received on '{topic}' within {timeout_sec:.0f}s. "
                "Is a robot_description publisher (e.g. robot_state_publisher) running?"
            )
        return result["data"]
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception as e:
                logger.warning(f"Error destroying robot_description_resolver node: {e}")
