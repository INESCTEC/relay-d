"""ROS message -> the plain sample dict that DataRecorder keeps and writes to HDF5.

No state and no file access: one function per message type, picked by ``encode_message``.
"""

import numpy as np
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import String, Float32, Float64, Int32, Bool

from relay_d.utils.ros_media_codec import decode_image, decode_organized_pointcloud


def encode_message(msg, modality=None):
    """The sample dict of ``msg``. ``modality`` is the input's configured "rgb" | "depth" | "scan" (cameras, point clouds)."""
    if isinstance(msg, (Image, CompressedImage)):
        return _encode_image(msg, modality)
    elif isinstance(msg, PointCloud2):
        return _encode_pointcloud(msg, modality)
    elif isinstance(msg, JointState) or (
        hasattr(msg, "name") and hasattr(msg, "position")
        and hasattr(msg, "velocity") and hasattr(msg, "effort")
        and hasattr(msg, "header")
    ):
        return _encode_joint_state(msg)
    elif isinstance(msg, PoseStamped):
        return _encode_pose(msg)
    elif isinstance(msg, Twist):
        return _encode_twist(msg)
    elif isinstance(msg, TransformStamped):
        return _encode_transform_stamped(msg)
    elif isinstance(msg, (String, Float32, Float64, Int32, Bool)):
        return _encode_simple(msg)
    else:
        return _encode_generic(msg)


def _encode_image(img_msg, modality):
    if isinstance(img_msg, CompressedImage):
        return {
            "type": "compressed_image",
            "format": img_msg.format,
            "data": np.frombuffer(img_msg.data, np.uint8),
            "header": {
                "stamp": img_msg.header.stamp.sec * 1000000000
                + img_msg.header.stamp.nanosec,
                "frame_id": img_msg.header.frame_id,
            },
        }
    else:
        is_depth = modality == "depth"
        cv_image = decode_image(img_msg, is_depth=is_depth)
        if cv_image is None:
            return None
        return {
            "type": "image",
            "encoding": img_msg.encoding,
            "width": img_msg.width,
            "height": img_msg.height,
            "data": cv_image,
            "header": {
                "stamp": img_msg.header.stamp.sec * 1000000000
                + img_msg.header.stamp.nanosec,
                "frame_id": img_msg.header.frame_id,
            },
        }


def _encode_pointcloud(pc_msg, modality):
    field_names = [field.name for field in pc_msg.fields]
    wants_scan = modality == "scan"
    is_scan_raster = pc_msg.height > 1 and wants_scan

    if is_scan_raster:
        points = decode_organized_pointcloud(pc_msg)
        if points is None:
            return None
    else:
        points = None  # unorganized clouds are saved as metadata only: decoding them would just fill the buffer

    return {
        "type": "pointcloud",
        "organized": is_scan_raster,
        "width": pc_msg.width,
        "height": pc_msg.height,
        "fields": field_names,
        "points": points,
        "header": {
            "stamp": pc_msg.header.stamp.sec * 1000000000
            + pc_msg.header.stamp.nanosec,
            "frame_id": pc_msg.header.frame_id,
        },
    }


def _encode_joint_state(joint_msg):
    names = list(joint_msg.name) if joint_msg.name else []
    positions = list(joint_msg.position) if joint_msg.position else []
    velocities = list(joint_msg.velocity) if joint_msg.velocity else []
    efforts = list(joint_msg.effort) if joint_msg.effort else []

    positions = [float(p) for p in positions] if positions else []
    velocities = [float(v) for v in velocities] if velocities else []
    efforts = [float(e) for e in efforts] if efforts else []

    return {
        "type": "joint_state",
        "names": names,
        "positions": positions,
        "velocities": velocities,
        "efforts": efforts,
        "header": {
            "stamp": joint_msg.header.stamp.sec * 1000000000
            + joint_msg.header.stamp.nanosec,
            "frame_id": joint_msg.header.frame_id,
        },
    }


def _encode_pose(pose_msg):
    return {
        "type": "pose",
        "position": [
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ],
        "orientation": [
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w,
        ],
        "header": {
            "stamp": pose_msg.header.stamp.sec * 1000000000
            + pose_msg.header.stamp.nanosec,
            "frame_id": pose_msg.header.frame_id,
        },
    }


def _encode_twist(twist_msg):
    return {
        "type": "twist",
        "linear": [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z],
        "angular": [
            twist_msg.angular.x,
            twist_msg.angular.y,
            twist_msg.angular.z,
        ],
    }


def _encode_transform_stamped(transform_msg):
    return {
        "type": "transform_stamped",
        "parent_frame": transform_msg.header.frame_id,
        "child_frame": transform_msg.child_frame_id,
        "translation": [
            transform_msg.transform.translation.x,
            transform_msg.transform.translation.y,
            transform_msg.transform.translation.z,
        ],
        "rotation": [
            transform_msg.transform.rotation.x,
            transform_msg.transform.rotation.y,
            transform_msg.transform.rotation.z,
            transform_msg.transform.rotation.w,
        ],
        "header": {
            "stamp": transform_msg.header.stamp.sec * 1000000000
            + transform_msg.header.stamp.nanosec,
            "frame_id": transform_msg.header.frame_id,
        },
    }


def _encode_simple(msg):
    return {
        "type": type(msg).__name__.lower(),
        "data": msg.data if hasattr(msg, "data") else str(msg),
    }


def _encode_generic(msg):
    msg_data = {"type": type(msg).__name__, "fields": {}}

    if hasattr(msg, "__slots__"):
        for slot in msg.__slots__:
            try:
                value = getattr(msg, slot)
                if isinstance(value, (int, float, str, bool)):
                    msg_data["fields"][slot] = value
                else:
                    msg_data["fields"][slot] = str(value)
            except Exception:
                msg_data["fields"][slot] = "Error reading field"

    return msg_data
