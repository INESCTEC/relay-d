"""
ros_media_codec.py
-------------------
Pure decode functions for sensor_msgs/Image and organized sensor_msgs/PointCloud2
messages, shared between the acquisition recorder
(relay_d/acquisition/qt_app/ui/record/data_recorder.py) and the live inference
observation builder (relay_d/dispatch/obs_builder.py) so both sides decode
pixel/point data identically — a live obs and its recorded-then-postprocessed
counterpart should never silently diverge because of two hand-written copies of
the same decode logic.

No ROS node / Qt dependency — plain functions over already-received message
objects, callable from either module without pulling in the other's
dependencies.

CompressedImage is intentionally NOT handled here: it isn't supported as an
`output.streams`/bridge-map obs on the acquisition side either (variable-length
bytes, no fixed per-frame shape), so there is nothing for a live obs to mirror.
Unorganized PointCloud2 (height <= 1) is likewise unsupported on both sides —
no fixed per-frame shape to decode into.
"""
import numpy as np
from sensor_msgs_py import point_cloud2

from relay_d.utils.coloring_logger import logger


def decode_image(img_msg, cv_bridge, is_depth: bool = False):
    """Decode a sensor_msgs/Image into an (H, W, C) or (H, W) numpy array.

    `is_depth` selects cv_bridge's "passthrough" mode instead of "bgr8" so the
    native depth encoding (typically 32FC1 float meters or 16UC1 uint16
    millimeters) is preserved instead of being reinterpreted as color.

    Returns None on decode failure.
    """
    try:
        desired_encoding = "passthrough" if is_depth else "bgr8"
        return cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding)
    except Exception as e:
        logger.error(f"Error decoding image message: {e}")
        return None


def decode_organized_pointcloud(pc_msg):
    """Decode an ORGANIZED sensor_msgs/PointCloud2 (height > 1) into a fixed
    (height, width, len(fields)) float32 array, with NaN (invalid/no-return)
    points zeroed rather than dropped, so every frame has the same shape.

    Returns None for unorganized/ragged clouds (height <= 1) — those have no
    fixed per-frame shape and are not supported here — or on decode failure.
    """
    try:
        if pc_msg.height <= 1:
            return None
        field_names = [field.name for field in pc_msg.fields]
        # skip_nans=False preserves the (height, width) raster layout —
        # skipping NaNs would misalign the reshape below.
        raw = point_cloud2.read_points(pc_msg, field_names=field_names, skip_nans=False)
        # read_points returns a structured array (named per-field dtype) on
        # this ROS distro — flatten to plain float32 columns before reshaping
        # into the (H, W, fields) raster.
        if isinstance(raw, np.ndarray) and raw.dtype.names:
            from numpy.lib import recfunctions as rfn
            flat = rfn.structured_to_unstructured(raw, dtype=np.float32)
        else:
            flat = np.array(list(raw), dtype=np.float32)
        points = flat.reshape(pc_msg.height, pc_msg.width, len(field_names))
        return np.nan_to_num(points, nan=0.0)
    except Exception as e:
        logger.error(f"Error decoding pointcloud message: {e}")
        return None
