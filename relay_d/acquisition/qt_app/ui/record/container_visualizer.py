"""Live ROS 2 message visualizer for Record page container cards."""

from __future__ import annotations

import datetime
import itertools
import time

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from sensor_msgs.msg import CompressedImage, Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, Float64, Int32, String
from tf2_msgs.msg import TFMessage

from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from relay_d.utils.numerics import stamp_to_sec
from relay_d.utils.ros_media_codec import imgmsg_to_cv2
from ..settings.config_manager import config


def _set_text_widget(container, names, text: str) -> bool:
    """Set text on the first existing widget in ``names`` that supports ``setText``."""
    for name in names:
        widget = getattr(container, name, None)
        if widget is not None and hasattr(widget, "setText"):
            widget.setText(text)
            return True
    return False


class ContainerVisualizer:
    """Periodically renders the latest ROS messages into each container card."""

    def __init__(self):
        self.visualization_timers = {}  # container_id -> QTimer
        self.update_mutex = QMutex()
        self.update_rate = user_settings.get("visualizer_update_rate", 20) or 20
        self.framerate_window = {}  # container_id -> list of arrival times (rolling 2s FPS)
        self._last_msg_ref = {}  # container_id -> last rendered message object (skip unchanged ticks)
        config.settings_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, settings):
        if "visualizer_update_rate" in settings:
            rate = settings["visualizer_update_rate"]
            if rate and rate > 0 and rate != self.update_rate:
                self.update_rate = rate
                for timer in self.visualization_timers.values():
                    timer.setInterval(1000 // self.update_rate)

        if "freeze_containers" in settings:
            is_frozen = settings["freeze_containers"]
            logger.info(f"Container visualization freeze: {is_frozen}")
            for timer in self.visualization_timers.values():
                if is_frozen:
                    timer.stop()
                else:
                    timer.start(1000 // self.update_rate)

    def setup_container_visualization(self, container, topic_subscribers):
        try:
            container_id = getattr(container, "container_id", None)
            if not container_id:
                logger.error("Container missing container_id for visualization")
                return False

            timer = QTimer()
            timer.timeout.connect(
                lambda: self.update_container_display(container, topic_subscribers)
            )
            timer.start(1000 // self.update_rate)
            self.visualization_timers[container_id] = timer
            return True
        except Exception as e:
            logger.error(f"Failed to setup container visualization: {e}")
            return False

    def update_container_display(self, container, topic_subscribers):
        self.update_mutex.lock()
        try:
            container_id = getattr(container, "container_id", None)
            if not container_id:
                return

            latest_msg = topic_subscribers.get_content(container_id)
            if latest_msg is None:
                self._last_msg_ref.pop(container_id, None)
                self._update_no_data_display(container)
                return

            # Skip redundant decoding/scaling when no new message arrived since the last tick.
            if latest_msg is self._last_msg_ref.get(container_id):
                return
            self._last_msg_ref[container_id] = latest_msg

            if isinstance(latest_msg, (Image, CompressedImage)):
                self._update_image_display(container, latest_msg)
                self._update_rate_and_timestamp(container, container_id, latest_msg)
            elif isinstance(latest_msg, PointCloud2):
                self._update_pointcloud_display(container, latest_msg)
            elif isinstance(latest_msg, JointState):
                self._update_joint_display(container, latest_msg)
            elif isinstance(latest_msg, PoseStamped):
                p = latest_msg.pose
                self._update_pose_or_transform(
                    container, p.position, p.orientation,
                    parent_frame=getattr(getattr(latest_msg, "header", None), "frame_id", ""),
                    title="Pose Data",
                )
                self._update_rate_and_timestamp(container, container_id, latest_msg)
            elif isinstance(latest_msg, Twist):
                self._update_twist_display(container, latest_msg)
            elif isinstance(latest_msg, TFMessage):
                if latest_msg.transforms:
                    tf = latest_msg.transforms[0]
                    self._update_pose_or_transform(
                        container, tf.transform.translation, tf.transform.rotation,
                        parent_frame=tf.header.frame_id, child_frame=tf.child_frame_id, title="TF Data",
                    )
                    self._update_rate_and_timestamp(container, container_id, tf)
            elif isinstance(latest_msg, TransformStamped):
                self._update_pose_or_transform(
                    container, latest_msg.transform.translation, latest_msg.transform.rotation,
                    parent_frame=latest_msg.header.frame_id, child_frame=latest_msg.child_frame_id,
                    title="TF Transform",
                )
                self._update_rate_and_timestamp(container, container_id, latest_msg)
            elif isinstance(latest_msg, (String, Float32, Float64, Int32, Bool)):
                self._update_simple_data_display(container, latest_msg)
            else:
                self._update_unknown_display(container, latest_msg)
        except Exception as e:
            logger.error(f"Error updating container display: {e}")
        finally:
            self.update_mutex.unlock()

    def _update_rate_and_timestamp(self, container, container_id, msg):
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None and hasattr(container, "label_timestamp"):
            dt_str = datetime.datetime.fromtimestamp(stamp_to_sec(stamp)).strftime("%H:%M:%S.%f")[:-3]
            container.label_timestamp.setText(dt_str)

        now = time.time()
        window = [t for t in self.framerate_window.setdefault(container_id, []) if now - t < 2.0]
        window.append(now)
        self.framerate_window[container_id] = window
        if hasattr(container, "label_frame_rate"):
            container.label_frame_rate.setText(f"{len(window) / 2.0:.1f} FPS")

    def _update_image_display(self, container, image_msg):
        try:
            is_rgb = False
            if isinstance(image_msg, CompressedImage):
                cv_image = cv2.imdecode(np.frombuffer(image_msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                try:
                    cv_image = imgmsg_to_cv2(image_msg, "rgb8")
                    is_rgb = True
                except ValueError:
                    cv_image = imgmsg_to_cv2(image_msg, "bgr8")

            if cv_image is None:
                return
            self._display_image(container, cv_image, is_rgb=is_rgb)
        except Exception as e:
            logger.error(f"Error updating image display: {e}")
            _set_text_widget(
                container, ("label_actuator_state", "dataVisualizationLabel", "text_data_display"),
                f"Image Error: {str(e)[:50]}",
            )

    def _display_image(self, container, cv_image, is_rgb: bool = False):
        label = getattr(container, "label_image", None)
        if label is None or label.width() <= 0 or label.height() <= 0:
            return
        rgb = np.ascontiguousarray(cv_image if is_rgb else cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        qt_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _update_pointcloud_display(self, container, pc_msg):
        try:
            num_points = pc_msg.width * pc_msg.height
            field_names = [field.name for field in pc_msg.fields]
            if hasattr(container, "text_data_display"):
                try:
                    sample_points = list(
                        itertools.islice(
                            point_cloud2.read_points(pc_msg, field_names=field_names, skip_nans=True),
                            5,
                        )
                    )
                    lines = [f"Point Cloud ({num_points} points)", f"Fields: {', '.join(field_names)}", "", "Sample points:"]
                    lines.extend(f"Point {i + 1}: {pt}" for i, pt in enumerate(sample_points))
                    container.text_data_display.setText("\n".join(lines))
                except Exception as e:
                    container.text_data_display.setText(f"Point Cloud ({num_points} points)\nError reading points: {e}")
        except Exception as e:
            logger.error(f"Error updating point cloud display: {e}")

    def _update_joint_display(self, container, joint_msg):
        try:
            names = joint_msg.name
            if hasattr(container, "dataVisualizationLabel"):
                lines = [f"Joint States ({len(names)} joints)\n"]
                for i, name in enumerate(names[:10]):
                    pos = joint_msg.position[i] if i < len(joint_msg.position) else 0.0
                    vel = joint_msg.velocity[i] if i < len(joint_msg.velocity) else 0.0
                    eff = joint_msg.effort[i] if i < len(joint_msg.effort) else 0.0
                    lines.append(f"{name}: pos={pos:.3f}, vel={vel:.3f}, eff={eff:.3f}")
                if len(names) > 10:
                    lines.append(f"... and {len(names) - 10} more")
                container.dataVisualizationLabel.setText("\n".join(lines))
                container.dataVisualizationLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)

            if hasattr(container, "positionLabel") and joint_msg.position:
                container.positionLabel.setText(f"Pos {sum(joint_msg.position) / len(joint_msg.position):.3f}")
            if hasattr(container, "velocityLabel") and joint_msg.velocity:
                container.velocityLabel.setText(f"Vel {sum(abs(v) for v in joint_msg.velocity) / len(joint_msg.velocity):.3f}")
            if hasattr(container, "torqueLabel") and joint_msg.effort:
                container.torqueLabel.setText(f"Eff {sum(abs(e) for e in joint_msg.effort) / len(joint_msg.effort):.3f}")
            if hasattr(container, "text_data_display"):
                container.text_data_display.setText(f"{getattr(container, 'container_id', 'joints')} · {len(names)} joints")
        except Exception as e:
            logger.error(f"Error updating joint display: {e}")

    def _update_pose_or_transform(self, container, pos, quat, parent_frame="", child_frame="", title="Pose Data"):
        try:
            if hasattr(container, "lineEdit_pos_x"):
                if parent_frame and hasattr(container, "lineEdit_parent_frame"):
                    container.lineEdit_parent_frame.setText(parent_frame)
                if child_frame and hasattr(container, "lineEdit_child_frame"):
                    container.lineEdit_child_frame.setText(child_frame)
                container.lineEdit_pos_x.setText(f"{pos.x:.4f}")
                container.lineEdit_pos_y.setText(f"{pos.y:.4f}")
                container.lineEdit_pos_z.setText(f"{pos.z:.4f}")
                container.lineEdit_quat_x.setText(f"{quat.x:.4f}")
                container.lineEdit_quat_y.setText(f"{quat.y:.4f}")
                container.lineEdit_quat_z.setText(f"{quat.z:.4f}")
                container.lineEdit_quat_w.setText(f"{quat.w:.4f}")
                return

            lines = [title, ""]
            if parent_frame or child_frame:
                lines.append(f"Parent: {parent_frame}  Child: {child_frame}")
            lines.extend([
                f"Position: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})",
                f"Orientation: ({quat.x:.4f}, {quat.y:.4f}, {quat.z:.4f}, {quat.w:.4f})",
            ])
            _set_text_widget(container, ("text_data_display", "dataVisualizationLabel", "label_actuator_state"), "\n".join(lines))
        except Exception as e:
            logger.error(f"Error updating {title} display: {e}")

    def _update_twist_display(self, container, twist_msg):
        try:
            lin, ang = twist_msg.linear, twist_msg.angular
            text = (
                f"Twist Data\n\n"
                f"Linear:  ({lin.x:.4f}, {lin.y:.4f}, {lin.z:.4f})\n"
                f"Angular: ({ang.x:.4f}, {ang.y:.4f}, {ang.z:.4f})"
            )
            _set_text_widget(container, ("text_data_display", "dataVisualizationLabel", "label_actuator_state"), text)
        except Exception as e:
            logger.error(f"Error updating twist display: {e}")

    def _update_simple_data_display(self, container, msg):
        try:
            value = msg.data if hasattr(msg, "data") else str(msg)
            text = f"{type(msg).__name__}\n\nValue: {value}"
            _set_text_widget(container, ("label_actuator_state", "text_data_display", "dataVisualizationLabel"), text)
        except Exception as e:
            logger.error(f"Error updating simple data display: {e}")

    def _update_unknown_display(self, container, msg):
        try:
            lines = [f"Type: {type(msg).__name__}"]
            for slot in getattr(msg, "__slots__", ())[:10]:
                val = getattr(msg, slot, None)
                if val is not None:
                    lines.append(f"  {slot}: {val}")
            _set_text_widget(container, ("text_data_display", "dataVisualizationLabel", "label_actuator_state"), "\n".join(lines))
        except Exception as e:
            logger.error(f"Error updating unknown display: {e}")

    def _update_no_data_display(self, container):
        try:
            _set_text_widget(container, ("text_data_display", "dataVisualizationLabel", "label_actuator_state"), "No data received")
            label = getattr(container, "label_image", None)
            if label is not None:
                label.clear()
        except Exception as e:
            logger.error(f"Error updating no data display: {e}")

    def stop_all_visualizations(self):
        for timer in list(self.visualization_timers.values()):
            timer.stop()
        self.visualization_timers.clear()
        self._last_msg_ref.clear()
        self.framerate_window.clear()
