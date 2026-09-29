import cv2
import numpy as np
import json
import time
import tempfile
import os
import rclpy
import datetime
import math

from relay_d.utils.coloring_logger import logger
from ..settings.config_manager import config

from sensor_msgs.msg import Image, CompressedImage, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import String, Float32, Float64, Int32, Bool
from tf2_msgs.msg import TFMessage
from sensor_msgs_py import point_cloud2
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QTextEdit, QProgressBar
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, QMutex, QUrl, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from cv_bridge import CvBridge


class ContainerVisualizer:
    """
    Handles visualization of ROS message content in containers
    """

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.visualization_timers = {}  # container_id -> QTimer
        self.update_mutex = QMutex()
        self.media_players = {}  # container_id -> QMediaPlayer
        self.temp_video_files = {}  # container_id -> temp file path

        # Visualization update rate (Hz)
        self.update_rate = config.get("update_rate", 20) or 20

        self.last_msg_time = {}  # container_id -> last header.stamp
        self.last_msg_arrival = {}  # container_id -> wall-time of last message
        self.framerate_window = {}  # container_id -> list of arrival times (for rolling FPS)

        # Connect to config changes for update_rate
        config.settings_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, settings):
        """Handle settings changes via signal (Triggered when Save is clicked)"""
        # 1. Update Update Rate
        if "update_rate" in settings:
            rate = settings["update_rate"]
            if rate and rate > 0:
                self.update_rate = rate
                for timer in self.visualization_timers.values():
                    timer.setInterval(1000 // self.update_rate)

        # 2. PERFORMANCE FIX: Start/Stop timers based on freeze state
        if "freeze_containers" in settings:
            is_frozen = settings["freeze_containers"]
            logger.info(f"Container visualization freeze: {is_frozen}")

            for container_id, timer in self.visualization_timers.items():
                if is_frozen:
                    timer.stop()  # Stops the CPU usage for this container
                else:
                    timer.start(1000 // self.update_rate)  # Restarts visualization

    def setup_container_visualization(self, container, topic_subscribers):
        """
        Set up visualization for a container

        Args:
            container: The container widget
            topic_subscribers: The TopicSubscribers instance
        """
        try:
            container_id = getattr(container, "container_id", None)
            if not container_id:
                logger.error("Container missing container_id for visualization")
                return False

            # Create and start timer for this container
            timer = QTimer()
            timer.timeout.connect(
                lambda: self.update_container_display(container, topic_subscribers)
            )

            # Always start timer - freeze will stop/start it via signal
            timer.start(1000 // self.update_rate)

            self.visualization_timers[container_id] = timer

            logger.info(f"Set up visualization for container {container_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup container visualization: {e}")
            return False

    def update_container_display(self, container, topic_subscribers):
        """
        Update the visual display of a container with latest message data

        Args:
            container: The container widget
            topic_subscribers: The TopicSubscribers instance
        """
        self.update_mutex.lock()

        try:
            container_id = getattr(container, "container_id", None)
            if not container_id:
                return

            # Get latest message data
            latest_msg = topic_subscribers.get_content(container_id)

            # Debug: Log what we got
            if container_id and "TF" in str(getattr(container, "topic_path", "")):
                logger.info(
                    f"DEBUG Viz: container={container_id}, latest_msg={latest_msg}, type={type(latest_msg) if latest_msg else None}"
                )

            # Debug logging for TF containers
            if latest_msg is not None and isinstance(latest_msg, TransformStamped):
                logger.debug(
                    f"Visualizer: Received TransformStamped for {container_id}"
                )
                logger.debug(
                    f"Visualizer: Parent={latest_msg.header.frame_id}, Child={latest_msg.child_frame_id}"
                )

            if latest_msg is None:
                self._update_no_data_display(container)
                return

            # Update display based on message type
            msg_type = type(latest_msg).__name__

            if isinstance(latest_msg, (Image, CompressedImage)):
                self._update_image_display(container, latest_msg)

                ## Add framerate and timestamp
                container_id = getattr(container, "container_id", None)

                # === Timestamp (ROS time) ===
                if hasattr(latest_msg, "header") and hasattr(
                    latest_msg.header, "stamp"
                ):
                    ros_time = latest_msg.header.stamp
                    if ros_time:
                        time_sec = ros_time.sec + ros_time.nanosec * 1e-9
                        dt_str = datetime.datetime.fromtimestamp(time_sec).strftime(
                            "%H:%M:%S.%f"
                        )[:-3]
                        if hasattr(container, "label_timestamp"):
                            container.label_timestamp.setText(f"Time: {dt_str}")

                # === Framerate ===
                now = time.time()
                window = self.framerate_window.setdefault(container_id, [])
                window.append(now)
                # Keep only last 2 seconds of timestamps
                window = [t for t in window if now - t < 2.0]
                self.framerate_window[container_id] = window

                fps = len(window) / 2.0  # messages per second (avg over 2s)
                if hasattr(container, "label_frame_rate"):
                    container.label_frame_rate.setText(f"FPS: {fps:.1f}")

            elif isinstance(latest_msg, PointCloud2):
                self._update_pointcloud_display(container, latest_msg)
            elif isinstance(latest_msg, JointState):
                self._update_joint_display(container, latest_msg)
            elif isinstance(latest_msg, PoseStamped):
                self._update_pose_display(container, latest_msg)
            elif isinstance(latest_msg, Twist):
                self._update_twist_display(container, latest_msg)
            elif isinstance(latest_msg, TFMessage):
                self._update_tf_display(container, latest_msg)
            elif isinstance(latest_msg, TransformStamped):
                self._update_transform_stamped_display(container, latest_msg)
            elif isinstance(latest_msg, (String, Float32, Float64, Int32, Bool)):
                self._update_simple_data_display(container, latest_msg)
            else:
                self._update_unknown_display(container, latest_msg)

            # Update message count and timestamp
            message_count = topic_subscribers.get_message_count(container_id)
            self._update_status_info(container, msg_type, message_count)

        except Exception as e:
            logger.error(f"Error updating container display: {e}")

        finally:
            self.update_mutex.unlock()

    def _update_image_display(self, container, image_msg):
        """Update container with image data - specifically for QVideoWidget"""
        try:
            container_id = getattr(container, "container_id", "unknown")

            # Convert ROS image message to OpenCV format
            if isinstance(image_msg, CompressedImage):
                # Decompress compressed image
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # Convert regular image message
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

            if cv_image is None:
                return

            # Check for QVideoWidget named 'widget_video'
            if hasattr(container, "widget_video") and isinstance(
                container.widget_video, QVideoWidget
            ):
                self._display_image_in_video_widget(container, cv_image, container_id)
            else:
                # Fallback to QLabel display
                self._display_image_in_label(container, cv_image, image_msg)

            # Update image info if text area exists
            height, width = cv_image.shape[:2]
            info_text = f"Size: {width}x{height}\nChannels: {cv_image.shape[2] if len(cv_image.shape) > 2 else 1}\nEncoding: {getattr(image_msg, 'encoding', 'unknown')}"
            self._update_text_info(container, info_text)

        except Exception as e:
            logger.error(f"Error updating image display: {e}")
            # Fallback to text update
            self._update_text_status(container, f"Image Error: {str(e)[:50]}")

    def _display_image_in_video_widget(self, container, cv_image, container_id):
        """Display image in QVideoWidget using QMediaPlayer"""
        try:
            # Create or get existing media player
            if container_id not in self.media_players:
                media_player = QMediaPlayer()
                media_player.setVideoOutput(container.widget_video)
                self.media_players[container_id] = media_player
                logger.info(f"Created media player for container {container_id}")
            else:
                media_player = self.media_players[container_id]

            # Convert OpenCV image to a format suitable for QMediaPlayer
            # For real-time display, we'll convert the image to a QLabel approach
            # since QMediaPlayer is more suited for video files

            # Alternative approach: Convert to QPixmap and display on an overlay
            self._display_image_as_pixmap_overlay(container, cv_image)

        except Exception as e:
            logger.error(f"Error displaying image in video widget: {e}")
            # Fallback to label display
            self._display_image_in_label(container, cv_image, None)

    def _display_image_as_pixmap_overlay(self, container, cv_image):
        """Display image as pixmap overlay on video widget"""
        try:
            # Convert OpenCV image to Qt format
            height, width, channels = cv_image.shape
            bytes_per_line = channels * width

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Create QImage
            qt_image = QImage(
                rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

            # Scale image to fit video widget
            video_widget = container.widget_video
            video_size = video_widget.size()

            if video_size.width() > 0 and video_size.height() > 0:
                # Scale maintaining aspect ratio
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    video_size.width(),
                    video_size.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,  # Keep aspect ratio
                )

                # Check if we need to create an overlay label
                if not hasattr(container, "_video_overlay_label"):
                    from PyQt5.QtWidgets import QLabel

                    overlay_label = QLabel(video_widget)
                    # Style defined in .ui file
                    overlay_label.resize(video_widget.size())
                    overlay_label.show()
                    container._video_overlay_label = overlay_label
                    logger.info(
                        f"Created video overlay label for {getattr(container, 'container_id', 'unknown')}"
                    )

                # Update overlay with new image
                container._video_overlay_label.setPixmap(scaled_pixmap)
                container._video_overlay_label.resize(video_widget.size())

            logger.debug(f"Updated video widget overlay with image {width}x{height}")

        except Exception as e:
            logger.error(f"Error displaying image as pixmap overlay: {e}")

    def _display_image_in_label(self, container, cv_image, image_msg):
        """Fallback method to display image in QLabel"""
        try:
            # Check if container has valid size
            if not hasattr(container, "size") or container.size().width() <= 0:
                logger.warning("Container has invalid size, skipping image update")
                return

            # Resize image to fit container
            height, width = cv_image.shape[:2]
            container_size = container.size()
            container_width = max(container_size.width() - 20, 50)  # Minimum 50px
            container_height = max(container_size.height() - 60, 50)  # Minimum 50px

            # Calculate scaling factor to maintain aspect ratio
            scale_x = container_width / width
            scale_y = container_height / height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale

            new_width = max(int(width * scale), 1)
            new_height = max(int(height * scale), 1)

            cv_image = cv2.resize(cv_image, (new_width, new_height))

            # Convert to Qt format
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qt_image)

            # Try to find image display widget with multiple possible names
            image_widget = None
            possible_names = ["label_image", "image_display", "imageLabel", "imgLabel"]

            for name in possible_names:
                if hasattr(container, name):
                    widget = getattr(container, name)
                    if hasattr(widget, "setPixmap"):
                        image_widget = widget
                        break

            if image_widget:
                image_widget.setPixmap(pixmap)
            else:
                # Create a basic status update if no image widget found
                self._update_text_status(container, f"Image: {width}x{height}")

        except Exception as e:
            logger.error(f"Error displaying image in label: {e}")

    def _update_pointcloud_display(self, container, pc_msg):
        """Update container with point cloud data"""
        try:
            # Extract basic point cloud information
            num_points = pc_msg.width * pc_msg.height
            point_step = pc_msg.point_step
            row_step = pc_msg.row_step

            # Get field names
            field_names = [field.name for field in pc_msg.fields]

            # Update point cloud info
            if hasattr(container, "label_data_info"):
                info_text = f"Points: {num_points}\nFields: {', '.join(field_names)}\nStep: {point_step}"
                container.label_data_info.setText(info_text)

            # If container has a text display, show some sample points
            if hasattr(container, "text_data_display"):
                try:
                    # Read first few points
                    points = list(
                        point_cloud2.read_points(
                            pc_msg, field_names=field_names, skip_nans=True
                        )
                    )
                    sample_points = points[:5]  # Show first 5 points

                    display_text = f"Point Cloud ({num_points} points)\n"
                    display_text += f"Fields: {', '.join(field_names)}\n\n"
                    display_text += "Sample points:\n"

                    for i, point in enumerate(sample_points):
                        display_text += f"Point {i + 1}: {point}\n"

                    container.text_data_display.setText(display_text)
                except Exception as e:
                    if hasattr(container, "text_data_display"):
                        container.text_data_display.setText(
                            f"Point Cloud\n{num_points} points\nError reading data: {str(e)}"
                        )

        except Exception as e:
            logger.error(f"Error updating pointcloud display: {e}")

    def _update_joint_display(self, container, joint_msg):
        """Update container with joint state data"""
        try:
            # Update the dataVisualizationLabel with detailed joint info
            if hasattr(container, "dataVisualizationLabel"):
                # Create display text showing first few joints in detail
                display_lines = []
                display_lines.append(f"Joint States ({len(joint_msg.name)} joints)\n")

                # Show joints in a more compact format
                num_joints = len(joint_msg.name)
                for i, name in enumerate(joint_msg.name):
                    position = (
                        joint_msg.position[i] if i < len(joint_msg.position) else 0.0
                    )
                    velocity = (
                        joint_msg.velocity[i] if i < len(joint_msg.velocity) else 0.0
                    )
                    effort = joint_msg.effort[i] if i < len(joint_msg.effort) else 0.0

                    # Format: name: pos=X.XX, vel=X.XX, eff=X.XX
                    display_lines.append(
                        f"{name}: pos={position:.3f}, vel={velocity:.3f}, eff={effort:.3f}"
                    )

                    # Limit display to avoid too much text
                    if i >= 9:
                        if num_joints > 10:
                            display_lines.append(f"... and {num_joints - 10} more")
                        break

                display_text = "\n".join(display_lines)
                container.dataVisualizationLabel.setText(display_text)
                container.dataVisualizationLabel.setAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                )

            # Update individual metric labels if they exist
            if hasattr(container, "positionLabel") and joint_msg.position:
                avg_pos = sum(joint_msg.position) / len(joint_msg.position)
                container.positionLabel.setText(f"Avg Pos: {avg_pos:.3f}")

            if hasattr(container, "velocityLabel") and joint_msg.velocity:
                avg_vel = sum(abs(v) for v in joint_msg.velocity) / len(
                    joint_msg.velocity
                )
                container.velocityLabel.setText(f"Avg Vel: {avg_vel:.3f} rad/s")

            if hasattr(container, "torqueLabel") and joint_msg.effort:
                avg_eff = sum(abs(e) for e in joint_msg.effort) / len(joint_msg.effort)
                container.torqueLabel.setText(f"Avg Eff: {avg_eff:.3f} Nm")

            # Update topic name label at bottom
            if hasattr(container, "text_data_display"):
                container.text_data_display.setText(
                    f"{len(joint_msg.name)} Joint States"
                )

        except Exception as e:
            logger.error(f"Error updating joint display: {e}")

    def _update_pose_display(self, container, pose_msg):
        """Update container with pose data"""
        try:
            pose = pose_msg.pose

            # Check if we have the new pose widget with individual fields
            if hasattr(container, "lineEdit_pos_x"):
                # New pose widget format - update individual fields
                if hasattr(pose_msg, "header") and hasattr(pose_msg.header, "frame_id"):
                    if hasattr(container, "lineEdit_parent_frame"):
                        container.lineEdit_parent_frame.setText(
                            pose_msg.header.frame_id
                        )

                container.lineEdit_pos_x.setText(f"{pose.position.x:.4f}")
                container.lineEdit_pos_y.setText(f"{pose.position.y:.4f}")
                container.lineEdit_pos_z.setText(f"{pose.position.z:.4f}")

                container.lineEdit_quat_x.setText(f"{pose.orientation.x:.4f}")
                container.lineEdit_quat_y.setText(f"{pose.orientation.y:.4f}")
                container.lineEdit_quat_z.setText(f"{pose.orientation.z:.4f}")
                container.lineEdit_quat_w.setText(f"{pose.orientation.w:.4f}")

                # Convert quaternion to Euler angles (roll, pitch, yaw)
                roll, pitch, yaw = self._quaternion_to_euler(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
                import math

                if hasattr(container, "lineEdit_roll"):
                    container.lineEdit_roll.setText(f"{math.degrees(roll):.2f}")
                    container.lineEdit_pitch.setText(f"{math.degrees(pitch):.2f}")
                    container.lineEdit_yaw.setText(f"{math.degrees(yaw):.2f}")
            else:
                # Legacy format - use text display
                display_text = "Pose Data\n\n"
                display_text += f"Position:\n"
                display_text += f"  x: {pose.position.x:.4f}\n"
                display_text += f"  y: {pose.position.y:.4f}\n"
                display_text += f"  z: {pose.position.z:.4f}\n\n"
                display_text += f"Orientation:\n"
                display_text += f"  x: {pose.orientation.x:.4f}\n"
                display_text += f"  y: {pose.orientation.y:.4f}\n"
                display_text += f"  z: {pose.orientation.z:.4f}\n"
                display_text += f"  w: {pose.orientation.w:.4f}\n"

                if hasattr(container, "text_data_display"):
                    container.text_data_display.setText(display_text)
                elif hasattr(container, "label_data_info"):
                    short_text = f"Pose\nPos: ({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
                    container.label_data_info.setText(short_text)

        except Exception as e:
            logger.error(f"Error updating pose display: {e}")

    def _quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _update_twist_display(self, container, twist_msg):
        """Update container with twist (velocity) data"""
        try:
            linear = twist_msg.linear
            angular = twist_msg.angular

            display_text = "Twist Data\n\n"
            display_text += f"Linear:\n"
            display_text += f"  x: {linear.x:.4f}\n"
            display_text += f"  y: {linear.y:.4f}\n"
            display_text += f"  z: {linear.z:.4f}\n\n"
            display_text += f"Angular:\n"
            display_text += f"  x: {angular.x:.4f}\n"
            display_text += f"  y: {angular.y:.4f}\n"
            display_text += f"  z: {angular.z:.4f}\n"

            if hasattr(container, "text_data_display"):
                container.text_data_display.setText(display_text)
            elif hasattr(container, "label_data_info"):
                short_text = (
                    f"Twist\nLin: ({linear.x:.2f}, {linear.y:.2f}, {linear.z:.2f})"
                )
                container.label_data_info.setText(short_text)

        except Exception as e:
            logger.error(f"Error updating twist display: {e}")

    def _update_tf_display(self, container, tf_msg):
        """Update container with TF (transform) data"""
        try:
            if not tf_msg.transforms:
                return

            # Get the first transform
            transform = tf_msg.transforms[0]

            # Check if we have the new pose widget with individual fields
            if hasattr(container, "lineEdit_pos_x"):
                # New pose widget format - update individual fields
                if hasattr(container, "lineEdit_parent_frame"):
                    container.lineEdit_parent_frame.setText(transform.header.frame_id)
                if hasattr(container, "lineEdit_child_frame"):
                    container.lineEdit_child_frame.setText(transform.child_frame_id)

                container.lineEdit_pos_x.setText(
                    f"{transform.transform.translation.x:.4f}"
                )
                container.lineEdit_pos_y.setText(
                    f"{transform.transform.translation.y:.4f}"
                )
                container.lineEdit_pos_z.setText(
                    f"{transform.transform.translation.z:.4f}"
                )

                container.lineEdit_quat_x.setText(
                    f"{transform.transform.rotation.x:.4f}"
                )
                container.lineEdit_quat_y.setText(
                    f"{transform.transform.rotation.y:.4f}"
                )
                container.lineEdit_quat_z.setText(
                    f"{transform.transform.rotation.z:.4f}"
                )
                container.lineEdit_quat_w.setText(
                    f"{transform.transform.rotation.w:.4f}"
                )

                # Convert quaternion to Euler angles (roll, pitch, yaw)
                roll, pitch, yaw = self._quaternion_to_euler(
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
                import math

                if hasattr(container, "lineEdit_roll"):
                    container.lineEdit_roll.setText(f"{math.degrees(roll):.2f}")
                    container.lineEdit_pitch.setText(f"{math.degrees(pitch):.2f}")
                    container.lineEdit_yaw.setText(f"{math.degrees(yaw):.2f}")
            else:
                # Legacy format - use text display
                display_text = "TF Data\n\n"
                display_text += f"Parent Frame: {transform.header.frame_id}\n"
                display_text += f"Child Frame: {transform.child_frame_id}\n\n"
                display_text += f"Translation:\n"
                display_text += f"  x: {transform.transform.translation.x:.4f}\n"
                display_text += f"  y: {transform.transform.translation.y:.4f}\n"
                display_text += f"  z: {transform.transform.translation.z:.4f}\n\n"
                display_text += f"Rotation (Quaternion):\n"
                display_text += f"  x: {transform.transform.rotation.x:.4f}\n"
                display_text += f"  y: {transform.transform.rotation.y:.4f}\n"
                display_text += f"  z: {transform.transform.rotation.z:.4f}\n"
                display_text += f"  w: {transform.transform.rotation.w:.4f}\n"

                if hasattr(container, "text_data_display"):
                    container.text_data_display.setText(display_text)
                elif hasattr(container, "label_data_info"):
                    short_text = (
                        f"TF: {transform.header.frame_id} -> {transform.child_frame_id}"
                    )
                    container.label_data_info.setText(short_text)

        except Exception as e:
            logger.error(f"Error updating TF display: {e}")

    def _update_transform_stamped_display(self, container, transform_msg):
        """Update container with TransformStamped data (from TF lookup)"""
        try:
            container_id = getattr(container, "container_id", "unknown")
            logger.debug(f"Updating TransformStamped display for {container_id}")

            if hasattr(container, "lineEdit_pos_x"):
                if hasattr(container, "lineEdit_parent_frame"):
                    container.lineEdit_parent_frame.setText(
                        transform_msg.header.frame_id
                    )
                if hasattr(container, "lineEdit_child_frame"):
                    container.lineEdit_child_frame.setText(transform_msg.child_frame_id)

                container.lineEdit_pos_x.setText(
                    f"{transform_msg.transform.translation.x:.4f}"
                )
                container.lineEdit_pos_y.setText(
                    f"{transform_msg.transform.translation.y:.4f}"
                )
                container.lineEdit_pos_z.setText(
                    f"{transform_msg.transform.translation.z:.4f}"
                )

                container.lineEdit_quat_x.setText(
                    f"{transform_msg.transform.rotation.x:.4f}"
                )
                container.lineEdit_quat_y.setText(
                    f"{transform_msg.transform.rotation.y:.4f}"
                )
                container.lineEdit_quat_z.setText(
                    f"{transform_msg.transform.rotation.z:.4f}"
                )
                container.lineEdit_quat_w.setText(
                    f"{transform_msg.transform.rotation.w:.4f}"
                )

                roll, pitch, yaw = self._quaternion_to_euler(
                    transform_msg.transform.rotation.x,
                    transform_msg.transform.rotation.y,
                    transform_msg.transform.rotation.z,
                    transform_msg.transform.rotation.w,
                )
                import math

                if hasattr(container, "lineEdit_roll"):
                    container.lineEdit_roll.setText(f"{math.degrees(roll):.2f}")
                    container.lineEdit_pitch.setText(f"{math.degrees(pitch):.2f}")
                    container.lineEdit_yaw.setText(f"{math.degrees(yaw):.2f}")

                logger.debug(f"Successfully updated TF display for {container_id}")
            else:
                display_text = "TF Transform\n\n"
                display_text += f"Parent Frame: {transform_msg.header.frame_id}\n"
                display_text += f"Child Frame: {transform_msg.child_frame_id}\n\n"
                display_text += f"Translation:\n"
                display_text += f"  x: {transform_msg.transform.translation.x:.4f}\n"
                display_text += f"  y: {transform_msg.transform.translation.y:.4f}\n"
                display_text += f"  z: {transform_msg.transform.translation.z:.4f}\n\n"
                display_text += f"Rotation (Quaternion):\n"
                display_text += f"  x: {transform_msg.transform.rotation.x:.4f}\n"
                display_text += f"  y: {transform_msg.transform.rotation.y:.4f}\n"
                display_text += f"  z: {transform_msg.transform.rotation.z:.4f}\n"
                display_text += f"  w: {transform_msg.transform.rotation.w:.4f}\n"

                if hasattr(container, "text_data_display"):
                    container.text_data_display.setText(display_text)
                elif hasattr(container, "label_data_info"):
                    short_text = f"TF: {transform_msg.header.frame_id} -> {transform_msg.child_frame_id}"
                    container.label_data_info.setText(short_text)

        except Exception as e:
            logger.error(
                f"Error updating TransformStamped display for container {getattr(container, 'container_id', 'unknown')}: {e}"
            )

    def _update_unknown_display(self, container, msg):
        """Update container with unknown/unsupported message types"""
        try:
            msg_type = type(msg).__name__

            display_text = f"Unknown Message Type\n\n"
            display_text += f"Type: {msg_type}\n\n"

            # Try to display some basic fields
            if hasattr(msg, "__slots__"):
                display_text += "Fields:\n"
                for slot in msg.__slots__[:10]:  # Limit to first 10 fields
                    try:
                        value = getattr(msg, slot, None)
                        if value is not None:
                            display_text += f"  {slot}: {value}\n"
                    except:
                        pass

            if hasattr(container, "text_data_display"):
                container.text_data_display.setText(display_text)
            elif hasattr(container, "label_data_info"):
                container.label_data_info.setText(f"{msg_type} data")

        except Exception as e:
            logger.error(f"Error updating unknown display: {e}")

    def _update_simple_data_display(self, container, msg):
        """Update container with simple data types"""
        try:
            msg_type = type(msg).__name__

            if hasattr(msg, "data"):
                value = msg.data
            else:
                value = str(msg)

            display_text = f"{msg_type}\n\nValue: {value}"

            if hasattr(container, "text_data_display"):
                container.text_data_display.setText(display_text)
            elif hasattr(container, "label_data_info"):
                container.label_data_info.setText(f"{msg_type}: {value}")

        except Exception as e:
            logger.error(f"Error updating simple data display: {e}")

    def _update_text_status(self, container, text):
        """Update container with text status - try multiple possible widgets"""
        try:
            # Try different possible text widgets
            text_widgets = [
                "label_status",
                "status_label",
                "statusLabel",
                "label_data_info",
                "data_info_label",
                "dataInfoLabel",
                "text_display",
                "textDisplay",
                "label_info",
                "info_label",
                "infoLabel",
            ]

            updated = False
            for widget_name in text_widgets:
                if hasattr(container, widget_name):
                    widget = getattr(container, widget_name)
                    if hasattr(widget, "setText"):
                        widget.setText(text)
                        updated = True
                        break

            if not updated:
                # Log available widgets for debugging
                widget_names = [
                    attr
                    for attr in dir(container)
                    if not attr.startswith("_")
                    and hasattr(getattr(container, attr, None), "setText")
                ]
                logger.debug(f"Available text widgets in container: {widget_names}")

        except Exception as e:
            logger.error(f"Error updating text status: {e}")

    def _update_text_info(self, container, text):
        """Update container with detailed text info"""
        try:
            # Try different possible info widgets
            info_widgets = [
                "label_data_info",
                "data_info_label",
                "dataInfoLabel",
                "text_data_display",
                "textDataDisplay",
                "data_display",
                "info_text",
                "infoText",
                "details_label",
            ]

            for widget_name in info_widgets:
                if hasattr(container, widget_name):
                    widget = getattr(container, widget_name)
                    if hasattr(widget, "setText"):
                        widget.setText(text)
                        return

        except Exception as e:
            logger.error(f"Error updating text info: {e}")

    def _update_no_data_display(self, container):
        """Update container when no data is available"""
        try:
            no_data_text = "No data received"

            if hasattr(container, "text_data_display"):
                container.text_data_display.setText(no_data_text)
            elif hasattr(container, "label_data_info"):
                container.label_data_info.setText(no_data_text)

            # Clear image if present
            if hasattr(container, "label_image"):
                container.label_image.clear()
            elif hasattr(container, "image_display"):
                container.image_display.clear()

        except Exception as e:
            logger.error(f"Error updating no data display: {e}")

    def _update_status_info(self, container, msg_type, message_count):
        """Update container status information"""
        try:
            status_text = f"Type: {msg_type}\nMessages: {message_count}"

            if hasattr(container, "label_status"):
                container.label_status.setText(f"Active ({message_count} msgs)")

            if hasattr(container, "label_message_type"):
                container.label_message_type.setText(msg_type)

            if hasattr(container, "label_message_count"):
                container.label_message_count.setText(str(message_count))

        except Exception as e:
            logger.error(f"Error updating status info: {e}")

    def stop_container_visualization(self, container):
        """Stop visualization for a container"""
        try:
            container_id = getattr(container, "container_id", None)
            if container_id:
                # Stop timer
                if container_id in self.visualization_timers:
                    timer = self.visualization_timers[container_id]
                    timer.stop()
                    del self.visualization_timers[container_id]

                # Stop media player
                if container_id in self.media_players:
                    media_player = self.media_players[container_id]
                    media_player.stop()
                    del self.media_players[container_id]

                # Clean up temp video files
                if container_id in self.temp_video_files:
                    temp_file = self.temp_video_files[container_id]
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    del self.temp_video_files[container_id]

                # Remove overlay label if exists
                if hasattr(container, "_video_overlay_label"):
                    container._video_overlay_label.deleteLater()
                    delattr(container, "_video_overlay_label")

                logger.info(f"Stopped visualization for container {container_id}")

        except Exception as e:
            logger.error(f"Error stopping container visualization: {e}")

    def stop_all_visualizations(self):
        """Stop all container visualizations"""
        try:
            # Stop all timers
            for container_id, timer in list(self.visualization_timers.items()):
                timer.stop()
            self.visualization_timers.clear()

            # Stop all media players
            for container_id, media_player in list(self.media_players.items()):
                media_player.stop()
            self.media_players.clear()

            # Clean up temp files
            for container_id, temp_file in list(self.temp_video_files.items()):
                try:
                    os.remove(temp_file)
                except:
                    pass
            self.temp_video_files.clear()

            logger.info("Stopped all container visualizations")

        except Exception as e:
            logger.error(f"Error stopping all visualizations: {e}")

    def set_update_rate(self, rate_hz):
        """Set the visualization update rate"""
        self.update_rate = max(1, min(rate_hz, 30))  # Limit between 1-30 Hz

        # Update existing timers
        for timer in self.visualization_timers.values():
            timer.setInterval(1000 // self.update_rate)

        logger.info(f"Set visualization update rate to {self.update_rate} Hz")
