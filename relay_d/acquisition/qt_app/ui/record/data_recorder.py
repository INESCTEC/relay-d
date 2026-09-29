import os
import h5py
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
import cv2
from relay_d.utils.coloring_logger import logger
from ...utils.stream_builder import StreamBuilder
from .data_normalization import DataNormalizer
import time
import json
from threading import Lock, Thread
from collections import defaultdict
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import String, Float32, Float64, Int32, Bool
from sensor_msgs_py import point_cloud2
from datetime import datetime
from relay_d.utils.ros_media_codec import decode_image, decode_organized_pointcloud

class DataRecorder(Node):
    def __init__(self, yaml_parser):
        super().__init__("data_recorder")
        self.yaml_parser = yaml_parser
        self.cv_bridge = CvBridge()
        self.data_normalizer = DataNormalizer()

        self.recorded_data = defaultdict(list)
        self.recording_metadata = {}
        self.current_demo_number = 0

        self.recording_frequency = 30

        # Update from yaml_parser if available
        if self.yaml_parser and self.yaml_parser.is_loaded():
            config = self.yaml_parser.yaml_data.get("config", {})
            self.recording_frequency = config.get("recording_frequency", 30)
        self.max_buffer_size = 1000000

        self.output_directory = None
        self.recorded_data_dir = None
        self.h5_file = None
        self.h5_file_path = None

        self.gripper_states = {}
        self.timestamps = []
        self.timestamp_spreads = []
        self._sync_tolerance = None
        self._sync_mode = "soft"

        self._stream_builder = StreamBuilder(self.yaml_parser)
        self._fill_gaps_containers = set()
        self._fill_defaults = {}
        self._fill_count_summary = {}
        self._modalities = {}

        self.topic_subscribers = None

        self.recording_thread = None
        self.stop_recording_flag = False
        self.recording_start_time = None
        self.recording_sample_count = 0
        self.decimation_counter = 0.0

        self.is_recording = False
        self.is_paused = False
        self.recording_lock = Lock()

        self._clock = self.get_clock()

        logger.info("DataRecorder initialized")

    def _stop_recording_due_to_error(self, reason):
        logger.error(f"Stopping recording due to error: {reason}")
        if self.is_recording:
            self.stop_recording()

    def pause_recording(self):
        try:
            with self.recording_lock:
                if not self.is_recording:
                    logger.warning("No recording in progress to pause")
                    return False

                if self.is_paused:
                    logger.warning("Recording is already paused")
                    return False

                self.is_paused = True
                logger.info("Recording paused")
                return True

        except Exception as e:
            logger.error(f"Failed to pause recording: {e}")
            return False

    def resume_recording(self):
        try:
            with self.recording_lock:
                if not self.is_recording:
                    logger.warning("No recording in progress to resume")
                    return False

                if not self.is_paused:
                    logger.warning("Recording is not paused")
                    return False

                self.is_paused = False
                logger.info("Recording resumed")
                return True

        except Exception as e:
            logger.error(f"Failed to resume recording: {e}")
            return False

    def start_recording(self, topic_subscribers, demo_name=None, flush_stale_data=False):
        try:
            with self.recording_lock:
                if self.is_recording:
                    logger.warning("Recording already in progress")
                    return False

                self.topic_subscribers = topic_subscribers

                if demo_name is None:
                    demo_name = f"demo_{self.current_demo_number}"

                self.recording_metadata = {
                    "demo_name": demo_name,
                    "start_time": time.time(),
                    "recording_frequency": self.recording_frequency,
                    "containers": {},
                    "yaml_config": self.yaml_parser.get_current_config_path()
                    if self.yaml_parser.is_loaded()
                    else None,
                }

                if self.yaml_parser.is_loaded():
                    metadata_dict = self.yaml_parser.get_metadata()
                    if metadata_dict:
                        for key, value in metadata_dict.items():
                            self.recording_metadata[key] = value
                        logger.info(
                            f"Loaded metadata from config: {list(metadata_dict.keys())}"
                        )
                    # Always re-read recording_frequency so it matches the loaded config,
                    # not the stale value from __init__ (which ran before config was loaded).
                    config = self.yaml_parser.yaml_data.get("config", {})
                    self.recording_frequency = config.get("recording_frequency", self.recording_frequency)
                    logger.info(f"Recording frequency: {self.recording_frequency} Hz")
                    self._sync_tolerance = config.get("sync_tolerance_sec", None)
                    self._sync_mode = config.get("sync_mode", "soft")
                    self.recording_metadata["sync_tolerance_sec"] = self._sync_tolerance
                    if self._sync_tolerance is not None:
                        logger.info(
                            f"Sync check enabled: tolerance={self._sync_tolerance}s, "
                            f"mode={self._sync_mode}"
                        )

                self.recorded_data.clear()
                self.timestamps.clear()
                self.timestamp_spreads = []
                self.gripper_states.clear()
                self._last_valid_raw = {}
                self._stream_builder = StreamBuilder(self.yaml_parser)
                self._fill_gaps_containers = self._build_fill_gaps_set()
                self._modalities = self._build_modality_lookup()
                if self._fill_gaps_containers:
                    logger.info(f"Gap-fill (hold) enabled for: {sorted(self._fill_gaps_containers)}")

                if self.yaml_parser.is_loaded():
                    config_data = self.yaml_parser.get_config_data()
                    for config_item in config_data:
                        if config_item["name"] == "gripper":
                            gripper_count = config_item["count"]
                            for i in range(gripper_count):
                                self.gripper_states[f"gripper_{i + 1}"] = []

                if flush_stale_data and hasattr(self.topic_subscribers, "flush_latest_data"):
                    self.topic_subscribers.flush_latest_data()
                    logger.info("Flushed stale data before recording start")

                self._create_h5_file(demo_name)

                return self._start_standard_recording(demo_name)

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def _start_standard_recording(self, demo_name):
        try:
            logger.info("Starting standard (non-synchronized) recording")

            self.is_recording = True
            self.stop_recording_flag = False

            self.recording_thread = Thread(
                target=self._standard_recording_loop,
                args=(self.topic_subscribers,),
                daemon=True,
            )
            self.recording_thread.start()

            logger.info(f"Started standard recording session: {demo_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start standard recording: {e}")
            return False

    def _standard_recording_loop(self, topic_subscribers):
        try:
            logger.info(
                f"Recording loop started at {self.recording_frequency} Hz — "
                "all inputs (topics + TF) sampled on the same tick"
            )

            # Warmup: spin TF lookups and wait for every input to produce at least
            # one sample before we start writing timestamps. This prevents leading
            # None entries caused by an empty TF buffer or topics that haven't
            # published yet.
            warmup_timeout = 2.0
            warmup_start = time.monotonic()
            while time.monotonic() - warmup_start < warmup_timeout:
                if hasattr(topic_subscribers, "update_tf_lookups"):
                    topic_subscribers.update_tf_lookups()
                snapshot = topic_subscribers.get_content()
                if snapshot and all(v is not None for v in snapshot.values()):
                    break
                time.sleep(0.05)
            missing = [
                k for k, v in (topic_subscribers.get_content() or {}).items()
                if v is None
            ]
            if missing:
                logger.warning(
                    f"Warmup finished; {len(missing)} input(s) still have no data "
                    f"and will use sample-and-hold until first message: {missing}"
                )
            else:
                logger.info("Warmup complete — all inputs have initial data")

            # Seed hold defaults for inputs with fill_gaps: true and a fill_default
            # value configured. Without seeding, leading Nones accumulate until the
            # first message arrives (which defeats gap-fill for the leading period).
            for _cid in list(topic_subscribers.subscribers.keys()):
                if _cid in self._fill_gaps_containers and _cid not in self._last_valid_raw:
                    _default = self._fill_defaults.get(_cid)
                    if _default is None:
                        continue  # no default configured — skip seeding
                    _status = (topic_subscribers.subscriber_status or {}).get(_cid, {})
                    _topic = _status.get("topic", "")
                    _cls = topic_subscribers.message_type_cache.get(_topic) if _topic else None
                    if _cls is not None:
                        _type_name = _cls.__name__.lower()
                        self._last_valid_raw[_cid] = {"type": _type_name, "data": _default}
                        logger.info(
                            f"Seeded hold default ({_default!r}) for '{_cid}' ({_type_name})"
                        )

            self.recording_start_time = self._clock.now()
            self.recording_sample_count = 0
            self.decimation_counter = 0.0

            interval = 1.0 / self.recording_frequency
            # Monotonic next-tick anchor — corrects for loop body processing time
            # so the actual sample rate stays close to recording_frequency.
            next_tick = time.monotonic()

            while not self.stop_recording_flag:
                try:
                    if self.is_paused:
                        time.sleep(0.01)
                        next_tick = time.monotonic()  # don't try to catch up after a pause
                        continue

                    # Flush any TF messages queued in DDS since the last external
                    # spin. Non-blocking: skip if the Qt timer or API spin thread
                    # already holds the lock.
                    if hasattr(topic_subscribers, "_spin_lock"):
                        if topic_subscribers._spin_lock.acquire(blocking=False):
                            try:
                                rclpy.spin_once(topic_subscribers, timeout_sec=0.0)
                            except Exception:
                                pass
                            finally:
                                topic_subscribers._spin_lock.release()

                    # Refresh all TF transforms before snapshotting data so that
                    # every input (topics and TF alike) reflects the same moment.
                    if hasattr(topic_subscribers, "update_tf_lookups"):
                        topic_subscribers.update_tf_lookups()

                    # Atomic snapshot of data AND timestamps — both dicts are read
                    # under a single lock so topic callbacks cannot update one after
                    # the other has been read, which would inflate the apparent spread.
                    all_data, snap_timestamps = (
                        topic_subscribers.get_content_and_timestamps()
                    )

                    # Re-look up TF containers at the header.stamp of the most recently-stamped
                    # topic message. This ensures TF and topic data share the same robot-clock
                    # timestamp without any additional subscriptions.
                    ref_ros_time = None
                    ref_nsec = 0
                    for msg in all_data.values():
                        if msg is not None and hasattr(msg, "header"):
                            s = msg.header.stamp
                            nsec = s.sec * 1_000_000_000 + s.nanosec
                            if nsec > ref_nsec:
                                ref_nsec = nsec
                                ref_ros_time = rclpy.time.Time(
                                    seconds=s.sec, nanoseconds=s.nanosec
                                )
                    if ref_ros_time is not None and topic_subscribers.tf_lookup_containers:
                        for tf_cid in list(topic_subscribers.tf_lookup_containers.keys()):
                            tf_msg = topic_subscribers.lookup_tf_at_time(
                                tf_cid, ref_ros_time, timeout_sec=0.0
                            )
                            if tf_msg is not None:
                                all_data[tf_cid] = tf_msg

                    # Refresh snapshot timestamps for TF containers to reflect the
                    # time-aligned data now in all_data. If lookup_tf_at_time succeeded,
                    # the new header.stamp ≈ ref_ros_time → spread ≈ 0 → no false warning.
                    # If it failed, all_data[tf_cid] kept the raw broadcaster timestamp
                    # → snap_timestamps stays stale → warning fires correctly.
                    for tf_cid in topic_subscribers.tf_lookup_containers:
                        tf_val = all_data.get(tf_cid)
                        if tf_val is not None and hasattr(tf_val, "header"):
                            s = tf_val.header.stamp
                            snap_timestamps[tf_cid] = s.sec + s.nanosec * 1e-9

                    # Timestamp sync check — compares ROS message timestamps across
                    # all inputs that are not fill_gaps. Soft mode logs a warning;
                    # strict mode skips the sample entirely.
                    spread = 0.0
                    if self._sync_tolerance is not None:
                        sync_cids = [
                            cid for cid in all_data
                            if cid not in self._fill_gaps_containers
                            and snap_timestamps.get(cid) is not None
                        ]
                        if len(sync_cids) >= 2:
                            ts_vals = [snap_timestamps[cid] for cid in sync_cids]
                            spread = max(ts_vals) - min(ts_vals)
                            if spread > self._sync_tolerance:
                                ref_ts = max(ts_vals)
                                sorted_cids = sorted(
                                    sync_cids, key=lambda c: snap_timestamps[c], reverse=True
                                )
                                lag_parts = "  ".join(
                                    f"{cid} ({snap_timestamps[cid] - ref_ts:+.3f}s)"
                                    for cid in sorted_cids
                                )
                                logger.warning(
                                    f"Timestamp spread {spread:.4f}s exceeds tolerance "
                                    f"{self._sync_tolerance}s\n  {lag_parts}"
                                )
                                if self._sync_mode == "strict":
                                    next_tick += interval
                                    sleep_time = next_tick - time.monotonic()
                                    if sleep_time > 0:
                                        time.sleep(sleep_time)
                                    else:
                                        next_tick = time.monotonic()
                                    continue
                    self.timestamp_spreads.append(spread)

                    current_time = self._clock.now().to_msg()
                    timestamp_nsec = (
                        current_time.sec * 1000000000 + current_time.nanosec
                    )
                    self.timestamps.append(timestamp_nsec)

                    # Streams share the same snapshot as all containers.
                    if self._stream_builder.has_streams():
                        self._stream_builder.evaluate(all_data)

                    data_processed = 0
                    all_container_ids = list(self.topic_subscribers.subscribers.keys())
                    for container_id in all_container_ids:
                        msg_data = all_data.get(container_id)
                        if msg_data is not None:
                            msg_data = self._stream_builder.apply_joint_filter(container_id, msg_data)
                            processed_data = self._process_message_for_recording(
                                msg_data, container_id
                            )
                            if processed_data:
                                self._last_valid_raw[container_id] = processed_data
                                self.recorded_data[container_id].append(processed_data)
                                data_processed += 1
                            elif container_id in self._last_valid_raw:
                                self.recorded_data[container_id].append(
                                    self._last_valid_raw[container_id]
                                )
                            else:
                                self.recorded_data[container_id].append(None)
                        elif container_id in self._last_valid_raw:
                            self.recorded_data[container_id].append(
                                self._last_valid_raw[container_id]
                            )
                        else:
                            self.recorded_data[container_id].append(None)

                    self.recording_sample_count += 1

                    if self.recording_sample_count % 100 == 0:
                        logger.info(
                            f"Recorded {self.recording_sample_count} samples, "
                            f"{data_processed} with actual data"
                        )

                    if self.recording_sample_count > self.max_buffer_size:
                        logger.warning(
                            f"Recording buffer full ({self.max_buffer_size} samples)."
                        )

                    # Drift-corrected sleep: subtract the time already spent in the
                    # loop body so the inter-sample interval stays at 1/recording_frequency.
                    next_tick += interval
                    sleep_time = next_tick - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # Loop body took longer than one interval; skip the sleep and
                        # re-anchor so we don't try to recover with zero-sleep bursts.
                        next_tick = time.monotonic()

                except Exception as e:
                    logger.error(f"Error in recording loop: {e}")

            logger.info(
                f"Standard recording loop finished. Total samples: {self.recording_sample_count}"
            )
            logger.info(
                f"Final recorded data: {dict((k, len(v)) for k, v in self.recorded_data.items())}"
            )

        except Exception as e:
            logger.error(f"Fatal error in standard recording loop: {e}")

    def stop_recording(self):
        try:
            with self.recording_lock:
                if not self.is_recording:
                    logger.warning("No recording in progress")
                    return False

                self.stop_recording_flag = True
                self.is_recording = False
                self.is_paused = False

                if self.recording_thread and self.recording_thread.is_alive():
                    self.recording_thread.join(timeout=2.0)

                expected = len(self.timestamps)
                for cid, samples in self.recorded_data.items():
                    if len(samples) != expected:
                        logger.warning(
                            f"Sample count mismatch: '{cid}' has {len(samples)} samples, "
                            f"expected {expected}. Gap-fill will be applied before saving."
                        )

                self.recording_metadata["end_time"] = time.time()
                self.recording_metadata["duration"] = (
                    self.recording_metadata["end_time"]
                    - self.recording_metadata["start_time"]
                )
                self.recording_metadata["total_samples"] = len(self.timestamps)

                self._save_recorded_data()

                if self._fill_count_summary:
                    total_filled = sum(self._fill_count_summary.values())
                    summary_lines = "\n".join(
                        f"  {cid}: {count} samples gap-filled"
                        for cid, count in sorted(self._fill_count_summary.items())
                    )
                    logger.info(
                        f"Gap-fill summary — {total_filled} total samples filled by hold:\n"
                        f"{summary_lines}"
                    )

                if self.h5_file:
                    self.h5_file.close()
                    self.h5_file = None

                logger.info(
                    f"Recording stopped. Saved {len(self.timestamps)} samples in {self.h5_file_path}"
                )
                logger.info(
                    f"Recording duration: {self.recording_metadata['duration']:.2f} seconds"
                )

                self.current_demo_number += 1
                return True

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False

    def _create_h5_file(self, demo_name):
        """Create H5 file for recording data, reusing session folder if already set."""
        try:
            if self.recorded_data_dir and os.path.exists(self.recorded_data_dir):
                output_dir = self.recorded_data_dir
            elif self.output_directory:
                timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = os.path.join(self.output_directory, timestamp_folder)
            else:
                timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = os.path.join(
                    os.path.expanduser("~"), "lfd_recordings", timestamp_folder
                )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.recorded_data_dir = output_dir
            self.h5_file_path = os.path.join(str(output_dir), f"{demo_name}.h5")
            self.h5_file = h5py.File(self.h5_file_path, "w", libver="latest")
            logger.info(f"Created H5 file: {self.h5_file_path}")
        except Exception as e:
            logger.error(f"Failed to create H5 file: {e}")
            raise

    def _process_message_for_recording(self, msg, container_id=None):
        try:
            msg_type = type(msg).__name__

            if isinstance(msg, (Image, CompressedImage)):
                return self._process_image_message(msg, container_id)
            elif isinstance(msg, PointCloud2):
                return self._process_pointcloud_message(msg, container_id)
            elif isinstance(msg, JointState) or (
                hasattr(msg, "name") and hasattr(msg, "position")
                and hasattr(msg, "velocity") and hasattr(msg, "effort")
                and hasattr(msg, "header")
            ):
                return self._process_joint_message(msg)
            elif isinstance(msg, PoseStamped):
                return self._process_pose_message(msg)
            elif isinstance(msg, Twist):
                return self._process_twist_message(msg)
            elif isinstance(msg, TransformStamped):
                return self._process_transform_stamped_message(msg)
            elif isinstance(msg, (String, Float32, Float64, Int32, Bool)):
                return self._process_simple_message(msg)
            else:
                return self._process_generic_message(msg)

        except Exception as e:
            logger.error(f"Error processing message for recording: {e}")
            return None

    def _process_image_message(self, img_msg, container_id=None):
        try:
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
                is_depth = self._modalities.get(container_id) == "depth"
                cv_image = decode_image(img_msg, self.cv_bridge, is_depth=is_depth)
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

        except Exception as e:
            logger.error(f"Error processing image message: {e}")
            return None

    def _process_pointcloud_message(self, pc_msg, container_id=None):
        try:
            field_names = [field.name for field in pc_msg.fields]
            wants_scan = self._modalities.get(container_id) == "scan"
            is_scan_raster = pc_msg.height > 1 and wants_scan

            if is_scan_raster:
                points = decode_organized_pointcloud(pc_msg)
                if points is None:
                    return None
            else:
                raw = list(
                    point_cloud2.read_points(
                        pc_msg, field_names=field_names, skip_nans=True
                    )
                )
                points = np.array(raw) if raw else np.array([])

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

        except Exception as e:
            logger.error(f"Error processing pointcloud message: {e}")
            return None

    def _process_joint_message(self, joint_msg):
        try:
            names = list(joint_msg.name) if joint_msg.name else []
            positions = list(joint_msg.position) if joint_msg.position else []
            velocities = list(joint_msg.velocity) if joint_msg.velocity else []
            efforts = list(joint_msg.effort) if joint_msg.effort else []

            positions = [float(p) for p in positions] if positions else []
            velocities = [float(v) for v in velocities] if velocities else []
            efforts = [float(e) for e in efforts] if efforts else []

            if self.data_normalizer.has_limits():
                try:
                    if positions:
                        pos_dict = dict(zip(names, positions))
                        normalized_pos = self.data_normalizer.normalize_joint_data(pos_dict, "position")
                        if normalized_pos is not None:
                            positions = [normalized_pos[name] for name in names]

                    if velocities:
                        vel_dict = dict(zip(names, velocities))
                        normalized_vel = self.data_normalizer.normalize_joint_data(vel_dict, "velocity")
                        if normalized_vel is not None:
                            velocities = [normalized_vel[name] for name in names]

                    if efforts:
                        eff_dict = dict(zip(names, efforts))
                        normalized_eff = self.data_normalizer.normalize_joint_data(eff_dict, "effort")
                        if normalized_eff is not None:
                            efforts = [normalized_eff[name] for name in names]

                except Exception as e:
                    logger.error(f"Normalization failed: {e}")
                    if self.is_recording:
                        self._stop_recording_due_to_error(f"Normalization error: {e}")
                    return None

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

        except Exception as e:
            logger.error(f"Error processing joint message: {e}")
            return None

    def _process_pose_message(self, pose_msg):
        try:
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

        except Exception as e:
            logger.error(f"Error processing pose message: {e}")
            return None

    def _process_twist_message(self, twist_msg):
        try:
            return {
                "type": "twist",
                "linear": [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z],
                "angular": [
                    twist_msg.angular.x,
                    twist_msg.angular.y,
                    twist_msg.angular.z,
                ],
            }

        except Exception as e:
            logger.error(f"Error processing twist message: {e}")
            return None

    def _process_transform_stamped_message(self, transform_msg):
        try:
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

        except Exception as e:
            logger.error(f"Error processing TransformStamped message: {e}")
            return None

    def _process_simple_message(self, msg):
        try:
            return {
                "type": type(msg).__name__.lower(),
                "data": msg.data if hasattr(msg, "data") else str(msg),
            }

        except Exception as e:
            logger.error(f"Error processing simple message: {e}")
            return None

    def _process_generic_message(self, msg):
        try:
            msg_data = {"type": type(msg).__name__, "fields": {}}

            if hasattr(msg, "__slots__"):
                for slot in msg.__slots__:
                    try:
                        value = getattr(msg, slot)
                        if isinstance(value, (int, float, str, bool)):
                            msg_data["fields"][slot] = value
                        else:
                            msg_data["fields"][slot] = str(value)
                    except:
                        msg_data["fields"][slot] = "Error reading field"

            return msg_data

        except Exception as e:
            logger.error(f"Error processing generic message: {e}")
            return None

    def _build_output_map_lookup(self) -> dict:
        """
        Build container_id -> output_map lookup from the yaml config input section.
        If output_map is not set for a container, falls back to the container_id itself.
        """
        lookup = {}
        if self.yaml_parser and self.yaml_parser.is_loaded():
            input_section = (
                self.yaml_parser.yaml_data.get("config", {}).get("input", {})
            )
            for container_id, spec in input_section.items():
                output_map = spec.get("output_map")
                lookup[container_id] = output_map if output_map else container_id
        return lookup

    def _build_modality_lookup(self) -> dict:
        """
        Build container_id -> "rgb"|"depth"|"scan" from the yaml config input
        section's `modality` field. Camera inputs without an explicit
        `modality` default to "rgb" (backward compatible).
        """
        lookup = {}
        if self.yaml_parser and self.yaml_parser.is_loaded():
            input_section = self.yaml_parser.yaml_data.get("config", {}).get("input", {})
            for container_id, spec in input_section.items():
                if isinstance(spec, dict):
                    lookup[container_id] = spec.get("modality", "rgb")
        return lookup

    def _build_fill_gaps_set(self) -> set:
        """Return the set of container IDs that have fill_gaps: true in the config.
        Also populates self._fill_defaults with any configured fill_default values.
        """
        fill_set = set()
        self._fill_defaults = {}
        if self.yaml_parser and self.yaml_parser.is_loaded():
            input_section = self.yaml_parser.yaml_data.get("config", {}).get("input", {})
            for container_id, spec in input_section.items():
                if isinstance(spec, dict) and spec.get("fill_gaps", False):
                    fill_set.add(container_id)
                    default_val = spec.get("fill_default", None)
                    if default_val is not None:
                        self._fill_defaults[container_id] = default_val
        return fill_set

    def _forward_fill(self, data_list):
        """Hold-last-value forward-fill: replace None with the most recent valid entry.

        Leading Nones that have no prior value are left as-is (handled upstream by
        seeding _last_valid_raw before the recording loop starts).
        Returns the original list unchanged when all entries are None.
        """
        if not any(item is not None for item in data_list):
            return data_list
        filled = list(data_list)
        last_valid = None
        for i, item in enumerate(filled):
            if item is not None:
                last_valid = item
            elif last_valid is not None:
                filled[i] = last_valid
        return filled

    def _save_recorded_data(self):
        try:
            if not self.h5_file:
                logger.error("No H5 file available for saving")
                return False

            demo_name = self.recording_metadata["demo_name"]
            demo_group = self.h5_file.create_group(f"data/{demo_name}")

            output_map_lookup = self._build_output_map_lookup()
            self._fill_count_summary = {}

            if self.timestamps:
                demo_group.create_dataset("timestamps", data=np.array(self.timestamps))

            if self.timestamp_spreads:
                demo_group.create_dataset(
                    "timestamp_spreads",
                    data=np.array(self.timestamp_spreads, dtype=np.float32),
                )

            new_streams = self._stream_builder.finalize()
            if new_streams:
                streams_grp = demo_group.create_group("streams")
                for stream_name, arr in new_streams.items():
                    streams_grp.create_dataset(
                        stream_name, data=arr, compression="gzip", compression_opts=4
                    )
                    logger.info(f"Saved stream '{stream_name}': shape {arr.shape}")

            for container_id, data_list in self.recorded_data.items():
                if not data_list:
                    continue

                non_null_raw = sum(1 for d in data_list if d is not None)

                if container_id in self._fill_gaps_containers:
                    filled_list = self._forward_fill(data_list)
                    fill_count = sum(
                        1 for a, b in zip(data_list, filled_list)
                        if a is None and b is not None
                    )
                    if fill_count > 0:
                        self._fill_count_summary[container_id] = fill_count
                else:
                    filled_list = data_list
                    fill_count = 0

                fill_suffix = f", {fill_count} gap-filled by hold" if fill_count > 0 else ""
                logger.info(
                    f"{container_id} — {len(data_list)} samples "
                    f"({non_null_raw} real{fill_suffix})"
                )

                # Use output_map name so the raw HDF5 matches what data_config.yaml expects
                group_name = output_map_lookup.get(container_id, container_id)
                if group_name != container_id:
                    logger.info(f"Saving '{container_id}' as '{group_name}' (output_map)")
                container_group = demo_group.create_group(group_name)

                data_by_type = defaultdict(list)
                for data_item in filled_list:
                    if data_item is not None:
                        data_type = data_item.get("type", "unknown")
                        data_by_type[data_type].append(data_item)

                for data_type, type_data_list in data_by_type.items():
                    self._save_data_type_to_h5_safe(
                        container_group, data_type, type_data_list
                    )

            if self.gripper_states:
                gripper_group = demo_group.create_group("gripper_states")
                for gripper_name, states in self.gripper_states.items():
                    if states:
                        if isinstance(states[0], dict):
                            state_values = [s["state"] for s in states]
                            timestamps = [s["timestamp"] for s in states]
                            gripper_group.create_dataset(
                                f"{gripper_name}_states", data=np.array(state_values)
                            )
                            gripper_group.create_dataset(
                                f"{gripper_name}_timestamps", data=np.array(timestamps)
                            )
                        else:
                            gripper_group.create_dataset(
                                gripper_name, data=np.array(states)
                            )

            self.recording_metadata["fill_count_summary"] = dict(self._fill_count_summary)
            self._save_metadata_to_h5(demo_group)

            logger.info(f"Saved recording data to {self.h5_file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving recorded data: {e}")
            return False

    def _save_data_type_to_h5_safe(self, group, data_type, data_list):
        try:
            if data_type == "image":
                images = []
                for item in data_list:
                    if "data" in item and item["data"] is not None:
                        img_data = item["data"]
                        if isinstance(img_data, np.ndarray):
                            images.append(img_data)

                if images:
                    first_shape = images[0].shape
                    first_dtype = images[0].dtype

                    consistent_shape = all(
                        img.shape == first_shape and img.dtype == first_dtype
                        for img in images
                    )

                    if consistent_shape:
                        images_array = np.stack(images)
                        group.create_dataset(
                            "images", data=images_array, compression="gzip"
                        )
                    else:
                        logger.warning(
                            "Images have inconsistent shapes, saving individually"
                        )
                        for i, img in enumerate(images):
                            group.create_dataset(
                                f"image_{i}", data=img, compression="gzip"
                            )

                if data_list:
                    first_item = data_list[0]
                    group.attrs["image_width"] = first_item.get("width", 0)
                    group.attrs["image_height"] = first_item.get("height", 0)
                    group.attrs["image_encoding"] = first_item.get(
                        "encoding", "unknown"
                    )

            elif data_type == "compressed_image":
                group.attrs["data_type"] = "compressed_image"
                group.attrs["format"] = (
                    data_list[0].get("format", "unknown") if data_list else "unknown"
                )

                compressed_data = []
                for item in data_list:
                    if "data" in item and item["data"] is not None:
                        compressed_data.append(item["data"])

                if compressed_data:
                    dt = h5py.vlen_dtype(np.dtype("uint8"))
                    group.create_dataset(
                        "compressed_data", data=compressed_data, dtype=dt
                    )

            elif data_type == "joint_state":
                positions = []
                velocities = []
                efforts = []

                for item in data_list:
                    if "positions" in item:
                        pos = item["positions"]
                        if isinstance(pos, (list, tuple)) and len(pos) > 0:
                            positions.append([float(p) for p in pos])
                        elif pos is None or len(pos) == 0:
                            positions.append([])

                    if "velocities" in item:
                        vel = item["velocities"]
                        if isinstance(vel, (list, tuple)) and len(vel) > 0:
                            velocities.append([float(v) for v in vel])
                        elif vel is None or len(vel) == 0:
                            velocities.append([])

                    if "efforts" in item:
                        eff = item["efforts"]
                        if isinstance(eff, (list, tuple)) and len(eff) > 0:
                            efforts.append([float(e) for e in eff])
                        elif eff is None or len(eff) == 0:
                            efforts.append([])

                # Check if we have any valid data
                has_positions = (
                    any(len(p) > 0 for p in positions) if positions else False
                )
                has_velocities = (
                    any(len(v) > 0 for v in velocities) if velocities else False
                )
                has_efforts = any(len(e) > 0 for e in efforts) if efforts else False

                if has_positions:
                    # Check if all positions have the same length
                    non_empty_lengths = [len(p) for p in positions if len(p) > 0]
                    if non_empty_lengths and len(set(non_empty_lengths)) == 1:
                        # All have same length - create array
                        if positions:
                            # Replace empty lists with zeros for consistent shape
                            first_len = len(positions[0]) if positions[0] else 1
                            consistent_positions = [
                                p if len(p) > 0 else [0.0] * first_len
                                for p in positions
                            ]
                            group.create_dataset(
                                "joint_positions", data=np.array(consistent_positions)
                            )
                        else:
                            group.create_dataset("joint_positions", data=np.array([[]]))
                    else:
                        logger.warning(
                            "Joint positions have inconsistent lengths, saving as variable-length"
                        )
                        # Save as variable-length dataset using h5py vlen dtype
                        dt = h5py.vlen_dtype(np.float64)
                        # Filter out empty positions
                        valid_positions = [
                            np.array(p, dtype=np.float64) for p in positions if p
                        ]
                        if valid_positions:
                            group.create_dataset(
                                "joint_positions", data=valid_positions, dtype=dt
                            )

                if has_velocities:
                    non_empty_lengths = [len(v) for v in velocities if len(v) > 0]
                    if non_empty_lengths and len(set(non_empty_lengths)) == 1:
                        if velocities:
                            first_len = len(velocities[0]) if velocities[0] else 1
                            consistent_velocities = [
                                v if len(v) > 0 else [0.0] * first_len
                                for v in velocities
                            ]
                            group.create_dataset(
                                "joint_velocities", data=np.array(consistent_velocities)
                            )
                        else:
                            group.create_dataset(
                                "joint_velocities", data=np.array([[]])
                            )
                    else:
                        logger.warning(
                            "Joint velocities have inconsistent lengths, saving as variable-length"
                        )
                        dt = h5py.vlen_dtype(np.float64)
                        valid_velocities = [
                            np.array(v, dtype=np.float64) for v in velocities if v
                        ]
                        if valid_velocities:
                            group.create_dataset(
                                "joint_velocities", data=valid_velocities, dtype=dt
                            )

                if has_efforts:
                    non_empty_lengths = [len(e) for e in efforts if len(e) > 0]
                    if non_empty_lengths and len(set(non_empty_lengths)) == 1:
                        if efforts:
                            first_len = len(efforts[0]) if efforts[0] else 1
                            consistent_efforts = [
                                e if len(e) > 0 else [0.0] * first_len for e in efforts
                            ]
                            group.create_dataset(
                                "joint_efforts", data=np.array(consistent_efforts)
                            )
                        else:
                            group.create_dataset("joint_efforts", data=np.array([[]]))
                    else:
                        logger.warning(
                            "Joint efforts have inconsistent lengths, saving as variable-length"
                        )
                        dt = h5py.vlen_dtype(np.float64)
                        valid_efforts = [
                            np.array(e, dtype=np.float64) for e in efforts if e
                        ]
                        if valid_efforts:
                            group.create_dataset(
                                "joint_efforts", data=valid_efforts, dtype=dt
                            )

                if data_list and "names" in data_list[0]:
                    joint_names = data_list[0]["names"]
                    group.attrs["joint_names"] = json.dumps(joint_names)

            elif data_type == "pose":
                positions = [
                    item["position"] for item in data_list if "position" in item
                ]
                orientations = [
                    item["orientation"] for item in data_list if "orientation" in item
                ]

                if positions:
                    group.create_dataset("positions", data=np.array(positions))
                if orientations:
                    group.create_dataset("orientations", data=np.array(orientations))

            elif data_type == "twist":
                linear = [item["linear"] for item in data_list if "linear" in item]
                angular = [item["angular"] for item in data_list if "angular" in item]

                if linear:
                    group.create_dataset("linear_velocities", data=np.array(linear))
                if angular:
                    group.create_dataset("angular_velocities", data=np.array(angular))

            elif data_type == "pointcloud":
                group.attrs["data_type"] = "pointcloud"
                if data_list:
                    first_item = data_list[0]
                    group.attrs["width"] = first_item.get("width", 0)
                    group.attrs["height"] = first_item.get("height", 0)
                    group.attrs["fields"] = json.dumps(first_item.get("fields", []))
                    group.attrs["organized"] = bool(first_item.get("organized", False))

                    if first_item.get("organized"):
                        frames = [
                            item["points"] for item in data_list
                            if isinstance(item.get("points"), np.ndarray)
                            and item["points"].ndim == 3
                        ]
                        consistent = (
                            bool(frames)
                            and all(f.shape == frames[0].shape for f in frames)
                        )
                        if consistent:
                            group.create_dataset(
                                "points", data=np.stack(frames), compression="gzip"
                            )
                        else:
                            logger.warning(
                                "Organized point cloud frames have inconsistent "
                                "shapes; per-frame points not saved."
                            )
                    else:
                        logger.warning(
                            "Point cloud is not an organized 'scan' raster; "
                            "Robomimic training data requires a fixed shape, so "
                            "per-frame points are not saved (metadata attrs only)."
                        )

            elif data_type == "transform_stamped":
                translations = [
                    item["translation"] for item in data_list if "translation" in item
                ]
                rotations = [
                    item["rotation"] for item in data_list if "rotation" in item
                ]
                parent_frames = [
                    item["parent_frame"] for item in data_list if "parent_frame" in item
                ]
                child_frames = [
                    item["child_frame"] for item in data_list if "child_frame" in item
                ]

                if translations:
                    group.create_dataset("translations", data=np.array(translations))
                if rotations:
                    group.create_dataset("rotations", data=np.array(rotations))
                if parent_frames:
                    group.attrs["parent_frames"] = json.dumps(parent_frames)
                if child_frames:
                    group.attrs["child_frames"] = json.dumps(child_frames)

            else:
                json_data = []
                for item in data_list:
                    try:
                        json_str = json.dumps(item)
                        json_data.append(json_str)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Could not serialize item to JSON: {e}")
                        json_data.append(
                            json.dumps(
                                {
                                    "error": "serialization_failed",
                                    "type": str(type(item)),
                                }
                            )
                        )

                if json_data:
                    dt = h5py.string_dtype(encoding="utf-8")
                    group.create_dataset("data", data=json_data, dtype=dt)

        except Exception as e:
            logger.error(f"Error saving data type {data_type}: {e}")
            try:
                json_data = [
                    json.dumps({"error": "save_failed", "original_error": str(e)})
                ]
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset("error_data", data=json_data, dtype=dt)
            except:
                logger.error(f"Failed to save error data for {data_type}")

    def _save_metadata_to_h5(self, group):
        try:
            metadata_group = group.create_group("metadata")

            for key, value in self.recording_metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = json.dumps(value) if value else ""

            logger.info("Saved metadata to H5 file")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def get_recording_status(self):
        return {
            "is_recording": self.is_recording,
            "is_paused": self.is_paused,
            "samples_recorded": len(self.timestamps),
            "recording_duration": time.time()
            - self.recording_metadata.get("start_time", time.time())
            if self.is_recording
            else 0,
            "output_directory": self.output_directory,
            "current_demo": self.current_demo_number,
            "active_inputs": len(self.topic_subscribers.subscribers)
            if self.topic_subscribers
            else 0,
        }

    def cleanup(self):
        try:
            if self.is_recording:
                self.stop_recording()

            if self.h5_file:
                self.h5_file.close()
                self.h5_file = None

            logger.info("DataRecorder cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
