import os
import h5py
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from ...utils.stream_builder import StreamBuilder
from relay_d.utils.numerics import inputs_in_sync, stamp_to_sec
from ..config.builder.config_check import RECORDING_CHANGING, removed_keys
import time
import json
from threading import Lock, Thread
from collections import defaultdict
from datetime import datetime
from .message_encoding import encode_message

def _sample_nbytes(sample: dict) -> int:
    """RAM held by one recorded sample: every array in it (image `data`, point cloud `points`, ...)."""
    return sum(v.nbytes for v in sample.values() if isinstance(v, np.ndarray))


def start_demo(recorder, topic_subscribers, containers, dataset_name, demo_name=None):
    """The per-Start steps the GUI and AppAPI share: subscribe every input again (re-detects the message type and QoS,
    resets custom TF buffers; a broken custom TF topic raises TfTopicSubscriptionError), then start the recorder (it
    flushes the data from before Start and first retries an unsaved demo, see ``recorder.retried_file``).
    Returns the demo name, or None if the recording did not start. An input that could not be subscribed only logs:
    the recorder holds it empty until data arrives (as the GUI always did)."""
    results = topic_subscribers.create_subscribers_for_containers(containers)
    logger.info(f"Subscribers created: {sum(results.values())}/{len(results)}")
    demo_name = demo_name or f"{dataset_name}_{time.time_ns() // 1_000_000}"  # ms: two Starts in one second must not clash ("w-")
    return demo_name if recorder.start_recording(topic_subscribers, demo_name) else None


def _write_stacked(group, name: str, frames: list):
    """Write equally shaped arrays as one (N, ...) dataset, frame by frame.

    np.stack would build a second full copy first (a full 4 GB buffer would need 8 GB at Stop).
    With compression each frame is its own chunk, so writing a frame never recompresses its neighbours.
    """
    first = frames[0]
    options = user_settings.h5_compression()
    dataset = group.create_dataset(
        name, shape=(len(frames), *first.shape), dtype=first.dtype,
        chunks=(1, *first.shape) if options else None, **options,
    )
    for i, frame in enumerate(frames):
        dataset[i] = frame
    return dataset


class DataRecorder(Node):
    def __init__(self, yaml_parser):
        super().__init__("data_recorder")
        self.yaml_parser = yaml_parser

        self.recorded_data = defaultdict(list)
        self.recording_metadata = {}
        self.current_demo_number = 0
        self.retried_file = None  # path of the demo saved by the last retry_save()

        self.recording_frequency = None  # read from the config at each start_recording
        self.max_buffer_bytes = int(user_settings.get("rec_default_max_buffer_mb")) * 1024**2
        self._sync_tolerance = None
        self._sync_mode = "soft"
        # shortcut: samples live in RAM until Stop, so a crash loses the demo being recorded; upgrade to a journal or
        # chunked HDF5 writes if demos get long enough for that to hurt. Past this many bytes of frames the loop stops
        # sampling (Stop still saves what is buffered) instead of running the PC out of memory.
        self._buffered_bytes = 0
        self._unsaved = False  # a save failed and the samples are still only in memory
        self.buffer_full = False  # the loop stopped sampling because max_buffer_bytes was reached
        self.loop_failed = False  # the loop stopped sampling because of repeated errors

        self.output_directory = None
        self.recorded_data_dir = None
        self.h5_file = None
        self.h5_file_path = None

        self.timestamps = []
        self.timestamp_spreads = []

        self._stream_builder = StreamBuilder(self.yaml_parser)
        self._fill_gaps_containers = set()
        self._fill_count_summary = {}
        self._held_counts = defaultdict(int)  # per input: ticks recorded without a fresh sample
        self._modalities = {}

        self.topic_subscribers = None

        self.recording_thread = None
        self.stop_recording_flag = False
        self.recording_sample_count = 0

        self.is_recording = False
        self.recording_lock = Lock()

        self._clock = self.get_clock()

        logger.info("DataRecorder initialized")

    def start_recording(self, topic_subscribers, demo_name=None):
        try:
            with self.recording_lock:
                if self.is_recording:
                    logger.warning("Recording already in progress")
                    return False

                self.retried_file = None
                if self._unsaved:
                    logger.warning("Previous recording was not saved; retrying the save before starting")
                    if not self.retry_save():
                        logger.error(
                            "Cannot start: the previous recording is still unsaved (fix the disk "
                            "problem and try again). Starting would erase it."
                        )
                        return False

                self.topic_subscribers = topic_subscribers
                self.recording_frequency = None  # only the loaded config sets it: no config, no recording

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
                    config = self.yaml_parser.yaml_data.get("config", {})
                    blocking = [msg for key, msg in removed_keys(config) if key in RECORDING_CHANGING]
                    if blocking:
                        logger.error("Cannot start: " + " ".join(blocking))
                        return False
                    self.recording_frequency = config.get("recording_frequency")
                    self.max_buffer_bytes = int(
                        float(config.get("max_buffer_mb") or user_settings.get("rec_default_max_buffer_mb")) * 1024**2
                    )
                    self.recording_metadata["recording_frequency"] = self.recording_frequency
                    logger.info(f"Recording frequency: {self.recording_frequency} Hz")
                    self._sync_tolerance = config.get("sync_tolerance_sec") or None  # missing or 0 = off
                    self._sync_mode = config.get("sync_mode", "soft")
                    self.recording_metadata["sync_tolerance_sec"] = self._sync_tolerance
                    if self._sync_tolerance is not None:
                        logger.info(
                            f"Sync check enabled: tolerance={self._sync_tolerance}s, "
                            f"mode={self._sync_mode}"
                        )

                if not self.recording_frequency or self.recording_frequency <= 0:
                    logger.error(f"Cannot start: recording_frequency must be > 0 Hz, got {self.recording_frequency!r}")
                    return False

                self.recorded_data.clear()
                self._held_counts.clear()
                self._buffered_bytes = 0
                self.buffer_full = False
                self.loop_failed = False
                self.timestamps.clear()
                self.timestamp_spreads = []
                self._last_valid_raw = {}
                self._stream_builder = StreamBuilder(self.yaml_parser)
                self._fill_gaps_containers = self._build_fill_gaps_set()
                self._modalities = self._build_modality_lookup()
                if self._fill_gaps_containers:
                    logger.info(f"Gap-fill (hold) enabled for: {sorted(self._fill_gaps_containers)}")

                self.topic_subscribers.flush_latest_data()  # samples from before Start are not part of the demo
                logger.info("Flushed stale data before recording start")

                self._create_h5_file(demo_name)

                self.is_recording = True
                self.stop_recording_flag = False
                self.recording_thread = Thread(
                    target=self._standard_recording_loop,
                    args=(self.topic_subscribers,),
                    daemon=True,
                )
                self.recording_thread.start()
                logger.info(f"Started recording session: {demo_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
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
            warmup_timeout = float(user_settings.get("rec_warmup_s"))
            warmup_start = time.monotonic()
            while time.monotonic() - warmup_start < warmup_timeout and not self.stop_recording_flag:  # Stop must not wait out a long warm-up
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

            self.recording_sample_count = 0

            interval = 1.0 / self.recording_frequency
            # Monotonic next-tick anchor — corrects for loop body processing time
            # so the actual sample rate stays close to recording_frequency.
            next_tick = time.monotonic()

            consecutive_errors = 0
            while not self.stop_recording_flag:
                try:
                    # No spin_once here: TopicSubscribers' own spin thread drains the callbacks, and spinning the
                    # node on the global executor would take it out of that thread's executor.
                    # Refresh all TF transforms before snapshotting data so that
                    # every input (topics and TF alike) reflects the same moment.
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
                            snap_timestamps[tf_cid] = stamp_to_sec(tf_val.header.stamp)

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
                            if not inputs_in_sync(sync_cids, snap_timestamps, self._sync_tolerance):
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
                    current_time = self._clock.now().to_msg()
                    timestamp_nsec = (
                        current_time.sec * 1000000000 + current_time.nanosec
                    )

                    # Encode the whole tick first, append it only once all of it worked: a tick that
                    # fails half way would leave some inputs one sample behind the timestamps for good.
                    tick = {}
                    for container_id in self.recorded_data.keys() | self.topic_subscribers.subscribers.keys():
                        msg_data = all_data.get(container_id)
                        processed_data = None
                        if msg_data is not None:
                            msg_data = self._stream_builder.apply_joint_filter(container_id, msg_data)
                            processed_data = self._process_message_for_recording(
                                msg_data, container_id
                            )
                        tick[container_id] = processed_data

                    # Streams share the same snapshot as all containers.
                    if self._stream_builder.has_streams():
                        self._stream_builder.evaluate(all_data)

                    self.timestamp_spreads.append(spread)
                    self.timestamps.append(timestamp_nsec)
                    data_processed = 0
                    for container_id, processed_data in tick.items():
                        if processed_data:
                            self._last_valid_raw[container_id] = processed_data
                            self._buffered_bytes += _sample_nbytes(processed_data)
                            data_processed += 1
                        else:
                            self._held_counts[container_id] += 1  # held (or None until the first sample)
                        self.recorded_data[container_id].append(
                            processed_data or self._last_valid_raw.get(container_id)
                        )

                    self.recording_sample_count += 1
                    consecutive_errors = 0

                    if self.recording_sample_count % 100 == 0:
                        logger.info(
                            f"Recorded {self.recording_sample_count} samples, "
                            f"{data_processed} with actual data"
                        )

                    if self._buffered_bytes > self.max_buffer_bytes:
                        self.buffer_full = True
                        logger.error(
                            f"Recording buffer full ({self._buffered_bytes / 1024**3:.1f} GB): "
                            f"sampling stopped at {self.recording_sample_count} samples. "
                            "Press Stop to save what was recorded."
                        )
                        break

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
                    # Don't retry at full speed: wait one tick, and give up if it keeps failing.
                    consecutive_errors += 1
                    if consecutive_errors >= 50:
                        logger.error("Recording loop failed 50 times in a row; sampling stopped.")
                        self.loop_failed = True
                        break
                    time.sleep(interval)

            logger.info(
                f"Standard recording loop finished. Total samples: {self.recording_sample_count}"
            )
            logger.info(
                f"Final recorded data: {dict((k, len(v)) for k, v in self.recorded_data.items())}"
            )

        except Exception as e:
            self.loop_failed = True
            logger.error(f"Fatal error in standard recording loop: {e}")

    def stop_recording(self):
        """Stop the loop, save the buffered samples to the H5 file and close it.

        Returns True only if the data was saved. If saving fails, returns False and
        keeps the samples in memory (``recorded_data``) so nothing is lost silently.
        """
        try:
            with self.recording_lock:
                if not self.is_recording:
                    logger.warning("No recording in progress")
                    return False

                self.stop_recording_flag = True
                self.is_recording = False

                if self.recording_thread and self.recording_thread.is_alive():
                    # The loop checks the stop flag every tick; wait long enough that it is really done
                    # before saving (saving while it still appends would corrupt the samples).
                    self.recording_thread.join(timeout=30.0)
                    if self.recording_thread.is_alive():
                        logger.error("Recording loop did not stop within 30 s; saving what is buffered")

                expected = len(self.timestamps)
                for cid, samples in list(self.recorded_data.items()):
                    if len(samples) != expected:
                        logger.error(
                            f"Sample count mismatch: '{cid}' has {len(samples)} samples, expected {expected}."
                        )

                self.recording_metadata["end_time"] = time.time()
                self.recording_metadata["duration"] = (
                    self.recording_metadata["end_time"]
                    - self.recording_metadata["start_time"]
                )
                self.recording_metadata["total_samples"] = len(self.timestamps)

                self._unsaved = True  # until the file is saved AND closed: a failed close must not lose the demo at the next Start
                saved = self._save_recorded_data()

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
                    try:
                        self.h5_file.close()
                    finally:
                        self.h5_file = None

                if not saved:
                    logger.error(f"Recording NOT saved to {self.h5_file_path}; data is still in memory")
                    return False
                self._unsaved = False

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

    def retry_save(self):
        """Save the samples left in memory by a failed stop_recording(). True on success."""
        try:
            self._create_h5_file(self.recording_metadata["demo_name"], mode="w")  # replaces its own partial file
            saved = self._save_recorded_data()
            self.h5_file.close()  # saved only once closed
        except Exception as e:
            logger.error(f"Retry save failed: {e}")
            saved = False
            try:
                if self.h5_file:
                    self.h5_file.close()
            except Exception:
                pass
        self.h5_file = None
        if saved:
            self._unsaved = False
            self.retried_file = self.h5_file_path  # the caller adds it to its session list
            self.current_demo_number += 1
        return saved

    def _recordings_base(self):
        """Folder the sessions go into, read when a demo starts: an explicit ``output_directory`` (AppAPI), else the setting."""
        return self.output_directory or user_settings.get("hdf5_save_folder") or os.path.join(os.path.expanduser("~"), "lfd_recordings")

    @staticmethod
    def _session_folder_name():
        """Folder name of a recording session: the ``session_folder_format`` setting (strftime pattern) + ``_Recording``."""
        name = datetime.now().strftime(user_settings.get("session_folder_format") or "%Y-%m-%d_%H-%M-%S")
        return (name.replace(os.sep, "_") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "_Recording"

    def _create_h5_file(self, demo_name, mode="w-"):
        """Create the H5 file of a demo in the current session folder. A session folder is reused only while it is still
        inside the recordings folder: after a change in Settings the next demo opens a new session there."""
        try:
            base = self._recordings_base()
            same_base = self.recorded_data_dir and os.path.normpath(os.path.dirname(self.recorded_data_dir)) == os.path.normpath(base)
            if same_base and os.path.exists(self.recorded_data_dir):
                output_dir = self.recorded_data_dir
            else:
                output_dir = os.path.join(base, self._session_folder_name())
                logger.info(f"Recording session folder: {output_dir} (recordings folder: {base})")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.recorded_data_dir = output_dir
            self.h5_file_path = os.path.join(str(output_dir), f"{demo_name}.h5")
            # "w-": never overwrite a demo already there (the demo counter restarts in every process)
            self.h5_file = h5py.File(self.h5_file_path, mode, libver="latest")
            logger.info(f"Created H5 file: {self.h5_file_path}")
        except Exception as e:
            logger.error(f"Failed to create H5 file: {e}")
            raise

    def _process_message_for_recording(self, msg, container_id=None):
        try:
            return encode_message(msg, self._modalities.get(container_id))
        except Exception as e:
            logger.error(f"Error processing {type(msg).__name__} for recording ({container_id}): {e}")
            return None

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
        """Return the set of container IDs that have fill_gaps: true in the config."""
        fill_set = set()
        if self.yaml_parser and self.yaml_parser.is_loaded():
            input_section = self.yaml_parser.yaml_data.get("config", {}).get("input", {})
            for container_id, spec in input_section.items():
                if isinstance(spec, dict) and spec.get("fill_gaps", False):
                    fill_set.add(container_id)
        return fill_set

    def _save_recorded_data(self):
        try:
            if not self.h5_file:
                logger.error("No H5 file available for saving")
                return False

            demo_name = self.recording_metadata["demo_name"]
            demo_group = self.h5_file.create_group(f"data/{demo_name}")

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
                        stream_name, data=arr, **user_settings.h5_compression()
                    )
                    logger.info(f"Saved stream '{stream_name}': shape {arr.shape}")

            for container_id, data_list in list(self.recorded_data.items()):
                if not data_list:
                    continue

                # The loop already held the last sample over gaps; held ticks are not real samples.
                fill_count = self._held_counts.get(container_id, 0)
                non_null_raw = len(data_list) - fill_count
                if fill_count > 0 and container_id in self._fill_gaps_containers:
                    self._fill_count_summary[container_id] = fill_count

                # Samples taken before the first message arrived are None. Dropping them would
                # leave this container shorter than `timestamps` and shift it against the rest.
                first_valid = next((d for d in data_list if d is not None), None)
                if first_valid is None:
                    logger.error(f"{container_id}: no data received during the whole recording")
                    filled_list = data_list
                else:
                    filled_list = [first_valid if d is None else d for d in data_list]

                fill_suffix = f", {fill_count} gap-filled by hold" if fill_count > 0 else ""
                logger.info(
                    f"{container_id} — {len(data_list)} samples "
                    f"({non_null_raw} real{fill_suffix})"
                )

                container_group = demo_group.create_group(container_id)

                data_by_type = defaultdict(list)
                for data_item in filled_list:
                    if data_item is not None:
                        data_type = data_item.get("type", "unknown")
                        data_by_type[data_type].append(data_item)

                for data_type, type_data_list in data_by_type.items():
                    self._save_data_type_to_h5_safe(
                        container_group, data_type, type_data_list
                    )

            self.recording_metadata["fill_count_summary"] = dict(self._fill_count_summary)
            self._save_metadata_to_h5(demo_group)

            logger.info(f"Saved recording data to {self.h5_file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving recorded data: {e}")
            return False

    def _save_joint_series(self, group, dataset_name, series):
        """Save one per-sample joint list (positions/velocities/efforts) as a dataset.

        Equal lengths give a regular array (empty samples become zeros); mixed lengths are saved
        as variable-length rows. Nothing is written if every sample is empty.
        """
        series = [[float(v) for v in row] for row in series]
        lengths = {len(row) for row in series if row}
        if not lengths:
            return
        if len(lengths) == 1:
            first_len = next(iter(lengths))
            group.create_dataset(
                dataset_name, data=np.array([row if row else [0.0] * first_len for row in series])
            )
        else:
            logger.warning(f"{dataset_name} have inconsistent lengths, saving as variable-length")
            rows = [np.array(row, dtype=np.float64) for row in series]  # empty rows too: one row per timestamp
            group.create_dataset(dataset_name, data=rows, dtype=h5py.vlen_dtype(np.float64))

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
                        _write_stacked(group, "images", images)
                    else:
                        logger.warning(
                            "Images have inconsistent shapes, saving individually"
                        )
                        for i, img in enumerate(images):
                            group.create_dataset(
                                f"image_{i}", data=img, **user_settings.h5_compression()
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
                for item_key, dataset_name in (
                    ("positions", "joint_positions"),
                    ("velocities", "joint_velocities"),
                    ("efforts", "joint_efforts"),
                ):
                    self._save_joint_series(
                        group, dataset_name, [item[item_key] or [] for item in data_list if item_key in item]
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
                            _write_stacked(group, "points", frames)
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
            # Propagate: _save_recorded_data reports the save as failed and the samples stay in memory.
            logger.error(f"Error saving data type {data_type}: {e}")
            raise

    def _save_metadata_to_h5(self, group):
        try:
            metadata_group = group.create_group("metadata")

            for key, value in self.recording_metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = json.dumps(value, default=str) if value else ""  # e.g. a YAML date

            logger.info("Saved metadata to H5 file")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
