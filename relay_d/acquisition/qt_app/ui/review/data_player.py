"""Republishes a recorded demo's HDF5 data onto /playback/* topics + TF, paced
like `ros2 bag play`, driven by DemoPlaybackReader.

Owns a throwaway rclpy node created in play() and destroyed in stop() - the
same short-lived-node idiom TopicSubscribers already uses for its dedicated
TF listener node, rather than the app-lifetime Node subclassing DataRecorder
and TopicSubscribers use for themselves.
"""

import time
from enum import Enum
from threading import Event, Lock, Thread

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState

try:
    from tf2_ros import TransformBroadcaster
except ImportError:
    TransformBroadcaster = None

from relay_d.utils.coloring_logger import logger
from .demo_playback_reconstructor import DemoPlaybackReader, ChannelKind

# tf2_ros.TransformBroadcaster hardcodes its publish topic to the absolute
# "/tf" - a namespace remap does not affect it, only an explicit topic remap
# does. Applied to this node so its TF traffic lands on /playback/tf instead
# of colliding with any live /tf.
_TF_REMAP_ARGS = [
    "--ros-args",
    "-r", "/tf:=/playback/tf",
    "-r", "/tf_static:=/playback/tf_static",
]


class PlaybackState(Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


class DataPlayer:
    def __init__(self, register_node_fn, unregister_node_fn):
        self._register_node_fn = register_node_fn
        self._unregister_node_fn = unregister_node_fn

        self._reader = None
        self._node = None
        self._joint_pub = None
        self._other_pubs = {}
        self._tf_broadcaster = None

        self._thread = None
        self._stop_event = Event()
        self._pause_event = Event()
        self._lock = Lock()
        self._frame_index = 0
        self._state = PlaybackState.IDLE

    @property
    def state(self):
        return self._state

    @property
    def progress(self):
        with self._lock:
            total = self._reader.num_frames if self._reader else 0
            return self._frame_index, total

    @property
    def time_progress(self):
        """(elapsed_sec, duration_sec) of the current playback position."""
        if self._reader is None:
            return 0.0, 0.0
        with self._lock:
            frame_index = self._frame_index
        return self._reader.elapsed_sec(frame_index), self._reader.duration_sec

    def channel_summary(self) -> str:
        return self._reader.describe() if self._reader else ""

    def has_playable_channels(self) -> bool:
        return bool(self._reader) and self._reader.has_playable_channels()

    def load(self, h5_path: str):
        """Load a new demo. Tears down any previous playback first."""
        self.stop()
        if self._reader is not None:
            self._reader.close()
        self._reader = DemoPlaybackReader(h5_path)
        with self._lock:
            self._frame_index = 0
        self._state = PlaybackState.IDLE

    def play(self) -> bool:
        if self._reader is None:
            logger.error("DataPlayer.play() called with no demo loaded")
            return False

        if self._state == PlaybackState.PLAYING:
            return True

        if self._state == PlaybackState.PAUSED:
            self._pause_event.clear()
            self._state = PlaybackState.PLAYING
            return True

        # IDLE or FINISHED - (re)start from frame 0. Only build the node and
        # publishers if they don't already exist (FINISHED leaves them up).
        if self._node is None:
            try:
                self._setup_node()
            except Exception as e:
                logger.error(f"Failed to set up playback node: {e}")
                self._teardown_node()
                return False

        with self._lock:
            self._frame_index = 0
        self._stop_event.clear()
        self._pause_event.clear()
        self._state = PlaybackState.PLAYING
        self._thread = Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        return True

    def pause(self) -> bool:
        if self._state != PlaybackState.PLAYING:
            return False
        self._pause_event.set()
        self._state = PlaybackState.PAUSED
        return True

    def stop(self) -> bool:
        if self._state == PlaybackState.IDLE and self._node is None:
            return True

        self._stop_event.set()
        self._pause_event.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        with self._lock:
            self._frame_index = 0

        self._teardown_node()
        self._state = PlaybackState.IDLE
        return True

    def close(self):
        self.stop()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # ------------------------------------------------------------------
    # Node / publisher lifecycle
    # ------------------------------------------------------------------

    def _setup_node(self):
        node_name = f"data_player_{id(self)}"
        self._node = rclpy.create_node(node_name, cli_args=_TF_REMAP_ARGS)

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        has_joint = any(c.kind == ChannelKind.JOINT_STATE for c in self._reader.channels)
        if has_joint:
            self._joint_pub = self._node.create_publisher(
                JointState, "/playback/joint_states", qos
            )

        self._other_pubs = {}
        for ch in self._reader.channels:
            msg_cls = ch.msg_class()
            if msg_cls is None:
                continue
            self._other_pubs[ch.name] = self._node.create_publisher(
                msg_cls, f"/playback/{ch.name}", qos
            )

        has_tf = any(c.kind == ChannelKind.TRANSFORM for c in self._reader.channels)
        if has_tf:
            if TransformBroadcaster is None:
                logger.warning("tf2_ros not available - TF channels will not be published")
            else:
                self._tf_broadcaster = TransformBroadcaster(self._node)

        self._register_node_fn(self._node)

    def _teardown_node(self):
        if self._node is None:
            return
        try:
            self._unregister_node_fn(self._node)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception as e:
            logger.warning(f"Error destroying playback node: {e}")
        self._node = None
        self._joint_pub = None
        self._other_pubs = {}
        self._tf_broadcaster = None

    # ------------------------------------------------------------------
    # Playback loop
    # ------------------------------------------------------------------

    def _playback_loop(self):
        reader = self._reader
        num_frames = reader.num_frames
        timestamps = reader.timestamps_nsec
        transform_channels = [c for c in reader.channels if c.kind == ChannelKind.TRANSFORM]
        publishable_channels = [
            c for c in reader.channels
            if c.kind not in (ChannelKind.JOINT_STATE, ChannelKind.TRANSFORM)
            and c.name in self._other_pubs
        ]

        frame_index = 0
        next_tick = time.monotonic()

        try:
            while frame_index < num_frames and not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.02)
                    next_tick = time.monotonic()
                    continue

                # Stamp with our own clock rather than the recorded epoch
                # timestamp - avoids TF-buffer extrapolate-into-the-past
                # warnings on nodes running with use_sim_time=False. Pacing
                # (real-time-relative speed) comes from sleeping according to
                # the recorded inter-sample deltas below, not from the stamp.
                stamp = self._node.get_clock().now().to_msg()

                if self._joint_pub is not None:
                    self._joint_pub.publish(
                        reader.build_combined_joint_state(frame_index, stamp)
                    )

                for ch in publishable_channels:
                    msg = ch.build(frame_index, stamp)
                    if msg is not None:
                        self._other_pubs[ch.name].publish(msg)

                if transform_channels and self._tf_broadcaster is not None:
                    transforms = [
                        t for t in (ch.build(frame_index, stamp) for ch in transform_channels)
                        if t is not None
                    ]
                    if transforms:
                        self._tf_broadcaster.sendTransform(transforms)

                frame_index += 1
                with self._lock:
                    self._frame_index = frame_index

                if frame_index >= num_frames:
                    break

                delta_sec = (
                    int(timestamps[frame_index]) - int(timestamps[frame_index - 1])
                ) / 1e9
                next_tick += max(delta_sec, 0.0)
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()

        except Exception as e:
            logger.error(f"Error in playback loop: {e}")

        if not self._stop_event.is_set():
            self._state = PlaybackState.FINISHED
