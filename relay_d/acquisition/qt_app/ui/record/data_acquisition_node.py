from __future__ import annotations

import array
import operator
import threading
import yaml
from typing import Any

import rclpy
import rclpy.time
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:
    Buffer = None
    TransformListener = None

from relay_d.utils.coloring_logger import logger


class DataAcquisitionNode(Node):
    """
    ROS 2 node that parses a YAML config and acquires data from multiple topics.

    Supports two input types:
      - topic inputs:  {topic: /some/topic}
      - TF inputs:     {parent_frame: X, child_frame: Y}

    Stream field paths use dot-notation: "<input_name>.<field.path>[:]?"
    The "[:]" suffix converts the value to a flat Python list (handles array.array).

    All extraction logic is pre-compiled via operator.attrgetter at init time.
    No string operations occur in callbacks or read_stream.

    YAML schema:
        config:
          inputs:
            <name>: {topic: /path} | {parent_frame: X, child_frame: Y}
          outputs:
            streams:
              <stream_name>:
                - <input_name>.<field_path>[:]?
    """

    def __init__(
        self,
        inputs: dict,
        streams: dict,
        node_name: str = "data_acquisition_node",
    ) -> None:
        super().__init__(node_name)

        self._inputs_cfg: dict = inputs
        self._streams_cfg: dict = streams

        # input_name -> latest msg or Transform; protected by _state_lock
        self._state: dict[str, Any] = {}
        self._state_lock = threading.Lock()

        # input_name -> subscription (or "tf"); write-once after init
        self._subscribers: dict[str, Any] = {}

        # input_name -> topic_path; entries removed as topics are resolved
        self._pending_topics: dict[str, str] = {}

        # stream_name -> [(input_name, getter_fn, is_slice), ...]
        self._extractors: dict[str, list[tuple]] = {}

        # topic_path -> msg class; write-once per topic
        self._type_cache: dict[str, type] = {}

        # input_name -> {parent_frame, child_frame}
        self._tf_inputs: dict[str, dict] = {}

        self._poll_timer = None
        self._tf_timer = None

        self._init_tf()
        self._classify_inputs()
        self._build_extractors()

        if self._pending_topics:
            self._poll_timer = self.create_timer(2.0, self._poll_topics)
            logger.info(
                f"[DataAcquisitionNode] Poll timer started for: {list(self._pending_topics.keys())}"
            )

        if self._tf_inputs:
            self._tf_timer = self.create_timer(0.1, self._update_tf_state)
            logger.info(
                f"[DataAcquisitionNode] TF timer started for: {list(self._tf_inputs.keys())}"
            )

    # ------------------------------------------------------------------
    # Classmethod constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml_file(
        cls,
        path: str,
        node_name: str = "data_acquisition_node",
    ) -> "DataAcquisitionNode":
        """Load config from a YAML file and construct the node."""
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        cfg = raw.get("config", {})
        inputs = cfg.get("inputs", {})
        outputs = cfg.get("outputs", {})
        streams = outputs.get("streams", {})

        if not inputs:
            raise ValueError(f"YAML at '{path}' has no config.inputs section")
        if not streams:
            raise ValueError(f"YAML at '{path}' has no config.outputs.streams section")

        return cls(inputs=inputs, streams=streams, node_name=node_name)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_tf(self) -> None:
        self._tf_buffer = None
        self._tf_listener = None

        if Buffer is None:
            logger.warning("[DataAcquisitionNode] tf2_ros not available — TF inputs will be skipped")
            return

        try:
            self._tf_buffer = Buffer(cache_time=RclDuration(seconds=10.0))
            self._tf_listener = TransformListener(self._tf_buffer, self)
            logger.info("[DataAcquisitionNode] TF2 buffer and listener initialized")
        except Exception as exc:
            logger.warning(f"[DataAcquisitionNode] TF2 init failed: {exc}")

    def _classify_inputs(self) -> None:
        """Sort inputs into topic subscribers vs TF lookups."""
        for name, spec in self._inputs_cfg.items():
            if not isinstance(spec, dict):
                logger.warning(f"[DataAcquisitionNode] Input '{name}' is not a dict — skipped")
                continue

            if "topic" in spec:
                self._pending_topics[name] = spec["topic"]
                with self._state_lock:
                    self._state[name] = None
            elif "parent_frame" in spec and "child_frame" in spec:
                self._tf_inputs[name] = {
                    "parent_frame": spec["parent_frame"],
                    "child_frame": spec["child_frame"],
                }
                with self._state_lock:
                    self._state[name] = None
                self._subscribers[name] = "tf"
            else:
                logger.warning(
                    f"[DataAcquisitionNode] Input '{name}' has neither 'topic' "
                    "nor 'parent_frame'/'child_frame' — skipped"
                )

    def _build_extractors(self) -> None:
        """
        Pre-compile operator.attrgetter extractors for every stream field spec.

        Spec format: "<input_name>.<field_path>[:]?"
        The first dot separates input name from the field path.
        String splitting happens here only — never in callbacks or read_stream.
        """
        for stream_name, specs in self._streams_cfg.items():
            if not isinstance(specs, list):
                logger.warning(
                    f"[DataAcquisitionNode] Stream '{stream_name}' value is not a list — skipped"
                )
                continue

            extractor_list = []
            for spec in specs:
                try:
                    first_dot = spec.index(".")
                except ValueError:
                    raise ValueError(
                        f"Stream spec '{spec}' in stream '{stream_name}' has no dot separator. "
                        "Expected format: '<input_name>.<field_path>'"
                    )

                input_name = spec[:first_dot]
                raw_field = spec[first_dot + 1:]

                is_slice = raw_field.endswith("[:]")
                field_path = raw_field[:-3] if is_slice else raw_field

                getter = operator.attrgetter(field_path)
                extractor_list.append((input_name, getter, is_slice))

                logger.debug(
                    f"[DataAcquisitionNode] Extractor: stream={stream_name} "
                    f"input={input_name} field={field_path} slice={is_slice}"
                )

            self._extractors[stream_name] = extractor_list

    # ------------------------------------------------------------------
    # Poll timer — discovers and subscribes to pending topics
    # ------------------------------------------------------------------

    def _poll_topics(self) -> None:
        """Timer callback (2 s): resolve pending topic types and create subscriptions."""
        if not self._pending_topics:
            if self._poll_timer is not None:
                self._poll_timer.cancel()
                self._poll_timer = None
            return

        try:
            live: dict[str, list[str]] = dict(self.get_topic_names_and_types())
        except Exception as exc:
            logger.warning(f"[DataAcquisitionNode] get_topic_names_and_types failed: {exc}")
            return

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        resolved = []

        for input_name, topic_path in list(self._pending_topics.items()):
            type_strings = live.get(topic_path)
            if not type_strings:
                continue

            type_str = type_strings[0]
            try:
                msg_class = self._type_cache.get(topic_path)
                if msg_class is None:
                    msg_class = get_message(type_str)
                    self._type_cache[topic_path] = msg_class
            except Exception as exc:
                logger.error(
                    f"[DataAcquisitionNode] get_message('{type_str}') failed "
                    f"for input '{input_name}': {exc}"
                )
                continue

            sub = self.create_subscription(
                msg_class,
                topic_path,
                self._make_topic_callback(input_name),
                qos,
            )
            self._subscribers[input_name] = sub
            resolved.append(input_name)
            logger.info(
                f"[DataAcquisitionNode] Subscribed: input='{input_name}' "
                f"topic='{topic_path}' type='{type_str}'"
            )

        for name in resolved:
            del self._pending_topics[name]

        if not self._pending_topics:
            logger.info("[DataAcquisitionNode] All topic subscriptions established")
            if self._poll_timer is not None:
                self._poll_timer.cancel()
                self._poll_timer = None

    # ------------------------------------------------------------------
    # Topic callback factory
    # ------------------------------------------------------------------

    def _make_topic_callback(self, input_name: str):
        """Return a subscriber callback that stores the latest message under input_name."""
        def _cb(msg):
            with self._state_lock:
                self._state[input_name] = msg
        return _cb

    # ------------------------------------------------------------------
    # TF timer — periodic transform lookup
    # ------------------------------------------------------------------

    def _update_tf_state(self) -> None:
        """
        Timer callback (0.1 s): look up all TF inputs and update state.

        Stores stamped.transform (geometry_msgs.msg.Transform) — NOT the full
        TransformStamped — so that field paths like 'translation.x' resolve
        directly via attrgetter without requiring a 'transform.' prefix.
        """
        if self._tf_buffer is None:
            return

        for input_name, tf_spec in self._tf_inputs.items():
            try:
                stamped = self._tf_buffer.lookup_transform(
                    tf_spec["parent_frame"],
                    tf_spec["child_frame"],
                    rclpy.time.Time(seconds=0.0),
                    timeout=RclDuration(seconds=0.0),  # non-blocking: return latest or raise
                )
                with self._state_lock:
                    self._state[input_name] = stamped.transform
            except Exception:
                pass  # Transform not yet available — state remains None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_stream(self, stream_name: str) -> list | None:
        """
        Extract and return the current values for a stream.

        Returns a flat list of values (scalars extended for sliced specs).
        Returns None if any required input has not yet received data.

        No string operations are performed here; all extraction uses
        pre-compiled operator.attrgetter callables from _build_extractors.
        """
        extractors = self._extractors.get(stream_name)
        if not extractors:
            logger.warning(f"[DataAcquisitionNode] Unknown stream: '{stream_name}'")
            return None

        with self._state_lock:
            snapshot = {inp: self._state.get(inp) for inp, _, _ in extractors}

        result = []
        for input_name, getter, is_slice in extractors:
            obj = snapshot.get(input_name)
            if obj is None:
                return None

            try:
                val = getter(obj)
            except AttributeError as exc:
                logger.error(
                    f"[DataAcquisitionNode] Extractor failed for stream='{stream_name}' "
                    f"input='{input_name}': {exc}"
                )
                return None

            if is_slice:
                # array.array from ROS fixed-size sequences; list/tuple for variable ones
                if isinstance(val, array.array):
                    result.extend(list(val))
                else:
                    result.extend(list(val[:]))
            else:
                result.append(val)

        return result

    def read_all_streams(self) -> dict[str, list | None]:
        """Read all configured streams and return a dict of stream_name -> values."""
        return {name: self.read_stream(name) for name in self._extractors}

    def is_ready(self) -> bool:
        """Return True when every configured input has received at least one data point."""
        with self._state_lock:
            return all(v is not None for v in self._state.values())

    def get_stream_names(self) -> list[str]:
        return list(self._extractors.keys())

    def get_input_names(self) -> list[str]:
        return list(self._inputs_cfg.keys())

    def get_pending_inputs(self) -> list[str]:
        """Return the names of topic inputs not yet subscribed."""
        return list(self._pending_topics.keys())
