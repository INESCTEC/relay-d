"""Builds 'stream' vectors (config.output.streams) from per-tick input messages.

Pure yaml-config + message-dict logic, no ROS node / Qt dependency - shared by
DataRecorder (live recording, one tick at a time) and offline reprocessing
(scripts/reprocess_recorded_data.py, one reconstructed frame at a time via
qt_app/ui/review/demo_playback_reconstructor.py).
"""

import types as _types

import numpy as np

from relay_d.utils.coloring_logger import logger
from relay_d.utils.numerics import enriched_tf, entry_inputs, evaluate_entries, parse_spec, select_joints, stream_entry

try:
    from sensor_msgs.msg import CompressedImage, Image, PointCloud2

    _UNSUPPORTED_STREAM_MSG_TYPES = (Image, CompressedImage, PointCloud2)
except ImportError:  # pragma: no cover - keeps this module importable without ROS installed
    _UNSUPPORTED_STREAM_MSG_TYPES = ()


class StreamBuilder:
    """Parses config.output.streams specs and evaluates them tick-by-tick.

    Construct with a loaded (or unloaded) YamlParser; if the parser has no
    yaml_data (from either load_yaml_file() or load_from_yaml_string()), the
    builder simply has no streams (has_streams() is False, evaluate()/finalize()
    are no-ops) - same behavior as the extractor-building this replaces.
    """

    def __init__(self, yaml_parser):
        self.yaml_parser = yaml_parser
        self._stream_extractors = {}
        self._stream_buffers = {}
        self._stream_last_valid = {}
        self._tf_input_names = set()
        self._joint_name_filters = {}
        self._joint_filter_warned = {}
        self._joint_last_full = {}  # per joint-filtered input: its last message with every configured joint
        self._unsupported_type_warned = set()
        self.build_errors = []
        self._build_stream_extractors()

    def has_streams(self) -> bool:
        return bool(self._stream_extractors)

    def get_build_errors(self) -> list:
        """Every spec-parsing error collected while building extractors
        (same messages already sent to logger.error, kept here so callers —
        e.g. ConfigPage — can surface them to the user instead of only the
        console)."""
        return self.build_errors

    def referenced_input_names(self) -> set:
        """Every input_name referenced by any stream spec (both operands of a
        binary arithmetic spec are included)."""
        names = set()
        for extractors in self._stream_extractors.values():
            names |= entry_inputs(extractors)
        return names

    def _build_stream_extractors(self):
        self._stream_extractors = {}
        self._tf_input_names = set()
        self._joint_name_filters = {}
        self.build_errors = []
        # Checking yaml_data directly (rather than is_loaded()) so this also
        # works right after YamlParser.load_from_yaml_string() — is_loaded()
        # is gated on current_config, which only load_yaml_file() sets.
        if not (self.yaml_parser and self.yaml_parser.yaml_data):
            return

        input_section = self.yaml_parser.yaml_data.get("config", {}).get("input", {})
        for name, spec in input_section.items():
            if isinstance(spec, dict) and "parent_frame" in spec and "child_frame" in spec:
                self._tf_input_names.add(name)

        for entry in self.yaml_parser.get_input_data():
            names = entry.get("joint_names", [])
            if names:
                self._joint_name_filters[entry["name"]] = frozenset(names)
                logger.info(f"Joint filter for '{entry['name']}': {list(names)}")

        def _error(msg):
            logger.error(msg)
            self.build_errors.append(msg)

        streams = self.yaml_parser.get_streams()
        for stream_name, entry in streams.items():
            specs = stream_entry(entry)[0]  # rotation_format is applied at conversion
            extractor_list = []
            for spec in specs:
                try:
                    extractor_list.append(parse_spec(spec, self._tf_input_names))
                except ValueError as e:
                    _error(f"Stream spec '{spec}': {e} — skipped")
            if not extractor_list:
                reason = (
                    "has no specs configured" if not specs
                    else "every spec failed to parse (see preceding errors)"
                )
                msg = f"Stream '{stream_name}': {reason} — stream will not be saved."
                logger.error(msg)
                self.build_errors.append(msg)
                continue
            self._stream_extractors[stream_name] = extractor_list
            logger.info(f"Built extractor for stream '{stream_name}' ({len(extractor_list)} specs)")

    def _filter_joints(self, input_name: str, msg):
        """``msg`` reduced to the configured joint_names. A message missing any of them (e.g. a gripper-only publisher
        on the same topic) gives the last full one instead, None before any: as live inference, never a narrower row."""
        allowed = self._joint_name_filters.get(input_name)
        if not allowed or not hasattr(msg, "name"):
            return msg
        names = list(msg.name)
        missing = allowed - set(names)
        if missing:
            if not self._joint_filter_warned.get(input_name):
                self._joint_filter_warned[input_name] = True
                logger.warning(
                    f"[joint_names filter] Input '{input_name}': configured joints not found in a message; "
                    f"such messages are ignored (the last full one is kept).\n"
                    f"  Config names : {sorted(allowed)}\n"
                    f"  Robot names  : {sorted(names)}\n"
                    f"  Missing      : {sorted(missing)}\n"
                    f"  Update 'joint_names' in the config to match exactly.\n"
                    f"  Tip: ros2 topic echo /joint_states --once | grep -A20 name"
                )
            return self._joint_last_full.get(input_name)
        filtered = _types.SimpleNamespace(header=msg.header)
        filtered.name, filtered.position, filtered.velocity, filtered.effort = select_joints(msg, allowed)
        self._joint_last_full[input_name] = filtered
        return filtered

    def apply_joint_filter(self, input_name: str, msg):
        """_filter_joints for the raw input recording, which cannot take None: the message as is until the first full one."""
        filtered = self._filter_joints(input_name, msg)
        return msg if filtered is None else filtered

    def _warn_unsupported_type(self, stream_name: str, input_name: str, msg) -> None:
        """Log once per stream why it's producing no data: the input resolved
        to an Image/CompressedImage/PointCloud2 message, which output.streams
        can't represent as a float32 vector (see module docstring)."""
        if stream_name in self._unsupported_type_warned:
            return
        self._unsupported_type_warned.add(stream_name)
        logger.error(
            f"Stream '{stream_name}': input '{input_name}' is a "
            f"{type(msg).__name__}, which is not supported in output.streams — "
            f"pixel/point data can't be represented as a float32 vector. This "
            f"spec will be skipped for every tick. Expose '{input_name}' as an "
            f"obs directly instead, via a top-level 'output:' bridge-map entry "
            f"(sibling to 'streams:'), e.g.:\n"
            f"    output:\n"
            f"      {stream_name}: {input_name}/images\n"
            f"      streams: ...\n"
            f"See the 'output.streams' comment block in config_template.yaml."
        )

    def _input(self, stream_name, input_name, all_data, prepared_cache=None):
        """The message of ``input_name`` ready for the getters, or None (no data / unsupported type)."""
        if prepared_cache is not None and input_name in prepared_cache:
            return prepared_cache[input_name]
        msg = all_data.get(input_name)
        if msg is None:
            result = None
        elif isinstance(msg, _UNSUPPORTED_STREAM_MSG_TYPES):
            self._warn_unsupported_type(stream_name, input_name, msg)
            result = None
        elif input_name in self._tf_input_names:
            result = enriched_tf(msg)  # euler.* / rot6d.* / rot_matrix.*, as obs_builder sees TF
        else:
            result = self._filter_joints(input_name, msg)
        if prepared_cache is not None and not isinstance(msg, _UNSUPPORTED_STREAM_MSG_TYPES):
            prepared_cache[input_name] = result
        return result

    def evaluate(self, all_data: dict):
        # Every row first, appended after: an error half way must not leave earlier streams one row ahead for good.
        # Cache prepared TF/joint messages per tick so multi-spec streams compute enriched_tf / _filter_joints once.
        prepared_cache = {}
        rows = {}
        for stream_name, extractors in self._stream_extractors.items():
            try:
                rows[stream_name] = evaluate_entries(
                    extractors, lambda n, s=stream_name: self._input(s, n, all_data, prepared_cache)
                )
            except (AttributeError, TypeError, IndexError, ValueError, ZeroDivisionError):  # e.g. a short joint array: hold the last row
                rows[stream_name] = None
        for stream_name, row in rows.items():
            buf = self._stream_buffers.setdefault(stream_name, [])
            if row is not None:
                self._stream_last_valid[stream_name] = row
                buf.append(row)
            else:
                # Sample-and-hold: reuse the last valid row (None before the first: filtered at save time),
                # so all streams have identical row counts for Robomimic alignment.
                buf.append(self._stream_last_valid.get(stream_name))

    def finalize(self) -> dict:
        """Return {stream_name: (N, M) float32 array} for every stream with data."""
        result = {}
        for stream_name, rows in self._stream_buffers.items():
            # evaluate() already holds the last row; only the rows before the first valid one are None
            first = next((r for r in rows if r is not None), None)
            if first is not None:
                result[stream_name] = np.array([first if r is None else r for r in rows], dtype=np.float32)
        return result
