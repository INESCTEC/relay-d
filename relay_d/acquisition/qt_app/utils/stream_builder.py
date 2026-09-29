"""Builds 'stream' vectors (config.output.streams) from per-tick input messages.

Pure yaml-config + message-dict logic, no ROS node / Qt dependency - shared by
DataRecorder (live recording, one tick at a time) and offline reprocessing
(scripts/reprocess_recorded_data.py, one reconstructed frame at a time via
qt_app/ui/review/demo_playback_reconstructor.py).
"""

import array as _array
import operator
import re
import types as _types

import numpy as np

from relay_d.utils.coloring_logger import logger

try:
    from sensor_msgs.msg import CompressedImage, Image, PointCloud2

    _UNSUPPORTED_STREAM_MSG_TYPES = (Image, CompressedImage, PointCloud2)
except ImportError:  # pragma: no cover - keeps this module importable without ROS installed
    _UNSUPPORTED_STREAM_MSG_TYPES = ()


def fill_none_gaps(data_list):
    """Forward-fill then back-fill None entries so all streams have equal length.

    Forward-fill handles Nones in the middle/end; back-fill covers leading Nones
    that arise when a stream's inputs haven't produced a valid row yet.
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
    last_valid = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid
    return filled


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
            for extractor in extractors:
                if extractor[0] == "simple":
                    names.add(extractor[1])
                else:  # "binary"
                    _kind, _op, (in_a, _ga), (in_b, _gb) = extractor
                    names.add(in_a)
                    names.add(in_b)
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

        # Matches "<operand> <op> <operand>" with a single +/- surrounded by
        # whitespace. Operand chars exclude space/+/-, so ROS-style identifiers,
        # dotted field paths and a trailing [:] are captured but the operator is
        # not, making the split unambiguous.
        binary_re = re.compile(
            r"^(?P<a>[\w.\[\]:]+)\s+(?P<op>[+\-])\s+(?P<b>[\w.\[\]:]+)$"
        )
        binary_ops = {"+": operator.add, "-": operator.sub}

        def _parse_operand(operand):
            """Resolve one dotted operand to (input_name, getter, is_slice).

            Returns None (and logs) on an operand with no dot separator so the
            caller can skip the whole spec.
            """
            operand = operand.strip()
            try:
                first_dot = operand.index(".")
            except ValueError:
                msg = f"Stream spec operand '{operand}' has no dot separator — skipped"
                logger.error(msg)
                self.build_errors.append(msg)
                return None
            input_name = operand[:first_dot]
            raw_field = operand[first_dot + 1:]
            is_slice = raw_field.endswith("[:]")
            field_path = raw_field[:-3] if is_slice else raw_field
            if input_name in self._tf_input_names:
                field_path = "transform." + field_path
            return input_name, operator.attrgetter(field_path), is_slice

        streams = self.yaml_parser.get_streams()
        for stream_name, specs in streams.items():
            if isinstance(specs, dict):  # dict form — extract spec list, ignore normalize/limits
                specs = specs.get("specs", [])
            extractor_list = []
            for spec in specs:
                # Detect sin(...) / cos(...) wrappers, e.g. sin(joint_1.position[:])
                math_op = None
                inner_spec = spec.strip()
                if inner_spec.startswith(("sin(", "cos(")) and inner_spec.endswith(")"):
                    func_name = inner_spec[:3]
                    inner_spec = inner_spec[4:-1].strip()
                    math_op = np.sin if func_name == "sin" else np.cos

                # Binary arithmetic spec: <operand> +|- <operand>
                bin_match = binary_re.match(inner_spec)
                if bin_match:
                    if math_op is not None:
                        msg = (
                            f"Stream spec '{spec}' combines sin/cos with arithmetic "
                            f"— unsupported, skipped"
                        )
                        logger.error(msg)
                        self.build_errors.append(msg)
                        continue
                    parsed_a = _parse_operand(bin_match.group("a"))
                    parsed_b = _parse_operand(bin_match.group("b"))
                    if parsed_a is None or parsed_b is None:
                        continue  # _parse_operand already logged
                    in_a, get_a, slice_a = parsed_a
                    in_b, get_b, slice_b = parsed_b
                    if slice_a or slice_b:
                        msg = (
                            f"Stream spec '{spec}': arithmetic operands must be scalars "
                            f"([:] not allowed) — skipped"
                        )
                        logger.error(msg)
                        self.build_errors.append(msg)
                        continue
                    op_func = binary_ops[bin_match.group("op")]
                    extractor_list.append(
                        ("binary", op_func, (in_a, get_a), (in_b, get_b))
                    )
                    continue

                # Guard against a +/- operator without surrounding spaces, which
                # would otherwise mis-parse as a single garbage field path.
                if ("+" in inner_spec or "-" in inner_spec):
                    msg = (
                        f"Stream spec '{spec}': arithmetic operators must have spaces "
                        f"around them (e.g. 'a.x - b.x') — skipped"
                    )
                    logger.error(msg)
                    self.build_errors.append(msg)
                    continue

                # Simple single-operand spec.
                parsed = _parse_operand(inner_spec)
                if parsed is None:
                    continue  # _parse_operand already logged the no-dot error
                input_name, getter, is_slice = parsed
                extractor_list.append(("simple", input_name, getter, is_slice, math_op))
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

    def apply_joint_filter(self, input_name: str, msg):
        allowed = self._joint_name_filters.get(input_name)
        if not allowed or not hasattr(msg, "name") or not msg.name:
            return msg
        names = list(msg.name)
        indices = [i for i, n in enumerate(names) if n in allowed]
        if not indices:
            if not self._joint_filter_warned.get(input_name):
                self._joint_filter_warned[input_name] = True
                logger.warning(
                    f"[joint_names filter] Input '{input_name}': none of the configured "
                    f"names matched the robot's published names.\n"
                    f"  Config names : {sorted(allowed)}\n"
                    f"  Robot names  : {sorted(names)}\n"
                    f"  Filter is inactive. Update 'joint_names' in the config to match exactly.\n"
                    f"  Tip: ros2 topic echo /joint_states --once | grep -A20 name"
                )
            return msg
        n = len(indices)
        filtered = _types.SimpleNamespace()
        filtered.name     = [names[i] for i in indices]
        filtered.position = [msg.position[i] for i in indices] if msg.position else [0.0] * n
        filtered.velocity = [msg.velocity[i] for i in indices] if msg.velocity else [0.0] * n
        filtered.effort   = [msg.effort[i]   for i in indices] if msg.effort   else [0.0] * n
        filtered.header   = msg.header
        return filtered

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

    def evaluate(self, all_data: dict):
        for stream_name, extractors in self._stream_extractors.items():
            row = []
            ok = True
            for extractor in extractors:
                if extractor[0] == "simple":
                    _kind, input_name, getter, is_slice, math_op = extractor
                    msg = all_data.get(input_name)
                    if msg is None:
                        ok = False
                        break
                    if isinstance(msg, _UNSUPPORTED_STREAM_MSG_TYPES):
                        self._warn_unsupported_type(stream_name, input_name, msg)
                        ok = False
                        break
                    msg = self.apply_joint_filter(input_name, msg)
                    try:
                        val = getter(msg)
                        if is_slice:
                            vals = list(val) if isinstance(val, _array.array) else list(val[:])
                            if math_op is not None:
                                vals = [float(math_op(v)) for v in vals]
                            row.extend(vals)
                        else:
                            v = float(val)
                            row.append(float(math_op(v)) if math_op is not None else v)
                    except (AttributeError, TypeError):
                        ok = False
                        break
                else:  # "binary": op_func(operand_a, operand_b), both scalars
                    _kind, op_func, (in_a, get_a), (in_b, get_b) = extractor
                    msg_a = all_data.get(in_a)
                    msg_b = all_data.get(in_b)
                    if msg_a is None or msg_b is None:
                        ok = False
                        break
                    if isinstance(msg_a, _UNSUPPORTED_STREAM_MSG_TYPES):
                        self._warn_unsupported_type(stream_name, in_a, msg_a)
                        ok = False
                        break
                    if isinstance(msg_b, _UNSUPPORTED_STREAM_MSG_TYPES):
                        self._warn_unsupported_type(stream_name, in_b, msg_b)
                        ok = False
                        break
                    msg_a = self.apply_joint_filter(in_a, msg_a)
                    msg_b = self.apply_joint_filter(in_b, msg_b)
                    try:
                        row.append(float(op_func(float(get_a(msg_a)), float(get_b(msg_b)))))
                    except (AttributeError, TypeError):
                        ok = False
                        break
            buf = self._stream_buffers.setdefault(stream_name, [])
            if ok:
                self._stream_last_valid[stream_name] = row
                buf.append(row)
            elif stream_name in self._stream_last_valid:
                # Sample-and-hold: reuse last valid row when input is temporarily missing.
                # Ensures all streams have identical row counts for Robomimic alignment.
                buf.append(self._stream_last_valid[stream_name])
            else:
                buf.append(None)   # no valid row yet — filtered at save time

    def finalize(self) -> dict:
        """Return {stream_name: (N, M) float32 array} for every stream with data."""
        result = {}
        for stream_name, rows in self._stream_buffers.items():
            filled_rows = fill_none_gaps(rows)
            valid = [r for r in filled_rows if r is not None]
            if valid:
                result[stream_name] = np.array(valid, dtype=np.float32)
        return result
