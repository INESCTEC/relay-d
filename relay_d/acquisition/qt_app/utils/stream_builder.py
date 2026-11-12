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
from relay_d.utils.numerics import enriched_tf, read_scalar, strip_trig

try:
    from sensor_msgs.msg import CompressedImage, Image, PointCloud2

    _UNSUPPORTED_STREAM_MSG_TYPES = (Image, CompressedImage, PointCloud2)
except ImportError:  # pragma: no cover - keeps this module importable without ROS installed
    _UNSUPPORTED_STREAM_MSG_TYPES = ()


# Same spec grammar as relay_d/dispatch/obs_builder.py, so a recorded stream has the width and values
# the live obs will have: "<operand> <op> <operand>" needs spaces around the operator.
_BINOP_RE = re.compile(r"^(.+?)\s+([+\-*/])\s+(.+)$")
_BIN_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


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
                elif extractor[0] == "binary":
                    names.add(extractor[2][0])
                    names.add(extractor[3][0])
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

        def _parse_operand(operand, spec):
            """Resolve one dotted operand to (input_name, getter, is_slice), or None (logged)."""
            operand = operand.strip()
            if "." not in operand:
                _error(f"Stream spec '{spec}': operand '{operand}' has no dot separator — skipped")
                return None
            input_name, raw_field = operand.split(".", 1)
            is_slice = raw_field.endswith("[:]")
            field_path = raw_field[:-3] if is_slice else raw_field
            if any(c in field_path for c in "+-*/"):
                _error(
                    f"Stream spec '{spec}': arithmetic operators must have spaces "
                    f"around them (e.g. 'a.x - b.x') — skipped"
                )
                return None
            if input_name in self._tf_input_names:
                field_path = "transform." + field_path
            return input_name, operator.attrgetter(field_path), is_slice

        streams = self.yaml_parser.get_streams()
        for stream_name, specs in streams.items():
            if isinstance(specs, dict):  # dict form — extract spec list, ignore rotation_format
                specs = specs.get("specs", [])
            extractor_list = []
            for spec in specs:
                # sin(...)/cos(...) enclosing the whole spec, e.g. sin(joint_1.position[:]) or sin(a.x - b.x)
                math_op, inner_spec = strip_trig(spec.strip())

                if inner_spec.startswith("zero:"):
                    try:
                        extractor_list.append(("zero", int(inner_spec[5:])))
                    except ValueError:
                        _error(f"Stream spec '{spec}': zero:N needs an integer N — skipped")
                    continue

                # Binary arithmetic spec: <operand> +|-|*|/ <operand>, each side may have its own sin/cos
                bin_match = _BINOP_RE.match(inner_spec)
                if bin_match:
                    sides = []
                    for side in (bin_match.group(1).strip(), bin_match.group(3).strip()):
                        if _BINOP_RE.match(side):
                            _error(f"Stream spec '{spec}': only one arithmetic operation per spec — skipped")
                            break
                        side_op, side_spec = strip_trig(side)
                        parsed = _parse_operand(side_spec, spec)
                        if parsed is None:
                            break
                        if parsed[2]:
                            _error(
                                f"Stream spec '{spec}': arithmetic operands must be scalars "
                                f"([:] not allowed) — skipped"
                            )
                            break
                        sides.append((parsed[0], parsed[1], side_op))
                    if len(sides) == 2:
                        extractor_list.append(("binary", _BIN_OPS[bin_match.group(2)], sides[0], sides[1], math_op))
                    continue

                # Simple single-operand spec.
                parsed = _parse_operand(inner_spec, spec)
                if parsed is None:
                    continue
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
        # An array not filled for every joint (e.g. no velocity) reads as zeros instead of raising IndexError.
        full = len(names)
        filtered.position = [msg.position[i] for i in indices] if len(msg.position) == full else [0.0] * n
        filtered.velocity = [msg.velocity[i] for i in indices] if len(msg.velocity) == full else [0.0] * n
        filtered.effort   = [msg.effort[i]   for i in indices] if len(msg.effort)   == full else [0.0] * n
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

    def _input(self, stream_name, input_name, all_data):
        """The message of ``input_name`` ready for the getters, or None (no data / unsupported type)."""
        msg = all_data.get(input_name)
        if msg is None:
            return None
        if isinstance(msg, _UNSUPPORTED_STREAM_MSG_TYPES):
            self._warn_unsupported_type(stream_name, input_name, msg)
            return None
        if input_name in self._tf_input_names:
            return enriched_tf(msg)  # euler.* / rot6d.* / rot_matrix.*, as obs_builder sees TF
        return self.apply_joint_filter(input_name, msg)

    def evaluate(self, all_data: dict):
        for stream_name, extractors in self._stream_extractors.items():
            row = []
            ok = True
            for extractor in extractors:
                try:
                    if extractor[0] == "zero":
                        row.extend([0.0] * extractor[1])
                    elif extractor[0] == "simple":
                        _kind, input_name, getter, is_slice, math_op = extractor
                        msg = self._input(stream_name, input_name, all_data)
                        if msg is None:
                            ok = False
                            break
                        if is_slice:
                            val = getter(msg)
                            vals = list(val) if isinstance(val, _array.array) else list(val[:])
                            row.extend(float(math_op(v)) if math_op is not None else float(v) for v in vals)
                        else:
                            row.append(read_scalar(getter, msg, math_op))
                    else:  # "binary": outer_op(side_op(a) <op> side_op(b)), both scalars
                        _kind, op_func, (in_a, get_a, op_a), (in_b, get_b, op_b), math_op = extractor
                        msg_a = self._input(stream_name, in_a, all_data)
                        msg_b = self._input(stream_name, in_b, all_data)
                        if msg_a is None or msg_b is None:
                            ok = False
                            break
                        v = float(op_func(read_scalar(get_a, msg_a, op_a), read_scalar(get_b, msg_b, op_b)))
                        row.append(float(math_op(v)) if math_op is not None else v)
                except (AttributeError, TypeError, IndexError, ValueError, ZeroDivisionError):  # e.g. a short joint array: hold the last row
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
            # evaluate() already holds the last row; only the rows before the first valid one are None
            first = next((r for r in rows if r is not None), None)
            if first is not None:
                result[stream_name] = np.array([first if r is None else r for r in rows], dtype=np.float32)
        return result
