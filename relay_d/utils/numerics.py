"""Maths shared by training-data conversion and live inference.

Both sides must apply the same rule, otherwise the policy sees different numbers at run time
than it was trained on. Keep them here, not in two copies.
"""

import operator
import re
import types
from typing import Optional

import numpy as np


def stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time (sec, nanosec), e.g. a header.stamp -> float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def stream_entry(stream_val):
    """An ``output.streams`` entry as (spec list, rotation_format). Accepts the plain-list form or the dict form
    ``{specs: [...], rotation_format: quat | rpy}`` (quat, the default, keeps the values as they are). Recording,
    conversion and inference all read streams through this, so they cannot disagree on a stream's layout."""
    if isinstance(stream_val, dict):
        return stream_val.get("specs", []), str(stream_val.get("rotation_format", "quat")).lower()
    return stream_val, "quat"


def stamp_to_ns(stamp) -> int:
    """builtin_interfaces/Time (sec, nanosec) -> integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def select_joints(msg, wanted):
    """A JointState's joints that are in `wanted`, as (names, position, velocity, effort) lists.

    Kept in the message's order: that is how the recorder stores them, so live inference must read
    them the same way. An array not filled for every joint (some drivers send no velocity) reads as zeros.
    """
    names = list(msg.name)
    idx = [i for i, n in enumerate(names) if n in wanted]

    def pick(arr):
        return [float(arr[i]) for i in idx] if len(arr) == len(names) else [0.0] * len(idx)

    return [names[i] for i in idx], pick(msg.position), pick(msg.velocity), pick(msg.effort)


def quat_to_euler(q) -> np.ndarray:
    """Quaternion(s) (..., 4) as (x, y, z, w) -> (..., 3) roll, pitch, yaw in radians (extrinsic xyz)."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def enriched_tf(tf_stamped) -> types.SimpleNamespace:
    """TransformStamped -> object whose ``transform`` also has euler.roll/pitch/yaw, rot6d.i0..i5
    (first two columns of R, column-major) and rot_matrix.r00..r22, so stream specs such as
    ``tf.euler.yaw`` read the same values when recording and at inference."""
    t, r = tf_stamped.transform.translation, tf_stamped.transform.rotation
    roll, pitch, yaw = (float(v) for v in quat_to_euler([r.x, r.y, r.z, r.w]))
    R = quat_to_rot_matrix(r.x, r.y, r.z, r.w)
    return types.SimpleNamespace(transform=types.SimpleNamespace(
        translation=t,
        rotation=r,
        euler=types.SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw),
        rot_matrix=types.SimpleNamespace(**{f"r{i}{j}": R[i, j] for i in range(3) for j in range(3)}),
        rot6d=types.SimpleNamespace(**{f"i{i}": float(v) for i, v in enumerate(list(R[:, 0]) + list(R[:, 1]))}),
    ))


def inputs_in_sync(names, timestamps, threshold: Optional[float]) -> bool:
    """True if the given inputs' timestamps all fall within `threshold` seconds
    of each other.

    Always True when threshold is None (sync check disabled) or fewer than
    two of `names` have a recorded timestamp (nothing to compare).
    """
    if threshold is None:
        return True
    ts = [timestamps[n] for n in names if n in timestamps]
    if len(ts) < 2:
        return True
    return (max(ts) - min(ts)) <= threshold


TRIG_OPS = {"sin": np.sin, "cos": np.cos}


def strip_trig(spec: str):
    """Strip a sin(...)/cos(...) wrapper that encloses the WHOLE spec.

    Returns (math_op | None, inner_spec).

    Paren-balanced on purpose: "sin(a.x - b.x)" IS a wrapper (its opening
    paren closes on the last char), but "sin(a.x) - cos(b.x)" is NOT (the
    first paren closes early) and must fall through to arithmetic parsing
    instead of being corrupted by a naive startswith/endswith check.
    """
    for name, fn in TRIG_OPS.items():
        prefix = name + "("
        if spec.startswith(prefix) and spec.endswith(")"):
            depth = 0
            for i, ch in enumerate(spec[len(name):], start=len(name)):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        if i == len(spec) - 1:
                            return fn, spec[len(prefix):-1].strip()
                        break   # closes early -> not an enclosing wrapper
    return None, spec


def read_scalar(getter, msg, math_op) -> float:
    """Extract exactly one float from `msg` via `getter`, applying optional trig.

    np.asarray(...).reshape(-1)[0]: robust to numpy>=2.0, which raises on
    float() of a 1-element ndarray (e.g. a static input's `.data` field,
    always stored as an ndarray).
    """
    v = float(np.asarray(getter(msg)).reshape(-1)[0])
    return float(math_op(v)) if math_op is not None else v


BIN_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}

# "<spec> <op> <spec>": the spaces around the operator are required, which keeps it from matching dotted field paths
BINOP_RE = re.compile(r"^(.+?)\s+([+\-*/])\s+(.+)$")


def _parse_field(spec: str, math_op, tf_inputs, known_inputs):
    """'<input>.<field_path>' or '<input>.<field_path>[:]' -> (input_name, attrgetter, is_slice, math_op)."""
    if "." not in spec:
        raise ValueError(f"'{spec}' is not a valid field spec: expected '<input_name>.<field_path>'")
    input_name, raw_field = spec.split(".", 1)
    if known_inputs is not None and input_name not in known_inputs:
        raise ValueError(f"unknown input '{input_name}'")
    is_slice = raw_field.endswith("[:]")
    field_path = raw_field[:-3] if is_slice else raw_field
    if any(c in field_path for c in "+-*/"):
        raise ValueError(f"unexpected operator inside field path '{field_path}': arithmetic between two "
                         f"specs needs spaces around the operator, e.g. 'a.x - b.x'")
    if input_name in tf_inputs:
        field_path = "transform." + field_path
    return input_name, operator.attrgetter(field_path), is_slice, math_op


def parse_spec(spec: str, tf_inputs, known_inputs=None):
    """One stream / observation spec -> entry (input_name, payload, is_slice, math_op):

      field     : (input_name,  attrgetter, is_slice, math_op | None)
      zero:N    : ("__zero__",  N,          False,    None)
      a <op> b  : ("__binop__", (l_input, l_getter, l_op, op_func, r_input, r_getter, r_op), False, outer_op | None)

    The recorder and live inference both parse with this, so a stream has the width and values the obs will have.
    Raises ValueError; `known_inputs` (optional) rejects an input name not in it.
    """
    math_op, spec = strip_trig(spec.strip())
    if spec.startswith("zero:"):
        try:
            return ("__zero__", int(spec[5:]), False, None)
        except ValueError:
            raise ValueError("zero:N needs an integer N") from None
    m = BINOP_RE.match(spec)
    if m is None:
        return _parse_field(spec, math_op, tf_inputs, known_inputs)
    sides = []
    for side in (m.group(1).strip(), m.group(3).strip()):
        if BINOP_RE.match(side):
            raise ValueError(f"only one arithmetic operation per spec is supported (chained: '{side}')")
        side_op, side_spec = strip_trig(side)
        inp, getter, is_slice, side_op = _parse_field(side_spec, side_op, tf_inputs, known_inputs)
        if is_slice:
            raise ValueError("slice specs ('[:]') cannot be arithmetic operands: arithmetic is scalar-only")
        sides.append((inp, getter, side_op))
    (l_inp, l_get, l_op), (r_inp, r_get, r_op) = sides
    return ("__binop__", (l_inp, l_get, l_op, BIN_OPS[m.group(2)], r_inp, r_get, r_op), False, math_op)


def entry_inputs(entries) -> set:
    """The input names parsed entries read (both operands of an arithmetic one)."""
    names = set()
    for inp, payload, _, _ in entries:
        if inp == "__binop__":
            names.update((payload[0], payload[4]))
        elif inp != "__zero__":
            names.add(inp)
    return names


def evaluate_entries(entries, get) -> Optional[list]:
    """The float row of parsed entries. ``get(input_name)`` is that input's latest message or None.

    None if an input has no data; a field that cannot be read raises (the caller holds the last row).
    """
    row = []
    for inp, payload, is_slice, math_op in entries:
        if inp == "__zero__":
            row.extend([0.0] * payload)
        elif inp == "__binop__":
            l_inp, l_get, l_op, op_func, r_inp, r_get, r_op = payload
            l_msg, r_msg = get(l_inp), get(r_inp)
            if l_msg is None or r_msg is None:
                return None
            v = float(op_func(read_scalar(l_get, l_msg, l_op), read_scalar(r_get, r_msg, r_op)))
            row.append(float(math_op(v)) if math_op is not None else v)
        else:
            msg = get(inp)
            if msg is None:
                return None
            if is_slice:
                val = payload(msg)
                vals = list(val) if hasattr(val, "__iter__") else [val]
                row.extend(float(math_op(v)) if math_op is not None else float(v) for v in vals)
            else:
                row.append(read_scalar(payload, msg, math_op))
    return row


def median_rate(timestamps_ns) -> Optional[float]:
    """Sampling rate (Hz) of a recording from its per-sample timestamps in nanoseconds: 1 / median step.

    The median ignores a pause or an occasional late tick. None with fewer than two increasing samples.
    """
    steps = np.diff(np.asarray(timestamps_ns, dtype=np.float64))
    steps = steps[steps > 0]
    return 1e9 / float(np.median(steps)) if steps.size else None


RATE_TOLERANCE = 0.10  # a recording more than 10% off its configured rate has the wrong time base for training
