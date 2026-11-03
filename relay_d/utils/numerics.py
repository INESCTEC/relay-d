"""Maths shared by training-data conversion and live inference.

Both sides must apply the same rule, otherwise the policy sees different numbers at run time
than it was trained on. Keep them here, not in two copies.
"""

from typing import Optional

import numpy as np


def normalize(x, lower: float, upper: float, norm_min: float = -1.0, norm_max: float = 1.0):
    """Linearly map ``x`` from [lower, upper] to [norm_min, norm_max].

    A degenerate range (upper <= lower) leaves ``x`` unchanged. ``x`` may be a scalar or an array.
    """
    span = upper - lower
    if span <= 0:
        return x
    return (x - lower) / span * (norm_max - norm_min) + norm_min


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


def median_rate(timestamps_ns) -> Optional[float]:
    """Sampling rate (Hz) of a recording from its per-sample timestamps in nanoseconds: 1 / median step.

    The median ignores a pause or an occasional late tick. None with fewer than two increasing samples.
    """
    steps = np.diff(np.asarray(timestamps_ns, dtype=np.float64))
    steps = steps[steps > 0]
    return 1e9 / float(np.median(steps)) if steps.size else None


RATE_TOLERANCE = 0.10  # a recording more than 10% off its configured rate has the wrong time base for training
