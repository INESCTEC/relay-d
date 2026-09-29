"""Post-hoc viability checks for recorded demo .h5 files.

Pure h5py/numpy/json logic - no PyQt5, no rclpy - so it can validate any demo
file on disk, whether or not a live DataRecorder/RecordPage is involved.
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum

import h5py
import numpy as np


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


_SEVERITY = {CheckStatus.PASS: 0, CheckStatus.WARN: 1, CheckStatus.FAIL: 2}

# Groups under data/<demo_name>/ that are not "raw input" containers and
# should be skipped when scanning for per-input sample-count/joint checks.
_NON_INPUT_GROUPS = {"streams", "metadata", "gripper_states"}

# Dataset keys (in priority order) used to determine an input group's sample count.
_PRIMARY_DATASET_KEYS = (
    "joint_positions",
    "positions",
    "translations",
    "images",
    "compressed_data",
)


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str


@dataclass
class DemoValidationResult:
    file_path: str
    demo_name: str = ""
    overall_status: CheckStatus = CheckStatus.PASS
    checks: list = field(default_factory=list)
    samples: int = 0
    duration: float = 0.0

    def summary_text(self) -> str:
        return "\n".join(
            f"[{c.status.value.upper()}] {c.name}: {c.message}" for c in self.checks
        )


class DemoValidator:
    """Runs a fixed set of viability checks against a single demo's HDF5 group."""

    DEFAULT_THRESHOLDS = {
        "min_samples": 10,
        "min_duration_sec": 0.5,
        "max_gap_factor": 3.0,  # sample gap > 3x expected interval -> warn
        "max_gap_fail_factor": 8.0,  # -> fail
        "max_gap_pct_warn": 0.05,  # 5% of samples with anomalous gaps -> warn
        "max_gap_pct_fail": 0.20,  # 20% -> fail
        "max_fill_pct_warn": 0.20,  # per-container gap-fill % -> warn
        "max_fill_pct_fail": 0.50,  # -> fail
    }

    def __init__(self, thresholds: dict = None):
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def validate(self, h5_path: str) -> DemoValidationResult:
        result = DemoValidationResult(file_path=h5_path)

        if not os.path.isfile(h5_path):
            result.checks.append(
                CheckResult("file_exists", CheckStatus.FAIL, "File not found")
            )
            result.overall_status = CheckStatus.FAIL
            return result

        try:
            with h5py.File(h5_path, "r") as f:
                demo_name, demo_group = self._find_demo_group(f)
                if demo_group is None:
                    result.checks.append(
                        CheckResult(
                            "structure",
                            CheckStatus.FAIL,
                            "No data/<demo_name> group found",
                        )
                    )
                    result.overall_status = CheckStatus.FAIL
                    return result

                result.demo_name = demo_name
                self._check_min_length(demo_group, result)
                self._check_timestamps_and_gaps(demo_group, result)
                self._check_sample_counts(demo_group, result)
                self._check_joint_counts(demo_group, result)
                self._check_gap_fill(demo_group, result)
        except Exception as e:
            result.checks.append(CheckResult("read_error", CheckStatus.FAIL, str(e)))

        result.overall_status = self._aggregate(result.checks)
        return result

    def _find_demo_group(self, f):
        if "data" not in f:
            return "", None
        names = list(f["data"].keys())
        if not names:
            return "", None
        demo_name = names[0]
        return demo_name, f["data"][demo_name]

    def _aggregate(self, checks) -> CheckStatus:
        worst = CheckStatus.PASS
        for c in checks:
            if _SEVERITY[c.status] > _SEVERITY[worst]:
                worst = c.status
        return worst

    def _get_metadata_attrs(self, grp):
        meta = grp.get("metadata")
        return meta.attrs if meta is not None else {}

    def _check_min_length(self, grp, result):
        n = grp["timestamps"].shape[0] if "timestamps" in grp else 0
        result.samples = n

        attrs = self._get_metadata_attrs(grp)
        duration = float(attrs.get("duration", 0.0))
        result.duration = duration

        if n == 0:
            result.checks.append(
                CheckResult("min_length", CheckStatus.FAIL, "Demo has zero samples")
            )
        elif n < self.thresholds["min_samples"] or duration < self.thresholds["min_duration_sec"]:
            result.checks.append(
                CheckResult(
                    "min_length",
                    CheckStatus.WARN,
                    f"Very short demo: {n} samples, {duration:.2f}s",
                )
            )
        else:
            result.checks.append(
                CheckResult(
                    "min_length", CheckStatus.PASS, f"{n} samples, {duration:.2f}s"
                )
            )

    def _check_timestamps_and_gaps(self, grp, result):
        if "timestamps" not in grp or grp["timestamps"].shape[0] < 2:
            result.checks.append(
                CheckResult(
                    "timestamp_gaps",
                    CheckStatus.WARN,
                    "Not enough timestamps to check gaps",
                )
            )
            return

        ts = grp["timestamps"][:].astype(np.float64) * 1e-9
        diffs = np.diff(ts)
        expected = float(np.median(diffs)) if len(diffs) else 0.0

        if expected <= 0:
            result.checks.append(
                CheckResult(
                    "timestamp_gaps",
                    CheckStatus.WARN,
                    "Could not infer expected sampling interval",
                )
            )
        else:
            fail_mask = diffs > expected * self.thresholds["max_gap_fail_factor"]
            warn_mask = diffs > expected * self.thresholds["max_gap_factor"]
            pct_warn = float(np.mean(warn_mask))

            if pct_warn == 0:
                result.checks.append(
                    CheckResult(
                        "timestamp_gaps", CheckStatus.PASS, "No abnormal timestamp gaps"
                    )
                )
            elif np.any(fail_mask) or pct_warn > self.thresholds["max_gap_pct_fail"]:
                result.checks.append(
                    CheckResult(
                        "timestamp_gaps",
                        CheckStatus.FAIL,
                        f"{pct_warn * 100:.1f}% of samples have gaps > "
                        f"{self.thresholds['max_gap_fail_factor']}x expected interval",
                    )
                )
            elif pct_warn > self.thresholds["max_gap_pct_warn"]:
                result.checks.append(
                    CheckResult(
                        "timestamp_gaps",
                        CheckStatus.WARN,
                        f"{pct_warn * 100:.1f}% of samples have gaps > "
                        f"{self.thresholds['max_gap_factor']}x expected interval",
                    )
                )
            else:
                result.checks.append(
                    CheckResult(
                        "timestamp_gaps", CheckStatus.PASS, "Gaps within tolerance"
                    )
                )

        if "timestamp_spreads" in grp:
            attrs = self._get_metadata_attrs(grp)
            tol = attrs.get("sync_tolerance_sec")
            if tol not in (None, "", "None"):
                try:
                    tol = float(tol)
                except (TypeError, ValueError):
                    tol = None
            else:
                tol = None

            if tol:
                spreads = grp["timestamp_spreads"][:]
                over = float(np.mean(spreads > tol))
                if over > self.thresholds["max_gap_pct_fail"]:
                    result.checks.append(
                        CheckResult(
                            "sync_spread",
                            CheckStatus.FAIL,
                            f"{over * 100:.1f}% of samples exceed sync tolerance {tol}s",
                        )
                    )
                elif over > self.thresholds["max_gap_pct_warn"]:
                    result.checks.append(
                        CheckResult(
                            "sync_spread",
                            CheckStatus.WARN,
                            f"{over * 100:.1f}% of samples exceed sync tolerance {tol}s",
                        )
                    )
                else:
                    result.checks.append(
                        CheckResult(
                            "sync_spread", CheckStatus.PASS, "Sync spread within tolerance"
                        )
                    )

    def _check_sample_counts(self, grp, result):
        n = grp["timestamps"].shape[0] if "timestamps" in grp else 0
        mismatches = []

        if "streams" in grp:
            for name, ds in grp["streams"].items():
                if ds.shape[0] != n:
                    mismatches.append(f"streams/{name}: {ds.shape[0]} rows vs {n} timestamps")

        for name, item in grp.items():
            if name in _NON_INPUT_GROUPS or not isinstance(item, h5py.Group):
                continue
            for key in _PRIMARY_DATASET_KEYS:
                if key in item:
                    rows = item[key].shape[0]
                    if rows != n:
                        mismatches.append(f"{name}/{key}: {rows} rows vs {n} timestamps")
                    break

        if mismatches:
            result.checks.append(
                CheckResult(
                    "sample_count_consistency", CheckStatus.FAIL, "; ".join(mismatches)
                )
            )
        else:
            result.checks.append(
                CheckResult(
                    "sample_count_consistency",
                    CheckStatus.PASS,
                    "All streams/inputs match timestamp count",
                )
            )

    def _check_joint_counts(self, grp, result):
        issues = []
        for name, item in grp.items():
            if not isinstance(item, h5py.Group) or "joint_names" not in item.attrs:
                continue
            try:
                names = json.loads(item.attrs["joint_names"])
            except Exception:
                continue
            for key in ("joint_positions", "joint_velocities", "joint_efforts"):
                if key in item and item[key].ndim == 2 and item[key].shape[1] != len(names):
                    issues.append(
                        f"{name}/{key}: {item[key].shape[1]} cols vs {len(names)} joint_names"
                    )

        if issues:
            result.checks.append(
                CheckResult("joint_count_consistency", CheckStatus.FAIL, "; ".join(issues))
            )
        else:
            result.checks.append(
                CheckResult(
                    "joint_count_consistency",
                    CheckStatus.PASS,
                    "Joint counts consistent with joint_names",
                )
            )

    def _check_gap_fill(self, grp, result):
        attrs = self._get_metadata_attrs(grp)
        raw_summary = attrs.get("fill_count_summary")

        if not raw_summary:
            result.checks.append(
                CheckResult(
                    "gap_fill_pct", CheckStatus.PASS, "No samples were gap-filled"
                )
            )
            return

        try:
            summary = json.loads(raw_summary)
        except Exception:
            summary = {}

        n = grp["timestamps"].shape[0] if "timestamps" in grp else 0
        if not summary or n == 0:
            result.checks.append(
                CheckResult(
                    "gap_fill_pct", CheckStatus.PASS, "No samples were gap-filled"
                )
            )
            return

        worst_name, worst_count = max(summary.items(), key=lambda kv: kv[1])
        worst_pct = worst_count / n

        if worst_pct > self.thresholds["max_fill_pct_fail"]:
            result.checks.append(
                CheckResult(
                    "gap_fill_pct",
                    CheckStatus.FAIL,
                    f"'{worst_name}' is {worst_pct * 100:.1f}% gap-filled by hold",
                )
            )
        elif worst_pct > self.thresholds["max_fill_pct_warn"]:
            result.checks.append(
                CheckResult(
                    "gap_fill_pct",
                    CheckStatus.WARN,
                    f"'{worst_name}' is {worst_pct * 100:.1f}% gap-filled by hold",
                )
            )
        else:
            result.checks.append(
                CheckResult(
                    "gap_fill_pct",
                    CheckStatus.PASS,
                    f"Gap-fill within tolerance (worst: '{worst_name}' {worst_pct * 100:.1f}%)",
                )
            )
