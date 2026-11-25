"""DemoValidator, the check that runs after every recording."""

import h5py
import numpy as np
import pytest

from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus, DemoValidator


def _demo(path, rate, configured=30, n=60):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("timestamps", data=(np.arange(n) * 1e9 / rate).astype(np.int64))
        g.create_group("metadata").attrs["recording_frequency"] = configured
    return str(path)


@pytest.mark.parametrize("rate, status", [(30.0, CheckStatus.PASS), (28.0, CheckStatus.PASS), (21.0, CheckStatus.FAIL)])
def test_recorded_rate_is_checked_against_the_configured_one(tmp_path, rate, status):
    checks = {c.name: c for c in DemoValidator().validate(_demo(tmp_path / "d.h5", rate)).checks}
    assert checks["sample_rate"].status == status
    if status == CheckStatus.FAIL:
        assert "21.0 Hz" in checks["sample_rate"].message  # the operator is told what the PC actually managed


def _checks(path):
    return {c.name: c for c in DemoValidator().validate(path).checks}


def test_an_input_one_row_short_fails_the_sample_count_check(tmp_path):
    path = _demo(tmp_path / "d.h5", 30.0)
    with h5py.File(path, "a") as f:
        f["data/demo_0"].create_group("arm").create_dataset("joint_positions", data=np.zeros((59, 2)))
    check = _checks(path)["sample_count_consistency"]
    assert check.status == CheckStatus.FAIL and "arm/joint_positions: 59 rows vs 60" in check.message


def test_one_tenfold_timestamp_gap_fails(tmp_path):
    path = _demo(tmp_path / "d.h5", 30.0)
    with h5py.File(path, "a") as f:
        ts = f["data/demo_0/timestamps"]
        step = ts[1] - ts[0]
        ts[30:] += 9 * step  # one gap of 10 intervals
    assert _checks(path)["timestamp_gaps"].status == CheckStatus.FAIL


def test_samples_over_the_sync_tolerance_fail_the_spread_check(tmp_path):
    path = _demo(tmp_path / "d.h5", 30.0)
    with h5py.File(path, "a") as f:
        f["data/demo_0/metadata"].attrs["sync_tolerance_sec"] = 0.01
        f["data/demo_0"].create_dataset("timestamp_spreads", data=np.r_[np.full(15, 0.05), np.zeros(45)])  # 25% over
    assert _checks(path)["sync_spread"].status == CheckStatus.FAIL


def test_an_input_mostly_held_fails_the_gap_fill_check(tmp_path):
    path = _demo(tmp_path / "d.h5", 30.0)
    with h5py.File(path, "a") as f:
        f["data/demo_0/metadata"].attrs["fill_count_summary"] = '{"cam": 31, "arm": 2}'  # cam: 52% of 60
    check = _checks(path)["gap_fill_pct"]
    assert check.status == CheckStatus.FAIL and "'cam'" in check.message
