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
