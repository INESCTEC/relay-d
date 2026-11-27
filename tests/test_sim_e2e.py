"""Opt-in end-to-end simulation pipeline test using relayd-sim (ManiSkill 3 + ROS 2 + CUDA).

Skipped during regular ``pytest`` runs so unit tests stay fast and GPU/simulator-free.
Run explicitly with:

    RELAYD_SIM_E2E=1 pytest -v -s tests/test_sim_e2e.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sim_e2e_test.py"


def _load_e2e_module():
    spec = importlib.util.spec_from_file_location("relayd_sim_e2e_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.sim
@pytest.mark.skipif(
    os.environ.get("RELAYD_SIM_E2E") != "1",
    reason="Set RELAYD_SIM_E2E=1 to run the full ManiSkill 3 record -> convert -> train -> deploy pipeline",
)
def test_relayd_sim_full_pipeline(tmp_path):
    pytest.importorskip("rclpy")
    mod = _load_e2e_module()
    sim_repo = Path(os.environ.get("RELAYD_SIM_REPO", mod._DEFAULT_SIM_REPO))
    sim_py = Path(os.environ.get("RELAYD_SIM_PYTHON", mod._DEFAULT_SIM_PYTHON))
    if not (sim_repo / "maniskill_bridge.py").is_file() or not sim_py.is_file():
        pytest.skip(f"relayd-sim not installed at {sim_repo} / {sim_py}")

    backends_env = os.environ.get("RELAYD_SIM_BACKENDS", "both")
    backends = ["lerobot", "robomimic"] if backends_env == "both" else [b.strip() for b in backends_env.split(",")]
    cfg = mod.E2EConfig(
        sim_repo=sim_repo,
        sim_python=sim_py,
        backends=backends,
        work_dir=tmp_path / "sim_e2e",
    )
    summary = mod.run_e2e(cfg)
    for backend in backends:
        assert summary["deployed"][backend]["returncode"] == 0
        assert summary["deployed"][backend]["n_cmds"] > 0
