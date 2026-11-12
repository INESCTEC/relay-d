"""Maths shared by training-data conversion and live inference, and the converter rules built on it."""

import h5py
import numpy as np
import pytest

from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import (
    YAMLDrivenConverter,
)
from relay_d.utils.numerics import (
    inputs_in_sync,
    quat_to_euler,
    quat_to_rot_matrix,
    read_scalar,
    strip_trig,
)


def test_quat_to_euler_known_rotations():
    s = np.sqrt(0.5)
    q = np.array([[0, 0, 0, 1], [0, 0, s, s], [s, 0, 0, s]], dtype=np.float64)  # id, yaw 90, roll 90
    rpy = quat_to_euler(q)
    assert np.allclose(rpy, [[0, 0, 0], [0, 0, np.pi / 2], [np.pi / 2, 0, 0]])


def test_inputs_in_sync():
    ts = {"a": 1.00, "b": 1.04, "c": 2.0}
    assert inputs_in_sync(["a", "b"], ts, 0.05)
    assert not inputs_in_sync(["a", "b"], ts, 0.03)
    assert not inputs_in_sync(["a", "b", "c"], ts, 0.05)
    assert inputs_in_sync(["a", "c"], ts, None)  # check disabled
    assert inputs_in_sync(["a", "missing"], ts, 0.0)  # fewer than two timestamps: nothing to compare


def test_quat_to_rot_matrix_known_rotation():
    s = np.sqrt(0.5)
    assert np.allclose(quat_to_rot_matrix(0, 0, 0, 1), np.eye(3))
    assert np.allclose(quat_to_rot_matrix(0, 0, s, s), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # yaw 90


def test_strip_trig_only_strips_a_wrapper_around_the_whole_spec():
    fn, inner = strip_trig("sin(a.x - b.x)")
    assert fn is np.sin and inner == "a.x - b.x"
    assert strip_trig("sin(a.x) - cos(b.x)") == (None, "sin(a.x) - cos(b.x)")  # first paren closes early
    assert strip_trig("a.x") == (None, "a.x")


def test_read_scalar_takes_one_float_and_applies_trig():
    from types import SimpleNamespace

    msg = SimpleNamespace(data=np.array([0.0]))  # a 1-element array: float() of it fails on numpy 2
    assert read_scalar(lambda m: m.data, msg, None) == 0.0
    assert read_scalar(lambda m: m.data, msg, np.cos) == 1.0


@pytest.fixture
def converter():
    return YAMLDrivenConverter(PostprocessConfigLoader())


def _mask(converter, n, train, val):
    with h5py.File("m.h5", "w", driver="core", backing_store=False) as f:
        g = f.create_group("mask")
        ok = converter._create_data_mask(g, [f"demo_{i}" for i in range(n)], train, val)
        return ok, {k: len(g[k]) for k in g}


def test_split_mask_normal_and_few_demos(converter):
    assert _mask(converter, 10, 0.8, 0.2) == (True, {"train": 8, "valid": 2})
    ok, sizes = _mask(converter, 2, 0.8, 0.1)
    assert ok and sizes["train"] >= 1  # int(2 * 0.8) would be 1 here, int(1 * 0.8) would be 0


def test_split_mask_trains_on_the_leftovers(converter):
    assert _mask(converter, 10, 0.8, 0.1) == (True, {"train": 9, "valid": 1})  # no unused "eval" split
    assert _mask(converter, 100, 0.29, 0.71)[1] == {"train": 29, "valid": 71}  # int(100 * 0.29) is 28


def test_split_mask_rejects_bad_fractions(converter):
    assert _mask(converter, 10, 0.9, 0.2)[0] is False
    assert "more than 1" in converter.last_error
    assert _mask(converter, 10, -0.1, 0.1)[0] is False
    assert _mask(converter, 10, "0.8", 0.1)[0] is False
    assert _mask(converter, 10, 0.0, 0.5)[0] is False  # no train mask: robomimic could not start


def test_zero_quaternion_in_an_rpy_stream_fails_instead_of_writing_nan():
    from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import _quat_to_rpy

    good = np.array([[0, 0, 0, 1.0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=np.float32)
    assert np.allclose(_quat_to_rpy(good, "eef"), [[0, 0, 0], [0, 0, np.pi / 2]], atol=1e-6)
    with pytest.raises(ValueError, match=r"stream 'eef' has 1 row"):
        _quat_to_rpy(np.array([[0, 0, 0, 1.0], [0, 0, 0, 0.0]], dtype=np.float32), "eef")


def test_action_is_the_next_step_target():
    from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import _action_part

    x = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    raw = {"q": ["j.position[:]"]}
    # absolute: value at t+1 (the value at t is where the robot already is); the last step holds
    assert _action_part(x, "q", raw, []).ravel().tolist() == [1.0, 3.0, 3.0]
    # delta: change from t to t+1; the last step is zero
    assert _action_part(x, "q", raw, ["q"]).ravel().tolist() == [1.0, 2.0, 0.0]


def test_split_mask_sorts_demo_numbers_naturally(converter):
    with h5py.File("d.h5", "w", driver="core", backing_store=False) as f:
        assert converter._create_data_mask(f.create_group("mask"), [f"demo_{i}" for i in range(11)], 0.9, 0.1)
        assert [n.decode() for n in f["mask/valid"][:]] == ["demo_10"]  # the last demo, not demo_9


def test_median_rate_ignores_a_pause_and_needs_two_samples():
    from relay_d.utils.numerics import median_rate

    ts = [0, 50e6, 100e6, 150e6, 2_150e6, 2_200e6]  # 20 Hz with a 2 s pause in the middle
    assert median_rate(ts) == pytest.approx(20.0)
    assert median_rate([5]) is None and median_rate([]) is None
