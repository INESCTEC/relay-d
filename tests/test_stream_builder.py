"""StreamBuilder: one spec grammar with live inference, tick alignment between streams."""

from types import SimpleNamespace

import pytest

from relay_d.acquisition.qt_app.utils.stream_builder import StreamBuilder


def _builder(streams, inputs=None, joint_names=None):
    data = [{"name": n, "joint_names": j} for n, j in (joint_names or {}).items()]
    return StreamBuilder(SimpleNamespace(yaml_data={"config": {"input": inputs or {}}},
                                         get_input_data=lambda: data, get_streams=lambda: streams))


def test_an_error_half_way_does_not_leave_earlier_streams_a_row_ahead():
    class Broken:
        @property
        def v(self):
            raise RuntimeError("driver bug")

    sb = _builder({"a": ["x.v"], "b": ["y.v"]})
    sb.evaluate({"x": SimpleNamespace(v=1.0), "y": SimpleNamespace(v=2.0)})
    with pytest.raises(RuntimeError):
        sb.evaluate({"x": SimpleNamespace(v=3.0), "y": Broken()})
    sb.evaluate({"x": SimpleNamespace(v=4.0), "y": SimpleNamespace(v=5.0)})
    out = sb.finalize()
    assert out["a"][:, 0].tolist() == [1.0, 4.0] and out["b"][:, 0].tolist() == [2.0, 5.0]  # was a: 3 rows, b: 2


def test_a_message_missing_a_joint_holds_the_last_full_row_as_live_inference():
    sb = _builder({"q": ["js.position[:]"]}, joint_names={"js": ["j1", "j2"]})
    hdr = SimpleNamespace(stamp=None)
    full = SimpleNamespace(header=hdr, name=["j1", "j2"], position=[1.0, 2.0], velocity=[], effort=[])
    gripper = SimpleNamespace(header=hdr, name=["finger", "j1"], position=[0.5, 9.0], velocity=[], effort=[])
    sb.evaluate({"js": gripper})  # before any full message: no row yet
    sb.evaluate({"js": full})
    sb.evaluate({"js": gripper})
    assert sb.finalize()["q"].tolist() == [[1.0, 2.0]] * 3  # was a 1-wide row [9.0]: the dataset misaligned
    assert sb.apply_joint_filter("js", gripper).position == [1.0, 2.0]  # the raw input recording holds it too


def test_spec_errors_are_collected_not_raised():
    sb = _builder({"s": ["nodot", "a.x+b.x", "a.x - b.x - c.x", "zero:n", "a.p[:] - b.x", "a.x"]})
    errors = sb.get_build_errors()
    assert len(errors) == 5 and all("skipped" in e for e in errors)
    assert sb.referenced_input_names() == {"a"}
