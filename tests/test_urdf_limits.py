import pytest

from relay_d.acquisition.qt_app.ui.config.builder.urdf_limits import parse_joint_limits

URDF = """<robot name="r">
  <joint name="fixed" type="fixed"/>
  <joint name="j1" type="revolute"><limit lower="-1.5" upper="1.5" velocity="2.0" effort="30"/></joint>
  <joint name="j2" type="prismatic"><limit lower="0" upper="0.1" velocity="0.5" effort="10"/></joint>
</robot>"""


def test_parse_joint_limits_order_filter_and_errors():
    assert [j["name"] for j in parse_joint_limits(URDF)] == ["j1", "j2"]
    only = parse_joint_limits(URDF, ["j2", "missing", "j1"])
    assert [j["name"] for j in only] == ["j2", "j1"]
    assert only[1]["lower"] == -1.5 and only[1]["effort"] == 30.0
    with pytest.raises(ValueError):
        parse_joint_limits("<nope/>")
    with pytest.raises(ValueError):
        parse_joint_limits("not xml")


def test_limits_for_position_velocity_and_errors():
    from relay_d.acquisition.qt_app.ui.config.builder.urdf_limits import limits_for

    joints = parse_joint_limits(URDF)
    assert limits_for("position", joints, ["j2", "j1"]) == [{"lower_lim": 0.0, "upper_lim": 0.1}, {"lower_lim": -1.5, "upper_lim": 1.5}]
    assert limits_for("velocity", joints, ["j1"]) == [{"lower_lim": -2.0, "upper_lim": 2.0}]
    for args in (("position", ["nope"]), ("acceleration", ["j1"])):
        with pytest.raises(ValueError):
            limits_for(args[0], joints, args[1])
