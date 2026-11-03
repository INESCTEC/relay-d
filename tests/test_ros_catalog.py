from relay_d.acquisition.qt_app.ui.config.builder import ros_catalog as rc


class _Node:
    def get_topic_names_and_types(self):
        return [
            ("/joint_states", ["sensor_msgs/msg/JointState"]),
            ("/camera/depth", ["sensor_msgs/msg/Image"]),
            ("/tf", ["tf2_msgs/msg/TFMessage"]),
            ("/force", ["std_msgs/msg/Float64"]),
            ("/rosout", ["rcl_interfaces/msg/Log"]),
        ]


class _Buffer:
    def all_frames_as_yaml(self):
        return "base:\n  parent: world\neef:\n  parent: base\n"


def _msg(**fields):
    return type("M", (), {"get_fields_and_field_types": staticmethod(lambda: fields)})


MESSAGES = {
    "sensor_msgs/msg/JointState": _msg(header="std_msgs/Header", name="sequence<string>", position="sequence<double>", effort="sequence<double>"),
    "geometry_msgs/msg/PoseStamped": _msg(header="std_msgs/Header", pose="geometry_msgs/Pose"),
    "geometry_msgs/msg/Pose": _msg(position="geometry_msgs/Point", orientation="geometry_msgs/Quaternion"),
    "geometry_msgs/msg/Point": _msg(x="double", y="double", z="double"),
    "geometry_msgs/msg/Quaternion": _msg(x="double", y="double", z="double", w="double"),
    "x/msg/Mixed": _msg(frame="string", ok="boolean", cov="double[9]", covariance="double[36]", n="uint8"),
}


def test_topics_are_classified_and_hidden_ones_dropped():
    topics = {t.name: t.kind for t in rc.list_topics(_Node())}
    assert topics == {"/camera/depth": "image", "/force": "scalar", "/joint_states": "joint_state", "/tf": "tf"}
    assert rc.guess_modality("/camera/depth") == "depth" and rc.guess_modality("/camera/color") == "rgb"


def test_tf_frames_and_fields_and_specs():
    assert rc.tf_frames(_Buffer()) == ["base", "eef", "world"]
    load = MESSAGES.__getitem__
    assert rc.message_fields("sensor_msgs/msg/JointState", load) == [("position", True), ("effort", True)]
    assert [p for p, _ in rc.message_fields("geometry_msgs/msg/PoseStamped", load)] == [
        "pose.position.x", "pose.position.y", "pose.position.z",
        "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w",
    ]
    assert rc.message_fields("x/msg/Mixed", load) == [("cov", True), ("n", False)]
    assert rc.spec_for("joint_1", "position", True) == "joint_1.position[:]"
    assert rc.spec_for("joint_1", "position", True, 2) == "joint_1.position[2]"
