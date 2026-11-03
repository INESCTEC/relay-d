"""Config builder page: build a config from a fake ROS graph, save it, and load it like a hand-written file (no ROS, no display)."""

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt  # noqa: E402

from relay_d.acquisition.qt_app.ui.config.builder import builder_page as bp  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.offline_config import load_offline_config  # noqa: E402
from test_ros_catalog import MESSAGES, _Buffer, _Node  # noqa: E402

URDF = '<robot name="r"><joint name="j1" type="revolute"><limit lower="-1" upper="1" velocity="2" effort="3"/></joint>' \
       '<joint name="j2" type="revolute"><limit lower="-2" upper="2" velocity="2" effort="3"/></joint></robot>'


@pytest.fixture
def page(qapp, monkeypatch):
    boxes = []
    for name in ("info", "error", "warning"):
        monkeypatch.setattr(bp.CustomMessageBox, name, staticmethod(lambda *a, _n=name, **k: boxes.append((_n, a))))
    used = []
    p = bp.BuilderPage(lambda: _Node(), lambda topic: _Buffer(), used.append, lambda: None, loader=MESSAGES.__getitem__)
    p.boxes, p.used = boxes, used
    return p


def _tick(page, *paths):
    tree = page.tree_fields
    for i in range(tree.topLevelItemCount()):
        if tree.topLevelItem(i).text(0) in paths:
            tree.topLevelItem(i).setCheckState(0, Qt.Checked)


def test_build_save_and_load_like_a_handwritten_config(page, tmp_path, monkeypatch):
    assert [t.name for t in page._topics] == ["/camera/depth", "/force", "/joint_states", "/tf"]  # sorted, /rosout hidden
    page.add_topic_input("/joint_states", "sensor_msgs/msg/JointState", name="joint_1")
    page.draft.inputs["joint_1"].joint_names = ["j1", "j2"]
    page.add_topic_input("/camera/depth", "sensor_msgs/msg/Image", name="image_depth")
    page.add_tf_input("base", "eef", "/tf", name="tf_eef")
    assert page.draft.inputs["image_depth"].modality == "depth" and page.draft.inputs["tf_eef"].tf_topic == ""

    page.refresh()
    page.combo_obs_input.setCurrentText("joint_1")
    _tick(page, "position")
    assert page._specs() == ["joint_1.position[:]"]
    page.edit_stream_name.setText("robot_joint_pos")
    page.check_normalize.setChecked(True)
    assert page.table_limits.rowCount() == 2  # one row per joint
    page._fill_limits(URDF)  # what "Fill limits from URDF" does with the file's text
    assert page._limits() == [{"lower_lim": -1.0, "upper_lim": 1.0}, {"lower_lim": -2.0, "upper_lim": 2.0}]
    page.add_stream()

    page.combo_obs_input.setCurrentText("tf_eef")
    _tick(page, "translation.x", "translation.y", "translation.z")
    page.edit_stream_name.setText("eef_pos")
    page.add_stream()
    page.combo_obs_input.setCurrentText("image_depth")
    page.edit_camera_key.setText("agentview_depth")
    page.add_camera()
    assert page.draft.cameras == {"agentview_depth": ("image_depth", "images")}

    page.refresh_actions()
    page.table_actions.item(0, 1).setCheckState(Qt.Checked)
    assert page.draft.actions == ["robot_joint_pos"] and "2 values" in page.label_action_width.text()

    page.refresh_review()
    assert page.btn_save.isEnabled(), [str(p) for p in page.problems()]
    path = tmp_path / "built.yaml"
    monkeypatch.setattr(bp.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    page.save_and_use()
    assert page.used == [str(path)]
    parser, problems = load_offline_config(str(path))
    assert parser is not None and problems == []
    assert list(parser.get_streams()) == ["robot_joint_pos", "eef_pos"]  # order kept


def test_problems_block_saving(page):
    page.add_topic_input("/joint_states", "sensor_msgs/msg/JointState", name="joint_1")
    page.draft.streams["q"] = bp.StreamDraft(name="q", specs=["missing.position[:]"])
    page.refresh_review()
    assert not page.btn_save.isEnabled() and any("missing" in p.message for p in page.problems())


def test_edit_loaded_config_and_bad_input_rejected(page):
    assert page.add_topic_input("joint_states", "sensor_msgs/msg/JointState", name="x") is None  # no leading slash
    page._current_config = lambda: {"config": {"dataset_name": "d", "input": {"joint_1": {"topic": "/js"}}, "output": {}}}
    page.load_current()
    assert list(page.draft.inputs) == ["joint_1"] and page.edit_dataset.text() == "d"
