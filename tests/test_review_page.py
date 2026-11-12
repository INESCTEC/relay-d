"""Review page: recorded-demo list and playback settings."""

from types import SimpleNamespace

import h5py
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")
pytest.importorskip("cv2")
from PyQt5.QtWidgets import QWidget  # noqa: E402

from relay_d.acquisition.qt_app.ui.review import review_page as rp  # noqa: E402


@pytest.fixture
def make_page(qapp):
    pages = []

    def make(folder):
        page = rp.ReviewPage(SimpleNamespace(page_review=QWidget()))
        page.session_dir = str(folder)
        page.refresh_demo_list()
        pages.append(page)
        return page

    yield make
    for p in pages:
        p._close_current_file()


def test_the_robot_description_of_a_playback_is_remembered(make_page, tmp_path, monkeypatch):
    """The setting says 'last playback': a typed path counts. (The ROS-only modules are replaced by stubs.)"""
    import sys
    import types

    package = "relay_d.acquisition.qt_app.ui.review"
    resolver = types.ModuleType(f"{package}.robot_description_resolver")
    resolver.resolve_from_file = lambda path: "<robot/>"
    resolver.RobotDescriptionError = type("RobotDescriptionError", (Exception,), {})
    player = types.ModuleType(f"{package}.data_player")
    player.PlaybackState = SimpleNamespace(PAUSED="paused")
    monkeypatch.setitem(sys.modules, resolver.__name__, resolver)
    monkeypatch.setitem(sys.modules, player.__name__, player)

    with h5py.File(tmp_path / "demo.h5", "w") as f:
        f.create_group("data/demo_0")
    page = make_page(tmp_path)
    page.list_demos.setCurrentRow(0)
    saved, launched = {}, []
    monkeypatch.setattr(rp.config, "save_settings", lambda values: saved.update(values))
    monkeypatch.setattr(page, "_launch_playback", lambda demo, xml: launched.append((demo, xml)))
    page.edit_urdf_path.setText("/robots/arm.urdf")
    page._on_play_clicked()
    assert saved == {"review_last_robot_description": "/robots/arm.urdf"} and len(launched) == 1


def _two_camera_demo(path):
    """A recording like the sim's: two cameras, raw joint states with names, a Float64MultiArray input, a stream."""
    import json

    import numpy as np

    with h5py.File(path, "w") as f:
        d = f.create_group("data/demo_0")
        d["cam_wrist/images"] = np.zeros((5, 8, 8, 3), np.uint8)
        d["cam_static/images"] = np.full((5, 8, 8, 3), 255, np.uint8)
        d["joint_states/joint_positions"] = np.zeros((5, 2))
        d["joint_states"].attrs["joint_names"] = json.dumps(["j1", "j2"])
        action = d.create_dataset("cmd/data", (5,), dtype=h5py.vlen_dtype(np.float64))
        for i in range(5):
            action[i] = np.arange(3.0) + i
        d["streams/action"] = np.zeros((5, 3), np.float32)
        d["timestamps"] = np.arange(5)
        # a message without its own encoder, as message_encoding._encode_generic records it
        d["custom/data"] = [json.dumps({"type": "Float64MultiArray", "fields": {
            "_layout": "std_msgs.msg.MultiArrayLayout(dim=[], data_offset=0)",
            "_data": f"array('d', [{i}.0, 0.5])"}}).encode() for i in range(5)]


def test_every_camera_is_shown_side_by_side(make_page, tmp_path):
    _two_camera_demo(tmp_path / "demo.h5")
    page = make_page(tmp_path)
    page.list_demos.setCurrentRow(0)
    assert sorted(name for name, *_ in page._image_sources) == ["cam_static", "cam_wrist"]
    page.slider_scrub.setValue(4)
    assert all(not view.pixmap().isNull() for *_, view in page._image_sources)
    assert sorted(card.title() for *_, card, _ in page._image_sources) == ["cam_static  ·  8x8 px", "cam_wrist  ·  8x8 px"]
    assert page.label_frame_index.text() == "Frame 4 / 4   ·   0.00 s"  # timestamps of the test demo are 0..4 ns


def test_every_recorded_dataset_gets_a_channel(make_page, tmp_path):
    _two_camera_demo(tmp_path / "demo.h5")
    page = make_page(tmp_path)
    page.list_demos.setCurrentRow(0)
    labels = [b.text().split("\n")[0] for b in page.channel_buttons]
    assert labels == ["Images", "action", "cmd/data", "custom/data", "joint_states/joint_positions", "timestamps"]
    page._select_channel("dataset", "cmd/data")  # one array per sample, stacked to (5, 3)
    page._select_channel("dataset", "joint_states/joint_positions")  # legend from joint_names
    assert page.layout_display_graph.count() == 1



def test_a_json_message_is_decoded_plotted_and_readable(make_page, tmp_path):
    from PyQt5.QtWidgets import QListWidget, QTextEdit

    _two_camera_demo(tmp_path / "demo.h5")
    page = make_page(tmp_path)
    page.list_demos.setCurrentRow(0)
    page._select_channel("text", "custom/data")
    layout = page.layout_display_graph
    panes = layout.itemAt(0).widget()  # plots over the sample reader
    assert panes.widget(0).plotItem.titleLabel.text == "custom/data: data"  # the numeric field, plotted (5, 2)
    sample_list, sample_view = panes.findChild(QListWidget), panes.findChild(QTextEdit)
    sample_list.setCurrentRow(3)
    assert '"data": [3.0, 0.5]' in sample_view.toPlainText()  # number lists on one line
    assert '"layout": "std_msgs.msg.MultiArrayLayout(dim=[], data_offset=0)"' in sample_view.toPlainText()
    assert page._playback_cursor_line.value() == 3  # the plot cursor follows the picked sample


def test_decode_generic_and_numeric_fields():
    assert rp.decode_generic(b'{"type": "T", "fields": {"_data": "array(\'f\', [1.0, 2.5])", "_n": 3}}') == {
        "type": "T", "fields": {"data": [1.0, 2.5], "n": 3}}
    assert rp.decode_generic(b"not json") == "not json"
    fields = rp.numeric_fields([{"fields": {"data": [1.0, 2.0], "n": 3, "s": "x"}},
                                {"fields": {"data": [3.0, 4.0], "n": 4, "s": "y"}}])
    assert {k: v.shape for k, v in fields.items()} == {"data": (2, 2), "n": (2,)}
