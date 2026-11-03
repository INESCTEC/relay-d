"""Review page with a LeRobot dataset: episode list, camera image, state / action curves, view-only."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")
pytest.importorskip("av")
pytest.importorskip("pyarrow")
pytest.importorskip("cv2")
from PyQt5.QtWidgets import QWidget  # noqa: E402

from relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer import LeRobotDatasetWriter  # noqa: E402
from relay_d.acquisition.qt_app.ui.review import review_page as rp  # noqa: E402

LENGTHS = [12, 7]


def _make_dataset(root, cameras=("cam_a", "cam_b")):
    with LeRobotDatasetWriter(root, fps=10, robot_type="arm") as writer:
        for ep, n in enumerate(LENGTHS):
            rng = np.random.default_rng(ep)
            images = {f"observation.images.{c}": rng.integers(0, 255, (n, 32, 48, 3), dtype=np.uint8) for c in cameras}
            writer.add_episode(rng.normal(size=(n, 4)).astype(np.float32), rng.normal(size=(n, 2)).astype(np.float32),
                               images, task="stack", state_names=["j1", "j2", "j3", "j4"])


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
        p._close_lerobot()


def test_lists_episodes_of_a_dataset_in_the_folder(make_page, tmp_path):
    _make_dataset(tmp_path / "lerobot_dataset_1")  # the layout Post-process writes: dataset inside the folder
    page = make_page(tmp_path)
    assert page.list_demos.count() == 2 and not page.combo_source.isVisibleTo(page.combo_source.parent() or page.combo_source)
    assert "Episode 0 · 12 frames · stack" in page.list_demos.item(0).text() and "PASS" in page.list_demos.item(0).text()
    assert "2 episodes" in page.text_validation.toPlainText() and "cam_a" in page.text_validation.toPlainText()
    assert not page.btn_delete.isEnabled() and not page.tab_widget.isTabEnabled(1)  # view-only, no ROS playback


def test_selecting_an_episode_shows_camera_frames_and_curves(make_page, tmp_path):
    _make_dataset(tmp_path / "ds")
    page = make_page(tmp_path / "ds")  # the dataset folder itself
    page.list_demos.setCurrentRow(1)
    assert [b.text().split("\n")[0] for b in page.channel_buttons] == ["Image", "observation.state", "action"]
    assert page.slider_scrub.maximum() == 6 and page.label_frame_index.text().startswith("Frame 0 / 6")
    assert page.label_image.pixmap() is not None and not page.label_image.pixmap().isNull()
    assert [page.combo_camera.itemText(i) for i in range(page.combo_camera.count())] == [
        "observation.images.cam_a", "observation.images.cam_b"]

    page.slider_scrub.setValue(4)
    assert page.label_frame_index.text().startswith("Frame 4 / 6")
    page.combo_camera.setCurrentIndex(1)  # another camera, same frame
    assert page.label_frame_index.text().startswith("Frame 4 / 6")

    page.channel_buttons[1].click()  # observation.state: 4 curves named after the joints, x axis in seconds
    plot = page.layout_display_graph.itemAt(0).widget()
    assert len(plot.getPlotItem().listDataItems()) == 4
    assert [i.name() for i in plot.getPlotItem().listDataItems()] == ["j1", "j2", "j3", "j4"]
    assert page._playback_cursor_line.value() == pytest.approx(0.4)
    page.slider_scrub.setValue(6)  # the cursor follows the slider
    assert page._playback_cursor_line.value() == pytest.approx(0.6)


def test_long_episodes_are_decimated(make_page, tmp_path, monkeypatch):
    _make_dataset(tmp_path / "ds", cameras=("c",))
    real_get = rp.user_settings.get
    monkeypatch.setattr(rp.user_settings, "get", lambda key, default=None: 5 if key == "review_plot_max_points" else real_get(key, default))
    page = make_page(tmp_path / "ds")
    page.list_demos.setCurrentRow(0)
    page.channel_buttons[1].click()
    item = page.layout_display_graph.itemAt(0).widget().getPlotItem().listDataItems()[0]
    assert len(item.xData) <= 5 + 1


def test_h5_demos_and_a_dataset_in_the_same_folder_are_two_sources(make_page, tmp_path):
    _make_dataset(tmp_path / "ds")
    with h5py.File(tmp_path / "demo.h5", "w") as f:
        f.create_group("data/demo_0")
    page = make_page(tmp_path)
    assert [page.combo_source.itemText(i) for i in range(2)] == ["Recorded demos (.h5)", "LeRobot: ds"]
    assert page.btn_delete.isEnabled()  # the first source is the h5 demos
    page.combo_source.setCurrentIndex(1)
    assert page.list_demos.count() == 2 and not page.btn_delete.isEnabled()
    page.combo_source.setCurrentIndex(0)
    assert page.btn_delete.isEnabled() and page.tab_widget.isTabEnabled(1)


def test_a_broken_dataset_is_reported_not_raised(make_page, tmp_path):
    (tmp_path / "ds" / "meta").mkdir(parents=True)
    (tmp_path / "ds" / "meta" / "info.json").write_text("{}")
    page = make_page(tmp_path / "ds")
    assert page.list_demos.count() == 0 and page.text_validation.toPlainText()


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
