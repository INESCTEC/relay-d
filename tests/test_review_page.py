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
