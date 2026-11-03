"""LeRobot reader used by the Review page: episodes, curves, camera frames, checks (no Qt, no ROS)."""

import shutil
import types

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("pyarrow")
pytest.importorskip("cv2")
import pyarrow.parquet as pq  # noqa: E402

from relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer import LeRobotDatasetWriter  # noqa: E402
from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus  # noqa: E402
from relay_d.acquisition.qt_app.ui.review.lerobot_reader import (  # noqa: E402
    LeRobotDataset,
    LeRobotError,
    is_lerobot_dataset,
)

FPS = 10
LENGTHS = [12, 7, 9]


def _frames(n, episode):
    """BGR frames (writer default): B=200, G grows with the frame number, R=30."""
    frames = np.zeros((n, 32, 48, 3), np.uint8)
    for i in range(n):
        frames[i, :, :, 0] = 200
        frames[i, :, :, 1] = 20 + 8 * i + 40 * episode
        frames[i, :, :, 2] = 30
    return frames


def _make(root, with_camera=True):
    with LeRobotDatasetWriter(root, fps=FPS, robot_type="arm") as writer:
        for ep, n in enumerate(LENGTHS):
            rng = np.random.default_rng(ep)
            writer.add_episode(
                rng.normal(size=(n, 5)).astype(np.float32), rng.normal(size=(n, 3)).astype(np.float32),
                {"observation.images.cam": _frames(n, ep)} if with_camera else {},
                task="pick the cube", state_names=["a", "b", "c", "d", "e"],
            )
    return root


@pytest.fixture
def ds(tmp_path):
    reader = LeRobotDataset(_make(tmp_path / "ds"))
    yield reader
    reader.close()


def test_metadata_and_episodes(ds, tmp_path):
    assert is_lerobot_dataset(tmp_path / "ds") and not is_lerobot_dataset(tmp_path)
    assert ds.fps == FPS and ds.camera_keys == ["observation.images.cam"]
    assert ds.names("observation.state") == ["a", "b", "c", "d", "e"]
    assert ds.names("action") == ["action_0", "action_1", "action_2"]  # no names in info.json: generic ones
    eps = ds.episodes()
    assert [e.length for e in eps] == LENGTHS and eps[0].tasks == ["pick the cube"]
    assert eps[1].label == "Episode 1 · 7 frames · pick the cube"
    assert "3 episodes" in ds.summary() and "arm" in ds.summary()


def test_arrays_are_sliced_per_episode(ds):
    for ep, n in zip(ds.episodes(), LENGTHS):
        a = ds.arrays(ep)
        assert a["observation.state"].shape == (n, 5) and a["action"].shape == (n, 3)
        assert a["time"].shape == (n,) and a["time"][0] == 0.0 and np.allclose(np.diff(a["time"]), 1 / FPS, atol=1e-4)
    rng = np.random.default_rng(1)
    assert np.allclose(ds.arrays(ds.episodes()[1])["observation.state"], rng.normal(size=(7, 5)).astype(np.float32))


def test_frames_decode_to_the_right_picture_in_rgb(ds):
    cam = ds.camera_keys[0]
    eps = ds.episodes()
    for ep_index, i in ((1, 3), (0, 11), (2, 0), (1, 4), (1, 5)):  # jumps, then sequential reads
        frame = ds.frame(eps[ep_index], cam, i)
        assert frame.shape == (32, 48, 3) and frame.dtype == np.uint8
        r, g, b = frame[8:24, 12:36].reshape(-1, 3).mean(axis=0)
        assert abs(r - 30) < 12 and abs(b - 200) < 12  # RGB order, not BGR
        assert abs(g - (20 + 8 * i + 40 * ep_index)) < 14
    with pytest.raises(IndexError):
        ds.frame(eps[0], cam, 12)


def test_validation_passes_on_a_good_dataset(ds):
    result = ds.validate()
    assert result.overall_status == CheckStatus.PASS, result.summary_text()
    assert result.samples == sum(LENGTHS) and result.duration == pytest.approx(sum(LENGTHS) / FPS)
    assert all(ds.episode_check(e) == CheckStatus.PASS for e in ds.episodes())


def test_validation_fails_when_a_video_is_missing(tmp_path):
    root = _make(tmp_path / "ds")
    shutil.rmtree(root / "videos")
    reader = LeRobotDataset(root)
    result = reader.validate()
    assert result.overall_status == CheckStatus.FAIL and "missing" in result.summary_text()
    assert reader.episode_check(reader.episodes()[0]) == CheckStatus.FAIL


def test_validation_fails_when_data_rows_are_missing(tmp_path):
    root = _make(tmp_path / "ds")
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 3), path)  # the last episode loses 3 rows
    reader = LeRobotDataset(root)
    result = reader.validate()
    assert result.overall_status == CheckStatus.FAIL and "episode 2" in result.summary_text()


def test_dataset_without_camera_warns(tmp_path):
    reader = LeRobotDataset(_make(tmp_path / "ds", with_camera=False))
    assert reader.camera_keys == []
    result = reader.validate()
    assert result.overall_status == CheckStatus.WARN and "no camera" in result.summary_text()


def test_not_a_dataset_is_a_clear_error(tmp_path):
    with pytest.raises(LeRobotError, match="not a LeRobot dataset"):
        LeRobotDataset(tmp_path)


# ---------------------------------------------------------------- datasets from other tools / big datasets


def test_plain_list_columns_are_read_as_matrices(tmp_path):
    """Datasets from other tools store state/action as list<float>, not fixed-size lists."""
    import pyarrow as pa

    from relay_d.acquisition.qt_app.ui.review import lerobot_reader as lr

    plain = pa.chunked_array([pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=pa.list_(pa.float32()))])
    out = lr._vector_column(plain)
    assert out.shape == (2, 3) and out.dtype == np.float32 and out[1, 2] == 6.0
    assert lr._vector_column(pa.chunked_array([pa.array([1.5, 2.5])])).shape == (2, 1)
    ragged = pa.chunked_array([pa.array([[1.0], [2.0, 3.0]], type=pa.list_(pa.float32()))])
    with pytest.raises(LeRobotError, match="different lengths"):
        lr._vector_column(ragged)


def test_every_row_is_converted_once_not_once_per_episode(tmp_path, monkeypatch):
    from relay_d.acquisition.qt_app.ui.review import lerobot_reader as lr

    root = tmp_path / "ds"
    with LeRobotDatasetWriter(root, fps=FPS) as writer:
        for ep in range(30):
            rng = np.random.default_rng(ep)
            writer.add_episode(rng.normal(size=(20, 4)).astype(np.float32), rng.normal(size=(20, 2)).astype(np.float32),
                               {}, task="t")
    converted = []
    original = lr._vector_column
    monkeypatch.setattr(lr, "_vector_column", lambda col: converted.append(len(col)) or original(col))
    reader = LeRobotDataset(root)
    result = reader.validate()
    statuses = [reader.episode_check(e) for e in reader.episodes()]  # reuse validate()'s result
    assert result.overall_status == CheckStatus.WARN and all(s == CheckStatus.PASS for s in statuses)
    assert sum(converted) == 30 * 20 * 2  # state + action of every episode, once (was episodes x all rows, twice)
    rng = np.random.default_rng(7)  # the slice of episode 7 holds exactly what was written for it
    state, action = rng.normal(size=(20, 4)).astype(np.float32), rng.normal(size=(20, 2)).astype(np.float32)
    got = reader.arrays(reader.episodes()[7])
    assert np.allclose(got["observation.state"], state) and np.allclose(got["action"], action)


def test_pyav_fallback_decodes_the_same_frames_and_seeks_both_ways(tmp_path, monkeypatch):
    import cv2

    root = _make(tmp_path / "ds")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: types.SimpleNamespace(isOpened=lambda: False, release=lambda: None))
    reader = LeRobotDataset(root)
    cam, eps = reader.camera_keys[0], reader.episodes()
    for ep_index, i in ((1, 3), (1, 4), (1, 5), (0, 11), (0, 2), (2, 8)):  # forward, sequential, backward, jumps
        frame = reader.frame(eps[ep_index], cam, i)
        r, g, b = frame[8:24, 12:36].reshape(-1, 3).mean(axis=0)
        assert abs(r - 30) < 12 and abs(b - 200) < 12 and abs(g - (20 + 8 * i + 40 * ep_index)) < 14
    reader.close()
    assert not reader._av
