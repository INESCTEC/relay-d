"""Tests for the LeRobotDataset v3.0 writer (no ROS / Qt needed)."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

av = pytest.importorskip("av")

from relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer import (  # noqa: E402
    LeRobotDatasetWriter,
    LeRobotWriterError,
)

FPS = 10


def _episode(n, state_dim=5, action_dim=3, size=(33, 48), seed=0):
    """Random episode; the odd height (33) exercises the even-size crop."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, state_dim)).astype(np.float32)
    action = rng.normal(size=(n, action_dim)).astype(np.float32)
    frames = rng.integers(0, 255, size=(n, *size, 3), dtype=np.uint8)
    return state, action, {"observation.images.cam": frames}


def _video_frames(path):
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    lengths = [12, 7, 9]
    with LeRobotDatasetWriter(root, fps=FPS, robot_type="test") as writer:
        for i, n in enumerate(lengths):
            state, action, images = _episode(n, seed=i)
            writer.add_episode(
                state, action, images, task="pick the cube",
                state_names=["a", "b", "c", "d", "e"],
            )
    return root, lengths


def test_layout_and_info(dataset):
    root, lengths = dataset
    for rel in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.cam/chunk-000/file-000.mp4",
    ):
        assert (root / rel).exists(), rel

    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS
    assert info["total_episodes"] == len(lengths)
    assert info["total_frames"] == sum(lengths)
    assert info["total_tasks"] == 1
    assert info["splits"] == {"train": "0:3"}
    feats = info["features"]
    assert feats["observation.state"]["shape"] == [5]
    assert feats["observation.state"]["names"] == ["a", "b", "c", "d", "e"]
    assert feats["action"]["shape"] == [3]
    # 33 -> 32: yuv420p needs even dimensions
    assert feats["observation.images.cam"]["shape"] == [32, 48, 3]
    assert feats["observation.images.cam"]["dtype"] == "video"
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        assert key in feats


def test_data_parquet(dataset):
    root, lengths = dataset
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    assert table.num_rows == sum(lengths)
    assert set(table.column_names) == {
        "observation.state", "action", "timestamp", "frame_index",
        "episode_index", "index", "task_index",
    }
    assert table["index"].to_pylist() == list(range(sum(lengths)))
    assert table["episode_index"].to_pylist() == [0] * 12 + [1] * 7 + [2] * 9
    ts = np.array(table["timestamp"].to_pylist())
    first = ts[:12]
    np.testing.assert_allclose(first, np.arange(12) / FPS, rtol=1e-6)
    assert len(table["observation.state"][0].as_py()) == 5


def test_episode_rows_and_video(dataset):
    root, lengths = dataset
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    assert eps["length"] == lengths
    assert eps["dataset_from_index"] == [0, 12, 19]
    assert eps["dataset_to_index"] == [12, 19, 28]
    key = "videos/observation.images.cam"
    assert eps[f"{key}/from_timestamp"] == pytest.approx([0.0, 1.2, 1.9])
    assert eps[f"{key}/to_timestamp"] == pytest.approx([1.2, 1.9, 2.8])
    assert eps["stats/observation.state/mean"][0] and len(eps["stats/observation.state/mean"][0]) == 5
    assert eps["stats/observation.images.cam/mean"][0] is not None
    assert len(eps["stats/observation.images.cam/mean"][0]) == 3  # per-channel

    # all episodes share one mp4; its frame count is the sum of episode lengths
    assert _video_frames(root / "videos/observation.images.cam/chunk-000/file-000.mp4") == sum(lengths)


def test_tasks_and_stats(dataset):
    root, _ = dataset
    tasks = pq.read_table(root / "meta/tasks.parquet")
    assert tasks["task"].to_pylist() == ["pick the cube"]
    assert tasks["task_index"].to_pylist() == [0]
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [28]
    assert len(stats["observation.state"]["std"]) == 5
    assert len(stats["observation.images.cam"]["min"]) == 3


def test_bgr_input_is_converted_to_rgb(tmp_path):
    state, action, _ = _episode(4)
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    frames[..., 0] = 255  # pure blue in BGR
    with LeRobotDatasetWriter(tmp_path / "ds", fps=FPS) as writer:
        writer.add_episode(state, action, {"observation.images.cam": frames}, task="t")
    stats = json.loads((tmp_path / "ds/meta/stats.json").read_text())
    mean = [c[0][0] for c in stats["observation.images.cam"]["mean"]]
    assert mean[2] > 0.8 and mean[0] < 0.2  # blue ended up in the RGB blue channel


def test_state_action_only_dataset(tmp_path):
    state, action, _ = _episode(6)
    with LeRobotDatasetWriter(tmp_path / "ds", fps=FPS) as writer:
        writer.add_episode(state, action, None, task="t")
    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["video_path"] is None
    assert not (tmp_path / "ds/videos").exists()


def test_rejects_changed_shapes_and_bad_input(tmp_path):
    state, action, images = _episode(5)
    writer = LeRobotDatasetWriter(tmp_path / "ds", fps=FPS)
    writer.add_episode(state, action, images, task="t")
    with pytest.raises(LeRobotWriterError):
        writer.add_episode(state[:, :4], action, images, task="t")
    with pytest.raises(LeRobotWriterError):
        writer.add_episode(state, action, {}, task="t")
    with pytest.raises(LeRobotWriterError):
        writer.add_episode(state[:3], action, images, task="t")
    writer.finalize()
    with pytest.raises(LeRobotWriterError):
        LeRobotDatasetWriter(tmp_path / "ds", fps=FPS)  # not empty
    with pytest.raises(LeRobotWriterError):
        LeRobotDatasetWriter(tmp_path / "other", fps=0)


def test_file_rollover(tmp_path):
    """A tiny size limit forces new data/video files; episode rows must point at them."""
    with LeRobotDatasetWriter(
        tmp_path / "ds", fps=FPS, data_files_size_in_mb=0.001, video_files_size_in_mb=0.001
    ) as writer:
        for i in range(3):
            state, action, images = _episode(20, seed=i)
            writer.add_episode(state, action, images, task="t")
    eps = pq.read_table(tmp_path / "ds/meta/episodes/chunk-000/file-000.parquet").to_pydict()
    assert eps["data/file_index"] == [0, 1, 2]
    key = "videos/observation.images.cam"
    assert eps[f"{key}/file_index"] == [0, 1, 2]
    assert eps[f"{key}/from_timestamp"] == pytest.approx([0.0, 0.0, 0.0])
    for i in range(3):
        assert (tmp_path / f"ds/data/chunk-000/file-{i:03d}.parquet").exists()
        assert _video_frames(tmp_path / f"ds/videos/observation.images.cam/chunk-000/file-{i:03d}.mp4") == 20


def test_failed_conversion_removes_partial_output(tmp_path):
    root = tmp_path / "ds"
    with pytest.raises(RuntimeError):
        with LeRobotDatasetWriter(root, fps=10) as w:
            w.add_episode(np.zeros((5, 3)), np.zeros((5, 2)), None, task="t")
            raise RuntimeError("boom")
    assert not root.exists()
    # the folder can be reused straight away
    with LeRobotDatasetWriter(root, fps=10) as w:
        w.add_episode(np.zeros((5, 3)), np.zeros((5, 2)), None, task="t")
    assert (root / "meta/info.json").exists()
