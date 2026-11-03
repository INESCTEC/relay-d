"""Self-contained writer for the LeRobotDataset v3.0 on-disk format.

Layout produced (``codebase_version`` "v3.0")::

    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet                      (many episodes per file)
    videos/<camera key>/chunk-000/file-000.mp4           (many episodes per file)

Only ``pyarrow`` and ``av`` (PyAV) are needed, so this works in an environment
that cannot install ``lerobot`` itself (it requires numpy>=2, while ROS
``cv_bridge`` pins numpy<2). Episodes inside a shared parquet/mp4 file are
located through ``meta/episodes`` (``dataset_from_index``/``dataset_to_index``
and ``videos/<key>/from_timestamp``/``to_timestamp``).

Usage::

    with LeRobotDatasetWriter(root, fps=20) as writer:
        writer.add_episode(state, action, {"cam": frames_rgb}, task="pick the cube")
"""

from __future__ import annotations

import json
import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # optional "lerobot" extra: check_dependencies() reports a missing install clearly
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised by tests via sys.modules
    pa = pq = None

from relay_d.utils.coloring_logger import logger

CODEBASE_VERSION = "v3.0"

DEFAULT_CHUNK_SIZE = 1000  # max files per chunk directory
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
_QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")
_STAT_KEYS = ("min", "max", "mean", "std", "count") + _QUANTILE_KEYS

# Video encoder settings. H.264 + yuv420p decodes everywhere (PyAV, torchcodec);
# g=2 gives frequent keyframes so per-frame seeking during training is cheap.
_VCODEC = "libx264"
_PIX_FMT = "yuv420p"
_GOP = 2
_CRF = 23
_PRESET = "veryfast"

_MAX_STAT_FRAMES = 100  # frames sampled per episode for image statistics
_MAX_STAT_SIDE = 150  # sampled frames are strided down to about this size


class LeRobotWriterError(RuntimeError):
    pass


class LeRobotDependencyError(LeRobotWriterError):
    """A required library is missing in the running Python. Fatal for the whole
    conversion (every episode would fail the same way), never a per-episode skip."""


def check_dependencies(need_video: bool) -> None:
    """Raise LeRobotDependencyError if pyarrow (or PyAV, for cameras) can't be imported.

    Both come from the optional ``lerobot`` extra (``pip install ".[lerobot]"``).
    """
    import importlib
    import sys

    required = [("pyarrow", "pyarrow")]
    if need_video:
        required.append(("av", "av"))
    for module, package in required:
        try:
            importlib.import_module(module)
        except ImportError as e:
            raise LeRobotDependencyError(
                f"'{package}' is not installed for the Python running this app "
                f"({sys.executable}). It is part of the optional LeRobot extra; install it with:\n"
                f'    {sys.executable} -m pip install ".[lerobot]"   (from the repository root)\n'
                f"or just:\n"
                f"    {sys.executable} -m pip install {package}"
            ) from e


def _chunk_file_next(chunk: int, file: int, chunks_size: int):
    if file == chunks_size - 1:
        return chunk + 1, 0
    return chunk, file + 1


def _vector_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-column stats for an (N, D) float array, or (N,) -> shape (1,) stats."""
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr.astype(np.float64, copy=False)
    out = {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    q = np.quantile(arr, _QUANTILES, axis=0)
    for key, row in zip(_QUANTILE_KEYS, q):
        out[key] = row
    return out


def _image_stats(frames: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-channel stats over sampled frames, scaled to [0, 1], shape (C, 1, 1)."""
    n = frames.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(n, _MAX_STAT_FRAMES)).astype(int))
    h, w = frames.shape[1:3]
    step = max(1, max(h, w) // _MAX_STAT_SIDE)
    # one strided indexing op: never materializes the full-resolution sampled frames
    sampled = frames[idx, ::step, ::step, :].astype(np.float64) / 255.0
    # (S, h, w, C) -> (C, S*h*w)
    flat = sampled.transpose(3, 0, 1, 2).reshape(sampled.shape[3], -1)
    out = {
        "min": flat.min(axis=1),
        "max": flat.max(axis=1),
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
    }
    q = np.quantile(flat, _QUANTILES, axis=1)
    for key, row in zip(_QUANTILE_KEYS, q):
        out[key] = row
    out = {k: v.reshape(-1, 1, 1) for k, v in out.items()}
    out["count"] = np.array([n], dtype=np.int64)
    return out


def _aggregate_stats(episode_stats: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Combine per-episode stats of one feature (exact mean/std, weighted quantiles)."""
    counts = np.array([float(s["count"][0]) for s in episode_stats])
    total = counts.sum()

    def stack(key):
        return np.stack([s[key] for s in episode_stats])

    w = counts.reshape((-1,) + (1,) * (stack("mean").ndim - 1))
    means = stack("mean")
    total_mean = (means * w).sum(axis=0) / total
    var = ((stack("std") ** 2 + (means - total_mean) ** 2) * w).sum(axis=0) / total
    out = {
        "min": stack("min").min(axis=0),
        "max": stack("max").max(axis=0),
        "mean": total_mean,
        "std": np.sqrt(var),
        "count": np.array([int(total)], dtype=np.int64),
    }
    for key in _QUANTILE_KEYS:
        out[key] = (stack(key) * w).sum(axis=0) / total  # approximate, as in lerobot
    return out


def _hf_metadata(features: Dict[str, Dict]) -> Dict[bytes, bytes]:
    """Hugging Face ``datasets`` schema metadata so the parquet loads with typed columns."""
    hf = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        value = {"dtype": ft["dtype"], "_type": "Value"}
        shape = ft["shape"]
        if tuple(shape) == (1,):
            hf[key] = value
        else:
            hf[key] = {"feature": value, "length": int(shape[0]), "_type": "Sequence"}
    return {b"huggingface": json.dumps({"info": {"features": hf}}).encode()}


def _pandas_index_metadata() -> Dict[bytes, bytes]:
    """pandas metadata so ``pd.read_parquet`` restores ``task`` as the index."""
    meta = {
        "index_columns": ["task"],
        "column_indexes": [
            {"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}
        ],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    return {b"pandas": json.dumps(meta).encode()}


class _VideoFile:
    """One open mp4 (PyAV container) that episodes are appended to."""

    def __init__(self, path: Path, fps: int, width: int, height: int,
                 crf: int = _CRF, preset: str = _PRESET, gop: int = _GOP):
        check_dependencies(need_video=True)
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(
            _VCODEC,
            rate=fps,
            options={"g": str(gop), "crf": str(crf), "preset": preset, "bf": "0"},
        )
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = _PIX_FMT
        self.stream.codec_context.time_base = Fraction(1, fps)
        self.frames = 0
        self.bytes = 0

    def write(self, frames_rgb: np.ndarray) -> None:
        import av

        for frame_rgb in frames_rgb:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_rgb), format="rgb24")
            frame.pts = self.frames
            frame.time_base = Fraction(1, self.fps)
            self.frames += 1
            for packet in self.stream.encode(frame):
                self.bytes += packet.size
                self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode(None):
            self.bytes += packet.size
            self.container.mux(packet)
        self.container.close()


class LeRobotDatasetWriter:
    """Write episodes into a LeRobotDataset v3.0 directory."""

    def __init__(
        self,
        root: str | os.PathLike,
        fps: int,
        robot_type: Optional[str] = None,
        chunks_size: int = DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        images_are_bgr: bool = True,
        video_crf: int = _CRF,
        video_preset: str = _PRESET,
        video_gop: int = _GOP,
    ):
        check_dependencies(need_video=False)
        fps = int(fps)
        if fps <= 0:
            raise LeRobotWriterError(f"fps must be a positive integer, got {fps}")
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise LeRobotWriterError(f"Output directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.robot_type = robot_type
        self.chunks_size = chunks_size
        self.data_max_mb = data_files_size_in_mb
        self.video_max_mb = video_files_size_in_mb
        self.video_crf, self.video_preset, self.video_gop = int(video_crf), str(video_preset), int(video_gop)
        self.images_are_bgr = images_are_bgr

        self._features: Optional[Dict[str, Dict]] = None
        self._camera_keys: List[str] = []

        self._tasks: Dict[str, int] = {}
        self._episodes: List[Dict] = []
        self._episode_stats: List[Dict[str, Dict[str, np.ndarray]]] = []
        self._total_frames = 0

        # data parquet state
        self._data_chunk = 0
        self._data_file = 0
        self._data_sink: Optional[pa.OSFile] = None
        self._data_writer: Optional[pq.ParquetWriter] = None
        self._data_file_frames = 0
        self._arrow_schema: Optional[pa.Schema] = None

        # per-camera video state: key -> {"chunk", "file", "video": _VideoFile}
        self._videos: Dict[str, Dict] = {}
        self._finalized = False

    # ------------------------------------------------------------------ public

    def __enter__(self) -> "LeRobotDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self._abort()

    def add_episode(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: Optional[Dict[str, np.ndarray]],
        task: str,
        state_names: Optional[Sequence[str]] = None,
        action_names: Optional[Sequence[str]] = None,
    ) -> int:
        """Append one episode and return its episode_index.

        Args:
            state:  (N, D) observation.state.
            action: (N, A) action.
            images: camera key -> (N, H, W, 3) uint8 frames (BGR unless ``images_are_bgr=False``).
            task:   natural-language task description.
        """
        if self._finalized:
            raise LeRobotWriterError("Writer already finalized")
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.ndim != 2 or action.ndim != 2:
            raise LeRobotWriterError("state and action must be 2-D (N, dim) arrays")
        n = state.shape[0]
        if n == 0 or action.shape[0] != n:
            raise LeRobotWriterError(
                f"state ({state.shape[0]}) and action ({action.shape[0]}) need the same non-zero length"
            )

        images = {k: self._prepare_frames(k, v, n) for k, v in (images or {}).items()}
        if self._features is None:
            self._init_schema(state, action, images, state_names, action_names)
        self._check_episode(state, action, images)

        episode_index = len(self._episodes)
        task_index = self._tasks.setdefault(task, len(self._tasks))
        start = self._total_frames

        cols = {
            "observation.state": state,
            "action": action,
            "timestamp": (np.arange(n, dtype=np.float32) / np.float32(self.fps)).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "index": np.arange(start, start + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        }

        stats = {name: _vector_stats(arr) for name, arr in cols.items()}
        for key, frames in images.items():
            stats[key] = _image_stats(frames)

        row = {
            "episode_index": episode_index,
            "tasks": [task],
            "length": n,
        }
        row.update(self._write_data(cols, n, start))
        for key, frames in images.items():
            row.update(self._write_video(key, frames))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        self._episodes.append(row)
        self._episode_stats.append(stats)
        self._total_frames += n
        return episode_index

    def finalize(self) -> None:
        """Close writers and write metadata. Must be called, or the parquet files are corrupt."""
        if self._finalized:
            return
        self._finalized = True
        if not self._episodes:
            self._abort()
            raise LeRobotWriterError("No episodes were added")
        try:
            self._close_data_writer()
            for key, state in self._videos.items():
                state["video"].close()
            self._write_tasks()
            self._write_episodes()
            global_stats = {
                feat: _aggregate_stats([s[feat] for s in self._episode_stats])
                for feat in self._episode_stats[0]
            }
            self._write_json(STATS_PATH, {f: {k: np.asarray(v).tolist() for k, v in s.items()} for f, s in global_stats.items()})
            self._write_info()
        except BaseException:
            self._abort()
            raise
        logger.info(
            f"LeRobot dataset written: {len(self._episodes)} episodes, "
            f"{self._total_frames} frames -> {self.root}"
        )

    # ----------------------------------------------------------------- schema

    def _prepare_frames(self, key: str, frames: np.ndarray, n: int) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.uint8:
            raise LeRobotWriterError(
                f"Camera '{key}' must be (N, H, W, 3) uint8, got {frames.shape} {frames.dtype}"
            )
        if frames.shape[0] != n:
            raise LeRobotWriterError(f"Camera '{key}' has {frames.shape[0]} frames, expected {n}")
        if self.images_are_bgr:
            frames = frames[..., ::-1]
        h, w = frames.shape[1], frames.shape[2]
        eh, ew = h - (h % 2), w - (w % 2)  # yuv420p needs even dimensions
        if (eh, ew) != (h, w):
            frames = frames[:, :eh, :ew]
        return frames

    def _init_schema(self, state, action, images, state_names, action_names) -> None:
        def names(prefix, given, dim):
            if given is not None and len(given) == dim:
                return [str(x) for x in given]
            return [f"{prefix}_{i}" for i in range(dim)]

        feats: Dict[str, Dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [state.shape[1]],
                "names": names("state", state_names, state.shape[1]),
            },
            "action": {
                "dtype": "float32",
                "shape": [action.shape[1]],
                "names": names("action", action_names, action.shape[1]),
            },
        }
        for key, frames in images.items():
            h, w = frames.shape[1], frames.shape[2]
            self._camera_keys.append(key)
            feats[key] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": "h264",
                    "video.pix_fmt": _PIX_FMT,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "video.g": self.video_gop,
                    "video.crf": self.video_crf,
                    "video.preset": self.video_preset,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        for key in ("timestamp",):
            feats[key] = {"dtype": "float32", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index"):
            feats[key] = {"dtype": "int64", "shape": [1], "names": None}
        self._features = feats

        def vec(dim):
            return pa.list_(pa.float32(), dim)

        fields = [
            pa.field("observation.state", vec(state.shape[1])),
            pa.field("action", vec(action.shape[1])),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
        self._arrow_schema = pa.schema(fields, metadata=_hf_metadata(feats))

    def _check_episode(self, state, action, images) -> None:
        assert self._features is not None
        if state.shape[1] != self._features["observation.state"]["shape"][0]:
            raise LeRobotWriterError(
                f"observation.state width changed: {state.shape[1]} vs "
                f"{self._features['observation.state']['shape'][0]}"
            )
        if action.shape[1] != self._features["action"]["shape"][0]:
            raise LeRobotWriterError(
                f"action width changed: {action.shape[1]} vs {self._features['action']['shape'][0]}"
            )
        if set(images) != set(self._camera_keys):
            raise LeRobotWriterError(
                f"Camera set changed between episodes: {sorted(images)} vs {sorted(self._camera_keys)}"
            )
        for key, frames in images.items():
            h, w, _ = self._features[key]["shape"]
            if frames.shape[1:3] != (h, w):
                raise LeRobotWriterError(
                    f"Camera '{key}' resolution changed: {frames.shape[1:3]} vs {(h, w)}"
                )

    # ------------------------------------------------------------------- data

    def _data_path(self) -> Path:
        return self.root / DATA_PATH.format(chunk_index=self._data_chunk, file_index=self._data_file)

    def _close_data_writer(self) -> None:
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        if self._data_sink is not None:
            self._data_sink.close()
            self._data_sink = None

    def _write_data(self, cols: Dict[str, np.ndarray], n: int, start: int) -> Dict:
        assert self._arrow_schema is not None
        if self._data_writer is not None and self._data_file_frames > 0:
            size_mb = self._data_sink.tell() / 1e6
            projected = size_mb + (size_mb / self._data_file_frames) * n
            if projected >= self.data_max_mb:
                self._close_data_writer()
                self._data_chunk, self._data_file = _chunk_file_next(
                    self._data_chunk, self._data_file, self.chunks_size
                )
                self._data_file_frames = 0
        if self._data_writer is None:
            path = self._data_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._data_sink = pa.OSFile(str(path), "wb")
            self._data_writer = pq.ParquetWriter(
                self._data_sink, self._arrow_schema, compression="snappy", use_dictionary=True
            )

        arrays = []
        for field in self._arrow_schema:
            arr = cols[field.name]
            if pa.types.is_fixed_size_list(field.type):
                dim = field.type.list_size
                arrays.append(
                    pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), dim)
                )
            else:
                arrays.append(pa.array(arr, type=field.type))
        self._data_writer.write_table(pa.Table.from_arrays(arrays, schema=self._arrow_schema))
        self._data_file_frames += n
        return {
            "data/chunk_index": self._data_chunk,
            "data/file_index": self._data_file,
            "dataset_from_index": start,
            "dataset_to_index": start + n,
        }

    # ------------------------------------------------------------------ video

    def _video_path(self, key: str, chunk: int, file: int) -> Path:
        return self.root / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)

    def _write_video(self, key: str, frames: np.ndarray) -> Dict:
        assert self._features is not None
        h, w, _ = self._features[key]["shape"]
        n = frames.shape[0]
        state = self._videos.get(key)
        if state is not None and state["video"].frames > 0:
            video: _VideoFile = state["video"]
            projected = (video.bytes + (video.bytes / video.frames) * n) / 1e6
            if projected >= self.video_max_mb:
                video.close()
                state["chunk"], state["file"] = _chunk_file_next(
                    state["chunk"], state["file"], self.chunks_size
                )
                state["video"] = None
        if state is None:
            state = {"chunk": 0, "file": 0, "video": None}
            self._videos[key] = state
        if state["video"] is None:
            state["video"] = _VideoFile(
                self._video_path(key, state["chunk"], state["file"]), self.fps, w, h,
                crf=self.video_crf, preset=self.video_preset, gop=self.video_gop,
            )

        video = state["video"]
        from_ts = video.frames / self.fps
        video.write(frames)
        to_ts = video.frames / self.fps
        return {
            f"videos/{key}/chunk_index": state["chunk"],
            f"videos/{key}/file_index": state["file"],
            f"videos/{key}/from_timestamp": float(from_ts),
            f"videos/{key}/to_timestamp": float(to_ts),
        }

    # --------------------------------------------------------------- metadata

    def _write_json(self, rel: str, obj) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)
            f.write("\n")

    def _write_tasks(self) -> None:
        tasks = sorted(self._tasks.items(), key=lambda kv: kv[1])
        schema = pa.schema(
            [pa.field("task_index", pa.int64()), pa.field("task", pa.string())],
            metadata=_pandas_index_metadata(),
        )
        table = pa.Table.from_arrays(
            [pa.array([i for _, i in tasks], pa.int64()), pa.array([t for t, _ in tasks], pa.string())],
            schema=schema,
        )
        path = self.root / TASKS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    def _write_episodes(self) -> None:
        columns: Dict[str, list] = {}
        for row, stats in zip(self._episodes, self._episode_stats):
            full = dict(row)
            for feat, feat_stats in stats.items():
                for stat in _STAT_KEYS:
                    if stat in feat_stats:
                        full[f"stats/{feat}/{stat}"] = np.asarray(feat_stats[stat]).tolist()
            for name, value in full.items():
                columns.setdefault(name, []).append(value)
        table = pa.table({name: pa.array(values) for name, values in columns.items()})
        path = self.root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy")

    def _write_info(self) -> None:
        assert self._features is not None
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_max_mb,
            "video_files_size_in_mb": self.video_max_mb,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self._camera_keys else None,
            "features": self._features,
        }
        self._write_json(INFO_PATH, info)

    def _abort(self) -> None:
        """Release file handles and delete the half-written dataset (the constructor
        checked that the folder was empty, so everything in it is ours)."""
        self._finalized = True
        try:
            self._close_data_writer()
        except Exception:
            pass
        for state in self._videos.values():
            try:
                if state["video"] is not None:
                    state["video"].close()
            except Exception:
                pass
        shutil.rmtree(self.root, ignore_errors=True)
