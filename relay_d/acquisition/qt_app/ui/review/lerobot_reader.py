"""Read a LeRobotDataset v3.0 folder for viewing: episodes, state / action curves, camera frames, checks.

Pure Python (pyarrow + OpenCV, no Qt, no ROS). View-only: it never writes to the dataset.
``pyarrow`` comes with the ``lerobot`` extra; camera frames are decoded with OpenCV, which is a core
dependency (PyAV is the fallback if OpenCV cannot open a file).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .demo_validator import CheckResult, CheckStatus, DemoValidationResult, _SEVERITY

STATE_KEY, ACTION_KEY = "observation.state", "action"
_MAX_OPEN_VIDEOS = 4


class LeRobotError(Exception):
    """The folder is not a readable LeRobot dataset (or pyarrow is missing)."""


def is_lerobot_dataset(path) -> bool:
    return (Path(path) / "meta" / "info.json").is_file()


def _pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise LeRobotError("Reading LeRobot datasets needs pyarrow: pip install 'relay-d[lerobot]'") from None
    return pq


@dataclass
class Episode:
    index: int
    length: int
    tasks: List[str]
    data_file: tuple  # (chunk, file) of the data parquet
    from_index: int
    to_index: int
    videos: Dict[str, dict] = field(default_factory=dict)  # camera key -> {chunk, file, from_ts, to_ts}

    @property
    def label(self) -> str:
        task = f" · {self.tasks[0]}" if self.tasks else ""
        return f"Episode {self.index} · {self.length} frames{task}"


def _vector_column(col) -> np.ndarray:
    """A list column (fixed-size or plain ``list<float>``) as an (N, D) float array; a scalar column as (N, 1)."""
    import pyarrow as pa

    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(col)
    t = col.type
    if pa.types.is_fixed_size_list(t):
        return col.flatten().to_numpy(zero_copy_only=False).reshape(n, t.list_size)
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        lengths = np.diff(col.offsets.to_numpy())
        if n and (lengths != lengths[0]).any():
            raise LeRobotError("a vector column has rows of different lengths")
        width = int(lengths[0]) if n else 0
        return col.flatten().to_numpy(zero_copy_only=False).reshape(n, width)
    return col.to_numpy(zero_copy_only=False).reshape(n, 1)


class LeRobotDataset:
    def __init__(self, root):
        self.root = Path(root)
        try:
            self.info: Dict[str, Any] = json.loads((self.root / "meta" / "info.json").read_text())
        except (OSError, ValueError) as e:
            raise LeRobotError(f"{self.root} is not a LeRobot dataset: {e}") from None
        self.fps = int(self.info.get("fps") or 0)
        self.features: Dict[str, dict] = self.info.get("features", {})
        self.camera_keys = [k for k, v in self.features.items() if v.get("dtype") == "video"]
        self._episodes: Optional[List[Episode]] = None
        self._tables: Dict[tuple, Any] = {}
        self._row_index: Dict[tuple, np.ndarray] = {}  # (chunk, file) -> the global `index` column
        self._status: Dict[int, CheckStatus] = {}  # episode -> result of validate()
        self._av: "OrderedDict[str, dict]" = OrderedDict()  # PyAV fallback: open containers
        self._videos: "OrderedDict[str, dict]" = OrderedDict()  # path -> {"cap": VideoCapture, "next": frame no.}

    # ------------------------------------------------------------------ metadata

    def names(self, key: str) -> List[str]:
        """Names of the columns of a vector feature (joint names, ...)."""
        feature = self.features.get(key, {})
        names = feature.get("names")
        width = (feature.get("shape") or [0])[0]
        if isinstance(names, list) and len(names) == width:
            return [str(n) for n in names]
        return [f"{key}[{i}]" for i in range(width)]

    def episodes(self) -> List[Episode]:
        if self._episodes is None:
            pq = _pyarrow()
            files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            if not files:
                raise LeRobotError("meta/episodes has no parquet files")
            rows = []
            for f in files:
                rows.extend(pq.read_table(f).to_pylist())
            episodes = []
            for r in sorted(rows, key=lambda r: r["episode_index"]):
                videos = {
                    k: {
                        "chunk": r.get(f"videos/{k}/chunk_index"), "file": r.get(f"videos/{k}/file_index"),
                        "from_ts": r.get(f"videos/{k}/from_timestamp"), "to_ts": r.get(f"videos/{k}/to_timestamp"),
                    }
                    for k in self.camera_keys
                }
                episodes.append(Episode(
                    index=int(r["episode_index"]), length=int(r["length"]), tasks=list(r.get("tasks") or []),
                    data_file=(int(r["data/chunk_index"]), int(r["data/file_index"])),
                    from_index=int(r["dataset_from_index"]), to_index=int(r["dataset_to_index"]), videos=videos,
                ))
            self._episodes = episodes
        return self._episodes

    def summary(self) -> str:
        eps = self.episodes()
        tasks = sorted({t for e in eps for t in e.tasks})
        return "\n".join([
            f"LeRobot dataset {self.info.get('codebase_version', '?')}: {self.root.name}",
            f"{len(eps)} episodes, {self.info.get('total_frames', sum(e.length for e in eps))} frames, {self.fps} fps",
            f"Robot type: {self.info.get('robot_type') or 'not set'}",
            f"Cameras: {', '.join(self.camera_keys) or 'none'}",
            f"Tasks: {', '.join(tasks) or 'none'}",
        ])

    # ------------------------------------------------------------------ data

    def _table(self, chunk: int, file: int):
        key = (chunk, file)
        if key not in self._tables:
            path = self.root / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"
            try:
                self._tables[key] = _pyarrow().read_table(path)
            except OSError as e:
                raise LeRobotError(f"cannot read {path}: {e}") from None
            if len(self._tables) > 3:  # a few files are enough for browsing episodes in order
                self._tables.pop(next(iter(self._tables)))
        return self._tables[key]

    def _episode_rows(self, episode: Episode):
        """The slice of the data file that holds one episode (rows are ordered by the global ``index``)."""
        table = self._table(*episode.data_file)
        if "index" in table.column_names:
            idx = self._row_index.get(episode.data_file)
            if idx is None:
                idx = self._row_index[episode.data_file] = np.asarray(table.column("index").to_numpy())
            start, stop = np.searchsorted(idx, [episode.from_index, episode.to_index], side="left")
        else:  # no global index: fall back to the episode column
            rows = np.nonzero(np.asarray(table.column("episode_index").to_numpy()) == episode.index)[0]
            start, stop = (rows[0], rows[-1] + 1) if len(rows) else (0, 0)
        return table.slice(int(start), int(stop - start))

    def arrays(self, episode: Episode) -> Dict[str, np.ndarray]:
        """``time`` (T,) plus ``observation.state`` / ``action`` (T, D) of one episode."""
        rows = self._episode_rows(episode)
        out = {"time": np.asarray(rows.column("timestamp").to_numpy()).astype(np.float64)}
        for key in (STATE_KEY, ACTION_KEY):
            if key in rows.column_names:
                out[key] = _vector_column(rows.column(key))
        return out

    # ------------------------------------------------------------------ video

    def _video_path(self, episode: Episode, camera: str) -> Path:
        v = episode.videos.get(camera)
        if not v or v["chunk"] is None:
            raise LeRobotError(f"episode {episode.index} has no video for '{camera}'")
        return self.root / "videos" / camera / f"chunk-{v['chunk']:03d}" / f"file-{v['file']:03d}.mp4"

    def frame(self, episode: Episode, camera: str, i: int) -> np.ndarray:
        """Frame ``i`` of an episode as an RGB uint8 array."""
        import cv2

        if not 0 <= i < episode.length:
            raise IndexError(f"frame {i} out of range for an episode of {episode.length} frames")
        path = self._video_path(episode, camera)
        number = int(round(episode.videos[camera]["from_ts"] * self.fps)) + i
        entry = self._videos.pop(str(path), None)
        if entry is None:
            entry = {"cap": cv2.VideoCapture(str(path)), "next": 0}
        self._videos[str(path)] = entry  # most recently used last
        while len(self._videos) > _MAX_OPEN_VIDEOS:
            _, old = self._videos.popitem(last=False)
            old["cap"].release()
        cap = entry["cap"]
        if not cap.isOpened():
            return self._frame_av(path, number)
        if entry["next"] != number:  # sequential playback needs no seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, bgr = cap.read()
        entry["next"] = number + 1
        if not ok:
            return self._frame_av(path, number)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _frame_av(self, path: Path, number: int) -> np.ndarray:
        """Fallback decoder: one open container per file, seek to the keyframe before the frame, decode forward."""
        try:
            import av
        except ImportError:
            raise LeRobotError(f"cannot decode {path.name}: OpenCV failed and PyAV is not installed") from None
        entry = self._av.pop(str(path), None)
        if entry is None:
            container = av.open(str(path))
            entry = {"container": container, "stream": container.streams.video[0], "frames": None, "next": None}
        self._av[str(path)] = entry
        while len(self._av) > _MAX_OPEN_VIDEOS:
            self._av.popitem(last=False)[1]["container"].close()
        container, stream = entry["container"], entry["stream"]
        fps = float(stream.average_rate or self.fps)
        sequential = entry["frames"] is not None and entry["next"] is not None and entry["next"] <= number <= entry["next"] + 60
        if not sequential:
            container.seek(int(number / fps / stream.time_base), stream=stream, backward=True)
            entry["frames"], entry["next"] = container.decode(stream), None
        for frame in entry["frames"]:
            n = int(round(frame.pts * stream.time_base * fps))
            entry["next"] = n + 1
            if n == number:
                return frame.to_ndarray(format="rgb24")
            if n > number:
                break
        entry["frames"] = None
        raise LeRobotError(f"{path.name} has no frame {number}")

    def close(self) -> None:
        for entry in self._videos.values():
            entry["cap"].release()
        self._videos.clear()
        for entry in self._av.values():
            entry["container"].close()
        self._av.clear()
        self._tables.clear()
        self._row_index.clear()

    # ------------------------------------------------------------------ checks

    def validate(self) -> DemoValidationResult:
        """Structure checks of the whole dataset (same result type as the demo validator)."""
        result = DemoValidationResult(file_path=str(self.root), demo_name=self.root.name)
        add = lambda name, status, msg: result.checks.append(CheckResult(name, status, msg))  # noqa: E731
        P, W, F = CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL

        version = self.info.get("codebase_version")
        add("version", P if version == "v3.0" else W, f"codebase_version {version!r}")
        if self.fps <= 0:
            add("fps", F, f"fps is {self.info.get('fps')!r}")
        try:
            episodes = self.episodes()
        except LeRobotError as e:
            add("episodes", F, str(e))
            return self._finish(result)
        total = sum(e.length for e in episodes)
        result.samples = total
        result.duration = total / self.fps if self.fps > 0 else 0.0
        declared = self.info.get("total_episodes")
        add("episodes", P if declared in (None, len(episodes)) else F,
            f"{len(episodes)} episodes" + ("" if declared in (None, len(episodes)) else f", info.json says {declared}"))
        declared = self.info.get("total_frames")
        add("frames", P if declared in (None, total) else F,
            f"{total} frames" + ("" if declared in (None, total) else f", info.json says {declared}"))
        if not self.camera_keys:
            add("cameras", W, "no camera in this dataset (ACT and Diffusion need one)")

        data_bad, missing = [], set()
        self._status = {}
        for e in episodes:
            problems = []  # data problems of this episode
            try:
                a = self.arrays(e)
            except LeRobotError as err:
                add(f"episode {e.index}", F, str(err))
                self._status[e.index] = F
                continue
            n = len(a["time"])
            if n != e.length:
                problems.append(f"episode {e.index}: {n} rows, expected {e.length}")
            else:
                if n > 1 and not (np.diff(a["time"]) > 0).all():
                    problems.append(f"episode {e.index}: timestamps do not increase")
                for key in (STATE_KEY, ACTION_KEY):
                    if key in a and not np.isfinite(a[key]).all():
                        problems.append(f"episode {e.index}: NaN/inf in {key}")
            gone = [str(self._video_path(e, k).relative_to(self.root)) for k in self.camera_keys
                    if e.videos[k]["chunk"] is not None and not self._video_path(e, k).is_file()]
            missing.update(gone)
            data_bad.extend(problems)
            self._status[e.index] = F if problems or gone else P
        shown = "; ".join(data_bad[:5]) + (f" (+{len(data_bad) - 5} more)" if len(data_bad) > 5 else "")
        add("data", F if data_bad else P, shown or "state / action rows match the episode lengths, finite and ordered")
        missing = sorted(missing)
        add("videos", F if missing else P, f"missing: {', '.join(missing[:3])}" if missing else "all video files exist")
        return self._finish(result)

    def episode_check(self, episode: Episode) -> CheckStatus:
        """Status of one episode for the list colour (runs ``validate`` once for the whole dataset)."""
        if not self._status:
            self.validate()
        return self._status.get(episode.index, CheckStatus.FAIL)

    @staticmethod
    def _finish(result: DemoValidationResult) -> DemoValidationResult:
        result.overall_status = max((c.status for c in result.checks), key=lambda s: _SEVERITY[s], default=CheckStatus.PASS)
        return result
