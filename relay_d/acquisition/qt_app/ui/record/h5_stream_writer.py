"""Background streaming HDF5 writer for heavy array modalities (images, organized point clouds).

Instead of holding gigabytes of decoded camera / scan frames in RAM until Stop, ``DataRecorder``
enqueues each heavy frame into ``StreamingH5Writer``. A dedicated daemon thread writes single-frame
chunks into extendable datasets inside the open ``<demo>.h5.partial`` file (``libhdf5`` releases the
GIL during compression and disk I/O), keeping memory usage bounded by the in-flight queue and
making Stop finish in under a second.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Dict, Optional, Set, Tuple

import h5py
import numpy as np

from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger

_SENTINEL = object()


def is_streamable_sample(sample: Optional[dict]) -> bool:
    """True for heavy array samples whose frames should stream straight to HDF5."""
    if not isinstance(sample, dict):
        return False
    dtype = sample.get("type")
    if dtype == "image":
        return isinstance(sample.get("data"), np.ndarray)
    if dtype == "pointcloud":
        return (
            bool(sample.get("organized"))
            and isinstance(sample.get("points"), np.ndarray)
            and sample["points"].ndim == 3
        )
    return False


def _sample_array(sample: dict) -> np.ndarray:
    return sample["data"] if sample["type"] == "image" else sample["points"]


class _ContainerStreamState:
    """Per-container HDF5 dataset state owned exclusively by the writer thread."""

    def __init__(self, data_type: str, group: h5py.Group, shape: Tuple[int, ...], dtype: np.dtype):
        self.data_type = data_type
        self.group = group
        self.shape = shape
        self.dtype = dtype
        self.dataset: Optional[h5py.Dataset] = None
        self.count = 0
        self.inconsistent_warned = False


class StreamingH5Writer:
    """Streams heavy array frames to an open ``h5py.File`` on a background thread.

    Thread-safety contract: while the writer thread is alive (between ``__init__`` and
    ``finish()``), no other thread may read or mutate ``h5_file``.
    """

    def __init__(
        self,
        h5_file: h5py.File,
        demo_name: str,
        max_queue_bytes: int,
        flush_every: int = 64,
    ):
        self._h5_file = h5_file
        self._demo_name = demo_name
        self._max_queue_bytes = max(1, int(max_queue_bytes))
        self._flush_every = max(1, int(flush_every))

        self._queue: queue.Queue = queue.Queue()
        self._bytes_lock = threading.Lock()
        self._queued_bytes = 0
        self._written_items = 0
        self._error: Optional[Exception] = None
        self._closed = False
        self._abort_flag = False

        self._states: Dict[str, _ContainerStreamState] = {}
        self._demo_group: Optional[h5py.Group] = None
        self._writes_since_flush = 0
        self._streamed_containers: Set[str] = set()
        self._counts: Dict[str, int] = {}

        self._thread = threading.Thread(
            target=self._run, name=f"h5_stream_{demo_name}", daemon=True
        )
        self._thread.start()

    @property
    def queued_bytes(self) -> int:
        with self._bytes_lock:
            return self._queued_bytes

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def frame_count(self, container_id: str) -> int:
        return self._counts.get(container_id, 0)

    def enqueue(self, container_id: str, sample: dict, repeat: int = 1) -> bool:
        """Queue ``sample`` to be written ``repeat`` times (>=1, used to backfill leading Nones).

        Returns False if the writer already failed or has been closed.
        """
        if self._closed or self._error is not None or self._abort_flag:
            return False
        repeat = max(1, int(repeat))
        arr = _sample_array(sample)
        nbytes = int(arr.nbytes)
        with self._bytes_lock:
            self._queued_bytes += nbytes
        self._streamed_containers.add(container_id)
        self._counts[container_id] = self._counts.get(container_id, 0) + repeat
        self._queue.put((container_id, sample, repeat, nbytes))
        return True

    def finish(
        self,
        timeout: float = 30.0,
        stall_timeout: Optional[float] = None,
    ) -> Tuple[bool, Set[str], Optional[Exception]]:
        """Drain all queued frames, flush the HDF5 file, and join the writer thread.

        Uses stall-aware draining: as long as frames are actively being written to disk (e.g. flushing
        a large buffer to a slow HDD/USB drive), draining continues up to ``max(timeout, 180.0)`` when
        ``timeout >= 15.0``, aborting only if no write progress happens for ``stall_timeout`` seconds.
        """
        if not self._closed:
            self._closed = True
            self._queue.put(_SENTINEL)

        if self._thread.is_alive():
            eff_stall = float(stall_timeout) if stall_timeout is not None else min(15.0, float(timeout))
            max_wait = max(float(timeout), 180.0) if (stall_timeout is not None or timeout >= 15.0) else float(timeout)
            now = time.monotonic()
            hard_deadline = now + max_wait
            stall_deadline = now + eff_stall
            with self._bytes_lock:
                last_written = self._written_items

            while self._thread.is_alive():
                now = time.monotonic()
                if now >= hard_deadline or now >= stall_deadline:
                    break
                step = min(0.25, hard_deadline - now, stall_deadline - now)
                self._thread.join(timeout=max(0.01, step))
                with self._bytes_lock:
                    cur_written = self._written_items
                if cur_written != last_written:
                    last_written = cur_written
                    stall_deadline = time.monotonic() + eff_stall

        if self._thread.is_alive():
            self._abort_flag = True
            self._thread.join(timeout=0.5)
        if self._thread.is_alive():
            err = TimeoutError(f"StreamingH5Writer stalled / did not finish draining within {timeout:.1f}s")
            self._error = self._error or err
        return self._error is None, set(self._streamed_containers), self._error

    # ------------------------------------------------------------------
    # Writer thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                container_id, sample, repeat, nbytes = item
                try:
                    if self._error is None and not self._abort_flag:
                        self._write_item(container_id, sample, repeat)
                except Exception as exc:
                    self._error = exc
                    logger.error(f"StreamingH5Writer failed while writing '{container_id}': {exc}")
                finally:
                    with self._bytes_lock:
                        self._queued_bytes = max(0, self._queued_bytes - nbytes)
                        self._written_items += 1
            if self._error is None and not self._abort_flag:
                self._h5_file.flush()
        except Exception as exc:
            self._error = self._error or exc
            logger.error(f"StreamingH5Writer fatal error: {exc}")

    def _ensure_demo_group(self) -> h5py.Group:
        if self._demo_group is None:
            self._demo_group = self._h5_file.require_group(f"data/{self._demo_name}")
        return self._demo_group

    def _create_state(self, container_id: str, sample: dict, arr: np.ndarray) -> _ContainerStreamState:
        demo_group = self._ensure_demo_group()
        group = demo_group.require_group(container_id)
        data_type = sample["type"]
        state = _ContainerStreamState(data_type, group, arr.shape, arr.dtype)
        options = user_settings.h5_compression()

        if data_type == "image":
            group.attrs["image_width"] = sample.get("width", 0)
            group.attrs["image_height"] = sample.get("height", 0)
            group.attrs["image_encoding"] = sample.get("encoding", "unknown")
            ds_name = "images"
        else:
            group.attrs["data_type"] = "pointcloud"
            group.attrs["width"] = sample.get("width", 0)
            group.attrs["height"] = sample.get("height", 0)
            group.attrs["fields"] = json.dumps(sample.get("fields", []))
            group.attrs["organized"] = True
            ds_name = "points"

        state.dataset = group.create_dataset(
            ds_name,
            shape=(0, *arr.shape),
            maxshape=(None, *arr.shape),
            dtype=arr.dtype,
            chunks=(1, *arr.shape),
            **options,
        )
        self._states[container_id] = state
        return state

    def _write_item(self, container_id: str, sample: dict, repeat: int) -> None:
        arr = _sample_array(sample)
        state = self._states.get(container_id)
        if state is None:
            state = self._create_state(container_id, sample, arr)

        if arr.shape == state.shape and arr.dtype == state.dtype and state.dataset is not None:
            start = state.count
            new_len = start + repeat
            state.dataset.resize(new_len, axis=0)
            for idx in range(start, new_len):
                state.dataset[idx] = arr
            state.count = new_len
        elif state.data_type == "image":
            # Mid-demo resolution/dtype change: mirror the batch writer by saving per-frame datasets.
            if not state.inconsistent_warned:
                state.inconsistent_warned = True
                logger.warning(
                    f"'{container_id}': image shape/dtype changed mid-demo "
                    f"(expected {state.shape}/{state.dtype}, got {arr.shape}/{arr.dtype}); "
                    "saving mismatched frames individually"
                )
            options = user_settings.h5_compression()
            for offset in range(repeat):
                state.group.create_dataset(f"image_{state.count + offset}", data=arr, **options)
            state.count += repeat
        else:
            if not state.inconsistent_warned:
                state.inconsistent_warned = True
                logger.warning(
                    f"'{container_id}': organized point cloud shape changed mid-demo "
                    f"(expected {state.shape}, got {arr.shape}); skipping mismatched frames."
                )
            state.count += repeat

        self._writes_since_flush += repeat
        if self._writes_since_flush >= self._flush_every:
            self._h5_file.flush()
            self._writes_since_flush = 0
