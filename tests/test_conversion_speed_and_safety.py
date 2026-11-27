"""Tests for Workstreams 2 & 4: conversion speed (direct chunk copy, single-pass + parallel LeRobot) and safety."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter


def _make_demo_h5(path, n_frames=30, cams=("cam0", "cam1"), nan_stream=False, compression="gzip", compression_opts=4):
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_a")
        g.create_dataset("timestamps", data=np.arange(n_frames, dtype=np.int64) * 50_000_000)
        sg = g.create_group("streams")
        state = rng.standard_normal((n_frames, 4)).astype(np.float32)
        if nan_stream:
            state[5, 2] = np.nan
        sg.create_dataset("state", data=state)
        sg.create_dataset("act", data=rng.standard_normal((n_frames, 4)).astype(np.float32))
        for idx, cam in enumerate(cams):
            cg = g.create_group(cam)
            opts = {"compression": compression} if compression else {}
            if compression == "gzip" and compression_opts is not None:
                opts["compression_opts"] = compression_opts
            frames = rng.integers(0, 255, size=(n_frames, 33, 48, 3), dtype=np.uint8)
            # Tag channel 0 (blue in BGR) distinctly per camera for RGB-conversion checks
            frames[..., 0] = 200 if idx == 0 else 40
            frames[..., 2] = 30 if idx == 0 else 210
            cg.create_dataset("images", data=frames, chunks=(1, 33, 48, 3), **opts)


def test_robomimic_direct_chunk_copy_bit_identical_and_fallback(tmp_path, monkeypatch):
    from relay_d.utils import user_settings

    src_same = tmp_path / "same.h5"
    src_diff = tmp_path / "diff.h5"
    _make_demo_h5(src_same, n_frames=12, cams=("cam0",), compression="gzip", compression_opts=4)
    _make_demo_h5(src_diff, n_frames=12, cams=("cam0",), compression="lzf", compression_opts=None)

    monkeypatch.setattr(user_settings, "h5_compression", lambda: {"compression": "gzip", "compression_opts": 4})

    cfg = {
        "recording_frequency": 20,
        "dataset_name": "test",
        "actions_to_extract": ["act"],
        "output": {"cam0": "cam0/images", "streams": {"state": ["s.x"], "act": ["a.x"]}},
    }

    # Spy on direct chunk copy path
    direct_calls = []
    orig_can = YAMLDrivenConverter._can_direct_chunk_copy

    def spy_can(s, d):
        res = orig_can(s, d)
        direct_calls.append(res)
        return res

    monkeypatch.setattr(YAMLDrivenConverter, "_can_direct_chunk_copy", staticmethod(spy_can))

    out1 = tmp_path / "out_same.h5"
    conv = YAMLDrivenConverter(cfg)
    assert conv.convert_multiple_files("robomimic", [str(src_same)], str(out1)) is True
    assert direct_calls == [True]
    assert not (tmp_path / "out_same.h5.partial").exists()
    with h5py.File(src_same, "r") as fs, h5py.File(out1, "r") as fd:
        np.testing.assert_array_equal(fs["data/demo_a/cam0/images"][:], fd["data/demo_0/obs/cam0"][:])

    # Different compression falls back seamlessly to slice copy and still matches bit-for-bit
    out2 = tmp_path / "out_diff.h5"
    assert conv.convert_multiple_files("robomimic", [str(src_diff)], str(out2)) is True
    assert direct_calls == [True, False]
    with h5py.File(src_diff, "r") as fs, h5py.File(out2, "r") as fd:
        np.testing.assert_array_equal(fs["data/demo_a/cam0/images"][:], fd["data/demo_0/obs/cam0"][:])


def test_robomimic_atomic_partial_cleanup_on_failure(tmp_path, monkeypatch):
    src = tmp_path / "demo.h5"
    _make_demo_h5(src, n_frames=10, cams=("cam0",))
    cfg = {
        "recording_frequency": 20,
        "dataset_name": "test",
        "actions_to_extract": ["act"],
        "output": {"cam0": "cam0/images", "streams": {"state": ["s.x"], "act": ["a.x"]}},
    }
    conv = YAMLDrivenConverter(cfg)
    monkeypatch.setattr(conv, "_create_data_mask", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tmp_path / "dataset.h5"
    assert conv.convert_multiple_files("robomimic", [str(src)], str(out)) is False
    assert not out.exists()
    assert not (tmp_path / "dataset.h5.partial").exists()


def test_converter_skips_demo_with_nan_stream(tmp_path):
    good = tmp_path / "good.h5"
    bad = tmp_path / "bad.h5"
    _make_demo_h5(good, n_frames=10, cams=(), nan_stream=False)
    _make_demo_h5(bad, n_frames=10, cams=(), nan_stream=True)
    cfg = {
        "recording_frequency": 20,
        "dataset_name": "test",
        "actions_to_extract": ["act"],
        "output": {"streams": {"state": ["s.x"], "act": ["a.x"]}},
    }
    conv = YAMLDrivenConverter(cfg)
    out = tmp_path / "out.h5"
    assert conv.convert_multiple_files("robomimic", [str(good), str(bad)], str(out)) is True
    assert conv.converted_demos == 1
    assert len(conv.skipped_demos) == 1


def test_lerobot_single_pass_parallel_cameras_stats_and_atomicity(tmp_path):
    av = pytest.importorskip("av")
    from relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer import (
        LeRobotDatasetWriter,
        LeRobotWriterError,
        _Frames,
        _aggregate_stats,
        _image_stats,
    )

    rng = np.random.default_rng(7)
    n = 140  # > 100 (_MAX_STAT_FRAMES) and > 2*64 batches
    state = rng.normal(size=(n, 5)).astype(np.float32)
    action = rng.normal(size=(n, 3)).astype(np.float32)
    cam0 = rng.integers(0, 255, size=(n, 65, 80, 3), dtype=np.uint8)
    cam1 = rng.integers(0, 255, size=(n, 64, 80, 3), dtype=np.uint8)
    # Distinct BGR values to verify BGR->RGB conversion in both video and stats
    cam0[..., 0] = 240  # Blue in BGR -> channel 2 in RGB
    cam0[..., 2] = 15   # Red in BGR  -> channel 0 in RGB

    class _ReadCounter:
        def __init__(self, arr):
            self.arr, self.shape, self.dtype, self.total_read = arr, arr.shape, arr.dtype, 0

        def __getitem__(self, idx):
            out = self.arr[idx]
            self.total_read += len(out) if out.ndim == 4 else 1
            return out

    src0, src1 = _ReadCounter(cam0), _ReadCounter(cam1)
    root = tmp_path / "lr_ds"
    with LeRobotDatasetWriter(root, fps=20) as w:
        w.add_episode(
            state,
            action,
            {"observation.images.cam0": src0, "observation.images.cam1": src1},
            task="test task",
        )
        # While still open (before finalize), final root must NOT contain a half-written dataset.
        assert not (root / "meta/info.json").exists()

    # Single-pass guarantee: every frame read exactly once per camera (no double HDF5 pass!).
    assert src0.total_read == n and src1.total_read == n

    # Exact parity with reference _image_stats()
    ref0 = _image_stats(_Frames(cam0, True, 64, 80))
    ref1 = _image_stats(_Frames(cam1, True, 64, 80))
    stats = json.loads((root / "meta/stats.json").read_text())
    for key in ("min", "max", "mean", "std", "q01", "q50", "q99"):
        np.testing.assert_allclose(stats["observation.images.cam0"][key], ref0[key].tolist(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stats["observation.images.cam1"][key], ref1[key].tolist(), rtol=1e-6, atol=1e-6)

    # Decoded video frames must have blue in RGB channel 2 (verifying bgr24 -> yuv420p -> rgb24 round-trip)
    v0 = root / "videos/observation.images.cam0/chunk-000/file-000.mp4"
    with av.open(str(v0)) as container:
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(decoded) == n
    assert decoded[0][..., 2].mean() > 200 and decoded[0][..., 0].mean() < 40

    # No leftover staging directories next to root
    assert [p for p in tmp_path.iterdir() if ".partial_" in p.name] == []

    # Reject NaN/Inf in state or action before writing
    bad_state = state.copy()
    bad_state[3, 1] = np.nan
    with LeRobotDatasetWriter(tmp_path / "lr_nan", fps=20) as w:
        with pytest.raises(LeRobotWriterError, match="NaN/Inf"):
            w.add_episode(bad_state, action, None, task="t")
        w.add_episode(state, action, None, task="t")

    # Negative-zero variance guard in _aggregate_stats
    ep = {
        "min": np.array([1.0]),
        "max": np.array([1.0]),
        "mean": np.array([1.0]),
        "std": np.array([0.0]),
        "count": np.array([10]),
        "q01": np.array([1.0]),
        "q10": np.array([1.0]),
        "q50": np.array([1.0]),
        "q90": np.array([1.0]),
        "q99": np.array([1.0]),
    }
    agg = _aggregate_stats([ep, ep])
    assert np.isfinite(agg["std"]).all() and agg["std"][0] == 0.0

    # Exact parity of 256-bin uint8 histogram stats against float64 np.quantile
    from relay_d.acquisition.qt_app.ui.postprocess.lerobot_writer import _QUANTILES, _QUANTILE_KEYS, _stats_from_rgb_samples

    sample = rng.integers(0, 256, size=(13, 37, 41, 3), dtype=np.uint8)
    fast = _stats_from_rgb_samples(sample, 13)
    flat = (sample.astype(np.float64) / 255.0).transpose(3, 0, 1, 2).reshape(3, -1)
    np.testing.assert_allclose(fast["min"].reshape(-1), flat.min(axis=1), atol=1e-12)
    np.testing.assert_allclose(fast["max"].reshape(-1), flat.max(axis=1), atol=1e-12)
    np.testing.assert_allclose(fast["mean"].reshape(-1), flat.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(fast["std"].reshape(-1), flat.std(axis=1), atol=1e-12)
    q_ref = np.quantile(flat, _QUANTILES, axis=1)
    for k, row in zip(_QUANTILE_KEYS, q_ref):
        np.testing.assert_allclose(fast[k].reshape(-1), row, atol=1e-12)
