"""Helpers shared by several test modules (`from helpers import ...`: conftest.py puts tests/ on sys.path)."""

import json  # noqa: F401  (used by the recording writer below)
import time
import types

import h5py
import numpy as np


def wait_until(cond, seconds=10):
    """Poll ``cond()`` until it is true (or ``seconds`` pass); returns the last value."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


# --- a small recording and its acquisition config (used by the conversion tests)
FPS = 10
LENGTHS = [14, 9]

DATA_CONFIG = {
    "dataset_name": "stack_cubes",
    "recording_frequency": FPS,
    "actions_to_extract": ["robot_joint_pos"],
    "input": {"camera_rgb": {"topic": "/cam", "modality": "rgb"}},
    "output": {
        "agentview_image": "camera_rgb/images",
        "streams": {
            "robot_joint_pos": ["joint_1.position[:]"],
            "eef_translation": [
                "tf_eef.translation.x",
                "tf_eef.translation.y",
                "tf_eef.translation.z",
            ],
        },
    },
}


def write_recording(path, lengths, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, n in enumerate(lengths):
            demo = data.create_group(f"demo_{i}")
            streams = demo.create_group("streams")
            streams.create_dataset("robot_joint_pos", data=rng.normal(size=(n, 4)).astype(np.float32))
            streams.create_dataset("eef_translation", data=rng.normal(size=(n, 3)).astype(np.float32))
            cam = demo.create_group("camera_rgb")
            cam.create_dataset("images", data=rng.integers(0, 255, (n, 24, 32, 3), dtype=np.uint8))


# --- DataRecorder helpers (the recorder_cls fixture is in conftest.py)
def make_recorder(cls, tmp_path, timestamps, recorded):
    parser = types.SimpleNamespace(is_loaded=lambda: False, yaml_data=None)
    rec = cls(parser)
    rec.h5_file = h5py.File(tmp_path / "demo.h5", "w")
    rec.recording_metadata = {"demo_name": "d"}
    rec.timestamps = timestamps
    rec.recorded_data.update(recorded)
    return rec


def image_sample():
    return {"type": "image", "data": np.zeros((2, 2, 3), np.uint8), "width": 2, "height": 2, "encoding": "rgb8"}


def joint_sample(n=1):
    return {"type": "joint_state", "names": ["a"] * n, "positions": [1.0] * n, "velocities": [], "efforts": []}
