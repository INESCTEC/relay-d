"""LeRobot inference backend: backend detection, state order, camera conversion, and the runner with fake LeRobot parts."""

import json
import types

import numpy as np
import pytest

from relay_d.dispatch.api import detect_backend
from relay_d.dispatch.lerobot_runner import (
    STATE,
    build_batch,
    dataset_info,
    resolve_checkpoint,
    state_layout_from_names,
)


def _lerobot_folder(root, with_subfolder=False):
    folder = root / "pretrained_model" if with_subfolder else root
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text("{}")
    return folder


def test_backend_is_detected_from_the_checkpoint(tmp_path):
    ck = _lerobot_folder(tmp_path / "run" / "pretrained_model")
    assert detect_backend(ck) == "lerobot"
    last = tmp_path / "checkpoints" / "last"
    _lerobot_folder(last, with_subfolder=True)
    assert detect_backend(last) == "lerobot" and resolve_checkpoint(last) == last / "pretrained_model"
    pth = tmp_path / "model.pth"
    pth.write_bytes(b"x")
    assert detect_backend(pth) == "robomimic"
    with pytest.raises(ValueError, match="cannot tell the backend"):
        detect_backend(tmp_path / "nothing")
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError):
        detect_backend(tmp_path / "notes.txt")


def test_state_layout_is_read_back_from_the_names_the_converter_writes():
    names = ["robot_joint_pos.0", "robot_joint_pos.1", "robot_joint_pos.2", "gripper", "eef.0", "eef.1"]
    assert state_layout_from_names(names) == [("robot_joint_pos", 3), ("gripper", 1), ("eef", 2)]
    assert state_layout_from_names(["a", "b"]) == [("a", 1), ("b", 1)]
    assert state_layout_from_names([]) == []


def test_dataset_info_is_found_through_train_config(tmp_path):
    ck = _lerobot_folder(tmp_path / "pretrained_model")
    assert dataset_info(ck) is None  # no train_config.json
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": {STATE: {"names": ["a.0", "a.1"]}}}))
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(root)}}))
    assert dataset_info(ck)["features"][STATE]["names"] == ["a.0", "a.1"]
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(tmp_path / "gone")}}))
    assert dataset_info(ck) is None  # the dataset was moved or deleted


def _frame(h, w, bgr=(10, 20, 30)):
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = bgr
    return frame


def test_build_batch_orders_the_state_and_converts_the_camera():
    obs = {"b": np.array([3.0, 4.0]), "a": np.array([1.0, 2.0]), "cam": _frame(4, 6)}
    batch = build_batch(obs, ["a", "b"], {"cam": ("observation.images.cam", (3, 4, 6))})
    assert batch[STATE].tolist() == [1.0, 2.0, 3.0, 4.0] and batch[STATE].dtype == np.float32
    image = batch["observation.images.cam"]
    assert image.shape == (3, 4, 6) and image.dtype == np.float32  # CHW
    assert image[:, 0, 0].tolist() == pytest.approx([30 / 255, 20 / 255, 10 / 255])  # BGR input -> RGB channels, in [0, 1]


def test_build_batch_passes_the_task_sentence_only_when_there_is_one():
    obs = {"a": np.array([1.0])}
    assert "task" not in build_batch(obs, ["a"], {})
    assert build_batch(obs, ["a"], {}, task="pick the cube")["task"] == "pick the cube"


def test_dataset_task_is_read_from_the_training_dataset(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from relay_d.dispatch.lerobot_runner import dataset_task

    ck = _lerobot_folder(tmp_path / "pretrained_model")
    assert dataset_task(ck) is None  # no train_config.json
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(root)}}))
    pq.write_table(pa.table({"task_index": [0], "task": ["stack the cubes"]}), root / "meta" / "tasks.parquet")
    assert dataset_task(ck) == "stack the cubes"
    pq.write_table(pa.table({"task_index": [0, 1], "task": ["a", "b"]}), root / "meta" / "tasks.parquet")
    assert dataset_task(ck) is None  # several tasks: which one is ambiguous, so --task must say


def test_build_batch_crops_an_odd_frame_like_the_dataset_writer_did_and_rejects_other_sizes():
    obs = {"s": np.zeros(1), "cam": _frame(5, 7)}
    assert build_batch(obs, ["s"], {"cam": ("observation.images.cam", (3, 4, 6))})["observation.images.cam"].shape == (3, 4, 6)
    with pytest.raises(ValueError, match="trained on"):
        build_batch({"s": np.zeros(1), "cam": _frame(8, 12)}, ["s"], {"cam": ("observation.images.cam", (3, 4, 6))})
    with pytest.raises(KeyError, match="missing"):
        build_batch({"s": np.zeros(1)}, ["s"], {"cam": ("observation.images.cam", (3, 4, 6))})


def test_runner_feeds_lerobot_and_returns_a_flat_float32_action(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from relay_d.dispatch import lerobot_runner as lr

    seen = {}

    class FakePolicy:
        def eval(self):
            return self

        def reset(self):
            seen["policy_reset"] = seen.get("policy_reset", 0) + 1

        def select_action(self, batch):
            seen["batch"] = batch
            return torch.full((1, 3), 2.0)

    class FakePipeline:
        def __init__(self, scale=1.0):
            self.scale, self.resets = scale, 0

        def __call__(self, x):
            return {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in x.items()} if isinstance(x, dict) else x * self.scale

        def reset(self):
            self.resets += 1

    feature = lambda shape: types.SimpleNamespace(shape=shape)  # noqa: E731
    cfg = types.SimpleNamespace(
        type="act", device="cpu",
        input_features={STATE: feature((3,)), "observation.images.cam": feature((3, 4, 6))},
    )
    _lerobot_folder(tmp_path / "pretrained_model")
    ck = tmp_path / "pretrained_model"

    fake = types.ModuleType("fake")
    monkeypatch.setitem(__import__("sys").modules, "lerobot", fake)
    configs = types.ModuleType("lerobot.configs.policies")
    configs.PreTrainedConfig = types.SimpleNamespace(from_pretrained=lambda path: cfg)
    factory = types.ModuleType("lerobot.policies.factory")
    factory.get_policy_class = lambda name: types.SimpleNamespace(from_pretrained=lambda path, config: FakePolicy())
    pipelines = (FakePipeline(), FakePipeline(10.0))
    factory.make_pre_post_processors = lambda **kw: pipelines
    for name, module in (("lerobot.configs", types.ModuleType("lerobot.configs")), ("lerobot.configs.policies", configs),
                         ("lerobot.policies", types.ModuleType("lerobot.policies")), ("lerobot.policies.factory", factory)):
        monkeypatch.setitem(__import__("sys").modules, name, module)

    runner = lr.LeRobotPolicyRunner(str(ck), device="cpu", fallback_state_keys=["x", "y"])
    assert runner.obs_keys == ["x", "y", "cam"] and seen["policy_reset"] == 1 and pipelines[0].resets == 1
    action = runner.get_action({"x": np.array([1.0]), "y": np.array([2.0, 3.0]), "cam": _frame(4, 6)})
    assert action.shape == (3,) and action.dtype == np.float32 and action.tolist() == [20.0, 20.0, 20.0]  # postprocessor x10
    assert seen["batch"][STATE].shape == (1, 3) and seen["batch"]["observation.images.cam"].shape == (1, 3, 4, 6)
    with pytest.raises(ValueError, match="expects 3"):  # a state of the wrong width names both sides
        runner.get_action({"x": np.array([1.0]), "y": np.array([2.0]), "cam": _frame(4, 6)})
    with pytest.raises(FileNotFoundError, match="not a LeRobot checkpoint"):
        lr.LeRobotPolicyRunner(str(tmp_path / "nowhere"))
    assert runner.fps is None  # no training dataset to read the rate from

    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 15, "features": {STATE: {"names": ["x", "y.0", "y.1"]}}}))
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(root)}}))
    runner = lr.LeRobotPolicyRunner(str(ck), device="cpu")
    assert runner.fps == 15 and runner.state_keys == ["x", "y"]  # the control rate the policy learned

    class TokenizerProcessorStep:  # what a language-conditioned policy (SmolVLA) has in its preprocessor
        pass

    pipelines[0].steps = [TokenizerProcessorStep()]
    with pytest.raises(ValueError, match="--task"):  # no dataset task to fall back on: fail at startup, not mid-run
        lr.LeRobotPolicyRunner(str(ck), device="cpu")
    runner = lr.LeRobotPolicyRunner(str(ck), device="cpu", task="stack the cubes")
    runner.get_action({"x": np.array([1.0]), "y": np.array([2.0, 3.0]), "cam": _frame(4, 6)})
    assert seen["batch"]["task"] == "stack the cubes"

    # fine-tuned from a pretrained model: its config keeps that model's inputs (6 values, camera1 at 256x256), the
    # real inputs are the dataset's, fed under the dataset's camera name (the preprocessor renames it to camera1)
    cfg.input_features = {STATE: feature((6,)), "observation.images.camera1": feature((3, 256, 256))}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 15, "features": {
        STATE: {"names": ["x", "y.0", "y.1"], "shape": [3]},
        "observation.images.cam": {"dtype": "video", "shape": [4, 6, 3]}}}))
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(root)}, "policy": {"pretrained_path": "lerobot/smolvla_base"}}))
    runner = lr.LeRobotPolicyRunner(str(ck), device="cpu", task="stack the cubes")
    assert runner.obs_keys == ["x", "y", "cam"]
    runner.get_action({"x": np.array([1.0]), "y": np.array([2.0, 3.0]), "cam": _frame(4, 6)})
    assert seen["batch"][STATE].shape == (1, 3) and seen["batch"]["observation.images.cam"].shape == (1, 3, 4, 6)
    (ck / "train_config.json").write_text(json.dumps({"dataset": {"root": str(tmp_path / "gone")}, "policy": {"pretrained_path": "x"}}))
    with pytest.raises(ValueError, match="training dataset"):  # without the dataset its real inputs are unknown
        lr.LeRobotPolicyRunner(str(ck), device="cpu", task="t")


def test_old_custom_backend_setting_falls_back_to_auto(settings_file):
    from relay_d.utils import user_settings as us

    settings_file.write_text(json.dumps({"infer_backend": "custom"}))
    assert us.reload()["infer_backend"] == "auto"
