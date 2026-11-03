"""postprocess_config.yaml: format options kept apart from the acquisition config."""

import json

import h5py
import pytest
import yaml

pytest.importorskip("av")

from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess import postprocess_options as po  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import (  # noqa: E402
    PostprocessConfigLoader,
)
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import (  # noqa: E402
    YAMLDrivenConverter,
)
from helpers import DATA_CONFIG, LENGTHS, write_recording  # noqa: E402

PostprocessOptions = po.PostprocessOptions

FILE_OPTIONS = {
    "postprocess": {
        "robomimic": {
            "training_model": "diffusion_policy",
            "splits": {"train_frac": 0.5, "val_frac": 0.5},
            "metadata": {"env_name": "FromFile", "env_version": "9", "type": 1, "env_kwargs": {"a": 1}},
        },
        "lerobot": {"task": "task from file", "robot_type": "iiwa"},
    }
}


@pytest.fixture(autouse=True)
def _reset_warned():
    po._warned_legacy.clear()


def _write(path, data):
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_file_values_win_over_legacy_and_defaults(tmp_path):
    opts, problems = PostprocessOptions.load(_write(tmp_path / "postprocess_config.yaml", FILE_OPTIONS))
    assert opts is not None and problems == []
    legacy = {"metadata": {"env_name": "Legacy"}, "training_model": "bc_rnn_gmm", "train_frac": 0.1}
    assert opts.robomimic_metadata(legacy)["env_name"] == "FromFile"
    assert opts.training_model(legacy) == "diffusion_policy"
    assert opts.splits(legacy) == {"train_frac": 0.5, "val_frac": 0.5, "test_frac": 0.0}
    assert opts.lerobot({"lerobot": {"task": "legacy", "state_streams": ["x"]}}) == {
        "task": "task from file", "robot_type": "iiwa", "state_streams": ["x"],
    }


def test_legacy_fallback_warns_once(caplog):
    legacy = {"metadata": {"env_name": "Legacy"}, "training_model": "bc_transformer_gmm", "val_frac": 0.3}
    opts = PostprocessOptions()
    messages = []
    import relay_d.acquisition.qt_app.ui.postprocess.postprocess_options as mod

    mod.logger = type("L", (), {"warning": staticmethod(messages.append), "info": staticmethod(lambda *_: None)})
    try:
        assert opts.robomimic_metadata(legacy)["env_name"] == "Legacy"
        assert opts.robomimic_metadata(legacy)["env_name"] == "Legacy"
        assert opts.training_model(legacy) == "bc_transformer_gmm"
        assert opts.splits(legacy)["val_frac"] == 0.3
    finally:
        import relay_d.utils.coloring_logger as cl

        mod.logger = cl.logger
    assert sum("'metadata'" in m for m in messages) == 1  # warned once, not per call
    assert any("deprecated" in m and "postprocess_config.yaml" in m for m in messages)


def test_defaults_when_nothing_set():
    opts = PostprocessOptions()
    assert opts.robomimic_metadata({}) == {}
    assert opts.training_model({}) == "bc_rnn_gmm"
    assert opts.splits({}) == {"train_frac": 0.8, "val_frac": 0.2, "test_frac": 0.0}
    assert opts.lerobot({}) == {}


def test_load_rejects_bad_files(tmp_path):
    assert PostprocessOptions.load(str(tmp_path / "nope.yaml"))[0] is None
    (tmp_path / "bad.yaml").write_text("a: [unclosed")
    assert PostprocessOptions.load(str(tmp_path / "bad.yaml"))[0] is None
    assert PostprocessOptions.load(_write(tmp_path / "x.yaml", {"config": {}}))[0] is None
    assert PostprocessOptions.load(_write(tmp_path / "y.yaml", {"postprocess": {"lerobot": 3}}))[0] is None
    opts, problems = PostprocessOptions.load(_write(tmp_path / "z.yaml", {"postprocess": {"foo": {}}}))
    assert opts is not None and problems and "unknown format section" in problems[0]


def test_find_sibling(tmp_path):
    cfg = _write(tmp_path / "acq.yaml", {"config": {}})
    assert PostprocessOptions.find_sibling(cfg) is None
    sibling = _write(tmp_path / "postprocess_config.yaml", FILE_OPTIONS)
    assert PostprocessOptions.find_sibling(cfg) == sibling
    assert PostprocessOptions.find_sibling(None) is None


def test_templates_are_valid():
    opts, problems = PostprocessOptions.load("relay_d/acquisition/qt_app/ui/config/templates/postprocess_config_template.yaml")
    assert opts is not None and problems == []
    assert opts.robomimic_metadata({})["env_name"] == "Lift"
    acq = yaml.safe_load(open("relay_d/acquisition/qt_app/ui/config/templates/config_template.yaml"))["config"]
    for legacy_key in ("metadata", "training_model", "train_frac", "lerobot"):
        assert legacy_key not in acq


def _converter(config=None):
    cfg = json.loads(json.dumps(config or DATA_CONFIG))
    return YAMLDrivenConverter(PostprocessConfigLoader(), data_config=cfg)


def test_robomimic_uses_options_file(tmp_path):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    conv = _converter()
    conv.options, _ = PostprocessOptions.load(_write(tmp_path / "postprocess_config.yaml", FILE_OPTIONS))
    out = tmp_path / "rm.h5"
    assert conv.convert_multiple_files("robomimic", [str(rec)], str(out))
    with h5py.File(out, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        assert env_args["env_name"] == "FromFile"
        # train 0.5 / valid 0.5 of 2 demos
        assert len(f["mask/train"]) == 1 and len(f["mask/valid"]) == 1


def test_robomimic_legacy_config_still_works(tmp_path):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    cfg = json.loads(json.dumps(DATA_CONFIG))
    cfg["metadata"] = {"env_name": "LegacyEnv", "env_version": "1", "type": 1, "env_kwargs": {}}
    conv = _converter(cfg)
    out = tmp_path / "rm.h5"
    assert conv.convert_multiple_files("robomimic", [str(rec)], str(out))
    with h5py.File(out, "r") as f:
        assert json.loads(f["data"].attrs["env_args"])["env_name"] == "LegacyEnv"


def test_lerobot_uses_options_file(tmp_path):
    rec = tmp_path / "recording_0.h5"
    write_recording(rec, LENGTHS)
    conv = _converter()
    conv.options, _ = PostprocessOptions.load(_write(tmp_path / "postprocess_config.yaml", FILE_OPTIONS))
    out = tmp_path / "lr"
    assert conv.convert_multiple_files("lerobot", [str(rec)], str(out))
    info = json.loads((out / "meta/info.json").read_text())
    assert info["robot_type"] == "iiwa"
    import pyarrow.parquet as pq

    assert pq.read_table(out / "meta/tasks.parquet")["task"].to_pylist() == ["task from file"]


def test_get_train_model_name_with_options(tmp_path):
    p = YamlParser()
    p.yaml_data = {"config": {"training_model": "legacy_model"}}
    assert p.get_train_model_name() == "legacy_model"
    opts, _ = PostprocessOptions.load(_write(tmp_path / "o.yaml", FILE_OPTIONS))
    assert p.get_train_model_name(opts) == "diffusion_policy"


def test_get_obs_config_without_sync_tolerance():
    p = YamlParser()
    p.yaml_data = {"config": {"input": {}, "output": {"streams": {}}}}
    p.yaml_input_data_raw = {}
    cfg = p.get_obs_config()["config"]
    assert "sync" not in cfg and "observations" in cfg
    p.yaml_data["config"]["sync_tolerance_sec"] = 0.05
    assert p.get_obs_config()["config"]["sync"] == {"threshold": 0.05}


def test_output_folder_and_dataset_are_named_after_the_format():
    from datetime import datetime

    from relay_d.acquisition.qt_app.ui.postprocess.postprocess_options import dataset_output_paths

    now = datetime(2026, 11, 2, 15, 10, 0)
    folder, dataset = dataset_output_paths("/rec", "robomimic", now)
    assert folder == "/rec/2026-11-02_15-10-00_Robomimic"
    assert dataset == f"{folder}/dataset_{int(now.timestamp())}_Robomimic.h5"
    folder, dataset = dataset_output_paths("/rec", "lerobot", now)
    assert folder == "/rec/2026-11-02_15-10-00_Lerobot"
    assert dataset == f"{folder}/dataset_{int(now.timestamp())}_Lerobot"  # a LeRobot dataset is a folder: no extension
