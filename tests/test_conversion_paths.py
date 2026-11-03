"""The GUI Post-process page, AppAPI and relay-d-convert write the same dataset and the same files next to it."""

import os
from datetime import datetime
from types import SimpleNamespace

import h5py
import pytest
import yaml

pytest.importorskip("av")

from relay_d.acquisition.convert_cli import main as cli_main  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess import offline_config  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter  # noqa: E402
from helpers import DATA_CONFIG, LENGTHS, write_recording  # noqa: E402

SIDECARS = ["acquisition_config.yaml", "action_dispatcher/action_config.yaml",
            "action_dispatcher/inference_script.py", "action_dispatcher/obs_config.yaml"]


@pytest.fixture
def workspace(tmp_path):
    rec_dir = tmp_path / "recordings" / "2026-11-03_10-00-00_Recording"
    rec_dir.mkdir(parents=True)
    write_recording(rec_dir / "stack_cubes_1.h5", LENGTHS)
    cfg = tmp_path / "configs" / "acquisition.yaml"
    cfg.parent.mkdir()
    cfg.write_text("# my comment\n" + yaml.safe_dump({"config": DATA_CONFIG}))
    return tmp_path, cfg, rec_dir


def _files(folder):
    return sorted(os.path.relpath(os.path.join(r, f), folder) for r, _, fs in os.walk(folder) for f in fs)


# --- B1: one writer for the files next to a dataset
@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_cli_writes_the_dispatcher_files_where_the_launcher_looks(workspace, fmt):
    tmp, cfg, rec_dir = workspace
    out = tmp / "out" / ("ds.h5" if fmt == "robomimic" else "ds")
    assert cli_main(["--config", str(cfg), "--format", fmt, "--input", str(rec_dir), "--output", str(out)]) == 0
    sidecars = [f for f in _files(tmp / "out") if not f.startswith("ds")]
    assert sidecars == SIDECARS  # next to the .h5 / in the folder above the LeRobot folder (launcher _config_file)
    assert (tmp / "out" / "acquisition_config.yaml").read_bytes() == cfg.read_bytes()  # byte copy, comments kept


def test_cli_keeps_the_dataset_when_the_extra_files_fail(workspace, monkeypatch, capsys):
    tmp, cfg, rec_dir = workspace

    def boom(folder):
        raise OSError("disk full")

    monkeypatch.setattr(offline_config, "save_dispatch_templates", boom)
    out = tmp / "out" / "ds.h5"
    assert cli_main(["--config", str(cfg), "--format", "robomimic", "--input", str(rec_dir), "--output", str(out)]) == 0
    assert out.exists()
    err = capsys.readouterr().err
    assert "disk full" in err and "Traceback" not in err


def test_acquisition_config_is_dumped_when_the_parser_was_edited_after_loading(workspace, tmp_path):
    _, cfg, _ = workspace
    parser, _ = offline_config.load_offline_config(str(cfg))
    parser.yaml_data["config"]["dataset_name"] = "edited"
    offline_config.write_dataset_files(str(tmp_path / "o"), parser)
    assert yaml.safe_load((tmp_path / "o" / "acquisition_config.yaml").read_text())["config"]["dataset_name"] == "edited"


# --- B2: AppAPI converts like the GUI
def _api(cfg, rec_dir, options=None):
    pytest.importorskip("rclpy")
    from relay_d.acquisition.api.api import AppAPI

    api = AppAPI.__new__(AppAPI)  # no ROS: only the conversion part
    api._load_conversion_config(str(cfg), options)
    api._session_h5_files = [str(p) for p in sorted(rec_dir.glob("*.h5"))]
    api._data_recorder = SimpleNamespace(recorded_data_dir=str(rec_dir))
    return api


@pytest.mark.parametrize("fmt", ["robomimic", "lerobot"])
def test_api_writes_a_new_dataset_folder_with_the_dispatcher_files(workspace, fmt, monkeypatch):
    tmp, cfg, rec_dir = workspace
    api = _api(cfg, rec_dir)
    from relay_d.acquisition.api import api as api_mod

    now = datetime(2026, 11, 3, 11, 0, 0)
    monkeypatch.setattr(api_mod, "datetime", SimpleNamespace(now=lambda: now))
    assert api.convert_recording_to_dataset(fmt)
    out_dir = tmp / "recordings" / now.strftime("%Y-%m-%d_%H-%M-%S" + ("_Robomimic" if fmt == "robomimic" else "_Lerobot"))
    assert [f for f in _files(out_dir) if not f.startswith("dataset_")] == SIDECARS
    assert _files(rec_dir) == ["stack_cubes_1.h5"]  # nothing written into the recording session


def test_api_refuses_a_bad_options_file(workspace):
    tmp, cfg, rec_dir = workspace
    bad = tmp / "opts.yaml"
    bad.write_text("postprocess:\n  robomimic:\n    training_model: bc\n")
    with pytest.raises(ValueError, match="training_model"):
        _api(cfg, rec_dir, str(bad))


def test_api_shutdown_converts_to_the_default_format_setting(settings_file):
    pytest.importorskip("rclpy")
    from relay_d.acquisition.api.api import AppAPI
    from relay_d.utils import user_settings

    user_settings.save({**user_settings.values(), "postprocess_default_format": "lerobot"})
    api = AppAPI.__new__(AppAPI)
    seen = []
    api._data_recorder = SimpleNamespace(is_recording=False)
    api.convert_recording_to_dataset = lambda *a: seen.append(a)
    api._rclpy_initialized = False
    api._topic_subscribers = SimpleNamespace(close=lambda: None)
    api.shutdown()
    assert seen == [("lerobot",)]


# --- B3: GUI options follow the loaded config
@pytest.fixture
def page(monkeypatch, qapp):
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QMainWindow
    import relay_d.acquisition.qt_app as qt_app

    monkeypatch.syspath_prepend(os.path.dirname(qt_app.__file__))
    from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser
    from relay_d.acquisition.qt_app.ui.containers.mainWindow_ui import Ui_MainWindow
    from relay_d.acquisition.qt_app.ui.postprocess import postprocess_page as pp

    monkeypatch.setattr(pp.CustomMessageBox, "warning", lambda *a, **k: None)
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    p = pp.PostProcessPage(ui, YamlParser())
    p._window = window
    return p


def test_gui_options_reset_on_each_config_load(workspace, page):
    tmp, cfg, _ = workspace
    picked = tmp / "picked.yaml"
    picked.write_text("postprocess:\n  lerobot:\n    task: picked\n")
    assert page.load_postprocess_options_from_path(str(picked))
    assert page.load_postprocess_config_from_path(str(cfg))  # no postprocess_config.yaml next to it
    assert page.yaml_converter.options.source is None  # defaults, not the earlier file
    sibling = cfg.parent / "postprocess_config.yaml"
    sibling.write_text("postprocess:\n  lerobot:\n    task: sibling\n")
    assert page.load_postprocess_options_from_path(str(picked))
    assert page.load_postprocess_config_from_path(str(cfg))
    assert page.yaml_converter.options.source == str(sibling)


def test_gui_refuses_to_convert_with_a_broken_options_file_next_to_the_config(workspace, page, monkeypatch):
    # as relay-d-convert and AppAPI: a broken postprocess_config.yaml is not silently replaced by the defaults
    tmp, cfg, rec_dir = workspace
    (cfg.parent / "postprocess_config.yaml").write_text("postprocess:\n  robomimic:\n    training_model: bc\n")
    errors = []
    from relay_d.acquisition.qt_app.ui.postprocess import postprocess_page as pp

    monkeypatch.setattr(pp.CustomMessageBox, "error", lambda msg, *a, **k: errors.append(msg))
    page.load_postprocess_config_from_path(str(cfg))
    page.output_format, page.h5_files = "robomimic", [str(rec_dir / "stack_cubes_1.h5")]
    converted = []
    monkeypatch.setattr(page.yaml_converter, "convert_multiple_files", lambda *a, **k: converted.append(a))
    page.convert_dataset()
    assert converted == [] and errors and "training_model" in errors[0]


# --- B4: inputs
def _converter():
    import json

    return YAMLDrivenConverter(PostprocessConfigLoader(), data_config=json.loads(json.dumps(DATA_CONFIG)))


def test_converted_robomimic_files_in_the_inputs_are_skipped(tmp_path):
    rec = tmp_path / "rec.h5"
    write_recording(rec, LENGTHS)
    conv = _converter()
    first = tmp_path / "first.h5"
    assert conv.convert_multiple_files("robomimic", [str(rec)], str(first))
    assert conv.convert_multiple_files("robomimic", [str(rec), str(first)], str(tmp_path / "second.h5"))
    with h5py.File(tmp_path / "second.h5", "r") as f:
        assert f["data"].attrs["total"] == len(LENGTHS)
    assert conv.conversion_summary() == f"Converted {len(LENGTHS)} of {len(LENGTHS)} demos"  # not "failed demos"


def test_same_named_files_of_several_sessions_sort_by_folder(tmp_path):
    for session, n in (("s1", 5), ("s2", 7)):
        (tmp_path / session).mkdir()
        write_recording(tmp_path / session / "rec.h5", [n])
    out = tmp_path / "out.h5"
    assert _converter().convert_multiple_files("robomimic", [str(tmp_path / "s2/rec.h5"), str(tmp_path / "s1/rec.h5")], str(out))
    with h5py.File(out, "r") as f:
        assert f["data/demo_0/actions"].shape[0] == 5
