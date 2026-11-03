"""ConfigDraft round trips, the shapes it writes, and the checks of the shared config validator."""

from pathlib import Path

import yaml

from relay_d.acquisition.qt_app.ui.config.builder.config_check import ERROR, check_config, has_errors, stream_width
from relay_d.acquisition.qt_app.ui.config.builder.draft import ConfigDraft, InputDraft, StreamDraft, spec_inputs, suggest_input_name
from helpers import DATA_CONFIG

TEMPLATE = Path(__file__).parents[1] / "relay_d/acquisition/qt_app/ui/config/templates/config_template.yaml"


def _messages(cfg, level=None):
    return [p.message for p in check_config(cfg) if level in (None, p.level)]


def test_round_trip_keeps_template_and_test_config():
    for cfg in (yaml.safe_load(TEMPLATE.read_text()), {"config": DATA_CONFIG}):
        assert ConfigDraft.from_config(cfg).to_config() == cfg


def test_remove_input_drops_dependent_streams_and_cameras():
    d = ConfigDraft()
    d.add_input(InputDraft(name="joint_a", topic="/js", joint_names=["j1", "j2"]))
    d.add_input(InputDraft(name="image_cam", topic="/img"))
    d.streams["q"] = StreamDraft(name="q", specs=["joint_a.position[:]"])
    d.cameras["agentview"] = ("image_cam", "images")
    d.remove_input("joint_a")
    d.remove_input("image_cam")
    assert not d.streams and not d.cameras and not d.inputs


def test_suggested_names_use_the_prefix_the_record_page_needs():
    assert suggest_input_name("image", "/camera/color/image_raw") == "image_camera_color"
    assert suggest_input_name("joint_state", "/joint_states") == "joint_states"
    assert spec_inputs("a.x - b.y") == {"a", "b"}


def _good():
    d = ConfigDraft(recording_frequency=30)
    d.add_input(InputDraft(name="joint_a", topic="/js", joint_names=["j1", "j2"]))
    d.streams["q"] = StreamDraft(name="q", specs=["joint_a.position[:]"], normalize=True, limits=[{"lower_lim": -1, "upper_lim": 1}] * 2)
    d.actions = ["q"]
    return d.to_config()


def test_valid_config_has_no_errors_and_width_counts_slices():
    cfg = _good()
    assert not has_errors(check_config(cfg))
    assert stream_width(["joint_a.position[:]", "joint_a.velocity[0]"], cfg["config"]["input"]) == 3


def test_validator_rules():
    def bad(mutate):
        cfg = _good()
        mutate(cfg["config"])
        return _messages(cfg, ERROR)

    assert any("not defined" in m for m in bad(lambda c: c["input"].pop("joint_a")))
    assert any("limits for 2 values" in m for m in bad(lambda c: c["output"]["streams"]["q"].update(limits=[{"lower_lim": 0, "upper_lim": 1}] * 3)))
    assert any("lower_lim must be smaller" in m for m in bad(lambda c: c["output"]["streams"]["q"].update(limits=[{"lower_lim": 1, "upper_lim": 1}] * 2)))
    assert any("actions_to_extract" in m for m in bad(lambda c: c.update(actions_to_extract=["nope"])))
    assert any("delta_actions" in m for m in bad(lambda c: c.update(delta_actions=["other"])))
    assert any("is normalized" in m for m in bad(lambda c: c.update(delta_actions=["q"])))  # q is normalized
    assert any("positive" in m for m in bad(lambda c: c.update(recording_frequency=0)))
    assert any("rpy needs exactly 4" in m for m in bad(lambda c: c["output"]["streams"]["q"].update(rotation_format="rpy")))
    assert any("both recorded into" in m for m in bad(lambda c: c["input"].update(joint_b={"topic": "/x", "output_map": "joint_a"})))
