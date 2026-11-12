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
    d.streams["q"] = StreamDraft(name="q", specs=["joint_a.position[:]"], explicit_dict=True)
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
    assert any("actions_to_extract" in m for m in bad(lambda c: c.update(actions_to_extract=["nope"])))
    assert any("delta_actions" in m for m in bad(lambda c: c.update(delta_actions=["other"])))
    for key in ("normalize", "normalized_range", "limits"):  # Relay-D no longer normalizes: old keys must not be ignored
        msgs = bad(lambda c: c["output"]["streams"]["q"].update({key: True}))
        assert any("Relay-D no longer normalizes" in m and "'q'" in m for m in msgs), key
    assert any("positive" in m for m in bad(lambda c: c.update(recording_frequency=0)))
    assert any("rpy needs exactly 4" in m for m in bad(lambda c: c["output"]["streams"]["q"].update(rotation_format="rpy")))
    assert any("'streams'" in m for m in bad(lambda c: c["input"].update(streams={"topic": "/x"})))
    assert any("recording_frequency is required" in m for m in bad(lambda c: c.pop("recording_frequency")))
    # actions are streams only (a camera entry is an observation, the converter cannot use it as an action)
    assert any("not a stream" in m for m in bad(lambda c: (c["output"].update(cam="joint_a/images"), c.update(actions_to_extract=["cam"]))))


def test_removed_keys_are_errors_that_say_what_to_do():
    """Every removed key fails loudly (never silently ignored) and names what to do."""
    def bad(mutate):
        cfg = _good()
        mutate(cfg["config"])
        return _messages(cfg, ERROR)

    cases = {
        "output_map": lambda c: c["input"]["joint_a"].update(output_map="g"),
        "fill_default": lambda c: c["input"]["joint_a"].update(fill_gaps=True, fill_default=0),
        "transfer_rate": lambda c: c["input"]["joint_a"].update(transfer_rate=10),
        "prefix_path": lambda c: c["output"].update(prefix_path="x"),
        "data": lambda c: c.update(data={"a": 1}),
        "metadata.recording_frequency": lambda c: c.update(metadata={"recording_frequency": 30}),
        "metadata.transfer_rate": lambda c: c.update(metadata={"transfer_rate": 30}),
    }
    for key, mutate in cases.items():
        assert any(f"'{key}' was removed" in m and "elete" in m for m in bad(mutate)), key


def test_stream_width_counts_zero_n_as_n_values():
    assert stream_width(["zero:3", "a.x"], {}) == 4


def test_camera_obs_for_dispatch_names_an_existing_input():
    from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser

    cfg = _good()
    cfg["config"]["input"]["image_cam"] = {"topic": "/img"}
    cfg["config"]["output"]["agentview_image"] = "image_cam/images"
    parser = YamlParser()
    assert parser.load_from_yaml_string(yaml.safe_dump(cfg))
    obs = parser.get_obs_config()["config"]
    assert obs["observations"]["agentview_image"] == {"source": "image_cam"}
    assert obs["observations"]["agentview_image"]["source"] in obs["input"]


def test_shipped_acquisition_template_passes_the_check():
    # "Generate Defaults" hands this file to users: it must load without errors and turn into an obs_config inference accepts.
    path = Path(__file__).parents[1] / "relay_d/acquisition/qt_app/ui/config/templates/config_template.yaml"
    assert [str(p) for p in check_config(yaml.safe_load(path.read_text())) if p.level == ERROR] == []
