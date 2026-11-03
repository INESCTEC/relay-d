"""The dispatch config templates users copy: they must parse, and their live entries must stay as documented."""

from importlib import resources

import yaml


def _load(name):
    return yaml.safe_load(resources.files("relay_d.dispatch").joinpath(f"config_templates/{name}").read_text())


def test_action_template_keeps_its_three_live_entries():
    actions = _load("action_config_template.yaml")["actions"]
    assert list(actions) == ["move_eef", "move_arm_joint", "robotiq_gripper"]
    assert actions["move_eef"]["goal_init"]["transforms"] == 1  # array pre-allocation, see the goal_init notes
    assert actions["move_arm_joint"]["mapping"] == {"joint_trajectory[0].positions": [0, 1, 2, 3, 4, 5, 6]}
    assert actions["robotiq_gripper"]["action"] is True and actions["robotiq_gripper"]["mapping"] == {"position": 0}


def test_live_template_mappings_point_inside_their_slice():
    # mapping values index the entry's slice (0-based), not the model output: `indices: [7]` has only index 0
    for name, cfg in _load("action_config_template.yaml")["actions"].items():
        if "mapping" in cfg and "transform" not in cfg:
            used = [i for v in cfg["mapping"].values() for i in (v if isinstance(v, list) else [v])]
            assert max(used) < len(cfg["indices"]), name


def test_obs_template_parses_and_every_documented_dispatch_key_is_known():
    assert "config" in _load("obs_config_template.yaml")
    text = resources.files("relay_d.dispatch").joinpath("config_templates/action_config_template.yaml").read_text()
    for key in ("indices", "denormalize", "normalized_range", "limits", "transform", "goal_init", "feedback", "mapping", "stop"):
        assert key in text  # the quick reference still documents every entry key


def test_inference_script_runs_the_cli_with_the_local_configs(monkeypatch):
    import runpy
    import sys
    import types

    fake = types.ModuleType("relay_d.dispatch.inference_node")
    seen = []
    fake.main = lambda: seen.append(list(sys.argv[1:]))
    monkeypatch.setitem(sys.modules, "relay_d.dispatch.inference_node", fake)
    monkeypatch.setattr(sys, "argv", ["inference_script.py", "--checkpoint", "m.pth", "--obs-config=o.yaml"])
    path = resources.files("relay_d.dispatch").joinpath("templates/inference_script_template.py")
    runpy.run_path(str(path), run_name="__main__")
    assert seen == [["--checkpoint", "m.pth", "--obs-config=o.yaml", "--action-config", "./action_config.yaml"]]
