"""The dispatch config templates users copy: they must parse, and their live entries must stay as documented."""

from importlib import resources
from pathlib import Path

import yaml


def _load(name):
    return yaml.safe_load(resources.files("relay_d.dispatch").joinpath(f"config_templates/{name}").read_text())


def test_action_template_has_one_live_arm_entry():
    actions = _load("action_config_template.yaml")["actions"]
    # move_eef is only an example: live next to move_arm_joint (same topic, same indices) it moved the arm twice per tick
    assert list(actions) == ["move_arm_joint", "robotiq_gripper"]
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
    for key in ("indices", "limits", "transform", "goal_init", "feedback", "mapping", "stop"):
        assert key in text  # the quick reference still documents every entry key
    for key in ("denormalize", "normalized_range", "apply_limits", "dedupe", "command:", "{I[", "vacuum", "magnetic",
                "type: scale", "type: clip", "after denorm"):
        assert key not in text  # removed: the dispatcher refuses them
    assert "static:" not in resources.files("relay_d.dispatch").joinpath("config_templates/obs_config_template.yaml").read_text()


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
    # the files next to the script, not the current folder
    assert seen == [["--checkpoint", "m.pth", "--obs-config=o.yaml", "--action-config", str(Path(str(path)).with_name("action_config.yaml"))]]


def test_cli_dry_run_flag_reaches_the_session(monkeypatch):
    import sys
    from types import SimpleNamespace

    from relay_d.dispatch import inference_node

    seen = []

    def fake_session(**kwargs):
        seen.append(kwargs["dry_run"])
        return SimpleNamespace(model=SimpleNamespace(fps=30.0), run=lambda **k: seen.append(k["hz"]) or True, close=lambda: None)

    monkeypatch.setattr(inference_node, "InferenceSession", fake_session)
    monkeypatch.setattr(inference_node, "reexec_if_needed", lambda *a: None)
    monkeypatch.setattr(inference_node, "sigterm_runs_the_safe_stop", lambda: None)  # keep this process's SIGTERM
    base = ["lfd-inference-node", "--checkpoint", "m.pth", "--obs-config", "o.yaml", "--action-config", "a.yaml",
            "--backend", "robomimic"]
    for extra, expected in (([], False), (["--dry-run"], True)):  # the Run tab passes exactly --dry-run
        seen.clear()
        monkeypatch.setattr(sys, "argv", base + extra)
        inference_node.main()
        assert seen == [expected, None]  # and no rate: run() takes the policy's own
