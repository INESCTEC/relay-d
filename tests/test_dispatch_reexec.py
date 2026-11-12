"""The inference node starts again in the environment that has the policy package (decision logic only; nothing is executed)."""

import sys

from relay_d.dispatch.reexec import GUARD, reexec_if_needed


def _run(tmp_path, **kw):
    target = tmp_path / "venv" / "bin" / "python"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("")
    calls = []
    args = dict(
        backend="lerobot", argv=["-m", "relay_d.dispatch.inference_node", "--checkpoint", "c"], environ={"PYTHONPATH": "/old"},
        current="/usr/bin/python3", python_for=lambda b: str(target),
        find_spec=lambda name: None if name == "lerobot" else type("S", (), {"origin": f"/opt/ros/lib/{name}/__init__.py"})(),
        execve=lambda *a: calls.append(a), echo=lambda m: None,
    )
    args.update(kw)
    return reexec_if_needed(**args), calls, str(target)


def test_missing_package_switches_to_the_environment_with_ros_and_relay_d_on_the_path(tmp_path):
    done, calls, target = _run(tmp_path)
    (path, argv, env), = calls
    assert done and path == target and argv == [target, "-m", "relay_d.dispatch.inference_node", "--checkpoint", "c"]
    assert env[GUARD] == "1" and env["PYTHONPATH"].split(":") == ["/opt/ros/lib", "/opt/ros/lib", "/old"]  # ros + relay_d roots, then the old path


def test_no_switch_when_the_package_is_here_already_switched_same_python_or_unknown(tmp_path):
    here = lambda name: object()  # noqa: E731  (every package is importable)
    for kw in (dict(find_spec=here), dict(environ={GUARD: "1"}), dict(backend="other"), dict(python_for=lambda b: sys.executable, current=sys.executable),
               dict(python_for=lambda b: str(tmp_path / "missing" / "python"))):
        done, calls, _ = _run(tmp_path, **kw)
        assert not done and not calls, kw
