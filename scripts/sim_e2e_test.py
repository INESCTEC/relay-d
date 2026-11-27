#!/usr/bin/env python3
"""Full end-to-end simulation pipeline test using ``relayd-sim`` (ManiSkill 3 + ROS 2 Jazzy).

Exercises the complete Relay-D loop against a real simulated Franka Panda PickCube scene
with 3 live RGB cameras (front, side, wrist), /joint_states, object poses, and /tf:

  Phase 1 — Record:  maniskill_bridge.py -> AppAPI (StreamingH5Writer) -> DemoValidator
  Phase 2 — Convert: relay-d-convert --check + --format lerobot & --format robomimic
  Phase 3 — Train:   relay-d-train on CUDA (LeRobot ACT & Robomimic BC-RNN-GMM)
  Phase 4 — Deploy:  lfd-inference-node (lazy ObsBuilder + /dev/shm SubprocessPolicyRunner
                     + ActionDispatcher) driving /sim/joint_command in closed loop

Usage:
    python3 scripts/sim_e2e_test.py
    python3 scripts/sim_e2e_test.py --backends lerobot --num-demos 2 --lerobot-steps 10
    RELAYD_SIM_E2E=1 pytest -v -s tests/test_sim_e2e.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

_EVENT_RE = re.compile(r"^(expert_start|expert_done)(?: success=(true|false))? seed=(\d+)$")
_DEFAULT_SIM_REPO = Path(__file__).resolve().parents[2] / "relayd-sim"
_DEFAULT_SIM_PYTHON = Path.home() / "Documents" / "python_environments" / "relayd-sim" / "bin" / "python3"


@dataclass
class E2EConfig:
    sim_repo: Path = _DEFAULT_SIM_REPO
    sim_python: Path = _DEFAULT_SIM_PYTHON
    robot: str = "panda"
    backends: List[str] = field(default_factory=lambda: ["lerobot", "robomimic"])
    num_demos: int = 2
    lerobot_steps: int = 10
    robomimic_epochs: int = 2
    rollout_sec: float = 8.0
    deploy_hz: float = 20.0
    device: str = "cuda"
    ros_domain_id: int = 98
    work_dir: Optional[Path] = None
    train_timeout_sec: float = 600.0
    deploy_startup_sec: float = 90.0


def _isolated_env(domain_id: int, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<no log>"


def _stop_process_group(proc: subprocess.Popen, sig: int = signal.SIGINT, timeout: float = 15.0) -> int:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5.0)
    return int(proc.returncode or 0)


class SimBridge:
    """Context manager launching ``relayd-sim/maniskill_bridge.py`` on an isolated ROS domain."""

    def __init__(self, cfg: E2EConfig, log_path: Path):
        self.cfg = cfg
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._log_file = None

    def __enter__(self) -> "SimBridge":
        bridge = self.cfg.sim_repo / "maniskill_bridge.py"
        if not bridge.is_file():
            raise FileNotFoundError(f"maniskill_bridge.py not found at {bridge}")
        if not self.cfg.sim_python.is_file():
            raise FileNotFoundError(f"relayd-sim Python not found at {self.cfg.sim_python}")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w")
        cmd = [str(self.cfg.sim_python), str(bridge), "--robot", self.cfg.robot, "--no-console"]
        print(f"[sim] Starting bridge: {' '.join(cmd)} (ROS_DOMAIN_ID={self.cfg.ros_domain_id})", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cfg.sim_repo),
            env=_isolated_env(self.cfg.ros_domain_id),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def wait_ready(self, node, timeout: float = 60.0) -> None:
        """Wait until /sim/run_expert, /sim/reset, and /sim/camera/front/image_raw are advertised."""
        from std_srvs.srv import Trigger

        run_cli = node.create_client(Trigger, "/sim/run_expert")
        reset_cli = node.create_client(Trigger, "/sim/reset")
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"maniskill_bridge exited early (code {self.proc.returncode}). Log:\n{_tail(self.log_path)}"
                    )
                topics = {name for name, _ in node.get_topic_names_and_types()}
                if (
                    run_cli.service_is_ready()
                    and reset_cli.service_is_ready()
                    and "/sim/camera/front/image_raw" in topics
                    and "/joint_states" in topics
                ):
                    print("[sim] Bridge ready (services + cameras live).", flush=True)
                    return
                time.sleep(0.25)
            raise TimeoutError(f"maniskill_bridge not ready within {timeout:.0f}s. Log:\n{_tail(self.log_path)}")
        finally:
            node.destroy_client(run_cli)
            node.destroy_client(reset_cli)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is not None:
            _stop_process_group(self.proc, sig=signal.SIGINT, timeout=10.0)
            self.proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


class _ExpertDriver:
    """Calls /sim/run_expert and /sim/reset and listens to /sim/episode_event."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node = rclpy.create_node("relayd_e2e_driver")
        self.events: queue.Queue = queue.Queue()
        self.resets = 0
        self._lock = threading.Lock()
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(String, "/sim/episode_event", self._on_event, latched)
        self.run_expert_cli = self.node.create_client(Trigger, "/sim/run_expert")
        self.reset_cli = self.node.create_client(Trigger, "/sim/reset")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

    def _on_event(self, msg) -> None:
        if msg.data == "reset":
            with self._lock:
                self.resets += 1
            return
        m = _EVENT_RE.match(msg.data)
        if m:
            self.events.put((m.group(1), m.group(2) == "true", int(m.group(3))))

    def drain(self) -> None:
        time.sleep(0.8)
        while not self.events.empty():
            self.events.get_nowait()

    def request_expert(self) -> None:
        from std_srvs.srv import Trigger

        assert self.run_expert_cli.wait_for_service(timeout_sec=10.0), "/sim/run_expert unavailable"
        self.run_expert_cli.call_async(Trigger.Request())

    def wait_event(self, kind: str, seed: Optional[int] = None, timeout: float = 90.0):
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                k, ok, s = self.events.get(timeout=min(left, 0.5))
            except queue.Empty:
                continue
            if k == kind and (seed is None or s == seed):
                return ok, s
        raise TimeoutError(f"Timed out waiting for {kind} (seed={seed}) after {timeout:.0f}s")

    def reset_scene(self, timeout: float = 15.0) -> None:
        from std_srvs.srv import Trigger

        assert self.reset_cli.wait_for_service(timeout_sec=10.0), "/sim/reset unavailable"
        with self._lock:
            before = self.resets
        self.reset_cli.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.resets > before:
                    return
            time.sleep(0.05)
        raise TimeoutError("No 'reset' event received from maniskill_bridge")

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()


def _phase1_record(cfg: E2EConfig, work_dir: Path, acq_cfg: Path, post_cfg: Path) -> Dict:
    """Phase 1: Start ManiSkill bridge and record scripted expert demos via AppAPI."""
    import rclpy
    from relay_d.acquisition.api import AppAPI
    from relay_d.acquisition.qt_app.ui.review.demo_validator import CheckStatus, DemoValidator

    print(f"\n=== [Phase 1/4] Recording {cfg.num_demos} scripted expert demos via AppAPI ===", flush=True)
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROS_DOMAIN_ID"] = str(cfg.ros_domain_id)

    with SimBridge(cfg, work_dir / "bridge_record.log") as bridge:
        rclpy.init()
        driver = _ExpertDriver()
        api = None
        kept: List[Path] = []
        stop_times: List[float] = []
        try:
            bridge.wait_ready(driver.node, timeout=60.0)
            driver.drain()
            api = AppAPI(str(acq_cfg), output_dir=str(raw_dir), postprocess_config_path=str(post_cfg))
            validator = DemoValidator()

            attempts = 0
            max_attempts = max(4, cfg.num_demos * 3)
            while len(kept) < cfg.num_demos and attempts < max_attempts:
                attempts += 1
                driver.request_expert()
                _, seed = driver.wait_event("expert_start", timeout=60.0)
                demo_name = f"pickcube_seed{seed}_{time.time_ns() // 1_000_000}"
                assert api.start_recording(demo_name), f"AppAPI.start_recording failed for {demo_name}"
                success, _ = driver.wait_event("expert_done", seed=seed, timeout=120.0)

                # Verify cameras streamed to HDF5 rather than sitting in RAM
                assert "image_camera_front" not in api._data_recorder.recorded_data, (
                    "Expected camera frames to stream via StreamingH5Writer, not stay in recorded_data"
                )
                t0 = time.perf_counter()
                stopped = api.stop_recording()
                stop_dt = time.perf_counter() - t0
                assert stopped, f"AppAPI.stop_recording failed for {demo_name}"

                h5_path = Path(api._data_recorder.h5_file_path)
                assert h5_path.is_file(), f"Missing recorded file {h5_path}"

                if not success:
                    print(f"  [record] seed={seed}: expert failed, deleting via AppAPI.delete_demo", flush=True)
                    assert api.delete_demo(str(h5_path))
                    continue

                val = validator.validate(str(h5_path))
                assert val.overall_status != CheckStatus.FAIL, f"DemoValidator failed:\n{val.summary_text()}"
                with h5py.File(h5_path, "r") as f:
                    demo = f["data"][demo_name]
                    n_frames = int(demo["timestamps"].shape[0])
                    cam_shape = tuple(demo["image_camera_front/images"].shape)
                stop_times.append(stop_dt)
                kept.append(h5_path)
                print(
                    f"  [record] {len(kept)}/{cfg.num_demos} kept: {h5_path.name} "
                    f"({n_frames} frames, front={cam_shape}, stop={stop_dt:.3f}s, check={val.overall_status.value})",
                    flush=True,
                )

            assert len(kept) == cfg.num_demos, f"Only kept {len(kept)}/{cfg.num_demos} demos after {attempts} attempts"
            session_dir = kept[0].parent
        finally:
            if api is not None:
                api._topic_subscribers.close()
            driver.close()
            rclpy.try_shutdown()

    return {
        "session_dir": session_dir,
        "demos": kept,
        "stop_times_s": stop_times,
    }


def _phase2_convert(
    cfg: E2EConfig, work_dir: Path, acq_cfg: Path, post_cfg: Path, session_dir: Path
) -> Dict[str, Dict[str, Path]]:
    """Phase 2: Run relay-d-convert (--check + conversion) for each requested backend."""
    print(f"\n=== [Phase 2/4] Converting demos via relay-d-convert ({', '.join(cfg.backends)}) ===", flush=True)
    datasets: Dict[str, Dict[str, Path]] = {}

    for fmt in cfg.backends:
        # Pre-check first
        check_cmd = [
            sys.executable, "-m", "relay_d.acquisition.convert_cli",
            "--config", str(acq_cfg), "--options", str(post_cfg),
            "--format", fmt, "--input", str(session_dir), "--check",
        ]
        res = subprocess.run(check_cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"relay-d-convert --check ({fmt}) failed:\n{res.stdout}\n{res.stderr}"

        out_dir = work_dir / f"dataset_{fmt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("pickcube" if fmt == "lerobot" else "pickcube.h5")

        t0 = time.perf_counter()
        conv_cmd = [
            sys.executable, "-m", "relay_d.acquisition.convert_cli",
            "--config", str(acq_cfg), "--options", str(post_cfg),
            "--format", fmt, "--input", str(session_dir), "--output", str(out_path),
        ]
        res = subprocess.run(conv_cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        assert res.returncode == 0, f"relay-d-convert ({fmt}) failed:\n{res.stdout}\n{res.stderr}"

        obs_cfg = out_dir / "action_dispatcher" / "obs_config.yaml"
        assert obs_cfg.is_file(), f"Missing generated obs_config.yaml at {obs_cfg}"

        if fmt == "lerobot":
            info_path = out_path / "meta" / "info.json"
            assert info_path.is_file(), f"Missing LeRobot meta/info.json at {info_path}"
            info = json.loads(info_path.read_text())
            videos = list((out_path / "videos").rglob("*.mp4"))
            parquets = list((out_path / "data").rglob("*.parquet"))
            assert info["total_episodes"] == cfg.num_demos
            assert len(videos) >= 3 and len(parquets) >= 1
            print(
                f"  [convert] lerobot OK in {dt:.2f}s: {info['total_episodes']} episodes, "
                f"{info['total_frames']} frames, {len(videos)} mp4s",
                flush=True,
            )
        else:
            assert out_path.is_file()
            with h5py.File(out_path, "r") as f:
                demos = list(f["data"].keys())
                assert len(demos) == cfg.num_demos
                assert "mask" in f and "train" in f["mask"]
                env_args = json.loads(f["data"].attrs["env_args"])
                assert env_args["env_kwargs"]["control_freq"] == 20
            print(
                f"  [convert] robomimic OK in {dt:.2f}s: {len(demos)} demos, "
                f"{out_path.stat().st_size / 1024**2:.1f} MB",
                flush=True,
            )

        datasets[fmt] = {"folder": out_dir, "path": out_path, "obs_config": obs_cfg}
    return datasets


def _find_checkpoint(fmt: str, run_dir: Path) -> Path:
    if fmt == "lerobot":
        candidates = sorted((run_dir / "output" / "checkpoints").rglob("pretrained_model"))
        candidates = [p for p in candidates if (p / "config.json").is_file() and (p / "model.safetensors").is_file()]
        assert candidates, f"No LeRobot pretrained_model found under {run_dir}"
        return candidates[-1]
    pths = sorted(run_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    assert pths, f"No Robomimic .pth checkpoint found under {run_dir}"
    return pths[-1]


def _phase3_train(cfg: E2EConfig, datasets: Dict[str, Dict[str, Path]]) -> Dict[str, Dict]:
    """Phase 3: Train real checkpoints on CUDA via relay-d-train CLI and wait for completion."""
    from relay_d.training import runs

    print(f"\n=== [Phase 3/4] Training policies via relay-d-train on {cfg.device} ===", flush=True)
    trained: Dict[str, Dict] = {}

    for fmt, ds in datasets.items():
        folder = ds["folder"]
        cmd = [
            sys.executable, "-m", "relay_d.training.train_cli",
            "--folder", str(folder),
            "--format", fmt,
            "--device", cfg.device,
            "--no-terminal",
        ]
        if fmt == "lerobot":
            cmd += [
                "--policy", "act",
                "--steps", str(cfg.lerobot_steps),
                "--save-every", str(cfg.lerobot_steps),
                "--batch-size", "2",
                "--num-workers", "0",
                "--extra=--policy.chunk_size=10 --policy.n_action_steps=5",
            ]
        else:
            cmd += [
                "--policy", "bc_rnn_gmm",
                "--epochs", str(cfg.robomimic_epochs),
                "--save-every", str(cfg.robomimic_epochs),
                "--batch-size", "4",
            ]

        t0 = time.perf_counter()
        print(f"  [train] Launching {fmt}: {' '.join(cmd)}", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"relay-d-train ({fmt}) failed to launch:\n{res.stdout}\n{res.stderr}"

        found_runs = runs.list_runs(folder)
        assert found_runs, f"No run recorded under {folder}"
        run_dir = found_runs[0].path

        deadline = time.monotonic() + cfg.train_timeout_sec
        status = runs.STARTING
        while time.monotonic() < deadline:
            cur = runs.read_run(run_dir)
            status = cur.status
            if status not in (runs.STARTING, runs.RUNNING):
                break
            time.sleep(2.0)

        if status != runs.FINISHED:
            runs.stop(run_dir)
            raise RuntimeError(
                f"Training run {run_dir.name} ({fmt}) ended with status={status}.\n"
                f"--- train.log tail ---\n{_tail(run_dir / 'train.log', 60)}"
            )

        ckpt = _find_checkpoint(fmt, run_dir)
        dt = time.perf_counter() - t0
        print(f"  [train] {fmt} finished in {dt:.1f}s -> checkpoint: {ckpt}", flush=True)
        trained[fmt] = {"run_dir": run_dir, "checkpoint": ckpt, "train_time_s": dt}
    return trained


class _DeployWatcher:
    """Monitors /sim/joint_command and /joint_states while lfd-inference-node drives the sim."""

    def __init__(self, node):
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        self._lock = threading.Lock()
        self.cmd_times: List[float] = []
        self.last_cmd: Optional[np.ndarray] = None
        self.qpos: Optional[np.ndarray] = None
        self._subs = [
            node.create_subscription(Float64MultiArray, "/sim/joint_command", self._on_cmd, 10),
            node.create_subscription(JointState, "/joint_states", self._on_js, 10),
        ]
        self._node = node

    def _on_cmd(self, msg) -> None:
        with self._lock:
            self.cmd_times.append(time.monotonic())
            self.last_cmd = np.asarray(msg.data, dtype=np.float64)

    def _on_js(self, msg) -> None:
        if len(msg.position) >= 7:
            with self._lock:
                self.qpos = np.asarray(msg.position[:7], dtype=np.float64)

    def reset_counts(self) -> None:
        with self._lock:
            self.cmd_times.clear()
            self.last_cmd = None

    def snapshot(self):
        with self._lock:
            return len(self.cmd_times), list(self.cmd_times), None if self.qpos is None else self.qpos.copy()

    def close(self) -> None:
        for sub in self._subs:
            self._node.destroy_subscription(sub)


def _phase4_deploy(
    cfg: E2EConfig,
    work_dir: Path,
    action_cfg: Path,
    datasets: Dict[str, Dict[str, Path]],
    trained: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Phase 4: Restart ManiSkill bridge and deploy each trained policy live via lfd-inference-node."""
    import rclpy

    print(f"\n=== [Phase 4/4] Deploying trained policies live via lfd-inference-node ({cfg.deploy_hz:g} Hz) ===", flush=True)
    os.environ["ROS_DOMAIN_ID"] = str(cfg.ros_domain_id)
    results: Dict[str, Dict] = {}

    with SimBridge(cfg, work_dir / "bridge_deploy.log") as bridge:
        rclpy.init()
        driver = _ExpertDriver()
        watcher = _DeployWatcher(driver.node)
        try:
            bridge.wait_ready(driver.node, timeout=60.0)

            for fmt in cfg.backends:
                driver.reset_scene()
                watcher.reset_counts()
                time.sleep(0.4)
                _, _, q_prev = watcher.snapshot()

                ckpt = trained[fmt]["checkpoint"]
                obs_cfg = datasets[fmt]["obs_config"]
                log_path = work_dir / f"deploy_{fmt}.log"

                cmd = [
                    sys.executable, "-m", "relay_d.dispatch.inference_node",
                    "--checkpoint", str(ckpt),
                    "--obs-config", str(obs_cfg),
                    "--action-config", str(action_cfg),
                    "--backend", fmt,
                    "--device", cfg.device,
                    "--hz", str(cfg.deploy_hz),
                ]
                print(f"  [deploy] Starting {fmt}: {' '.join(cmd)}", flush=True)
                with open(log_path, "w") as log_f:
                    proc = subprocess.Popen(
                        cmd,
                        env=_isolated_env(cfg.ros_domain_id),
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        t_first = None
                        path_len = 0.0
                        deadline = time.monotonic() + cfg.deploy_startup_sec
                        while time.monotonic() < deadline and proc.poll() is None:
                            n_cmds, _, q_now = watcher.snapshot()
                            if n_cmds > 0 and t_first is None:
                                t_first = time.monotonic()
                                deadline = t_first + cfg.rollout_sec
                            if q_now is not None and q_prev is not None:
                                path_len += float(np.abs(q_now - q_prev).sum())
                            if q_now is not None:
                                q_prev = q_now
                            time.sleep(0.05)

                        if proc.poll() is not None and t_first is None:
                            raise RuntimeError(
                                f"lfd-inference-node ({fmt}) exited early (code {proc.returncode}).\n"
                                f"--- deploy_{fmt}.log ---\n{_tail(log_path, 60)}"
                            )
                        assert t_first is not None, (
                            f"No /sim/joint_command received from {fmt} within {cfg.deploy_startup_sec}s.\n"
                            f"--- deploy_{fmt}.log ---\n{_tail(log_path, 60)}"
                        )
                    finally:
                        rc = _stop_process_group(proc, sig=signal.SIGTERM, timeout=15.0)

                n_cmds, cmd_times, _ = watcher.snapshot()
                dur = (cmd_times[-1] - cmd_times[0]) if len(cmd_times) >= 2 else 0.0
                achieved_hz = (len(cmd_times) - 1) / dur if dur > 0 else 0.0
                log_text = log_path.read_text(errors="replace")

                assert rc == 0, f"lfd-inference-node ({fmt}) exited with {rc}.\n{_tail(log_path, 40)}"
                assert "[SubprocessPolicyRunner]" in log_text, (
                    f"Expected isolated SubprocessPolicyRunner in deploy_{fmt}.log:\n{_tail(log_path, 40)}"
                )
                assert "Stale inputs" not in log_text, (
                    f"Stale input errors detected during {fmt} rollout:\n{_tail(log_path, 40)}"
                )
                assert achieved_hz >= cfg.deploy_hz * 0.6, (
                    f"Command rate too low for {fmt}: {achieved_hz:.1f} Hz (expected near {cfg.deploy_hz} Hz)"
                )
                assert path_len > 0.05, f"Simulated Panda arm barely moved under {fmt}: path_len={path_len:.4f} rad"

                leftover_shm = glob.glob("/dev/shm/relayd_ipc_*")
                assert not leftover_shm, f"Leaked /dev/shm IPC directories after {fmt} shutdown: {leftover_shm}"

                print(
                    f"  [deploy] {fmt} OK: {n_cmds} commands in {dur:.2f}s ({achieved_hz:.1f} Hz), "
                    f"arm path_len={path_len:.2f} rad, 0 stale drops, clean exit (rc=0), 0 /dev/shm leaks",
                    flush=True,
                )
                results[fmt] = {
                    "n_cmds": n_cmds,
                    "achieved_hz": achieved_hz,
                    "path_len_rad": path_len,
                    "returncode": rc,
                    "log": log_path,
                }
        finally:
            watcher.close()
            driver.close()
            rclpy.try_shutdown()

    return results


def run_e2e(cfg: E2EConfig) -> Dict:
    """Execute all 4 phases and return a structured summary dict."""
    panda_cfg_dir = cfg.sim_repo / "configs" / cfg.robot
    acq_cfg = panda_cfg_dir / "acquisition.yaml"
    post_cfg = panda_cfg_dir / "postprocess_config.yaml"
    action_cfg = panda_cfg_dir / "action_config.yaml"
    for p in (acq_cfg, post_cfg, action_cfg):
        if not p.is_file():
            raise FileNotFoundError(f"Missing relayd-sim config: {p}")

    t_all = time.perf_counter()
    if cfg.work_dir is not None:
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        work_ctx = None
        work_dir = cfg.work_dir.resolve()
    else:
        work_ctx = tempfile.TemporaryDirectory(prefix="relayd_sim_e2e_")
        work_dir = Path(work_ctx.name)

    try:
        rec = _phase1_record(cfg, work_dir, acq_cfg, post_cfg)
        datasets = _phase2_convert(cfg, work_dir, acq_cfg, post_cfg, rec["session_dir"])
        trained = _phase3_train(cfg, datasets)
        deployed = _phase4_deploy(cfg, work_dir, action_cfg, datasets, trained)
        total_s = time.perf_counter() - t_all
        print(f"\n=== ALL 4 PHASES PASSED in {total_s:.1f}s (work_dir={work_dir}) ===", flush=True)
        return {
            "work_dir": work_dir,
            "record": rec,
            "datasets": datasets,
            "trained": trained,
            "deployed": deployed,
            "total_time_s": total_s,
        }
    finally:
        if work_ctx is not None:
            work_ctx.cleanup()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-repo", type=Path, default=Path(os.environ.get("RELAYD_SIM_REPO", _DEFAULT_SIM_REPO)))
    ap.add_argument("--sim-python", type=Path, default=Path(os.environ.get("RELAYD_SIM_PYTHON", _DEFAULT_SIM_PYTHON)))
    ap.add_argument("--backends", choices=["both", "lerobot", "robomimic"], default="both")
    ap.add_argument("--num-demos", type=int, default=2)
    ap.add_argument("--lerobot-steps", type=int, default=10)
    ap.add_argument("--robomimic-epochs", type=int, default=2)
    ap.add_argument("--rollout-sec", type=float, default=8.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--ros-domain-id", type=int, default=98)
    ap.add_argument("--work-dir", type=Path, default=None, help="Keep artifacts in this folder instead of a temp dir")
    args = ap.parse_args(argv)

    backends = ["lerobot", "robomimic"] if args.backends == "both" else [args.backends]
    cfg = E2EConfig(
        sim_repo=args.sim_repo,
        sim_python=args.sim_python,
        backends=backends,
        num_demos=args.num_demos,
        lerobot_steps=args.lerobot_steps,
        robomimic_epochs=args.robomimic_epochs,
        rollout_sec=args.rollout_sec,
        deploy_hz=args.hz,
        device=args.device,
        ros_domain_id=args.ros_domain_id,
        work_dir=args.work_dir,
    )
    run_e2e(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
