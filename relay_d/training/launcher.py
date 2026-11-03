"""Start imitation-learning training (robomimic or LeRobot) as a detached process.

Pure Python: no Qt, no ROS. Used by the Train tab (``relay_d/acquisition/qt_app/ui/train``) and by the
``relay-d-train`` command. This is a thin launcher: it builds the training command for a dataset made by
the Post-process page (or ``relay-d-convert``), opens a terminal for it, and records the run on disk so
``relay_d.training.runs`` can follow it later, even after the GUI was closed.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py

from relay_d.utils import user_settings
from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from relay_d.acquisition.qt_app.ui.postprocess.robomimic_config_handler import (
    RobomimicConfigHandler,
    classify_observations,
)

ROBOMIMIC, LEROBOT = "robomimic", "lerobot"
RUNS_DIRNAME = "training_runs"

ROBOMIMIC_POLICIES = ["bc_rnn_gmm", "bc_transformer_gmm", "diffusion_policy"]
LEROBOT_POLICIES = ["act", "diffusion", "smolvla"]

# Editable parameters. key -> (where it goes, type, label). The robomimic "where" is a path into the
# config JSON; the LeRobot one is the command-line flag. A new LeRobot release that renames a flag is a
# one-line change here.
ROBOMIMIC_PARAMS = {
    "epochs": ("train.num_epochs", int, "Epochs"),
    "batch_size": ("train.batch_size", int, "Batch size"),
    "learning_rate": ("algo.optim_params.policy.learning_rate.initial", float, "Learning rate"),
    "seq_length": ("train.seq_length", int, "Sequence length"),
    "save_every": ("experiment.save.every_n_epochs", int, "Save checkpoint every N epochs"),
    "seed": ("train.seed", int, "Seed"),
    "device": ("train.cuda", str, "Device (cuda / cpu)"),
}
LEROBOT_PARAMS = {
    "steps": ("--steps", int, "Training steps"),
    "batch_size": ("--batch_size", int, "Batch size"),
    "learning_rate": ("--policy.optimizer_lr", float, "Learning rate"),
    "save_every": ("--save_freq", int, "Save checkpoint every N steps"),
    "num_workers": ("--num_workers", int, "Data loader workers"),
    "seed": ("--seed", int, "Seed"),
    "device": ("--policy.device", str, "Device (cuda / cpu)"),
}
_LEROBOT_SAVE_EVERY = 20_000
_LEROBOT_LR = {"act": 1e-5, "diffusion": 1e-4, "smolvla": 1e-4}
# Fine-tuned from a pretrained model on the Hugging Face Hub (downloaded on the first run) instead of trained from scratch,
# with the camera names that model was trained with: the dataset's cameras are renamed to them, in order.
_LEROBOT_PRETRAINED = {"smolvla": ("lerobot/smolvla_base", ["observation.images.camera1", "observation.images.camera2",
                                                            "observation.images.camera3"])}
_LEROBOT_MODULE = "lerobot.scripts.lerobot_train"
_ZERO_OK = ("seed", "num_workers")  # 0 is meaningful for these (num_workers=0: load data in the main process)


class TrainingError(Exception):
    """The run cannot be set up (bad dataset, bad parameter, no template, ...)."""


@dataclass
class DatasetInfo:
    format: str
    path: Path  # robomimic: the .h5 file; lerobot: the dataset directory
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def base(self) -> Path:
        """Folder next to the dataset: holds the configs, and the ``training_runs`` folder."""
        return self.path.parent

    @property
    def name(self) -> str:
        return self.path.stem if self.format == ROBOMIMIC else self.path.name


@dataclass
class TrainJob:
    dataset: DatasetInfo
    python: str = sys.executable
    policy: Optional[str] = None  # robomimic: None = the folder's saved robomimic_config.json, else the first template
    params: Dict[str, Any] = field(default_factory=dict)  # only the keys the user changed
    extra_args: str = ""


# ----------------------------------------------------------------------------- datasets


def _robomimic_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Summary of a robomimic training file, or None if it is not one (a raw recording has no ``mask``)."""
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f or "mask" not in f:
                return None
            demos = [k for k in f["data"] if isinstance(f["data"][k], h5py.Group)]
            obs_keys = list(f["data"][demos[0]]["obs"]) if demos and "obs" in f["data"][demos[0]] else []
            frames = lambda names: int(sum(f["data"][d].attrs.get("num_samples", 0) for d in names if d in f["data"]))  # noqa: E731
            train = [k.decode() if isinstance(k, bytes) else str(k) for k in f["mask"]["train"][:]] if "train" in f["mask"] else demos
            return {
                "episodes": len(demos),
                "frames": frames(demos),
                "train_frames": frames(train),  # what one epoch is made of
                "has_valid": "valid" in f["mask"],
                "obs_keys": obs_keys,
            }
    except (OSError, KeyError):
        return None


def _lerobot_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None
    cameras = [k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image")]
    return {
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "cameras": cameras,
        "robot_type": info.get("robot_type"),
    }


def find_datasets(path) -> List[DatasetInfo]:
    """Training datasets at ``path``, newest first.

    ``path`` can be the dataset itself (a robomimic ``.h5`` or a LeRobot folder) or the folder the
    Post-process page / ``relay-d-convert`` wrote it to.
    """
    path = Path(path)
    candidates = [path] if (path.is_file() or (path / "meta" / "info.json").exists()) else sorted(path.iterdir()) if path.is_dir() else []
    found = []
    for item in candidates:
        if item.is_file() and item.suffix == ".h5" and (s := _robomimic_summary(item)) is not None:
            found.append(DatasetInfo(ROBOMIMIC, item, s))
        elif item.is_dir() and (s := _lerobot_summary(item)) is not None:
            found.append(DatasetInfo(LEROBOT, item, s))
    return sorted(found, key=lambda d: d.path.stat().st_mtime, reverse=True)


# ----------------------------------------------------------------------------- parameters


def _get_path(cfg: dict, dotted: str, default=None):
    for key in dotted.split("."):
        if not isinstance(cfg, dict) or key not in cfg:
            return default
        cfg = cfg[key]
    return cfg


def _set_path(cfg: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    for key in parents:
        cfg = cfg.setdefault(key, {})
    cfg[last] = value


def _config_file(dataset: "DatasetInfo", name: str) -> Optional[Path]:
    """``name`` (e.g. ``acquisition_config.yaml``) that belongs to ``dataset``, next to it.

    ``relay-d-convert`` writes ``<output name>_<name>``, so a folder can hold several pairs: take the one named after
    this dataset, else the plain file the GUI writes, else any ``*<name>``.
    """
    for candidate in (dataset.base / f"{dataset.path.stem}_{name}", dataset.base / name):
        if candidate.is_file():
            return candidate
    return next(iter(sorted(dataset.base.glob(f"*{name}"))), None)


def _robomimic_policy(dataset: DatasetInfo, policy: Optional[str]) -> str:
    """The template name to use: explicit (the Train tab's Policy box / --policy), else the first one."""
    if policy and policy not in ROBOMIMIC_POLICIES:
        raise TrainingError(f"unknown robomimic policy '{policy}' (choose from {ROBOMIMIC_POLICIES})")
    return policy or ROBOMIMIC_POLICIES[0]


def _robomimic_modalities(dataset: DatasetInfo) -> Dict[str, list]:
    """Observation modalities for a config that has none: from the saved config, else from the file."""
    saved = dataset.base / "robomimic_config.json"
    if saved.exists():
        obs = _get_path(RobomimicConfigHandler(str(saved)).get_config(), "observation.modalities.obs")
        if obs:
            return {k: obs.get(k, []) for k in ("low_dim", "rgb", "depth", "scan")}
    import yaml

    data_config: dict = {}
    acq = _config_file(dataset, "acquisition_config.yaml")
    if acq is not None:
        data_config = (yaml.safe_load(acq.read_text()) or {}).get("config", {})
    with h5py.File(dataset.path, "r") as f:
        demo = f["data"][next(iter(f["data"]))]
        shape_cache = {k: (demo["obs"][k].shape[1:], demo["obs"][k].dtype) for k in demo["obs"]}
    return classify_observations(data_config, shape_cache)


def _robomimic_base_config(dataset: DatasetInfo, policy: Optional[str]) -> dict:
    saved = dataset.base / "robomimic_config.json"
    if policy is None and saved.exists():
        return RobomimicConfigHandler(str(saved)).get_config()
    model = _robomimic_policy(dataset, policy)
    template = PostprocessConfigLoader().get_output_train_config(ROBOMIMIC, model)
    if not template or not os.path.exists(template):
        raise TrainingError(f"no robomimic config template for '{model}'")
    handler = RobomimicConfigHandler(template)
    handler.set_observation_modalities(**_robomimic_modalities(dataset))
    return handler.get_config()


def default_params(dataset: DatasetInfo, policy: Optional[str] = None) -> Dict[str, Any]:
    """The values training would use if the user changes nothing (shown as the form's defaults)."""
    if dataset.format == LEROBOT:
        return {
            "steps": user_settings.get("train_lerobot_steps"),
            "batch_size": user_settings.get("train_lerobot_batch_size"),
            "save_every": _LEROBOT_SAVE_EVERY,
            "num_workers": user_settings.get("train_lerobot_num_workers"),
            "seed": user_settings.get("train_lerobot_seed"),
            "device": user_settings.get("train_default_device"),
            "learning_rate": _LEROBOT_LR[policy or "act"],
        }
    cfg = _robomimic_base_config(dataset, policy)
    out = {key: _get_path(cfg, path) for key, (path, _type, _label) in ROBOMIMIC_PARAMS.items()}
    train_frames = dataset.summary.get("train_frames")
    if train_frames and out.get("batch_size"):
        out["batch_size"] = max(1, min(int(out["batch_size"]), train_frames))  # a batch larger than the data makes no batch at all
    out["device"] = user_settings.get("train_default_device")  # the Settings page chooses the default device
    return out


def _coerce(params: Dict[str, Any], spec: dict) -> Dict[str, Any]:
    out = {}
    for key, value in params.items():
        if key not in spec:
            raise TrainingError(f"unknown parameter '{key}' (choose from {sorted(spec)})")
        kind = spec[key][1]
        try:
            out[key] = kind(value)
        except (TypeError, ValueError):
            raise TrainingError(f"parameter '{key}' must be {kind.__name__}, got {value!r}") from None
        if kind in (int, float) and out[key] <= 0 and key not in _ZERO_OK:
            raise TrainingError(f"parameter '{key}' must be greater than 0, got {value!r}")
        if key == "device" and out[key] not in ("cuda", "cpu"):
            raise TrainingError("device must be 'cuda' or 'cpu'")
    return out


# ----------------------------------------------------------------------------- commands


def build_command(job: TrainJob, run_dir: Path, write: bool = True) -> List[str]:
    """The argv that trains ``job``. With ``write`` the files it needs (robomimic config) are written to ``run_dir``."""
    ds = job.dataset
    try:
        extra = shlex.split(job.extra_args)
    except ValueError as e:
        raise TrainingError(f"extra arguments: {e}") from None

    if ds.format == ROBOMIMIC:
        params = _coerce(job.params, ROBOMIMIC_PARAMS)
        cfg = _robomimic_base_config(ds, job.policy)
        for key, value in params.items():
            _set_path(cfg, ROBOMIMIC_PARAMS[key][0], value == "cuda" if key == "device" else value)
        train_frames, batch = ds.summary.get("train_frames"), _get_path(cfg, ROBOMIMIC_PARAMS["batch_size"][0])
        if train_frames is not None and batch and batch > train_frames:
            if "batch_size" in params:
                # robomimic drops the last incomplete batch: with fewer samples than one batch an epoch has no batch and it crashes (StopIteration)
                raise TrainingError(
                    f"the training split has {train_frames} frames, fewer than the batch size {batch}: robomimic only trains on "
                    f"complete batches, so it would find none. Set the batch size to {train_frames} or less, or record more demos."
                )
            _set_path(cfg, ROBOMIMIC_PARAMS["batch_size"][0], train_frames)  # nobody chose a size: use what fits (like default_params)
        name = run_dir.name
        _set_path(cfg, "train.data", str(ds.path))
        _set_path(cfg, "train.output_dir", str(run_dir / "output"))
        _set_path(cfg, "experiment.name", name)
        if not ds.summary.get("has_valid", True):  # no validation split in the file: nothing to validate on
            _set_path(cfg, "experiment.validate", False)
            _set_path(cfg, "train.hdf5_validation_filter_key", None)  # robomimic refuses a filter key when validation is off
        config_path = run_dir / "robomimic_config.json"
        if write:
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg, indent=4))
        return [job.python, "-m", "robomimic.scripts.train", "--config", str(config_path),
                "--dataset", str(ds.path), "--name", name, *extra]

    policy = job.policy or "act"
    if policy not in LEROBOT_POLICIES:
        raise TrainingError(f"unknown LeRobot policy '{policy}' (choose from {LEROBOT_POLICIES})")
    if not ds.summary.get("cameras"):
        # ACT, Diffusion and SmolVLA need an image (or environment state, which Relay-D does not write) as input.
        raise TrainingError(
            "this LeRobot dataset has no camera: ACT, Diffusion and SmolVLA need at least one image input. "
            "Record a camera, or train a state-only policy with robomimic."
        )
    params = {**default_params(ds, policy), **_coerce(job.params, LEROBOT_PARAMS)}
    flags = [f"{LEROBOT_PARAMS[k][0]}={v}" for k, v in params.items()]
    if policy in _LEROBOT_PRETRAINED:
        base, cameras = _LEROBOT_PRETRAINED[policy]
        if len(ds.summary["cameras"]) > len(cameras):
            raise TrainingError(f"{policy} takes at most {len(cameras)} cameras, this dataset has {len(ds.summary['cameras'])}.")
        rename = dict(zip(ds.summary["cameras"], cameras))  # kept in the checkpoint: inference renames the same way
        source = [f"--policy.path={base}", f"--rename_map={json.dumps(rename)}"]
    else:
        source = [f"--policy.type={policy}"]
    return [job.python, "-m", _LEROBOT_MODULE,
            f"--dataset.repo_id=local/{ds.name}", f"--dataset.root={ds.path}",
            *source, "--policy.push_to_hub=false",  # push_to_hub defaults to true
            f"--output_dir={run_dir / 'output'}", f"--job_name={run_dir.name}", *flags, *extra]


# ----------------------------------------------------------------------------- terminal + run


_RUN_SCRIPT = """#!/usr/bin/env bash
# Written by Relay-D. Runs the training and records its state next to this file.
cd "$(dirname "$0")" || exit 1
echo $$ > pid
echo "Relay-D training run: {name}"
printf '$ %s\n\n' {quoted}
# Ctrl+C reaches the training (same process group) but must not kill this script before it records the end.
trap 'interrupted=1' INT
{command} 2>&1 | tee -a train.log
code=${{PIPESTATUS[0]}}
# robomimic catches its own crash, prints "run failed with error" and returns 0: that is a failure
if [ "$code" = 0 ] && grep -q "run failed with error" train.log 2>/dev/null; then code=1; fi
if [ -n "$interrupted" ]; then
    echo 130 > exit_code
    exit 0  # you pressed Ctrl+C: close the window now (a normal end or a crash still waits, so you can read it)
fi
echo $code > exit_code
echo
echo "Training exited with code $code. Press Enter to close this window."
read -r _ || true  # no tty (headless): do not wait, and exit 0
"""


def _terminal_candidates(title: str, command: List[str]) -> List[List[str]]:
    """Commands that open a terminal window running ``command``, best first. Empty without a display."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return []
    found = []
    for exe, args in (
        ("gnome-terminal", ["--title", title, "--"]),
        ("x-terminal-emulator", ["-T", title, "-e"]),
        ("konsole", ["-e"]),
        ("xterm", ["-T", title, "-e"]),
    ):
        if shutil.which(exe):
            found.append([exe, *args, *command])
    return found


def _open_terminal(candidates: List[List[str]]) -> Optional[str]:
    """Open the first terminal that works; return its name, or None if none did.

    A terminal that fails (no D-Bus session, unsupported flag) exits with an error within a moment,
    so the next one is tried. One that is still running is a foreground terminal (xterm, alacritty);
    one that exits 0 handed the window to its server (gnome-terminal). Both are fine.
    """
    for argv in candidates:
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            continue
        try:
            if proc.wait(timeout=1.0) != 0:
                continue
        except subprocess.TimeoutExpired:
            pass
        return argv[0]
    return None


def run_script(script: Path, title: str, terminal: bool = True) -> Optional[str]:
    """Run ``bash script`` in its own terminal window, else detached in the background (no display, no working terminal,
    or ``terminal`` False). Returns the terminal's name, or None for the background. Raises OSError if bash cannot start."""
    opened = _open_terminal(_terminal_candidates(title, ["bash", str(script)])) if terminal else None
    if opened is None:
        subprocess.Popen(["bash", str(script)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    return opened


def new_run_dir(base: Path, fmt: str) -> Path:
    stem = f"{fmt}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir, n = base / RUNS_DIRNAME / stem, 1
    while run_dir.exists():
        n += 1
        run_dir = base / RUNS_DIRNAME / f"{stem}_{n}"
    return run_dir


def start(job: TrainJob, terminal: bool = True) -> Path:
    """Start training in its own terminal window (or headless without one). Returns the run folder.

    The process belongs to the terminal (or its own session), not to the caller, so it keeps running
    when the GUI or the calling script exits.
    """
    run_dir = new_run_dir(job.dataset.base, job.dataset.format)
    argv = build_command(job, run_dir, write=False)  # validates everything before anything is created
    run_dir.mkdir(parents=True)
    build_command(job, run_dir)  # writes the files the command needs
    command = shlex.join(argv)
    script = run_dir / "run.sh"
    script.write_text(_RUN_SCRIPT.format(name=run_dir.name, command=command, quoted=shlex.quote(command)))

    meta = {
        "name": run_dir.name,
        "format": job.dataset.format,
        "dataset": str(job.dataset.path),
        "policy": job.policy,
        "python": job.python,
        "params": job.params,
        "extra_args": job.extra_args,
        "command": command,
        "started": time.time(),
        "terminal": None,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=4))
    try:
        meta["terminal"] = run_script(script, run_dir.name, terminal)
    except OSError as e:
        raise TrainingError(f"could not start the run: {e}") from None
    (run_dir / "run.json").write_text(json.dumps(meta, indent=4))
    return run_dir


def reopen_terminal(run_dir: Path) -> bool:
    """Open a terminal that follows the run's log (for a run started headless or whose window was closed)."""
    tail = f"tail -n 200 -f {shlex.quote(str(run_dir / 'train.log'))}"
    return _open_terminal(_terminal_candidates(f"{run_dir.name} log", ["bash", "-c", tail])) is not None


def check_environment(python: str, fmt: str, timeout: float = 60.0) -> Dict[str, Any]:
    """Can ``python`` start training? Imports what the run imports. Returns ``{ok, torch, cuda, error}``."""
    package = "robomimic" if fmt == ROBOMIMIC else "lerobot"
    module = "robomimic.scripts.train" if fmt == ROBOMIMIC else _LEROBOT_MODULE  # what the command runs
    code = (
        "import importlib, importlib.util as u, json\n"
        f"r = {{'package': u.find_spec({package!r}) is not None, 'torch': None, 'torchvision': None, 'cuda': False, 'import_error': None}}\n"
        "try:\n"
        "    import torch\n"
        "    r['torch'], r['cuda'] = torch.__version__, torch.cuda.is_available()\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import importlib.metadata as m\n"
        "    r['torchvision'] = m.version('torchvision')\n"
        "except Exception:\n"
        "    pass\n"
        "if r['package']:\n"
        "    try:\n"
        f"        importlib.import_module({module!r})\n"
        "    except BaseException as e:\n"
        "        r['import_error'] = f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(r))\n"
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout)
        result = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {"ok": False, "package": False, "torch": None, "cuda": False, "error": str(e) or type(e).__name__}
    if not result["package"]:
        error = f"'{package}' is not installed for {python}"
        error += f". Use the 'Set up {fmt} environment' button in the Train tab to build one"
        if fmt == LEROBOT:
            error += " (or: pip install 'lerobot[training]', plus 'lerobot[diffusion]' for Diffusion and 'lerobot[smolvla]' for SmolVLA)"
    elif not result["torch"]:
        error = f"torch is missing for {python}"
    else:
        error = result.get("import_error")
        if error:
            error = f"'{package}' is installed but cannot be imported: {error}"
            if "pip install" in error and fmt == LEROBOT:
                error += " (LeRobot needs: pip install 'lerobot[training]', plus 'lerobot[diffusion]' for Diffusion and 'lerobot[smolvla]' for SmolVLA)"
            if "torchvision" in error:
                error += (f" (torch {result['torch']} with torchvision {result.get('torchvision')}: "
                          "they must be built for each other, reinstall torchvision for this torch)")
    result["ok"], result["error"] = error is None, error
    return result
