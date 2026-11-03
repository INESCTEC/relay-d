"""User settings shared by the GUI and the command-line tools (no Qt, no ROS).

One table, ``SETTINGS``, says which settings exist, their type, default and whether a change applies
live or after a restart. The Settings page is generated from it, and every consumer reads a value with
``get(key)``. Precedence in the code that uses them: acquisition YAML > setting > built-in default;
a command-line flag beats all three.

The values live in ``~/.relayd_settings.json`` (override with the ``RELAYD_SETTINGS_FILE`` environment
variable). A missing or corrupt file means defaults; keys this module does not know are kept.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LIVE, RESTART = "live", "restart"
GROUPS = ("General", "Recording", "Review", "Post-process", "Training", "Inference")


@dataclass(frozen=True)
class Setting:
    key: str
    group: str
    label: str
    kind: str  # bool | int | float | choice | text | folder | file
    default: Any
    tooltip: str = ""
    choices: Tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    applies: str = LIVE


def _s(*args, **kwargs) -> Setting:
    return Setting(*args, **kwargs)


SETTINGS: Tuple[Setting, ...] = (
    # ---- General
    _s("theme", "General", "Theme", "choice", "light", "Colour theme of the window.", ("light", "dark")),
    _s("log_level", "General", "Log level", "choice", "INFO",
       "Console log level. The RELAYD_LOG_LEVEL environment variable, if set, wins.",
       ("DEBUG", "INFO", "WARNING", "ERROR")),
    _s("startup_page", "General", "Page shown at start", "choice", "help", "Page opened when the app starts.",
       ("help", "start", "data", "review", "train", "import", "settings"), applies=RESTART),
    _s("confirm_destructive", "General", "Ask before deleting / stopping", "bool", True,
       "Confirm before deleting a demo or stopping a training run."),
    _s("hdf5_save_folder", "General", "Recordings folder", "folder", "",
       "Where recordings and converted datasets are saved (default: ~/lfd_recordings for recordings)."),
    _s("hdf5_load_folder", "General", "Open-files folder", "folder", "", "Where file dialogs for recordings start."),
    _s("config_folder", "General", "Config folder", "folder", "", "Where the config / options file dialogs start."),
    _s("ros_spin_ms", "General", "ROS spin interval (ms)", "int", 10,
       "How often the GUI processes incoming ROS messages. Lower is more responsive and uses more CPU.",
       minimum=1, maximum=100),
    _s("ros_domain_id", "General", "ROS domain id (-1 = environment)", "int", -1,
       "Sets ROS_DOMAIN_ID for this app. Ignored if the variable is already set, or -1.",
       minimum=-1, maximum=232, applies=RESTART),
    # ---- Recording
    _s("rec_default_frequency", "Recording", "Default recording frequency (Hz)", "int", 30,
       "Used when the acquisition config has no recording_frequency.", minimum=1, maximum=1000),
    _s("rec_default_max_buffer_mb", "Recording", "Default memory buffer (MB)", "int", 4096,
       "Samples stay in RAM until Stop (a crash before Stop loses the demo being recorded); sampling stops at this size. Used when the config has no max_buffer_mb.",
       minimum=64, maximum=1_000_000),
    _s("rec_default_sync_mode", "Recording", "Default sync mode", "choice", "soft",
       "soft: warn when inputs drift apart; strict: skip the sample. Used when the config has no sync_mode.",
       ("soft", "strict")),
    _s("rec_default_sync_tolerance_sec", "Recording", "Default sync tolerance (s, 0 = off)", "float", 0.0,
       "Largest allowed spread between input timestamps. Used when the config has no sync_tolerance_sec.",
       minimum=0.0, maximum=10.0),
    _s("rec_warmup_s", "Recording", "Warm-up wait before recording (s)", "float", 2.0,
       "Time to wait for every input to deliver a first sample after Start.", minimum=0.0, maximum=60.0),
    _s("session_folder_format", "Recording", "Session folder name format", "text", "%Y-%m-%d_%H-%M-%S",
       "strftime pattern of the folder created for each recording session (\"_Recording\" is added at the end)."),
    _s("hdf5_compression", "Recording", "HDF5 compression", "choice", "gzip",
       "Compression of image / stream / point-cloud datasets in recordings and robomimic datasets.",
       ("gzip", "lzf", "none")),
    _s("hdf5_compression_level", "Recording", "gzip level (1-9)", "int", 4,
       "Higher is smaller and slower. Only used with gzip.", minimum=1, maximum=9),
    _s("visualizer_update_rate", "Recording", "Live view refresh (Hz)", "int", 20,
       "Refresh rate of the live containers on the Record page.", minimum=1, maximum=60),
    _s("freeze_containers", "Recording", "Freeze container layout", "bool", True,
       "Stop the live view from resizing containers while recording."),
    _s("tf_wait_timeout_s", "Recording", "TF wait timeout (s)", "float", 10.0,
       "How long a custom TF topic may take to deliver its transform (a per-input tf_wait_timeout in the config wins).",
       minimum=0.5, maximum=300.0),
    _s("topic_discovery_timeout_s", "Recording", "Topic discovery timeout (s)", "float", 5.0,
       "How long to wait for a topic to show up in the ROS graph when subscribing.", minimum=0.5, maximum=120.0),
    _s("qos_depth", "Recording", "Subscription queue depth", "int", 10,
       "QoS history depth of topic subscriptions (recording and inference).", minimum=1, maximum=1000,
       applies=RESTART),
    # ---- Review
    _s("review_default_fixed_frame", "Review", "Default RViz fixed frame", "text", "base_link",
       "Fixed frame used for playback when the robot description has none."),
    _s("review_urdf_timeout_s", "Review", "Robot description timeout (s)", "float", 5.0,
       "How long to wait for a robot description from a topic.", minimum=0.5, maximum=120.0),
    _s("review_playback_tick_ms", "Review", "Playback UI refresh (ms)", "int", 150,
       "How often the slider and plot cursor follow the playback.", minimum=20, maximum=2000),
    _s("review_plot_max_points", "Review", "Plot decimation (max points)", "int", 5000,
       "Long state / action curves are thinned to this many points.", minimum=200, maximum=200_000),
    _s("review_last_robot_description", "Review", "Last robot description file", "file", "",
       "Remembered URDF / xacro of the last playback."),
    # ---- Post-process
    _s("postprocess_default_format", "Post-process", "Default output format", "choice", "robomimic",
       "Format selected when the Post-process page opens.", ("robomimic", "lerobot")),
    _s("postprocess_default_training_model", "Post-process", "Default robomimic training model", "choice",
       "bc_rnn_gmm", "Used when postprocess_config.yaml has no training_model.",
       ("bc_rnn_gmm", "bc_transformer_gmm", "diffusion_policy")),
    _s("split_train", "Post-process", "Default train fraction", "float", 0.8,
       "Used when postprocess_config.yaml has no robomimic.splits.", minimum=0.0, maximum=1.0),
    _s("split_val", "Post-process", "Default validation fraction", "float", 0.2,
       "The rest (1 - train - validation) is the eval split.", minimum=0.0, maximum=1.0),
    _s("lerobot_video_crf", "Post-process", "LeRobot video quality (CRF, lower = better)", "int", 23,
       "x264 constant rate factor of the camera videos.", minimum=0, maximum=51),
    _s("lerobot_video_preset", "Post-process", "LeRobot video encoder preset", "choice", "veryfast",
       "Slower presets give smaller files.", ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")),
    _s("lerobot_gop", "Post-process", "LeRobot video keyframe interval", "int", 2,
       "Frames between keyframes. Small keeps random access fast for training.", minimum=1, maximum=250),
    _s("lerobot_chunks_size", "Post-process", "LeRobot files per chunk folder", "int", 1000,
       "Maximum files per chunk-NNN folder.", minimum=1, maximum=100_000),
    _s("lerobot_data_file_mb", "Post-process", "LeRobot data file size (MB)", "int", 100,
       "A new parquet file starts after this size.", minimum=1, maximum=10_000),
    _s("lerobot_video_file_mb", "Post-process", "LeRobot video file size (MB)", "int", 200,
       "A new mp4 file starts after this size.", minimum=1, maximum=10_000),
    # ---- Training
    _s("python_robomimic", "Training", "Python for robomimic", "file", "",
       "Interpreter that trains and runs robomimic policies (empty: the environment built by the Train tab, else the one running Relay-D)."),
    _s("python_lerobot", "Training", "Python for LeRobot", "file", "",
       "Interpreter that trains and runs LeRobot policies (empty: the environment built by the Train tab, else the one running Relay-D)."),
    _s("train_use_terminal", "Training", "Open a terminal window for each run", "bool", True,
       "Off: runs in the background and logs to train.log."),
    _s("train_default_device", "Training", "Default training device", "choice", "cuda", "", ("cuda", "cpu")),
    _s("train_lerobot_steps", "Training", "LeRobot default steps", "int", 100_000, "", minimum=1, maximum=100_000_000),
    _s("train_lerobot_batch_size", "Training", "LeRobot default batch size", "int", 8, "", minimum=1, maximum=4096),
    _s("train_lerobot_num_workers", "Training", "LeRobot default data workers", "int", 4, "", minimum=0, maximum=64),
    _s("train_lerobot_seed", "Training", "LeRobot default seed", "int", 1000, "", minimum=0, maximum=2_000_000_000),
    # ---- Inference
    _s("infer_device", "Inference", "Default device", "choice", "cuda", "Default of lfd-inference-node --device (the Python API keeps its own default).",
       ("cuda", "cpu")),
    _s("infer_hz", "Inference", "Default control rate (Hz, 0 = as fast as possible)", "float", 10.0,
       "Default of lfd-inference-node --hz for a policy without a rate of its own (robomimic); a LeRobot policy runs at its dataset fps. The Python API runs unthrottled unless told otherwise.", minimum=0.0, maximum=1000.0),
    _s("infer_max_obs_age_s", "Inference", "Stale input limit (s)", "float", 1.0,
       "Default of --max-obs-age: no commands are sent while an input is older than this.", minimum=0.05, maximum=60.0),
    _s("infer_obs_wait_s", "Inference", "Wait for observations at start (s)", "float", 10.0,
       "Default of --obs-wait.", minimum=0.5, maximum=300.0),
    _s("infer_backend", "Inference", "Default backend", "choice", "auto",
       "Default of --backend. auto: robomimic for a .pth file, LeRobot for a pretrained_model folder.",
       ("auto", "robomimic", "lerobot")),
)

BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}
_MIGRATIONS = {"update_rate": "visualizer_update_rate"}  # old key -> new key
_cache: Optional[Dict[str, Any]] = None


def settings_file() -> Path:
    return Path(os.environ.get("RELAYD_SETTINGS_FILE") or Path.home() / ".relayd_settings.json")


def defaults() -> Dict[str, Any]:
    return {s.key: s.default for s in SETTINGS}


def _valid(setting: Setting, value: Any) -> bool:
    if setting.kind == "bool":
        return isinstance(value, bool)
    if setting.kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if setting.kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if setting.kind == "choice":
        return value in setting.choices
    return isinstance(value, str)


def _read_file() -> Dict[str, Any]:
    try:
        data = json.loads(settings_file().read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    for old, new in _MIGRATIONS.items():
        if old in data and new not in data:
            data[new] = data[old]
    return data


def reload() -> Dict[str, Any]:
    """Re-read the file; defaults fill missing or invalid values, unknown keys are kept."""
    global _cache
    merged = _read_file()
    for s in SETTINGS:
        if s.key not in merged or not _valid(s, merged[s.key]):
            merged[s.key] = s.default
        elif s.kind == "float":
            merged[s.key] = float(merged[s.key])
    _cache = merged
    return merged


def values() -> Dict[str, Any]:
    return dict(_cache if _cache is not None else reload())


def get(key: str, default: Any = None) -> Any:
    data = _cache if _cache is not None else reload()
    return data.get(key, default)


def save(new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``new`` into the file (atomic write) and return all values. Invalid values are rejected."""
    for key, value in new.items():
        setting = BY_KEY.get(key)
        if setting is not None and not _valid(setting, value):
            raise ValueError(f"invalid value for '{key}': {value!r}")
    on_disk = _read_file()
    on_disk.update(new)
    # Keep only what differs from the default: a value equal to its default is not a choice, and
    # storing it would pin it forever, so a better default in a later release would never reach this user.
    for key in new:
        setting = BY_KEY.get(key)
        if setting is not None and on_disk[key] == setting.default and type(on_disk[key]) is type(setting.default):
            del on_disk[key]
    for old_key, new_key in _MIGRATIONS.items():
        if new_key in new:
            on_disk.pop(old_key, None)  # else the migration would bring the old value back
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(on_disk, f, indent=4)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return reload()


def changed_restart_keys(old: Dict[str, Any], new: Dict[str, Any]) -> list:
    """Keys whose change only applies after a restart."""
    return [s.key for s in SETTINGS if s.applies == RESTART and old.get(s.key) != new.get(s.key)]


def h5_compression() -> Dict[str, Any]:
    """``create_dataset`` keyword arguments for the ``hdf5_compression`` / ``hdf5_compression_level`` settings."""
    kind = get("hdf5_compression")
    if kind == "gzip":
        return {"compression": "gzip", "compression_opts": int(get("hdf5_compression_level"))}
    if kind == "lzf":
        return {"compression": "lzf"}
    return {}
