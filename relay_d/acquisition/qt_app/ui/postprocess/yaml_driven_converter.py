"""
YAML-driven dataset conversion: recorded HDF5 demos -> robomimic HDF5 or LeRobotDataset v3.0.

Numeric obs and actions come from each demo's `streams/` group (`output.streams`); other
`output:` entries (the bridge map) point an obs key at a recorded HDF5 path,
e.g. agentview_image -> camera_rgb/images.
"""

import h5py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from relay_d.utils.coloring_logger import logger
from relay_d.utils import user_settings
from relay_d.utils.numerics import RATE_TOLERANCE, median_rate, quat_to_euler
from relay_d.acquisition.qt_app.ui.config.builder.config_check import normalization_error, stream_width
from .postprocess_options import FORMAT_SUFFIX, PostprocessOptions, removed_key_errors
from .robomimic_config_handler import explicit_modalities
import re
import json
import os
import traceback
import yaml


def _parse_stream_entry(stream_val):
    """Return (spec_list, rotation_format).

    Accepts plain-list form (list of strings) or dict form:
      specs: [...]
      rotation_format: quat | rpy            # quat (default) or roll-pitch-yaw radians
    """
    if isinstance(stream_val, dict):
        return stream_val.get("specs", []), str(stream_val.get("rotation_format", "quat")).lower()
    return stream_val, "quat"


def _stream_output_width(data_config: Dict, stream_name: str) -> Optional[int]:
    """
    Post-conversion column width for a stream from its configured specs (config_check.stream_width:
    None if a ``[:]`` slice has no joint_names), except a stream with rotation_format: rpy outputs
    3 columns (roll, pitch, yaw) from its 4 quaternion specs.
    """
    streams = data_config.get("output", {}).get("streams", {})
    spec_list, rot_fmt = _parse_stream_entry(streams.get(stream_name, []))
    inputs = {n: v for n, v in (data_config.get("input") or {}).items() if isinstance(v, dict)}
    width = stream_width([str(x) for x in spec_list], inputs)
    if rot_fmt == "rpy":
        if width != 4:
            raise ValueError(
                f"stream '{stream_name}': rotation_format: rpy requires exactly 4 "
                f"quaternion specs (x,y,z,w), but its specs resolve to {width} columns."
            )
        return 3
    return width


def _action_part(part: np.ndarray, key: str, stream_configs: dict, delta_actions: list) -> np.ndarray:
    """One `actions_to_extract` entry as (N, width) columns: its stream's rotation format,
    then the target of step t: the delta to t+1 if listed in `delta_actions`, else the value at t+1
    (the value at t is the current state, which would teach the policy to stand still).
    Shared by both output formats."""
    part = part if part.ndim > 1 else part[:, np.newaxis]
    _, rot_fmt = _parse_stream_entry(stream_configs.get(key, []))
    part = _apply_rotation_format(part, rot_fmt, key)
    return _compute_delta(part) if key in delta_actions else _next_step(part)


def _compute_delta(data: np.ndarray) -> np.ndarray:
    """Return frame-to-frame delta (data[t+1] - data[t]); last step is zero-padded."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    delta = np.zeros_like(data)
    if len(data) > 1:
        delta[:-1] = data[1:] - data[:-1]
    return delta


def _quat_to_rpy(data: np.ndarray, stream_name: str = "") -> np.ndarray:
    """(N, 4) quaternion (x, y, z, w) -> (N, 3) roll-pitch-yaw euler angles, radians."""
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            f"rotation_format: rpy requires a 4-column quaternion stream (x,y,z,w); "
            f"stream '{stream_name}' has shape {tuple(data.shape)}."
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        rpy = quat_to_euler(data).astype(np.float32)
    if not np.isfinite(rpy).all():  # a zero-length quaternion (an unfilled message): fail, never write NaN
        bad = int((~np.isfinite(rpy).all(axis=1)).sum())
        raise ValueError(
            f"rotation_format: rpy: stream '{stream_name}' has {bad} row(s) with a zero-length or non-finite quaternion."
        )
    return rpy


def _apply_rotation_format(
    data: np.ndarray, rotation_format: str, stream_name: str = ""
) -> np.ndarray:
    """Convert *data* to the requested rotation representation (no-op for 'quat')."""
    if rotation_format == "quat":
        return data
    if rotation_format == "rpy":
        return _quat_to_rpy(data, stream_name)
    raise ValueError(
        f"stream '{stream_name}': unknown rotation_format '{rotation_format}' "
        f"(expected 'quat' or 'rpy')."
    )


# ---------------------------------------------------------------------------
# DataTransformEngine
# ---------------------------------------------------------------------------
def _end_flags(n: int) -> np.ndarray:
    """Rewards and dones of a demo: zero except the last step, which is 1."""
    flags = np.zeros(n, dtype=np.float32)
    if n > 0:
        flags[-1] = 1.0
    return flags


def _check_length(data: np.ndarray, num_samples: int, key: str) -> None:
    """The recorder writes one row per timestamp: another length means lost or extra frames, so the demo fails
    instead of being stretched out of step with the actions."""
    if len(data) != num_samples:
        raise ValueError(f"'{key}' has {len(data)} rows but the demo has {num_samples} timesteps")


def _demo_metadata(demo_data: h5py.Group) -> Tuple[Optional[float], Optional[str]]:
    """(recording_frequency, yaml_config file name) the recorder stored in data/<demo>/metadata, None where absent."""
    meta = demo_data.get("metadata")
    attrs = meta.attrs if isinstance(meta, h5py.Group) else {}
    try:
        hz = float(attrs["recording_frequency"]) if "recording_frequency" in attrs else None
    except (TypeError, ValueError):
        hz = None
    config = attrs.get("yaml_config")
    config = config.decode() if isinstance(config, bytes) else config
    return hz, (os.path.basename(str(config)) or None) if config else None


def _names_text(names: List[str]) -> str:
    """'(7) a.0..3, b.0..2' from LeRobot column names (a.0, a.1, ..., b)."""
    groups: Dict[str, int] = {}
    for name in names:
        key, _, idx = name.rpartition(".")
        key = key if key and idx.isdigit() else name
        groups[key] = groups.get(key, 0) + 1
    parts = [k if k in names else f"{k}.0..{n - 1}" for k, n in groups.items()]
    return f"({len(names)}) " + ", ".join(parts)


def _splits_error(train, val) -> Optional[str]:
    """Why robomimic.splits cannot make the masks, None if they can."""
    for name, frac in (("train", train), ("val", val)):
        if not isinstance(frac, (int, float)) or isinstance(frac, bool) or not 0 <= frac <= 1:
            return f"robomimic.splits: '{name}' must be a number between 0 and 1, got {frac!r}"
    if train == 0:  # every robomimic template trains on the "train" mask: no mask, no training
        return "robomimic.splits: 'train' must be more than 0"
    if train + val > 1 + 1e-9:
        return f"robomimic.splits: train ({train}) + val ({val}) is more than 1"
    return None


def _split_counts(total: int, train: float, val: float) -> Tuple[int, int]:
    """(train, valid) demo counts; demos left over by rounding or train + val < 1 also train."""
    # At least one training demo when training is asked for (round() would give 0 for few demos).
    train_count = max(1, round(total * train)) if train > 0 and total else 0
    val_count = min(round(total * val), total - train_count)
    if train_count:
        train_count = total - val_count
    return train_count, val_count


def _next_step(obs: np.ndarray) -> np.ndarray:
    """value[t+1]; the last step repeats itself so the length stays N."""
    return np.concatenate([obs[1:], obs[-1:]], axis=0)


# robomimic's train.py always reads env_args from the dataset; with rollout disabled it only needs these fields
# (type 1 = robosuite keeps robomimic's depth processing as before). Each conversion adds
# env_kwargs.control_freq = recording_frequency: robomimic copies env_args into the checkpoint (env_metadata),
# where inference reads the rate the policy was trained at.
ENV_ARGS = {"env_name": "relay_d", "type": 1, "env_kwargs": {}}

FPS_REQUIRED = "A positive 'recording_frequency' is required in the config (used as the fps)."


def natural_key(path: str) -> List:
    """Sort key for paths with numbers: rec_2.h5 before rec_10.h5. The whole path counts, so same-named
    files of several session folders keep a fixed order (folder first)."""
    parts = re.split(r"(\d+)", path)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _is_converted_robomimic(path: str) -> bool:
    """True (and logged) for a robomimic dataset written by a conversion (it has a ``mask`` group): not a recording."""
    try:
        with h5py.File(path, "r") as f:
            converted = "mask" in f
    except OSError:
        return False  # reported as unreadable by the conversion loop
    if converted:
        logger.warning(f"Skipping {path}: it is a converted robomimic dataset, not a recording.")
    return converted


def _recordings(input_files: List[str]) -> List[str]:
    """The input files in conversion order, without converted robomimic datasets."""
    return [f for f in sorted(input_files, key=natural_key) if not _is_converted_robomimic(f)]


class DataTransformEngine:
    """
    Bridge map (obs key -> recorded HDF5 path) of a data_config (the `config` sub-key).
    """

    def __init__(self, data_config: Optional[Dict] = None):
        self.data_config = data_config or {}

        # Core maps built once at construction
        self.bridge_map: Dict[str, str] = self._build_bridge_map()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_bridge_map(self) -> Dict[str, str]:
        """
        Build logical_key -> recorded_path mapping from data_config.yaml `output`.

        Example entry in data_config.yaml:
            output:
              robot0_eef_pos: robot0_eef/translations
        """
        bridge_map = {}
        output_section = self.data_config.get("output", {})

        for logical_key, recorded_path in output_section.items():
            if logical_key in ("prefix_path", "streams"):
                continue
            bridge_map[logical_key] = recorded_path

        logger.debug(
            f"Bridge map built with {len(bridge_map)} entries: {list(bridge_map.keys())}"
        )
        return bridge_map

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_path(self, logical_key: str) -> Optional[str]:
        """
        Resolve a logical key to the actual recorded HDF5 path via the bridge map.

        Args:
            logical_key: e.g. "robot0_eef_pos"

        Returns:
            Recorded path e.g. "robot0_eef/translations", or None if unmapped.
        """
        return self.bridge_map.get(logical_key)

    # ------------------------------------------------------------------
    # HDF5 nested data reader
    # ------------------------------------------------------------------

    def _get_nested_data(self, group: h5py.Group, path: str) -> Optional[h5py.Dataset]:
        """
        Traverse an HDF5 group using a slash-delimited path.
        Returns the Dataset if found, None otherwise.
        """
        if not path:
            return None

        parts = path.split("/")
        current: Any = group
        for part in parts:
            if not isinstance(current, (h5py.Group, h5py.File)):
                return None
            if part not in current:
                return None
            current = current[part]

        return current if isinstance(current, h5py.Dataset) else None


# ---------------------------------------------------------------------------
# YAMLDrivenConverter
# ---------------------------------------------------------------------------
class YAMLDrivenConverter:
    """
    Orchestrator: reads YAML configs, iterates over input HDF5 files,
    converts each demo to Robomimic 1.0 format, and writes the output.
    """

    def __init__(
        self,
        config_loader,
        data_config: Optional[Dict] = None,
    ):
        self.config_loader = config_loader
        # data_config is the `config` sub-dict from data_config.yaml
        self.data_config: Dict = data_config or {}
        # Per-batch cache: logical_key -> (feature_shape, dtype), built by
        # _prescan_shapes() from real recorded data before each conversion.
        self._shape_cache: Dict[str, Tuple[Tuple[int, ...], Any]] = {}
        # Short human-readable reason for the last failed conversion (shown by the GUI).
        self.last_error: Optional[str] = None
        # Per-format conversion options (postprocess_config.yaml); defaults without one.
        self.options: PostprocessOptions = PostprocessOptions()
        # Demos of the last conversion: how many were written, and "file::demo" of each one dropped.
        self.converted_demos: int = 0
        self.skipped_demos: List[str] = []
        # Bridge-map keys with zero data anywhere in the current batch and no
        # way to infer a shape for them — excluded from output entirely.
        self._excluded_keys: Set[str] = set()
        # Per-batch: output key -> "rgb"|"depth"|"scan", from the `modality` of the
        # input it reads. Only keys with an EXPLICIT modality are present here —
        # plain rgb (the common case) is instead detected from the real recorded dtype (uint8).
        self._modality_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data_config(self, data_config_path: str) -> bool:
        """
        Load (or reload) the data_config.yaml bridge map.

        Args:
            data_config_path: Absolute path to data_config.yaml

        Returns:
            True on success, False on error.
        """
        try:
            with open(data_config_path, "r") as f:
                raw = yaml.safe_load(f)
            self.data_config = raw.get("config", raw)
            logger.info(
                f"Loaded data_config from {data_config_path} "
                f"({len(self.data_config.get('output', {}))} bridge entries)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load data_config.yaml: {e}")
            return False

    def _prepare_batch(
        self,
        format_name: str,
        input_files: List[str],
        data_config: Optional[Dict] = None,
    ) -> Optional[Tuple["DataTransformEngine", List[str]]]:
        """
        Per-batch setup shared by every output format.

        Applies an optional data_config override, sorts the input files
        naturally, builds the transform engine and the real-data shape cache
        (zero-fill widths, excluded keys). Returns (engine, sorted_files), or
        None if the format config is missing.
        """
        if format_name not in FORMAT_SUFFIX:
            self.last_error = f"Unknown format: '{format_name}'"
            logger.error(self.last_error)
            return None

        if data_config:
            self.data_config = data_config
        if removed := removed_key_errors(self.data_config):
            self.last_error = " ".join(removed)
            logger.error(self.last_error)
            return None
        if not self.data_config.get("actions_to_extract"):
            self.last_error = "actions_to_extract is empty: the config names no stream to use as the action."
            logger.error(self.last_error)
            return None
        for name, entry in (self.data_config.get("output", {}).get("streams") or {}).items():
            if msg := normalization_error(name, entry):
                self.last_error = msg
                logger.error(msg)
                return None

        sorted_files = _recordings(input_files)

        # bridge_map/required_observations depend only on data_config, which is
        # identical for every demo in this batch — build the engine once.
        engine = DataTransformEngine(self.data_config)
        self._modality_map = explicit_modalities(self.data_config)

        # Real-data shape/dtype cache: for every observation the config asks
        # for, find the first demo in the whole batch where it actually
        # resolves and record its real (feature_shape, dtype). Used to
        # zero-fill demos missing that key instead of guessing from its name.
        streams_cfg = self.data_config.get("output", {}).get("streams", {})
        required_keys = set(streams_cfg.keys()) | set(engine.bridge_map)
        self._shape_cache, unresolved = self._prescan_shapes(
            sorted_files, engine, required_keys
        )
        if missing := sorted(k for k in unresolved if k in streams_cfg):
            # a config these recordings were not made with: every demo would get zero columns
            have, configs = self._recorded_streams(sorted_files)
            self.last_error = (
                f"Configured stream(s) {', '.join(missing)} are in none of the selected recordings; "
                f"they have {', '.join(sorted(have)) or 'no streams'}"
                + (f" (recorded with {', '.join(sorted(configs))})" if configs else "")
                + ". Load the config these demos were recorded with."
            )
            logger.error(self.last_error)
            return None
        self._excluded_keys = set(unresolved)
        for key in sorted(unresolved):
            logger.error(
                f"Observation '{key}' has no recorded data anywhere in this batch "
                f"and no inferable width; excluding it from the converted dataset."
            )

        return engine, sorted_files

    def _recorded_streams(self, files: List[str]) -> Tuple[Set[str], Set[str]]:
        """Stream names and recorded-with config names found in these recordings (for the mismatch message)."""
        streams: Set[str] = set()
        configs: Set[str] = set()
        for path in files:
            try:
                with h5py.File(path, "r") as f:
                    for name in self._find_demo_groups(f):
                        demo = f["data"][name]
                        if isinstance(demo.get("streams"), h5py.Group):
                            streams.update(demo["streams"].keys())
                        if config := _demo_metadata(demo)[1]:
                            configs.add(config)
            except OSError:
                continue
        return streams, configs

    def convert_multiple_files(
        self,
        format_name: str,
        input_files: List[str],
        output_path: str,
        data_config: Optional[Dict] = None,
    ) -> bool:
        """
        Convert and merge multiple input HDF5 files into one Robomimic-format output.

        Args:
            format_name:  Target format key (e.g. "robomimic")
            input_files:  List of input HDF5 file paths
            output_path:  Destination HDF5 path
            data_config:  Optional data_config override (the `config` sub-dict)

        Returns:
            True on success. False on failure, with the reason in ``self.last_error``
            (also when no demo could be converted). A demo that fails to write is
            dropped and the remaining ones are still converted.
        """
        self.last_error = None
        self.converted_demos, self.skipped_demos = 0, []
        if format_name == "lerobot":
            return self.convert_to_lerobot(
                input_files, output_path, data_config=data_config
            )

        prepared = self._prepare_batch(format_name, input_files, data_config)
        if prepared is None:
            return False
        engine, sorted_files = prepared
        freq = float(self.data_config.get("recording_frequency") or 0)  # required by the config check; 0 = no rate check

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            with h5py.File(output_path, "w") as out_f:
                data_grp = out_f.create_group("data")
                total_demos = 0
                processed: List[str] = []
                skipped: List[str] = []
                counter = 0

                for file_path in sorted_files:
                    if not file_path.endswith(".h5"):
                        logger.warning(f"Skipping non-HDF5 file: {file_path}")
                        skipped.append(os.path.basename(file_path))
                        continue

                    try:
                        with h5py.File(file_path, "r") as in_f:
                            for raw_demo_name in sorted(self._find_demo_groups(in_f)):
                                in_data_grp = in_f.get("data")
                                if not isinstance(in_data_grp, h5py.Group):
                                    continue
                                demo_data = in_data_grp.get(raw_demo_name)
                                if not isinstance(demo_data, h5py.Group):
                                    continue

                                if reason := self._demo_problem(demo_data, engine, freq):
                                    logger.error(f"Skipped {file_path}::{raw_demo_name}: {reason}")
                                    self.skipped_demos.append(f"{file_path}::{raw_demo_name}")
                                    continue

                                out_name = f"demo_{counter}"
                                converted = self._convert_single_demo(
                                    demo_data, engine, out_name
                                )

                                if converted:
                                    try:
                                        self._write_demo_to_output(
                                            data_grp, out_name, converted
                                        )
                                    except Exception as e:
                                        # drop the half-written group so the next demo can reuse the name
                                        if out_name in data_grp:
                                            del data_grp[out_name]
                                        logger.error(f"Skipped {file_path}::{raw_demo_name}: {e}")
                                        self.skipped_demos.append(f"{file_path}::{raw_demo_name}")
                                        continue
                                    processed.append(out_name)
                                    total_demos += converted["num_samples"]
                                    counter += 1
                                    logger.info(
                                        f"Converted {file_path}::{raw_demo_name} -> "
                                        f"{out_name} ({converted['num_samples']} samples)"
                                    )
                                else:
                                    logger.warning(
                                        f"Skipped {file_path}::{raw_demo_name} (conversion failed)"
                                    )
                                    self.skipped_demos.append(f"{file_path}::{raw_demo_name}")

                    except Exception as e:
                        fname = os.path.basename(file_path)
                        logger.error(f"Skipping {fname} — could not open: {e}")
                        skipped.append(fname)
                        continue

                if counter == 0:
                    self.last_error = "No demos could be converted."
                    logger.error(self.last_error)
                    self._discard_output(output_path)
                    return False

                # Robomimic requires `total` = number of demos (not total timesteps)
                data_grp.attrs["total"] = counter
                # env_args MUST be a JSON string (Robomimic ObsUtils requirement)
                env_args = {**ENV_ARGS, "env_kwargs": {"control_freq": freq}} if freq > 0 else ENV_ARGS
                data_grp.attrs["env_args"] = json.dumps(env_args)

                splits = self.options.splits()
                if not self._create_data_mask(out_f.create_group("mask"), processed, splits["train_frac"], splits["val_frac"]):
                    self._discard_output(output_path)
                    return False

            if skipped:
                logger.warning(
                    f"Skipped {len(skipped)} file(s) (corrupted or unreadable): {skipped}"
                )
            self.converted_demos = counter
            logger.info(
                f"Successfully wrote {counter} demos ({total_demos} timesteps) -> {output_path}"
            )
            return True

        except Exception as e:
            self.last_error = f"Fatal error during conversion: {e}"
            logger.error(self.last_error)
            traceback.print_exc()
            self._discard_output(output_path)
            return False

    @staticmethod
    def _discard_output(path: str) -> None:
        """Delete a half-written output file (a leftover would make the CLI refuse to run again)."""
        try:
            os.remove(path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # LeRobotDataset v3.0 output
    # ------------------------------------------------------------------

    def convert_to_lerobot(
        self,
        input_files: List[str],
        output_dir: str,
        data_config: Optional[Dict] = None,
        task: Optional[str] = None,
    ) -> bool:
        """
        Convert recorded HDF5 files into a LeRobotDataset v3.0 directory.

        Uses the same acquisition config as the robomimic conversion:
          - actions            <- `actions_to_extract` (as for robomimic)
          - observation.state  <- numeric `output.streams`, concatenated in
                                  declaration order (or `lerobot.state_streams`)
          - observation.images.<key> <- uint8 image obs from bridge-map entries
          - fps                <- `recording_frequency`
          - task               <- `lerobot.task`, else `dataset_name`

        Args:
            input_files: recorded HDF5 files
            output_dir:  dataset directory to create (must not exist or be empty)
            data_config: optional data_config override (the `config` sub-dict)
            task:        optional task text overriding the config
        """
        from .lerobot_writer import (
            LeRobotDatasetWriter,
            LeRobotDependencyError,
            LeRobotFatalError,
            LeRobotWriterError,
            check_dependencies,
        )

        self.last_error = None
        self.converted_demos, self.skipped_demos = 0, []
        prepared = self._prepare_batch("lerobot", input_files, data_config)
        if prepared is None:
            return False
        engine, sorted_files = prepared

        # Fail before reading any demo if a required library is missing. PyAV is only
        # needed when the RECORDED data really contains camera images: bridge-map keys
        # with no recorded data are excluded by _prepare_batch and never become cameras.
        def _is_recorded_camera(key: str) -> bool:
            cached = self._shape_cache.get(key)
            return (
                key not in self._excluded_keys
                and cached is not None
                and np.dtype(cached[1]) == np.uint8
                and len(cached[0]) == 3
                and cached[0][-1] == 3
            )

        camera_keys = [k for k in engine.bridge_map if _is_recorded_camera(k)]
        unrecorded = sorted(
            k for k in engine.bridge_map
            if k in self._excluded_keys and str(engine.bridge_map[k]).rstrip("/").endswith("images")
        )
        if unrecorded:
            logger.warning(
                f"No recorded images for {unrecorded}; the LeRobot dataset will have no "
                f"camera video for them."
            )
        needs_video = bool(camera_keys)
        try:
            check_dependencies(need_video=needs_video)
        except LeRobotDependencyError as e:
            self.last_error = str(e)
            logger.error(self.last_error)
            return False

        lerobot_cfg = self.options.lerobot()
        fps = self._lerobot_fps()
        if fps is None:
            self.last_error = FPS_REQUIRED
            return False
        task_text = self._lerobot_task(task)

        written = 0
        skipped: List[str] = []
        try:
            with LeRobotDatasetWriter(
                output_dir, fps=fps, robot_type=lerobot_cfg.get("robot_type"),
                # encoder and file-size settings (Settings page / ~/.relayd_settings.json)
                chunks_size=user_settings.get("lerobot_chunks_size"),
                data_files_size_in_mb=user_settings.get("lerobot_data_file_mb"),
                video_files_size_in_mb=user_settings.get("lerobot_video_file_mb"),
                video_crf=user_settings.get("lerobot_video_crf"),
                video_preset=user_settings.get("lerobot_video_preset"),
                video_gop=user_settings.get("lerobot_gop"),
            ) as writer:
                for file_path in sorted_files:
                    if not file_path.endswith(".h5"):
                        logger.warning(f"Skipping non-HDF5 file: {file_path}")
                        skipped.append(os.path.basename(file_path))
                        continue
                    try:
                        with h5py.File(file_path, "r") as in_f:
                            in_data_grp = in_f.get("data")
                            if not isinstance(in_data_grp, h5py.Group):
                                continue
                            for raw_demo_name in sorted(self._find_demo_groups(in_f)):
                                demo_data = in_data_grp.get(raw_demo_name)
                                if not isinstance(demo_data, h5py.Group):
                                    continue
                                if reason := self._demo_problem(demo_data, engine, fps):
                                    logger.error(f"Skipped {file_path}::{raw_demo_name}: {reason}")
                                    self.skipped_demos.append(f"{file_path}::{raw_demo_name}")
                                    continue
                                demo = self._convert_single_demo(
                                    demo_data, engine, f"lerobot_ep_{written}"
                                )
                                if not demo:
                                    logger.warning(
                                        f"Skipped {file_path}::{raw_demo_name} (conversion failed)"
                                    )
                                    self.skipped_demos.append(f"{file_path}::{raw_demo_name}")
                                    continue
                                try:
                                    state, state_names, images = self._lerobot_split_obs(
                                        demo, lerobot_cfg.get("state_streams")
                                    )
                                    writer.add_episode(
                                        state,
                                        demo["actions"],
                                        images,
                                        task=task_text,
                                        state_names=state_names,
                                        action_names=self._lerobot_action_names(
                                            demo["actions"].shape[1]
                                        ),
                                    )
                                except LeRobotFatalError:
                                    raise  # missing library or aborted dataset: every later episode would be wrong too
                                except LeRobotWriterError as e:
                                    logger.error(
                                        f"Skipped {file_path}::{raw_demo_name}: {e}"
                                    )
                                    self.skipped_demos.append(f"{file_path}::{raw_demo_name}")
                                    continue
                                written += 1
                                logger.info(
                                    f"Converted {file_path}::{raw_demo_name} -> "
                                    f"episode {written - 1} ({demo['num_samples']} frames)"
                                )
                    except LeRobotFatalError:
                        raise
                    except Exception as e:  # skip a broken file: reaching the writer would delete the whole output
                        fname = os.path.basename(file_path)
                        logger.error(f"Skipping {fname} — could not read: {e}")
                        skipped.append(fname)

                if written == 0:
                    logger.error("No episodes were converted; nothing to write.")
                    raise LeRobotWriterError("No episodes were converted")

            if skipped:
                logger.warning(
                    f"Skipped {len(skipped)} file(s) (corrupted or unreadable): {skipped}"
                )
            self.converted_demos = written
            logger.info(f"Successfully wrote {written} episodes -> {output_dir}")
            return True

        except LeRobotDependencyError as e:
            self.last_error = str(e)
            logger.error(self.last_error)
            return False
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Fatal error during LeRobot conversion: {e}")
            traceback.print_exc()
            return False

    def conversion_summary(self) -> str:
        """One line for the GUI / CLI: how many demos of the last conversion were written."""
        total = self.converted_demos + len(self.skipped_demos)
        text = f"Converted {self.converted_demos} of {total} demos"
        return text + (f" ({len(self.skipped_demos)} skipped, see log)" if self.skipped_demos else "")

    # ------------------------------------------------------------------
    # Pre-check: what a conversion would do, nothing written
    # ------------------------------------------------------------------

    def precheck(self, format_name: str, input_files: List[str]) -> Dict[str, Any]:
        """
        What converting ``input_files`` to ``format_name`` would do, without writing anything (no LeRobot/PyAV needed):
        the batch error the conversion would stop with, each demo's verdict from the same `_demo_problem` the
        conversion runs, and the output layout from the real config + recorded shapes.
        """
        self.last_error = None
        report: Dict[str, Any] = {"format": format_name, "error": None, "demos": [], "output": [],
                                  "converted": 0, "total": 0}
        prepared = self._prepare_batch(format_name, input_files)
        engine, files = prepared if prepared else (None, _recordings(input_files))
        freq = float(self.data_config.get("recording_frequency") or 0)
        if prepared and format_name == "lerobot":
            freq = self._lerobot_fps() or 0
            self.last_error = None if freq else FPS_REQUIRED
        elif prepared:
            splits = self.options.splits()
            self.last_error = _splits_error(splits["train_frac"], splits["val_frac"])
        report["error"] = self.last_error

        for path in files:
            if not path.endswith(".h5"):
                continue  # as the conversion: a skipped file, not a demo
            try:
                with h5py.File(path, "r") as f:
                    for name in sorted(self._find_demo_groups(f)):
                        demo = f["data"][name]
                        ts = demo.get("timestamps")
                        ts = ts[:] if isinstance(ts, h5py.Dataset) else None
                        streams = demo.get("streams")
                        first = next(iter(streams.values()), None) if isinstance(streams, h5py.Group) else None
                        recorded_hz, config = _demo_metadata(demo)
                        if report["error"]:
                            reason = self._rate_problem(demo, freq) or "not converted (batch error)"
                        else:
                            reason = self._demo_problem(demo, engine, freq)
                        report["demos"].append({
                            "file": path, "demo": name,
                            "samples": len(ts) if ts is not None else (len(first) if first is not None else 0),
                            "rate_hz": median_rate(ts) if ts is not None else None,
                            "recorded_with": config, "recorded_hz": recorded_hz,
                            "ok": not reason, "reason": reason,
                        })
            except OSError as e:
                logger.warning(f"Pre-check: could not read {path}: {e}")

        report["total"] = len(report["demos"])
        report["converted"] = sum(d["ok"] for d in report["demos"])
        if not report["error"]:
            report["output"] = self._output_preview(format_name, engine, report["converted"], freq)
        return report

    def _output_preview(self, format_name: str, engine: DataTransformEngine, demos: int, freq: float) -> List[List[str]]:
        """[name, detail] rows of what the conversion writes, from the shape cache of _prepare_batch."""
        streams_cfg = self.data_config.get("output", {}).get("streams", {})
        obs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}  # key -> (feature shape, dtype) as written
        for key, entry in streams_cfg.items():
            shape = self._shape_cache[key][0]
            obs[key] = ((3,) if _parse_stream_entry(entry)[1] == "rpy" else shape, np.dtype(np.float32))
        for key in sorted(engine.bridge_map):
            if key not in self._excluded_keys and key not in obs:
                shape, dtype = self._shape_cache[key]
                kept = np.dtype(dtype) == np.uint8 or key in self._modality_map  # else read as float32
                obs[key] = (shape, np.dtype(dtype) if kept else np.dtype(np.float32))
        widths = [(k, (obs[k][0] or (1,))[0] if k in obs else 0) for k in self.data_config.get("actions_to_extract", [])]
        dim = sum(w for _, w in widths)
        cols, start = [], 0
        for key, w in widths:
            cols.append(f"{key}[{start}]" if w == 1 else f"{key}[{start}:{start + w}]")
            start += w
        actions = f"({dim}) = " + ", ".join(cols)

        if format_name == "lerobot":
            from .lerobot_writer import LeRobotWriterError

            cfg = self.options.lerobot()
            fake = {"num_samples": 1, "obs": {k: np.zeros((1,) + s, d) for k, (s, d) in obs.items()}}
            try:  # the converter's own split, on one zero row per key
                _, state_names, images = self._lerobot_split_obs(fake, cfg.get("state_streams"))
                state = _names_text(state_names)
            except LeRobotWriterError as e:
                state, images = str(e), {}
            action_names = self._lerobot_action_names(dim)
            return ([["observation.state", state]]
                    + [[k, f"video 3x{a.shape[1]}x{a.shape[2]}"] for k, a in images.items()]
                    + [["action", _names_text(action_names) if action_names else actions], ["fps", f"{freq:g}"],
                       ["task", self._lerobot_task()], ["robot_type", str(cfg.get("robot_type") or "-")],
                       ["episodes", str(demos)]])

        rows = [[f"data/demo_N/obs/{k}", f"(N{''.join(f', {x}' for x in s)}) {d}"] for k, (s, d) in obs.items()]
        splits = self.options.splits()
        train, val = _split_counts(demos, splits["train_frac"], splits["val_frac"])
        return rows + [
            ["data/demo_N/actions", actions.replace("(", "(N, ", 1)],
            ["data/demo_N/rewards, dones", "(N,) float32, 1 at the last step"],
            ["mask/train, mask/valid",
             f"{train} / {val} demos (train_frac {splits['train_frac']:g}, val_frac {splits['val_frac']:g})"],
            ["env_args", f"control_freq {freq:g}" if freq else "no control_freq"],
        ]

    @staticmethod
    def _rate_problem(demo_data: h5py.Group, freq: float) -> str:
        """Why the demo's rate is unusable ("" if fine): more than RATE_TOLERANCE off ``freq`` means the policy, run at
        ``freq``, would move rate/freq faster than shown. A demo without timestamps (or freq 0) is not checked."""
        ts = demo_data.get("timestamps")
        rate = median_rate(ts[:]) if freq > 0 and isinstance(ts, h5py.Dataset) else None
        if rate is None or abs(rate - freq) <= RATE_TOLERANCE * freq:
            return ""
        recorded_hz, config = _demo_metadata(demo_data)
        if recorded_hz is not None and recorded_hz != freq:
            return (f"recorded at {recorded_hz:g} Hz with {config or 'another config'}; the loaded config says "
                    f"{freq:g} Hz: load the config these demos were recorded with")
        return (f"recorded at {rate:.1f} Hz, but recording_frequency is {freq:g}: this PC could not keep up. "
                f"Set recording_frequency to a rate this PC keeps up with and record again.")

    def _demo_problem(self, demo_data: h5py.Group, engine: DataTransformEngine, freq: float) -> str:
        """Why this demo would be skipped, "" if it converts. The one set of per-demo rules, used by the conversion
        loops and by precheck(); reads dataset shapes only (and the timestamps), never the arrays."""
        if reason := self._rate_problem(demo_data, freq):
            return reason
        streams_grp = demo_data.get("streams")
        if not isinstance(streams_grp, h5py.Group):
            return "no 'streams' group (not recorded by Relay-D?)"
        streams_cfg = self.data_config.get("output", {}).get("streams", {})
        present = [k for k in streams_cfg if k in streams_grp]
        if not present or not len(streams_grp[present[0]]):
            return "no data in the configured streams"
        n = len(streams_grp[present[0]])
        actions = self.data_config.get("actions_to_extract", [])
        for key in actions:
            if key not in streams_grp:  # zero actions would train on zeros: the demo is skipped
                return f"no {key} (action stream)"
        try:  # the recorder writes one row per timestamp: another length means lost or extra frames
            for key in dict.fromkeys(present + list(actions)):
                _check_length(streams_grp[key], n, key)
            for key in sorted(engine.bridge_map):
                if key in self._excluded_keys or key in streams_cfg:
                    continue
                ds = engine._get_nested_data(demo_data, engine.resolve_path(key) or "")
                if ds is not None:
                    _check_length(ds, n, key)
                    continue
                shape, dtype = self._shape_cache[key]
                if np.dtype(dtype) == np.uint8 or len(shape) >= 2 or key in self._modality_map:
                    # black frames would teach the policy to act blind (a missing numeric obs is zero-filled)
                    return f"no {'point cloud' if self._modality_map.get(key) == 'scan' else 'camera'} {key}"
        except ValueError as e:
            return str(e)
        return ""

    def _lerobot_task(self, task: Optional[str] = None) -> str:
        return task or self.options.lerobot().get("task") or self.data_config.get("dataset_name") or "unspecified task"

    def _lerobot_fps(self) -> Optional[int]:
        """fps from `recording_frequency`."""
        raw = self.data_config.get("recording_frequency")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            logger.error(
                "LeRobot output needs a positive `recording_frequency` in the "
                "acquisition config (used as the dataset fps)."
            )
            return None
        if value != round(value):
            logger.warning(
                f"recording_frequency {value} is not an integer; LeRobot requires an "
                f"integer fps, using {int(round(value))}."
            )
        return int(round(value))

    def _lerobot_split_obs(
        self, demo: Dict, state_streams: Optional[List[str]]
    ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
        """
        Split a converted demo's obs into (observation.state, state column
        names, {observation.images.<key>: (N, H, W, 3) uint8}).

        Numeric streams go to the state vector (declaration order, or the
        `lerobot.state_streams` override); uint8 (N, H, W, 3) arrays are
        cameras. Depth and scan observations are not supported yet and are
        skipped with a warning.
        """
        from .lerobot_writer import LeRobotWriterError

        n = demo["num_samples"]
        obs: Dict[str, np.ndarray] = demo["obs"]
        images: Dict[str, np.ndarray] = {}
        numeric: Dict[str, np.ndarray] = {}
        for key, arr in obs.items():
            arr = np.asarray(arr)
            modality = self._modality_map.get(key)
            if modality in ("depth", "scan"):
                logger.warning(f"Skipping '{key}': {modality} is not supported in LeRobot output yet.")
            elif arr.dtype == np.uint8 and arr.ndim == 4 and arr.shape[3] == 3:
                images[f"observation.images.{key}"] = arr
            elif arr.dtype == np.uint8 or arr.ndim > 2:
                logger.warning(
                    f"Skipping '{key}' (shape {arr.shape}, dtype {arr.dtype}): "
                    f"not a numeric vector or an RGB/BGR image."
                )
            else:
                numeric[key] = arr.astype(np.float32).reshape(n, -1)

        if state_streams:
            unknown = [k for k in state_streams if k not in numeric]
            if unknown:
                logger.warning(f"lerobot.state_streams entries not found in this demo: {unknown}")
            order = [k for k in state_streams if k in numeric]
        else:
            declared = list(self.data_config.get("output", {}).get("streams", {}).keys())
            order = [k for k in declared if k in numeric]
            order += [k for k in numeric if k not in order]

        if not order:
            raise LeRobotWriterError("No numeric streams available for observation.state")

        names: List[str] = []
        for key in order:
            width = numeric[key].shape[1]
            names.extend([key] if width == 1 else [f"{key}.{i}" for i in range(width)])
        state = np.concatenate([numeric[k] for k in order], axis=1)
        return state, names, images

    def _lerobot_action_names(self, dim: int) -> Optional[List[str]]:
        """Column names for `action` from `actions_to_extract` stream widths, if they add up."""
        keys = self.data_config.get("actions_to_extract", []) or []
        names: List[str] = []
        for key in keys:
            width = _stream_output_width(self.data_config, key) or 0
            if width <= 0:
                return None
            names.extend([key] if width == 1 else [f"{key}.{i}" for i in range(width)])
        return names if len(names) == dim else None

    # ------------------------------------------------------------------
    # Demo conversion
    # ------------------------------------------------------------------

    def _convert_single_demo_from_streams(
        self,
        demo_data: h5py.Group,
        demo_name: str,
    ) -> Optional[Dict]:
        """
        Stream-based conversion path.
        Reads pre-computed stream datasets from raw HDF5 `streams/` group and
        maps them directly to obs. Actions are assembled from
        `actions_to_extract` stream names.
        """
        try:
            streams_grp = demo_data.get("streams")
            if not isinstance(streams_grp, h5py.Group):
                return None

            stream_names = list(
                self.data_config.get("output", {}).get("streams", {}).keys()
            )
            if not stream_names:
                logger.warning(f"No stream names found in data_config for {demo_name}")
                return None

            num_samples = None
            for name in stream_names:
                if name in streams_grp:
                    num_samples = len(streams_grp[name])
                    break
            if not num_samples:
                logger.warning(f"No stream data found for {demo_name}")
                return None

            demo: Dict = {
                "num_samples": num_samples,
                "obs": {},
                "actions": None,
                "rewards": None,
                "dones": None,
            }

            stream_configs = self.data_config.get("output", {}).get("streams", {})
            for stream_name in stream_names:
                _, rot_fmt = _parse_stream_entry(stream_configs.get(stream_name, []))
                if stream_name in streams_grp:
                    data = np.array(streams_grp[stream_name], dtype=np.float32)
                    data = _apply_rotation_format(data, rot_fmt, stream_name)
                else:
                    cached = self._shape_cache.get(stream_name)
                    if cached is None:
                        logger.warning(
                            f"Stream '{stream_name}' missing in {demo_name} and has no "
                            f"recorded shape anywhere in the batch; skipping."
                        )
                        continue
                    feat_shape, dtype = cached
                    if rot_fmt == "rpy" and feat_shape == (4,):
                        # Keep zero-filled demos consistent with converted demos:
                        # the shape cache stores the raw recorded (quat) shape.
                        feat_shape = (3,)
                    logger.warning(
                        f"Stream '{stream_name}' missing in {demo_name}; zero-filling "
                        f"with shape {feat_shape} inferred from the rest of the batch."
                    )
                    data = np.zeros((num_samples,) + feat_shape, dtype=dtype)

                demo["obs"][stream_name] = data

            actions_to_extract = self.data_config.get("actions_to_extract", [])
            delta_actions = self.data_config.get("delta_actions", [])
            action_parts = []
            for stream_name in actions_to_extract:  # present and the right length: checked by _demo_problem
                part = np.array(streams_grp[stream_name], dtype=np.float32)
                action_parts.append(_action_part(part, stream_name, stream_configs, delta_actions))

            action_parts = [p for p in action_parts if p.ndim < 2 or p.shape[1] > 0]
            demo["actions"] = np.concatenate(action_parts, axis=1)  # all zero-width: ValueError, the demo fails

            demo["rewards"] = _end_flags(num_samples)
            demo["dones"] = _end_flags(num_samples)

            logger.info(
                f"Stream-based conversion: {demo_name}, "
                f"N={num_samples}, obs={list(demo['obs'].keys())}, "
                f"actions shape={demo['actions'].shape}"
            )
            return demo

        except Exception as e:
            logger.error(f"Error in stream-based conversion of '{demo_name}': {e}")
            traceback.print_exc()
            return None

    def _populate_bridge_obs(
        self,
        demo_data: h5py.Group,
        engine: DataTransformEngine,
        demo: Dict[str, Any],
        num_samples: int,
        demo_name: str,
    ) -> None:
        """
        Populate demo['obs'] for every bridge-map key (top-level
        `output:` entries other than `streams`) not already present in demo['obs'].

        This is how Image / organized PointCloud2 data reaches obs:
        those types can't be expressed as output.streams specs (numeric-only —
        see StreamBuilder), so they're referenced directly by their raw HDF5
        path instead, e.g. `output: agentview_image: camera_rgb/images`.
        """
        required_keys = sorted(
            k for k in engine.bridge_map
            if k not in self._excluded_keys and k not in demo["obs"]
        )
        if not required_keys:
            return

        logger.info(f"Converting demo: {demo_name}, bridge-map obs: {required_keys}")

        for obs_key in required_keys:
            recorded_path = engine.resolve_path(obs_key)
            dataset = None

            if recorded_path:
                dataset = engine._get_nested_data(demo_data, recorded_path)

            if dataset is not None:
                # Images are recorded as uint8 by the acquisition pipeline —
                # that's a data-driven signal, not a guess from the key name.
                # Depth/scan keys are explicitly tagged via data_config's
                # `input.*.modality` field, since their dtype (uint16/
                # float32) isn't distinguishable from low_dim data by itself.
                is_image = dataset.dtype == np.uint8 or obs_key in self._modality_map
                raw = np.array(dataset) if is_image else np.array(dataset, dtype=np.float32)

                demo["obs"][obs_key] = raw
                continue

            cached = self._shape_cache.get(obs_key)
            if cached is None:
                logger.warning(
                    f"'{obs_key}' not found in {demo_name} and has no recorded "
                    f"shape anywhere in the batch; skipping."
                )
                continue
            feat_shape, dtype = cached  # a missing camera / point cloud skipped the demo in _demo_problem
            logger.warning(f"'{obs_key}' not found in {demo_name}; zero-filling with shape {feat_shape}.")
            demo["obs"][obs_key] = np.zeros((num_samples,) + feat_shape, dtype=dtype)

    def _convert_single_demo(
        self,
        demo_data: h5py.Group,
        engine: DataTransformEngine,
        demo_name: str,
    ) -> Optional[Dict]:
        """
        Convert one demo group to an in-memory dict ready for HDF5 writing.

        Numeric obs and actions come from the `streams/` group, then bridge-map obs
        (cameras / point clouds) are merged in. Only for a demo `_demo_problem` passed.
        """
        demo = self._convert_single_demo_from_streams(demo_data, demo_name)
        if demo is None:
            return None
        try:
            self._populate_bridge_obs(demo_data, engine, demo, demo["num_samples"], demo_name)
        except ValueError as e:  # one bad demo is skipped, not the rest of its file
            logger.error(f"Error converting demo '{demo_name}': {e}")
            return None
        return demo

    # ------------------------------------------------------------------
    # HDF5 write helpers
    # ------------------------------------------------------------------

    def _write_demo_to_output(
        self, output_group: h5py.Group, demo_name: str, demo: Dict
    ):
        """
        Write an in-memory demo dict to the output HDF5 group.

        Robomimic-compliance notes:
          - demo_{i}.attrs['num_samples'] = N
          - images (uint8 arrays) saved with per-frame chunking
          - everything else saved as float32 with gzip compression
          - dones kept as float32 (the validation yaml marks it as bool but
            Robomimic's actual loader checks for non-zero, so float32 is fine)
        """
        dg = output_group.create_group(demo_name)
        dg.attrs["num_samples"] = demo["num_samples"]
        dg.attrs["model_file"] = ""  # Required by some Robomimic loaders

        # Top-level datasets
        for ds_name in ("actions", "rewards", "dones"):
            arr = demo[ds_name].astype(np.float32)
            dg.create_dataset(ds_name, data=arr, **user_settings.h5_compression())

        # obs (no next_obs: robomimic only loads it with hdf5_load_next_obs, false in every shipped config)
        grp = dg.create_group("obs")
        for key, data in demo["obs"].items():
            is_image = data.dtype == np.uint8 or key in self._modality_map

            kwargs: Dict[str, Any] = {**user_settings.h5_compression(), "dtype": data.dtype}
            if is_image and data.ndim >= 3:
                # Per-frame chunking for efficient sequential reads during training
                kwargs["chunks"] = (1,) + data.shape[1:]

            grp.create_dataset(key, data=data, **kwargs)

    def _create_data_mask(self, mask_group: h5py.Group, demo_names: List[str], train: float, val: float):
        """mask/train and mask/valid; demos left over by rounding or train + val < 1 also train."""
        vlen_str_type = h5py.special_dtype(vlen=str)

        if error := _splits_error(train, val):
            self.last_error = error
            logger.error(self.last_error)
            return False

        sorted_demo_names = sorted(demo_names, key=natural_key)
        train_count, val_count = _split_counts(len(sorted_demo_names), train, val)

        if(train_count != 0):
            mask_group.create_dataset(
                "train",
                data=[n.encode("utf-8") for n in sorted_demo_names[:train_count]],
                dtype=vlen_str_type,
            )

        if(val_count != 0):
            mask_group.create_dataset(
                "valid",
                data=[n.encode("utf-8") for n in sorted_demo_names[train_count:train_count + val_count]],
                dtype=vlen_str_type,
            )

        return True

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _find_demo_groups(self, input_file: h5py.File) -> List[str]:
        """Return the names of all groups directly under /data (any name, not only 'demo_*')."""
        groups: List[str] = []
        data_grp = input_file.get("data")
        if isinstance(data_grp, h5py.Group):
            for key in data_grp.keys():
                if isinstance(data_grp[key], h5py.Group):
                    groups.append(key)
        return groups

    # ------------------------------------------------------------------
    # Real-data shape/dtype inference
    # ------------------------------------------------------------------

    def _resolve_dataset_for_key(
        self,
        demo_data: h5py.Group,
        key: str,
        engine: DataTransformEngine,
    ) -> Optional[h5py.Dataset]:
        """Return the recorded dataset for `key` in this demo: streams/ group first, then bridge map."""
        streams_grp = demo_data.get("streams")
        if isinstance(streams_grp, h5py.Group) and key in streams_grp:
            ds = streams_grp[key]
            return ds if isinstance(ds, h5py.Dataset) else None

        path = engine.resolve_path(key)
        if not path:
            return None
        return engine._get_nested_data(demo_data, path)

    def _prescan_shapes(
        self,
        input_files: List[str],
        engine: DataTransformEngine,
        required_keys: Set[str],
    ) -> Tuple[Dict[str, Tuple[Tuple[int, ...], Any]], Set[str]]:
        """
        Metadata-only pass (reads only `.shape`/`.dtype`, never full array data):
        for every key in required_keys, find the first demo across the whole
        batch where it actually resolves and record its real (feature_shape, dtype).

        Returns (cache, keys_with_no_data_anywhere_in_the_batch).
        """
        cache: Dict[str, Tuple[Tuple[int, ...], Any]] = {}
        remaining = set(required_keys)
        if not remaining:
            return cache, remaining

        for file_path in input_files:
            if not remaining:
                break
            if not file_path.endswith(".h5"):
                continue
            try:
                with h5py.File(file_path, "r") as f:
                    for demo_name in self._find_demo_groups(f):
                        if not remaining:
                            break
                        demo_data = f["data"][demo_name]
                        for key in list(remaining):
                            ds = self._resolve_dataset_for_key(demo_data, key, engine)
                            if ds is not None:
                                cache[key] = (tuple(ds.shape[1:]), ds.dtype)
                                remaining.discard(key)
            except Exception as e:
                logger.warning(f"Prescan: could not open {file_path}: {e}")
                continue

        return cache, remaining
