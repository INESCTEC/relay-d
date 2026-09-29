"""
YAML-driven dataset conversion engine
Reads conversion configs and applies transformation rules to convert datasets
to the Robomimic 1.0 format using a two-step bridge mapping strategy.

Bridge Mapping (two-step lookup):
  Step A: robomimic.yaml defines required observations (e.g. robot0_eef_pos)
  Step B: data_config.yaml `output` section maps logical keys to recorded HDF5 paths
          (e.g. robot0_eef_pos -> robot0_eef/translations)
"""

import h5py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from relay_d.utils.coloring_logger import logger
import re
import json
import os
import traceback
import yaml


def _parse_stream_entry(stream_val):
    """Return (spec_list, normalize, norm_range, limits, rotation_format).

    Accepts plain-list form (list of strings) or dict form:
      specs: [...]
      normalize: true
      normalized_range: [-1.0, 1.0]
      limits:
        - {lower_lim: ..., upper_lim: ...}   # one per column, or one entry → broadcast
      rotation_format: quat | rpy            # quat (default) or roll-pitch-yaw radians
    """
    if isinstance(stream_val, dict):
        return (
            stream_val.get("specs", []),
            stream_val.get("normalize", False),
            stream_val.get("normalized_range", [-1.0, 1.0]),
            stream_val.get("limits", None),
            str(stream_val.get("rotation_format", "quat")).lower(),
        )
    return stream_val, False, None, None, "quat"


def _apply_normalization(
    data: "np.ndarray",
    limits: List[Dict],
    norm_range: List[float],
) -> "np.ndarray":
    """Normalize each column of *data* using per-element limits.

    If *limits* has a single entry it is broadcast to every column.
    Columns whose span is zero are left unchanged.
    """
    lo, hi = norm_range
    n_cols = data.shape[1] if data.ndim == 2 else 1

    if data.ndim == 2 and n_cols == 0:
        logger.warning(
            "Normalization skipped: stream has 0 columns — data was recorded before "
            "the dict-form fix. Re-record to get proper data."
        )
        return data

    if len(limits) == 1:
        limits = limits * n_cols

    if data.ndim == 2 and len(limits) != n_cols:
        raise ValueError(
            f"_apply_normalization: {len(limits)} limits but data has {n_cols} columns. "
            f"Trim the limits list in acquisition_config.yaml to match the recorded columns "
            f"(tip: use --joints with extract_joint_limits.py to filter to only the joints "
            f"listed in joint_names)."
        )

    out = data.copy()
    for i, lim in enumerate(limits):
        lower = float(lim["lower_lim"])
        upper = float(lim["upper_lim"])
        span = upper - lower
        if span <= 0:
            continue
        if data.ndim == 2:
            out[:, i] = (data[:, i] - lower) / span * (hi - lo) + lo
        else:
            out = (data - lower) / span * (hi - lo) + lo
    return out


def _infer_spec_width(data_config: Dict, stream_name: str) -> int:
    """
    Column width for a single stream, derived purely from its configured
    specs (no recorded data needed):
      - slice spec ([:]): width = len(joint_names) for that input, or 1
      - sin/cos wrapper:  same rule applied to the inner spec
      - scalar spec:      width = 1

    Returns 0 if the stream has no configured specs at all (e.g. it isn't
    an `output.streams` entry).
    """
    streams = data_config.get("output", {}).get("streams", {})
    inputs = data_config.get("input", {})

    spec_list, *_ = _parse_stream_entry(streams.get(stream_name, []))

    width = 0
    for spec in spec_list:
        inner = spec.strip()
        if inner.startswith(("sin(", "cos(")) and inner.endswith(")"):
            inner = inner[4:-1].strip()
        if inner.endswith("[:]"):
            input_name = inner.split(".")[0]
            jnames = inputs.get(input_name, {}).get("joint_names", [])
            width += len(jnames) if jnames else 1
        else:
            width += 1
    return width


def _stream_output_width(data_config: Dict, stream_name: str) -> int:
    """
    Post-conversion column width for a stream: same as _infer_spec_width, except
    a stream with rotation_format: rpy outputs 3 columns (roll, pitch, yaw) from
    its 4 quaternion specs.
    """
    width = _infer_spec_width(data_config, stream_name)
    streams = data_config.get("output", {}).get("streams", {})
    *_, rot_fmt = _parse_stream_entry(streams.get(stream_name, []))
    if rot_fmt == "rpy":
        if width != 4:
            raise ValueError(
                f"stream '{stream_name}': rotation_format: rpy requires exactly 4 "
                f"quaternion specs (x,y,z,w), but its specs resolve to {width} columns."
            )
        return 3
    return width


def _infer_n_actions(data_config: Dict) -> int:
    """
    Derive the total action width from actions_to_extract + stream specs.
    See _stream_output_width for the per-stream width rule.
    """
    to_extract = data_config.get("actions_to_extract", [])
    total = sum(_stream_output_width(data_config, name) for name in to_extract)
    return total if total > 0 else 1


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
    from scipy.spatial.transform import Rotation

    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            f"rotation_format: rpy requires a 4-column quaternion stream (x,y,z,w); "
            f"stream '{stream_name}' has shape {tuple(data.shape)}."
        )
    return Rotation.from_quat(data).as_euler("xyz", degrees=False).astype(np.float32)


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
class DataTransformEngine:
    """
    Engine for applying transformation rules based on YAML config.

    Accepts:
      config      – parsed robomimic.yaml
      data_config – parsed data_config.yaml (the `config` sub-key)
    """

    def __init__(self, config: Dict, data_config: Optional[Dict] = None):
        self.config = config
        self.data_config = data_config or {}
        self.metadata = self.data_config.get("metadata", {})

        # Core maps built once at construction
        self.bridge_map: Dict[str, str] = self._build_bridge_map()
        self.robot_indices: List[int] = self._discover_robot_indices()
        self.required_observations: Set[str] = self._extract_required_observations()

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

    def _discover_eef_key(self) -> Optional[str]:
        """Discover the EEF position key from bridge map (e.g., robot0_eef_pos)."""
        candidates = [
            key for key in self.bridge_map if key.endswith("_pos") and "eef" in key.lower()
        ]
        if len(candidates) > 1:
            logger.warning(
                f"Multiple candidate eef keys found {candidates}; using '{candidates[0]}'."
            )
        return candidates[0] if candidates else None

    def _discover_joint_key(self) -> Optional[str]:
        """Discover the joint position key from bridge map (e.g., robot0_joint_pos)."""
        candidates = [
            key for key in self.bridge_map if key.endswith("_pos") and "joint" in key.lower()
        ]
        if len(candidates) > 1:
            logger.warning(
                f"Multiple candidate joint keys found {candidates}; using '{candidates[0]}'."
            )
        return candidates[0] if candidates else None

    def _discover_robot_indices(self) -> List[int]:
        """
        Scan bridge_map keys to discover all robot indices present.
        For new format (e.g. my_robot_joint_pos), returns [0].
        """
        indices: Set[int] = set()
        pattern = re.compile(r"robot(\d+)_")
        for key in self.bridge_map:
            m = pattern.match(key)
            if m:
                indices.add(int(m.group(1)))
        result = sorted(indices) if indices else [0]
        logger.debug(f"Discovered robot indices: {result}")
        return result

    def _extract_required_observations(self) -> Set[str]:
        """
        Collect required observation keys from the bridge map and conversion rules.
        """
        return set(self.bridge_map.keys())

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
        if logical_key in self.bridge_map:
            return self.bridge_map[logical_key]

        # Fuzzy: try stripping known suffixes to find a parent key
        for suffix in ("_pos", "_quat", "_vel", "_image"):
            if logical_key.endswith(suffix):
                stripped = logical_key[: -len(suffix)]
                if stripped in self.bridge_map:
                    return self.bridge_map[stripped]

        logger.debug(f"No bridge mapping found for logical key: '{logical_key}'")
        return None

    # ------------------------------------------------------------------
    # next_obs generation
    # ------------------------------------------------------------------

    def generate_next_obs(self, obs_data: np.ndarray) -> np.ndarray:
        """
        Shift obs forward by 1 timestep; duplicate the last frame to maintain length N.

        next_obs[t]   = obs[t+1]  for t in 0..N-2
        next_obs[N-1] = obs[N-1]  (duplicate)
        """
        if len(obs_data) <= 1:
            return obs_data.copy()
        return np.concatenate([obs_data[1:], obs_data[-1:]], axis=0)

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def generate_actions(
        self,
        input_data: h5py.Group,
        obs_dict: Dict[str, np.ndarray],
        num_samples: int,
    ) -> np.ndarray:
        """
        Build the actions array (N, n_actions) from keys in actions_to_extract.

        Each key's raw recorded value is used as-is (no per-name transform);
        normalization is applied per-column only when that stream's own
        `output.streams` entry configures `normalize`/`limits`.
        """
        actions_to_extract = self.data_config.get("actions_to_extract", [])
        if not actions_to_extract:
            n_actions = _infer_n_actions(self.data_config)
            return np.zeros((num_samples, n_actions), dtype=np.float32)

        delta_actions = self.data_config.get("delta_actions", [])
        stream_configs = self.data_config.get("output", {}).get("streams", {})
        action_parts: List[np.ndarray] = []
        any_normalized = False

        for key in actions_to_extract:
            path = self.resolve_path(key)
            if not path:
                logger.warning(
                    f"actions_to_extract key '{key}' not found in bridge_map, skipping"
                )
                continue

            data = self._get_nested_data(input_data, path)
            if data is None and "/" in path:
                field_name = path.split("/")[-1]
                data = self._scan_for_field(input_data, field_name)
                if data is not None:
                    logger.info(
                        f"actions fallback: resolved '{key}' via field scan "
                        f"('{path}' not found, matched '{field_name}')"
                    )
            if data is None:
                logger.warning(
                    f"actions_to_extract key '{key}' path '{path}' not found in data, skipping"
                )
                continue

            raw_data = np.array(data, dtype=np.float32)
            if len(raw_data) == 0:
                continue

            raw_data = self._resize_data(raw_data, num_samples)
            part = raw_data if raw_data.ndim > 1 else raw_data[:, np.newaxis]

            _, normalize, norm_range, limits, rot_fmt = _parse_stream_entry(
                stream_configs.get(key, [])
            )
            part = _apply_rotation_format(part, rot_fmt, key)
            if normalize and limits:
                part = _apply_normalization(part, limits, norm_range)
                any_normalized = True

            if key in delta_actions:
                part = _compute_delta(part)
            action_parts.append(part)

        if not action_parts:
            n_actions = _infer_n_actions(self.data_config)
            return np.zeros((num_samples, n_actions), dtype=np.float32)

        actions = np.concatenate(action_parts, axis=1)

        if self.metadata.get("normalize_actions", False) and not any_normalized:
            logger.warning(
                "normalize_actions is set but none of the actions_to_extract entries "
                "configure normalize/limits under output.streams; skipping normalization."
            )

        return actions

    # ------------------------------------------------------------------
    # Rewards / dones / states
    # ------------------------------------------------------------------

    def generate_rewards(
        self,
        input_data: h5py.Group,
        actions: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """Zero rewards except the last step which is 1."""
        rewards = np.zeros(num_samples, dtype=np.float32)
        if num_samples > 0:
            rewards[-1] = 1.0
        return rewards

    def generate_dones(self, num_samples: int) -> np.ndarray:
        """All zeros except the last element which is 1."""
        dones = np.zeros(num_samples, dtype=np.float32)
        if num_samples > 0:
            dones[-1] = 1.0
        return dones

    # ------------------------------------------------------------------
    # HDF5 nested data reader
    # ------------------------------------------------------------------

    def _scan_for_field(
        self, group: h5py.Group, field_name: str
    ) -> Optional[h5py.Dataset]:
        """
        Scan all direct sub-groups for a dataset named field_name.
        Fallback when bridge map paths don't match the recorded container names.
        """
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group) and field_name in item:
                candidate = item[field_name]
                if isinstance(candidate, h5py.Dataset):
                    logger.debug(
                        f"Fallback scan: found '{field_name}' under sub-group '{key}'"
                    )
                    return candidate
        return None

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

    def _resize_data(
        self,
        data: np.ndarray,
        target_samples: int,
        preserve_dtype: bool = False,
    ) -> np.ndarray:
        """
        Resample data along axis-0 to target_samples using linear index interpolation.
        Preserves dtype when preserve_dtype=True (important for uint8 images).
        """
        if len(data) == target_samples:
            return data

        if len(data) == 0:
            return np.zeros((target_samples,) + data.shape[1:], dtype=data.dtype)

        indices = np.linspace(0, len(data) - 1, target_samples).astype(int)
        resampled = data[indices]
        return resampled if preserve_dtype else resampled.astype(np.float32)


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
        # Bridge-map keys with zero data anywhere in the current batch and no
        # way to infer a shape for them — excluded from output entirely.
        self._excluded_keys: Set[str] = set()
        # Per-batch: logical_key -> "rgb"|"depth"|"scan", from data_config's
        # `input` section's `modality` field. Only keys with an EXPLICIT
        # modality are present here — plain rgb (the common case) is instead
        # detected from the real recorded dtype (uint8).
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

    def convert_dataset(
        self,
        format_name: str,
        input_h5_path: str,
        output_h5_path: str,
        env_metadata: Optional[Dict] = None,
        data_config: Optional[Dict] = None,
    ) -> bool:
        """Convert a single input HDF5 file."""
        return self.convert_multiple_files(
            format_name,
            [input_h5_path],
            output_h5_path,
            env_metadata=env_metadata,
            data_config=data_config,
        )

    def convert_multiple_files(
        self,
        format_name: str,
        input_files: List[str],
        output_path: str,
        env_metadata: Optional[Dict] = None,
        data_config: Optional[Dict] = None,
    ) -> bool:
        """
        Convert and merge multiple input HDF5 files into one Robomimic-format output.

        Args:
            format_name:  Target format key (e.g. "robomimic")
            input_files:  List of input HDF5 file paths
            output_path:  Destination HDF5 path
            env_metadata: Optional metadata override dict
            data_config:  Optional data_config override (the `config` sub-dict)

        Returns:
            True on success.
        """
        fmt_config = self.config_loader.get_config(format_name)
        if not fmt_config:
            logger.error(f"Format config not found: '{format_name}'")
            return False

        if data_config:
            self.data_config = data_config

        # Metadata: prefer explicit arg, then data_config, then empty
        meta = env_metadata if env_metadata else self.data_config.get("metadata", {})
        env_args = self._build_env_args(meta)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def _natural_key(path: str) -> List:
            parts = re.split(r"(\d+)", os.path.basename(path))
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        sorted_files = sorted(input_files, key=_natural_key)

        # bridge_map/required_observations depend only on data_config, which is
        # identical for every demo in this batch — build the engine once.
        engine = DataTransformEngine(fmt_config, self.data_config)
        self._modality_map = self._build_modality_map()

        # Real-data shape/dtype cache: for every observation the config asks
        # for, find the first demo in the whole batch where it actually
        # resolves and record its real (feature_shape, dtype). Used to
        # zero-fill demos missing that key instead of guessing from its name.
        streams_cfg = self.data_config.get("output", {}).get("streams", {})
        required_keys = set(streams_cfg.keys()) | set(engine.required_observations)
        self._shape_cache, unresolved = self._prescan_shapes(
            sorted_files, engine, required_keys
        )
        self._excluded_keys = set()
        for key in sorted(unresolved):
            if key in streams_cfg:
                width = _stream_output_width(self.data_config, key) or 1
                self._shape_cache[key] = ((width,), np.float32)
                logger.warning(
                    f"Stream '{key}' has no recorded data anywhere in this batch; "
                    f"zero-filling every demo using its configured spec width ({width})."
                )
            else:
                self._excluded_keys.add(key)
                logger.error(
                    f"Observation '{key}' has no recorded data anywhere in this batch "
                    f"and no inferable width; excluding it from the converted dataset."
                )

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

                                out_name = f"demo_{counter}"
                                converted = self._convert_single_demo(
                                    demo_data, engine, out_name
                                )

                                if converted:
                                    self._write_demo_to_output(
                                        data_grp, out_name, converted
                                    )
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

                    except Exception as e:
                        fname = os.path.basename(file_path)
                        logger.error(f"Skipping {fname} — could not open: {e}")
                        skipped.append(fname)
                        continue

                # Robomimic requires `total` = number of demos (not total timesteps)
                data_grp.attrs["total"] = counter
                # env_args MUST be a JSON string (Robomimic ObsUtils requirement)
                data_grp.attrs["env_args"] = json.dumps(env_args)

                mask_grp = out_f.create_group("mask")
                train_frac = self.data_config.get("train_frac", 0.8)
                val_frac = self.data_config.get("val_frac", 0.2)
                test_frac = self.data_config.get("test_frac", 0.0)
                if(not self._create_data_mask(mask_grp, processed, train=train_frac, val=val_frac, test=test_frac)):
                    return False

            if skipped:
                logger.warning(
                    f"Skipped {len(skipped)} file(s) (corrupted or unreadable): {skipped}"
                )
            logger.info(
                f"Successfully wrote {counter} demos ({total_demos} timesteps) -> {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Fatal error during conversion: {e}")
            traceback.print_exc()
            return False

    def convert_raw_format(self, input_files: List[str], output_path: str) -> bool:
        """
        Pass-through copy: merge multiple HDF5 files preserving internal structure.
        No format translation is applied.
        """
        def _natural_key(path: str) -> List:
            parts = re.split(r"(\d+)", os.path.basename(path))
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        try:
            with h5py.File(output_path, "w") as out_f:
                data_grp = out_f.create_group("data")
                total = 0
                for file_path in sorted(input_files, key=_natural_key):
                    with h5py.File(file_path, "r") as in_f:
                        in_data = in_f.get("data")
                        if not isinstance(in_data, h5py.Group):
                            continue
                        for key in sorted(in_data.keys()):
                            item = in_data[key]
                            if isinstance(item, h5py.Group):
                                self._copy_group_preserving_data(
                                    item, data_grp.create_group(f"demo_{total}")
                                )
                                total += 1

                data_grp.attrs["total"] = total
                mask_grp = out_f.create_group("mask")
                self._create_valid_mask(mask_grp, [f"demo_{i}" for i in range(total)])
            return True
        except Exception as e:
            logger.error(f"Error in raw conversion: {e}")
            traceback.print_exc()
            return False

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
        maps them directly to obs/next_obs. Actions are assembled from
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
                "next_obs": {},
                "actions": None,
                "rewards": None,
                "dones": None,
                "states": None,
            }

            stream_configs = self.data_config.get("output", {}).get("streams", {})
            for stream_name in stream_names:
                _, normalize, norm_range, limits, rot_fmt = _parse_stream_entry(
                    stream_configs.get(stream_name, [])
                )
                if stream_name in streams_grp:
                    dataset = streams_grp[stream_name]
                    # Defensive guard: a stream should never legitimately carry
                    # uint8 image data (output.streams is numeric-only — see
                    # StreamBuilder), but if a misconfigured spec ever put one
                    # here, preserve its dtype instead of corrupting it with an
                    # unconditional float32 cast + rotation/normalization math.
                    is_image = dataset.dtype == np.uint8
                    data = np.array(dataset) if is_image else np.array(dataset, dtype=np.float32)
                    data = data[:num_samples]
                    if not is_image:
                        data = _apply_rotation_format(data, rot_fmt, stream_name)
                        if normalize and limits:
                            data = _apply_normalization(data, limits, norm_range)
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
                demo["next_obs"][stream_name] = np.concatenate(
                    [data[1:], data[-1:]], axis=0
                )

            actions_to_extract = self.data_config.get("actions_to_extract", [])
            delta_actions = self.data_config.get("delta_actions", [])
            action_parts = []
            for stream_name in actions_to_extract:
                if stream_name in streams_grp:
                    part = np.array(streams_grp[stream_name], dtype=np.float32)
                    if part.ndim == 1:
                        part = part[:, np.newaxis]
                    part = part[:num_samples]
                    _, normalize, norm_range, limits, rot_fmt = _parse_stream_entry(
                        stream_configs.get(stream_name, [])
                    )
                    part = _apply_rotation_format(part, rot_fmt, stream_name)
                    if normalize and limits:
                        part = _apply_normalization(part, limits, norm_range)
                    if stream_name in delta_actions:
                        part = _compute_delta(part)
                    action_parts.append(part)
                elif stream_name in demo["obs"]:
                    # Already zero-filled above from the shape cache.
                    part = demo["obs"][stream_name]
                    if part.ndim == 1:
                        part = part[:, np.newaxis]
                    action_parts.append(part)
                else:
                    logger.warning(
                        f"actions_to_extract: stream '{stream_name}' not found in {demo_name} "
                        f"(no data anywhere in the batch)"
                    )

            action_parts = [p for p in action_parts if p.ndim < 2 or p.shape[1] > 0]
            if action_parts:
                demo["actions"] = np.concatenate(action_parts, axis=1)
            else:
                n_actions = _infer_n_actions(self.data_config)
                demo["actions"] = np.zeros((num_samples, n_actions), dtype=np.float32)

            demo["rewards"] = np.zeros(num_samples, dtype=np.float32)
            if num_samples > 0:
                demo["rewards"][-1] = 1.0
            demo["dones"] = np.zeros(num_samples, dtype=np.float32)
            if num_samples > 0:
                demo["dones"][-1] = 1.0

            # Build states by concatenating every float (N, M) obs array in YAML stream order.
            # Image arrays (uint8) and 1-D arrays are skipped.
            state_parts = [
                v for v in demo["obs"].values()
                if isinstance(v, np.ndarray) and v.dtype != np.uint8 and v.ndim == 2
            ]
            demo["states"] = (
                np.concatenate(state_parts, axis=1)
                if state_parts
                else np.zeros((num_samples, 1), dtype=np.float32)
            )

            logger.info(
                f"Stream-based conversion: {demo_name}, "
                f"N={num_samples}, obs={list(demo['obs'].keys())}, "
                f"actions shape={demo['actions'].shape}"
            )
            return demo

        except Exception as e:
            logger.error(f"Error in stream-based conversion of '{demo_name}': {e}")
            import traceback
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
        Populate demo['obs']/['next_obs'] for every bridge-map key (top-level
        `output:` entries other than `streams`) not already present in demo['obs'].

        This is how Image / organized PointCloud2 data reaches obs/next_obs:
        those types can't be expressed as output.streams specs (numeric-only —
        see StreamBuilder), so they're referenced directly by their raw HDF5
        path instead, e.g. `output: agentview_image: camera_rgb/images`. Called
        from both branches of `_convert_single_demo` so bridge-map obs can
        coexist with (or stand in for) output.streams-based obs.
        """
        required_keys = sorted(
            k for k in engine.required_observations
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

                # Fallback: scan sub-groups for the field name when the config
                # path doesn't match the recorded container name.
                if dataset is None and "/" in recorded_path:
                    field_name = recorded_path.split("/")[-1]
                    dataset = engine._scan_for_field(demo_data, field_name)
                    if dataset is not None:
                        logger.info(
                            f"Fallback: resolved '{obs_key}' via field scan "
                            f"('{recorded_path}' not found, matched '{field_name}')"
                        )

            if dataset is not None:
                # Images are recorded as uint8 by the acquisition pipeline —
                # that's a data-driven signal, not a guess from the key name.
                # Depth/scan keys are explicitly tagged via data_config's
                # `input.*.modality` field, since their dtype (uint16/
                # float32) isn't distinguishable from low_dim data by itself.
                is_image = dataset.dtype == np.uint8 or obs_key in self._modality_map
                raw = np.array(dataset) if is_image else np.array(dataset, dtype=np.float32)

                if len(raw) != num_samples:
                    raw = engine._resize_data(raw, num_samples, preserve_dtype=is_image)

                demo["obs"][obs_key] = raw
                demo["next_obs"][obs_key] = engine.generate_next_obs(raw)
                continue

            cached = self._shape_cache.get(obs_key)
            if cached is None:
                logger.warning(
                    f"'{obs_key}' not found in {demo_name} and has no recorded "
                    f"shape anywhere in the batch; skipping."
                )
                continue
            feat_shape, dtype = cached
            fill = np.zeros((num_samples,) + feat_shape, dtype=dtype)
            demo["obs"][obs_key] = fill
            demo["next_obs"][obs_key] = fill.copy()

    def _convert_single_demo(
        self,
        demo_data: h5py.Group,
        engine: DataTransformEngine,
        demo_name: str,
    ) -> Optional[Dict]:
        """
        Convert one demo group to an in-memory dict ready for HDF5 writing.

        Steps:
          1. If a `streams/` group is present, start from stream-based
             conversion (numeric obs from output.streams).
          2. Always merge in bridge-map obs (top-level `output:` entries other
             than `streams`) — this is how Image / organized PointCloud2 data
             reaches obs/next_obs, and it coexists with output.streams rather
             than being mutually exclusive with it.
          3. If there was no `streams/` group at all, determine master
             timestep count N via the bridge map instead, then generate
             actions, rewards, dones, states as usual.
        """
        if "streams" in demo_data:
            demo = self._convert_single_demo_from_streams(demo_data, demo_name)
            if demo is None:
                return None
            self._populate_bridge_obs(
                demo_data, engine, demo, demo["num_samples"], demo_name
            )
            return demo

        try:
            num_samples = self._get_master_timesteps(demo_data, engine)
            if num_samples is None or num_samples == 0:
                logger.warning(f"Could not determine timesteps for {demo_name}")
                return None

            demo: Dict[str, Any] = {
                "num_samples": num_samples,
                "obs": {},
                "next_obs": {},
                "actions": None,
                "rewards": None,
                "dones": None,
                "states": None,
            }

            self._populate_bridge_obs(demo_data, engine, demo, num_samples, demo_name)

            # --- Actions, rewards, dones, states ---
            demo["actions"] = engine.generate_actions(
                demo_data, demo["obs"], num_samples
            )
            demo["rewards"] = engine.generate_rewards(
                demo_data, demo["actions"], num_samples
            )
            demo["dones"] = engine.generate_dones(num_samples)
            state_parts = [
                v for v in demo["obs"].values()
                if isinstance(v, np.ndarray) and v.dtype != np.uint8 and v.ndim == 2
            ]
            demo["states"] = (
                np.concatenate(state_parts, axis=1)
                if state_parts
                else np.zeros((num_samples, 1), dtype=np.float32)
            )

            return demo

        except Exception as e:
            logger.error(f"Error converting demo '{demo_name}': {e}")
            traceback.print_exc()
            return None

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
        for ds_name in ("states", "actions", "rewards", "dones"):
            arr = demo[ds_name].astype(np.float32)
            dg.create_dataset(ds_name, data=arr, compression="gzip")

        # obs and next_obs
        for group_name in ("obs", "next_obs"):
            grp = dg.create_group(group_name)
            for key, data in demo[group_name].items():
                is_image = data.dtype == np.uint8 or key in self._modality_map

                kwargs: Dict[str, Any] = {"compression": "gzip", "dtype": data.dtype}
                if is_image and data.ndim >= 3:
                    # Per-frame chunking for efficient sequential reads during training
                    kwargs["chunks"] = (1,) + data.shape[1:]

                grp.create_dataset(key, data=data, **kwargs)

    def _create_data_mask(self, mask_group: h5py.Group, demo_names: List[str], train: float, val: float, test: float):
        vlen_str_type = h5py.special_dtype(vlen=str)

        sorted_demo_names = sorted(demo_names)
        total_demos = len(sorted_demo_names)

        train_count = int(total_demos * train)
        val_count = int(total_demos * val)
        test_count = total_demos - train_count - val_count

        if(train_count + val_count + test_count != total_demos):
            logger.warning(f"Train/Val/Test split does not sum to total demos. Adjusting test count.")
            return False

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

        if(test_count != 0):
            mask_group.create_dataset(
                "eval",
                data=[n.encode("utf-8") for n in sorted_demo_names[train_count + val_count:]],
                dtype=vlen_str_type,
            )

        return True

    def _copy_group_preserving_data(self, source: h5py.Group, dest: h5py.Group):
        for k, v in source.attrs.items():
            dest.attrs[k] = v
        for key in source.keys():
            item = source[key]
            if isinstance(item, h5py.Dataset):
                ds = dest.create_dataset(key, data=item[:], dtype=item.dtype)
                for ak, av in item.attrs.items():
                    ds.attrs[ak] = av
            elif isinstance(item, h5py.Group):
                self._copy_group_preserving_data(item, dest.create_group(key))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _find_demo_groups(self, input_file: h5py.File) -> List[str]:
        """Return all group names directly under /data that start with 'demo_'."""
        groups: List[str] = []
        data_grp = input_file.get("data")
        if isinstance(data_grp, h5py.Group):
            for key in data_grp.keys():
                if isinstance(data_grp[key], h5py.Group):
                    groups.append(key)
        return groups

    def _get_master_timesteps(
        self,
        demo_data: h5py.Group,
        engine: DataTransformEngine,
    ) -> Optional[int]:
        """
        Determine N (number of timesteps) for a demo group.

        Priority:
          1. Explicit 'timestamps' dataset at demo root
          2. detailed_timestamps/recording_times
          3. Primary observation dataset via bridge map (joint_positions)
          4. Any first dataset found by scanning sub-groups
        """
        # 1. timestamps
        if "timestamps" in demo_data:
            ts = demo_data["timestamps"]
            if isinstance(ts, h5py.Dataset) and ts.shape[0] > 0:
                return ts.shape[0]

        # 2. detailed_timestamps
        dt = demo_data.get("detailed_timestamps")
        if isinstance(dt, h5py.Group):
            rt = dt.get("recording_times")
            if isinstance(rt, h5py.Dataset) and rt.shape[0] > 0:
                return rt.shape[0]

        # 3. Primary obs from bridge map (dynamically discovered)
        joint_key = engine._discover_joint_key()
        eef_key = engine._discover_eef_key()
        for primary_key in (joint_key, eef_key):
            if not primary_key:
                continue
            path = engine.resolve_path(primary_key)
            if path:
                ds = engine._get_nested_data(demo_data, path)
                if isinstance(ds, h5py.Dataset) and ds.shape[0] > 0:
                    return ds.shape[0]

        # 4. Scan any dataset in any sub-group
        for key in demo_data.keys():
            item = demo_data[key]
            if isinstance(item, h5py.Group):
                for sub_key in item.keys():
                    sub = item[sub_key]
                    if (
                        isinstance(sub, h5py.Dataset)
                        and len(sub.shape) >= 1
                        and sub.shape[0] > 0
                    ):
                        return sub.shape[0]
            elif isinstance(item, h5py.Dataset) and item.shape[0] > 0:
                return item.shape[0]

        return None

    def _build_env_args(self, metadata: Dict) -> Dict:
        """
        Build the env_args dictionary that will be JSON-serialised into
        /data.attrs['env_args'].

        Structure required by Robomimic ObsUtils:
          {
            "env_name": str,
            "type": int,           (Robomimic env-family enum, e.g. 1 = robosuite)
            "env_kwargs": { ... }
          }
        """
        return {
            "env_name": metadata.get("env_name", "LFD_Demo_Environment"),
            "env_version": metadata.get("env_version", ""),
            "type": metadata.get("type", 1),
            "env_kwargs": metadata.get("env_kwargs", {}),
        }

    def _build_modality_map(self) -> Dict[str, str]:
        """
        Build logical_key -> "rgb"|"depth"|"scan" from data_config's `input`
        section's `modality` field. Only inputs with an explicit modality are
        included — implicit rgb (the common case) is instead detected from
        the real recorded dtype (uint8) wherever it's checked.
        """
        modality_map: Dict[str, str] = {}
        for input_name, spec in self.data_config.get("input", {}).items():
            if not isinstance(spec, dict):
                continue
            modality = spec.get("modality")
            if modality in ("rgb", "depth", "scan"):
                key = spec.get("output_map") or input_name
                modality_map[key] = modality
        return modality_map

    def get_last_shape_cache(self) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
        """Real (feature_shape, dtype) per key from the last convert_multiple_files call."""
        return dict(self._shape_cache)

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
        ds = engine._get_nested_data(demo_data, path)
        if ds is None and "/" in path:
            ds = engine._scan_for_field(demo_data, path.split("/")[-1])
        return ds

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
