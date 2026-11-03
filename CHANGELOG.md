# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrade notes (read first)
- **Re-convert your datasets and retrain.** An absolute action is now the target of the next step (`action[t] = value[t+1]`), see *Changed*. Datasets converted before this release taught the policy to hold still.
- **LeRobot environment:** press **Set up lerobot environment** (Train tab) again: it adds `lerobot[smolvla]`. The `.[lerobot]` extra now needs `pyarrow<21` (pyarrow 21+ needs NumPy 2, ROS Jazzy has 1.26).
- **`relay_d.dispatch.DispatcherHelpers` is gone:** use `save_dispatch_templates(output_dir)`.
- **`lfd-inference-node --hz`** defaults to the rate the LeRobot policy was trained at (its dataset fps), no longer the `infer_hz` setting (10 Hz). robomimic checkpoints still use the setting: pass `--hz <recording_frequency>`.
- **Action configs copied from the template:** the `robotiq_gripper` entry must map `position: 0`, not `7` (see *Fixed*).
- **New folder names:** recording sessions end in `_Recording`, Post-process outputs in `_Robomimic` / `_Lerobot`. Existing folders keep their names.

### Added
- **SmolVLA**, a third LeRobot policy (Train tab, `relay-d-train --policy smolvla`). It is a 450M vision-language-action model fine-tuned from the pretrained `lerobot/smolvla_base` (`--policy.path`, downloaded once), not trained from scratch,
  and it follows the dataset's task sentence. The dataset's cameras (at most 3) are renamed to the model's `camera1..3` with `--rename_map`, which LeRobot keeps in the checkpoint. "Set up lerobot environment" installs
  `lerobot[smolvla]` (`transformers`, `num2words`). Measured on an RTX 4060 Laptop (8 GB): the fine-tune peaks at about 4.2 GB with the default batch size, and inference needs under 1 GB.
- **Task sentence at inference:** `lfd-inference-node --task "..."` / `InferenceSession(task=...)`, by default the training dataset's task (`meta/tasks.parquet`). A language policy without a task stops at startup. For a fine-tuned checkpoint
  the state size and cameras are taken from the training dataset, because the checkpoint config keeps the pretrained model's ones.
- **Safe stop:** a command or topic entry of `action_config.yaml` can set `stop: [...]` (e.g. `[0, 0, 0]` for a velocity), sent once when inference stops sending: inputs go stale, the model fails, `on_timeout: raise`, Ctrl+C, the end of
  `run()`, or `close()`. Before, the node just went quiet and a controller without a watchdog kept the last command. A broken `stop` (wrong length, a mapping that does not fit, set on an `action: true` entry) stops startup.
- **Recorded-rate check:** a demo recorded more than 10% off `recording_frequency` (a PC that could not keep up) fails the after-recording check and is left out of LeRobot datasets, whose fps time base would make the policy move
  faster than shown.
- **CI runs the ROS-dependent tests** in a `ros:jazzy` container job (dispatcher, recorder, `AppAPI`, inference session). `pytest` also runs locally with ROS sourced (`pyproject.toml` disables the ROS pytest plugins).
- Verified end to end against LeRobot 0.6.1: `LeRobotDataset` loads a Relay-D dataset (frames, RGB video, `action[t] = state[t+1]`, ACT-style action chunks via `delta_timestamps`), `lerobot_train` trains on it with the Train tab's
  command (ACT and SmolVLA), and `LeRobotPolicyRunner` runs the checkpoint, with state and action normalized by the dataset's own stats.

### Changed
- **Training actions are next-step targets:** an absolute action (a stream not in `delta_actions`) is the stream's value at t+1, as a delta already looked one step ahead; it was the value at t, so a state stream used as an action
  taught the policy to stand still. A stream that is both `normalize: true` and in `delta_actions` is refused (conversion and the config check): the dispatcher cannot turn a normalized delta back into real units.
- **LeRobot policies run at their training rate:** `--hz` defaults to the dataset's fps (read through the checkpoint), and `InferenceSession.run()` warns when the rate differs (a 30 fps policy at 10 Hz played its actions 3x slower).
- **Output names:** a recording session folder ends in `_Recording` (after the `session_folder_format` pattern). A Post-process output folder and the dataset inside it end in `_Robomimic` or `_Lerobot`
  (`2026-11-02_15-10-00_Robomimic/dataset_<time>_Robomimic.h5`, `2026-11-02_15-12-00_Lerobot/dataset_<time>_Lerobot/`). `AppAPI` writes `combined_dataset_Robomimic.h5` / `combined_dataset_Lerobot` into the session folder, and Review
  leaves `*_Robomimic.h5` out of the demo list. Training finds datasets by their content, as before.
- **Record page Stop** saves in a background thread: the window no longer freezes while a large demo is written, the buttons are locked until it is done, and closing the window waits for that save. Demo names carry milliseconds,
  so two Starts in one second no longer clash.
- The exported `inference_script.py` is a thin wrapper around `lfd-inference-node` (same options and defaults; only `--obs-config` / `--action-config` default to the files next to it).
- `relay_d.dispatch.save_dispatch_templates(output_dir)` replaces `DispatcherHelpers`; the action dispatcher walks field paths in one place (`_walk` / `_assign`).
- The app logo is a small PNG (11 KB instead of 3.9 MB), so `resources_rc.py` shrinks by about 250k lines; duplicate icon entries are removed from `resources.qrc`.

### Fixed
- **Conversion no longer guesses:** a configured path that is missing is not replaced by the first group with the same field name (an arm path could read the gripper), and a dataset whose length differs from the demo's timestamps
  fails that demo instead of being stretched. The train/validation masks sort demos by number (`demo_2` before `demo_10`).
- **`AppAPI.shutdown()`** keeps everything running and returns False when the last recording could not be saved (it was only in memory and got lost); True otherwise. Closing the GUI with an unsaved demo retries the save and asks
  before losing it.
- **Action dispatcher:** a `scale` / `strength_scale` with `in_min == in_max` stops startup (the magnetic gripper crashed the node the first time it switched on); an ack that arrives while a stale one is being read is no longer lost.
- **Action template:** `robotiq_gripper` mapped `position: 7`, but mapping values index the entry's slice (`indices: [7]` has only index 0), so the copied template failed its startup check. It is now `position: 0`.
- **`.[lerobot]` extra:** pyarrow is capped below 21: pyarrow 21+ needs NumPy 2 at import and ROS Jazzy's Python has NumPy 1.26, so LeRobot conversion failed on a standard ROS machine.
- **Joint states:** a message without every configured joint (e.g. a gripper-only publisher on the same topic) is ignored instead of shifting the values (`ObsBuilder`, recording streams), and an array the driver does not fill for
  every joint (often `velocity`) reads as zeros/empty instead of raising or being misread. Joint datasets keep one row per timestamp, an empty first row is zero-padded instead of failing the save.
- **Observations:** an obs whose `limits` count matches neither 1 nor its width is left unset (pre-flight fails) instead of reusing the last limit.
- **Recording:** a new demo never overwrites an existing file (the demo counter restarts in every process), and a demo saved by a retry at Start joins the session; a YAML date in the config metadata no longer drops the metadata
  keys after it; pressing Start again after a failed start no longer subscribes twice; unorganized point clouds are stored as metadata only instead of being decoded into the RAM buffer.
- **Environment setup:** the folder name in the setup script is shell-quoted correctly.

### Removed
- `relay_d.dispatch.DispatcherHelpers` (replaced by `save_dispatch_templates`).
- The converter's field-scan fallback (`_scan_for_field`) and the resampling of datasets of the wrong length (`_resize_data`): both hid broken data (see *Fixed*).

### Fixed (audit)
- **Action dispatcher:** a `mapping` that does not fit its message (a wrong field name or index) now stops startup; before, the first tick sent the other entries (the arm) and skipped that one (the gripper). An output that is not finite
  after its `transform` (a `scale` with `in_min == in_max`) skips the whole tick instead of being published.
- **Environment setup:** a build that is still running is recognised by its process (`<env>.setup.pid`), not by how recently its log grew (pip prints nothing into a pipe, so a long download looked "failed" after 15 minutes and a second
  build could start in the same folder). `lerobot` is pinned to the release the launcher's module path and flags were written for (`envsetup.LEROBOT_VERSION`).

### Changed (audit)
- `DataRecorder`'s message-to-sample encoding moved to `ui/record/message_encoding.py`; the ROS-free helpers of `ObsBuilder` (`inputs_in_sync`, `quat_to_rot_matrix`, `strip_trig`, `read_scalar`) moved to `utils/numerics.py` and are tested;
  `launcher.run_script` is the one place that starts a script in a terminal (training and environment setup).

### Changed (pages)
- **Record page without a config:** opening the Record page (sidebar button, or "Page shown at start") with no config loaded shows "Load a config before recording ..." and opens the Configuration page instead; the empty Record page itself
  says what to do, and pressing Start without a config now explains why nothing happens (it only logged before).
- **One look for every page** (`ui/page_style.py` + both themes): the same margins (24/20), a bold page title, a thin separator and the content below, on Record, Review, Train, Settings and Configuration (its title now sits above both tabs);
  tabs and group boxes are styled the same everywhere (they were unstyled), the Record page uses the same background as the others, and the dark theme's Settings form is readable (its scroll area had a light system background).

### Fixed (TF warning, settings, first start)
- **`TF_OLD_DATA ignoring data from the past` spam:** the TF listeners (Record page, dispatch `ObsBuilder`, custom TF topics) drop a transform
  that tf2 would reject anyway (older than the 10 s cache behind the newest of its frame) before tf2 prints a warning, and log one summary
  line naming the frames. The cause is another publisher on `/tf` with old stamps (a bag replayed without `--remap /tf:=/playback/tf`,
  a machine with a different clock): `ros2 topic info -v /tf` lists the publishers. rviz2 / robot_state_publisher print their own lines.
- **Settings apply to what is already running:** `AppAPI` no longer freezes the recordings folder at construction; the recorder re-reads
  `rec_default_*` at every start; `ConfigManager.get` reads the same store as the command-line tools; the Settings page shows values saved
  by other pages (Train's Python, Review's last URDF, ...) and no longer writes the old ones back; Review's fixed frame follows its setting
  unless it was edited.
- **Environments for training and running policies:** the Train tab builds a separate environment per format (`~/.relayd/envs/lerobot` and `~/.relayd/envs/robomimic`, Python 3.12, uv if available;
  robomimic from a pinned source commit, since PyPI stops at 0.3.0) and the button just *uses* an environment that is already built (the setup script is safe to re-run: it used to fail on an existing environment and mark
  it "failed"). The interpreter rule is the same everywhere: the `python_<format>` setting, else the built environment, else the running Python. The Train tab no longer saves its untouched default Python as a setting (that pinned
  it and hid environments built later). **`lfd-inference-node` and the generated inference script** start again in that environment when the backend's package is missing in the Python that launched them
  (ROS python packages and Relay-D passed on; guard `RELAYD_REEXEC`).
- **Robomimic `StopIteration` crash on small datasets:** robomimic only trains on complete batches (`drop_last=True`), so a dataset with fewer training frames than the batch size (default 100) makes no batch and crashes in `run_epoch`.
  The Train tab now caps a batch size nobody chose to the number of training frames, refuses an explicit one that cannot work with a clear message, and shows "N training frames: M batches per epoch". Also, robomimic catches its own crash
  and exits 0, so such a run showed as *finished*: a log containing "run failed with error" is now recorded as a failure.
- **Robomimic training without a validation split** no longer fails (`train.hdf5_validation_filter_key` is cleared when validation is turned off).
- **Train tab, LeRobot:** "'lerobot' is not installed for <python>" now says what to do, and a new **Set up LeRobot environment** button creates a separate environment
  (`~/.relayd/envs/lerobot`, with uv if available; `lerobot[training]` + `lerobot[diffusion]`) in a terminal window and uses it for LeRobot training (GUI and `relay-d-train`)
  unless "Python for LeRobot" is set. Other Python environments are not touched.
- **Settings Save button** raised `TypeError: 'bool' object is not iterable` (the `clicked` flag was taken as the list of keys to save).
- **Recordings folder:** the recorder reads the setting when a demo starts and opens a new session folder when it changed (it used a cached copy before).
- **First start:** if no recordings folder (`hdf5_save_folder`) is set, Relay-D asks for one before anything else can be used (cancelling asks to quit).

### Added (config builder)
- **Config page > "Build from ROS"**: build `acquisition_config.yaml` from the running ROS 2 graph, GUI only. Pick topics from the live
  graph (or type one), pick two TF frames, name the inputs, tick message fields to build `output.streams` (joint, quaternion, rpy,
  arithmetic specs), add cameras / point clouds, choose the action streams (order, delta), fill normalization limits by hand or from
  the URDF (file or the latched `/robot_description`), set the recording options, review the YAML and the problems, then **Save and use now**.
  "Edit loaded config" opens an existing file in the builder (unknown keys such as `metadata` are kept).
- A shared config check (`builder/config_check.py`): undefined inputs in a spec, duplicate `output_map`, limits count vs stream width,
  `rpy` needing 4 specs, unknown action names, and more. Errors block saving in the builder; loading a file on the Config page shows
  the same findings as warnings only (existing configs keep loading). The API and `load_offline_config` are unchanged: they still take a ready-made file.
- `parse_joint_limits` moved to `relay_d/.../config/builder/urdf_limits.py` (stdlib only); `scripts/extract_joint_limits.py` uses it
  (no more `xmltodict`, but it now needs `relay_d` importable). `run_in_background` moved to `qt_app/utils/background.py`.

### Fixed (audit)
- **Dispatch:** a message with a mapped field that cannot be set is no longer published; int/bool fields
  are cast correctly; a command whose send failed is retried; all outputs are computed before any is
  sent; `step()` has the same stale-input check as `run()`; `close()` stops started shell commands.
- **Inference:** stale check also covers held observations (`ObsBuilder.get_obs_ages()`); static TF
  inputs count as fresh; observations are evaluated once per `get_obs()`; the checkpoint is loaded once.
- **Subscriptions** match a best-effort publisher's QoS (`relay_d/utils/qos.py`).
- **Recording:** samples before a sensor's first message are padded, not dropped; a failed save is
  reported (and the GUI can retry it); stop waits for the sampling loop; the loop backs off on
  repeated errors; the GUI warns when the buffer is full.
- **Conversion:** a missing action key fails the demo; split fractions are validated; failed runs
  delete their half-written output; training and inference share one normalization rule
  (`relay_d/utils/numerics.py`).

### Changed (inference backends)
- `lfd-inference-node` / `InferenceSession` run exactly two backends: **robomimic** (`.pth`) and the new **LeRobot** one
  (`relay_d/dispatch/lerobot_runner.py`: ACT, Diffusion, ... from a `pretrained_model` folder). `--backend auto` (the default, and
  the new default of the `infer_backend` setting) picks it from the checkpoint. The LeRobot runner feeds `observation.state` in the
  order of the training dataset's `meta/info.json` and the cameras as RGB CHW floats, and matches LeRobot's own `select_action`
  exactly (checked for ACT and Diffusion on checkpoints trained from a Relay-D dataset). New `--state-keys` option.
- Post-process writes the `action_dispatcher/` files (obs_config, action template, inference script) for LeRobot datasets too.
- Removed: the custom backend (`model_runner.py`: BC-MLP / RNN / GMM, Diffusion, TorchScript), `--model-type`, `--obs-keys`,
  `--action-horizon`, the `infer_backend` value `custom` (an old saved one falls back to `auto`), the `dispatch` extra, and
  `debug/equivalence_test.py` + `debug/replay_parity.py` (they compared the custom backend with robomimic).

### Changed (simplification, no behaviour change)
- Removed dead code: debug dumps, unused pass-through methods and widget guards in the Record page, unused accessors,
  guards for widgets that always exist (Post-process, Review, recording buttons), duplicate constants
  (`NON_INPUT_GROUPS`, `TF_REMAP_ARGS`), the unused `sample` option of the GMM policy and `DemoValidator(thresholds=)`,
  the converter's repeated rewards / dones / next_obs / states code, unread YAML keys, and the unused imports.
- The two dispatch config templates lose their repeated and commented-out copies of entries (the live entries are
  unchanged) and the comments that contradicted the code. Module docstrings that repeated the templates are shorter.
- Tests: shared fixtures and helpers live in `tests/conftest.py` and `tests/helpers.py`.
- Fixed on the way: an empty "Recordings folder" setting made Post-process write into the current directory;
  `inference_script_template.py` ignored its `--hz` and `--obs-wait` options.

### Fixed (Ctrl+C)
- Ctrl+C in a training window ends the run and closes the window at once (it waited for Enter before); the run shows
  as "stopped". Ctrl+C in the terminal that started `RelayDApp` closes the app through its normal cleanup.

### Fixed (review)
- A zero-length quaternion in a `rotation_format: rpy` stream stops the conversion again (it had become NaN in the
  dataset when scipy was replaced by `relay_d.utils.numerics`).
- Train tab: the job is built when you click Start (and the inputs are locked while it is checked), and the run is
  started in the background. "Reset this tab" in Settings no longer saves or discards edits on other tabs; the robot
  description of the last playback is remembered. CI installs Qt and OpenCV so the page and viewer tests run.

### Fixed (audit 2)
- **Recording:** point clouds now count towards `max_buffer_mb` (they were never counted, so a scan recording could
  fill the RAM); images are written frame by frame instead of through `np.stack` (no second full copy at Stop);
  changing the Recordings folder starts a new session folder.
- **LeRobot viewer:** plain `list<float>` columns (datasets from other tools) are read; an episode is cut out of the
  data file instead of converting the whole column for every episode (1000 episodes: about 10 s to 0.24 s); each
  episode is checked once; the PyAV fallback keeps one open decoder and seeks instead of decoding from the start.
- **Train tab:** the environment check and Stop run in a background thread (they froze the window and the ROS spin timer).
- **Settings:** only values that differ from the default are stored, so a better default in a later release reaches
  everyone; the app's start-up / live-change logic is in tested pure functions (`qt_app/settings_apply.py`).
- The config files next to a dataset are chosen by the dataset's name when a folder holds several.
- Removed dead code for a settings widget that no longer exists; CI now fails on undefined names.

### Fixed (Train tab)
- A terminal that fails to open (no D-Bus, missing flag) is skipped for the next one, then the run falls back to the
  background instead of silently never starting; the run script no longer exits non-zero when it has no tty.
- LeRobot: removed the unconfirmed `--wandb.enable` flag (off by default), a camera is required for ACT / Diffusion
  (clear error instead of a LeRobot crash), docs no longer mention the non-existent `--eval_freq`.
- "Check environment" imports what the run imports, so a torch / torchvision mismatch is reported with both versions.
- Three dialogs in the recording controller had their title and message swapped.

### Removed
- The `scipy` dependency (quaternion to roll/pitch/yaw is now `relay_d.utils.numerics.quat_to_euler`).

### Added
- **Review tab opens LeRobot datasets** (`ui/review/lerobot_reader.py`): episode list with checks, camera selector
  and frame slider, state / action curves. View-only. When a folder has both recorded demos and a dataset, a
  source selector appears.
- **Complete Settings page**, generated from one table (`relay_d/utils/user_settings.py`, no Qt, also read by
  the command-line tools): theme (now persisted and applied at start), log level, startup page, ROS spin interval
  and domain id, recording defaults used when the YAML is silent (frequency, buffer, sync, warm-up, session folder
  name), HDF5 compression (replaces the dead `enable_compression`), TF / topic timeouts and QoS depth, review
  defaults, post-process defaults and LeRobot encoder settings, training and inference defaults. The old
  duplicate settings writer, the unused `auto_save` and the `recording_frequency` live override were removed.
- **Train tab and `relay-d-train`** (`relay_d/training/`): start robomimic or LeRobot (ACT, Diffusion) training
  on a converted dataset from the GUI or the command line. It opens a terminal that keeps running when the
  GUI closes, lets you edit epochs/steps, batch size, learning rate and more, checks the Python
  environment, and lists / stops / follows runs (`training_runs/` next to the dataset).
- **"Generate Defaults..." buttons** on the Config and Post-process pages: copy
  `config_template.yaml` and `postprocess_config_template.yaml` into a folder you pick. That
  folder is saved as the `config_folder` setting and is where the Insert / Load Config /
  Load Options dialogs open; when unset they use the HDF5 load folder from Settings.
- **Headless converter** `relay-d-convert` (also `scripts/convert_dataset.py`): convert recordings to
  robomimic or LeRobot without the GUI or ROS (`relay_d/acquisition/convert_cli.py`).
- A `test` job in CI running the ROS-free pytest suite; CI now runs on `main` only.
- **`postprocess_config.yaml`**: conversion options kept apart from the acquisition config
  (`postprocess.robomimic`: `metadata` env args, `training_model`, `splits`;
  `postprocess.lerobot`: `task`, `robot_type`, `state_streams`). Picked up automatically
  from the folder of the loaded config, or via "Load Options..." on the Post-process page,
  and copied into the dataset folder. The same keys inside the acquisition config still
  work but log a deprecation warning. See `relay_d/acquisition/qt_app/ui/config/templates/postprocess_config_template.yaml`;
  `AppAPI` accepts `postprocess_config_path`.
- **"Load Config..." on the Post-process page**: load an acquisition config for
  conversion only, without any live ROS topics/TF (no availability check, no
  Record-page containers; the Record page is never affected). "Use Config page"
  switches back to the Config page's config.
- **LeRobotDataset v3.0 output format** in the Post-process page and `AppAPI.convert_recording_to_dataset(output_format="lerobot")`.
  Reuses the existing acquisition config: `actions_to_extract` -> `action`,
  `output.streams` -> `observation.state`, image bridge-map entries ->
  `observation.images.<key>` mp4 videos, `recording_frequency` -> fps. Optional
  `config.lerobot.task` / `config.lerobot.state_streams`. Adds `pyarrow` and `av`
  dependencies and the first pytest tests (`tests/test_lerobot_writer.py`).
- Open-source project scaffolding: `LICENSE` (GNU GPLv3), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue/PR templates, and a CI
  workflow.
- `license`, `authors`, and classifier metadata in `pyproject.toml`.
- A `## Dependencies` section in the README enumerating all pip and
  ROS 2/system dependencies in one place.

### Changed
- The config templates moved from the top-level `config/` folder into the package
  (`relay_d/acquisition/qt_app/ui/config/templates/`) so installed copies include them.
- `DataRecorder.stop_recording()` returns False (and keeps the samples in memory) when the
  HDF5 save fails; it used to report success.
- Record page: the start/pause/resume/stop logic moved from `RecordPage` into
  `RecordingController`, which also owns the Start/Pause/Stop buttons. The Pause button is
  connected once and toggles Pause/Resume (no more disconnecting/reconnecting slots).
  `RecordPage` keeps its public methods and attributes as delegates.
- `ContainerManager.create_containers` no longer freezes the window with `time.sleep`: it
  waits while processing Qt events and ignores re-entrant requests. The unused
  `wait_for_all_data` was removed.
- `pyarrow` and `av` are now an optional extra (`pip install ".[lerobot]"`) instead of base
  dependencies; they are only needed for LeRobot conversion. Anyone who relied on them
  being installed with the package must install the extra.

### Removed
- Dead code: `DataNormalizer` (`record/data_normalization.py`, never configured), the unused
  `gripper_states` recording, `DataRecorder.get_recording_status`/`cleanup`,
  `YAMLDrivenConverter.convert_dataset`, the `ACT` scaffold and `"act"` model type in
  `model_runner.py`, the unused `UI.auto_arrange_containers`/`set_forced_layout`, and
  `data_acquisition_node.py` / `interface_menu.py`.

### Fixed
- The exported `inference_script` crashed at startup (`--model-type` was never defined).
- "Remove" on the Config page left the Post-process page converting with the removed config.
- A demo that failed to write no longer corrupts the rest of a robomimic conversion, and a
  conversion with no demos now reports failure instead of success.
- A failed copy of `acquisition_config.yaml` after conversion is now reported to the user.
- A config without an `input` section no longer raises `AttributeError` in `get_obs_config()`.
- Stopping a recording no longer logs a spurious warning per TF input.
- `scripts/reprocess_recorded_data.py` handles mixed absolute/relative paths and requires `--input_dir`.
- Post-process page: selecting an output format now unchecks the other format button
  (both RoboMimic and LeRobot used to stay highlighted).
- LeRobot conversion of large camera frames uses about half the memory (no unused `next_obs`
  copy, strided image statistics).
- `get_obs_config()` no longer raises `KeyError` when the config has no
  `sync_tolerance_sec`; the `sync` section is simply omitted (check disabled).
- LeRobot conversion no longer demands PyAV (`av`) for recordings that contain no camera
  images, even if the config declares an image bridge-map entry.
- Post-processing no longer silently skips the training config and `action_dispatcher/`
  files when a config has no `training_model`: it now defaults to `bc_rnn_gmm`, and any
  remaining error in that step is shown to the user.
- Removed leftover local absolute file paths (`/home/<user>/...`) from
  `pyuic5`-generated header comments in the compiled Qt UI modules.

## [0.1.1] — 2026-10-08

### Fixed
- Startup crash under NumPy 2: removed the compiled `cv_bridge` dependency (replaced by pure-numpy `imgmsg_to_cv2`/`cv2_to_imgmsg` in `relay_d.utils.ros_media_codec`) and dropped the `numpy<2` pin.
- Qt "xcb" plugin abort when the full `opencv-python` wheel is installed: `sanitize_qt_env()` undoes cv2's Qt plugin-path override.

## [0.1.0] — 2026-07-28

### Added
- Upload of RelayD two subsystems:
  - `relay_d.acquisition` — the `RelayDApp` GUI for recording robot sensor
    data to HDF5 and converting it to Robomimic format.
  - `relay_d.dispatch` — the `lfd-inference-node` CLI for running a trained
    LfD policy live on the robot.
- Shared `relay_d.utils` module (colored console logger, config loader).
- Packaging via `pyproject.toml` (setuptools backend), with `RelayDApp` and
  `lfd-inference-node` console-script entry points.
