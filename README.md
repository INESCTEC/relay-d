![logo](./img/relayd_logo.png)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

# Relay-D

**Description:** Relay-D is a standalone application for recording synchronized data from multi-modal sensor robotic systems built on ROS 2, and for running the trained policies back on the robot. It streamlines the entire Learning-from-Demonstration (LfD) training loop by facilitating three core tasks: recording robot sensor data, converting it into imitation-learning training datasets, and deploying the trained policy back onto the robot.

- **What it does** — records robot sensor topics and TF to HDF5 per a YAML config, converts recordings into [Robomimic](https://robomimic.github.io/)-format training datasets, and runs a trained LfD policy live on the robot (subscribing to sensor data, running a forward pass, and dispatching the resulting action).
- **Who it is for** — robotics researchers and engineers building imitation-learning pipelines on ROS 2 (Jazzy) robots.
- **The problem it solves** — LfD projects usually stitch together bespoke, one-off scripts for recording, format conversion, and inference. Relay-D replaces that with a single, consistent tool covering the whole loop, so datasets recorded for training are guaranteed to be compatible with the config format consumed at inference time.

**Screenshot:**

<!-- TODO: add a screenshot of the RelayDApp GUI (e.g. the Record page with live sensor feeds) here. -->

---

## 1. Project status

This project is currently **in progress**: it is actively being developed and core functionality (recording, conversion, and inference) is working end-to-end, but features and structure may still change.

---

## 2. Technology stack

- **Language:** Python ≥ 3.12
- **Frameworks and libraries:** ROS 2 Jazzy (`rclpy`, `tf2_ros`), PyQt5 (`RelayDApp` GUI), PyTorch, [Robomimic](https://robomimic.github.io/) (optional, imitation-learning backend), OpenCV, h5py
- **Other tools:** pip/setuptools packaging, GitHub Actions (CI)

Relay-D is a plain pip package (not a colcon/ament package) intended to run standalone inside a Python environment where ROS 2 Jazzy has already been sourced — it is not a ROS 2 node package in its own right, but it depends on a sourced ROS 2 environment at runtime.

It is composed of two subsystems that interoperate directly:

- **`relay_d.acquisition`** — the `RelayDApp` GUI: records robot sensor data (topics + TF) to HDF5 per a YAML config, and converts recordings to Robomimic format. Also exposes a headless `AppAPI` for programmatic recording.
- **`relay_d.dispatch`** — the `lfd-inference-node` CLI: runs a trained LfD policy live on the robot. Also exposes a programmatic `InferenceSession` API.
- **`relay_d.utils`** — small utilities shared by both subsystems (currently the colored console logger).

See [Architecture diagrams](#6-architecture-diagrams) for how these pieces fit together.

---

## 3. Dependencies

### Python (pip-installable, declared in `pyproject.toml`)

| Package | Used for |
|---|---|
| `numpy` (`<2`) | array/data handling; pinned below 2.0 for `cv_bridge` compatibility |
| `pyyaml` | YAML config parsing |
| `torch` | policy model loading/inference |
| `h5py` | HDF5 recording/dataset I/O |
| `opencv-python-headless` | image processing (no GUI/X11 dependency — PyQt5 supplies the GUI) |
| `pyqt5`, `PyQt5-sip` | `RelayDApp` GUI toolkit |
| `pyqtgraph` | live plotting widgets in the GUI's review page |
| `xmltodict` | parsing robot description / URDF-derived XML |

**Optional extra** — `pip install ".[robomimic]"`:

| Package | Used for |
|---|---|
| `robomimic` (`>=0.4`) | imitation-learning policy backend. **Must** be installed from an [ARISE-Initiative](https://github.com/ARISE-Initiative/robomimic) source checkout, not plain PyPI (see [Installation](#4-installation)) |

**Dev-only** — `pip install --group dev .` / `[dependency-groups] dev`:

| Package | Used for |
|---|---|
| `jurigged` | live code-reloading during development |
| `watchdog` | filesystem-change monitoring used by the dev reload workflow |

### System / ROS 2 (not pip-installable)

These must come from a sourced ROS 2 Jazzy environment (see [Installation](#4-installation)) rather than `pip`:

| Package | Used for |
|---|---|
| `rclpy` | ROS 2 Python client library — nodes, topics, actions |
| `tf2_ros`, `tf2_msgs` | transform lookups |
| `geometry_msgs`, `std_msgs`, `sensor_msgs`, `sensor_msgs_py`, `trajectory_msgs`, `builtin_interfaces` | ROS 2 message types |
| `cv_bridge` | ROS `Image` message ⇄ OpenCV array conversion |
| `rosidl_runtime_py` | introspecting ROS 2 message/service definitions |
| `xacro` | resolving robot description macros (used by `scripts/extract_joint_limits.py`) |

### Other system libraries

- **Qt5** shared libraries (via `PyQt5`) — required at runtime for the `RelayDApp` GUI.
- **libhdf5** (via `h5py`) — usually pulled in automatically by the `h5py` wheel; only relevant if building `h5py` from source.

Keep dependencies up to date to avoid security vulnerabilities and compatibility issues (see `pyproject.toml` and `CHANGELOG.md` for the currently pinned versions).

---

## 4. Installation

```bash
# 1. Source ROS 2
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-tf2-ros ros-jazzy-geometry-msgs ros-jazzy-std-msgs \
    ros-jazzy-sensor-msgs ros-jazzy-cv-bridge ros-jazzy-image-transport

# 2. Clone
cd ~/ros2_ws/src
git clone git@github.com:pascd/Relay-D.git
cd Relay-D

# 3. Install
pip install -e .

# Optional: robomimic backend for policy inference.
# IMPORTANT: PyPI's `robomimic` package only goes up to 0.3.0 and is NOT
# compatible with checkpoints trained from a newer ARISE-Initiative source
# checkout (this dispatcher targets >=0.4, e.g. 0.5.0). Install your
# robomimic source checkout in editable mode FIRST, then install this
# extra — do NOT rely on `pip install robomimic` alone.
# Version from source and pip may differ when training models
pip install -e /path/to/your/robomimic/checkout   # e.g. your ARISE-Initiative source clone
pip install -e ".[robomimic]"

# Verify afterward that the correct version/source is active:
python -c "import robomimic; print(robomimic.__version__, robomimic.__file__)"
# Expect your checkout's version and a path under your local checkout,
# NOT a site-packages/robomimic path installed from PyPI.
```

This is a plain pip package (not a colcon/ament package) — no `colcon build` step is required. It just needs to run inside a Python environment where ROS 2 has already been sourced.

---

## 5. Usage

### Recording data (RelayDApp)

```bash
RelayDApp
```

Three pages: **Config** (load/validate your YAML), **Record** (start/pause/stop, live sensor feeds), **Post-process** (convert raw recordings to Robomimic format).

### Running inference on the robot

```bash
lfd-inference-node \
    --checkpoint path/to/model.pth \
    --obs-config path/to/obs_config.yaml \
    --action-config path/to/action_config.yaml \
    --device cpu
```

See `relay_d/dispatch/inference_node.py` and `relay_d/dispatch/config_templates/*.yaml` for the full set of options and config syntax.

### Package layout

```
relay_d/
├── utils/              # shared utilities (coloring_logger)
├── dispatch/           # policy inference + action dispatch (lfd-inference-node)
└── acquisition/        # RelayDApp GUI + AppAPI (data recording + Robomimic conversion)
config/                 # top-level YAML config template for acquisition
scripts/                # standalone dev CLIs (extract_joint_limits.py, reprocess_recorded_data.py)
debug/                  # standalone offline dev/parity-testing tools, not part of the installed package
```

---

## 6. Architecture diagrams

The full pipeline — from YAML config, through recording and conversion, to training and live inference — plus the relationship between the `relay_d.acquisition` and `relay_d.dispatch` subsystems, is diagrammed in:

- [`docs/diagrams/architecture.md`](docs/diagrams/architecture.md) — pipeline flowchart (Mermaid)

---

## 7. Known issues

- **No automated test suite for ROS 2-dependent code paths.** CI only runs a dependency-installable-subset build/lint check; changes touching `rclpy`/`tf2_ros`/`cv_bridge` code paths must currently be validated manually (e.g. running `RelayDApp`, recording a demo, converting it). See [CONTRIBUTING.md](CONTRIBUTING.md).
- **`robomimic` version mismatch on PyPI.** The published `robomimic` package on PyPI only goes up to `0.3.0`, which is **not** compatible with checkpoints trained against a newer [ARISE-Initiative](https://github.com/ARISE-Initiative/robomimic) source checkout (this project targets `>=0.4`). You must install your own `robomimic` source checkout in editable mode before installing the `[robomimic]` extra — see [Installation](#4-installation).

---

## 8. License

Relay-D is licensed under the **GNU General Public License v3.0 or later** — see [LICENSE](LICENSE) for the full text.

Copyright © 2026 INESC TEC.

---

## 9. Documentation and resources

- [CHANGELOG.md](CHANGELOG.md) — release history and notable changes
- [CITATION.cff](CITATION.cff) — citation metadata; use this if you use Relay-D in academic work
- `relay_d/dispatch/config_templates/*.yaml` and `config/config_template.yaml` — reference config templates for the dispatch and acquisition subsystems
- [Architecture diagrams](docs/diagrams/architecture.md) — pipeline and subsystem overview

---

## 10. Community standards and contribution

This repository follows INESC TEC's organisation-wide governance documents. Please review these before contributing:

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Issue Reporting Template](docs/reporting_template.md)
- [Security Policy](SECURITY.md)

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment, coding conventions, and the pull-request process. By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

To report a security vulnerability, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

---

## 11. Credits and acknowledgements

- Pedro Dias – Developer, INESC TEC
- Artur Cordeiro – Developer, INESC TEC

---

## 12. Contacts

For support or inquiries, contact:

- Pedro Dias – pedro.dias@inesctec.pt <!-- TODO: confirm actual address -->
- Artur Cordeiro – artur.cordeiro@inesctec.pt <!-- TODO: confirm actual address -->
