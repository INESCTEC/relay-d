# Third-Party Dependency Licenses

Relay-D itself is licensed under the [GNU GPLv3](LICENSE). This document lists
the license of every direct dependency, so downstream users/redistributors
can check compatibility with their own use case. License data was pulled from
each package's published metadata (PyPI JSON API / SPDX `license_expression`)
or its upstream repository, not guessed from memory.

## Python — required (`pyproject.toml` → `[project] dependencies`)

| Package | License | Notes |
|---|---|---|
| `numpy` | BSD-3-Clause (+ small bundled 0BSD/MIT/Zlib/CC0-1.0 components) | |
| `pyyaml` | MIT | |
| `torch` | BSD-3-Clause (+ bundled Apache-2.0/BSD-2-Clause/BSL-1.0/MIT components) | |
| `h5py` | BSD-3-Clause | |
| `opencv-python-headless` | Apache-2.0 | Packaging scripts (the `opencv-python` project itself) are MIT; the bundled OpenCV library is Apache-2.0 (OpenCV ≥4.5) |
| `pyqt5` | **GPL-3.0** (or a paid Riverbank commercial license) | See ⚠️ note below |
| `PyQt5-sip` | BSD-2-Clause | |
| `pyqtgraph` | MIT | |
| `xmltodict` | MIT | |
| `scipy` | BSD-3-Clause | |

## Python — optional extra (`pip install ".[robomimic]"`)

| Package | License | Notes |
|---|---|---|
| `robomimic` | MIT | This project requires the [ARISE-Initiative source checkout](https://github.com/ARISE-Initiative/robomimic) (`>=0.4`), not the PyPI release (which only goes up to 0.3.0) — both are MIT |

## Python — dev only (`[dependency-groups] dev`)

| Package | License |
|---|---|
| `jurigged` | MIT |
| `watchdog` | Apache-2.0 |

## ROS 2 (system, sourced from a ROS 2 Jazzy install — not pip-installable)

| Package | License | Notes |
|---|---|---|
| `rclpy` | Apache-2.0 | |
| `tf2_ros`, `tf2_msgs` | BSD-3-Clause | (`ros2/geometry2`) |
| `std_msgs`, `geometry_msgs`, `sensor_msgs`, `trajectory_msgs`, `builtin_interfaces` | Apache-2.0 | per each package's `package.xml` (`ros2/common_interfaces` and related) |
| `sensor_msgs_py`, `rosidl_runtime_py` | Apache-2.0 | |
| `cv_bridge` | Apache-2.0 | (`ros-perception/vision_opencv`) |
| `xacro` | BSD-3-Clause | (`ros/xacro`) |

## Other system libraries

| Library | License | Notes |
|---|---|---|
| Qt5 (via `PyQt5`) | LGPLv3 / GPLv3 / Commercial (Qt Company) | Distro-packaged Qt5 (e.g. Ubuntu's `libqt5*`) is typically usable under LGPLv3; combined with PyQt5's GPL bindings below, GPLv3 obligations apply to the whole application regardless |
| libhdf5 (via `h5py` wheels) | HDF5 License (BSD-style, permissive) | |

## ⚠️ Why GPLv3 specifically

`PyQt5` — a **required**, non-optional dependency of the `RelayDApp` GUI — is
dual-licensed by Riverbank Computing under **GPLv3 or a paid commercial
license**. Distributing an application built on the GPL edition of PyQt5
effectively requires the whole application to be GPL-compatible.

No other dependency listed above restricts the choice of license the way
PyQt5 does — everything else here (Apache-2.0, BSD, MIT) is permissive and
compatible with GPLv3 distribution.
