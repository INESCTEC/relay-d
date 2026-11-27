#!/usr/bin/env bash
# Installs Relay-D on Ubuntu 24.04 with ROS 2 Jazzy already installed. Safe to run again (it updates).
#   - relay_d goes into the system Python (the ROS one), so any script can `import relay_d` without a venv.
#     Dependencies come from apt when apt has them (built against ROS's NumPy 1.26); pip adds the rest in ~/.local.
#   - torch / lerobot / robomimic go into their own venvs in ~/.relayd/envs (same as the Train tab button).
# Usage: scripts/install.sh [--no-lerobot] [--no-robomimic]
set -euo pipefail

envs=(lerobot robomimic)
for arg in "$@"; do
    case "$arg" in
        --no-lerobot) envs=("${envs[@]/lerobot}") ;;
        --no-robomimic) envs=("${envs[@]/robomimic}") ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done
py=/usr/bin/python3  # the ROS Python: a python3 earlier in PATH (pyenv, mise, conda) is not what ROS packages target
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. ROS 2 Jazzy must already be there (this script does not install it).
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    echo "ROS 2 Jazzy not found in /opt/ros/jazzy. Install it first (Ubuntu 24.04):" >&2
    echo "  https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html  (ros-jazzy-ros-base is enough: step 2 adds rviz2 and the rest)" >&2
    exit 1
fi
if ! "$py" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo "$py must be 3.12 (the Python of ROS Jazzy); found $("$py" --version 2>&1)" >&2
    exit 1
fi

# 2. System packages: Qt, OpenCV, NumPy, YAML from apt, built against the NumPy ROS uses; rviz2 and
#    robot_state_publisher for Review playback; sensor_msgs_py (relay_d imports it).
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-numpy python3-yaml python3-opencv python3-pyqt5 \
    ros-jazzy-tf2-ros ros-jazzy-geometry-msgs ros-jazzy-std-msgs ros-jazzy-sensor-msgs \
    ros-jazzy-image-transport ros-jazzy-xacro ros-jazzy-rviz2 ros-jazzy-robot-state-publisher ros-jazzy-sensor-msgs-py

# 3+4. What apt lacks (or has too old), then relay_d itself with --no-deps so pip never pulls
# opencv-python-headless / NumPy 2 over the apt ones (that breaks rclpy). pyarrow 21+ needs NumPy 2.
pip=("$py" -m pip install --user --break-system-packages)
"${pip[@]}" h5py "pyarrow<21" av "pyqtgraph>=0.13.7"
"${pip[@]}" --no-deps -e "$repo"

# 5. Training / inference venvs: the same script the Train tab runs, here in the foreground.
for fmt in "${envs[@]}"; do
    [ -n "$fmt" ] || continue
    script="$(cd "$repo" && "$py" -c "import shutil, sys
from relay_d.training import envsetup as e
d = e.default_env_dir('$fmt')
if e.status(d) == 'running':  # checked before the state files are deleted: a second build would corrupt the venv
    sys.exit(f'The $fmt environment is already being built (log: {d}.setup.log). Wait for it to finish, then run this again.')
d.parent.mkdir(parents=True, exist_ok=True)
for f in d.parent.glob(d.name + '.setup.*'): f.unlink()
s = d.with_name(d.name + '.setup.sh')
s.write_text(e.build_script('$fmt', d, shutil.which('uv'), sys.executable))
print(s)")"
    bash "$script" </dev/null
    if [ "$(cat "${script%.sh}.exit" 2>/dev/null)" != 0 ]; then
        echo "The $fmt environment failed; log: ${script%.sh}.log" >&2
        exit 1
    fi
done

echo
echo "Relay-D installed."
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "Add ~/.local/bin to PATH (RelayDApp, relay-d-convert, relay-d-train, lfd-inference-node live there)." ;; esac
echo "Every new shell needs ROS: source /opt/ros/jazzy/setup.bash   (or add it to ~/.bashrc)"
echo "Then: RelayDApp"
