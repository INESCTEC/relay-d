#!/usr/bin/env python3
"""
Extract joint position, velocity, and effort limits from a URDF robot description
and print them as ready-to-paste YAML blocks for acquisition_config.yaml.

Usage
-----
From a URDF file:
  python3 scripts/extract_joint_limits.py --urdf /path/to/robot.urdf

From the live ROS 2 parameter server (robot_state_publisher must be running):
  python3 scripts/extract_joint_limits.py --ros2

Filter / reorder joints (must match the joint_names order in acquisition_config.yaml):
  python3 scripts/extract_joint_limits.py --urdf robot.urdf \\
      --joints iiwa_joint_1 iiwa_joint_2 iiwa_joint_3 \\
               iiwa_joint_4 iiwa_joint_5 iiwa_joint_6 iiwa_joint_7

Change the output stream-name prefix (default: robot0):
  python3 scripts/extract_joint_limits.py --urdf robot.urdf --prefix robot0

Change the input name used in the spec (default: joint_1):
  python3 scripts/extract_joint_limits.py --urdf robot.urdf --input-name joint_1
"""

import argparse
import subprocess
import sys

try:
    import xmltodict
except ImportError:
    sys.exit("ERROR: xmltodict is required.  Install with: pip install xmltodict")


# ---------------------------------------------------------------------------
# URDF source helpers
# ---------------------------------------------------------------------------

def _load_from_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError as exc:
        sys.exit(f"ERROR: cannot read URDF file '{path}': {exc}")


def _load_from_ros2(node_name: str = "robot_state_publisher",
                    param_name: str = "robot_description") -> str:
    """Fetch robot_description from the ROS 2 parameter server using the CLI."""
    cmd = ["ros2", "param", "get", f"/{node_name}", param_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        sys.exit("ERROR: 'ros2' command not found — is your ROS 2 workspace sourced?")
    except subprocess.TimeoutExpired:
        sys.exit(f"ERROR: timed out waiting for: {' '.join(cmd)}")

    if result.returncode != 0:
        sys.exit(
            f"ERROR: ros2 param get failed (exit {result.returncode}).\n"
            f"  Make sure /{node_name} is running.\n"
            f"  stderr: {result.stderr.strip()}"
        )

    # ros2 param get output: "String value is:\n  <?xml ..."
    output = result.stdout
    marker = "String value is:"
    if marker in output:
        xml_string = output[output.index(marker) + len(marker):].strip()
    else:
        # Fallback: treat the whole output as XML (some ros2 versions differ)
        xml_string = output.strip()

    if not xml_string.startswith("<"):
        sys.exit(
            "ERROR: unexpected output from ros2 param get — could not find XML.\n"
            f"  Output was:\n{output}"
        )
    return xml_string


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def parse_joint_limits(xml_string: str, joint_filter: list | None = None) -> list:
    """
    Parse a URDF XML string and return a list of dicts for each
    revolute / prismatic joint, in URDF document order (or joint_filter order).

    Each dict:
      name     : str
      type     : str   ('revolute' | 'prismatic')
      lower    : float  position lower limit (rad or m)
      upper    : float  position upper limit
      velocity : float  max velocity (always positive)
      effort   : float  max effort (always positive)
    """
    try:
        robot = xmltodict.parse(xml_string)["robot"]
    except (KeyError, Exception) as exc:
        sys.exit(f"ERROR: failed to parse URDF XML: {exc}")

    raw_joints = robot.get("joint", [])
    if not isinstance(raw_joints, list):
        raw_joints = [raw_joints]

    # Build name → data map
    joint_map: dict = {}
    for j in raw_joints:
        jtype = j.get("@type", "")
        if jtype not in ("revolute", "prismatic"):
            continue
        name = j.get("@name", "")
        lim = j.get("limit", {}) or {}
        joint_map[name] = {
            "name": name,
            "type": jtype,
            "lower":    float(lim.get("@lower",    0.0)),
            "upper":    float(lim.get("@upper",    0.0)),
            "velocity": float(lim.get("@velocity", 0.0)),
            "effort":   float(lim.get("@effort",   0.0)),
        }

    if joint_filter:
        result = []
        for jname in joint_filter:
            if jname not in joint_map:
                print(f"WARNING: joint '{jname}' not found in URDF (skipped)", file=sys.stderr)
            else:
                result.append(joint_map[jname])
        return result

    return list(joint_map.values())


# ---------------------------------------------------------------------------
# YAML block generation
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a float to 4 decimal places, right-aligned in 8 chars."""
    return f"{value:8.4f}"


def _limits_block(stream_name: str, spec: str, joints: list,
                  lo_key: str, hi_key: str, symmetric: bool) -> str:
    """
    Produce one YAML dict-form stream block.

    symmetric=True  → lower_lim = -value, upper_lim = +value  (vel / eff)
    symmetric=False → lower_lim = joint[lo_key], upper_lim = joint[hi_key]  (pos)
    """
    lines = [
        f"      {stream_name}:",
        f"        specs:",
        f"          - {spec}",
        f"        normalize: true",
        f"        normalized_range: [-1.0, 1.0]",
        f"        limits:",
    ]
    for j in joints:
        if symmetric:
            val = j[hi_key]
            lo = -val
            hi =  val
        else:
            lo = j[lo_key]
            hi = j[hi_key]
        lines.append(
            f"          - {{lower_lim: {_fmt(lo)}, upper_lim: {_fmt(hi)}}}"
            f"   # {j['name']}  ({j['type']})"
        )
    return "\n".join(lines)


def print_yaml_blocks(joints: list, prefix: str, input_name: str) -> None:
    if not joints:
        sys.exit("ERROR: no matching revolute/prismatic joints found.")

    pos_block = _limits_block(
        f"{prefix}_joint_pos",
        f"{input_name}.position[:]",
        joints,
        lo_key="lower", hi_key="upper",
        symmetric=False,
    )
    vel_block = _limits_block(
        f"{prefix}_joint_vel",
        f"{input_name}.velocity[:]",
        joints,
        lo_key=None, hi_key="velocity",
        symmetric=True,
    )
    eff_block = _limits_block(
        f"{prefix}_joint_eff",
        f"{input_name}.effort[:]",
        joints,
        lo_key=None, hi_key="effort",
        symmetric=True,
    )

    separator = "\n"
    print("# ── paste into acquisition_config.yaml → output.streams ──────────────────\n")
    print(pos_block)
    print(separator)
    print(vel_block)
    print(separator)
    print(eff_block)
    print("\n# ──────────────────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--urdf", metavar="PATH",
        help="Path to a URDF (.urdf / .xml) file.",
    )
    source.add_argument(
        "--ros2", action="store_true",
        help="Fetch robot_description from the ROS 2 parameter server.",
    )

    p.add_argument(
        "--joints", nargs="+", metavar="JOINT_NAME",
        help="Filter and order output to these joint names "
             "(must match joint_names in acquisition_config.yaml). "
             "Defaults to all revolute/prismatic joints in URDF order.",
    )
    p.add_argument(
        "--prefix", default="robot0", metavar="PREFIX",
        help="Stream-name prefix (default: robot0). "
             "Produces <PREFIX>_joint_pos, <PREFIX>_joint_vel, <PREFIX>_joint_eff.",
    )
    p.add_argument(
        "--input-name", default="joint_1", metavar="NAME",
        help="Input source name used in the spec string "
             "(default: joint_1, matching acquisition_config.yaml).",
    )
    p.add_argument(
        "--node-name", default="robot_state_publisher", metavar="NODE",
        help="ROS 2 node that owns the robot_description parameter "
             "(default: robot_state_publisher). Only used with --ros2.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.ros2:
        xml_string = _load_from_ros2(node_name=args.node_name)
    else:
        xml_string = _load_from_file(args.urdf)

    joints = parse_joint_limits(xml_string, joint_filter=args.joints)
    print_yaml_blocks(joints, prefix=args.prefix, input_name=args.input_name)


if __name__ == "__main__":
    main()
