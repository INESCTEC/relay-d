"""Joint limits from a URDF string (stdlib only), used by the Config builder and ``scripts/extract_joint_limits.py``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional


def parse_joint_limits(xml_string: str, joint_filter: Optional[List[str]] = None) -> List[Dict]:
    """Revolute / prismatic joints as ``{name, type, lower, upper, velocity, effort}``, in URDF order (or ``joint_filter`` order).

    Names in ``joint_filter`` that are not in the URDF are skipped: compare the result with the request to find them.
    Raises ``ValueError`` if the XML is not a URDF.
    """
    try:
        root = ET.fromstring(xml_string.strip())
    except ET.ParseError as exc:
        raise ValueError(f"failed to parse URDF XML: {exc}") from exc
    if root.tag != "robot":
        raise ValueError(f"not a URDF: the root element is <{root.tag}>, expected <robot>")

    joints: Dict[str, Dict] = {}
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "prismatic"):
            continue
        lim = j.find("limit")
        attr = lim.attrib if lim is not None else {}
        name = j.get("name", "")
        joints[name] = {
            "name": name,
            "type": j.get("type"),
            **{k: float(attr.get(k, 0.0)) for k in ("lower", "upper", "velocity", "effort")},
        }
    if joint_filter:
        return [joints[n] for n in joint_filter if n in joints]
    return list(joints.values())


def limits_for(field: str, joints: List[Dict], names: List[str]) -> List[Dict]:
    """``limits:`` entries for the stream field ``position`` / ``velocity`` / ``effort`` of the joints ``names``, in that order.

    Position uses the URDF lower/upper, velocity and effort are symmetric (±max), like ``scripts/extract_joint_limits.py``.
    Raises ``ValueError`` for an unknown field, for joints missing from ``joints`` or for a joint whose URDF limit is 0 (not set).
    """
    if field not in ("position", "velocity", "effort"):
        raise ValueError(f"no URDF limit for '{field}' (only position, velocity and effort have one)")
    by_name = {j["name"]: j for j in joints}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"joints not in the URDF (or not revolute / prismatic): {missing}")
    out = []
    for n in names:
        j = by_name[n]
        lo, hi = (j["lower"], j["upper"]) if field == "position" else (-j[field], j[field])
        if lo >= hi:
            raise ValueError(f"the URDF has no usable {field} limit for joint '{n}' ({lo} .. {hi})")
        out.append({"lower_lim": lo, "upper_lim": hi})
    return out
