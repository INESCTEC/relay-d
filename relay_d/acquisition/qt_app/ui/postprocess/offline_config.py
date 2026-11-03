"""Load an acquisition config for post-processing only.

Unlike ``ConfigPage.insert_config_file`` this does NOT check that the configured
topics/TF are live on the ROS graph, so a recording can be converted without the
robot (or any subscriber) running. It only needs the YAML itself:

  - the file must parse and contain a top-level ``config`` section
  - every ``output.streams`` spec must parse (same check the Config page runs)

The returned ``YamlParser`` is a separate instance from the one the Record page
uses, so loading a config this way can never affect recording.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from relay_d.utils.coloring_logger import logger

from ..config.yaml_parser import YamlParser
from ...utils.stream_builder import StreamBuilder


def load_offline_config(path: str) -> Tuple[Optional[YamlParser], List[str]]:
    """Parse and validate ``path``. Returns ``(parser, problems)``.

    ``parser`` is ``None`` when the config cannot be used (``problems`` explains
    why). When ``parser`` is returned, ``problems`` may still hold non-fatal
    warnings (prefixed ``warning:``).
    """
    if not path or not os.path.isfile(path):
        return None, [f"File not found: {path}"]

    parser = YamlParser()
    if not parser.load_yaml_file(path):
        return None, [f"Could not parse YAML file: {path}"]

    data = parser.yaml_data
    if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
        return None, ["The file has no top-level 'config' section."]

    errors = [f"output.streams: {e}" for e in StreamBuilder(parser).get_build_errors()]
    if errors:
        return None, errors

    problems: List[str] = []
    output = data["config"].get("output") or {}
    if not output:
        problems.append("warning: the config has no 'output' entries, so nothing can be converted.")
    if not data["config"].get("recording_frequency") and not (
        data["config"].get("metadata") or {}
    ).get("recording_frequency"):
        problems.append(
            "warning: no 'recording_frequency' set (required as fps for the LeRobot format)."
        )
    for problem in problems:
        logger.warning(f"Offline config {path}: {problem}")
    return parser, problems
