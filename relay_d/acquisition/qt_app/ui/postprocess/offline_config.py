"""Load an acquisition config for post-processing only.

Unlike ``ConfigPage.insert_config_file`` this does NOT check that the configured
topics/TF are live on the ROS graph, so a recording can be converted without the
robot (or any subscriber) running. It only needs the YAML itself:

  - the file must parse and contain a top-level ``config`` section
  - every ``output.streams`` spec must parse (same check the Config page runs)
  - no removed key (``metadata``, ``training_model``, ...) is left in it

The returned ``YamlParser`` is a separate instance from the one the Record page
uses, so loading a config this way can never affect recording.
"""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Tuple

import yaml

from relay_d.dispatch import save_dispatch_templates
from relay_d.utils.coloring_logger import logger

from ..config.yaml_parser import YamlParser
from .postprocess_options import OPTIONS_FILENAME, PostprocessOptions, removed_key_errors
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

    errors = removed_key_errors(data["config"]) + [f"output.streams: {e}" for e in StreamBuilder(parser).get_build_errors()]
    if errors:
        return None, errors

    problems: List[str] = []
    output = data["config"].get("output") or {}
    if not output:
        problems.append("warning: the config has no 'output' entries, so nothing can be converted.")
    if not data["config"].get("recording_frequency"):
        problems.append(
            "warning: no 'recording_frequency' set (required as fps for the LeRobot format)."
        )
    for problem in problems:
        logger.warning(f"Offline config {path}: {problem}")
    return parser, problems


def load_conversion_config(config_path: str, options_path: Optional[str] = None) -> Tuple[YamlParser, PostprocessOptions]:
    """Config + format options of a headless conversion (relay-d-convert, AppAPI): options from ``options_path``,
    else a postprocess_config.yaml next to the config, else the defaults. Raises ValueError if a file is unusable."""
    parser, problems = load_offline_config(config_path)
    if parser is None:
        raise ValueError("unusable config:\n  " + "\n  ".join(problems))
    options_path = options_path or PostprocessOptions.find_sibling(config_path)
    if not options_path:
        return parser, PostprocessOptions()
    options, problems = PostprocessOptions.load(options_path)
    if options is None:
        raise ValueError(f"unusable options file {options_path}:\n  " + "\n  ".join(problems))
    return parser, options


def write_dataset_files(output_dir: str, parser: YamlParser, options_source: Optional[str] = None) -> None:
    """The files kept next to a converted dataset, the same for the GUI, AppAPI and relay-d-convert: the configs used
    (acquisition_config.yaml, postprocess_config.yaml) and action_dispatcher/ for lfd-inference-node.
    ``output_dir`` is the folder holding the dataset (for LeRobot the one above the dataset folder: where the
    training launcher looks). Raises on failure; the dataset itself is already written then."""
    os.makedirs(output_dir, exist_ok=True)
    # A byte copy keeps the comments, but only while the file is still what was converted with
    # (the Config page can edit the config in memory); otherwise the data used is dumped.
    target = os.path.join(output_dir, "acquisition_config.yaml")
    source = parser.get_current_config_path()
    if source and os.path.isfile(source) and _file_data(source) == parser.yaml_data:
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copyfile(source, target)
    elif not parser.save_yaml_file(file_path=target):
        raise RuntimeError("could not write acquisition_config.yaml")
    if options_source:
        shutil.copy2(options_source, os.path.join(output_dir, OPTIONS_FILENAME))
    dispatcher = os.path.join(output_dir, "action_dispatcher")
    os.makedirs(dispatcher, exist_ok=True)
    if not parser.save_yaml_file(file_path=os.path.join(dispatcher, "obs_config.yaml"), data=parser.get_obs_config()):
        raise RuntimeError("could not write action_dispatcher/obs_config.yaml")
    save_dispatch_templates(dispatcher)


def _file_data(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
