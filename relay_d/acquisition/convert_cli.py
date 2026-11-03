"""Headless dataset conversion: recorded .h5 files -> robomimic or LeRobot, no GUI, no ROS.

    relay-d-convert --config acquisition.yaml --format lerobot \\
        --input recordings/ --output my_lerobot_dataset

The acquisition config is only parsed and checked (no live topic/TF check), exactly like
"Load Config..." on the Post-process page. Format options come from ``--options`` or a
``postprocess_config.yaml`` next to the config (see relay_d/acquisition/qt_app/ui/config/templates/postprocess_config_template.yaml).

Exit codes: 0 success, 1 conversion failed, 2 bad arguments / unusable config or inputs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional


from .qt_app.ui.postprocess.offline_config import load_offline_config
from .qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from .qt_app.ui.postprocess.postprocess_options import PostprocessOptions
from .qt_app.ui.postprocess.yaml_driven_converter import YAMLDrivenConverter, natural_key

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def collect_inputs(paths: List[str]) -> List[str]:
    """Expand files and directories (searched recursively for ``*.h5``) into a sorted list."""
    found: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                found.extend(os.path.join(root, f) for f in files if f.endswith(".h5"))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(os.path.abspath(p) for p in found), key=natural_key)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="relay-d-convert",
        description="Convert recorded .h5 files to a training dataset without the GUI.",
    )
    ap.add_argument("--config", required=True, help="acquisition config (YAML) used for the recording")
    ap.add_argument("--options", help="postprocess_config.yaml (default: one next to --config, if any)")
    ap.add_argument("--format", required=True, choices=["robomimic", "lerobot"])
    ap.add_argument("--input", required=True, nargs="+", help=".h5 files and/or directories to search")
    ap.add_argument(
        "--output",
        required=True,
        help="robomimic: output .h5 file; lerobot: dataset directory (must not exist or be empty)",
    )
    ap.add_argument("--task", help="LeRobot task text (overrides postprocess_config.yaml / dataset_name)")
    return ap


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    parser, problems = load_offline_config(args.config)
    if parser is None:
        return _fail("unusable config:\n  " + "\n  ".join(problems), EXIT_USAGE)
    for problem in problems:
        print(problem, file=sys.stderr)

    options_path = args.options or PostprocessOptions.find_sibling(args.config)
    options = PostprocessOptions()
    if options_path:
        loaded, opt_problems = PostprocessOptions.load(options_path)
        if loaded is None:
            return _fail(f"unusable options file {options_path}:\n  " + "\n  ".join(opt_problems), EXIT_USAGE)
        options = loaded
        print(f"Using format options from {options_path}")

    try:
        inputs = collect_inputs(args.input)
    except FileNotFoundError as e:
        return _fail(f"input not found: {e}", EXIT_USAGE)
    if not inputs:
        return _fail("no .h5 files found in the given inputs", EXIT_USAGE)

    output = os.path.abspath(args.output)
    if args.format == "lerobot":
        if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
            return _fail(f"output must not exist or be an empty directory: {output}", EXIT_USAGE)
    else:
        if os.path.exists(output):
            return _fail(f"output already exists, refusing to overwrite: {output}", EXIT_USAGE)
        os.makedirs(os.path.dirname(output), exist_ok=True)

    converter = YAMLDrivenConverter(
        PostprocessConfigLoader(), data_config=parser.yaml_data["config"]
    )
    converter.options = options

    print(f"Converting {len(inputs)} file(s) to {args.format} -> {output}")
    if args.format == "lerobot":
        ok = converter.convert_to_lerobot(inputs, output, task=args.task)
    else:
        ok = converter.convert_multiple_files("robomimic", inputs, output)
    if not ok:
        return _fail(converter.last_error or "conversion failed (see the log above)", EXIT_FAILED)

    # Keep the configs used next to the result, like the GUI does.
    if args.format == "lerobot":
        dest_dir, prefix = output, ""
    else:
        dest_dir = os.path.dirname(output)
        prefix = os.path.splitext(os.path.basename(output))[0] + "_"
    shutil.copy2(args.config, os.path.join(dest_dir, f"{prefix}acquisition_config.yaml"))
    if options.source:
        shutil.copy2(options.source, os.path.join(dest_dir, f"{prefix}postprocess_config.yaml"))

    print(f"Done: {output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
