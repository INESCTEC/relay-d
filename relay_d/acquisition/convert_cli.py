"""Headless dataset conversion: recorded .h5 files -> robomimic or LeRobot, no GUI, no ROS.

    relay-d-convert --config acquisition.yaml --format lerobot \\
        --input recordings/ --output my_lerobot_dataset

The acquisition config is only parsed and checked (no live topic/TF check), exactly like
"Load Config..." on the Post-process page. Format options come from ``--options`` or a
``postprocess_config.yaml`` next to the config (see relay_d/acquisition/qt_app/ui/config/templates/postprocess_config_template.yaml).

With ``--check`` nothing is written: it prints what the conversion would do (each demo's verdict, the output layout)
and exits 0 if at least one demo would be converted, else 1.

Exit codes: 0 success, 1 conversion failed, 2 bad arguments / unusable config or inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


from .qt_app.ui.postprocess.offline_config import load_conversion_config, write_dataset_files
from .qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
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
        help="robomimic: output .h5 file; lerobot: dataset directory (must not exist or be empty)",
    )
    ap.add_argument("--task", help="LeRobot task text (overrides postprocess_config.yaml / dataset_name)")
    ap.add_argument("--check", action="store_true", help="write nothing: print what the conversion would do")
    return ap


def print_report(report: dict) -> None:
    """The pre-check report (YAMLDrivenConverter.precheck) as text."""
    rows = [(f"{os.path.basename(d['file'])}::{d['demo']}", str(d["samples"]),
             f"{d['rate_hz']:.1f} Hz" if d["rate_hz"] else "-", d["recorded_with"] or "-",
             "converts" if d["ok"] else f"skipped: {d['reason']}") for d in report["demos"]]
    header = ("Demo", "Samples", "Rate", "Recorded with", "Result")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(4)]
    for row in [header, *rows]:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)) + "  " + row[4])
    if report["error"]:
        print(f"\nerror: {report['error']}")
    if report["output"]:
        print(f"\nOutput ({report['format']}):")
        width = max(len(name) for name, _ in report["output"])
        for name, detail in report["output"]:
            print(f"  {name.ljust(width)}  {detail}")
    print(f"\n{report['converted']} of {report['total']} demos would be converted")


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        parser, options = load_conversion_config(args.config, args.options)
    except ValueError as e:
        return _fail(str(e), EXIT_USAGE)
    if options.source:
        print(f"Using format options from {options.source}")

    try:
        inputs = collect_inputs(args.input)
    except FileNotFoundError as e:
        return _fail(f"input not found: {e}", EXIT_USAGE)
    if not inputs:
        return _fail("no .h5 files found in the given inputs", EXIT_USAGE)

    converter = YAMLDrivenConverter(
        PostprocessConfigLoader(), data_config=parser.yaml_data["config"]
    )
    converter.options = options

    if args.check:
        report = converter.precheck(args.format, inputs)
        print_report(report)
        return EXIT_OK if report["converted"] and not report["error"] else EXIT_FAILED
    if not args.output:
        return _fail("--output is required (unless --check)", EXIT_USAGE)

    output = os.path.abspath(args.output)
    if args.format == "lerobot":
        if os.path.exists(output) and (not os.path.isdir(output) or os.listdir(output)):
            return _fail(f"output must not exist or be an empty directory: {output}", EXIT_USAGE)
    else:
        if os.path.exists(output):
            return _fail(f"output already exists, refusing to overwrite: {output}", EXIT_USAGE)
        os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Converting {len(inputs)} file(s) to {args.format} -> {output}")
    if args.format == "lerobot":
        ok = converter.convert_to_lerobot(inputs, output, task=args.task)
    else:
        ok = converter.convert_multiple_files("robomimic", inputs, output)
    if not ok:
        return _fail(converter.last_error or "conversion failed (see the log above)", EXIT_FAILED)

    # The configs used and action_dispatcher/, like the GUI: next to the .h5, above the LeRobot folder.
    dest_dir = os.path.dirname(output)
    if os.path.exists(os.path.join(dest_dir, "acquisition_config.yaml")):
        print(f"warning: replacing the config copies / action_dispatcher files already in {dest_dir}", file=sys.stderr)
    try:
        write_dataset_files(dest_dir, parser, options.source)
    except Exception as e:
        print(f"warning: the dataset was converted, but some extra files (config copies / action_dispatcher files) "
              f"could not be written: {str(e) or type(e).__name__}", file=sys.stderr)

    print(converter.conversion_summary())
    print(f"Done: {output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
