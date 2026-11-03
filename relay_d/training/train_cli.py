"""Start, list and stop training runs without the GUI.

    relay-d-train --folder ~/recordings/2026-10-23_10-00-00 --epochs 200
    relay-d-train --folder <folder> --format lerobot --policy act --steps 50000 --python ~/venvs/lerobot/bin/python
    relay-d-train --folder <folder> --list
    relay-d-train --folder <folder> --stop robomimic_20261023_101500

``--folder`` is the Post-process output folder (or the dataset itself). A run opens in its own terminal
window and keeps going after this command exits; use ``--no-terminal`` for a headless run (log in the run
folder). ``--dry-run`` prints the command and writes nothing.

Exit codes: 0 success, 1 the run could not be started, 2 bad arguments / no dataset found.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from relay_d.training import envsetup, launcher, runs
from relay_d.utils import user_settings

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="relay-d-train", description="Start imitation-learning training on a dataset.")
    ap.add_argument("--folder", required=True, help="Post-process output folder, or the dataset itself")
    ap.add_argument("--format", choices=["auto", "robomimic", "lerobot"], default="auto")
    ap.add_argument("--policy", help=f"robomimic: {launcher.ROBOMIMIC_POLICIES} (default: the folder's config); "
                                     f"lerobot: {launcher.LEROBOT_POLICIES} (default: act)")
    ap.add_argument("--python", help="interpreter with robomimic / lerobot installed (default: the python_<format> setting, else this one)")
    ap.add_argument("--epochs", type=int, help="robomimic: number of epochs")
    ap.add_argument("--steps", type=int, help="lerobot: number of training steps")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float, help="learning rate")
    ap.add_argument("--seq-length", type=int, help="robomimic: sequence length")
    ap.add_argument("--save-every", type=int, help="checkpoint every N epochs (robomimic) / steps (lerobot)")
    ap.add_argument("--num-workers", type=int, help="lerobot: data loader workers")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", choices=["cuda", "cpu"], help="default: the train_default_device setting")
    ap.add_argument("--extra", default="", help="extra arguments, appended to the training command as written; write --extra=\"--debug\" (with the =) if they start with --")
    ap.add_argument("--dry-run", action="store_true", help="print the command, write nothing, start nothing")
    ap.add_argument("--no-terminal", action="store_true", help="run headless (no terminal window); default: the train_use_terminal setting")
    ap.add_argument("--list", action="store_true", help="list the runs of --folder")
    ap.add_argument("--stop", metavar="RUN", help="stop a run (its folder name or path)")
    return ap


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _list(folder: str) -> int:
    found = runs.list_runs(folder)
    for r in found:
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.meta.get("started", 0)))
        print(f"{r.path.name:32} {r.status:9} {started}  {r.path}")
    if not found:
        print("no runs")
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return _list(args.folder)
    if args.stop:
        run_dir = Path(args.stop)
        if not run_dir.is_dir():
            run_dir = runs.runs_root(args.folder) / args.stop
        if not (run_dir / "run.json").exists():
            return _fail(f"no such run: {args.stop}", EXIT_USAGE)
        ok = runs.stop(run_dir)
        print(f"{run_dir.name}: {'stopped' if ok else 'could not be stopped'}")
        return EXIT_OK if ok else EXIT_FAILED

    path = Path(args.folder)
    if not path.exists():
        return _fail(f"not found: {path}", EXIT_USAGE)
    datasets = [d for d in launcher.find_datasets(path) if args.format in ("auto", d.format)]
    if not datasets:
        return _fail(f"no {'training ' if args.format == 'auto' else args.format + ' '}dataset found in {path}", EXIT_USAGE)
    dataset = datasets[0]
    if len(datasets) > 1:
        print(f"{len(datasets)} datasets found, using the newest: {dataset.path}", file=sys.stderr)

    if dataset.format == launcher.LEROBOT and (args.epochs or args.seq_length):
        return _fail("--epochs / --seq-length are robomimic options (LeRobot uses --steps)", EXIT_USAGE)
    if dataset.format == launcher.ROBOMIMIC and (args.steps or args.num_workers):
        return _fail("--steps / --num-workers are LeRobot options (robomimic uses --epochs)", EXIT_USAGE)

    chosen = {"epochs": args.epochs, "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
              "seq_length": args.seq_length, "save_every": args.save_every, "num_workers": args.num_workers,
              "seed": args.seed, "device": args.device or user_settings.get("train_default_device")}
    python = args.python or envsetup.python_for(dataset.format)  # setting, else the environment built by the Train tab
    job = launcher.TrainJob(dataset, python, args.policy, {k: v for k, v in chosen.items() if v is not None},
                            args.extra)
    try:
        if args.dry_run:
            print(" ".join(launcher.build_command(job, launcher.new_run_dir(dataset.base, dataset.format), write=False)))
            return EXIT_OK
        run_dir = launcher.start(job, terminal=user_settings.get("train_use_terminal") and not args.no_terminal)
    except launcher.TrainingError as e:
        return _fail(str(e), EXIT_USAGE)

    print(f"Started {run_dir.name}\n  log: {run_dir / 'train.log'}\n  stop: relay-d-train --folder {args.folder} --stop {run_dir.name}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
