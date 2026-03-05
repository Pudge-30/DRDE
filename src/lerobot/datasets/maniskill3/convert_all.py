#!/usr/bin/env python

"""Batch-convert ManiSkill3 trajectories to LeRobot datasets.

This script automates the validated pipeline:
1) replay ManiSkill trajectories with RGB observations
2) convert replayed trajectory to LeRobot format

Example:
python -m lerobot.datasets.maniskill3.convert_all \
  --raw-root /home/kemove/.cache/maniskill3_raw \
  --output-root /home/kemove/datasets/lerobot/maniskill3 \
  --count 50 \
  --num-envs 1
"""

from __future__ import annotations

import argparse
import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert ManiSkill3 data to LeRobot format.")
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable used to run ManiSkill modules.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/kemove/.cache/maniskill3_raw"),
        help="Root directory containing raw ManiSkill3 trajectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/kemove/datasets/lerobot/maniskill3"),
        help="Root directory where converted LeRobot datasets will be written.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb",
        help="Replay observation mode.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Replay at most this many episodes per source trajectory (default: all).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel envs during replay.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for LeRobot conversion output.",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="640x480",
        help="Image size passed to convert_to_lerobot, e.g. 640x480.",
    )
    parser.add_argument(
        "--chunks-size",
        type=int,
        default=1000,
        help="Episodes per chunk for convert_to_lerobot.",
    )
    parser.add_argument(
        "--include-replayed",
        action="store_true",
        help="Also include already replayed *.rgb.*.h5 files as source inputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion when target output already contains meta/info.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue converting remaining files when one file fails.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only first N matched source files (for smoke test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned commands, do not execute.",
    )
    return parser.parse_args()


def _run(cmd: list[str], dry_run: bool) -> None:
    logging.info("$ %s", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _sanitize_dataset_name(relative_h5: Path) -> str:
    stem = relative_h5.with_suffix("").as_posix()
    stem = stem.replace("/", "__")
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)
    return f"maniskill3_{stem}".lower()


def _select_replay_output(source_h5: Path) -> Path | None:
    candidates = [p for p in source_h5.parent.glob("*.h5") if ".rgb." in p.name]
    if not candidates:
        return None

    source_tokens = set(source_h5.stem.split("."))

    def _score(path: Path) -> tuple[int, float]:
        tokens = set(path.stem.split("."))
        overlap = len(tokens.intersection(source_tokens))
        return (overlap, path.stat().st_mtime)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _guess_replay_output_for_dry_run(source_h5: Path, obs_mode: str) -> Path:
    return source_h5.with_name(f"{source_h5.stem}.{obs_mode}.<control_mode>.<backend>.h5")


def _iter_source_h5(raw_root: Path, include_replayed: bool) -> list[Path]:
    files = sorted(raw_root.rglob("*.h5"))
    if include_replayed:
        return files
    return [p for p in files if ".rgb." not in p.name]


def _task_name_from_path(raw_root: Path, source_h5: Path) -> str:
    rel = source_h5.relative_to(raw_root)
    return rel.parts[0] if rel.parts else "maniskill3_task"


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    source_files = _iter_source_h5(raw_root, include_replayed=args.include_replayed)
    if args.max_files is not None:
        source_files = source_files[: args.max_files]

    logging.info("Found %d source trajectory files.", len(source_files))
    if not source_files:
        return

    converted = 0
    skipped = 0
    failed = 0

    for idx, source_h5 in enumerate(source_files, start=1):
        rel = source_h5.relative_to(raw_root)
        task_name = _task_name_from_path(raw_root, source_h5)
        dataset_name = _sanitize_dataset_name(rel)
        output_dir = output_root / dataset_name
        info_file = output_dir / "meta" / "info.json"

        logging.info("[%d/%d] Processing: %s", idx, len(source_files), rel)

        if args.skip_existing and info_file.exists():
            logging.info("Skip existing converted dataset: %s", output_dir)
            skipped += 1
            continue

        replay_cmd = [
            args.python_exec,
            "-m",
            "mani_skill.trajectory.replay_trajectory",
            "--traj-path",
            str(source_h5),
            "--obs-mode",
            args.obs_mode,
            "--save-traj",
            "--use-env-states",
            "--num-envs",
            str(args.num_envs),
        ]
        if args.count is not None:
            replay_cmd.extend(["--count", str(args.count)])

        convert_cmd = None
        try:
            _run(replay_cmd, dry_run=args.dry_run)
            if args.dry_run:
                replay_h5 = _guess_replay_output_for_dry_run(source_h5, args.obs_mode)
            else:
                replay_h5 = _select_replay_output(source_h5)
                if replay_h5 is None:
                    raise FileNotFoundError(
                        f"Cannot find replayed .h5 with '.rgb.' in {source_h5.parent} after replay."
                    )

            convert_cmd = [
                args.python_exec,
                "-m",
                "mani_skill.trajectory.convert_to_lerobot",
                "--traj-path",
                str(replay_h5),
                "--output-dir",
                str(output_dir),
                "--task-name",
                task_name,
                "--fps",
                str(args.fps),
                "--image-size",
                args.image_size,
                "--chunks-size",
                str(args.chunks_size),
            ]
            _run(convert_cmd, dry_run=args.dry_run)
            converted += 1
        except Exception:
            failed += 1
            logging.exception("Failed on source trajectory: %s", source_h5)
            if convert_cmd is not None:
                logging.error("Last convert command: %s", shlex.join(convert_cmd))
            if not args.continue_on_error:
                raise

    logging.info(
        "Done. converted=%d, skipped=%d, failed=%d, total=%d",
        converted,
        skipped,
        failed,
        len(source_files),
    )


if __name__ == "__main__":
    main()
