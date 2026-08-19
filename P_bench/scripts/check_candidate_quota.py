#!/usr/bin/env python3
"""Decide whether compiled QA quotas require another collection shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import candidate_quota, config, io_utils  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--supported-task", required=True, action="append",
        help="task that must reach the quota; repeat for every supported task")
    parser.add_argument(
        "--min-total-items", type=int,
        default=config.BACKGROUND_MIN_TOTAL_ITEMS)
    parser.add_argument(
        "--min-per-task", type=int,
        default=config.BACKGROUND_MIN_ITEMS_PER_TASK)
    parser.add_argument(
        "--length", type=int, action="append",
        help="required action length; defaults to the frozen L1-L6 set")
    parser.add_argument(
        "--min-per-length", type=int,
        default=config.BACKGROUND_MIN_ITEMS_PER_LENGTH)
    parser.add_argument("--min-unique-frames", type=int)
    parser.add_argument("--min-scene-families", type=int)
    parser.add_argument("--max-scene-family-fraction", type=float)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dataset = str(args.dataset).strip().lower()
    default_frames = config.BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET.get(dataset)
    default_families = \
        config.BACKGROUND_MIN_SCENE_FAMILIES_BY_DATASET.get(dataset)
    default_family_fraction = \
        config.BACKGROUND_MAX_SCENE_FAMILY_FRACTION.get(dataset)
    if (default_frames is None or default_families is None or
            default_family_fraction is None):
        parser.error("dataset has no frozen background quota")
    try:
        report = candidate_quota.summarize_artifact(
            args.benchmark, dataset=args.dataset,
            supported_tasks=args.supported_task,
            min_total_items=args.min_total_items,
            min_per_task=args.min_per_task,
            required_lengths=(
                args.length if args.length is not None
                else config.GEN_LENGTHS),
            min_per_length=args.min_per_length,
            min_unique_frames=(
                args.min_unique_frames
                if args.min_unique_frames is not None else default_frames),
            min_scene_families=(
                args.min_scene_families
                if args.min_scene_families is not None else
                default_families),
            max_scene_family_fraction=(
                args.max_scene_family_fraction
                if args.max_scene_family_fraction is not None else
                default_family_fraction))
        io_utils.atomic_write_json(
            args.out, report, allow_nan=False, sort_keys=True)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "dataset": report["dataset"],
        "decision": report["decision"],
        "complete": report["complete"],
        "task_shortfall": {
            task_id: values["count_shortfall"]
            for task_id, values in report["tasks"].items()
            if values["count_shortfall"]},
        "length_shortfall": report["length_shortfall"],
        "report": str(args.out.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
