#!/usr/bin/env python3
"""Compile the non-headline A4 checkpoint-direction diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import checkpoint_direction, dataset_case_browser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", required=True, type=Path,
        help="candidate run containing records/**/records.jsonl")
    parser.add_argument(
        "--output", type=Path,
        help=("output directory; defaults to candidate_qa/diagnostics/"
              "checkpoint_direction.v1 below the run"))
    parser.add_argument(
        "--update-report", type=Path,
        help="existing dataset-atlas index.html to update with A4 cases")
    args = parser.parse_args(argv)
    output = args.output or (
        args.run_root / "candidate_qa" / "diagnostics" /
        checkpoint_direction.SCHEMA_VERSION)
    result = checkpoint_direction.build_run_artifact(args.run_root, output)
    report = None
    if args.update_report is not None:
        report = dataset_case_browser.add_checkpoint_direction_diagnostic(
            args.update_report, output)
    print(json.dumps({
        "artifact": str(Path(result["artifact"]).resolve()),
        "headline_eligible": result["headline_eligible"],
        "record_count": result["record_count"],
        "item_count": result["item_count"],
        "report": None if report is None else str(report.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
