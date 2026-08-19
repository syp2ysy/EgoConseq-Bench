#!/usr/bin/env python3
"""Bundle multiple static dataset browsers into one review page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import dataset_case_browser  # noqa: E402


def _dataset(value: str) -> dataset_case_browser.DatasetReport:
    try:
        descriptor, raw_path = value.split("=", 1)
        dataset_id, status, label = descriptor.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "dataset must be ID:STATUS:LABEL=INDEX_HTML") from error
    if not raw_path or not label:
        raise argparse.ArgumentTypeError(
            "dataset must be ID:STATUS:LABEL=INDEX_HTML")
    try:
        return dataset_case_browser.DatasetReport(
            dataset_id=dataset_id, label=label,
            report_path=Path(raw_path), status=status)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", type=_dataset, required=True,
        help=("dataset browser as ID:STATUS:LABEL=INDEX_HTML; statuses are "
              "candidate, pilot, or calibration_only"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = dataset_case_browser.merge_dataset_reports(
        args.dataset, args.output)
    payload = dataset_case_browser.load_browser_payload(output)
    print(json.dumps({
        "report": str(output.resolve()),
        "case_count": len(payload["cases"]),
        "coverage": payload["coverage"],
        "datasets": payload.get("datasets"),
        "headline_eligible": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

