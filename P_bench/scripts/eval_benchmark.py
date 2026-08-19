"""Evaluate the R2R ABC candidate benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import candidate_preview, gate_authority
from pipeline.io_utils import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--source-authority-manifest", required=True, type=Path,
        help="committed manifest authenticating source shards and gate")
    parser.add_argument(
        "--source-authority-root", type=Path, default=ROOT,
        help="trusted root for manifest-relative paths (default: repository)")
    parser.add_argument("--gt-as-pred", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="accepted for CLI parity; v16 candidate artifacts are always "
             "non-headline")
    args = parser.parse_args(argv)
    predictions = None
    if args.predictions is not None:
        rows = read_jsonl(args.predictions, require_dict=True)
        predictions = {}
        for row in rows:
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("candidate prediction ID is invalid")
            if item_id in predictions:
                raise ValueError("candidate prediction IDs are not unique")
            predictions[item_id] = row.get("answer")
    source_authority = \
        gate_authority.resolve_preview_source_authority_from_manifest(
            args.source_authority_manifest, root=args.source_authority_root)
    report = candidate_preview.evaluate_preview_artifact(
        args.benchmark, predictions=predictions,
        gt_as_pred=args.gt_as_pred, allow_partial=args.allow_partial,
        expected_source_authority=source_authority)
    print(json.dumps(report, indent=2, sort_keys=True))
if __name__ == "__main__":
    main(sys.argv[1:])
