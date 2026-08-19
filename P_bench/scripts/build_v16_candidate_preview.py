#!/usr/bin/env python3
"""Build an honest ABC preview from authenticated main records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import candidate_preview, gate_authority


def _sha256(value: str) -> str:
    if (len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise argparse.ArgumentTypeError(
            "SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", required=True, action="append", type=Path,
        help="authenticated main records.jsonl; repeat to merge real shards")
    parser.add_argument(
        "--run-meta-sha256", required=True, action="append", type=_sha256,
        help=("trusted digest of the adjacent run_meta.json; repeat in the "
              "same order as --records"))
    parser.add_argument("--output", required=True, type=Path,
        help="new candidate_qa output directory")
    parser.add_argument("--report", required=True, type=Path,
        help="new candidate_qa_report output directory")
    parser.add_argument(
        "--collection-funnel", type=Path,
        help="optional collection_funnel.json copied into report metadata")
    parser.add_argument(
        "--source-authority-manifest", type=Path,
        help=("committed freeze/golden manifest independently binding the "
              "records and run metadata"))
    parser.add_argument("--main-max-items-per-task", type=int,
        help="preview-only per-shard cap for ABC tasks")
    parser.add_argument("--a1-common-support", action=argparse.BooleanOptionalAction,
        help=("balance A1 labels within public action cells; defaults on for "
              "uniform proposal-v2 sources"))
    return parser


def resolve_build_authority(args, *, root=ROOT):
    """Adapt parsed CLI arguments to the reusable authority resolver."""
    return gate_authority.resolve_build_preview_source_authority(
        records_paths=args.records,
        expected_run_meta_sha256=args.run_meta_sha256,
        source_authority_manifest=args.source_authority_manifest, root=root)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        source_authority = resolve_build_authority(args)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    result = candidate_preview.build_main_preview(
        args.records, args.output, args.report,
        expected_source_authority=source_authority,
        run_meta_sha256=args.run_meta_sha256,
        collection_funnel_path=args.collection_funnel,
        main_max_items_per_task=args.main_max_items_per_task,
        a1_common_support=args.a1_common_support)
    print(json.dumps({
        "benchmark": str((args.output / "benchmark.json").resolve()),
        "report": str((args.report / "index.html").resolve()),
        "headline_eligible": result["headline_eligible"],
        "coverage": result["coverage"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
