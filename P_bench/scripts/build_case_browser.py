#!/usr/bin/env python3
"""Build one offline case browser from validated candidate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import candidate_preview, case_browser, gate_authority  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", type=_source, required=True,
        help="validated candidate artifact as LABEL=PATH; repeat to merge")
    parser.add_argument(
        "--source-authority-manifest", action="append", type=Path,
        required=True,
        help=("committed manifest authenticating each labeled source; "
              "repeat in the same order as --source"))
    parser.add_argument(
        "--source-authority-root", type=Path, default=ROOT,
        help="trusted root for manifest-relative paths (default: repository)")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="candidate_qa_report directory to create")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    labels = [label for label, _path in args.source]
    if len(set(labels)) != len(labels):
        raise SystemExit("case-browser source labels must be unique")
    if len(args.source_authority_manifest) != len(args.source):
        parser.error("one source authority manifest per source is required")
    sources = []
    for index, ((label, path), manifest) in enumerate(zip(
            args.source, args.source_authority_manifest), 1):
        started = time.perf_counter()
        print(
            f"[{index}/{len(args.source)}] validating {label}: {path}",
            file=sys.stderr, flush=True)
        source_authority = \
            gate_authority.resolve_preview_source_authority_from_manifest(
                manifest, root=args.source_authority_root)
        benchmark = candidate_preview.validate_preview_artifact(
            path, expected_source_authority=source_authority)
        source = case_browser.browser_source_from_validated_artifact(
            label, path, benchmark)
        sources.append(source)
        print(
            f"[{index}/{len(args.source)}] validated {label}: "
            f"{len(source.benchmark['cases'])} cases in "
            f"{time.perf_counter() - started:.1f}s",
            file=sys.stderr, flush=True)
    print("rendering static case browser", file=sys.stderr, flush=True)
    output = case_browser.render_case_browser(sources, args.output)
    coverage = {}
    for source in sources:
        for task_id, count in source.benchmark["coverage"].items():
            coverage[task_id] = coverage.get(task_id, 0) + int(count)
    print(json.dumps({
        "report": str(output.resolve()),
        "case_count": sum(len(source.benchmark["cases"])
                          for source in sources),
        "coverage": coverage,
        "headline_eligible": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
