#!/usr/bin/env python3
"""Audit A2 public-input shortcuts without changing the QA artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import a2_shortcut_audit, gate_authority  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an out-of-band scene-clustered A2 shortcut audit")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-authority-manifest", type=Path, required=True,
        help="committed manifest authenticating source shards and gate")
    parser.add_argument(
        "--source-authority-root", type=Path, default=ROOT,
        help="trusted root for manifest-relative paths (default: repository)")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")
    try:
        source_authority = \
            gate_authority.resolve_preview_source_authority_from_manifest(
                args.source_authority_manifest,
                root=args.source_authority_root)
        report = a2_shortcut_audit.write_artifact_audit(
            args.benchmark, args.out,
            resamples=args.resamples, seed=args.seed,
            expected_source_authority=source_authority)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "item_count": report["item_count"],
        "scene_count": report["scene_count"],
        "blind_predictors": report["blind_predictors"],
        "max_statistic_cluster_bootstrap": report[
            "max_statistic_cluster_bootstrap"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
