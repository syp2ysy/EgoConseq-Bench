#!/usr/bin/env python3
"""Audit A1 blind and B2 rotation-only baselines without touching the artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import blind_baseline_audit, gate_authority  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an out-of-band A1 / B2 blind baseline audit")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source-authority-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-authority-root", type=Path, default=ROOT,
        help="trusted root for manifest-relative paths (default: repository)")
    args = parser.parse_args()
    try:
        authority = gate_authority.resolve_preview_source_authority_from_manifest(
            args.source_authority_manifest, root=args.source_authority_root)
        report = blind_baseline_audit.write_artifact_audit(
            args.benchmark, args.out,
            expected_source_authority=authority)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "a1_blind": {
            "item_count": report["a1_blind"]["item_count"],
            "blind_baselines": report["a1_blind"]["blind_baselines"],
        },
        "a1_blind_by_source": report["a1_blind_by_source"],
        "b2_rotation": {
            "item_count": report["b2_rotation"]["item_count"],
            "rotation_only_frontal_prior": (
                report["b2_rotation"]["rotation_only_frontal_prior"]),
            "precise_bearing_crosscheck": (
                report["b2_rotation"]["precise_bearing_crosscheck"]),
            "blind_baselines": report["b2_rotation"]["blind_baselines"],
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
