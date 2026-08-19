#!/usr/bin/env python3
"""Print a read-only JSON audit of historical B1K authority failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.b1k_authority_audit import audit_authority_failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(
        audit_authority_failures(Path(args.root)),
        sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
