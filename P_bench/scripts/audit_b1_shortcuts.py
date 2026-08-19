#!/usr/bin/env python3
"""Audit B1 numeric-rank shortcuts without changing the QA artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import qa_shortcut_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an out-of-band B1 numeric-rank shortcut audit")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = qa_shortcut_audit.write_artifact_audit(
            args.benchmark, args.out)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "item_count": report["item_count"],
        "full_numeric_rank_support": report["full_numeric_rank_support"],
        "blind_baselines": report["blind_baselines"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
