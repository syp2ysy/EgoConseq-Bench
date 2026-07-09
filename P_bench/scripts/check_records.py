"""Validator CLI: check invariants V1-V8 over a records.jsonl (exit 1 on violation)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--max-print", type=int, default=40)
    args = ap.parse_args()

    total, viol = validate.validate_file(args.records)
    print(f"checked {total} records; {len(viol)} violations")
    for v in viol[:args.max_print]:
        print("  ", v)
    if len(viol) > args.max_print:
        print(f"  ... and {len(viol) - args.max_print} more")
    sys.exit(1 if viol else 0)


if __name__ == "__main__":
    main()
