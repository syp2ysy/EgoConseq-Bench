#!/usr/bin/env python3
"""Periodically log distributed eval progress until final_metrics.json appears."""

import argparse
import datetime as dt
import json
import time
from pathlib import Path


def count_split(result_dir: Path, split_name: str) -> int:
    log_dir = result_dir / split_name / "log"
    if not log_dir.exists():
        return 0
    return sum(1 for _ in log_dir.glob("stats_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval_seconds", type=int, default=1200)
    args = parser.parse_args()

    expected = {"split_0": 460, "split_1": 460, "split_2": 460, "split_3": 459}
    args.log.parent.mkdir(parents=True, exist_ok=True)

    while True:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        final_path = args.result_dir / "final_metrics.json"
        with args.log.open("a") as f:
            if final_path.exists() and final_path.stat().st_size > 0:
                f.write(f"[{timestamp}] final_metrics ready: {final_path}\n")
                with final_path.open() as metrics_file:
                    metrics = json.load(metrics_file)
                f.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
                return

            total = 0
            parts = []
            for split_name, split_expected in expected.items():
                count = count_split(args.result_dir, split_name)
                total += count
                parts.append(f"{split_name}={count}/{split_expected}")
            f.write(f"[{timestamp}] {' '.join(parts)} total={total}/1839\n")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
