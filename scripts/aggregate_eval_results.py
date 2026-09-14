import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vln_baseline.eval_metrics import write_aggregate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate split VLN eval stats")
    parser.add_argument("output_dir")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--expected-splits", type=int, default=None)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = write_aggregate_metrics(
        Path(args.output_dir),
        args.split,
        expected_splits=args.expected_splits,
        expected_episodes=args.expected_episodes,
    )
    print(json.dumps(metrics, indent=2))
    if args.require_valid and not metrics.get("distributed_valid"):
        print("distributed_valid=false", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
