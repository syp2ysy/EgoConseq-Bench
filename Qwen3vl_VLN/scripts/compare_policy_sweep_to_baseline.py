import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


KEYS = [
    "success_rate",
    "spl",
    "navigation_error",
    "oracle_success",
    "avg_num_steps",
    "avg_path_length",
    "episodes_ge_400_steps",
    "fallback_total",
]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def best_depth_metrics(depth_root: Path, split: str) -> tuple[str, Dict[str, Any], Path]:
    rows = []
    for metrics_path in depth_root.glob(f"*_{split}/final_metrics.json"):
        metrics = load_json(metrics_path)
        rows.append((float(metrics.get("success_rate", 0.0)), metrics_path.parent.name, metrics, metrics_path))
    if not rows:
        raise SystemExit(f"No depth final_metrics.json found under {depth_root}")
    rows.sort(reverse=True)
    _, name, metrics, path = rows[0]
    return name, metrics, path


def first_final_metrics(root: Path) -> tuple[Dict[str, Any], Path]:
    paths = sorted(root.glob("*/final_metrics.json"))
    if not paths:
        raise SystemExit(f"No baseline final_metrics.json found under {root}")
    return load_json(paths[0]), paths[0]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def diff(a: Any, b: Any) -> str:
    try:
        return f"{float(a) - float(b):+.4f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare best depth eval policy against baseline with same policy")
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-reference", default=None)
    parser.add_argument("--depth-reference", default=None)
    parser.add_argument("--split", default="val_unseen")
    args = parser.parse_args()

    depth_name, depth_metrics, depth_path = best_depth_metrics(Path(args.depth_root), args.split)
    baseline_metrics, baseline_path = first_final_metrics(Path(args.baseline_root))
    depth_ref = load_json(Path(args.depth_reference)) if args.depth_reference else None
    baseline_ref = load_json(Path(args.baseline_reference)) if args.baseline_reference else None

    lines = [
        "# Policy Transfer Comparison",
        "",
        f"- best_depth_variant: `{depth_name}`",
        f"- depth_metrics: `{depth_path}`",
        f"- baseline_same_policy_metrics: `{baseline_path}`",
        "",
        "| metric | depth best | baseline same policy | depth-baseline | depth d/ref | baseline d/ref |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in KEYS:
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    fmt(depth_metrics.get(key)),
                    fmt(baseline_metrics.get(key)),
                    diff(depth_metrics.get(key), baseline_metrics.get(key)),
                    diff(depth_metrics.get(key), depth_ref.get(key)) if depth_ref else "",
                    diff(baseline_metrics.get(key), baseline_ref.get(key)) if baseline_ref else "",
                ]
            )
            + " |"
        )
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
