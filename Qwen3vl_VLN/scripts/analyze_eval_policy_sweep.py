import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_KEYS = [
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


def variant_name(path: Path, model_tag: str, split: str) -> str:
    name = path.parent.name
    prefix = f"{model_tag}_"
    suffix = f"_{split}"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def collect(output_root: Path, model_tag: str, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = f"{model_tag}_*_{split}/final_metrics.json"
    for metrics_path in sorted(output_root.glob(pattern)):
        metrics = load_json(metrics_path)
        row = {"variant": variant_name(metrics_path, model_tag, split), "path": str(metrics_path.parent)}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("success_rate") or 0.0), reverse=True)
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta(value: Any, ref: Any) -> str:
    if value is None or ref is None:
        return ""
    try:
        return f"{float(value) - float(ref):+.4f}"
    except (TypeError, ValueError):
        return ""


def best_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return rows[0] if rows else None


def markdown_table(rows: List[Dict[str, Any]], reference: Optional[Dict[str, Any]]) -> str:
    headers = [
        "rank",
        "variant",
        "SR",
        "dSR",
        "SPL",
        "dSPL",
        "NE",
        "dNE",
        "OSR",
        "steps",
        ">=400",
        "fallback",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["variant"]),
                    fmt(row.get("success_rate")),
                    delta(row.get("success_rate"), reference.get("success_rate") if reference else None),
                    fmt(row.get("spl")),
                    delta(row.get("spl"), reference.get("spl") if reference else None),
                    fmt(row.get("navigation_error")),
                    delta(row.get("navigation_error"), reference.get("navigation_error") if reference else None),
                    fmt(row.get("oracle_success")),
                    fmt(row.get("avg_num_steps"), digits=2),
                    fmt(row.get("episodes_ge_400_steps"), digits=0),
                    fmt(row.get("fallback_total"), digits=0),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze eval policy sweep results")
    parser.add_argument("output_root")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--reference-final", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = collect(output_root, args.model_tag, args.split)
    reference = load_json(Path(args.reference_final)) if args.reference_final else None
    best = best_row(rows)

    report = [
        f"# Eval Policy Sweep: {args.model_tag}",
        "",
        f"- output_root: `{output_root}`",
        f"- split: `{args.split}`",
        f"- variants_found: {len(rows)}",
    ]
    if best:
        report.append(f"- best_variant: `{best['variant']}`")
        report.append(f"- best_success_rate: {fmt(best.get('success_rate'))}")
    report.extend(["", markdown_table(rows, reference), ""])
    print("\n".join(report))


if __name__ == "__main__":
    main()
