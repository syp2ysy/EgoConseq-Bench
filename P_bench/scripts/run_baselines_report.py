"""Run non-vision baselines on the O5 manifest and write report.md.

O5 now uses one robot per question.  Counterfactual sensitivity is measured
across cases sharing a group_id.

Usage:
    python scripts/run_baselines_report.py [--manifest path] [--out path]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq.eval.baselines import evaluate_all_baselines
from egoconseq.manifest import Case, read_jsonl


def cases_to_o5_groups(cases: list[Case]) -> list[list[dict]]:
    """Convert manifest Cases into the grouped format expected by o5_metrics."""
    grouped: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        grouped[case.group_id or case.case_id].append(case)

    out: list[list[dict]] = []
    for group_id in sorted(grouped):
        group_cases = sorted(
            grouped[group_id],
            key=lambda c: float(c.body.get("radius_m", 0.0) if c.body else 0.0),
        )
        out_group = []
        for case in group_cases:
            label = case.answer.get("label", "INVALID") if case.answer else "INVALID"
            out_group.append({
                "case_id": case.case_id,
                "group_id": group_id,
                "radius_m": float(case.body.get("radius_m", 0.0) if case.body else 0.0),
                "gt": label,
                "pred": label,
            })
        out.append(out_group)
    return out


def summarize_cases(cases: list[Case]) -> dict:
    """Return label, group-kind, and scene counts for the report."""
    case_label_counts = Counter(
        c.answer.get("label", "INVALID") if c.answer else "INVALID"
        for c in cases
    )
    scenes = Counter(c.scene_id or "unknown" for c in cases)

    group_kind_by_id = {}
    for case in cases:
        group_id = case.group_id or case.case_id
        if group_id not in group_kind_by_id:
            group_kind_by_id[group_id] = (
                case.tags.get("group_kind", "unknown") if case.tags else "unknown"
            )

    return {
        "case_label_counts": case_label_counts,
        "group_kind_counts": Counter(group_kind_by_id.values()),
        "scene_counts": scenes,
        "n_cases": len(cases),
        "n_groups": len(group_kind_by_id),
    }


def fmt(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "NaN"
    return f"{v:.3f}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    manifest_path = args.manifest or os.path.join(
        _REPO_ROOT, "data", "demo", "o5", "manifest.jsonl"
    )
    out_path = args.out or os.path.join(
        _REPO_ROOT, "data", "demo", "o5", "report.md"
    )

    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    cases = read_jsonl(manifest_path)
    groups = cases_to_o5_groups(cases)
    summary = summarize_cases(cases)
    results = evaluate_all_baselines(groups, seed=42)

    label_counts = summary["case_label_counts"]
    group_kind_counts = summary["group_kind_counts"]
    n_cases = summary["n_cases"]
    n_groups = summary["n_groups"]
    n_flip = group_kind_counts.get("flip", 0)

    lines: list[str] = []
    lines.append("# EgoConseq-Bench O5 Report")
    lines.append("")
    lines.append("## Schema")
    lines.append("")
    lines.append("- Each question describes one cylindrical-chassis robot.")
    lines.append("- The prompt varies only chassis diameter; height is fixed to the Habitat default agent and is not mentioned.")
    lines.append("- Labels are binary: `contact` / `no_contact`.")
    lines.append("- Counterfactual flips are measured across cases sharing `group_id`.")
    lines.append("")

    lines.append("## Manifest Summary")
    lines.append("")
    lines.append(f"- Cases: **{n_cases}**")
    lines.append(f"- Groups: **{n_groups}**")
    for label in ("contact", "no_contact", "INVALID"):
        count = label_counts.get(label, 0)
        if count:
            pct = count / n_cases * 100 if n_cases else 0.0
            lines.append(f"- `{label}` cases: **{count}** ({pct:.1f}%)")
    for kind in ("flip", "all_no_contact", "all_contact", "unknown"):
        count = group_kind_counts.get(kind, 0)
        if count:
            pct = count / n_groups * 100 if n_groups else 0.0
            lines.append(f"- `{kind}` groups: **{count}** ({pct:.1f}%)")
    lines.append("")

    lines.append("## Baseline Metrics")
    lines.append("")
    lines.append("| Baseline | case_accuracy | false_safe_rate | flip_groups | correct_flip_rate | invariance_error |")
    lines.append("|----------|---------------|-----------------|-------------|-------------------|------------------|")
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {fmt(metrics['case_accuracy'])} | {fmt(metrics['false_safe_rate'])} "
            f"| {metrics['flip_groups']} | {fmt(metrics['correct_flip_rate'])} "
            f"| {fmt(metrics['invariance_error'])} |"
        )
    lines.append("")

    lines.append("## Gate 1")
    lines.append("")
    gate_flip = n_flip >= 30
    random_invariance = results["random"]["invariance_error"]
    blind_invariance = results["blind_text_only"]["invariance_error"]
    gate_blind_fails = (
        not math.isnan(float(blind_invariance))
        and float(blind_invariance) >= 0.9
    )
    lines.append(
        f"- Real GT-flip set exists: **{'PASS' if gate_flip else 'FAIL'}** "
        f"({n_flip} flip groups; target >= 30)."
    )
    lines.append(
        f"- Blind baseline fails to flip: **{'PASS' if gate_blind_fails else 'FAIL'}** "
        f"(blind invariance_error={fmt(blind_invariance)}; random={fmt(random_invariance)})."
    )
    lines.append("- Human QA: **PENDING** until `data/demo/o5/review.html` is inspected.")
    lines.append("")
    lines.append(
        "Note: `radius_only` is a diagnostic, not a validity gate. It can look strong "
        "on true flip groups because it hard-codes wide=contact/narrow=no_contact, "
        "but it should fail on all-contact and all-no-contact control groups."
    )

    report_text = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report_text + "\n", encoding="utf-8")

    print(f"[report] Written: {out_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
