#!/usr/bin/env python3
"""Write the five-model LaTeX table only from complete, comparable evaluations."""

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    "qwen3vl-4b": "Qwen3-VL-4B",
    "robobrain25-4b": "RoboBrain2.5-4B",
    "robointer-3b": "RoboInter-3B",
    "rynnbrain11-2b_thinking_off": "RynnBrain1.1-2B",
    "cosmos3-edge_thinking_off": "Cosmos3-Edge",
}
METRICS = {
    "A1": ("accuracy", "collision_recall"),
    "A2": ("accuracy",), "A3": ("accuracy",),
    "A4": ("joint_accuracy", "axis_accuracy"),
    "B1": ("accuracy_at_0.25m", "accuracy_at_0.5m"),
    "B2": ("joint_accuracy", "axis_accuracy"),
    "B3": ("accuracy_at_0.25m", "accuracy_at_0.5m"),
    "C1": ("accuracy",),
}


def percentage(value):
    if value is None or not 0 <= value <= 1:
        raise ValueError("Table requires a completed numeric metric in [0, 1]")
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent)
    results = parser.parse_args().benchmark_root / "evaluation/results"
    reference, counts, rows = None, None, []
    for model, name in MODELS.items():
        path = results / model / "full__extract-Qwen3-8B_scores.json"
        if not path.exists():
            raise SystemExit(f"Evaluation not available yet: {model}")
        data = json.loads(path.read_text())
        config = {k: v for k, v in data["config"].items() if k != "responses_sha256"}
        if reference is not None and config != reference:
            raise SystemExit(f"Mixed evaluation protocols/inputs: {model}")
        reference = config
        if not data["summary"]["all"]["complete"]:
            raise SystemExit(f"Evaluation incomplete: {model}; final table was not written")
        splits = data["summary"]["by_split"]
        totals = {s: {t: splits[s]["tasks"][t]["total"] for t in METRICS} for s in ("seen", "unseen")}
        if counts is not None and totals != counts:
            raise SystemExit(f"Different question counts: {model}")
        counts = totals
        cells = ["/".join(percentage(splits[s]["tasks"][t][metric]) for metric in metrics)
                 for s in ("seen", "unseen") for t, metrics in METRICS.items()]
        rows.append(" & ".join([name, *cells]) + r" \\")
    seen, unseen = (sum(counts[s].values()) for s in ("seen", "unseen"))
    header = " & ".join(["Model", *METRICS, *METRICS]) + r" \\"
    lines = [r"% Requires booktabs and graphicx. All values are percentages.",
             r"\begin{table*}[t]", r"\centering", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l*{16}{c}}", r"\toprule",
             rf" & \multicolumn{{8}}{{c}}{{Seen ({seen:,})}} & \multicolumn{{8}}{{c}}{{Unseen ({unseen:,})}} \\",
             r"\cmidrule(lr){2-9}\cmidrule(lr){10-17}", header, r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}}",
             r"\caption{Results under one evaluation protocol. A1: accuracy/collision recall; "
             r"A2, A3, C1: accuracy; A4, B2: joint/axis accuracy; "
             r"B1, B3: accuracy within $\pm0.25$/$\pm0.5$ m. All values are percentages.}",
             r"\label{tab:five-models}", r"\end{table*}"]
    path = results / "five_models_seen_unseen.tex"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
