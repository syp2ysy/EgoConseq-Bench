#!/usr/bin/env python3
"""Publish complete evaluations as a compact comparison JSON and plain LaTeX."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from post_QA import answers, templates

def percentage(value):
    """An exact Fraction as a percentage, rounded half up to two decimals."""
    return float((Decimal(value.numerator) * 100 / value.denominator).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def mean(rows, field="score"):
    return Fraction(sum(Fraction(str(row[field])) for row in rows), len(rows))


def _counter(rows, field):
    return dict(sorted(Counter(row.get(field, "unknown") for row in rows).items()))


def split_metrics(rows):
    """Every question stays in the denominator; any unscored question blocks the metrics."""
    tasks = {task: [r for r in rows if r["task_id"] == task] for task in templates.TASK_IDS}
    if any(r["status"] != "scored" for r in rows) or not all(tasks.values()):
        raise ValueError("Final results require every question and all eight tasks to be scored")
    diagnostics = {field: _counter(rows, field) for field in
                   ("status", "answer_status", "response_finish_reason", "extraction_method")}
    diagnostics['execution_status'] = diagnostics.pop('status')
    return {
        "questions": len(rows),
        "diagnostics": diagnostics,
        "collision_occurrence": {"accuracy": percentage(mean(tasks["A1"])),
                                 "recall": percentage(mean([r for r in tasks["A1"] if r["gt_collision"]]))},
        "collision_action_grounding": percentage(mean(tasks["A2"])),
        "collision_object_category": percentage(mean(tasks["A3"])),
        "target_distance": {moment: {"count": len(tasks[task]), "mra": percentage(mean(tasks[task]))}
                            for moment, task in (("intermediate", "B3"), ("final", "B1"))},
        "target_direction": {
            moment: {"count": len(tasks[task]), "joint_accuracy": percentage(mean(tasks[task])),
                     **{f"{axis}_accuracy": percentage(Fraction(sum(r["axis_correct"][axis] for r in tasks[task]),
                                                                len(tasks[task])))
                        for axis in answers.DIRECTION_AXES}}
            for moment, task in (("intermediate", "A4"), ("final", "B2"))},
        "future_view_selection": percentage(mean(tasks["C1"])),
        "average": percentage(sum(mean(group) for group in tasks.values()) / len(tasks)),
        "unit_accuracy": _unit_accuracy(rows),
    }


def _unit_accuracy(rows):
    units = {unit: [r for r in rows if r.get("unit") == unit] for unit in ("m", "cm")}
    return {unit: percentage(mean(group)) for unit, group in units.items() if group}


def pair_metrics(rows):
    """A minimal-pair set (two items differing only by the radius): item and pair accuracy."""
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    if any(len(group) != 2 for group in pairs.values()) or any(r["status"] != "scored" for r in rows):
        raise ValueError("Pair results need both scored members of every pair")
    return {"questions": len(rows), "pairs": len(pairs), "accuracy": percentage(mean(rows)),
            "pair_accuracy": percentage(Fraction(sum(all(r["correct"] for r in g) for g in pairs.values()), len(pairs))),
            "unit_accuracy": _unit_accuracy(rows)}


def _question_identity(rows, model_id):
    identity = {}
    for row in rows:
        key = row["split"], row["id"]
        if key in identity:
            raise ValueError(f"Duplicate question key for {model_id}: {key[0]}/{key[1]}")
        identity[key] = tuple(row.get(field) for field in ("task_id", "dataset", "ground_truth", "unit", "pair_id"))
    return identity


def build_report(models, config):
    report = {"status": "complete", "metric_unit": "percent", "models": []}
    reference = pair_reference = None
    source_references = {}
    for model in models:
        rows = model["results"]
        paired = {r["split"] for r in rows if "pair_id" in r}
        # Every model and ablation answers the same main sets; the radius pairs are a full-configuration
        # diagnostic, compared among the models that answered them.
        identity = _question_identity([r for r in rows if r["split"] not in paired], model["id"])
        if reference is not None and identity != reference:
            raise ValueError("Comparison models must have the same question identity")
        reference = identity
        if paired:
            identity = _question_identity([r for r in rows if r["split"] in paired], model["id"])
            if pair_reference is not None and identity != pair_reference:
                raise ValueError("Comparison models must have the same radius-pair identity")
            pair_reference = identity
        sources = model['sources']
        if source_references.setdefault(model['variant'], sources) != sources:
            raise ValueError('Comparison models with the same input variant must use the same QA sources')
        result = {"id": model["id"], "model": _paper_name(model), "variant": model["variant"]}
        splits = sorted({r["split"] for r in rows})
        result["main_splits"] = [s for s in splits if s not in paired]
        if result["main_splits"]:  # a minimal-pair set alone has no eight-task summary
            result["all"] = split_metrics([r for r in rows if r["split"] not in paired])
        for split in splits:
            group = [r for r in rows if r["split"] == split]
            result[split] = pair_metrics(group) if split in paired else split_metrics(group)
        report["models"].append(result)
    if not report["models"]:
        raise ValueError("No models to publish")
    report["config"] = config
    return report


PARAMETERS = {"full": "Full", "no_fov": "w/o FOV", "no_height": "w/o height",
              "no_radius": "w/o radius", "no_parameters": "w/o all"}


def _paper_name(model):
    # Checkpoint locations belong in provenance, not in the paper's method column.
    checkpoint = Path(model['model'])
    name = checkpoint.parent.parent.name if checkpoint.parent.name == 'snapshots' else checkpoint.name
    return name.rsplit('--', 1)[-1]


def _latex(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in text)


def latex_table(report, *, ablation=False):
    models = sorted(report['models'], key=lambda m: (m['model'].casefold(), list(PARAMETERS).index(m['variant'])))
    title = 'Parameter ablations' if ablation else 'Main comparison'
    headers = ('Method', 'Eval inputs') if ablation else ('Method',)

    lines = [r"\documentclass{article}", r"\usepackage[a4paper,margin=1.5cm]{geometry}",
             r"\usepackage{booktabs,graphicx}", r"\pagenumbering{gobble}", r"\begin{document}",
             r"\noindent All values are percentages; higher is better. Int.: intermediate state. "
             r"Recall treats collision as positive. Distance columns use mean relative accuracy (MRA); "
             r"direction columns use joint accuracy across the three axes."]
    groups = ((r"\shortstack{Collision\\Occurrence}", 2),
              (r"\shortstack{Collision\\Action\\Grounding}", 1),
              (r"\shortstack{Collision\\Object\\Category}", 1),
              (r"\shortstack{Target\\Distance}", 2),
              (r"\shortstack{Target\\Direction}", 2),
              (r"\shortstack{Future-View\\Selection}", 1), (r"\textbf{Avg.}", 1))
    metrics = ("Acc.", "Recall", "Acc.", "Acc.", r"\shortstack{Int.\\MRA}", r"\shortstack{Final\\MRA}",
               r"\shortstack{Int.\\Joint}", r"\shortstack{Final\\Joint}", "Acc.", "")
    main_splits = models[0]["main_splits"]
    for split in [*main_splits, "all"]:
        if split != main_splits[0]:
            lines.append(r"\clearpage")
        count = models[0][split]["questions"]
        lines += [r"\begin{table}[!ht]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
                  r"\renewcommand{\arraystretch}{1.15}",
                  rf"\caption{{{title}: {split.title()} ({count:,} questions per model).}}",
                  r"\resizebox{\linewidth}{!}{%",
                  rf"\begin{{tabular}}{{@{{}}{'l' * len(headers)}*{{9}}{{r}}@{{\hspace{{6pt}}}}r@{{}}}}", r"\toprule",
                  " & ".join([*headers, *[rf"\multicolumn{{{span}}}{{c}}{{{label}}}"
                                         for label, span in groups]]) + r" \\"]
        start, rules = len(headers) + 1, []
        for _, span in groups:
            rules.append(rf"\cmidrule(lr){{{start}-{start + span - 1}}}")
            start += span
        lines += [" ".join(rules), " & ".join([*[""] * len(headers), *metrics]) + r" \\", r"\midrule"]
        for index, model in enumerate(models):
            if ablation and index and model["model"] != models[index - 1]["model"]:
                lines.append(r"\midrule")
            labels = (model["model"], PARAMETERS[model["variant"]]) if ablation else (model["model"],)
            s = model[split]
            values = [s["collision_occurrence"]["accuracy"], s["collision_occurrence"]["recall"],
                      s["collision_action_grounding"], s["collision_object_category"],
                      *[s["target_distance"][moment]["mra"] for moment in ("intermediate", "final")],
                      s["target_direction"]["intermediate"]["joint_accuracy"],
                      s["target_direction"]["final"]["joint_accuracy"], s["future_view_selection"]]
            cells = [_latex(label) for label in labels]
            lines.append(" & ".join([*cells, *[f"{value:.2f}" for value in values],
                                      rf"\textbf{{{s['average']:.2f}}}"]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
                  r"\begin{minipage}{\linewidth}\footnotesize "
                  r"Avg. equally weights eight tasks. Distance uses VSI-Bench MRA (thresholds 0.50:0.95:0.05). "
                  r"Intermediate/final distance and direction are separate tasks; direction uses joint accuracy. "
                  r"Recall is excluded.",
                  r"\end{minipage}", r"\end{table}"]
    return "\n".join([*lines, r"\end{document}"]) + "\n"


def write_latex_tables(report, directory):
    if not report["models"][0]["main_splits"]:  # the LaTeX tables summarise main sets only
        return
    main, ablations = [], []
    for model in report["models"]:
        is_ablation = model['variant'] != 'full'
        (ablations if is_ablation else main).append(model)
    if ablations:
        ablations += [model for model in main if model['model'] in {m['model'] for m in ablations}]
    for name, models in (("results", main), ("ablation", ablations)):
        path = directory / f"{name}.tex"
        if models:
            path.write_text(latex_table({**report, "models": models}, ablation=name == "ablation"))
        else:
            path.unlink(missing_ok=True)


def _relative_symlink(source, staging_path, final_parent):
    source = source.resolve(strict=True)
    staging_path.symlink_to(os.path.relpath(source, start=final_parent))


def write_model_scores(report, directory):
    """Keep each model's summary and provenance alongside its question-level details."""
    scores = directory / "scores"
    scores.mkdir(exist_ok=True)
    inputs = {item["id"]: item for item in report["config"]["inputs"]}
    for model in report["models"]:
        config = (report["config"]["model_evaluations"][model["id"]]
                  if "model_evaluations" in report["config"] else report["config"])
        result = {"status": "complete", "metric_unit": report["metric_unit"], **model,
                  "config": {**config, "inputs": [inputs[model["id"]]]}}
        path = scores / f"{model['id']}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def write_report(models, config, output, *, replace=False, artifact_sources=None):
    report = build_report(models, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (staging / "details").mkdir()
        for model in models:
            path = staging / "details" / f"{model['id']}.jsonl"
            if artifact_sources and model["id"] in artifact_sources:
                source = artifact_sources[model["id"]] / "details" / path.name
                if not source.resolve().is_relative_to(output.resolve()):
                    _relative_symlink(source, path, output / "details")
                    continue
            with path.open("w") as handle:
                for row in model["results"]:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (staging / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        write_latex_tables(report, staging)
        write_model_scores(report, staging)
        if output.exists():
            if not replace:
                raise FileExistsError(f"Report already exists: {output}")
            shutil.rmtree(output)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def read_report(directory):
    report = json.loads((directory / "results.json").read_text())
    models = [{**info, "results": [json.loads(line) for line in (directory / "details" / f"{info['id']}.jsonl").open()]}
              for info in report["config"]["inputs"]]
    return report, models


def _report_model_config(report, model_id):
    config = report["config"]
    if "model_evaluations" in config:
        return config["model_evaluations"][model_id]
    return {k: v for k, v in config.items() if k != "inputs"}


def combine_reports(directories, output, *, replace=False):
    """Join disjoint sets of each model, then compare models on identical questions."""
    settings, models, artifact_sources, groups = {}, [], {}, defaultdict(list)
    canonical = None
    for directory in directories:
        report, scored = read_report(directory)
        for model in scored:
            current = _report_model_config(report, model["id"])
            if canonical is None:
                canonical = current
            elif current != canonical:
                raise ValueError("Cannot combine reports from incompatible evaluation protocols")
            groups[(model['model_id'], model['variant'])].append((model, current, directory))
    if not groups:
        raise ValueError("No completed reports to combine")
    for (alias, variant), parts in groups.items():
        first, current, directory = parts[0]
        if len(parts) == 1:
            models.append(first)
            settings[first['id']] = current
            artifact_sources[first['id']] = directory
            continue
        # A different checkpoint, prompt, decoding mode or input variant is a
        # different experiment. Only the declared sets may differ across runs.
        inference = {k: v for k, v in first.get('inference', {}).items() if k != 'sets'}
        combined = {'id': alias if variant == 'full' else f'{alias}_{variant}',
                    'model_id': alias, 'model': first['model'], 'variant': variant,
                    'inference': inference, 'sources': {}, 'runs': [], 'results': []}
        if 'model_revision' in first:
            combined['model_revision'] = first['model_revision']
        splits = set()
        for part, setting, source in parts:
            if (part['model'] != first['model'] or part.get('model_revision') != first.get('model_revision')
                    or {k: v for k, v in part.get('inference', {}).items() if k != 'sets'} != inference):
                raise ValueError(f'Cannot merge different checkpoint/inference settings for {alias}')
            present = {r['split'] for r in part['results']}
            if splits & present:
                raise ValueError(f'Duplicate or overlapping sets for {alias}: {sorted(splits & present)}')
            splits.update(present)
            combined['sources'].update(part['sources'])
            combined['results'].extend(part['results'])
            combined['runs'].extend(part.get('runs', [{
                'input': {k: v for k, v in part.items() if k != 'results'},
                'evaluation': setting, 'results_dir': str(source.resolve())}]))
        combined['results'].sort(key=lambda row: (row['split'], row['id']))
        models.append(combined)
        settings[combined['id']] = canonical
    evaluation = canonical.get("evaluation", "comparison")
    version = evaluation.rsplit("_", 1)[-1] if "_" in evaluation else ""
    config = {"evaluation": f"comparison_{version}" if version.startswith("v") else "comparison",
              "model_evaluations": settings,
              "inputs": [{k: v for k, v in model.items() if k != 'results'} for model in models]}
    write_report(models, config, output, replace=replace, artifact_sources=artifact_sources)
    return output / "results.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Completed result directories to compare")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(combine_reports(args.results, args.output_dir))
