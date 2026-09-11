"""Refresh frozen QA text from indexed inputs; never load records or images."""

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from pipeline import io_utils
from post_QA import templates
from post_QA.seen_build import artifact, sft
from visualization.benchmark_browser import build_combined_benchmark_browser


def export_benchmark_views(benchmark_root: Path) -> dict:
    """Derive parameter ablations from saved QA, without re-rendering any text."""
    variants = {"no_radius": ("radius",), "no_height": ("height",),
                "no_fov": ("fov",), "no_parameters": ("radius", "height", "fov")}
    result = {}
    for split in ("seen", "unseen"):
        root = Path(benchmark_root) / "benchmark" / split
        full = json.loads((root / "QA.json").read_text())
        for name, hidden in variants.items():
            rows = deepcopy(full)
            for item in rows:
                for message in item["messages"]:
                    if message["role"] == "user":
                        message["content"] = templates.remove_parameters(message["content"], hidden)
            io_utils.atomic_write_json(root / f"QA_{name}.json", rows)
        result[split] = {"qa_items_per_variant": len(full),
                         "files": [f"QA_{name}.json" for name in variants]}
    return result


def _benchmark_inputs(entry):
    usage = entry["usage"]
    model = {key: usage[key] for key in (
        "actions", "body_radius_m", "camera_height_m", "hfov_deg", "vfov_deg")}
    if entry["task_id"] in {"A4", "B1", "B2", "B3"}:
        model["target"] = "the marked point"
    if entry["task_id"] in {"A4", "B3"}:
        checkpoint = usage["checkpoint"]
        model["checkpoint"] = {"action_index": checkpoint["action_index_1based"],
                               "fraction": checkpoint["fraction"]}
    return artifact._inputs(model)


def refresh(benchmark_root: Path, sft_root: Path, *, seed: int = 20260908,
            keep_templates: bool = False) -> dict:
    """Replace only prompts/template IDs and their reports, hashes and HTML."""
    benchmark_root, sft_root = Path(benchmark_root).resolve(), Path(sft_root).resolve()
    result = {"version": templates.PROMPT_VERSION, "seed": seed, "benchmark": {}}
    with tempfile.TemporaryDirectory(prefix=".prompts-", dir=benchmark_root) as temporary:
        stage, pending = Path(temporary), {}

        def destination(path):
            path = Path(path)
            relative = (path.relative_to(benchmark_root) if path.is_relative_to(benchmark_root)
                        else Path("sft") / path.relative_to(sft_root))
            new = stage / relative
            new.parent.mkdir(parents=True, exist_ok=True)
            pending[path] = new
            return new

        def put(path, value):
            io_utils.atomic_write_json(destination(path), value, allow_nan=False)

        frozen_path = benchmark_root / "metadata/frozen.json"
        frozen = json.loads(frozen_path.read_text())
        update = {"version": templates.PROMPT_VERSION,
                  "updated_at_utc": datetime.now(timezone.utc).isoformat(), "splits": {}}
        for offset, split in enumerate(("seen", "unseen")):
            root = benchmark_root / "metadata" / split
            qa_path = benchmark_root / "benchmark" / split / "QA.json"
            index_path = root / "record_index.json"
            qa, index = json.loads(qa_path.read_text()), json.loads(index_path.read_text())
            by_id = {row["item_id"]: row for row in index["items"]}
            ids = ([by_id[row["id"]]["usage"]["template_id"] for row in qa] if keep_templates else
                   templates.balanced_template_ids((row["task_id"] for row in qa), seed=seed + offset))
            for item, template_id in zip(qa, ids):
                entry = by_id[item["id"]]
                entry["usage"]["template_id"] = template_id
                inputs = _benchmark_inputs(entry)
                for message in item["messages"]:
                    if message["role"] == "system":
                        message["content"] = templates.SYSTEM_PROMPT
                    elif message["role"] == "user":
                        message["content"] = templates.render_question(
                            item["task_id"], inputs, artifact._template(item["task_id"], template_id))
            put(qa_path, qa)
            index["qa_sha256"] = io_utils.sha256_file(destination(qa_path))
            put(index_path, index)
            update["splits"][split] = {"previous_qa_sha256": frozen["splits"][split]["qa_sha256"],
                                       "qa_sha256": index["qa_sha256"]}
            frozen["splits"][split].update(qa_sha256=index["qa_sha256"],
                record_index_sha256=io_utils.sha256_file(destination(index_path)))
            report = json.loads((root / "report.json").read_text())
            report.setdefault("prompts", {}).update(
                version=templates.PROMPT_VERSION, templates=dict(sorted(Counter(ids).items())))
            if not keep_templates:
                report["prompts"]["seed"] = seed + offset
            put(root / "report.json", report)
            result["benchmark"][split] = {"qa_items": len(qa), **report["prompts"]}
            print(f"[prompts] {split}: {len(qa)} QA", flush=True)
        frozen.setdefault("prompt_updates", []).append(update)
        put(frozen_path, frozen)
        for split, views in export_benchmark_views(stage).items():
            for filename in views["files"]:
                destination(benchmark_root / "benchmark" / split / filename)

        selection_path = sft_root / "metadata/selection.jsonl"
        with selection_path.open() as source:
            ids = ([json.loads(line)["template_id"] for line in source] if keep_templates else
                   templates.balanced_template_ids((json.loads(line)["task_id"] for line in source), seed=seed + 2))
        with ExitStack() as stack:
            source = stack.enter_context(selection_path.open())
            metadata = stack.enter_context(destination(selection_path).open("w"))
            outputs = []
            for path in sorted((sft_root / "json").glob("*.jsonl")):
                hidden = (() if path.stem == "full" else
                          ("height", "radius", "fov") if path.stem == "no_parameters" else
                          tuple(path.stem.removeprefix("no_").split("_")))
                outputs.append((stack.enter_context(destination(path).open("w")), hidden))
            for number, (line, template_id) in enumerate(zip(source, ids), 1):
                row = json.loads(line)
                row["template_id"] = template_id
                metadata.write(sft._json(row))
                for output, hidden in outputs:
                    output.write(sft._json(sft._render(row, hidden)))
                if number % 50000 == 0:
                    print(f"[prompts] SFT: {number}/{len(ids)} QA", flush=True)
        summary_path = sft_root / "metadata/summary.json"
        summary = json.loads(summary_path.read_text())
        sft._relative_image_metadata(summary, sft_root)
        summary["distributions"]["template"] = dict(sorted(Counter(ids).items()))
        summary["prompts"].update(version=templates.PROMPT_VERSION,
            full_sha256=io_utils.sha256_file(destination(sft_root / "json/full.jsonl")),
            selection_sha256=io_utils.sha256_file(destination(selection_path)))
        if not keep_templates:
            summary["prompts"]["template_seed"] = seed + 2
        summary["exclude_index_sha256"] = io_utils.sha256_file(destination(frozen_path))
        put(summary_path, summary)
        result["sft"] = {"qa_items": len(ids), "templates": summary["distributions"]["template"]}

        # The staged hierarchy matches the published hierarchy, so relative image
        # URLs remain identical. Browser generation reads metadata, not pixels.
        build_combined_benchmark_browser(stage)
        destination(benchmark_root / "index.html")

        # Keep only temporary hardlinks while switching the small set of outputs.
        # They are removed with the stage on success; failures restore old files.
        backups, replaced = {}, []
        try:
            for path, new in pending.items():
                if path.exists():
                    backup = stage / f"old-{len(backups)}"
                    os.link(path, backup)
                    backups[path] = backup
                os.replace(new, path)
                replaced.append(path)
        except BaseException:
            for path in reversed(replaced):
                if path in backups:
                    os.replace(backups[path], path)
                else:
                    path.unlink()
            raise
    return result
