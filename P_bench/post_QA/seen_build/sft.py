"""Build a training-only ms-swift dataset and aligned parameter ablations."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import io
import os
from pathlib import Path
import random
import shutil
import tempfile

from PIL import Image

from pipeline import abc1_record, benchmark_candidates, candidate_sources, io_utils, viz
from post_QA import answers, templates
from post_QA.seen_build import artifact, catalog, images, sft_selection, spec


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def distribution_keys(row):
    """The same distribution definitions for initial export and cleanup."""
    task, dataset = row["task_id"], row["dataset"]
    values = {"dataset_task": f"{dataset}/{task}", "length": f"{task}/L{row['action_length']}",
              "start": f"{dataset}/{task}/{row['starts_with']}", "answer": f"{task}/{row['answer']}",
              "template": row["template_id"]}
    if task in {"A4", "B3"}:
        checkpoint = row["inputs"]["checkpoint"]
        values.update(checkpoint_fraction=f"{task}/{checkpoint['fraction']}",
                      checkpoint_action=f"{task}/{checkpoint['action_index']}")
    if task == "A1":
        values["a1_buckets"] = f"{dataset}/{row['starts_with']}/{row['answer_bucket']}"
    if task == "A2":
        values["a2_cells"] = (f"{dataset}/L{row['action_length']}/{row['starts_with']}"
                              f"/o{row['a2_ordinal']}/{row['a2_rank']}")
    return values


def _render(row, hide_params=()):
    return {
        "id": row["id"], "task_id": row["task_id"], "dataset": row["dataset"],
        "images": list(row["images"]),
        "messages": [
            {"role": "system", "content": templates.SYSTEM_PROMPT},
            {"role": "user", "content": templates.render_question(
                row["task_id"], row["inputs"],
                artifact._template(row["task_id"], row["template_id"]),
                hide_params=hide_params)},
            {"role": "assistant", "content": row["answer"]},
        ],
    }


def _relative_image_metadata(summary, root):
    """Use the dataset root for both training images and diagnostic image paths."""
    summary["image_path_base"] = "."
    for failure in summary.get("unreadable_images", []):
        if Path(failure["path"]).is_absolute():
            failure["path"] = os.path.relpath(failure["path"], root)


def export_view(root: Path, *, hide_params=()) -> dict:
    """Derive ablations from saved full JSONL; rebuild full only when explicitly requested."""
    root = Path(root).resolve()
    hidden = set(hide_params)
    if hidden - {"height", "radius", "fov"}:
        raise ValueError("hide_params accepts height, radius, fov")
    ordered = [key for key in ("height", "radius", "fov") if key in hidden]
    name = ("no_parameters" if len(hidden) == 3 else
            "no_" + "_".join(ordered) if hidden else "full")
    path = root / "json" / f"{name}.jsonl"
    count = 0

    def write(stream):
        nonlocal count
        source_path = root / ("json/full.jsonl" if hidden else "metadata/selection.jsonl")
        with source_path.open() as source:
            for line in source:
                row = json.loads(line)
                if hidden:
                    for message in row["messages"]:
                        if message["role"] == "user":
                            message["content"] = templates.remove_parameters(message["content"], hidden)
                else:
                    row = _render(row)
                stream.write(_json(row).encode())
                count += 1

    io_utils.atomic_write_binary(path, write)
    if name == "full":
        summary_path = root / "metadata/summary.json"
        summary = json.loads(summary_path.read_text())
        summary["prompts"]["full_sha256"] = io_utils.sha256_file(path)
        _relative_image_metadata(summary, root)
        io_utils.atomic_write_json(summary_path, summary)
    return {"path": str(path), "qa_items": count, "hidden_parameters": ordered}


def _materialize_record(job):
    source, source_index, candidates, stage = job
    with source.records_path.open("rb") as stream:
        stream.seek(candidates[0].byte_offset)
        record = json.loads(stream.readline())
    if record["schema_version"] != abc1_record.SCHEMA_VERSION:
        record = abc1_record.normalize(record)
    image_root = stage / "images" / source.dataset
    cache, linked = {}, {}
    result = []
    failures = []
    raw = candidate_sources.resolve_record_asset(source.records_path.parent, record["image_path"])
    try:
        cache["rgb", str(raw)] = viz.authenticate_raw_rgb_image(
            raw, expected_resolution=record["sensor"]["resolution"])
    except OSError as error:
        return [], [{"record_uid": record["record_uid"], "kind": "initial",
                     "path": str(raw), "error": type(error).__name__}]
    with Image.open(io.BytesIO(cache["rgb", str(raw)].raw_bytes)) as image:
        if images.dark_fraction(image) >= images.EXTREME_DARK_FRACTION:
            return [], [{"record_uid": record["record_uid"], "kind": "initial",
                         "path": str(raw), "error": "extremely_dark"}]
    terminal_ok = {}
    usable = []
    for candidate in candidates:
        if candidate.task_id == "C1":
            for relative in candidate.terminal_paths:
                if relative not in terminal_ok:
                    path = candidate_sources.resolve_record_asset(source.records_path.parent, relative)
                    try:
                        with Image.open(path) as opened:
                            terminal_ok[relative] = images.dark_fraction(opened) < images.EXTREME_DARK_FRACTION
                        if not terminal_ok[relative]:
                            failures.append({"record_uid": record["record_uid"], "kind": "terminal",
                                             "path": str(path), "error": "extremely_dark"})
                    except OSError as error:
                        terminal_ok[relative] = False
                        failures.append({"record_uid": record["record_uid"], "kind": "terminal",
                                         "path": str(path), "error": type(error).__name__})
            if not all(terminal_ok[path] for path in candidate.terminal_paths):
                continue
        usable.append(candidate)
    if not usable:
        # A broken C1 option need not remove a readable record's other tasks.
        alternatives = benchmark_candidates.project_record(
            dataset=source.dataset, source_path=str(source.records_path),
            byte_offset=candidates[0].byte_offset, record_sha256="", record=record,
            allowed_tasks=set(spec.supported_tasks(source.dataset)) - {"C1"})
        if alternatives:
            row = alternatives[0]
            usable = [replace(row, template_id=templates.QUESTION_TEMPLATES[row.task_id][0]["id"])]

    def image_path(path, digest):
        path = Path(path)
        if path.is_relative_to(stage):
            return path.relative_to(stage).as_posix()
        key = str(path)
        if key not in linked:
            relative = Path("images") / source.dataset / "assets" / (
                f"{record['record_uid']}-{digest[:16]}.png")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                io_utils.link_or_copy_file(path, destination)
            linked[key] = relative.as_posix()
        return linked[key]

    for candidate in usable:
        model, private, choices, _ = artifact.project(
            candidate, source, record, image_root, image_cache=cache)
        paths = [image_path(model["initial_rgb"], model["initial_rgb_sha256"])]
        if candidate.task_id == "C1":
            correct = next(choice for choice in choices
                           if choice["id"] == private["canonical_answer"])
            distractors = [choice for choice in choices if choice is not correct]
            position = "ABCD".index(candidate.c1_label)
            choices = [correct if i == position else distractors.pop(0) for i in range(4)]
            paths.extend(image_path(choice["image"], choice["image_sha256"])
                         for choice in choices)
            private["canonical_answer"] = candidate.c1_label
        row = {
            "id": candidate.item_id, "task_id": candidate.task_id,
            "dataset": candidate.dataset, "scene_id": candidate.scene_id,
            "record_uid": candidate.record_id, "case_id": candidate.outcome_id,
            "source_index": source_index, "byte_offset": candidate.byte_offset,
            "point_id": candidate.point_id, "template_id": candidate.template_id,
            "images": paths, "inputs": artifact._inputs(model),
            "answer": answers.canonical_answer(candidate.task_id, private),
            "answer_bucket": candidate.answer_bucket,
            "action_length": candidate.action_length, "starts_with": candidate.starts_with,
        }
        if candidate.task_id == "A2":
            row.update(a2_ordinal=candidate.a2_ordinal, a2_rank=candidate.a2_rank)
        if candidate.task_id == "C1":
            row["c1_option_case_ids"] = [choice["outcome_id"] for choice in choices]
        result.append(row)
    return result, failures


def compile_records(records_root: Path, output_root: Path, *,
                    exclude_index: Path, seed: int = 20260906,
                    workers: int = 8) -> dict:
    """Cover the non-benchmark records and materialize only the selected QA."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    sources = catalog.load(records_root)
    excluded = {row["record_uid"] for row in
                json.loads(Path(exclude_index).read_text())["items"]}
    rng = random.Random(seed)
    records, record_totals, record_support = [], Counter(), Counter()
    excluded_count, unsupported = 0, 0
    original_views = set()
    for source in sources:
        with source.records_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                payload = stream.readline()
                if not payload:
                    break
                if not payload.strip():
                    continue
                record = json.loads(payload)
                if record["record_uid"] in excluded:
                    excluded_count += 1
                    continue
                record_totals[source.dataset] += 1
                rows = benchmark_candidates.project_record(
                    dataset=source.dataset, source_path=str(source.records_path),
                    byte_offset=offset, record_sha256="", record=record,
                    allowed_tasks=spec.supported_tasks(source.dataset))
                rows = sft_selection.representatives(rows, rng)
                record_support.update({row.task_id for row in rows})
                if rows:
                    records.append(rows)
                    original_views.add(str(source.records_path.parent / record["image_path"]))
                else:
                    unsupported += 1
                if sum(record_totals.values()) % 5000 == 0:
                    print(f"SFT scan: {sum(record_totals.values()):,} non-benchmark records", flush=True)
    selected, sampling = sft_selection.select(records, seed=seed)
    del records
    print(f"SFT selection: {len(selected):,} QA; materializing shared images", flush=True)
    print(f"SFT task totals: {dict(Counter(row.task_id for row in selected))}", flush=True)
    by_record = defaultdict(list)
    for row in selected:
        by_record[row.record_id].append(row)
    source_lookup = {str(source.records_path): (index, source)
                     for index, source in enumerate(sources)}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for directory in ("images", "json", "metadata"):
            (stage / directory).mkdir()
        jobs = []
        for rows in by_record.values():
            index, source = source_lookup[rows[0].source_path]
            jobs.append((source, index, rows, stage))
        task_counts, dataset_counts, distributions = Counter(), Counter(), defaultdict(Counter)
        image_refs = set()
        unreadable_images = []
        covered = 0
        with ProcessPoolExecutor(max_workers=workers) as pool, \
                (stage / "metadata/selection.jsonl").open("w") as metadata, \
                (stage / "json/full.jsonl").open("w") as output:
            # Bounded batches avoid queuing all record payloads/results at once.
            done = 0
            for begin in range(0, len(jobs), 512):
                for rows, failures in pool.map(_materialize_record, jobs[begin:begin + 512], chunksize=8):
                    covered += bool(rows)
                    unreadable_images.extend(failures)
                    for row in rows:
                        metadata.write(_json(row))
                        output.write(_json(_render(row)))
                        task, dataset = row["task_id"], row["dataset"]
                        task_counts[task] += 1
                        dataset_counts[dataset] += 1
                        image_refs.update(row["images"])
                        for key, value in distribution_keys(row).items():
                            distributions[key][value] += 1
                    done += 1
                if done % 5120 == 0 or done == len(jobs):
                    print(f"SFT images: {done:,}/{len(jobs):,} records", flush=True)
        report = {
            "seed": seed, "format": "ms-swift messages/images", "split": "training-only",
            "prompts": {"version": templates.PROMPT_VERSION,
                        "full_sha256": io_utils.sha256_file(stage / "json/full.jsonl"),
                        "selection_sha256": io_utils.sha256_file(stage / "metadata/selection.jsonl")},
            "sources": [{"path": str(source.records_path), "sha256": source.records_sha256}
                        for source in sources],
            "exclude_index": str(Path(exclude_index).resolve()),
            "exclude_index_sha256": io_utils.sha256_file(exclude_index),
            "record_count": sum(record_totals.values()), "covered_records": covered,
            "excluded_records": excluded_count, "records_without_qa": unsupported,
            "record_totals": dict(record_totals), "record_task_support": dict(record_support),
            "qa_items": sum(task_counts.values()), "task_totals": dict(sorted(task_counts.items())),
            "dataset_totals": dict(dataset_counts),
            "unique_initial_view_paths": len(original_views - {
                value["path"] for value in unreadable_images if value["kind"] == "initial"}),
            "image_files": len(image_refs),
            "sampling": sampling,
            "unreadable_images": unreadable_images,
            "distributions": {key: dict(sorted(value.items())) for key, value in distributions.items()},
        }
        sampling["task_shortfalls"] = {
            task: max(0, sampling["target_per_task"] - task_counts[task])
            for task in sampling["capacity_per_task"]}
        _relative_image_metadata(report, output_root)
        io_utils.atomic_write_json(stage / "metadata/summary.json", report)
        stage.rename(output_root)
    except Exception:
        shutil.rmtree(stage)
        raise
    return {"path": str(output_root), **report}
