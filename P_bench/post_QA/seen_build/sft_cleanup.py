"""Prune unusably dark training views and synchronize exported A3 names."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from PIL import Image

from pipeline import io_utils
from post_QA import categories
from post_QA.seen_build import images, sft
from visualization import benchmark_browser


def scan_dark_images(root, *, workers=16):
    root = Path(root)
    with (root / "metadata/selection.jsonl").open() as stream:
        paths = sorted({p for line in stream for p in json.loads(line)["images"]})

    def inspect(path):
        with Image.open(root / path) as image:
            return path, images.dark_fraction(image)

    rejected = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(paths), 2048):
            rejected.update((path, fraction) for path, fraction in
                            pool.map(inspect, paths[begin:begin + 2048])
                            if fraction >= images.EXTREME_DARK_FRACTION)
            if begin % 20480 == 0:
                print(f"[sft-clean] images {min(begin + 2048, len(paths))}/{len(paths)}; "
                      f"extremely dark {len(rejected)}", flush=True)
    return {"images_checked": len(paths), "dark_threshold": images.EXTREME_DARK_FRACTION,
            "dark_images": rejected}


def _a3_answer(path, offset, uid, case_id):
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        record = json.loads(stream.readline())
    if record["record_uid"] != uid:
        raise ValueError(f"A3 export no longer binds to its source record: {uid}")
    case = next(c for c in record["cases"] if c["case_id"] == case_id)
    return categories.a3_answer(record, case["task_outputs"]["A3"])


def clean(root, benchmark_root, *, workers=16, scan=None):
    """Change only QA membership / A3 answer text; never write source records."""
    root, benchmark_root = Path(root).resolve(), Path(benchmark_root).resolve()
    scan = scan_dark_images(root, workers=workers) if scan is None else scan
    report = {**{key: scan[key] for key in ("images_checked", "dark_threshold", "dark_images")},
              "updated_at_utc": datetime.now(timezone.utc).isoformat(), "benchmark_names": {}}
    bad = set(report["dark_images"])
    summary_path = root / "metadata/summary.json"
    summary = json.loads(summary_path.read_text())
    sources = summary["sources"]
    seen_sources = {Path(s["path"]).parent.name: s["path"] for s in sources}
    removed, renamed, before_refs, after_refs, before_uids, after_uids = {}, {}, set(), set(), set(), set()
    task_counts, dataset_counts, heights = Counter(), Counter(), Counter()
    distributions = {key: Counter() for key in summary["distributions"]}

    with tempfile.TemporaryDirectory(prefix=".sft-clean-", dir=root.parent) as temporary:
        stage, pending = Path(temporary), {}

        def destination(path):
            path = Path(path)
            if path not in pending:
                relative = (Path("benchmark") / path.relative_to(benchmark_root)
                            if path.is_relative_to(benchmark_root) else Path("sft") / path.relative_to(root))
                pending[path] = stage / relative
                pending[path].parent.mkdir(parents=True, exist_ok=True)
            return pending[path]

        def put(path, value):
            io_utils.atomic_write_json(destination(path), value, allow_nan=False)

        selection_path = root / "metadata/selection.jsonl"
        with selection_path.open() as incoming, destination(selection_path).open("w") as output:
            for number, line in enumerate(incoming, 1):
                row = json.loads(line)
                before_refs.update(row["images"])
                before_uids.add(row["record_uid"])
                if bad.intersection(row["images"]):
                    removed[row["id"]] = {key: row[key] for key in ("record_uid", "case_id", "task_id", "dataset")}
                    continue
                if row["task_id"] == "A3":
                    answer = _a3_answer(sources[row["source_index"]]["path"], row["byte_offset"],
                                        row["record_uid"], row["case_id"])
                    if answer != row["answer"]:
                        renamed[row["id"]] = answer
                        row["answer"] = row["answer_bucket"] = answer
                        line = sft._json(row)
                output.write(line)
                after_refs.update(row["images"])
                after_uids.add(row["record_uid"])
                task_counts[row["task_id"]] += 1
                dataset_counts[row["dataset"]] += 1
                heights[str(row["inputs"]["camera"]["optical_center_height_m"])] += 1
                for key, value in sft.distribution_keys(row).items():
                    distributions[key][value] += 1
                if number % 50000 == 0:
                    print(f"[sft-clean] QA {number}; removed {len(removed)}", flush=True)
        for path in sorted((root / "json").glob("*.jsonl")):
            with path.open() as incoming, destination(path).open("w") as output:
                for line in incoming:
                    row = json.loads(line)
                    if row["id"] in removed:
                        continue
                    if row["id"] in renamed:
                        for message in row["messages"]:
                            if message["role"] == "assistant":
                                message["content"] = renamed[row["id"]]
                        line = sft._json(row)
                    output.write(line)

        frozen_path = benchmark_root / "metadata/frozen.json"
        frozen = json.loads(frozen_path.read_text())
        for split in ("seen", "unseen"):
            base = benchmark_root / "metadata" / split
            qa_path = benchmark_root / "benchmark" / split / "QA.json"
            index_path = base / "record_index.json"
            qa, index = json.loads(qa_path.read_text()), json.loads(index_path.read_text())
            by_id = {q["id"]: q for q in qa}
            changed = 0
            for entry in index["items"]:
                if entry["task_id"] != "A3":
                    continue
                path = Path(entry["source"]["records_path"])
                if not path.is_absolute():
                    path = Path(seen_sources[entry["dataset"]])
                answer = _a3_answer(path, entry["source"]["byte_offset"], entry["record_uid"],
                                    entry["usage"]["outcome_id"])
                for message in by_id[entry["item_id"]]["messages"]:
                    if message["role"] == "assistant":
                        changed += message["content"] != answer
                        message["content"] = answer
                entry["usage"]["contact_category"] = answer
            put(qa_path, qa)
            index["qa_sha256"] = io_utils.sha256_file(destination(qa_path))
            put(index_path, index)
            frozen["splits"][split].update(qa_sha256=index["qa_sha256"],
                record_index_sha256=io_utils.sha256_file(destination(index_path)))
            selection_report = json.loads((base / "report.json").read_text())
            buckets = selection_report["answer_buckets"]
            for key in list(buckets):
                if "/A3/" in key:del buckets[key]
            buckets.update(Counter(f"{q['dataset']}/A3/{q['messages'][-1]['content']}"
                                   for q in qa if q["task_id"] == "A3"))
            put(base / "report.json", selection_report)
            report["benchmark_names"][split] = changed
        frozen["category_name_sync"] = {"updated_at_utc": report["updated_at_utc"],
                                       "changed_answers": report["benchmark_names"]}
        put(frozen_path, frozen)
        orphaned = before_refs - after_refs
        report.update(removed_qa=len(removed), removed_items=removed, renamed_qa=len(renamed),
                      removed_by_task=dict(Counter(r["task_id"] for r in removed.values())),
                      removed_by_dataset=dict(Counter(r["dataset"] for r in removed.values())),
                      removed_images=len(orphaned),
                      removed_image_bytes=sum((root / p).stat().st_size for p in orphaned),
                      lost_records=sorted(before_uids - after_uids), qa_items=sum(task_counts.values()))
        summary.update(qa_items=report["qa_items"], task_totals=dict(task_counts),
                       dataset_totals=dict(dataset_counts), image_files=len(after_refs),
                       covered_records=len(after_uids),
                       distributions={key: dict(value) for key, value in distributions.items()},
                       records_without_qa=summary["record_count"] - len(after_uids))
        summary["unique_initial_view_paths"] -= len(before_uids - after_uids)
        summary["prompts"].update(full_sha256=io_utils.sha256_file(destination(root / "json/full.jsonl")),
                                 selection_sha256=io_utils.sha256_file(destination(selection_path)))
        summary["exclude_index_sha256"] = io_utils.sha256_file(destination(frozen_path))
        if "parameters" in summary:
            summary["parameters"].update(qa_items=report["qa_items"], height_distribution_m=dict(heights))
        summary["sampling"]["task_shortfalls"] = {
            task: max(0, summary["sampling"]["target_per_task"] - task_counts[task])
            for task in summary["sampling"]["capacity_per_task"]}
        summary["quality_cleanup"] = {key: value for key, value in report.items()
                                      if key not in {"dark_images", "removed_items"}}
        summary["quality_cleanup"]["capacity_scope"] = "original selection; task counts reflect cleanup"
        put(summary_path, summary)
        put(root / "metadata/quality_cleanup.json", report)

        browser_stage = stage / "benchmark"
        benchmark_browser.build_combined_benchmark_browser(browser_stage)
        destination(benchmark_root / "index.html")
        for relative in orphaned:
            path = root / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or Path(relative).parts[0] != "images":
                raise ValueError(f"Refusing deletion outside SFT images: {relative}")
            pending[path] = None
        backups, replaced = {}, []
        try:
            for path, new in pending.items():
                if path.exists():
                    backup = stage / f"old-{len(backups)}"
                    os.link(path, backup)
                    backups[path] = backup
                if new is None:path.unlink()
                else:os.replace(new, path)
                replaced.append(path)
        except BaseException:
            for path in reversed(replaced):
                if path in backups:os.replace(backups[path], path)
                else:path.unlink()
            raise
    return report
