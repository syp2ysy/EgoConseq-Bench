"""Synchronize configured relative camera height without changing source data."""

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

from pipeline import config, io_utils
from post_QA import templates
from post_QA.seen_build import catalog, sft


def _json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _height_prompt(prompt, height):
    prompt, count = re.subn(
        r"(?m)^- (?:Camera optical-center height above ground|Relative camera height|"
        r"Camera optical-center height relative to the local ground reference): [-\d.]+ m$",
        "- Camera optical-center height relative to the local ground reference: "
        f"{height:.1f} m", prompt)
    if count != 1:
        raise ValueError("frozen prompt must contain one camera-height setting")
    return prompt


def repair(seen_records, unseen_records, benchmark_root, sft_root):
    """Read original height settings; replace only QA/metadata and regenerate HTML."""
    benchmark_root, sft_root = Path(benchmark_root).resolve(), Path(sft_root).resolve()
    report = {"version": "configured-relative-height.v1",
              "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "height_source": "sensor.nominal_camera_offset_m",
              "records_unchanged": True, "images_unchanged": True,
              "gt_unchanged": True, "datasets": {}, "benchmark": {}}
    heights = {}
    for root in (Path(seen_records), Path(unseen_records)):
        split = json.loads((root/"manifest.json").read_text())["split"]
        for source in catalog.load(root):
            counts = Counter()
            with source.records_path.open("rb") as stream:
                for line in stream:
                    record = json.loads(line)
                    height = float(record["sensor"]["nominal_camera_offset_m"])
                    if height not in config.BENCH_CAMERA_HEIGHTS_M:
                        raise ValueError(f"unexpected configured camera height: {record['record_uid']}")
                    heights[record["record_uid"]] = height
                    counts[f"{height:.1f}"] += 1
            key = f"{split}/{source.dataset}"
            total = report["datasets"].setdefault(key, {"records": 0, "height_distribution_m": {}})
            total["records"] += sum(counts.values())
            total["height_distribution_m"] = dict(Counter(total["height_distribution_m"]) + counts)
            print(f"[height] {key}: {dict(counts)}", flush=True)

    with tempfile.TemporaryDirectory(prefix=".parameters-", dir=benchmark_root) as temporary:
        stage, pending = Path(temporary), {}

        def destination(path):
            path = Path(path).resolve()
            if path not in pending:
                pending[path] = stage / str(len(pending))
            return pending[path]

        def put(path, value):
            destination(path).write_bytes(_json(value))

        frozen_path = benchmark_root / "metadata/frozen.json"
        frozen = json.loads(frozen_path.read_text())
        frozen.update(parameter_revision=report["version"], frozen_at_utc=report["updated_at_utc"])
        for split in ("seen", "unseen"):
            root = benchmark_root / "metadata" / split
            qa_path = benchmark_root / "benchmark" / split / "QA.json"
            index_path = root / "record_index.json"
            qa, index = json.loads(qa_path.read_text()), json.loads(index_path.read_text())
            by_id = {row["item_id"]: row for row in index["items"]}
            counts, changed = Counter(), 0
            for item in qa:
                entry = by_id[item["id"]]
                height, usage = heights[entry["record_uid"]], entry["usage"]
                changed += usage["camera_height_m"] != height
                usage["camera_height_m"] = height
                usage["nominal_camera_offset_m"] = height
                usage.pop("camera_height_source", None)
                usage.pop("ground_y_world_m", None)
                for message in item["messages"]:
                    if message["role"] == "system":
                        message["content"] = templates.SYSTEM_PROMPT
                    elif message["role"] == "user":
                        message["content"] = _height_prompt(message["content"], height)
                counts[f"{height:.1f}"] += 1
            put(qa_path, qa)
            index["qa_sha256"] = io_utils.sha256_file(destination(qa_path))
            put(index_path, index)
            frozen["splits"][split].update(qa_sha256=index["qa_sha256"],
                record_index_sha256=io_utils.sha256_file(destination(index_path)))
            report["benchmark"][split] = {"qa_items": len(qa), "height_changed": changed,
                                          "height_distribution_m": dict(counts)}
            selection_report = json.loads((root/"report.json").read_text())
            selection_report["prompts"]["version"] = templates.PROMPT_VERSION
            selection_report["parameters"] = report["benchmark"][split]
            put(root/"report.json", selection_report)
        put(frozen_path, frozen)

        summary_path, selection_path = sft_root/"metadata/summary.json", sft_root/"metadata/selection.jsonl"
        summary = json.loads(summary_path.read_text())
        sft._relative_image_metadata(summary, sft_root)
        counts, changed = Counter(), 0
        with ExitStack() as stack:
            incoming = stack.enter_context(selection_path.open())
            metadata = stack.enter_context(destination(selection_path).open("wb"))
            outputs = []
            for path in sorted((sft_root/"json").glob("*.jsonl")):
                hidden = (() if path.stem == "full" else
                          ("height", "radius", "fov") if path.stem == "no_parameters" else
                          tuple(path.stem.removeprefix("no_").split("_")))
                outputs.append((stack.enter_context(destination(path).open("wb")), hidden))
            for line in incoming:
                row = json.loads(line)
                height = heights[row["record_uid"]]
                camera = row["inputs"]["camera"]
                changed += camera["optical_center_height_m"] != height
                camera["optical_center_height_m"] = height
                metadata.write(_json(row))
                for output, hidden in outputs:
                    output.write(_json(sft._render(row, hidden)))
                counts[f"{height:.1f}"] += 1
        report["sft"] = {"qa_items": sum(counts.values()), "height_changed": changed,
                         "height_distribution_m": dict(counts)}
        summary["parameters"] = report["sft"]
        summary["exclude_index_sha256"] = io_utils.sha256_file(destination(frozen_path))
        summary["prompts"].update(version=templates.PROMPT_VERSION,
            full_sha256=io_utils.sha256_file(destination(sft_root/"json/full.jsonl")),
            selection_sha256=io_utils.sha256_file(destination(selection_path)))
        put(summary_path, summary)
        put(benchmark_root/"metadata/parameter_audit.json", report)

        backups, replaced = {}, []
        try:
            for path, new in pending.items():
                if path.exists():
                    backup = stage / (new.name + ".old")
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

    from visualization.benchmark_browser import build_combined_benchmark_browser
    build_combined_benchmark_browser(benchmark_root)
    return report
