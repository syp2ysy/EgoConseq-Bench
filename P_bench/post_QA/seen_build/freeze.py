"""Seal published QA and record exclusions; do not copy images or records."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from pipeline import io_utils
from post_QA import templates


def freeze(root: Path) -> dict:
    root = Path(root).resolve()
    output = root / "metadata/frozen.json"
    if output.exists():
        raise FileExistsError(output)
    document = {"schema": "egoconseq.benchmark-freeze.v1", "status": "frozen",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "total": 0, "splits": {}, "items": []}
    for split in ("seen", "unseen"):
        directory = f"benchmark/{split}"
        folder = root / directory
        metadata = root / "metadata" / split
        qa_path = folder / "QA.json"
        index_path = metadata / "record_index.json"
        qa = json.loads(qa_path.read_text())
        index = json.loads(index_path.read_text())
        patterns = {task: [(row["id"], re.compile(re.sub(
            r"\\\{[^}]+\\\}", ".+?", re.escape(row["question"])))) for row in rows]
                    for task, rows in templates.QUESTION_TEMPLATES.items()}
        counts = Counter()
        for item in qa:
            question = item["messages"][1]["content"].split("Question: ", 1)[1].split("\n\n", 1)[0]
            counts[next(name for name, pattern in patterns[item["task_id"]]
                        if pattern.fullmatch(question))] += 1
        report = json.loads((metadata / "report.json").read_text())
        report["prompts"] = {"version": templates.PROMPT_VERSION, "templates": dict(sorted(counts.items()))}
        io_utils.atomic_write_json(metadata / "report.json", report)
        images = sorted({path for item in qa for path in item["images"]})
        image_hash = hashlib.sha256()
        with ThreadPoolExecutor(max_workers=8) as pool:
            digests = pool.map(io_utils.sha256_file, (folder / path for path in images))
            for path, digest in zip(images, digests):
                image_hash.update(f"{path}\t{digest}\n".encode())
        document["splits"][split] = {
            "directory": directory, "qa_count": len(qa),
            "task_totals": dict(sorted(Counter(q["task_id"] for q in qa).items())),
            "qa_sha256": io_utils.sha256_file(qa_path),
            "record_index_sha256": io_utils.sha256_file(index_path),
            "image_count": len(images), "images_sha256": image_hash.hexdigest(),
        }
        document["total"] += len(qa)
        document["items"].extend({"record_uid": row["record_uid"], "dataset": row["dataset"],
                                   "split": split} for row in index["items"])
    io_utils.atomic_write_json(output, document)
    return document
