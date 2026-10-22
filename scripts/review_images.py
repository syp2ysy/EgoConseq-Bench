"""Offline image review. No simulator, QA answers, or inference in the QA builder.

prepare: initial-image inventory, optionally scene-stratified or restricted to debug seeds.
audit:   single-token yes/no scores on the selected GPU, cached by protocol and image hash.
merge:   freeze keep/reject: explicit quality rules, then human labels, then model thresholds.
sample:  an optional blind check of images on either side of those thresholds.
calibrate: count human keep/reject labels in each sampled group; no threshold search.

Use ``python -m scripts.review_images <command> --help``. The audit environment needs vLLM;
only prepare needs PyArrow. C1 option inventories come from the builder review_pending.csv.
Threshold changes never rerun inference. None of these commands launches the next step.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import time
import shutil

from pipeline.io_utils import atomic_write_json, sha256_file
from post_QA.build.images import Reviews, validate_review

MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots" / MODEL_REVISION
PROMPT = """Does this indoor navigation image contain severe reconstruction or rendering damage?
Severe means a large part of the visible scene is broken, stretched, missing, or replaced by
nonphysical fragments or clouds that substantially destroy the depicted surfaces or objects.
Small local holes, seams, isolated artifacts, ordinary blur, and imperfect textures are not severe.
Close-up views, low texture, plain walls, dark objects, and black areas alone are normal.
Simplified or stylized simulator assets, including low-detail trees and foliage, are not defects by themselves.
Judge visible rendering damage only, not navigation difficulty or how informative the view is.
Answer Yes for severe damage, or No otherwise. Reply with only Yes or No."""

def protocol_id(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    out = {}
    for row in rows:
        sha = row["image_sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or sha in out:
            raise ValueError(f"invalid or duplicate image hash in {path}: {sha}")
        out[sha] = row
    return out


def write_csv(path, rows, fields=None):
    """Do not silently change a frozen inventory or review table."""
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError("refusing to freeze an empty table")
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode()
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"{path} already exists; write a new version")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def load_cache(path, protocol):
    """Ignore only an interrupted final write; corruption in a complete record is an error."""
    result = {}
    if not path.exists():
        return result
    with path.open("rb+") as stream:
        while True:
            start, line = stream.tell(), stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(start)
                break
            row = json.loads(line)
            if row["protocol_version"] != protocol:
                raise ValueError(f"cache protocol differs: {path}")
            sha = row["image_sha256"]
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"invalid cached image hash: {sha}")
            score = row["bad_score"]
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid cached score: {sha}")
            if sha in result:
                raise ValueError(f"duplicate cached result: {sha}")
            result[sha] = row
    return result


def label_tokens(tokenizer):
    """Complete single-token yes/no spellings, including whitespace and case."""
    groups = {"yes": [], "no": []}
    for token in sorted(set(tokenizer.get_vocab().values())):
        label = tokenizer.decode([token]).strip().lower()
        if label in groups:
            groups[label].append(token)
    if not all(groups.values()):
        raise ValueError("tokenizer lacks a complete yes or no token")
    return groups


def bad_score(logprobs, groups):
    """P(Yes spellings | Yes or No), temperature 1; not calibrated defect probability."""
    values = {t: logprobs[t].logprob for ids in groups.values() for t in ids}
    maximum = max(values.values())
    if not math.isfinite(maximum) or any(math.isnan(v) or v == math.inf for v in values.values()):
        raise ValueError("invalid yes/no log probabilities")
    mass = {k: sum(math.exp(values[t] - maximum) for t in ids) for k, ids in groups.items()}
    return mass["yes"] / (mass["yes"] + mass["no"])


def load_thresholds(path, datasets):
    thresholds = json.loads(Path(path).read_text())
    for dataset in sorted(set(datasets)):
        value = thresholds.get(dataset)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"missing or invalid threshold for {dataset}")
    return {dataset: thresholds[dataset] for dataset in sorted(set(datasets))}


def merge_reviews(model, human, thresholds, inventory, rules=None):
    merged = {}
    for sha, row in model.items():
        reject = row["bad_score"] >= thresholds[inventory[sha]["dataset"]]
        merged[sha] = dict(image_sha256=sha, decision="reject" if reject else "keep",
                           reason="render_artifact" if reject else "below_threshold",
                           source="model", protocol_version=row["protocol_version"])
    for sha, row in human.items():
        if row["decision"] == "" and row["reason"] == "uncertain":
            merged.pop(sha, None)  # An explicit human uncertainty stays a coverage gap.
            continue
        validate_review(sha, row["decision"], row["reason"])
        merged[sha] = {**row, "source": "human"}
    for sha, row in (rules or {}).items():
        if (row["decision"] != "reject" or row["reason"] not in {"dark", "gs_bright_low_contrast"}
                or row["reason"] == "gs_bright_low_contrast" and inventory[sha]["dataset"] != "gs"):
            raise ValueError(f"invalid quality rule: {sha}")
        merged[sha] = {**row, "source": "rule"}
    return merged


def quality_rule(row):
    """Cheap exclusions before inference; GS plain close-ups are intentionally included."""
    reason = "dark" if float(row["dark_pixel_fraction"]) > .2 else None
    if reason is None and row["dataset"] == "gs":
        import numpy as np
        from PIL import Image
        raw = Path(row["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError(f"image changed: {row['path']}")
        with Image.open(io.BytesIO(raw)) as image:
            gray = np.asarray(image.convert("RGB"), dtype=np.float32) @ np.array([.299, .587, .114], dtype=np.float32)
        if gray.mean() >= 180 and gray.std() <= 15:
            reason = "gs_bright_low_contrast"
    if reason:
        return dict(image_sha256=row["image_sha256"], decision="reject", reason=reason,
                    source="rule", protocol_version="image-rules-v11")


def prepare(args):
    import pyarrow.parquet as pq

    labels = read_csv(args.labels) if args.labels else None
    out = {}
    for meta in args.meta:
        for file in sorted((meta / "observations").glob("*.parquet")):
            if args.datasets and file.stem not in args.datasets:
                continue
            table = pq.read_table(file, columns=["obs_id", "scene", "status", "image", "image_sha256", "dark_pixel_fraction"])
            for row in table.to_pylist():
                sha = row["image_sha256"]
                if row["status"] != "ok" or labels is not None and sha not in labels:
                    continue
                out.setdefault(sha, dict(image_sha256=sha, path=str((meta / row["image"]).resolve()),
                    dataset=file.stem, scene=row["scene"], obs_id=row["obs_id"], role="initial",
                    dark_pixel_fraction=row["dark_pixel_fraction"]))
    if labels and labels.keys() - out.keys():
        raise ValueError(f"{len(labels.keys() - out.keys())} labelled images are missing from meta")
    if args.per_dataset:
        grouped = defaultdict(lambda: defaultdict(list))
        for row in out.values():
            grouped[row["dataset"]][row["scene"]].append(row)
        selected = {}
        for dataset, scenes in sorted(grouped.items()):
            ordered = [sorted(g, key=lambda r: protocol_id([args.seed, r["image_sha256"]]))
                       for _, g in sorted(scenes.items(), key=lambda x: protocol_id([args.seed, x[0]]))]
            # Round-robin samples all scenes without replacement.
            rows = [g[i] for i in range(max(map(len, ordered))) for g in ordered if i < len(g)]
            selected.update((r["image_sha256"], r) for r in rows[:args.per_dataset])
        out = selected
    write_csv(args.output, (out[k] for k in sorted(out)))
    if args.rules_output:
        with ThreadPoolExecutor(8) as pool:
            rules = [row for row in pool.map(quality_rule, (out[k] for k in sorted(out))) if row]
        write_csv(args.rules_output, rules, fields=["image_sha256", "decision", "reason", "source", "protocol_version"])
        print(json.dumps({"quality_rule_rejections": len(rules), "rules_output": str(args.rules_output)}), flush=True)
    print(json.dumps({"images": len(out), "output": str(args.output)}), flush=True)


def audit(args):
    if args.report.exists():
        raise ValueError(f"{args.report} already exists; use a new run report path")
    # One process per explicitly selected GPU; parallel workers use separate caches.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    if not os.environ["CUDA_VISIBLE_DEVICES"].isdigit():
        raise ValueError("select one GPU per audit process with CUDA_VISIBLE_DEVICES")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoProcessor

    model_path = args.model.resolve()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    groups = label_tokens(processor.tokenizer)
    tokens = sorted(t for ids in groups.values() for t in ids)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()) | {p.name for p in model_path.glob("*.json")}
                   | {p.name for p in model_path.glob("*.txt")} | {p.name for p in model_path.glob("*.jinja")})
    model_hashes = {}
    for file in files:
        path = model_path / file
        if not path.is_file():
            raise FileNotFoundError(f"missing model file: {path}")
        # HF stores weights in immutable content-addressed blobs; non-HF files are hashed directly.
        blob = path.resolve().name
        model_hashes[file] = blob if path.is_symlink() and len(blob) == 64 else sha256_file(path)
    protocol = {"schema": "severe-render-score.v1", "model": str(model_path), "model_files": model_hashes,
        "prompt": PROMPT, "label_tokens": groups,
        "score": "sum P(yes spellings) / sum P(yes and no spellings)", "preprocessing": {"decode": "PIL RGB, no crop",
            "processor_size": {"longest_edge": 640 * 480, "shortest_edge": 65536}},
        "generation": {"temperature": 1., "max_tokens": 1, "seed": 0, "allowed_token_ids": tokens, "logprobs": len(tokens)},
        "engine": {"dtype": "bfloat16", "max_model_len": 2048, "max_num_seqs": 32,
                   "max_num_batched_tokens": 8192, "gpu_memory_utilization": .88, "enforce_eager": True, "logprobs_mode": "processed_logprobs"},
        "versions": {p: version(p) for p in ("vllm", "transformers", "torch", "Pillow")}}
    pid = protocol_id(protocol)
    directory = args.cache / pid
    directory.mkdir(parents=True, exist_ok=True)
    # One writer per protocol, without a server or queue framework.
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_write_json(directory / "protocol.json", protocol, durable=True)
        results = load_cache(directory / "results.jsonl", pid)
        inventory = read_csv(args.images)
        pending = [r for sha, r in sorted(inventory.items()) if sha not in results]
        started = time.monotonic()
        init_seconds, inference_seconds = 0., 0.
        print(json.dumps({"protocol": pid, "images": len(inventory), "cached": len(inventory) - len(pending),
                          "pending": len(pending)}), flush=True)
        if pending:
            from PIL import Image
            from vllm import LLM, SamplingParams

            llm = LLM(model=str(model_path), tensor_parallel_size=1, seed=0, mm_processor_cache_gb=0,
                      limit_mm_per_prompt={"image": 1, "video": 0}, **protocol["engine"],
                      mm_processor_kwargs={"size": protocol["preprocessing"]["processor_size"]})
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT}]}], tokenize=False, add_generation_prompt=True)
            sampling = SamplingParams(**protocol["generation"])
            def read_image(row):
                raw = Path(row["path"]).read_bytes()  # hash and decode the same bytes, once
                if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
                    raise ValueError(f"image changed: {row['path']}")
                with Image.open(io.BytesIO(raw)) as img:
                    return {"prompt": prompt, "multi_modal_data": {"image": img.convert("RGB")}}

            init_seconds = time.monotonic() - started
            with ThreadPoolExecutor(8) as pool, (directory / "results.jsonl").open("a") as stream:
                # Decode the next batch while vLLM runs this one; retain at most two batches.
                reads = [pool.submit(read_image, row) for row in pending[:args.batch]]
                for offset in range(0, len(pending), args.batch):
                    tick = time.monotonic()
                    batch = pending[offset:offset + args.batch]
                    requests = [future.result() for future in reads]
                    reads = [pool.submit(read_image, row) for row in pending[offset + args.batch:offset + 2 * args.batch]]
                    outputs = llm.generate(requests, sampling, use_tqdm=False)
                    if len(outputs) != len(batch):
                        raise ValueError("vLLM returned an incomplete image batch")
                    for row, output in zip(batch, outputs):
                        score = bad_score(output.outputs[0].logprobs[0], groups)
                        result = {"image_sha256": row["image_sha256"], "bad_score": score, "protocol_version": pid}
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        results[row["image_sha256"]] = result
                    stream.flush()
                    os.fsync(stream.fileno())
                    inference_seconds += time.monotonic() - tick
                    done = offset + len(batch)
                    speed = done / inference_seconds
                    print(json.dumps({"reviewed": done, "pending_total": len(pending),
                                      "images_per_second": round(speed, 2),
                                      "eta_seconds": round((len(pending) - done) / speed)}), flush=True)
        report = {"protocol_version": pid, "inventory_sha256": sha256_file(args.images),
            "images": len(inventory), "newly_scored": len(pending), "initialization_seconds": init_seconds,
            "input_batch_size": args.batch, "decode_workers": 8, "prefetch_batches": 1,
            "inference_seconds": inference_seconds, "images_per_second": len(pending) / inference_seconds if inference_seconds else None,
            "results": str(directory / "results.jsonl")}
        # Run reports are separate from immutable raw model scores, including cache-only replays.
        atomic_write_json(args.report, report, durable=True)
        print(json.dumps(report, ensure_ascii=False), flush=True)


def merge(args):
    directory = args.results.parent
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = json.loads((directory / "protocol.json").read_text())
        pid = protocol_id(protocol)
        model = load_cache(args.results, pid)
        human = read_csv(args.human) if args.human else {}
        inventory = {sha: row for path in args.images for sha, row in read_csv(path).items()}
        rules_path = getattr(args, "rules", None)
        rules = read_csv(rules_path) if rules_path else {}
        rules = {sha: row for sha, row in rules.items() if sha in inventory}
        thresholds = load_thresholds(args.thresholds, [r["dataset"] for r in inventory.values()])
        missing = inventory.keys() - model.keys() - human.keys() - rules.keys()
        if missing:
            raise ValueError(f"{len(missing)} inventory images have no score, human decision or quality rule")
        model = {sha: row for sha, row in model.items() if sha in inventory}
        human = {sha: row for sha, row in human.items() if sha in inventory}
        merged = merge_reviews(model, human, thresholds, inventory, rules)
        fields = ("image_sha256", "decision", "reason", "source", "protocol_version")
        rows = [{k: merged[s].get(k, "human" if k == "source" else "") for k in fields} for s in sorted(merged)]
        for row in rows:
            validate_review(row["image_sha256"], row["decision"], row["reason"])
        write_csv(args.output, rows)
        manifest = {"protocol_version": pid, "protocol": protocol, "thresholds": thresholds,
            "comparison": "bad_score >= threshold",
            "raw_sha256": sha256_file(args.results), "human_sha256": sha256_file(args.human) if args.human else None,
            "rules_sha256": sha256_file(rules_path) if rules_path else None,
            "quality_rules": json.loads(rules_path.with_suffix(".json").read_text())
                if rules_path and rules_path.with_suffix(".json").exists() else None,
            "precedence": "mandatory rule rejects > human decisions/uncertainty > model thresholds",
            "table_sha256": sha256_file(args.output), "states": dict(Counter(r["decision"] for r in rows)),
            "reasons": dict(Counter(r["reason"] for r in rows)),
            "human_unresolved": sum(r["decision"] == "" and sha not in rules for sha, r in human.items()),
            "inventories": {str(p): sha256_file(p) for p in args.images}}
        sidecar = args.output.with_suffix(".json")
        if sidecar.exists() and json.loads(sidecar.read_text()) != manifest:
            raise ValueError(f"{sidecar} already exists with different provenance; use a new output path")
        atomic_write_json(sidecar, manifest, durable=True)
        Reviews(args.output)


def cached_scores(path):
    protocol = json.loads((path.parent / "protocol.json").read_text())
    with (path.parent / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return protocol, load_cache(path, protocol_id(protocol))


def sample(args):
    if args.output.exists():
        raise ValueError("use a new annotation directory")
    protocol, scores = cached_scores(args.results)
    inventory = read_csv(args.images)
    if inventory.keys() - scores.keys():
        raise ValueError("score this inventory before sampling")
    thresholds = load_thresholds(args.thresholds, [r["dataset"] for r in inventory.values()])
    groups, rows = defaultdict(list), []
    for row in inventory.values():
        score = scores[row["image_sha256"]]["bad_score"]
        predicted = "reject" if score >= thresholds[row["dataset"]] else "keep"
        groups[row["dataset"], predicted].append({**row, "predicted": predicted, "bad_score": score})
    for _, group in sorted(groups.items()):
        rows.extend(sorted(group, key=lambda r: protocol_id([args.seed, r["image_sha256"]]))[:args.per_decision])
    rows.sort(key=lambda r: protocol_id([args.seed, "blind-order", r["image_sha256"]]))
    write_csv(args.output / "sample_manifest.csv", rows)
    # The page shows images only, never model scores, predictions or previous human labels.
    public = [{"image_sha256": r["image_sha256"], "image": os.path.relpath(r["path"], args.output),
               "dataset": r["dataset"]} for r in rows]
    (args.output / "items.js").write_text("const ITEMS = " + json.dumps(public) + ";\n")
    shutil.copyfile(Path(__file__).parent / "review_ui/image_labels.html", args.output / "index.html")
    atomic_write_json(args.output / "sampling.json", {"protocol_version": protocol_id(protocol),
        "inventory_sha256": sha256_file(args.images), "seed": args.seed, "thresholds": thresholds,
        "requested_per_decision": args.per_decision,
        "available": {f"{ds}/{decision}": len(groups[ds, decision]) for ds in thresholds for decision in ("keep", "reject")},
        "sampled": dict(Counter(f"{r['dataset']}/{r['predicted']}" for r in rows)),
        "scope": "Separate random checks of predicted keep and reject; do not pool them as a population defect rate."})
    print(json.dumps({"images": len(rows), "page": str(args.output / "index.html")}), flush=True)


def calibrate(args):
    samples, labels = read_csv(args.sample), read_csv(args.labels)
    sampling = json.loads((args.sample.parent / "sampling.json").read_text())
    datasets = {ds: {decision: dict(sampled=0, human_keep=0, human_reject=0, uncertain=0, missing=0)
                     for decision in ("keep", "reject")} for ds in sampling["thresholds"]}
    for sha, sample_row in samples.items():
        group = datasets[sample_row["dataset"]][sample_row["predicted"]]
        group["sampled"] += 1
        row = labels.get(sha)
        if row is None:
            group["missing"] += 1
        elif row["decision"] == "" and row["reason"] == "uncertain":
            group["uncertain"] += 1
        else:
            validate_review(sha, row["decision"], row["reason"])
            group["human_" + row["decision"]] += 1
    report = {"protocol_version": sampling["protocol_version"], "thresholds": sampling["thresholds"],
        "labels_sha256": sha256_file(args.labels), "sample_sha256": sha256_file(args.sample), "datasets": datasets,
        "scope": "Reject/human_keep counts unwanted removals; keep/human_reject counts residual defects in this sample. No population accuracy claim or automatic threshold choice."}
    if args.report.exists():
        raise ValueError("use a new calibration report path")
    atomic_write_json(args.report, report, durable=True)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="write a hash-deduplicated original-image inventory")
    p.add_argument("--meta", nargs="+", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=["r2r", "gs", "b1k"], help="only prepare these datasets")
    p.add_argument("--labels", type=Path, help="restrict inventory to these seed hashes")
    p.add_argument("--per-dataset", type=int, default=0, help="0: all; otherwise scene-balanced sample")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rules-output", type=Path, help="optional dark>20%% / GS mean>=180,std<=15 exclusions before scoring")
    p = commands.add_parser("audit", help="score only the supplied inventory on CUDA_VISIBLE_DEVICES (default: 1)")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--cache", type=Path, default=Path("data/analysis/image_review/cache"))
    p.add_argument("--batch", type=int, default=64, help="requests queued to vLLM, with the next batch decoded ahead")
    p.add_argument("--report", type=Path, required=True)
    p = commands.add_parser("sample", help="write a blind, mixed annotation page")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True, help="JSON mapping dataset to bad-score threshold")
    p.add_argument("--per-decision", type=int, default=50, help="random images per dataset and predicted decision")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("calibrate", help="count human decisions on the blind check; does not tune thresholds")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p = commands.add_parser("merge", help="freeze model results with human overrides")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--human", type=Path)
    p.add_argument("--rules", type=Path, help="mandatory hash-keyed dark/GS quality exclusions; override human and model labels")
    p.add_argument("--images", nargs="+", type=Path, required=True, help="freeze only these inventories; require complete coverage")
    p.add_argument("--thresholds", type=Path, required=True, help="JSON mapping dataset to bad-score threshold")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "batch", 1) < 1 or getattr(args, "per_decision", 1) < 1:
        parser.error("invalid batch/sample size")
    if getattr(args, "per_dataset", 0) < 0:
        parser.error("per-dataset must be nonnegative")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
