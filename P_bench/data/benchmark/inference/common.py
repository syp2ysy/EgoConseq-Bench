#!/usr/bin/env python3
"""Shared QA input, HTTP transport, response storage and resume logic."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("full", "no_radius", "no_height", "no_fov", "no_parameters")


def build_messages(item, image_root):
    """Use the saved prompts and images verbatim; exclude the assistant GT."""
    images = iter(item["images"])
    messages = []
    for message in item["messages"]:
        if message["role"] not in {"system", "user"}:
            continue
        content = []
        for index, text in enumerate(message["content"].split("<image>")):
            if index:
                content.append({"type": "image", "image": str(
                    (image_root / next(images)).resolve())})
            if text:
                content.append({"type": "text", "text": text})
        messages.append({"role": message["role"], "content": content})
    return messages


def save_output(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_questions(root, limit_per_task=0, variant="full"):
    rows, sources, counts = [], {}, Counter()
    filename = "QA.json" if variant == "full" else f"QA_{variant}.json"
    for split in ("seen", "unseen"):
        path = root / "benchmark" / split / filename
        content = path.read_bytes()
        sources[split] = {"qa_path": path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(content).hexdigest()}
        for item in json.loads(content):
            key = split, item["task_id"]
            if limit_per_task and counts[key] >= limit_per_task:
                continue
            counts[key] += 1
            rows.append((split, path.parent, item))
    return rows, sources


def api_messages(item, image_root):
    messages = build_messages(item, image_root)
    for message in messages:
        for block in message["content"]:
            if block["type"] == "image":
                path = Path(block.pop("image"))
                mime = mimetypes.guess_type(path.name)[0]
                block.update(type="image_url", image_url={"url":
                    f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()})
    return messages


def api_predictor(args):
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ.get("INFER_API_KEY", "EMPTY")}
    with urlopen(Request(args.base_url.rstrip("/") + "/models", headers=headers), timeout=30) as response:
        models = json.load(response)["data"]
    served = next((m for m in models if m["id"] == args.model), None)
    if served is None:
        raise ValueError(f"Endpoint does not serve the requested model: {args.model}")

    def predict_one(row):
        _, root, item = row
        body = {"model": args.model, "messages": api_messages(item, root),
                "temperature": 0, "max_tokens": args.max_new_tokens,
                "chat_template_kwargs": args.chat_template_kwargs}
        request = Request(args.base_url.rstrip("/") + "/chat/completions",
                          data=json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=1800) as response:
            choice = json.load(response)["choices"][0]
        message = choice["message"]
        result = {"answer": message.get("content") or "", "finish_reason": choice["finish_reason"]}
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            result["reasoning"] = reasoning
        return result

    def predict(batch):
        with ThreadPoolExecutor(max_workers=getattr(args, "api_workers", 1)) as pool:
            return list(pool.map(predict_one, batch))

    checkpoint = Path(served.get("root") or args.model)
    return predict, {"base_url": args.base_url, "served_model": served["id"],
                     "model_path": str(checkpoint),
                     "model_revision": checkpoint.name if checkpoint.parent.name == "snapshots" else None}


def run(create_predictor, *, model, backend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=model)
    parser.add_argument("--backend", choices=("api",) if backend == "api" else (backend, "api"), default=backend)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                        help="auto keeps the model chat template default; on/off requests enable_thinking")
    parser.add_argument("--limit-per-task", type=int, default=0,
                        help="QA per task per split for a smoke run; 0 uses all QA")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1",
                        help="Local OpenAI-compatible text-generation endpoint")
    parser.add_argument("--check-only", action="store_true",
                        help="No model/API call: exit 0 if complete, 3 if questions remain")
    args = parser.parse_args()
    if min(args.batch_size, args.api_workers, args.max_new_tokens) < 1 or args.limit_per_task < 0:
        parser.error("batch-size, api-workers and max-new-tokens must be positive; limit-per-task must be nonnegative")
    if args.backend == "spatiolm" and args.thinking != "auto":
        parser.error("SpatioLM's official chat interface has no enable_thinking switch; use --thinking auto.")
    args.chat_template_kwargs = {} if args.thinking == "auto" else {"enable_thinking": args.thinking == "on"}

    if args.output is None:
        variant = "" if args.variant == "full" else f"_{args.variant}"
        mode = "" if args.thinking == "auto" else f"_thinking_{args.thinking}"
        suffix = "_smoke" if args.limit_per_task else ""
        args.output = args.benchmark_root / "inference/responses" / f"{Path(args.model).name}{variant}{mode}{suffix}_response.json"
    rows, sources = load_questions(args.benchmark_root, args.limit_per_task, args.variant)
    config = {"model": args.model, "backend": args.backend, "variant": args.variant,
              "sources": sources, "do_sample": False,
              "chat_template_kwargs": args.chat_template_kwargs,
              "max_new_tokens": args.max_new_tokens, "limit_per_task": args.limit_per_task,
              "inference_code_sha256": hashlib.sha256(
                  Path(__file__).read_bytes() + Path(sys.argv[0]).read_bytes()).hexdigest()}
    result = {"config": config, "predictions": []}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result["config"] != config:
            parser.error("Existing responses use different inputs/model/settings; choose a new output.")
    completed = {(p["split"], p["id"]) for p in result["predictions"]}
    questions = {(split, item["id"]): item for split, _, item in rows}
    if len(completed) != len(result["predictions"]) or not completed.issubset(questions):
        parser.error("Existing responses contain duplicated or foreign question IDs")
    for p in result["predictions"]:
        item = questions[p["split"], p["id"]]
        if (p["task_id"], p["dataset"]) != (item["task_id"], item["dataset"]):
            parser.error("Existing response task/dataset does not match its question")
    pending = [row for row in rows if (row[0], row[2]["id"]) not in completed]
    print(f"Questions: {len(rows)}; remaining: {len(pending)}", flush=True)
    if args.check_only:
        sys.exit(3 if pending else 0)
    if not pending:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)

    factory = api_predictor if args.backend == "api" and backend != "api" else create_predictor
    predict, metadata = factory(args)
    for key in ("model_revision", "model_path", "image_processor"):
        if key in result and result[key] != metadata.get(key):
            parser.error(f"Model runtime changed ({key}); use a different output")
    result.update(metadata)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        predictions = predict(batch)
        if len(predictions) != len(batch):
            raise ValueError("Predictor returned a different number of answers than questions")
        for (split, _, item), prediction in zip(batch, predictions):
            result["predictions"].append({
                "split": split, "id": item["id"], "dataset": item["dataset"],
                "task_id": item["task_id"], **prediction,
            })
        save_output(args.output, result)
        print(f"Saved {len(result['predictions'])}/{len(rows)} answers", flush=True)
    print(f"Completed: {args.output}", flush=True)
