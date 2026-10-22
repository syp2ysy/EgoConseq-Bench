#!/usr/bin/env python3
"""Shared QA input, HTTP transport, response storage and resume logic."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("full", "no_radius", "no_height", "no_fov", "no_parameters")
ANSWER_ONLY_PROMPT = ("Return only the final answer in the requested format. "
                      "Do not include explanations, calculations, or reasoning.")


def completion_metadata(token_ids, eos_token_id, max_new_tokens, *, eos_omitted=False):
    """Classify generated IDs only; completion_tokens includes a consumed EOS.

    eos_omitted is for RxBrain's pinned loop: EOS is its only early exit and
    is removed before decode(). Do not use it for an arbitrary text helper.
    """
    count = len(token_ids)
    if not isinstance(max_new_tokens, int) or not 0 <= count <= max_new_tokens or max_new_tokens < 1:
        raise RuntimeError("Generated token count exceeds or lacks a valid generation budget")
    if eos_omitted:
        stopped = count < max_new_tokens
        count += int(stopped)
    else:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or [])
        stopped = bool(token_ids) and token_ids[-1] in eos_ids
        if not stopped and count < max_new_tokens:
            raise RuntimeError("Generation ended before its budget without EOS")
    return {"finish_reason": "stop" if stopped else "length", "completion_tokens": count}


@contextmanager
def record_generation(model):
    """Observe single-sample, new-ID-only generate() calls without changing them.

    RoboRefer and SpatioLM pass inputs_embeds to their language models, so their
    returned sequences contain no prompt IDs. Keep their official text decoding.
    """
    original = model.generate
    metadata = {}

    def generate(*args, **kwargs):
        output = original(*args, **kwargs)
        config = kwargs.get("generation_config") or getattr(model, "generation_config", None)
        eos = kwargs.get("eos_token_id", getattr(config, "eos_token_id", None))
        budget = kwargs.get("max_new_tokens", getattr(config, "max_new_tokens", None))
        metadata.update(completion_metadata(output[0].tolist(), eos, budget))
        return output

    model.generate = generate
    try:
        yield metadata
    finally:
        model.generate = original
    if not metadata:
        raise RuntimeError("No generation output was recorded")


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


def require_no_null(value, path="root"):
    if value is None:
        raise ValueError(f"Unexpected null at {path}; omit optional metadata or record an explicit answer status")
    if isinstance(value, dict):
        for key, child in value.items():
            require_no_null(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_no_null(child, f"{path}[{index}]")


def save_output(path, result, *, allow_null=False):
    if not allow_null:
        # Author processor configs use null for options that are not enabled.
        # Preserve those settings verbatim; response fields still require values.
        require_no_null({key: value for key, value in result.items() if key != "image_processor"})
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def benchmark_sets(root, sets=None):
    """{name: (QA directory, image root, declaration)} of the release sets to evaluate (default: seen)."""
    release = root / "release.json"
    manifest = json.loads(release.read_text())
    names = list(sets or ["seen"])
    unknown = sorted(set(names) - set(manifest["sets"]))
    if unknown:
        raise ValueError(f"{release} declares no set {unknown}")
    return {name: (root / manifest["sets"][name]["dir"], root / manifest["sets"][name]["image_root"],
                   manifest["sets"][name])
            for name in names}


def load_questions(root, limit_per_task=0, variant="full", sets=None):
    rows, sources, counts = [], {}, Counter()
    declared = list(json.loads((root / "release.json").read_text())["files"].values())
    filename = "QA.json" if variant == "full" else f"QA_{variant}.json"
    for split, (directory, image_root, _) in benchmark_sets(root, sets).items():
        path = directory / filename
        content = path.read_bytes()
        sources[split] = {"qa_path": path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(content).hexdigest()}
        relative = path.relative_to(root).parts
        expected = [entry["sha256"] for entry in declared
                    if Path(entry["path"]).parts[-len(relative):] == relative]
        if expected != [sources[split]["sha256"]]:
            raise ValueError(f"QA hash does not match frozen release: {path}")
        kept = set()
        for item in json.loads(content):
            unit = item.get("pair_id", item["id"])  # the two items of a radius pair stay together
            key = split, item["task_id"]
            if limit_per_task and unit not in kept and counts[key] >= limit_per_task:
                continue
            counts[key] += unit not in kept
            kept.add(unit)
            rows.append((split, image_root, item))
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
                **generation_settings(args), "max_tokens": args.max_new_tokens,
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
    metadata = {"base_url": args.base_url, "served_model": served["id"], "model_path": str(checkpoint)}
    if checkpoint.parent.name == "snapshots":
        metadata["model_revision"] = checkpoint.name
    return predict, metadata


def add_generation_arguments(parser):
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)


def generation_settings(args):
    return {name: getattr(args, name) for name in
            ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty", "seed")}


def run(create_predictor, *, model, backend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=model)
    parser.add_argument("--backend", choices=("api",) if backend == "api" else (backend, "api"), default=backend)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--sets", nargs="+", help="QA sets of a release root (default: seen)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    add_generation_arguments(parser)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                        help="auto keeps the model chat template default; on/off requests enable_thinking")
    parser.add_argument("--answer-only", action="store_true",
                        help="Append a uniform final-answer-only instruction without modifying the QA files")
    parser.add_argument("--limit-per-task", type=int, default=0,
                        help="QA per task per split for a smoke run; 0 uses all QA")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1",
                        help="Local OpenAI-compatible text-generation endpoint")
    parser.add_argument("--check-only", action="store_true",
                        help="No model/API call: exit 0 if complete, 3 if questions remain")
    parser.add_argument("--official-settings", type=Path,
                        help="Reviewed author inference settings for this checkpoint")
    args = parser.parse_args()
    args.official_settings = json.loads(args.official_settings.read_text()) if args.official_settings else {}
    official_generation = args.official_settings.get("generation", {})
    if args.official_settings:
        args.max_new_tokens = official_generation.get("max_new_tokens", official_generation.get("max_tokens", args.max_new_tokens))
        for name in generation_settings(args):
            if name in official_generation:
                setattr(args, name, official_generation[name])
    if not args.official_settings and args.backend != "api" and (args.temperature != 0 or args.presence_penalty != 0):
        parser.error("Custom sampling is supported by the API backend; use --backend api")
    if min(args.batch_size, args.api_workers, args.max_new_tokens) < 1 or args.limit_per_task < 0:
        parser.error("batch-size, api-workers and max-new-tokens must be positive; limit-per-task must be nonnegative")
    if args.backend == "spatiolm" and args.thinking != "auto":
        parser.error("SpatioLM's official chat interface has no enable_thinking switch; use --thinking auto.")
    args.chat_template_kwargs = {} if args.thinking == "auto" else {"enable_thinking": args.thinking == "on"}
    if args.official_settings:
        args.chat_template_kwargs = {**args.official_settings.get("chat_template_kwargs", {}), **args.chat_template_kwargs}

    if args.output is None:
        variant = "" if args.variant == "full" else f"_{args.variant}"
        mode = "" if args.thinking == "auto" else f"_thinking_{args.thinking}"
        mode += "_answer_only" if args.answer_only else ""
        suffix = "_smoke" if args.limit_per_task else ""
        sets = f"_{'+'.join(args.sets)}" if args.sets else ""
        args.output = args.benchmark_root / "inference/responses" / f"{Path(args.model).name}{sets}{variant}{mode}{suffix}_response.json"
    rows, sources = load_questions(args.benchmark_root, args.limit_per_task, args.variant, args.sets)
    if args.answer_only:
        rows = [(split, root, {**item, "messages": [
            {**message, "content": message["content"] + "\n\n" + ANSWER_ONLY_PROMPT}
            if message["role"] == "user" else message for message in item["messages"]]})
            for split, root, item in rows]
    config = {"model": args.model, "backend": args.backend, "variant": args.variant,
              **({"sets": args.sets} if args.sets else {}),
              "sources": sources, "do_sample": args.temperature > 0,
              "generation": generation_settings(args),
              "chat_template_kwargs": args.chat_template_kwargs,
              "max_new_tokens": args.max_new_tokens, "limit_per_task": args.limit_per_task,
              "inference_code_sha256": hashlib.sha256(
                  Path(__file__).read_bytes() + Path(sys.argv[0]).read_bytes()).hexdigest()}
    if args.answer_only:
        config["answer_only_prompt"] = ANSWER_ONLY_PROMPT
    if args.official_settings:
        config["official_settings"] = args.official_settings
        config["generation"] = official_generation
        config["do_sample"] = official_generation.get("do_sample", args.temperature > 0)
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
    metadata = {key: value for key, value in metadata.items() if value is not None}
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
        # Author processor/generation metadata can legitimately contain nulls.
        save_output(args.output, result, allow_null=bool(args.official_settings))
        print(f"Saved {len(result['predictions'])}/{len(rows)} answers", flush=True)
    print(f"Completed: {args.output}", flush=True)
