#!/usr/bin/env python3
"""Validate responses, evaluate one model at a time, and rebuild compatible tables."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from copy import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Lock
import time

BENCHMARK = Path(__file__).resolve().parents[1]
REPO = BENCHMARK.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BENCHMARK / "inference"))
sys.path.insert(0, str(BENCHMARK / "evaluation"))

from common import benchmark_sets, load_questions, save_output
from post_QA import answers
import extract
from extract import (CATEGORY_SYSTEM, GENERATION, JUDGE, UNPARSED, api_predictor, canonical_ground_truth,
                     extract_response, judge_entry, score_extracted)
from table import combine_reports, read_report, write_report

CHECKPOINT_INTERVAL_SECONDS = 30


def scoring_fingerprint():
    paths = (Path(__file__), Path(extract.__file__), Path(answers.__file__), Path(__file__).with_name("table.py"))
    return hashlib.sha256(b"\0".join(path.read_bytes() for path in paths)).hexdigest()


def checkpoint_identity(checkpoint):
    """Record the resolved snapshot and detect local checkpoint/config changes."""
    root = Path(checkpoint).resolve()
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix in {".json", ".jinja", ".safetensors", ".model"}):
            stat = path.stat()
            entry = [path.name, stat.st_size, stat.st_mtime_ns, str(path.resolve())]
            if path.suffix in {".json", ".jinja"}:
                entry.append(hashlib.sha256(path.read_bytes()).hexdigest())
            files.append(entry)
    digest = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
    revision = root.name if root.parent.name == "snapshots" else "local-manifest:" + digest
    return {"checkpoint": str(root), "model_revision": revision, "checkpoint_manifest_sha256": digest}


def make_protocol():
    from run import cached_checkpoint
    identity = checkpoint_identity(cached_checkpoint(JUDGE['model'], revision=JUDGE['revision']))
    return {"evaluation": "extract_and_judge_v11", "protocol_sha256": scoring_fingerprint(),
            "truncated_response_policy": "closed_selected_answer_only",
            "category_policy": "coarse_is_a_v3",
            "judge": {"model": JUDGE['model'], **identity, "enable_thinking": JUDGE['enable_thinking'],
                      "api_workers": JUDGE['api_workers'], **GENERATION, "max_new_tokens": JUDGE['max_new_tokens']}}


def load_inputs(paths, root, sets=None):
    """Validate the entire batch before any rules, judge requests or result writes.

    sets: the QA sets of a release root every response must answer (default: as the responses declare).
    """
    models, questions, qa_cache = [], {}, {}
    reference = None
    for path in paths:
        content = path.read_bytes()
        document = json.loads(content)
        variant = document["config"]["variant"]
        if sets is not None and document["config"].get("sets") != sets:
            raise ValueError(f"{path.name}: responses must answer the same QA sets ({sets})")
        sets = document["config"].get("sets")
        if variant not in qa_cache:
            qa, sources = load_questions(root, variant=variant, sets=sets)
            for _, _, item in qa:
                item["_canonical_ground_truth"] = canonical_ground_truth(item)
            qa_cache[variant] = qa, sources
        qa, sources = qa_cache[variant]
        items = {(split, q["id"]): q for split, _, q in qa}
        predictions = {(p["split"], p["id"]): p for p in document["predictions"]}
        if (len(items) != len(qa) or len(predictions) != len(document["predictions"])
                or items.keys() != predictions.keys() or document["config"]["sources"] != sources):
            raise ValueError(f"QA/response hashes or question IDs do not match: {path.name}")
        for split, (_, _, declared) in benchmark_sets(root, sets).items():
            if {q["task_id"] for s, _, q in qa if s == split} != set(declared["tasks"]):
                raise ValueError(f"{path.name}: {split} must contain every declared task")
        identity = {(split, q["id"], q["task_id"], q["dataset"], answers._message(q, "assistant"))
                    for split, _, q in qa}
        if reference is not None and identity != reference:
            raise ValueError("Models must use the same questions and ground truth")
        reference = identity
        name = path.stem.removesuffix("_response")
        if any(m["id"] == name for m in models):
            raise ValueError(f"Duplicate response name: {name}")
        model = {"id": name, "model": document["config"]["model"], "variant": variant,
                 "model_id": name.partition('_' + '+'.join(sets))[0],
                 "response_path": str(path.resolve()), "response_sha256": hashlib.sha256(content).hexdigest(),
                 "sources": sources, "inference": {k: v for k, v in document["config"].items()
                                                     if k not in {"model", "sources", "variant"}}, "results": [],
                 "release": {'path': str((root / 'release.json').resolve()),
                             'sha256': hashlib.sha256((root / 'release.json').read_bytes()).hexdigest()}}
        if document.get('model_revision'):
            model['model_revision'] = document['model_revision']
        for key, item in items.items():
            prediction = predictions[key]
            if any(prediction[k] != item[k] for k in ("task_id", "dataset")):
                raise ValueError(f"Response task/dataset mismatch: {name}/{key}")
            entry = {"split": key[0], "id": key[1], "dataset": item["dataset"], "task_id": item["task_id"],
                     "ground_truth": answers._message(item, "assistant"),
                     "response_finish_reason": prediction["finish_reason"], "status": "pending",
                     **{k: item[k] for k in ("unit", "pair_id") if k in item}}
            if item["task_id"] == "A1":
                entry["gt_collision"] = item["_canonical_ground_truth"] == "collision"
            model["results"].append(entry)
            questions[(name, *key)] = (item, prediction["answer"])
        models.append(model)
    return models, questions


def model_config(protocol, model):
    return {**protocol, "inputs": [{k: v for k, v in model.items() if k != "results"}]}


def read_saved(model, protocol, directory):
    """Reuse finished or checkpointed work only under an identical evaluation config."""
    config = model_config(protocol, model)
    final = directory / "results"
    checkpoint = directory / "work/checkpoint.json"
    if (final / "results.json").exists():
        report, saved = read_report(final)
        if report["config"] != config:
            raise ValueError(f"Results belong to another evaluation: {final}")
        return saved[0], config, True
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["config"] != config:
            raise ValueError(f"Checkpoint config belongs to another evaluation: {checkpoint}")
        return state["model"], config, False
    return model, config, False


@contextmanager
def judge_session(args, protocol):
    """Own the local judge on this GPU."""
    from run import local_server
    args.model = protocol["judge"]["checkpoint"]
    with open(f"/tmp/pbench-gpu-{args.gpus.replace(',', '-')}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with local_server(args.model, args, args.output_dir / "judge.log", judge=True) as url:
            args.base_url = url
            yield api_predictor(args)


def evaluate_model(model, questions, config, directory, get_predictor, cache, cache_lock, save_cache):
    """Finish this model, in checkpointed chunks, before starting another."""
    checkpoint = directory / "work/checkpoint.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = directory / "eval.log"
    rows = model["results"]

    def save(status):
        progress = {"scored": sum(r["status"] == "scored" for r in rows), "total": len(rows),
                    "errors": sum(r["status"] == "error" for r in rows)}
        state = {"config": config, "model": model, "status": status, "progress": progress,
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        save_output(checkpoint, state, allow_null=True)
        save_cache()
        line = f"{state['updated_at']} {model['id']} {status}: {progress}"
        with log_path.open("a") as log:
            log.write(line + "\n")
        print(line, flush=True)

    def score_indices(indices):
        pending = {}
        for index in indices:
            row = rows[index]
            if row["status"] == "scored":
                continue
            item, response = questions[model["id"], row["split"], row["id"]]
            value = extract_response(item, response, row["response_finish_reason"])
            if value is not UNPARSED and (row["task_id"] != "A3" or value["reason"] != "answered"):
                row.update(score_extracted(item, value), status="scored", extraction_method="rules")
                row.pop("error", None)
            else:
                pending[index] = value
        if pending:
            def predict(request):
                return get_predictor()(request)
            with ThreadPoolExecutor(max_workers=JUDGE['api_workers']) as pool:
                futures = {pool.submit(judge_entry, rows[i], *questions[model["id"], rows[i]["split"], rows[i]["id"]],
                                       predict, cache, cache_lock, extracted=value): i for i, value in pending.items()}
                last_save, completed = time.monotonic(), 0
                while futures:
                    timeout = max(0, CHECKPOINT_INTERVAL_SECONDS - (time.monotonic() - last_save))
                    done, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
                    for future in done:
                        rows[futures.pop(future)] = future.result()
                    completed += len(done)
                    if completed >= 16 or time.monotonic() - last_save >= CHECKPOINT_INTERVAL_SECONDS:
                        save("running")
                        last_save, completed = time.monotonic(), 0
        save("running")
        if any(rows[i]["status"] != "scored" for i in indices):
            raise RuntimeError(f"Judge failed for {model['id']}; resume from {checkpoint}")

    save("running")
    try:
        rest = [i for i in range(len(rows)) if rows[i]["status"] != "scored"]
        for offset in range(0, len(rest), 128):
            score_indices(rest[offset:offset + 128])
        write_report([model], config, directory / "results")
        save("complete")
        checkpoint.unlink()
        checkpoint.parent.rmdir()
    except BaseException:
        save("failed")
        raise


def run_evaluations(args):
    gpus = args.gpus.split(',')
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('Use distinct GPU indices, for example --gpus 0,1,2,3')
    models, questions = load_inputs(args.responses, args.benchmark_root, args.sets)
    protocol = make_protocol()
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol differs; use a new --output-dir")
    states = [read_saved(m, protocol, args.output_dir / "models" / m["id"]) for m in models]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_output(protocol_path, protocol)
    # One category table for every set: a category pair always receives the same judgment.
    cache_path = args.category_cache
    judge = {"judge": protocol["judge"], "prompt": CATEGORY_SYSTEM}
    cache = {}
    if cache_path.exists():
        document = json.loads(cache_path.read_text())
        if {k: document[k] for k in judge} != judge:
            raise ValueError(f"{cache_path} belongs to another judge or category prompt")
        cache = document["pairs"]
    cache_lock = extract.CategoryLocks()
    write_lock = Lock()

    def save_cache():
        with write_lock:
            with cache_lock.guard:
                snapshot = dict(cache)
            save_output(cache_path, {**judge, "pairs": snapshot}, allow_null=True)

    def worker(gpu, jobs):
        worker_args = copy(args)
        worker_args.gpus = gpu
        worker_args.environment = dict(args.environment, CUDA_VISIBLE_DEVICES=gpu)
        worker_args.output_dir = args.output_dir / 'judges' / f'gpu{gpu}'
        with ExitStack() as stack:
            predictor, startup_error, startup_lock = None, None, Lock()
            def get_predictor():
                nonlocal predictor, startup_error
                with startup_lock:
                    if startup_error is not None:
                        raise RuntimeError(f'Judge startup failed: {startup_error}') from startup_error
                    if predictor is None:
                        worker_args.output_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            predictor = stack.enter_context(judge_session(worker_args, protocol))
                        except Exception as error:
                            startup_error = error
                            raise RuntimeError(f'Judge startup failed: {error}') from error
                return predictor
            for model, config, complete in jobs:
                directory = args.output_dir / 'models' / model['id']
                if not complete:
                    evaluate_model(model, questions, config, directory, get_predictor, cache, cache_lock,
                                   save_cache)
                else:
                    print(f"Already complete: {model['id']}", flush=True)
                    work = directory / 'work'
                    (work / 'checkpoint.json').unlink(missing_ok=True)
                    if work.exists() and not any(work.iterdir()):
                        work.rmdir()

    if len(gpus) == 1:
        worker(gpus[0], states)
    else:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, states[i::len(gpus)])
                       for i, gpu in enumerate(gpus) if i < len(states)]
            for future in futures:
                future.result()
    completed = [args.output_dir / 'models' / model['id'] / 'results' for model, _, _ in states]
    combine_reports(completed, args.output_dir / "summary", replace=True)
    print(f"Complete: {len(completed)} models; {args.output_dir / 'summary/results.json'}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, nargs="+",
                        help="Default: all saved responses under the benchmark inference directory")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK)
    parser.add_argument("--sets", nargs="+", help="QA sets of a release root the responses answer")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_dir = (args.output_dir or args.benchmark_root / "evaluation/runs").resolve()
    if args.responses is None:
        args.responses = sorted((args.benchmark_root / "inference").rglob("*_response.json"))
        if args.sets:
            args.responses = [p for p in args.responses
                              if json.loads(p.read_text())["config"].get("sets") == args.sets]
    if not args.responses:
        parser.error("No saved responses found")
    args.responses = [p.resolve() for p in args.responses]
    args.category_cache = args.output_dir / "category_cache.json"
    args.environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONDONTWRITEBYTECODE="1",
                            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.background:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log = args.output_dir / "eval.log"
        command = [sys.executable, "-B", str(Path(__file__).resolve()), *[v for v in sys.argv[1:] if v != "--background"]]
        with log.open("a") as output:
            process = subprocess.Popen(command, env=args.environment, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Background PID {process.pid}; log: {log}")
        return 0
    try:
        groups = {}
        for path in args.responses:
            names = tuple(json.loads(path.read_text())["config"].get("sets", []))
            if args.sets is not None and names != tuple(args.sets):
                raise ValueError(f"{path.name}: responses must answer the requested QA sets")
            groups.setdefault(names, []).append(path)
        if len(groups) > 1 or args.output_dir == args.benchmark_root / "evaluation/runs":
            summaries = []
            for names, paths in sorted(groups.items()):
                group = copy(args)
                group.responses, group.sets = paths, list(names) or None
                group.output_dir = args.output_dir / ("+".join(names) or "default")
                status = _run_locked(group)
                if status:
                    return status
                summaries.append(group.output_dir / "summary")
            output = (args.benchmark_root / "evaluation/results" if
                      args.output_dir == args.benchmark_root / "evaluation/runs" else args.output_dir / "summary")
            combine_reports(summaries, output, replace=True)
            return 0
        return _run_locked(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


def _run_locked(args):
    """One output directory belongs to one set group, even when all sets share a root."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An evaluation already owns this output directory")
        return run_evaluations(args)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
