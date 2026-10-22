#!/usr/bin/env python3
"""Sequential single-GPU inference; evaluation/eval.py scores the saved responses."""

import argparse
from contextlib import contextmanager, nullcontext
from copy import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from urllib.error import URLError
from urllib.request import urlopen

from common import VARIANTS, add_generation_arguments, generation_settings, load_questions, save_output

ROOT = Path(__file__).resolve().parent
# alias: (script, checkpoint, backend, Python environment, default tensor parallelism)
MODELS = {
    "qwen25vl-3b": ("infer_transformers.py", "Qwen/Qwen2.5-VL-3B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen25vl-sat": ("infer_transformers.py", "array/Qwen2.5-VL-SAT", "transformers", "QWEN_PYTHON", 1),
    "cambrian-s-3b": ("infer_cambrian.py", "nyu-visionx/Cambrian-S-3B", "cambrian", "CAMBRIAN_PYTHON", 1),
    "cambrian-s-7b": ("infer_cambrian.py", "nyu-visionx/Cambrian-S-7B", "cambrian", "CAMBRIAN_PYTHON", 1),
    "qwen3vl-2b": ("infer_transformers.py", "Qwen/Qwen3-VL-2B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen3vl-4b": ("infer_transformers.py", "Qwen/Qwen3-VL-4B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen3vl-8b": ("infer_transformers.py", str(ROOT.parents[2] / "weights" /
                    "Qwen3-VL-8B-Instruct"), "transformers", "QWEN_PYTHON", 1),
    "qwen35-2b": ("infer_transformers.py", "Qwen/Qwen3.5-2B", "transformers", "MODERN_PYTHON", 1),
    "rynnbrain11-2b": ("infer_transformers.py", "Alibaba-DAMO-Academy/RynnBrain1.1-2B", "transformers", "MODERN_PYTHON", 1),
    "robobrain25-4b": ("infer_transformers.py", "BAAI/RoboBrain2.5-4B", "transformers", "QWEN_PYTHON", 1),
    "robointer-3b": ("infer_transformers.py", "InternRobotics/RoboInter-VLM_qwenvl25_3b", "transformers", "QWEN_PYTHON", 1),
    "cosmos-reason2-2b": ("infer_transformers.py", "nvidia/Cosmos-Reason2-2B", "transformers", "QWEN_PYTHON", 1),
    "cosmos-reason2-8b": ("infer_transformers.py", "nvidia/Cosmos-Reason2-8B", "transformers", "QWEN_PYTHON", 1),
    "embodied-r15": ("infer_transformers.py", "IffYuan/Embodied-R1.5", "transformers", "QWEN_PYTHON", 1),
    "spatiolm-8b": ("infer_spatiolm.py", "xiaomi-research/SpatioLM-Understanding-InternVL3.5", "spatiolm", "SPATIOLM_PYTHON", 1),
    "sensenova-si13-qwen3vl-8b": ("infer_transformers.py", str(ROOT.parents[2] / "weights" /
                                 "SenseNova-SI-1.3-Qwen3-VL-8B"), "api", "MODERN_PYTHON", 1),
    "cosmos3-edge": ("infer_cosmos3.py", "nvidia/Cosmos3-Edge", "transformers", "COSMOS_PYTHON", 1),
    "hy-embodied": ("infer_transformers.py", "tencent/Hy-Embodied-VLM-1.0", "api", "HY_PYTHON", 4),
}
PYTHONS = {
    "QWEN_PYTHON": "/home/zhangshan/miniconda3/envs/qwen3vl/bin/python",
    "MODERN_PYTHON": "/home/zhangshan/miniconda3/envs/vllm_serve/bin/python",
    "COSMOS_PYTHON": "/home/zhangshan/.local/share/pbench/cosmos3-runtime/bin/python",
    "CAMBRIAN_PYTHON": "/home/zhangshan/.local/share/pbench/cambrian-runtime/bin/python",
}


SMALL_MODELS = ("qwen3vl-4b", "robobrain25-4b", "robointer-3b", "rynnbrain11-2b", "cosmos3-edge")
DEFAULT_THINKING = {"rynnbrain11-2b": "off", "cosmos3-edge": "off"}


def judge_settings():
    """Load the evaluator's fixed judge settings only when a judge is needed."""
    for directory in (ROOT.parents[2], ROOT.parent / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from extract import JUDGE
    return JUDGE


def cached_checkpoint(model, revision=None):
    """Resolve local weights or a cached snapshot; an explicit revision never falls back."""
    if revision is None:
        for local in (Path(model), ROOT.parents[2] / "weights" / Path(model).name):
            if local.is_dir():
                return str(local.resolve())
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    repo = cache / ("models--" + model.replace("/", "--"))
    if revision is not None:
        checkpoint = repo / "snapshots" / revision
    else:
        ref = repo / "refs/main"
        snapshots = list((repo / "snapshots").glob("*"))
        if ref.exists():
            checkpoint = repo / "snapshots" / ref.read_text().strip()
        elif len(snapshots) == 1:
            checkpoint = snapshots[0]
        else:
            raise ValueError(f"Cannot resolve a unique cached checkpoint for {model}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing checkpoint configuration: {checkpoint}")
    if not any(checkpoint.rglob("*.safetensors")):
        raise FileNotFoundError(f"Missing cached weights: {checkpoint}")
    return str(checkpoint)


def _sets(args):
    return f"_{'+'.join(args.sets)}" if getattr(args, "sets", None) else ""


def output_path(alias, args):
    variant = _sets(args) + ("" if args.variant == "full" else f"_{args.variant}")
    thinking = args.thinking or DEFAULT_THINKING.get(alias, "auto")
    mode = "" if thinking == "auto" else f"_thinking_{thinking}"
    mode += "_answer_only" if getattr(args, "answer_only", False) else ""
    suffix = "_smoke" if args.limit_per_task else ""
    directory = (getattr(args, "response_dir", None) or
                 (args.run_dir / "responses" if args.run_dir else args.benchmark_root / "inference/responses"))
    return directory / f"{alias}{variant}{mode}{suffix}_response.json"


def server_command(model, args, port, *, judge=False, tp=1, runtime="MODERN_PYTHON"):
    python = os.environ.get(runtime, PYTHONS.get(runtime, f"<{runtime}>"))
    settings = judge_settings() if judge else {}
    command = [python, "-B", "-m", "vllm.entrypoints.openai.api_server",
               "--model", model, "--host", "127.0.0.1", "--port", str(port),
               "--dtype", "bfloat16", "--max-model-len", str(settings['max_model_len'] if judge else args.max_model_len),
               "--gpu-memory-utilization", str(settings["gpu_memory_utilization"] if judge else args.gpu_memory_utilization),
               "--max-num-seqs", str(settings["api_workers"] if judge else args.api_workers),
               "--max-num-batched-tokens", str(settings['max_num_batched_tokens'] if judge else 8192),
               "--enforce-eager", "--enable-prefix-caching",
               "--tensor-parallel-size", str(tp)]
    if judge:
        command += ["--generation-config", "vllm", "--reasoning-parser", "qwen3"]
        if settings["language_model_only"]:
            command.append("--language-model-only")
    else:
        command += ["--generation-config", "vllm",
                    "--limit-mm-per-prompt", '{"image":{"count":5,"width":640,"height":480},"video":0}',
                    "--trust-remote-code"]
        adapter = Path(model) / "adapter_config.json"
        if adapter.is_file():
            config = json.loads(adapter.read_text())
            base = cached_checkpoint(config["base_model_name_or_path"])
            command[command.index("--model") + 1] = base
            command += ["--enable-lora", "--max-lora-rank", str(config["r"]),
                        "--enable-tower-connector-lora", "--lora-modules",
                        json.dumps({"name": model, "path": model, "base_model_name": base})]
    return command


def stop_process(process):
    # A crashed parent can leave workers in its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def wait_for_server(process, port, model, log):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited ({process.returncode}); see {log}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                if any(item["id"] == model for item in json.load(response)["data"]):
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Server startup exceeded 30 minutes; see {log}")


@contextmanager
def local_server(model, args, log, *, judge=False, tp=1, runtime="MODERN_PYTHON"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = server_command(model, args, port, judge=judge, tp=tp, runtime=runtime)
    with log.open("a") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, env=args.environment, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for_server(process, port, model, log)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            stop_process(process)


def run_command(command, args, log):
    print(shlex.join(command), flush=True)
    with log.open("a") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, env=args.environment, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            stop_process(process)


def inference_command(alias, checkpoint, args):
    script, _, backend, runtime, _ = MODELS[alias]
    profile = getattr(args, "model_profiles", {}).get(alias)
    if not profile and (args.base_url or (args.serve and alias not in {
            "cosmos3-edge", "robointer-3b", "spatiolm-8b", "cambrian-s-3b", "cambrian-s-7b"})):
        backend = "api"
    python = (profile["python"] if profile else sys.executable if backend == "api" else
              os.environ.get(runtime, PYTHONS.get(runtime, f"<{runtime}>")))
    batch = min(args.batch_size, 4) if backend != "api" else args.batch_size
    sampling = [part for name, value in generation_settings(args).items()
                for part in ("--" + name.replace("_", "-"), str(value))]
    return [python, "-B", str(ROOT / script), "--model", checkpoint, "--backend", backend, *sampling,
            "--output", str(output_path(alias, args)), "--benchmark-root", str(args.benchmark_root),
            "--variant", args.variant, *(["--sets", *args.sets] if getattr(args, "sets", None) else []),
            "--batch-size", str(batch), "--api-workers", str(args.api_workers),
            "--max-new-tokens", str(args.max_new_tokens), "--thinking",
            args.thinking or DEFAULT_THINKING.get(alias, "auto"), "--limit-per-task", str(args.limit_per_task),
            *(["--answer-only"] if getattr(args, "answer_only", False) else []),
            *(["--official-settings", str(profile["settings_path"])]
              if profile and profile.get("settings_path") else [])]


def needs_work(command, args):
    code = subprocess.run([sys.executable, *command[1:], "--check-only"], env=args.environment).returncode
    if code not in (0, 3):
        raise subprocess.CalledProcessError(code, command)
    return code == 3


def gpu_processes(gpus):
    """Count GPU workloads, excluding the machine's persistent Xorg display server."""
    output = subprocess.check_output(
        ["nvidia-smi", "-i", gpus, "-q", "-x"], text=True, timeout=10)
    document = ET.fromstring(output)
    if not document.findall("gpu"):
        raise ValueError("nvidia-smi did not report the requested GPU")
    return sorted({int(process.findtext("pid")) for process in document.findall(".//process_info")
                   if not (process.findtext("type") == "G"
                           and Path(process.findtext("process_name", "")).name == "Xorg")})


def wait_for_idle(args, lock):
    required = args.idle_minutes * 60
    idle_since, previous_check, last_report = None, time.monotonic(), float("-inf")
    path = (args.run_dir or args.benchmark_root / "inference") / "pipeline_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Waiting for {args.idle_minutes:g} continuous idle minutes on GPU {args.gpus}.", flush=True)
    while True:
        now = time.monotonic()
        state = {"pid": os.getpid(), "status": "waiting_for_gpu_idle", "gpus": args.gpus,
                 "required_idle_seconds": required, "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            pids = gpu_processes(args.gpus)
            state["gpu_processes"] = pids
            if pids or now - previous_check > 10:
                idle_since = None
            elif idle_since is None:
                idle_since = now
                idle_started_at = state["updated_at"]
            if idle_since is not None and now - idle_since >= required:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    idle_since = None
                else:
                    if not gpu_processes(args.gpus):
                        args.idle_gate = {"required_seconds": required, "observed_seconds": now - idle_since,
                                          "idle_since": idle_started_at, "acquired_at": state["updated_at"]}
                        state.update(status="gpu_idle_confirmed", idle_gate=args.idle_gate)
                        save_output(path, state)
                        print(f"GPU idle confirmed: {args.idle_gate}", flush=True)
                        return
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    idle_since = None
        except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
            fcntl.flock(lock, fcntl.LOCK_UN)
            idle_since = None
            state["gpu_query_error"] = str(error)
        state["idle_seconds"] = 0 if idle_since is None else round(now - idle_since, 1)
        if now - last_report >= 60:
            save_output(path, state)
            print(f"GPU wait: {state}", flush=True)
            last_report = now
        previous_check = now
        time.sleep(1)


def execute(aliases, checkpoints, args):
    logs = args.run_dir / "logs" if args.run_dir else args.benchmark_root / "inference/logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows, sources = load_questions(args.benchmark_root, variant=args.variant, sets=getattr(args, "sets", None))
    state = {"pid": os.getpid(), "status": "running", "sources": sources,
             "questions_per_model": len(rows),
             "tasks": sorted({item["task_id"] for _, _, item in rows}),
             "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0,
                            **generation_settings(args),
                            "vllm_max_model_len": args.max_model_len},
             "models": {alias: {"checkpoint": checkpoints[alias], "variant": args.variant,
                                "sources": sources,
                                "inference": "pending",
                                "response": str(output_path(alias, args))} for alias in aliases}}
    if hasattr(args, "idle_gate"):
        state["idle_gate"] = args.idle_gate
    status_path = (args.run_dir or args.benchmark_root / "inference") / "pipeline_status.json"

    def update(alias=None, stage=None, value=None, error=None):
        if alias:
            state["models"][alias][stage] = value
            if error:
                state["models"][alias][stage + "_error"] = str(error)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_output(status_path, state)

    update()
    try:
        for alias in aliases:
            try:
                command = inference_command(alias, checkpoints[alias], args)
                update(alias, "inference", "running")
                if needs_work(command, args):
                    if command[command.index("--backend") + 1] == "api" and not args.base_url:
                        tp = args.tensor_parallel_size or MODELS[alias][4]
                        runtime = MODELS[alias][3] if MODELS[alias][2] == "api" else "MODERN_PYTHON"
                        with local_server(checkpoints[alias], args, logs / f"{alias}_server.log", tp=tp, runtime=runtime) as url:
                            run_command([*command, "--base-url", url], args, logs / f"{alias}_infer.log")
                    else:
                        if args.base_url:
                            command += ["--base-url", args.base_url]
                        run_command(command, args, logs / f"{alias}_infer.log")
                update(alias, "inference", "complete")
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                update(alias, "inference", "failed", error)
                print(f"FAILED {alias} inference: {error}", flush=True)
        state["status"] = ("complete" if all(job["inference"] == "complete" for job in state["models"].values())
                           else "failed")
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        update()
    print(f"Pipeline {state['status']}: {status_path}", flush=True)
    return 0 if state["status"] == "complete" else 1


def profile_args(args, alias):
    """Replay the full-input run's model, decoding and runtime settings."""
    job = copy(args)
    profile = args.model_profiles[alias]
    config = profile["config"]
    for name, value in profile["runtime"].items():
        setattr(job, name, value)
    # A smaller cache budget permits co-resident jobs without changing decoding.
    job.gpu_memory_utilization = min(job.gpu_memory_utilization, args.gpu_memory_utilization)
    for name, value in config["generation"].items():
        if hasattr(job, name):
            setattr(job, name, value)
    job.max_new_tokens = config["max_new_tokens"]
    job.answer_only = bool(config.get("answer_only_prompt"))
    job.thinking = profile["thinking"]
    job.batch_size = profile["runtime"]["api_batch_size"] if config["backend"] == "api" else 1
    job.environment = {**args.environment, **profile.get("env", {})}
    return job


def check_ablation_response(path, baseline, root, split, variant):
    result = json.loads(path.read_text())
    original = json.loads(Path(baseline["path"]).read_text())
    if hashlib.sha256(Path(baseline["path"]).read_bytes()).hexdigest() != baseline["sha256"]:
        raise ValueError(f"Full-input baseline changed: {baseline['path']}")
    rows, sources = load_questions(root, variant=variant, sets=[split])
    expected = {(s, q["id"]): (q["task_id"], q["dataset"]) for s, _, q in rows}
    predictions = result["predictions"]
    actual = {(p["split"], p["id"]): (p["task_id"], p["dataset"]) for p in predictions}
    if len(predictions) != len(expected) or actual != expected:
        raise ValueError(f"Incomplete or mismatched ablation responses: {path}")
    if result["config"]["variant"] != variant or result["config"]["sources"] != sources:
        raise ValueError(f"Wrong ablation inputs: {path}")
    ignored = {"variant", "sources", "inference_code_sha256"}
    for key in (set(original["config"]) | set(result["config"])) - ignored:
        if original["config"].get(key) != result["config"].get(key):
            raise ValueError(f"Ablation changed {key}: {path}")
    for key in ("image_processor", "model_revision", "model_path"):
        if original.get(key) != result.get(key):
            raise ValueError(f"Ablation changed {key}: {path}")
    return {"responses": len(predictions),
            "empty": sum(not p.get("answer", "").strip() for p in predictions),
            "truncated": sum(p.get("finish_reason") == "length" for p in predictions)}


def execute_ablations(aliases, args):
    """One model load per API sweep; each set/variant gets its own response file."""
    logs = args.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    args.response_dir.mkdir(parents=True, exist_ok=True)
    state = {"pid": os.getpid(), "status": "running", "gpus": args.gpus, "models": {}}
    status_path = args.run_dir / "pipeline_status.json"

    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_output(status_path, state)

    save()
    try:
        for alias in aliases:
            profile = args.model_profiles[alias]
            job = profile_args(args, alias)
            record = state["models"][alias] = {
                "status": "running", "jobs": {},
                "runtime": {"gpu_memory_utilization": job.gpu_memory_utilization,
                            "api_workers": job.api_workers, "batch_size": job.batch_size}}
            save()
            try:
                pending = []
                for variant in args.variants:
                    for split in args.sets:
                        current = copy(job)
                        current.variant, current.sets = variant, [split]
                        command = inference_command(alias, profile["checkpoint"], current)
                        path = output_path(alias, current)
                        entry = record["jobs"][f"{split}/{variant}"] = {
                            "status": "pending", "response": str(path)}
                        if needs_work(command, current):
                            pending.append((current, command, entry))
                        else:
                            entry.update(status="complete", **check_ablation_response(
                                path, profile["baselines"][split], args.benchmark_root, split, variant))
                save()
                server = (local_server(profile["checkpoint"], job, logs / f"{alias}_server.log")
                          if pending and profile["config"]["backend"] == "api" else nullcontext(None))
                with server as url:
                    for current, command, entry in pending:
                        entry["status"] = "running"
                        save()
                        if url:
                            command += ["--base-url", url]
                        log = logs / f"{alias}_{current.sets[0]}_{current.variant}.log"
                        run_command(command, current, log)
                        entry.update(status="complete", **check_ablation_response(
                            output_path(alias, current), profile["baselines"][current.sets[0]],
                            args.benchmark_root, current.sets[0], current.variant))
                        save()
                record["status"] = "complete"
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                record.update(status="failed", error=str(error))
                print(f"FAILED {alias}: {error}", flush=True)
            save()
        state["status"] = "complete" if all(m["status"] == "complete" for m in state["models"].values()) else "failed"
    except BaseException:
        state["status"] = "interrupted"
        raise
    finally:
        save()
    return 0 if state["status"] == "complete" else 1


def main():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--model-manifest", type=Path)
    profiles_path = early.parse_known_args()[0].model_manifest
    profiles = json.loads(profiles_path.read_text())["models"] if profiles_path else {}
    if profiles:
        MODELS.clear()
        MODELS.update({alias: (p["script"], p["checkpoint"], p["config"]["backend"], p["python"], 1)
                       for alias, p in profiles.items()})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["list", "all", "small", *MODELS])
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--model-manifest", type=Path,
                        help="Frozen model settings and full-input responses for an ablation sweep")
    parser.add_argument("--variants", nargs="+", choices=("no_radius", "no_height", "no_fov"),
                        help="Run single-parameter ablations separately for every requested set")
    parser.add_argument("--response-dir", type=Path, help="Save responses separately from queue logs")
    parser.add_argument("--sets", nargs="+", help="QA sets of a release root (default: seen)")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--wait-for-gpu", action="store_true", help="Queue behind the current pipeline on these GPUs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-workers", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    add_generation_arguments(parser)
    parser.add_argument("--run-dir", type=Path, help="Keep this run's responses, logs and status together")
    parser.add_argument("--idle-minutes", type=float, default=0,
                        help="Require this many continuously observed GPU-idle minutes before starting")
    parser.add_argument("--thinking", choices=("auto", "on", "off"))
    parser.add_argument("--answer-only", action="store_true",
                        help="Request the final answer only, identically for every model and task")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM memory budget; also caps the frozen profile's budget in ablation sweeps")
    parser.add_argument("--base-url", help="Use an existing inference API; do not start or stop it")
    parser.add_argument("--cached", action="store_true", help="Pin local cached revisions; disable network downloads")
    parser.add_argument("--serve", action="store_true", help="Use managed vLLM servers where supported; custom models use native adapters")
    parser.add_argument("--background", action="store_true", help="Detach the queue and write inference/pipeline.log")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model_profiles = profiles
    if args.models == ["list"]:
        for alias, (script, model, backend, runtime, _) in MODELS.items():
            print(f"{alias:24} {script:25} {backend:13} {runtime:16} {model}")
        return 0
    groups = {"small": SMALL_MODELS, "all": list(MODELS)}
    aliases = [alias for name in args.models for alias in groups.get(name, (name,))]
    if any(a not in MODELS for a in aliases) or len(set(aliases)) != len(aliases):
        parser.error("Choose distinct models or non-overlapping model groups")
    if args.model_manifest and not (args.variants and args.sets and args.run_dir):
        parser.error("--model-manifest requires --variants, --sets and --run-dir")
    if args.variants and (not args.model_manifest or args.limit_per_task or args.base_url):
        parser.error("Ablation sweeps require a model manifest and full sets; evaluate saved responses separately")
    if args.variants and len(set(args.variants)) != len(args.variants):
        parser.error("Choose distinct ablation variants")
    if args.idle_minutes < 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid idle duration or sampling parameters")
    if min(args.batch_size, args.api_workers, args.max_new_tokens) < 1 or args.limit_per_task < 0:
        parser.error("Invalid batch size, worker count or generation limit")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("GPU memory utilization must be in (0, 1]")
    args.benchmark_root = args.benchmark_root.resolve()
    if args.run_dir:
        args.run_dir = args.run_dir.resolve()
        args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.response_dir:
        args.response_dir = args.response_dir.resolve()
    elif args.variants:
        args.response_dir = args.benchmark_root / "inference/responses"
    args.environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONDONTWRITEBYTECODE="1")
    if args.cached:
        args.environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    checkpoints = {alias: cached_checkpoint(MODELS[alias][1]) if args.cached else MODELS[alias][1] for alias in aliases}
    if args.dry_run:
        for alias in aliases:
            if args.variants:
                job = profile_args(args, alias)
                for variant in args.variants:
                    for split in args.sets:
                        job.variant, job.sets = variant, [split]
                        print(shlex.join(inference_command(alias, checkpoints[alias], job)))
            else:
                print(shlex.join(inference_command(alias, checkpoints[alias], args)))
        return 0
    if args.background:
        log = (args.run_dir or args.benchmark_root / "inference") / "pipeline.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   *[v for v in sys.argv[1:] if v != "--background"]]
        with log.open("w") as output:
            process = subprocess.Popen(command, env=args.environment, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Background PID {process.pid}; log: {log}")
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    # One queue owns the requested GPU set.
    with open(f"/tmp/pbench-gpu-{args.gpus.replace(',', '-')}.lock", "w") as lock:
        if args.idle_minutes:
            wait_for_idle(args, lock)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_gpu else fcntl.LOCK_NB))
        return execute_ablations(aliases, args) if args.variants else execute(aliases, checkpoints, args)


if __name__ == "__main__":
    sys.exit(main())
