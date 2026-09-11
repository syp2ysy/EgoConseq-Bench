#!/usr/bin/env python3
"""Sequential single-GPU inference, followed by one shared LLM scoring pass."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from common import VARIANTS, load_questions, save_output

ROOT = Path(__file__).resolve().parent
# alias: (script, checkpoint, backend, Python environment, default tensor parallelism)
MODELS = {
    "qwen25vl-3b": ("infer_qwen25vl.py", "Qwen/Qwen2.5-VL-3B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen3vl-2b": ("infer_qwen3vl.py", "Qwen/Qwen3-VL-2B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen3vl-4b": ("infer_qwen3vl.py", "Qwen/Qwen3-VL-4B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen3vl-8b": ("infer_qwen3vl.py", "Qwen/Qwen3-VL-8B-Instruct", "transformers", "QWEN_PYTHON", 1),
    "qwen35-2b": ("infer_qwen35.py", "Qwen/Qwen3.5-2B", "transformers", "MODERN_PYTHON", 1),
    "rynnbrain11-2b": ("infer_rynnbrain.py", "Alibaba-DAMO-Academy/RynnBrain1.1-2B", "transformers", "MODERN_PYTHON", 1),
    "robobrain25-4b": ("infer_robobrain.py", "BAAI/RoboBrain2.5-4B", "transformers", "QWEN_PYTHON", 1),
    "robointer-3b": ("infer_robointer.py", "InternRobotics/RoboInter-VLM_qwenvl25_3b", "transformers", "QWEN_PYTHON", 1),
    "cosmos-reason2-2b": ("infer_cosmos_reason2.py", "nvidia/Cosmos-Reason2-2B", "transformers", "QWEN_PYTHON", 1),
    "cosmos-reason2-8b": ("infer_cosmos_reason2.py", "nvidia/Cosmos-Reason2-8B", "transformers", "QWEN_PYTHON", 1),
    "embodied-r15": ("infer_embodied_r15.py", "IffYuan/Embodied-R1.5", "transformers", "QWEN_PYTHON", 1),
    "spatiolm-8b": ("infer_spatiolm.py", "xiaomi-research/SpatioLM-Understanding-InternVL3.5", "spatiolm", "SPATIOLM_PYTHON", 1),
    "sensenova-si15-8b": ("infer_sensenova.py", "sensenova/SenseNova-SI-1.5-InternVL3-8B", "api", "MODERN_PYTHON", 1),
    "cosmos3-edge": ("infer_cosmos3.py", "nvidia/Cosmos3-Edge", "transformers", "COSMOS_PYTHON", 1),
    "hy-embodied": ("infer_hy_embodied.py", "tencent/Hy-Embodied-VLM-1.0", "api", "HY_PYTHON", 4),
}
PYTHONS = {
    "QWEN_PYTHON": "/home/zhangshan/miniconda3/envs/qwen3vl/bin/python",
    "MODERN_PYTHON": "/home/zhangshan/miniconda3/envs/vllm_serve/bin/python",
    "COSMOS_PYTHON": "/home/zhangshan/.local/share/pbench/cosmos3-runtime/bin/python",
}


SMALL_MODELS = ("qwen3vl-4b", "robobrain25-4b", "robointer-3b", "rynnbrain11-2b", "cosmos3-edge")
DEFAULT_THINKING = {"rynnbrain11-2b": "off", "cosmos3-edge": "off"}


def cached_checkpoint(model):
    """Resolve a cached revision without network access or downloads."""
    if Path(model).is_dir():
        return str(Path(model).resolve())
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    repo = cache / ("models--" + model.replace("/", "--"))
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


def output_path(alias, args):
    variant = "" if args.variant == "full" else f"_{args.variant}"
    thinking = args.thinking or DEFAULT_THINKING.get(alias, "auto")
    mode = "" if thinking == "auto" else f"_thinking_{thinking}"
    suffix = "_smoke" if args.limit_per_task else ""
    return args.benchmark_root / "inference/responses" / f"{alias}{variant}{mode}{suffix}_response.json"


def server_command(model, args, port, *, judge=False, tp=1, runtime="MODERN_PYTHON"):
    python = os.environ.get(runtime, PYTHONS.get(runtime, f"<{runtime}>"))
    command = [python, "-B", "-m", "vllm.entrypoints.openai.api_server",
               "--model", model, "--host", "127.0.0.1", "--port", str(port),
               "--dtype", "bfloat16", "--max-model-len", str(16384 if judge else args.max_model_len),
               "--gpu-memory-utilization", "0.9", "--max-num-seqs", "16" if judge else "32",
               "--max-num-batched-tokens", "8192", "--enforce-eager", "--enable-prefix-caching",
               "--tensor-parallel-size", str(tp)]
    if judge:
        command += ["--generation-config", "vllm"]
    else:
        command += ["--limit-mm-per-prompt", '{"image":{"count":5,"width":640,"height":480},"video":0}',
                    "--trust-remote-code"]
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
    if args.base_url or (args.serve and alias not in {"cosmos3-edge", "robointer-3b", "spatiolm-8b"}):
        backend = "api"
    python = sys.executable if backend == "api" else os.environ.get(runtime, PYTHONS.get(runtime, f"<{runtime}>"))
    batch = min(args.batch_size, 4) if backend != "api" else args.batch_size
    return [python, "-B", str(ROOT / script), "--model", checkpoint, "--backend", backend,
            "--output", str(output_path(alias, args)), "--benchmark-root", str(args.benchmark_root),
            "--variant", args.variant, "--batch-size", str(batch), "--api-workers", str(args.api_workers),
            "--max-new-tokens", str(args.max_new_tokens), "--thinking",
            args.thinking or DEFAULT_THINKING.get(alias, "auto"), "--limit-per-task", str(args.limit_per_task)]


def needs_work(command, args):
    code = subprocess.run([sys.executable, *command[1:], "--check-only"], env=args.environment).returncode
    if code not in (0, 3):
        raise subprocess.CalledProcessError(code, command)
    return code == 3


def evaluation_command(alias, args, base_url):
    path = output_path(alias, args)
    output = args.benchmark_root / "evaluation/results" / path.stem.removesuffix("_response") / (
        f"{args.variant}__extract-Qwen3-8B.json")
    return [sys.executable, "-B", str(ROOT.parent / "evaluation/eval.py"),
            "--responses", str(path), "--benchmark-root", str(args.benchmark_root),
            "--extractor-model", args.judge_checkpoint, "--base-url", base_url,
            "--output", str(output), "--batch-size", "16", "--api-workers", "16",
            "--max-new-tokens", "4096"]


def execute(aliases, checkpoints, args):
    logs = args.benchmark_root / "inference/logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows, sources = load_questions(args.benchmark_root, variant=args.variant)
    state = {"pid": os.getpid(), "status": "running", "sources": sources,
             "questions_per_model": len(rows),
             "tasks": sorted({item["task_id"] for _, _, item in rows}),
             "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                            "vllm_max_model_len": args.max_model_len},
             "judge_checkpoint": args.judge_checkpoint,
             "models": {alias: {"checkpoint": checkpoints[alias], "inference": "pending",
                                "evaluation": "pending" if args.evaluate else "not_requested",
                                "response": str(output_path(alias, args))} for alias in aliases}}
    status_path = args.benchmark_root / "inference/pipeline_status.json"

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
        ready = [alias for alias in aliases if state["models"][alias]["inference"] == "complete"]
        if args.evaluate and ready:
            pending = []
            for alias in ready:
                try:
                    if needs_work(evaluation_command(alias, args, "http://127.0.0.1:1/v1"), args):
                        pending.append(alias)
                    else:
                        update(alias, "evaluation", "complete")
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    update(alias, "evaluation", "failed", error)
            if pending:
                with local_server(args.judge_checkpoint, args, logs / "judge_server.log", judge=True) as url:
                    for alias in pending:
                        update(alias, "evaluation", "running")
                        try:
                            command = evaluation_command(alias, args, url)
                            run_command(command, args, logs / f"{alias}_eval.log")
                            update(alias, "evaluation", "complete")
                        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                            update(alias, "evaluation", "failed", error)
                            print(f"FAILED {alias} evaluation: {error}", flush=True)
        state["status"] = "complete" if all(
            job["inference"] == "complete" and job["evaluation"] in {"complete", "not_requested"}
            for job in state["models"].values()) else "failed"
        if state["status"] == "complete" and args.evaluate and args.variant == "full" and set(aliases) == set(SMALL_MODELS):
            run_command([sys.executable, "-B", str(ROOT.parent / "evaluation/table.py"),
                         "--benchmark-root", str(args.benchmark_root)], args, logs / "table.log")
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        update()
    print(f"Pipeline {state['status']}: {status_path}", flush=True)
    return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["list", "all", "small", *MODELS])
    parser.add_argument("--benchmark-root", type=Path, default=ROOT.parent)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-workers", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=("auto", "on", "off"))
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--base-url", help="Use an existing inference API; do not start or stop it")
    parser.add_argument("--cached", action="store_true", help="Pin local cached revisions; disable network downloads")
    parser.add_argument("--serve", action="store_true", help="Use managed vLLM servers; Cosmos3, RoboInter and SpatioLM use native adapters")
    parser.add_argument("--evaluate", action="store_true", help="After inference, score complete responses with cached Qwen3-8B")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.models == ["list"]:
        for alias, (script, model, backend, runtime, _) in MODELS.items():
            print(f"{alias:24} {script:25} {backend:13} {runtime:16} {model}")
        return 0
    aliases = list(SMALL_MODELS) if args.models == ["small"] else list(MODELS) if args.models == ["all"] else args.models
    if any(a not in MODELS for a in aliases) or len(set(aliases)) != len(aliases):
        parser.error("Use list/all/small alone, or distinct model aliases")
    if args.evaluate and args.limit_per_task:
        parser.error("Evaluation requires complete QA files; use a separate subset benchmark-root for smoke tests")
    if min(args.batch_size, args.api_workers, args.max_new_tokens) < 1 or args.limit_per_task < 0:
        parser.error("Invalid batch size, worker count or generation limit")
    args.benchmark_root = args.benchmark_root.resolve()
    args.environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONDONTWRITEBYTECODE="1")
    if args.cached:
        args.environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    checkpoints = {alias: cached_checkpoint(MODELS[alias][1]) if args.cached else MODELS[alias][1] for alias in aliases}
    args.judge_checkpoint = cached_checkpoint("Qwen/Qwen3-8B") if args.evaluate else None
    if args.dry_run:
        for alias in aliases:
            print(shlex.join(inference_command(alias, checkpoints[alias], args)))
        if args.evaluate:
            print("Judge checkpoint:", args.judge_checkpoint)
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    # One queue owns the requested GPU set, including its inference and judge stages.
    with open(f"/tmp/pbench-gpu-{args.gpus.replace(',', '-')}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(aliases, checkpoints, args)


if __name__ == "__main__":
    sys.exit(main())
