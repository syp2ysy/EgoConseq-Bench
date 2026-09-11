"""Scene-owned B1K collection, appending into the existing records catalog."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from pipeline import collection_cli, io_utils, scene_partitions


ROOT = Path(__file__).resolve().parents[1]


def scan_records(stream):
    """Count scenes and retain every contiguous range, including revisits."""
    counts, ranges, digest = Counter(), [], hashlib.sha256()
    offset = 0
    for line in stream:
        record = json.loads(line)
        scene = record["scene_id"]
        counts[scene] += 1
        digest.update(line)
        if not ranges or ranges[-1]["scene_id"] != scene:
            ranges.append({"scene_id": scene, "start_byte": offset,
                           "end_byte": offset, "record_count": 0})
        offset += len(line)
        ranges[-1]["end_byte"] = offset
        ranges[-1]["record_count"] += 1
    return counts, ranges, digest.hexdigest()


def update_progress(root, **fields):
    """Use the same lock as appenders so job updates cannot erase counters."""
    path = root / "b1k/expansion.json"
    with (root / "b1k/records.jsonl").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        state.update(fields)
        io_utils.atomic_write_json(path, state, durable=True)
    return state


def seal_catalog(root):
    """Update metadata once writers have stopped; never rewrite records."""
    dataset = root / "b1k"
    collection_cli.append_compact_records(
        dataset / "records.jsonl", [], progress_path=dataset / "expansion.json")
    with (dataset / "records.jsonl").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        counts, ranges, digest = scan_records(stream)
        meta_path = dataset / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(record_count=sum(counts.values()), records_sha256=digest,
                    scene_byte_ranges=ranges)
        meta["expansion"] = {
            key: value for key, value in json.loads(
                (dataset / "expansion.json").read_text()).items()
            if key in ("run_id", "seed", "source_manifest", "initial_count",
                       "record_count", "target_records", "phase")}
        io_utils.atomic_write_json(meta_path, meta, durable=True)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["datasets"]:
            if entry["dataset"] == "b1k":
                entry.update(record_count=sum(counts.values()), scene_count=len(counts),
                             records_sha256=digest,
                             run_meta_sha256=io_utils.sha256_file(meta_path))
        manifest["record_count"] = sum(row["record_count"] for row in manifest["datasets"])
        io_utils.atomic_write_json(manifest_path, manifest, durable=True)


def collect_scene(gpu, job, *, root, python, data_root, source_manifest, seed, lock_fd):
    """One process loads one scene and uses it for the entire pose budget."""
    dataset = root / "b1k"
    progress = dataset / ".expansion-progress" / job["scene_id"]
    progress.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMNIGIBSON_DATA_PATH=str(data_root),
               OMNIGIBSON_APPDATA_PATH=str(data_root / "appdata"),
               OMNIGIBSON_GPU_ID=str(gpu), OMNIGIBSON_HEADLESS="True",
               OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [str(python), "-u", str(ROOT / "scripts/collect.py"),
               "--backend", "b1k", "--scenes", job["scene_id"],
               "--b1k-data-root", str(data_root),
               "--b1k-source-manifest", str(source_manifest),
               "--benchmark-partition", "train_seen", "--seed", str(seed),
               "--collection-shard-id", job["shard_id"],
               "--append-records", str(dataset / "records.jsonl"),
               "--out", str(progress), "--poses-per-scene", str(job["poses"]),
               "--pose-candidates-per-scene", "20000",
               "--record-idle-stop-s", "300", "--scene-wallclock-stop-s", "14400",
               "--ordinary-actions-per-pose", "24", "--oracle-evaluation", "nominal",
               "--allow-dirty-code"]
    golden = ROOT / ("data/metadata/golden/abc1_three_dataset_1h_20260813_aa04f8e/"
                     "records/b1k/g3/gates_bedroom-transaction/gates_bedroom/records.jsonl")
    if golden.is_file():
        command.extend(["--pose-exclusions", str(golden)])
    with (dataset / f"expansion-gpu{gpu}.log").open("ab") as log:
        return subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT,
                              pass_fds=(lock_fd,)).returncode


def run_expansion(*, root, target_records, gpus, python, data_root,
                  source_manifest, seed=20260905):
    root = Path(root).resolve()
    dataset = root / "b1k"
    records = dataset / "records.jsonl"
    progress = dataset / "expansion.json"
    with (dataset / "expansion.lock").open("a") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collection_cli.append_compact_records(records, [])  # Recover only a torn tail.
        if progress.exists():
            state = json.loads(progress.read_text())
            if state["target_records"] != target_records or state["seed"] != seed:
                raise ValueError("resume with the existing target and seed")
            if state["phase"] == "complete":
                seal_catalog(root)
                return state
        else:
            with records.open("rb") as stream:
                counts, _, _ = scan_records(stream)
            partition = scene_partitions.load()
            catalog = json.loads(Path(source_manifest).read_text())["scenes"]
            scenes = sorted(
                [row["scene_id"] for row in catalog if
                 partition.partition("b1k", row["scene_id"]) == "train_seen"],
                key=lambda scene: (counts[scene] > 0, -counts[scene], scene))
            run_id = f"expand-{seed}-{int(time.time())}"
            state = {"phase": "collecting", "run_id": run_id, "seed": seed,
                     "source_manifest": str(Path(source_manifest).resolve()),
                     "initial_count": sum(counts.values()),
                     "record_count": sum(counts.values()),
                     "target_records": target_records, "byte_offset": records.stat().st_size,
                     "jobs": [{"scene_id": scene, "shard_id": f"{run_id}-{scene}",
                               "poses": 600, "status": "pending"} for scene in scenes]}
            io_utils.atomic_write_json(progress, state, durable=True)
        collection_cli.append_compact_records(records, [], progress_path=progress)
        jobs = state["jobs"]
        pending = [job for job in jobs if job["status"] != "completed"]
        running, free = {}, list(gpus)
        update_progress(root, phase="collecting", pid=os.getpid())
        try:
            with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                while pending or running:
                    state = json.loads(progress.read_text())
                    if state["record_count"] >= target_records:
                        pending.clear()
                    while free and pending:
                        gpu, job = free.pop(0), pending.pop(0)
                        job.update(status="running", gpu=gpu)
                        update_progress(root, jobs=jobs)
                        future = executor.submit(
                            collect_scene, gpu, dict(job), root=root, python=python,
                            data_root=data_root, source_manifest=source_manifest,
                            seed=seed, lock_fd=run_lock.fileno())
                        running[future] = (gpu, job)
                    if not running:
                        break
                    finished, _ = wait(running, timeout=30, return_when=FIRST_COMPLETED)
                    for future in finished:
                        gpu, job = running.pop(future)
                        code = future.result()
                        job.update(status="completed" if code == 0 else "failed", returncode=code)
                        update_progress(root, jobs=jobs)
                        free.append(gpu)
                        print(json.dumps({"scene": job["scene_id"], "gpu": gpu,
                                          "returncode": code}), flush=True)
            collection_cli.append_compact_records(records, [], progress_path=progress)
            state = json.loads(progress.read_text())
            state = update_progress(
                root, phase="complete" if state["record_count"] >= target_records else "shortfall",
                finished_time=time.time(), jobs=jobs)
        except Exception as error:
            update_progress(root, phase="failed", error=str(error), jobs=jobs)
            raise
        finally:
            seal_catalog(root)
        if state["phase"] == "complete":
            scratch = dataset / ".expansion-progress"
            if scratch.exists():
                shutil.rmtree(scratch)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/metadata/train/seen_updates/current")
    parser.add_argument("--target-records", type=int, default=12000)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    result = run_expansion(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["phase"] == "complete" else 1
