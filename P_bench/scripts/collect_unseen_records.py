#!/usr/bin/env python3
"""Collect held-out scenes on four GPUs, then index the original shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import io_utils, scene_partitions
from post_QA.seen_build import catalog


def build_jobs(args):
    data = args.data_root.resolve()
    partitions = scene_partitions.load()
    b1k_source = json.loads(args.b1k_source_manifest.read_text())
    b1k = sorted(row["scene_id"] for row in b1k_source["scenes"]
                  if partitions.partition("b1k", row["scene_id"]) == "test_unseen")
    episodes = data / "r2r_vlnce_v1-3/test/test.json.gz"
    with gzip.open(episodes, "rt") as stream:
        r2r = sorted({Path(row["scene_id"]).stem
                      for row in json.load(stream)["episodes"]})
    gs_manifest = data / "gs/splits/val.json"
    gs = sorted(row["scene_id"] for row in json.loads(gs_manifest.read_text())["scenes"]
                if row["split"] == "val")
    counts = dict(zip(("b1k", "r2r", "gs"), args.targets))
    scene_counts = {"b1k": len(b1k), "r2r": len(r2r), "gs": len(gs)}
    jobs = []
    for index, (dataset, scenes, split) in enumerate((
            ("b1k", b1k[::2], "train"), ("b1k", b1k[1::2], "train"),
            ("r2r", r2r, "test"), ("gs", gs, "val"))):
        worker = f"b1k-{index}" if dataset == "b1k" else dataset
        gpu = args.gpus[index]
        output = args.root.resolve() / dataset / worker
        poses = math.ceil(counts[dataset] / scene_counts[dataset])
        python = args.b1k_python if dataset == "b1k" else args.python
        command = [str(python), "-B", str(ROOT / "scripts/collect.py"),
                   "--backend", dataset, "--source-split", split,
                   "--benchmark-partition", "test_unseen",
                   "--poses-per-scene", str(poses),
                   "--pose-candidates-per-scene", "20000",
                   "--record-idle-stop-s", "300", "--scene-wallclock-stop-s", "14400",
                   "--ordinary-actions-per-pose", "24", "--oracle-evaluation", "nominal",
                   "--seed", str(args.seed + index), "--allow-dirty-code", "--resume"]
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
               "MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet"}
        if dataset == "b1k":
            b1k_root = data / "behavior-1k-v3.9.1"
            command += ["--b1k-data-root", str(b1k_root),
                        "--b1k-source-manifest", str(args.b1k_source_manifest.resolve())]
            env.update(OMNIGIBSON_GPU_ID=str(gpu), OMNIGIBSON_HEADLESS="True",
                       OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_DATA_PATH=str(b1k_root),
                       OMNIGIBSON_APPDATA_PATH=str(b1k_root / "appdata"))
        elif dataset == "r2r":
            command += ["--r2r-episodes", str(episodes),
                        "--mp3d-root", str(data / "scene_datasets/mp3d"),
                        "--semantic-query-workers", "8"]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            command += ["--gs-data-root", str(data / "gs"),
                        "--gs-source-manifest", str(gs_manifest)]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        jobs.append(dict(dataset=dataset, worker=worker, gpu=gpu, scenes=scenes,
                         poses_per_scene=poses, output_dir=str(output),
                         command=command, environment=env))
    return jobs


def collect_worker(job, seed):
    """One scene subprocess at a time; a failed scene leaves prior shards intact."""
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "collection.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"scenes": {}}
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(job["environment"])
    failed = False
    for scene in job["scenes"]:
        out = output / scene
        if (state["scenes"].get(scene, {}).get("status") == "completed"
                and (out / "records.jsonl").is_file()
                and (out / "manifest.json").is_file()):
            continue
        out.mkdir(parents=True, exist_ok=True)
        command = job["command"] + ["--scenes", scene, "--out", str(out),
                  "--collection-shard-id", f"unseen-{seed}-{job['worker']}-{scene}"]
        with (out / "collect.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            state["scenes"][scene] = {"status": "running", "pid": process.pid}
            io_utils.atomic_write_json(state_path, state)
            code = process.wait()
        complete = code == 0 and (out / "manifest.json").is_file()
        state["scenes"][scene].update(status="completed" if complete else "failed",
                                       exit_code=code)
        if complete:
            count = json.loads((out / "run_meta.json").read_text())["record_count"]
            state["scenes"][scene].update(record_count=count,
                                         shortfall=max(0, job["poses_per_scene"] - count))
        io_utils.atomic_write_json(state_path, state)
        print(json.dumps({"worker": job["worker"], "scene": scene, "exit_code": code}), flush=True)
        failed |= not complete
    return not failed


def finish_catalog(root, jobs):
    rows = []
    for job in jobs:
        output = Path(job["output_dir"])
        state = json.loads((output / "collection.json").read_text())
        for scene in job["scenes"]:
            path = output / scene / "records.jsonl"
            if state["scenes"].get(scene, {}).get("status") == "completed":
                rows.append({"dataset": job["dataset"], "path": str(path)})
    return catalog.write(rows, root / "manifest.json", split="test/unseen")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/metadata/test/unseen")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--b1k-source-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--b1k-python", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--targets", type=int, nargs=3, default=[3000, 3000, 2000],
                        metavar=("B1K", "R2R", "GS"))
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)
    jobs = build_jobs(args)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.root / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["split"] != "test/unseen":
            parser.error("output already contains a non-unseen catalog")
        state_path = args.root / "collection.json"
        if state_path.exists() and json.loads(state_path.read_text())["jobs"] != jobs:
            parser.error("resume with the existing source paths, targets, GPUs and seed")
        state = {"split": "test/unseen", "status": "running", "pid": os.getpid(), "jobs": jobs}
        io_utils.atomic_write_json(state_path, state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(collect_worker, job, args.seed) for job in jobs]
            results = [future.result() for future in as_completed(futures)]
        result = finish_catalog(args.root, jobs)
        actual = {row["dataset"]: sum(shard["record_count"] for shard in row["shards"])
                  for row in result["datasets"]}
        shortfalls = {ds: max(0, target - actual.get(ds, 0))
                      for ds, target in zip(("b1k", "r2r", "gs"), args.targets)}
        state.update(status="completed" if all(results) and not any(shortfalls.values()) else "partial",
                     record_count=result["record_count"], shortfalls=shortfalls)
        io_utils.atomic_write_json(state_path, state)
        print(json.dumps({"status": state["status"], "records": result["record_count"]}), flush=True)
        return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
