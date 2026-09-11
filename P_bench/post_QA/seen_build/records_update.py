"""Resume record refinement; optionally compile an explicitly requested benchmark."""

from __future__ import annotations

from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from pipeline import io_utils
from post_QA.seen_build import inventory, release, selection, supply


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "build_seen_benchmark.py"
WORKERS = ("b1k-0", "b1k-1", "gs", "r2r")


def _running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _collect(plan: Path, work: Path, logs: Path, *, seed: int,
             python: Path, b1k_python: Path, b1k_data_root: Path) -> None:
    """Each worker owns its scenes and keeps one append-only log."""
    logs.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        processes = []
        for gpu, worker in enumerate(WORKERS):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
            executable = python
            if worker.startswith("b1k"):
                executable = b1k_python
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env.update(
                    OMNIGIBSON_DATA_PATH=str(b1k_data_root),
                    OMNIGIBSON_APPDATA_PATH=str(b1k_data_root / "appdata"),
                    OMNIGIBSON_GPU_ID=str(gpu),
                    OMNIGIBSON_HEADLESS="True", OMNI_KIT_ACCEPT_EULA="YES")
            else:
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),
                           HABITAT_SIM_LOG="quiet", MAGNUM_LOG="quiet")
            log = stack.enter_context((logs / f"{worker}.log").open("ab"))
            processes.append((worker, subprocess.Popen([
                str(executable), "-u", str(CLI), "collect", "--plan", str(plan),
                "--work-dir", str(work), "--worker", worker, "--seed", str(seed),
            ], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)))
        failures = [(worker, code) for worker, process in processes
                    if (code := process.wait()) != 0]
    if failures:
        raise RuntimeError(f"collection failed; resume this command: {failures}")


def _materialize(plan: Path, work: Path, target: Path) -> None:
    if (target / "manifest.json").is_file():
        return
    stage = target.with_name(target.name + ".partial")
    if stage.exists():
        shutil.rmtree(stage)
    release.materialize_records(plan, work, stage)
    stage.rename(target)


def finish_records(output_root: Path, *, seed: int = 20260904) -> dict:
    """Close a user-stopped repair, retaining source cases for unfinished rows."""
    root = Path(output_root).resolve()
    with (root / "update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = (root / "current").resolve()
        round_index = int(current.name.split("-")[-1]) + 1
        work = root / "work" / f"round-{round_index:03d}"
        target = root / "versions" / f"records-{round_index:03d}"
        stage = target.with_name(target.name + ".partial")
        io_utils.atomic_write_json(root / "status.json", {
            "phase": "finishing_requested_stop", "round": round_index})
        if stage.exists():
            shutil.rmtree(stage)
        summary = release.materialize_records(
            work / "plan.jsonl", work, stage, reuse_uncollected=True)
        stage.rename(target)
        candidates = selection.enumerate_candidates(target, seed=seed)
        report = {"schema": "egoconseq.abc1-supply.v1",
                  "records_root": str(target),
                  **supply.analyze_candidates(candidates, seed=seed)}
        io_utils.atomic_write_json(root / "supply.json", report, allow_nan=False)
        pending = root / ".current.tmp"
        pending.unlink(missing_ok=True)
        pending.symlink_to(target.relative_to(root), target_is_directory=True)
        os.replace(pending, root / "current")
        shutil.rmtree(work)
        for previous in sorted((root / "versions").glob("records-*")):
            if previous not in (current, target) and (previous / "manifest.json").is_file():
                shutil.rmtree(previous)
        result = {"phase": "stopped_by_user", "records_updated": True,
                  "records": str(root / "current"), "round": round_index,
                  "shortfall_total": report["shortfall_total"], **summary}
        io_utils.atomic_write_json(root / "status.json", result, allow_nan=False)
        return result


def update_records(
        plan_path: Path, work_dir: Path, output_root: Path, *,
        python: Path, b1k_python: Path, b1k_data_root: Path,
        wait_pids: tuple[int, ...] = (), max_rounds: int = 3,
        seed: int = 20260904, benchmark_output: Path | None = None,
        device: str = "cuda:0") -> dict:
    """Checkpoint at complete record generations, keeping only the latest two."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def status(phase: str, **values) -> dict:
        value = {"phase": phase, **values}
        io_utils.atomic_write_json(root / "status.json", value, allow_nan=False)
        print(json.dumps(value, sort_keys=True), flush=True)
        return value

    with (root / "update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("waiting_for_collection", pids=list(wait_pids))
            while any(_running(pid) for pid in wait_pids):
                time.sleep(15)
            versions = root / "versions"
            versions.mkdir(exist_ok=True)
            existing = sorted(versions.glob("records-*"))
            existing = [path for path in existing
                        if (path / "manifest.json").is_file()
                        and not path.name.endswith(".partial")]
            current = existing[-1] if existing else versions / "records-000"
            if not existing:
                status("finishing_legacy_collection")
                _collect(plan_path, work_dir, root / "logs", seed=seed,
                         python=python, b1k_python=b1k_python,
                         b1k_data_root=b1k_data_root)
                status("materializing_records", output=str(current))
                _materialize(plan_path, work_dir, current)
                # Every referenced staged image now has a link in compact records.
                if (work_dir / "tmp").exists():
                    shutil.rmtree(work_dir / "tmp")
            round_index = int(current.name.split("-")[-1])
            while True:
                pending = root / ".current.tmp"
                pending.unlink(missing_ok=True)
                pending.symlink_to(current.relative_to(root), target_is_directory=True)
                os.replace(pending, root / "current")
                for previous in sorted(versions.glob("records-*")):
                    if (previous / "manifest.json").is_file() and previous.name < f"records-{round_index - 1:03d}":
                        shutil.rmtree(previous)
                benchmark_error = None
                if benchmark_output is not None:
                    status("compiling_benchmark", round=round_index,
                           output=str(benchmark_output))
                    try:
                        release.compile_records(current, benchmark_output, seed=seed, device=device)
                    except selection.SelectionShortfall as error:
                        benchmark_error = str(error)
                        status("benchmark_shortfall", error=benchmark_error)
                    else:
                        return status("complete", records=str(root / "current"),
                                      round=round_index, benchmark=str(benchmark_output))
                report_path = root / "supply.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                if report.get("records_root") != str(current):
                    status("measuring_supply", round=round_index, records=str(current))
                    candidates = selection.enumerate_candidates(current, seed=seed)
                    report = {"schema": "egoconseq.abc1-supply.v1",
                              "records_root": str(current),
                              **supply.analyze_candidates(candidates, seed=seed)}
                    del candidates
                    io_utils.atomic_write_json(report_path, report, allow_nan=False)
                if report["feasible"] or round_index >= max_rounds:
                    return status(
                        ("benchmark_shortfall" if benchmark_error else
                         "complete" if report["feasible"] else "budget_exhausted"),
                        records=str(root / "current"), round=round_index,
                        shortfall_total=report["shortfall_total"],
                        **({"error": benchmark_error} if benchmark_error else {}))
                repair_work = root / "work" / f"round-{round_index + 1:03d}"
                repair_plan = repair_work / "plan.jsonl"
                if not repair_plan.exists():
                    inventory.write_plan(inventory.sources_from_manifest(current),
                                         repair_plan, repair_report=report,
                                         split=json.loads((current / "manifest.json").read_text()).get("split", "train/seen"))
                status("repairing_actions", round=round_index + 1,
                       shortfall_total=report["shortfall_total"])
                _collect(repair_plan, repair_work, root / "logs",
                         seed=seed + round_index + 1, python=python,
                         b1k_python=b1k_python, b1k_data_root=b1k_data_root)
                target = versions / f"records-{round_index + 1:03d}"
                status("materializing_records", output=str(target))
                _materialize(repair_plan, repair_work, target)
                shutil.rmtree(repair_work)
                current = target
                round_index += 1
        except Exception as error:
            status("failed", error=str(error))
            raise


def refine_surfaces(
        records_root: Path, output_root: Path, work_dir: Path, *,
        python: Path, b1k_python: Path, b1k_data_root: Path,
        seed: int = 20260906) -> dict:
    """Refine every current record once, then atomically switch generations."""
    source_root = Path(records_root).resolve()
    root = Path(output_root).resolve()
    work = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def status(phase: str, **values) -> dict:
        value = {"phase": phase, **values}
        io_utils.atomic_write_json(root / "status.json", value, allow_nan=False)
        print(json.dumps(value, sort_keys=True), flush=True)
        return value

    with (root / "update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan = work / "plan.jsonl"
            status("planning_surfaces", records=str(source_root))
            inventory.write_surface_plan(
                inventory.sources_from_manifest(source_root), plan,
                seed=seed)
            status("refining_surfaces", work=str(work))
            _collect(
                plan, work, work / "logs", seed=seed, python=python,
                b1k_python=b1k_python, b1k_data_root=b1k_data_root)

            versions = root / "versions"
            versions.mkdir(exist_ok=True)
            numbered = sorted(
                path for path in versions.glob("records-[0-9][0-9][0-9]")
                if path.is_dir())
            next_index = max(
                [int(path.name.rsplit("-", 1)[1]) for path in numbered] + [-1]
            ) + 1
            target = versions / f"records-{next_index:03d}"
            stage = target.with_name(target.name + ".partial")
            if target.exists():
                raise FileExistsError(target)
            if stage.exists():
                shutil.rmtree(stage)
            status("materializing_surfaces", output=str(stage))
            summary = release.materialize_records(plan, work, stage)
            stage.rename(target)

            pending = root / ".current.tmp"
            pending.unlink(missing_ok=True)
            pending.symlink_to(
                target.relative_to(root), target_is_directory=True)
            os.replace(pending, root / "current")
            retained = {target}
            if source_root.parent == versions:
                retained.add(source_root)
            else:
                previous = [path for path in numbered if path != target]
                if previous:
                    retained.add(previous[-1])
            for old in numbered:
                if old not in retained and (old / "manifest.json").is_file():
                    shutil.rmtree(old)
            shutil.rmtree(work)
            return status(
                "complete", records=str(root / "current"),
                version=target.name, **summary)
        except Exception as error:
            status("failed", error=str(error), work=str(work))
            raise
