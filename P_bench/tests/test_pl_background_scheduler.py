"""The controller keeps every GPU busy without crossing catalog passes."""

from __future__ import annotations

from types import SimpleNamespace

from pipeline import background_scheduler


def _job(job_id: str, *, gpu_id: int, round_index: int,
         dataset: str = "r2r", catalog_pass: int = 0) -> dict:
    return {
        "job_id": job_id,
        "gpu_id": gpu_id,
        "round_index": round_index,
        "dataset": dataset,
        "catalog_pass": catalog_pass,
        "scenes": [job_id],
        "environment": {},
        "log_path": f"/tmp/{job_id}.log",
    }


def _manifest() -> dict:
    return {
        "rounds": [
            {"round_index": 0, "catalog_pass": 0, "jobs": [
                _job("g0-first", gpu_id=0, round_index=0),
                _job("g1-first", gpu_id=1, round_index=0, dataset="gs"),
            ]},
            {"round_index": 1, "catalog_pass": 0, "jobs": [
                _job("g0-second", gpu_id=0, round_index=1),
                _job("g1-second", gpu_id=1, round_index=1, dataset="gs"),
            ]},
            {"round_index": 2, "catalog_pass": 1, "jobs": [
                _job("g0-pass1", gpu_id=0, round_index=2,
                     catalog_pass=1),
            ]},
        ],
    }


def _runtime(job_id: str, *, status: str, catalog_status=None) -> dict:
    value = {"job_id": job_id, "status": status}
    if catalog_status is not None:
        value["catalog_status"] = catalog_status
    return value


def test_finished_gpu_gets_its_next_scene_without_round_barrier():
    manifest = _manifest()
    state = {"jobs": {
        "g0-first": _runtime(
            "g0-first", status="completed", catalog_status="completed"),
        "g1-first": _runtime("g1-first", status="running"),
    }}

    ready = background_scheduler.next_jobs(
        manifest, state, catalog_pass=0)

    assert [job["job_id"] for job in ready] == ["g0-second"]


def test_scheduler_never_crosses_an_incomplete_catalog_pass():
    manifest = _manifest()
    state = {"jobs": {
        "g0-first": _runtime(
            "g0-first", status="completed", catalog_status="completed"),
        "g0-second": _runtime(
            "g0-second", status="completed", catalog_status="completed"),
        "g1-first": _runtime("g1-first", status="running"),
    }}

    assert background_scheduler.next_jobs(
        manifest, state, catalog_pass=1) == []
    assert background_scheduler.catalog_pass_complete(
        manifest, state, catalog_pass=0) is False


def test_completed_dataset_jobs_are_skipped_without_blocking_other_work():
    manifest = _manifest()
    state = {"jobs": {}}

    ready = background_scheduler.next_jobs(
        manifest, state, catalog_pass=0, completed_datasets={"r2r"})

    assert [job["job_id"] for job in ready] == ["g1-first"]


def test_launch_jobs_allows_other_gpus_to_keep_running():
    manifest = _manifest()
    state = {"status": "running", "current_round": 0, "jobs": {
        "g1-first": {
            **_runtime("g1-first", status="running"),
            "gpu_id": 1,
            "round_index": 0,
        },
    }}
    initialized = []

    processes = background_scheduler.launch_jobs(
        manifest, state, [manifest["rounds"][1]["jobs"][0]],
        launcher=lambda *_args: SimpleNamespace(pid=123),
        initialize=lambda _manifest, job, **_kwargs:
            initialized.append(job["job_id"]),
        now=10.0)

    assert set(processes) == {"g0-second"}
    assert initialized == ["g0-second"]
    assert state["jobs"]["g0-second"]["round_index"] == 1
    assert state["jobs"]["g1-first"]["status"] == "running"
