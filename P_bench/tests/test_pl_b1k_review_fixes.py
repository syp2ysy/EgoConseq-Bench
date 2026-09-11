"""Regressions from the Task-4 real-install review and pilot failures."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


def test_isolated_worker_reports_progress_every_30_seconds(monkeypatch):
    """A live B1K child must never leave the supervisor silent for minutes."""
    from pipeline import b1k_process

    class Process:
        pid = 123

        def __init__(self):
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) < 3:
                raise subprocess.TimeoutExpired("collect", timeout)
            return 0

    ticks = iter([100.0, 130.0, 160.0])
    monkeypatch.setattr(b1k_process.time, "monotonic", lambda: next(ticks))
    process = Process()
    heartbeats = []

    assert b1k_process.wait_isolated_process(
        process, timeout_s=300, heartbeat_interval_s=30,
        on_heartbeat=heartbeats.append) == 0
    assert process.waits == [30.0, 30.0, 30.0]
    assert heartbeats == [30.0, 60.0]


def test_b1k_heartbeat_reports_pid_elapsed_and_log_bytes(tmp_path, capsys):
    from scripts import run_b1k_collection_shard as runner

    log_path = tmp_path / "worker.log"
    log_path.write_bytes(b"progress")
    runner._emit_worker_heartbeat(
        pid=321, log_path=log_path, elapsed_s=30.0)

    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "elapsed_s": 30.0,
        "event": "b1k_worker_heartbeat",
        "log_bytes": 8,
        "pid": 321,
    }


def test_pose_sampling_rejects_only_no_finite_depth_and_continues():
    """A no-hit RTX view is a failed pose, not a failed scene process."""
    from pipeline.b1k_sim import B1KSimSession, NoFiniteDepthObservation

    nav = SimpleNamespace(
        authority="b1k_geometry", geometry_authority_sha256="a" * 64,
        query_many=lambda poses: [SimpleNamespace(
            navigable=True, clearance_m=1.0, geometry_source=None)
            for _pose in poses])
    session = object.__new__(B1KSimSession)
    session._geometry = SimpleNamespace(
        sample_position=lambda *_args, **_kwargs: np.zeros(3),
        bind=lambda *_args, **_kwargs: nav)
    session.render = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(NoFiniteDepthObservation(
            "B1K linear depth contains no finite hit")))
    rejected = []

    result = session.sample_random_pose(
        np.random.default_rng(0), [0.2], yaws=[0.0], max_tries=3,
        on_reject=rejected.append)

    assert result is None
    assert rejected == ["low_valid_depth"] * 3

    session.render = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(ValueError("renderer contract broke")))
    with pytest.raises(ValueError, match="renderer contract broke"):
        session.sample_random_pose(
            np.random.default_rng(0), [0.2], yaws=[0.0], max_tries=1,
            on_reject=rejected.append)


def test_pose_sampling_skips_pose_with_no_minimum_action_support():
    """A pose no public action can leave must not spend an RTX render."""
    from pipeline.b1k_sim import B1KSimSession

    class UnsupportedNav:
        authority = "b1k_geometry"
        geometry_authority_sha256 = "a" * 64

        def query_many(self, poses):
            return [SimpleNamespace(
                navigable=False, clearance_m=0.0,
                geometry_source="unsupported_floor") for _pose in poses]

    session = object.__new__(B1KSimSession)
    session._geometry = SimpleNamespace(
        sample_position=lambda *_args, **_kwargs: np.zeros(3),
        bind=lambda *_args, **_kwargs: UnsupportedNav())
    session.render = lambda *_args, **_kwargs: pytest.fail(
        "unsupported pose reached RTX render")
    rejected = []

    result = session.sample_random_pose(
        np.random.default_rng(0), [0.2], yaws=[0.0], max_tries=2,
        on_reject=rejected.append)

    assert result is None
    assert rejected == ["action_support_unavailable"] * 2


@pytest.mark.parametrize(
    ("error", "status", "terminal_status", "reason_field", "reason"),
    [
        (RuntimeError("render failed"), 1, "failed", "failure_reason",
         "RuntimeError"),
        (KeyboardInterrupt(), 130, "interrupted", "interruption_reason",
         "keyboard_interrupt"),
    ],
)
def test_b1k_collect_exceptions_durably_exit_before_isaac_teardown(
        tmp_path, monkeypatch, error, status, terminal_status, reason_field,
        reason):
    """Exception paths retain exact 1/130 instead of later SIGSEGV 139."""
    from scripts import collect

    out = tmp_path / "out"
    out.mkdir()
    funnel = out / "collection_funnel.json"
    funnel.write_text(json.dumps({"status": "running"}))
    args = SimpleNamespace(backend="b1k", out=str(out))
    monkeypatch.setattr(
        collect, "build_parser",
        lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(
        collect, "run_collection",
        lambda *_args: (_ for _ in ()).throw(error))
    exits = []

    result = collect.main(hard_exit=exits.append)

    assert result == status
    assert exits == [status]
    persisted = json.loads(funnel.read_text())
    assert persisted["status"] == terminal_status
    assert persisted[reason_field] == reason


def test_collection_runner_rejects_equals_form_owned_options():
    """Argparse's --option=value form cannot override supervisor authority."""
    from scripts import run_b1k_collection_shard as runner

    with pytest.raises(ValueError, match="--out"):
        runner._validate_forwarded_args(["--out=/stale", "--poses-per-scene=1"])


def test_install_manifest_binds_omnigibson_editable_checkout(tmp_path):
    """A matching wheel version cannot substitute a different OG checkout."""
    from pipeline import b1k_source_builder

    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    data_root.mkdir()
    source_root.mkdir()
    observed = {
        **b1k_source_builder.PINNED_INSTALL,
        "omnigibson_editable_root": str(source_root / "OmniGibson"),
    }

    manifest = b1k_source_builder.build_install_manifest(
        data_root=data_root, source_root=source_root,
        observed=observed, scene_ids=["a", "b", "c"])

    assert manifest["omnigibson_editable_root"] == str(
        (source_root / "OmniGibson").resolve())
    with pytest.raises(ValueError, match="editable root"):
        b1k_source_builder.build_install_manifest(
            data_root=data_root, source_root=source_root,
            observed={**observed,
                      "omnigibson_editable_root": str(tmp_path / "other")},
            scene_ids=["a", "b", "c"])


def _fragment(scene_id: str, scene_json: Path, authority: dict) -> dict:
    from pipeline import io_utils

    return {
        "schema": "b1k-scene-authority-fragment.v2",
        "scene_id": scene_id,
        "scene_json_sha256": io_utils.sha256_file(scene_json),
        "scene_authority": authority,
        "bootstrap": {
            "schema": "b1k-scene-authority-bootstrap.v1",
            "scene_id": scene_id,
            "scene_authority": authority,
            "geometry": {},
            "reset_pose_errors": {},
        },
    }


def test_authority_fragment_recomputes_scene_and_bootstrap_bindings(tmp_path):
    """Fragments bind independent bytes and bootstrap authority exactly."""
    from scripts import build_b1k_source_manifest as builder
    from tests.test_pl_b1k_pilot_tools import _authority, _fake_install

    scene_id = _fake_install(tmp_path, count=3)[0]
    scene_json = builder.b1k_source_builder.canonical_scene_json(
        tmp_path, scene_id)
    path = tmp_path / "fragment.json"
    value = _fragment(scene_id, scene_json, _authority())
    path.write_text(json.dumps(value))

    assert builder._load_valid_fragment(path, scene_id, tmp_path) == value

    for mutation, match in (
        (lambda row: row.update(extra=True), "exact schema"),
        (lambda row: row.update(scene_json_sha256="0" * 64),
         "scene JSON changed"),
        (lambda row: row["bootstrap"].update(scene_id="other"),
         "bootstrap scene"),
        (lambda row: row["bootstrap"].update(scene_authority={}),
         "bootstrap authority"),
    ):
        changed = json.loads(json.dumps(value))
        mutation(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=match):
            builder._load_valid_fragment(path, scene_id, tmp_path)


def test_completed_runner_output_is_bound_to_outer_child_contract(
        tmp_path, monkeypatch):
    """A source-valid stale child output cannot satisfy a new runner task."""
    from scripts import run_b1k_collection_shard as runner
    from tests.test_pl_b1k_pilot_tools import (
        _seal_collection_output, _write_completed_collection_output,
    )

    scene_id = "scene-0"
    output = tmp_path / "collection" / scene_id
    manifest = tmp_path / "source.json"
    manifest.write_text("{}")
    contract = runner._expected_child_contract(
        scene_id=scene_id, output=output, data_root=tmp_path,
        source_manifest=manifest, shard_id="shard",
        revision="a" * 40, code_dirty=False)
    _write_completed_collection_output(output, scene_id, contract=contract)

    observed = runner._validate_completed_output(
        output, scene_id, expected=contract)
    assert observed["source_validated_records"] == 6

    metadata_path = output / "run_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["params"]["out"] = str(tmp_path / "stale")
    metadata_path.write_text(json.dumps(metadata))
    _seal_collection_output(output, status="completed", record_count=6)
    with pytest.raises(ValueError, match="child contract"):
        runner._validate_completed_output(output, scene_id, expected=contract)


@pytest.mark.parametrize(("name", "partial"), (
    ("_validate_completed_output", False),
    ("_validate_partial_output", True),
))
def test_b1k_terminal_wrappers_share_one_validation_path(
        monkeypatch, tmp_path, name, partial):
    """Keeps completed and partial validation policy in one implementation."""
    from scripts import run_b1k_collection_shard as runner

    calls = []

    def validate(output, scene_id, *, expected, partial):
        calls.append((output, scene_id, expected, partial))
        return {"terminal_status": "partial_valid" if partial else "completed"}

    monkeypatch.setattr(runner, "_validate_terminal_output", validate)
    expected = {"scene_id": "scene-0"}
    observed = getattr(runner, name)(
        tmp_path, "scene-0", expected=expected)

    assert observed["terminal_status"] == \
        ("partial_valid" if partial else "completed")
    assert calls == [(tmp_path, "scene-0", expected, partial)]


def test_collection_runner_rejects_stale_effective_collect_policy(
        tmp_path, monkeypatch):
    """A keep=1 artifact cannot satisfy an outer keep=2 collection task."""
    from scripts import run_b1k_collection_shard as runner
    from tests.test_pl_b1k_pilot_tools import (
        _collection_runner_args, _fake_install, _seal_collection_output,
        _write_completed_collection_output,
    )

    scene_id = _fake_install(tmp_path, count=1)[0]
    args = _collection_runner_args(tmp_path, [scene_id], resume=False)
    args.collect_args.extend(["--keep-per-length", "2"])

    def stale_child(command, *, env, log_path, timeout_s):
        del env, log_path, timeout_s
        output = Path(command[command.index("--out") + 1])
        _write_completed_collection_output(output, scene_id)
        # Once the supervisor digest exists, make this fixture prove that the
        # independently compared effective policy -- not a missing digest --
        # rejects the stale keep=1 metadata.
        option = "--b1k-supervisor-contract-sha256"
        if option in command:
            metadata_path = output / "run_meta.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["params"]["b1k_supervisor_contract_sha256"] = (
                command[command.index(option) + 1])
            metadata_path.write_text(json.dumps(metadata))
            _seal_collection_output(
                output, status="completed", record_count=6)
        return 0

    monkeypatch.setattr(runner, "_run_isolated_collection", stale_child)

    assert runner._run_shard(args) == 1
    progress = json.loads(
        (tmp_path / "collection" / "shard-progress.json").read_text())
    assert progress["completed_scene_ids"] == []
    assert progress["failed_scenes"][0]["scene_id"] == scene_id
    assert "child contract" in progress["failed_scenes"][0]["error"]


def test_isolated_supervisors_reap_children_on_keyboard_interrupt(
        tmp_path, monkeypatch):
    """Ctrl-C cannot orphan a start-new-session OmniGibson child."""
    from pipeline import b1k_process
    from scripts import build_b1k_source_manifest
    from scripts import run_b1k_collection_shard

    kills = []
    monkeypatch.setattr(
        b1k_process.os, "killpg",
        lambda process_id, sig: kills.append((process_id, sig)))

    class InterruptedProcess:
        pid = 4321

        def __init__(self):
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise KeyboardInterrupt
            return -signal.SIGTERM

    cases = [
        (build_b1k_source_manifest,
         build_b1k_source_manifest._run_isolated_scene_child, {}),
        (run_b1k_collection_shard,
         run_b1k_collection_shard._run_isolated_collection, {
             "env": {}, "log_path": tmp_path / "child.log",
         }),
    ]
    processes = []
    for module, run_child, extra in cases:
        process = InterruptedProcess()
        processes.append(process)
        monkeypatch.setattr(
            module.subprocess, "Popen", lambda *_args, _p=process, **_kwargs: _p)
        with pytest.raises(KeyboardInterrupt):
            run_child(["fake-child"], timeout_s=1200, **extra)

    assert kills == [(4321, signal.SIGTERM)] * 2
    assert [process.waits for process in processes] == [
        [1200.0, 10], [30.0, 10]]


def test_derive_scene_persists_runtime_initialization_failure(
        tmp_path, monkeypatch):
    """Runtime construction failure is durable and exits before teardown."""
    from pipeline import b1k_sim
    from scripts import build_b1k_source_manifest as manifest_cli

    output = tmp_path / "scene.json"
    exits = []
    shutdowns = []
    monkeypatch.setattr(
        b1k_sim, "load_b1k_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("Isaac init failed")))
    monkeypatch.setattr(
        manifest_cli, "_shutdown_runtime",
        lambda: shutdowns.append("shutdown"))

    result = manifest_cli._derive_scene(SimpleNamespace(
        data_root=str(tmp_path), output=str(output), scene="scene-0"),
        hard_exit=exits.append)

    assert result == 1
    assert exits == [1]
    assert shutdowns == []
    assert json.loads(Path(f"{output}.failure.json").read_text()) == {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-0",
        "error_type": "RuntimeError",
        "error": "Isaac init failed",
    }


def test_authority_progress_v2_detects_fragment_byte_tamper(tmp_path):
    """Resume binds fragment bytes independently instead of rescanning names."""
    from scripts import build_b1k_source_manifest as builder
    from tests.test_pl_b1k_pilot_tools import _authority_fragment, _fake_install

    scene_ids = _fake_install(tmp_path, count=3)
    output = tmp_path / "shard"
    output.mkdir()
    fragment = output / f"{scene_ids[0]}.json"
    fragment.write_text(json.dumps(_authority_fragment(tmp_path, scene_ids[0])))
    identity = builder._fragment_identity(fragment, output)
    (output / "shard-progress.json").write_text(json.dumps({
        "schema": "b1k-source-manifest-shard-progress.v2",
        "data_root": str(tmp_path.resolve()),
        "scene_timeout_s": 1200,
        "requested_scene_ids": scene_ids,
        "completed_scene_ids": [scene_ids[0]],
        "completed_scenes": [{
            "scene_id": scene_ids[0], "fragment": identity}],
        "failed_scenes": [], "attempts": [], "complete": False,
    }))
    fragment.write_text(fragment.read_text() + "\n")

    with pytest.raises(ValueError, match="fragment changed"):
        builder._derive_shard(SimpleNamespace(
            data_root=str(tmp_path), output_dir=str(output), resume=True,
            scenes=scene_ids, scene_timeout_s=1200))


def test_observed_install_reads_editable_direct_url(tmp_path, monkeypatch):
    """verify-install observes PEP-610 checkout identity, not only version."""
    from scripts import build_b1k_source_manifest as builder

    source = tmp_path / "source"
    editable = source / "OmniGibson"
    data = tmp_path / "data"
    editable.mkdir(parents=True)
    (data / "behavior-1k-assets").mkdir(parents=True)
    (data / "omnigibson-robot-assets").mkdir()
    (data / "behavior-1k-assets" / "VERSION").write_text("3.9.0")
    (data / "omnigibson-robot-assets" / "VERSION").write_text("3.8.2")

    class Distribution:
        @staticmethod
        def read_text(name):
            assert name == "direct_url.json"
            return json.dumps({
                "url": editable.as_uri(), "dir_info": {"editable": True},
            })

    monkeypatch.setattr(
        builder.importlib.metadata, "distribution", lambda _name: Distribution())
    monkeypatch.setattr(
        builder.importlib.metadata, "version", lambda name: {
            "omnigibson": "3.9.1", "bddl": "3.7.0",
            "isaacsim": "5.1.0.0", "torch": "2.7.0+cu128",
        }[name])
    monkeypatch.setattr(
        builder.subprocess, "run",
        lambda command, **_kwargs: SimpleNamespace(
            stdout=("26f2c7ef7b9cf96bd0414f81e1e751e493762779\n"
                    if "rev-parse" in command else "v3.9.1\n")))
    monkeypatch.setattr(builder.platform, "python_version", lambda: "3.11.15")

    observed = builder._observed_install(source, data)

    assert observed["omnigibson_editable_root"] == str(editable.resolve())
