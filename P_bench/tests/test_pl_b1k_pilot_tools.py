"""B1K installation, manifest, probe, and shard tooling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import (
    b1k_geometry, b1k_semantic, collection_closeout,
    formal_output_coverage, record,
)
from tests.test_pl_b1k_runtime import _FakeRuntime, _authority_inputs


def _identity(path: Path, root: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _seal_collection_output(
        output: Path, *, status: str, record_count: int) -> None:
    paths = {
        "records_sha256": output / "records.jsonl",
        "run_meta_sha256": output / "run_meta.json",
        "funnel_sha256": output / "collection_funnel.json",
    }
    (output / "collection_finalization.json").write_text(json.dumps({
        "schema": collection_closeout.FINALIZATION_SCHEMA,
        "status": status,
        "source_validation": "passed",
        "record_count": int(record_count),
        "run_contract_sha256": "b" * 64,
        **{
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in paths.items()
        },
    }, sort_keys=True))


def _authority() -> dict:
    value = {
        "schema": "b1k-derived-scene-authority.v1",
        "frame": "pbench_world_xyz",
        "ground_obstacle_band_m": [0.05, 0.3],
        "floor_components": [],
        "collision_components": [],
        "runtime_instances": [],
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def _authority_fragment(root: Path, scene_id: str) -> dict:
    from pipeline import b1k_source_builder, io_utils

    authority = _authority()
    return {
        "schema": "b1k-scene-authority-fragment.v2",
        "scene_id": scene_id,
        "scene_json_sha256": io_utils.sha256_file(
            b1k_source_builder.canonical_scene_json(root, scene_id)),
        "scene_authority": authority,
        "bootstrap": {
            "schema": "b1k-scene-authority-bootstrap.v1",
            "scene_id": scene_id,
            "scene_authority": authority,
            "geometry": {},
            "reset_pose_errors": {},
        },
    }


def _catalog_audit_authority_fragment(root: Path, scene_id: str) -> dict:
    """Return a v2 fragment with the fixed state-2--4 replay certificate."""
    from pipeline import b1k_source_builder, io_utils

    authority = _authority()
    authority.pop("sha256")
    authority.update({
        "canonical_replay_protocol":
            "b1k-canonical-fixed-snapshot-replay.v2",
        "canonical_state_sha256": "a" * 64,
    })
    authority["sha256"] = record.canonical_atom_sha256(authority)
    triangle_authority = {
        key: value for key, value in authority.items()
        if key not in {
            "sha256", "canonical_replay_protocol",
            "canonical_state_sha256",
        }
    }
    triangle_sha256 = record.canonical_atom_sha256(triangle_authority)
    return {
        "schema": "b1k-scene-authority-fragment.v2",
        "scene_id": scene_id,
        "scene_json_sha256": io_utils.sha256_file(
            b1k_source_builder.canonical_scene_json(root, scene_id)),
        "scene_authority": authority,
        "bootstrap": {
            "schema": "b1k-scene-authority-bootstrap.v2",
            "scene_id": scene_id,
            "scene_authority": authority,
            "canonical_replay": {
                "protocol": "b1k-canonical-fixed-snapshot-replay.v2",
                "canonical_state": "state_2",
                "canonical_state_sha256": "a" * 64,
                "triangle_authority_sha256_by_state": {
                    "state_2": triangle_sha256,
                    "state_3": triangle_sha256,
                    "state_4": triangle_sha256,
                },
                "state_2_to_state_3_errors": {
                    "max_object_position_error_m": 0.0,
                    "max_object_orientation_error_rad": 0.0,
                    "max_joint_position_error_rad": 0.0,
                },
                "state_2_to_state_4_errors": {
                    "max_object_position_error_m": 0.0,
                    "max_object_orientation_error_rad": 0.0,
                    "max_joint_position_error_rad": 0.0,
                },
                "state_3_to_state_4_errors": {
                    "max_object_position_error_m": 0.0,
                    "max_object_orientation_error_rad": 0.0,
                    "max_joint_position_error_rad": 0.0,
                },
            },
        },
    }


def test_catalog_audit_fragment_enforces_replay_error_contract(tmp_path):
    """Catches accepting replay diagnostics that exceed frozen GT gates."""
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_id = _fake_install(tmp_path, count=1)[0]
    fragment = _catalog_audit_authority_fragment(tmp_path, scene_id)
    valid_errors = {
        "max_object_position_error_m": 0.0,
        "max_object_orientation_error_rad": 0.0,
        "max_joint_position_error_rad": 0.0,
    }
    replay = fragment["bootstrap"]["canonical_replay"]
    for field in (
            "state_2_to_state_3_errors", "state_2_to_state_4_errors",
            "state_3_to_state_4_errors"):
        replay[field] = dict(valid_errors)
    replay["state_2_to_state_4_errors"][
        "max_object_position_error_m"] = 1.0
    path = tmp_path / f"{scene_id}.json"
    path.write_text(json.dumps(fragment, sort_keys=True))

    with pytest.raises(ValueError, match="replay.*position drift"):
        manifest_cli._load_current_audit_fragment(
            path, scene_id, tmp_path)


def test_catalog_audit_fragment_binds_replay_triangles_to_scene_authority(
        tmp_path):
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_id = _fake_install(tmp_path, count=1)[0]
    fragment = _catalog_audit_authority_fragment(tmp_path, scene_id)
    replay = fragment["bootstrap"]["canonical_replay"]
    replay["triangle_authority_sha256_by_state"] = {
        state: "f" * 64 for state in ("state_2", "state_3", "state_4")}
    path = tmp_path / f"{scene_id}.json"
    path.write_text(json.dumps(fragment, sort_keys=True))

    with pytest.raises(ValueError, match="triangle authority differs"):
        manifest_cli._load_current_audit_fragment(
            path, scene_id, tmp_path)


def _fake_install(root: Path, count: int = 5) -> list[str]:
    assets = root / "behavior-1k-assets"
    scenes = assets / "scenes"
    objects = assets / "objects"
    scene_ids = [f"scene-{index}" for index in range(count)]
    for index, scene_id in enumerate(scene_ids):
        scene_dir = scenes / scene_id
        (scene_dir / "json").mkdir(parents=True)
        (scene_dir / "layout").mkdir()
        category = f"category-{index}"
        model = f"model-{index}"
        scene_json = scene_dir / "json" / f"{scene_id}_best.json"
        scene_json.write_text(json.dumps({
            "state": {"registry": {"object_registry": {}}},
            "objects_info": {"init_info": {
                f"{category}_{model}_0": {
                    "class_module": "omnigibson.objects.dataset_object",
                    "class_name": "DatasetObject",
                    "args": {"category": category, "model": model},
                },
            }},
        }, sort_keys=True))
        (scene_dir / "layout" / "floor_trav_0.png").write_bytes(
            f"layout-{index}".encode())
        encrypted = objects / category / model / "usd" / (
            f"{model}.encrypted.usd")
        encrypted.parent.mkdir(parents=True)
        encrypted.write_bytes(f"encrypted-{index}".encode())
    return scene_ids


def test_source_entry_binds_real_scene_json_as_canonical_initial_state(
        tmp_path):
    """Catches fabricating an initial-state file not loaded by OmniGibson."""
    from pipeline import b1k_source_builder

    scene_id = _fake_install(tmp_path, count=1)[0]

    entry = b1k_source_builder.build_scene_entry(
        tmp_path, scene_id, _authority())

    assert entry["scene_json"] == entry["initial_state"]
    assert entry["initial_state_replay"] == {
        "schema": "b1k-canonical-scene-json-replay.v1",
        "canonical_input": "scene_json",
        "runtime_replay": "omnigibson_dump_state_load_state",
    }
    assert [item["path"] for item in entry["encrypted_assets"]] == [
        "behavior-1k-assets/objects/category-0/model-0/usd/"
        "model-0.encrypted.usd"]
    assert entry["layouts"] == [_identity(
        tmp_path / "behavior-1k-assets/scenes/scene-0/layout/"
        "floor_trav_0.png", tmp_path)]


def test_source_entry_rejects_unknown_canonical_replay_protocol(tmp_path):
    """Catches trusting a hash-valid authority with unknown replay semantics."""
    from pipeline import b1k_source_builder

    scene_id = _fake_install(tmp_path, count=1)[0]
    authority = _authority()
    authority.update({
        "canonical_replay_protocol": "b1k-canonical-dump-load-replay.v999",
        "canonical_state_sha256": "a" * 64,
    })
    authority["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in authority.items() if key != "sha256"
    })

    with pytest.raises(ValueError, match="replay protocol"):
        b1k_source_builder.build_scene_entry(
            tmp_path, scene_id, authority)


def test_source_entry_rejects_complete_obsolete_replay_binding(tmp_path):
    """Catches silently adapting a complete v1 authority into v2."""
    from pipeline import b1k_source_builder

    scene_id = _fake_install(tmp_path, count=1)[0]
    authority = _authority()
    authority.update({
        "canonical_replay_protocol":
            "b1k-canonical-dump-load-replay.v1",
        "canonical_state_sha256": "a" * 64,
    })
    authority["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in authority.items() if key != "sha256"
    })

    with pytest.raises(ValueError, match="replay protocol"):
        b1k_source_builder.build_scene_entry(
            tmp_path, scene_id, authority)


@pytest.mark.parametrize("binding", [
    {"canonical_replay_protocol": None},
    {"canonical_replay_protocol": "b1k-canonical-dump-load-replay.v1"},
    {"canonical_state_sha256": "a" * 64},
    {
        "canonical_replay_protocol": "b1k-canonical-dump-load-replay.v1",
        "canonical_state_sha256": None,
    },
    {
        "canonical_replay_protocol": None,
        "canonical_state_sha256": None,
    },
])
def test_source_entry_rejects_partial_or_null_canonical_replay_binding(
        tmp_path, binding):
    """Catches treating present-null replay keys as absent legacy fields."""
    from pipeline import b1k_source_builder

    scene_id = _fake_install(tmp_path, count=1)[0]
    authority = _authority()
    authority.update(binding)
    authority["sha256"] = record.canonical_atom_sha256({
        key: value for key, value in authority.items() if key != "sha256"
    })

    with pytest.raises(ValueError, match="canonical replay fields"):
        b1k_source_builder.build_scene_entry(
            tmp_path, scene_id, authority)


def test_pilot_manifest_names_subset_without_claiming_full_catalog(tmp_path):
    """Catches presenting a three-scene pilot as the installed train catalog."""
    from pipeline import b1k_source_builder

    scene_ids = _fake_install(tmp_path, count=5)
    entries = [
        b1k_source_builder.build_scene_entry(
            tmp_path, scene_id, _authority())
        for scene_id in scene_ids[:3]
    ]

    manifest = b1k_source_builder.build_source_manifest(
        entries,
        installed_scene_ids=scene_ids,
        selection_scope="three_scene_pilot",
        simulator={"omnigibson": "3.9.1", "isaac_sim": "5.1.0.0"},
        asset_versions={
            "behavior-1k-assets": "3.9.0",
            "omnigibson-robot-assets": "3.8.2",
        },
    )

    assert manifest["installed_catalog_count"] == 5
    assert manifest["selected_scene_count"] == 3
    assert manifest["selection_scope"] == "three_scene_pilot"
    assert manifest["catalog_complete"] is False
    assert [entry["scene_id"] for entry in manifest["scenes"]] == scene_ids[:3]


def test_four_gpu_shards_are_deterministic_disjoint_and_resumable(tmp_path):
    """Catches overlapping shards or commands that silently pick CUDA device 0."""
    from pipeline import b1k_source_builder

    scene_ids = _fake_install(tmp_path, count=11)
    plan = b1k_source_builder.build_shard_plan(
        scene_ids, data_root=tmp_path,
        output_root=tmp_path / "full-manifest", gpu_ids=[0, 1, 2, 3])

    assert len(plan["shards"]) == 4
    assigned = [
        scene for shard in plan["shards"] for scene in shard["scene_ids"]]
    assert sorted(assigned) == sorted(scene_ids)
    assert len(assigned) == len(set(assigned))
    for index, shard in enumerate(plan["shards"]):
        assert shard["gpu_id"] == index
        assert f"OMNIGIBSON_GPU_ID={index}" in shard["command"]
        assert "CUDA_VISIBLE_DEVICES" not in shard["command"]
        assert "--resume" in shard["command"]
        assert "--scene-timeout-s 1200" in shard["command"]
        assert f"shard-{index:02d}" in shard["output_dir"]


def test_collection_shard_plan_is_scene_disjoint_and_process_isolated(
        tmp_path):
    """Catches a collection plan reusing one OG process across scenes."""
    from pipeline import b1k_source_builder

    scene_ids = _fake_install(tmp_path, count=11)
    plan = b1k_source_builder.build_collection_shard_plan(
        scene_ids, data_root=tmp_path,
        source_manifest=tmp_path / "full-source-manifest.json",
        output_root=tmp_path / "collection", gpu_ids=[0, 1, 2, 3],
        code_revision="a" * 40,
        collect_args=["--poses-per-scene", "20"])

    assert plan["schema"] == "b1k-collection-shard-plan.v1"
    assert plan["installed_catalog_count"] == 11
    assigned = [
        scene for shard in plan["shards"] for scene in shard["scene_ids"]]
    assert sorted(assigned) == sorted(scene_ids)
    assert len(assigned) == len(set(assigned))
    for index, shard in enumerate(plan["shards"]):
        command = shard["command"]
        assert shard["gpu_id"] == index
        assert f"OMNIGIBSON_GPU_ID={index}" in command
        assert "CUDA_VISIBLE_DEVICES" not in command
        assert "run_b1k_collection_shard.py run" in command
        assert "--resume" in command
        assert "--scene-timeout-s" in command
        assert "--collect-args --poses-per-scene 20" in command
        assert f"shard-{index:02d}" in shard["output_dir"]


def test_collection_shard_plan_cli_persists_four_resumable_commands(tmp_path):
    """Catches the full-catalog collection plan existing only as an API."""
    scene_ids = _fake_install(tmp_path, count=7)
    output = tmp_path / "collection-plan.json"
    result = subprocess.run([
        sys.executable, str(Path(__file__).parents[1] / "scripts" /
                            "build_b1k_source_manifest.py"),
        "plan-collection-shards", "--data-root", str(tmp_path),
        "--source-manifest", str(tmp_path / "full-source-manifest.json"),
        "--output-root", str(tmp_path / "collection"),
        "--gpu-ids", "0", "1", "2", "3",
        "--code-revision", "a" * 40,
        "--output", str(output),
        "--collect-args", "--poses-per-scene", "20",
    ], cwd=Path(__file__).parents[1], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    plan = json.loads(output.read_text())
    assert plan["installed_catalog_count"] == len(scene_ids)
    assert len(plan["shards"]) == 4
    assert all("run_b1k_collection_shard.py run" in shard["command"]
               for shard in plan["shards"])


def test_expected_child_policy_names_only_params_a_real_child_records(
        tmp_path):
    """Catches a supervisor demanding params the collector never persists.

    The fixtures below echo the expected policy straight back as the child's
    params, so they agree by construction and cannot see this class of bug at
    all. Here the child side is built by the writer that actually produces
    ``run_meta.json``, which prunes backend-scoped params: a supervisor reading
    its expectations off the raw parser asks for GS roots no b1k child records
    and fails the readback on every scene it collects.
    """
    from pipeline import collection_cli, collection_runtime, collection_support
    from scripts import run_b1k_collection_shard as runner

    manifest = tmp_path / "source-manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "shard" / "scene-a"
    arguments = [
        "--backend", "b1k", "--scenes", "scene-a",
        "--out", str(output),
        "--b1k-data-root", str(tmp_path),
        "--b1k-source-manifest", str(manifest),
        "--collection-shard-id", "pilot-02-scene-a",
        "--code-revision", "a" * 40,
        "--poses-per-scene", "1",
    ]
    policy = runner._effective_child_policy(
        scene_id="scene-a", output=output, data_root=tmp_path,
        source_manifest=manifest, collection_shard_id="pilot-02-scene-a",
        revision="a" * 40, code_dirty=False,
        collect_args=["--poses-per-scene", "1"])

    parser = collection_cli.build_parser()
    parsed = parser.parse_args(arguments)
    collection_runtime._resolve_collection_mode_defaults(parsed, parser)
    recorded = collection_support.collection_run_contract(
        parsed, ["scene-a"], [1.5], [(90.0, 60.0)])["params"]

    assert not set(policy) - set(recorded)
    assert "gs_data_root" not in policy
    assert "gs_source_manifest" not in policy


def _write_completed_collection_output(
        output: Path, scene_id: str, *, contract: dict | None = None) -> None:
    if contract is None:
        from scripts import run_b1k_collection_shard as runner

        root = output.parents[1]
        contract = runner._expected_child_contract(
            scene_id=scene_id, output=output, data_root=root,
            source_manifest=root / "source-manifest.json",
            shard_id="pilot-02", revision="a" * 40, code_dirty=True,
            collect_args=["--poses-per-scene", "1"])
    output.mkdir(parents=True, exist_ok=True)
    records = [_formal_record(scene_id, length) for length in range(1, 7)]
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    params = dict(contract.get("collect_policy") or {})
    params.update({
        "backend": "b1k", "scenes": [scene_id],
        "keep_per_length": 1,
    })
    if contract is not None:
        params.update({
            "out": contract["out"],
            "b1k_data_root": contract["data_root"],
            "b1k_source_manifest": contract["source_manifest_path"],
            "collection_shard_id": contract["collection_shard_id"],
            "code_revision": contract["code_revision"],
            "allow_dirty_code": contract["code_dirty"],
        })
        if contract.get("collect_policy_sha256"):
            params["b1k_supervisor_contract_sha256"] = contract[
                "collect_policy_sha256"]
    (output / "collection_funnel.json").write_text(json.dumps({
        "schema_version": "egoconseq.collection-funnel.v3",
        "backend": "b1k",
        "status": "completed",
        "run_contract_sha256": "b" * 64,
        "code_revision": contract["code_revision"],
        "code_dirty": contract["code_dirty"],
        "scene_counts": {
            "total": 1, "completed": 1, "interrupted": 0, "failed": 0,
        },
        "per_scene": [{"scene_id": scene_id, "status": "completed"}],
    }))
    (output / "run_meta.json").write_text(json.dumps({
        "run_contract_sha256": "b" * 64,
        "code_revision": contract["code_revision"],
        "code_dirty": contract["code_dirty"],
        "params": params,
        "source_catalog": {
            "datasets": ["b1k"], "scene_ids": [scene_id],
            "manifest_sha256": [contract["source_manifest_sha256"]],
        },
        "formal_action_length_coverage": formal_output_coverage.summarize(
            records_path, SimpleNamespace(**params)),
    }))
    _seal_collection_output(
        output, status="completed", record_count=len(records))


def _formal_record(scene_id: str, length: int) -> dict:
    group_id = f"{scene_id}-L{length}"
    return {
        "intervention": {"group_id": f"pose-{scene_id}"},
        "selection": {"action_group_ids": [group_id]},
        "outcomes": [{
            "action_group_id": group_id,
            "actions": [
                {"type": "forward", "m": 0.5}
                for _index in range(length)
            ],
        }],
    }


def _write_partial_collection_output(
        output: Path, scene_id: str, lengths: list[int]) -> None:
    from scripts import run_b1k_collection_shard as runner

    root = output.parents[1]
    contract = runner._expected_child_contract(
        scene_id=scene_id, output=output, data_root=root,
        source_manifest=root / "source-manifest.json",
        shard_id="pilot-02", revision="a" * 40, code_dirty=True,
        collect_args=["--poses-per-scene", "1"])
    output.mkdir(parents=True, exist_ok=True)
    records = [_formal_record(scene_id, length) for length in lengths]
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    params = dict(contract.get("collect_policy") or {})
    params.update({
        "backend": "b1k", "scenes": [scene_id],
        "keep_per_length": 1,
        "out": contract["out"],
        "b1k_data_root": contract["data_root"],
        "b1k_source_manifest": contract["source_manifest_path"],
        "collection_shard_id": contract["collection_shard_id"],
        "code_revision": contract["code_revision"],
        "allow_dirty_code": contract["code_dirty"],
        "b1k_supervisor_contract_sha256":
            contract["collect_policy_sha256"],
    })
    coverage = formal_output_coverage.summarize(
        records_path, SimpleNamespace(**params))
    (output / "collection_funnel.json").write_text(json.dumps({
        "schema_version": "egoconseq.collection-funnel.v3",
        "backend": "b1k",
        "status": "failed",
        "failure_reason": "formal_action_length_coverage_shortfall",
        "run_contract_sha256": "b" * 64,
        "code_revision": contract["code_revision"],
        "code_dirty": contract["code_dirty"],
        "scene_counts": {
            "total": 1, "completed": 1, "interrupted": 0, "failed": 0,
        },
        "per_scene": [{"scene_id": scene_id, "status": "completed"}],
    }))
    (output / "run_meta.json").write_text(json.dumps({
        "run_contract_sha256": "b" * 64,
        "code_revision": contract["code_revision"],
        "code_dirty": contract["code_dirty"],
        "params": params,
        "source_catalog": {
            "datasets": ["b1k"], "scene_ids": [scene_id],
            "manifest_sha256": [contract["source_manifest_sha256"]],
        },
        "formal_action_length_coverage": coverage,
    }))
    _seal_collection_output(
        output, status="partial", record_count=len(records))


def _collection_runner_args(
        tmp_path: Path, scene_ids: list[str], *, resume: bool) -> SimpleNamespace:
    manifest = tmp_path / "source-manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "schema_version": "egoconseq.b1k-source-manifest.v1",
            "scenes": [{"scene_id": scene_id} for scene_id in scene_ids],
        }))
    return SimpleNamespace(
        data_root=str(tmp_path), source_manifest=str(manifest),
        output_dir=str(tmp_path / "collection"), gpu_id=2,
        shard_id="pilot-02", code_revision="a" * 40,
        resume=resume, scene_timeout_s=1200, scenes=scene_ids,
        allow_dirty_code=True,
        collect_args=["--poses-per-scene", "1"],
    )


def test_collection_shard_runs_every_scene_in_an_independent_child(
        tmp_path, monkeypatch):
    """Catches native teardown of one scene aborting the next scene."""
    from scripts import run_b1k_collection_shard as collection_shard

    scene_ids = _fake_install(tmp_path, count=3)
    args = _collection_runner_args(tmp_path, scene_ids, resume=False)
    calls = []

    def fake_child(command, *, env, log_path, timeout_s):
        calls.append((command, dict(env), Path(log_path), timeout_s))
        scene_id = command[command.index("--scenes") + 1]
        output = Path(command[command.index("--out") + 1])
        _write_completed_collection_output(output, scene_id)
        return 0

    monkeypatch.setattr(
        collection_shard, "_run_isolated_collection", fake_child)
    result = collection_shard._run_shard(args)

    assert result == 0
    assert len(calls) == 3
    for index, (command, env, log_path, timeout_s) in enumerate(calls):
        scene_id = scene_ids[index]
        assert command.count("--scenes") == 1
        assert command[command.index("--scenes") + 1] == scene_id
        assert command[command.index("--out") + 1] == str(
            (tmp_path / "collection" / scene_id).resolve())
        assert command[command.index("--collection-shard-id") + 1] == (
            f"pilot-02-{scene_id}")
        assert env["OMNIGIBSON_GPU_ID"] == "2"
        assert env["OMNIGIBSON_DATA_PATH"] == str(tmp_path.resolve())
        assert env["OMNIGIBSON_APPDATA_PATH"] == str(
            (tmp_path / "appdata").resolve())
        assert env["OMNIGIBSON_HEADLESS"] == "True"
        assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"
        assert timeout_s == 1200
        assert log_path == (tmp_path / "collection" / "logs" /
                            f"{scene_id}.log").resolve()
    progress = json.loads(
        (tmp_path / "collection" / "shard-progress.json").read_text())
    assert progress["completed_scene_ids"] == scene_ids
    assert progress["failed_scenes"] == []
    assert progress["complete"] is True
    assert [row["returncode"] for row in progress["attempts"]] == [0, 0, 0]


def test_collection_shard_records_exact_failure_and_continues(
        tmp_path, monkeypatch):
    """Catches a native child status being masked or stopping the shard."""
    from scripts import run_b1k_collection_shard as collection_shard

    scene_ids = _fake_install(tmp_path, count=3)
    args = _collection_runner_args(tmp_path, scene_ids, resume=False)

    def fake_child(command, *, env, log_path, timeout_s):
        scene_id = command[command.index("--scenes") + 1]
        if scene_id == scene_ids[0]:
            return -11
        output = Path(command[command.index("--out") + 1])
        _write_completed_collection_output(output, scene_id)
        return 0

    monkeypatch.setattr(
        collection_shard, "_run_isolated_collection", fake_child)
    result = collection_shard._run_shard(args)

    assert result == 1
    progress = json.loads(
        (tmp_path / "collection" / "shard-progress.json").read_text())
    assert progress["completed_scene_ids"] == scene_ids[1:]
    assert progress["failed_scenes"] == [{
        "scene_id": scene_ids[0],
        "returncode": -11,
        "error": "collect.py returned nonzero",
    }]
    assert [row["scene_id"] for row in progress["attempts"]] == scene_ids


def test_collection_shard_resume_skips_only_digest_verified_outputs(
        tmp_path, monkeypatch):
    """Catches resume trusting a substituted completed source shard."""
    from scripts import run_b1k_collection_shard as collection_shard

    scene_ids = _fake_install(tmp_path, count=3)
    calls = []

    def fake_child(command, *, env, log_path, timeout_s):
        scene_id = command[command.index("--scenes") + 1]
        calls.append(scene_id)
        output = Path(command[command.index("--out") + 1])
        _write_completed_collection_output(output, scene_id)
        return 0

    monkeypatch.setattr(
        collection_shard, "_run_isolated_collection", fake_child)
    assert collection_shard._run_shard(
        _collection_runner_args(tmp_path, scene_ids, resume=False)) == 0
    assert calls == scene_ids

    assert collection_shard._run_shard(
        _collection_runner_args(tmp_path, scene_ids, resume=True)) == 0
    assert calls == scene_ids

    records = tmp_path / "collection" / scene_ids[0] / "records.jsonl"
    records.write_text('{"substituted":true}\n')
    with pytest.raises(ValueError, match="completed artifact changed"):
        collection_shard._run_shard(
            _collection_runner_args(tmp_path, scene_ids, resume=True))


def test_collection_shard_reuses_source_valid_partial_and_aggregates_coverage(
        tmp_path, monkeypatch):
    """Catches treating every per-scene length shortfall as data loss."""
    from scripts import run_b1k_collection_shard as collection_shard

    scene_ids = _fake_install(tmp_path, count=2)
    calls = []

    def fake_child(command, *, env, log_path, timeout_s):
        scene_id = command[command.index("--scenes") + 1]
        calls.append(scene_id)
        output = Path(command[command.index("--out") + 1])
        lengths = [1, 2, 3] if scene_id == scene_ids[0] else [4, 5, 6]
        _write_partial_collection_output(output, scene_id, lengths)
        return 1

    monkeypatch.setattr(
        collection_shard, "_run_isolated_collection", fake_child)
    assert collection_shard._run_shard(
        _collection_runner_args(tmp_path, scene_ids, resume=False)) == 0
    progress = json.loads(
        (tmp_path / "collection" / "shard-progress.json").read_text())
    assert progress["partial_valid_scene_ids"] == scene_ids
    assert progress["completed_scene_ids"] == []
    assert progress["failed_scenes"] == []
    assert progress["aggregate_formal_action_length_coverage"][
        "counts_by_length"] == {
            f"L{length}": 1 for length in range(1, 7)}
    assert progress["aggregate_formal_action_length_coverage"][
        "complete"] is True
    assert progress["complete"] is True

    assert collection_shard._run_shard(
        _collection_runner_args(tmp_path, scene_ids, resume=True)) == 0
    assert calls == scene_ids


def test_collection_shard_partial_resume_fails_closed_on_tamper(
        tmp_path, monkeypatch):
    """Catches resume trusting a changed partial-valid records shard."""
    from scripts import run_b1k_collection_shard as collection_shard

    scene_ids = _fake_install(tmp_path, count=1)

    def fake_child(command, *, env, log_path, timeout_s):
        scene_id = command[command.index("--scenes") + 1]
        output = Path(command[command.index("--out") + 1])
        _write_partial_collection_output(output, scene_id, [1])
        return 1

    monkeypatch.setattr(
        collection_shard, "_run_isolated_collection", fake_child)
    assert collection_shard._run_shard(
        _collection_runner_args(tmp_path, scene_ids, resume=False)) == 1
    records = tmp_path / "collection" / scene_ids[0] / "records.jsonl"
    records.write_text(records.read_text() + '{"substituted":true}\n')
    with pytest.raises(ValueError, match="partial-valid artifact changed"):
        collection_shard._run_shard(
            _collection_runner_args(tmp_path, scene_ids, resume=True))


def test_derive_shard_isolates_each_scene_in_a_child_process(
        tmp_path, monkeypatch):
    """Catches one native OG teardown aborting the remaining shard scenes."""
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=3)
    output_dir = tmp_path / "shard"
    calls = []

    def fake_run(command, *, timeout_s):
        assert timeout_s == 1200
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        scene_id = command[command.index("--scene") + 1]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_authority_fragment(tmp_path, scene_id)))
        return 0

    monkeypatch.setattr(
        manifest_cli, "_run_isolated_scene_child", fake_run)

    result = manifest_cli._derive_shard(SimpleNamespace(
        data_root=str(tmp_path), output_dir=str(output_dir),
        resume=True, scenes=scene_ids))

    assert result == 0
    assert [command[command.index("--scene") + 1] for command in calls] == (
        scene_ids)
    assert all(command[1].endswith("build_b1k_source_manifest.py")
               and "derive-scene" in command for command in calls)
    progress = json.loads((output_dir / "shard-progress.json").read_text())
    assert progress["completed_scene_ids"] == scene_ids
    assert progress["failed_scenes"] == []
    assert progress["complete"] is True


def test_derive_shard_records_native_child_failure_and_continues(
        tmp_path, monkeypatch):
    """Catches one failed scene preventing resumable progress on later ones."""
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=3)
    output_dir = tmp_path / "shard"

    def fake_run(command, *, timeout_s):
        assert timeout_s == 1200
        scene_id = command[command.index("--scene") + 1]
        if scene_id == scene_ids[0]:
            return -11
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_authority_fragment(tmp_path, scene_id)))
        return 0

    monkeypatch.setattr(
        manifest_cli, "_run_isolated_scene_child", fake_run)

    result = manifest_cli._derive_shard(SimpleNamespace(
        data_root=str(tmp_path), output_dir=str(output_dir),
        resume=True, scenes=scene_ids))

    assert result == 1
    progress = json.loads((output_dir / "shard-progress.json").read_text())
    assert progress["completed_scene_ids"] == scene_ids[1:]
    assert progress["failed_scenes"] == [{
        "scene_id": scene_ids[0],
        "attempt_id": f"{scene_ids[0]}:attempt-0000",
        "returncode": -11,
        "error": (
            "FileNotFoundError: [Errno 2] No such file or directory: '"
            f"{output_dir / (scene_ids[0] + '.json')}'"),
    }]
    assert progress["complete"] is False


def test_derive_shard_records_scene_timeout_and_continues(
        tmp_path, monkeypatch):
    """Catches one pathological scene occupying a shard indefinitely."""
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=3)
    output_dir = tmp_path / "shard"
    calls = []

    def fake_child(command, *, timeout_s):
        scene_id = command[command.index("--scene") + 1]
        calls.append((scene_id, timeout_s))
        if scene_id == scene_ids[0]:
            raise subprocess.TimeoutExpired(command, timeout_s)
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_authority_fragment(tmp_path, scene_id)))
        return 0

    monkeypatch.setattr(
        manifest_cli, "_run_isolated_scene_child", fake_child)

    result = manifest_cli._derive_shard(SimpleNamespace(
        data_root=str(tmp_path), output_dir=str(output_dir),
        resume=True, scenes=scene_ids, scene_timeout_s=1200))

    assert result == 1
    assert calls == [(scene_id, 1200) for scene_id in scene_ids]
    progress = json.loads((output_dir / "shard-progress.json").read_text())
    assert progress["completed_scene_ids"] == scene_ids[1:]
    assert progress["failed_scenes"] == [{
        "scene_id": scene_ids[0],
        "attempt_id": f"{scene_ids[0]}:attempt-0000",
        "returncode": None,
        "error": "TimeoutExpired: scene exceeded 1200 seconds",
    }]


def test_derive_scene_preserves_failure_before_process_shutdown(
        tmp_path, monkeypatch):
    """Catches Isaac's zero-status shutdown masking extraction failures."""
    from pipeline import b1k_sim
    from scripts import build_b1k_source_manifest as manifest_cli

    output = tmp_path / "scene.json"
    shutdowns = []
    exits = []
    monkeypatch.setattr(b1k_sim, "load_b1k_runtime", lambda: object())
    monkeypatch.setattr(
        manifest_cli, "_derive_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("runtime geometry unavailable")))
    monkeypatch.setattr(
        manifest_cli, "_shutdown_runtime",
        lambda: shutdowns.append("shutdown"))

    result = manifest_cli._derive_scene(SimpleNamespace(
        data_root=str(tmp_path), output=str(output), scene="scene-0"),
        hard_exit=lambda status: exits.append(status))

    assert result == 1
    assert exits == [1]
    assert shutdowns == []
    failure = json.loads(Path(f"{output}.failure.json").read_text())
    assert failure == {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-0",
        "error_type": "ValueError",
        "error": "runtime geometry unavailable",
    }


def _rectangle(x0, x1, z0, z1, *, y=0.2):
    return np.array([
        [[x0, y, z0], [x1, y, z0], [x1, y, z1]],
        [[x0, y, z0], [x1, y, z1], [x0, y, z1]],
    ], dtype=np.float64)


def test_slice_probe_uses_component_triangles_not_whole_mesh_convex_hull():
    """Catches filling the missing quadrant of a concave slice."""
    from pipeline import b1k_probe

    # Area is 3 m^2.  A whole-mesh convex hull would report 3.5 m^2.
    l_shape = np.concatenate([
        _rectangle(0.0, 1.0, 0.0, 2.0),
        _rectangle(1.0, 2.0, 0.0, 1.0),
    ])
    collision = [b1k_geometry.TriangleComponent("collision", l_shape)]
    visual = [b1k_geometry.TriangleComponent("visual", l_shape)]

    result = b1k_probe.collision_visual_slice_probe(
        collision, visual, floor_height_m=0.0)

    assert result["collision_slice_area_m2"] == pytest.approx(3.0)
    assert result["collision_minus_visual_area_m2"] == pytest.approx(0.0)
    assert result[
        "collision_to_visual_vertex_sampled_directed_hausdorff_approx_m"
    ] == pytest.approx(0.0)


def test_semantic_disc_query_aabb_prefilter_skips_far_instances():
    """Catches timing every scene triangle for a local 2.5-cm query."""
    near = b1k_semantic.RuntimeInstanceSpec(
        "/near", "near", "near", "near.n.01",
        _rectangle(-0.01, 0.01, -0.01, 0.01))
    far = b1k_semantic.RuntimeInstanceSpec(
        "/far", "far", "far", "far.n.01",
        _rectangle(10.0, 10.02, 10.0, 10.02))
    floor = [b1k_geometry.TriangleComponent(
        "/floor", _rectangle(-1.0, 1.0, -1.0, 1.0, y=0.0))]
    _geometry, semantics, _atom = b1k_semantic.build_b1k_authorities(
        floor_components=floor,
        collision_components=[
            b1k_geometry.TriangleComponent("/near", near.triangles),
            b1k_geometry.TriangleComponent("/far", far.triangles),
        ],
        instances=[near, far])
    assigned = semantics.assign(
        np.array([[0.0, 0.2, 0.0]]), tol=0.025)

    assert assigned.tolist() == [2]  # Stable IDs sort /far before /near.
    assert semantics.last_assign_diagnostics == {
        "query_point_count": 1,
        "aabb_candidate_count": 1,
        "indexed_surface_query_count": 1,
    }


def test_reset_errors_are_measured_at_the_same_hard_gate():
    """Catches making RGB repeatability, instead of pose, the reset gate."""
    from pipeline.b1k_sim import _physical_state_errors

    expected = {
        "object_poses": {"/object": (
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))},
        "joint_positions_rad": {"/object": np.array([0.25])},
    }
    actual = {
        "object_poses": {"/object": (
            np.array([1e-7, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))},
        "joint_positions_rad": {"/object": np.array([0.2500002])},
    }

    measured = _physical_state_errors(expected, actual)

    assert measured["max_object_position_error_m"] == pytest.approx(1e-7)
    assert measured["max_object_orientation_error_rad"] == 0.0
    assert measured["max_joint_position_error_rad"] == pytest.approx(2e-7)


def test_b1k_oracle_precheck_report_keeps_arc_deltas_by_radius():
    """Catches a pilot reporting A2 disagreement without measured deltas."""
    from pipeline import collection_runtime

    collection_runtime._reset_b1k_oracle_precheck_diagnostics()
    collection_runtime._record_b1k_oracle_precheck_diagnostic(
        scene_id="scene", pose_index=2, frame_id="frame", radius=0.2,
        length=3, action_tag="action-a", label="collision",
        precheck={
            "accepted": False,
            "reason": "contact_arc_mismatch",
            "full_collision": True,
            "depth_collision": True,
            "full_contact_arc_m": 0.42,
            "depth_contact_arc_m": 0.09,
            "contact_arc_difference_m": 0.33,
            "corridor_coverage": 0.97,
        })
    collection_runtime._record_b1k_oracle_precheck_diagnostic(
        scene_id="scene", pose_index=2, frame_id="frame", radius=0.2,
        length=1, action_tag="action-b", label="safe",
        precheck={
            "accepted": True,
            "reason": "accepted",
            "full_collision": False,
            "depth_collision": False,
            "full_contact_arc_m": None,
            "depth_contact_arc_m": None,
            "contact_arc_difference_m": None,
            "corridor_coverage": 1.0,
        })

    report = collection_runtime._b1k_oracle_precheck_report()

    assert report["contact_tolerance_m"] == 0.30
    assert report["consensus_by_radius_m"]["0.2"] == {
        "total": 2,
        "accepted": 1,
        "reasons": {"accepted": 1, "contact_arc_mismatch": 1},
        "contact_arc_difference_m": {
            "count": 1, "minimum": 0.33, "maximum": 0.33,
            "values": [0.33],
        },
    }


def test_b1k_depth_alignment_diagnostic_compares_raw_axial_depth_to_hit():
    """Catches diagnosing a contact mismatch without the actual depth ray."""
    from pipeline import actions, collection_runtime

    frame = SimpleNamespace(
        frame_id="frame", position=np.array([0.0, 0.0, 0.0]),
        yaw_rad=0.0, depth=np.full((5, 5), 3.0, dtype=np.float32),
        K=np.array([[2.0, 0.0, 2.0],
                    [0.0, 2.0, 2.0],
                    [0.0, 0.0, 1.0]]),
        sensor=SimpleNamespace(nominal_camera_offset_m=0.15))

    class Nav:
        def closest_obstacle(self, pose):
            assert pose == pytest.approx((0.0, 1.0, 0.0))
            return {
                "world_point": [0.0, 0.15, -1.0],
                "obstacle_identity": "/World/wall",
            }

    class Sim:
        def proposal_nav(self, position, yaw, *, radius_m):
            assert position == pytest.approx(frame.position)
            assert yaw == 0.0
            assert radius_m == 0.2
            return Nav()

    result = collection_runtime._b1k_depth_alignment_diagnostic(
        sim=Sim(), frame=frame,
        actions=[actions.Forward(2.0)], radius=0.2,
        full_physical={"first_contact_arc_m": 1.0},
        depth_physical={"first_contact_arc_m": 1.5,
                        "center_local": [0.0, 1.5]})

    assert result["depth_input"] == {
        "modality": "depth_linear",
        "renderer_semantics": "distance_to_image_plane",
        "unprojection_semantics": "axial_camera_z",
        "units": "m",
        "no_hit_value_after_adapter": 0.0,
    }
    assert result["camera"]["forward_pbench_world_xyz"] == pytest.approx(
        [0.0, 0.0, -1.0])
    assert result["full_contact"]["obstacle_identity"] == "/World/wall"
    assert result["full_contact"]["local_xyz_m"] == pytest.approx(
        [0.0, 0.15, 1.0])
    assert result["full_contact"]["projected_pixel_uv"] == [2, 2]
    assert result["full_contact"]["expected_axial_depth_m"] == 1.0
    assert result["full_contact"]["raw_depth_patch_m"] == [
        [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]


def test_runtime_bootstrap_derives_authority_and_queries_every_cspace(tmp_path):
    """Catches writing a manifest authority without adapter C-space replay."""
    from pipeline.b1k_sim import bootstrap_scene_authority

    runtime = _FakeRuntime()
    result = bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json", runtime=runtime)
    _floor, collisions, instances = _authority_inputs()
    expected = b1k_semantic.derive_scene_authority_atom(
        floor_components=_floor,
        collision_components=collisions,
        instances=instances)

    assert {
        key: value for key, value in result["scene_authority"].items()
        if key not in {
            "sha256", "canonical_replay_protocol",
            "canonical_state_sha256",
        }
    } == {
        key: value for key, value in expected.items() if key != "sha256"
    }
    assert result["geometry"] == {
        "floor_component_count": 1,
        "collision_component_count": 1,
        "runtime_instance_count": 1,
        "collision_triangle_count": 2,
        "cspace_query_by_radius_m": {
            "0.15": "navigable",
            "0.2": "navigable",
            "0.25": "navigable",
        },
    }
    assert result["reset_pose_errors"] == {
        "max_object_position_error_m": 0.0,
        "max_object_orientation_error_rad": 0.0,
        "max_joint_position_error_rad": 0.0,
    }


def test_runtime_bootstrap_binds_stable_canonical_replay_authority(tmp_path):
    """Catches bootstrapping from only one dump/load replay."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _CanonicalReplayRuntime

    runtime = _CanonicalReplayRuntime([0.0, 0.0, 0.0])
    result = bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json", runtime=runtime)

    assert runtime.loads == 4
    assert result["schema"] == "b1k-scene-authority-bootstrap.v2"
    assert result["canonical_replay"]["protocol"] == \
        "b1k-canonical-fixed-snapshot-replay.v2"
    assert result["canonical_replay"]["canonical_state"] == "state_2"
    assert len(result["canonical_replay"]["canonical_state_sha256"]) == 64
    replay_digests = result["canonical_replay"][
        "triangle_authority_sha256_by_state"]
    assert set(replay_digests) == {"state_2", "state_3", "state_4"}
    assert len(set(replay_digests.values())) == 1
    assert len(result["canonical_replay"][
        "precanonical_triangle_authority_sha256"]) == 64
    assert result["scene_authority"]["canonical_replay_protocol"] == \
        result["canonical_replay"]["protocol"]
    assert result["scene_authority"]["canonical_state_sha256"] == \
        result["canonical_replay"]["canonical_state_sha256"]
    assert result["scene_authority"]["sha256"] == \
        record.canonical_atom_sha256({
            key: value for key, value in result["scene_authority"].items()
            if key != "sha256"
        })


def test_runtime_bootstrap_reloads_one_frozen_snapshot(tmp_path):
    """Catches chaining each replay through a newly dumped state."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _FixedSnapshotReplayRuntime

    runtime = _FixedSnapshotReplayRuntime()
    result = bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json", runtime=runtime)

    assert runtime.loaded_serialized == [
        b"snapshot-0", b"snapshot-1", b"snapshot-1", b"snapshot-1"]
    assert runtime.dump_count == 2
    assert result["canonical_replay"]["canonical_state"] == "state_2"
    assert result["canonical_replay"]["state_2_to_state_3_errors"] == {
        "max_object_position_error_m": 0.0,
        "max_object_orientation_error_rad": 0.0,
        "max_joint_position_error_rad": 0.0,
    }


def test_runtime_bootstrap_permits_only_initial_canonicalization(tmp_path):
    """Catches comparing canonical state 2 to pre-canonical state 0."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _CanonicalReplayRuntime

    result = bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json",
        runtime=_CanonicalReplayRuntime([4e-6, 4e-6, 4e-6]))

    assert result["reset_pose_errors"]["max_object_position_error_m"] == \
        pytest.approx(4e-6)
    assert result["canonical_replay"]["state_1_to_state_2_errors"] == {
        "max_object_position_error_m": 0.0,
        "max_object_orientation_error_rad": 0.0,
        "max_joint_position_error_rad": 0.0,
    }


def test_runtime_bootstrap_rejects_accumulating_canonical_replay_drift(
        tmp_path):
    """Catches accepting individually small but accumulating replay drift."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _CanonicalReplayRuntime

    with pytest.raises(RuntimeError, match=(
            r"canonical replay state_2 -> state_4 position drift")):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_CanonicalReplayRuntime(
                [4e-6, 4.75e-6, 5.5e-6, 6.25e-6]))


def test_runtime_bootstrap_rejects_canonical_triangle_digest_drift(tmp_path):
    """Catches state-stable replay whose runtime triangle authority changes."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _CanonicalReplayRuntime

    with pytest.raises(RuntimeError, match="triangle authority drift"):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_CanonicalReplayRuntime(
                [0.0, 0.0, 0.0], triangle_drift_at=2))


def test_runtime_bootstrap_keeps_one_micrometre_replay_gate(tmp_path):
    """Catches weakening the existing 1e-6 m stable-replay tolerance."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _CanonicalReplayRuntime

    bootstrap_scene_authority(
        "scene-0", tmp_path / "scene-0.json",
        runtime=_CanonicalReplayRuntime(
            [0.0, 0.0, 0.999e-6, 0.999e-6]))
    with pytest.raises(RuntimeError, match=r"1e-06 m"):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_CanonicalReplayRuntime(
                [0.0, 0.0, 1.001e-6, 1.001e-6]))


@pytest.mark.parametrize("invalid_kind", [
    "position_nan", "orientation_zero", "joint_nan",
])
def test_runtime_bootstrap_rejects_invalid_stable_replay_state(
        tmp_path, invalid_kind):
    """Catches NaN or zero-quaternion state bypassing tolerance comparisons."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _InvalidCanonicalReplayRuntime

    with pytest.raises(RuntimeError, match="invalid physical state"):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_InvalidCanonicalReplayRuntime(invalid_kind))


def test_runtime_bootstrap_rejects_invalid_precanonical_state(tmp_path):
    """Catches treating an invalid state 0 as numeric canonicalization."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _CanonicalReplayRuntime

    runtime = _CanonicalReplayRuntime([0.0, 0.0, 0.0])
    runtime._state["object_poses"]["/World/table"][1][:] = 0.0

    with pytest.raises(
            RuntimeError, match="canonical replay state_0.*invalid"):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json", runtime=runtime)


@pytest.mark.parametrize("malformed_kind", [
    "missing_object_poses",
    "list_object_poses",
    "empty_object_poses",
    "missing_joint_positions",
    "list_joint_positions",
])
def test_runtime_bootstrap_rejects_malformed_canonical_state_mapping(
        tmp_path, malformed_kind):
    """Catches missing/list/empty physical state passing as an empty mapping."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _MalformedCanonicalReplayRuntime

    with pytest.raises(RuntimeError, match="invalid physical state"):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_MalformedCanonicalReplayRuntime(malformed_kind))


def test_runtime_bootstrap_rejects_broadcastable_joint_shape_drift(tmp_path):
    """Catches NumPy broadcasting unequal canonical joint vector shapes."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _MalformedCanonicalReplayRuntime

    with pytest.raises(RuntimeError, match="joint drift.*shape") as captured:
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_MalformedCanonicalReplayRuntime(
                "broadcast_joint_shape"))

    failure = tmp_path / "failure.json.failure.json"
    failure.write_text(json.dumps({
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-0",
        "error_type": "RuntimeError",
        "error": str(captured.value),
    }))
    from pipeline.b1k_authority_audit import audit_authority_failures
    report = audit_authority_failures(tmp_path)
    assert report["category_counts"] == {"reset_joint_drift": 1}


def test_runtime_bootstrap_rejects_first_replay_joint_shape_change(tmp_path):
    """Catches state-0 diagnostics broadcasting a changed joint vector."""
    from pipeline.b1k_sim import bootstrap_scene_authority
    from tests.test_pl_b1k_runtime import _FirstReplayJointShapeRuntime

    with pytest.raises(RuntimeError, match="joint drift.*shape"):
        bootstrap_scene_authority(
            "scene-0", tmp_path / "scene-0.json",
            runtime=_FirstReplayJointShapeRuntime())


def test_b1k_authority_failure_audit_is_read_only_and_separates_clearance(
        tmp_path):
    """Catches losing failures or grouping empty C-space with replay drift."""
    from pipeline.b1k_authority_audit import audit_authority_failures

    reset = tmp_path / "shard-00" / "scene-a.json.failure.json"
    reset.parent.mkdir()
    reset.write_text(json.dumps({
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-a",
        "error_type": "RuntimeError",
        "error": "B1K object reset position drift for /World/tree",
    }))
    clearance = tmp_path / "shard-01" / "scene-b.json.failure.json"
    clearance.parent.mkdir()
    clearance.write_text(json.dumps({
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-b",
        "error_type": "ValueError",
        "error": "B1K radius 0.15 has empty clearance-conditioned free space",
    }))
    progress = tmp_path / "shard-00" / "shard-progress.json"
    progress.parent.mkdir(exist_ok=True)
    progress.write_text(json.dumps({
        "schema": "b1k-source-manifest-shard-progress.v2",
        "failed_scenes": [
            {
                "scene_id": "scene-a", "returncode": 1,
                "error": "FileNotFoundError: scene-a fragment is absent",
            },
            {
                "scene_id": "scene-c", "returncode": None,
                "error": "TimeoutExpired: scene exceeded 1200 seconds",
            },
            {
                "scene_id": "scene-d", "returncode": 1,
                "error": "FileNotFoundError: scene-d fragment is absent",
                "child_failure": {
                    "schema": "b1k-scene-authority-failure.v1",
                    "scene_id": "scene-d",
                    "error_type": "RuntimeError",
                    "error": "B1K canonical replay triangle authority drift",
                },
            },
        ],
        "attempts": [{
            "scene_id": "scene-d", "status": "failed", "returncode": 1,
            "error": "FileNotFoundError: scene-d fragment is absent",
            "child_failure": {
                "schema": "b1k-scene-authority-failure.v1",
                "scene_id": "scene-d",
                "error_type": "RuntimeError",
                "error": "B1K canonical replay triangle authority drift",
            },
        }],
    }))
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }

    report = audit_authority_failures(tmp_path)

    assert report["schema"] == "b1k-authority-failure-audit.v1"
    assert report["failure_event_count"] == 4
    assert report["evidence_record_count"] == 8
    assert report["unique_failed_scene_count"] == 4
    assert report["category_counts"] == {
        "empty_clearance": 1,
        "reset_position_drift": 1,
        "timeout": 1,
        "triangle_authority_drift": 1,
    }
    reset_row = next(
        row for row in report["failures"] if row["scene_id"] == "scene-a")
    assert reset_row["category"] == "reset_position_drift"
    assert len(reset_row["evidence"]) == 2
    digest_row = next(
        row for row in report["failures"] if row["scene_id"] == "scene-d")
    assert digest_row["category"] == "triangle_authority_drift"
    assert {row["source_kind"] for row in digest_row["evidence"]} == {
        "shard_progress", "shard_attempt", "child_failure",
    }
    assert before == {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.json")
    }

    completed = subprocess.run([
        sys.executable, "scripts/audit_b1k_authority_failures.py",
        "--root", str(tmp_path),
    ], cwd=Path(__file__).resolve().parents[1], check=True,
        capture_output=True, text=True)
    assert json.loads(completed.stdout) == report


def test_b1k_authority_failure_audit_retains_each_failed_retry(tmp_path):
    """Catches collapsing distinct historical attempts for the same scene."""
    from pipeline.b1k_authority_audit import audit_authority_failures

    progress = tmp_path / "shard-00" / "shard-progress.json"
    progress.parent.mkdir()
    progress.write_text(json.dumps({
        "schema": "b1k-source-manifest-shard-progress.v2",
        "failed_scenes": [],
        "attempts": [
            {
                "scene_id": "scene-a", "status": "failed", "returncode": 1,
                "error": "B1K object reset position drift for /World/tree",
            },
            {
                "scene_id": "scene-a", "status": "failed", "returncode": 1,
                "error": "B1K canonical replay triangle authority drift",
            },
        ],
    }))

    report = audit_authority_failures(tmp_path)

    assert report["failure_event_count"] == 2
    assert report["unique_failed_scene_count"] == 1
    assert report["evidence_record_count"] == 2
    assert [row["category"] for row in report["failures"]] == [
        "reset_position_drift", "triangle_authority_drift",
    ]


def test_install_manifest_refuses_version_drift(tmp_path):
    """Catches recording an observed but unpinned B1K runtime as verified."""
    from pipeline import b1k_source_builder

    scenes = _fake_install(tmp_path, count=3)
    observed = {
        "source_commit": "26f2c7ef7b9cf96bd0414f81e1e751e493762779",
        "source_tag": "v3.9.1",
        "python": "3.11.15",
        "omnigibson": "3.9.1",
        "bddl": "3.7.0",
        "isaac_sim": "5.1.0.0",
        "torch": "2.7.0+cu128",
        "behavior-1k-assets": "3.9.0",
        "omnigibson-robot-assets": "3.8.2",
        "omnigibson_editable_root": str(
            (tmp_path / "source" / "OmniGibson").resolve()),
    }

    manifest = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed=observed, scene_ids=scenes)
    assert manifest["verified"] is True
    assert manifest["installed_catalog_count"] == 3

    with pytest.raises(ValueError, match="omnigibson"):
        b1k_source_builder.build_install_manifest(
            data_root=tmp_path, source_root=tmp_path / "source",
            observed={**observed, "omnigibson": "3.9.2"},
            scene_ids=scenes)


def test_manifest_cli_assembles_existing_authority_fragments(tmp_path):
    """Catches a documented builder command that cannot emit a usable manifest."""
    from pipeline import b1k_source_builder

    scene_ids = _fake_install(tmp_path, count=3)
    fragments = tmp_path / "fragments"
    fragments.mkdir()
    for scene_id in scene_ids:
        (fragments / f"{scene_id}.json").write_text(json.dumps(
            _authority_fragment(tmp_path, scene_id), sort_keys=True))
    observed = {
        **b1k_source_builder.PINNED_INSTALL,
        "omnigibson_editable_root": str(
            (tmp_path / "source" / "OmniGibson").resolve()),
    }
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed=observed,
        scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install))
    output = tmp_path / "pilot-source.json"

    subprocess.run([
        sys.executable, "scripts/build_b1k_source_manifest.py", "assemble",
        "--data-root", str(tmp_path),
        "--install-manifest", str(install_path),
        "--expected-install-manifest-sha256",
        hashlib.sha256(install_path.read_bytes()).hexdigest(),
        "--fragment-dir", str(fragments),
        "--selection-scope", "three_scene_pilot",
        "--output", str(output),
        "--scenes", *scene_ids,
    ], cwd=Path(__file__).resolve().parents[1], check=True)

    built = json.loads(output.read_text())
    assert built["installed_catalog_count"] == 3
    assert built["selected_scene_count"] == 3
    assert built["selection_scope"] == "three_scene_pilot"


def _write_catalog_audit_shards(
        root: Path, scene_ids: list[str], *, excluded: dict[str, list[dict]]
        ) -> list[Path]:
    """Write four complete source-authority histories for audit assembly."""
    from scripts import build_b1k_source_manifest as manifest_cli

    shards = []
    for shard_index, assigned in enumerate(
            [scene_ids[index::4] for index in range(4)]):
        directory = root / f"shard-{shard_index:02d}"
        directory.mkdir()
        completed = {}
        attempts = []
        for scene_id in assigned:
            failures = excluded.get(scene_id, [])
            if failures:
                for attempt_number, failure in enumerate(failures):
                    attempt_id = f"{scene_id}:attempt-{attempt_number:04d}"
                    child_failure = {**failure, "attempt_id": attempt_id}
                    attempts.append({
                        "attempt_id": attempt_id,
                        "scene_id": scene_id,
                        "status": "failed",
                        "returncode": 1,
                        "error": "ValueError: scene authority derivation failed",
                        "child_failure": child_failure,
                    })
                continue
            fragment = directory / f"{scene_id}.json"
            fragment.write_text(json.dumps(
                _catalog_audit_authority_fragment(root, scene_id),
                sort_keys=True))
            completed[scene_id] = {
                "scene_id": scene_id,
                "fragment": manifest_cli._fragment_identity(fragment, directory),
            }
            attempts.append({
                "attempt_id": f"{scene_id}:attempt-0000",
                "scene_id": scene_id,
                "status": "completed",
                "returncode": 0,
            })
        directory.joinpath("shard-progress.json").write_text(json.dumps({
            "schema": "b1k-source-manifest-shard-progress.v2",
            "data_root": str(root.resolve()),
            "scene_timeout_s": 1200,
            "requested_scene_ids": assigned,
            "completed_scene_ids": [
                scene_id for scene_id in assigned if scene_id in completed],
            "completed_scenes": [
                completed[scene_id] for scene_id in assigned
                if scene_id in completed],
            "failed_scenes": [
                {
                    "scene_id": scene_id,
                    "attempt_id": attempts[-1]["attempt_id"],
                    "returncode": attempts[-1]["returncode"],
                    "error": attempts[-1]["error"],
                    "child_failure": attempts[-1]["child_failure"],
                }
                for scene_id in assigned
                if scene_id in excluded and excluded[scene_id]
            ],
            "attempts": attempts,
            "complete": len(completed) == len(assigned),
        }, sort_keys=True))
        shards.append(directory)
    return shards


def _catalog_audit_args(root: Path, install_path: Path, shard_dirs: list[Path],
                        output: Path, audit_output: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        data_root=str(root),
        install_manifest=str(install_path),
        expected_install_manifest_sha256=hashlib.sha256(
            install_path.read_bytes()).hexdigest(),
        fragment_dir=[str(path) for path in shard_dirs],
        output=str(output),
        audit_output=str(audit_output),
    )


def test_catalog_audit_assembly_emits_fixed_scope_manifest_and_audit(
        tmp_path):
    """Catches treating audited exclusions as unscoped user-selected scenes."""
    from pipeline import b1k_source_builder
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=51)
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed={
            **b1k_source_builder.PINNED_INSTALL,
            "omnigibson_editable_root": str(
                (tmp_path / "source" / "OmniGibson").resolve()),
        }, scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install, sort_keys=True))
    typed_failure = {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": scene_ids[-1],
        "error_type": "ValueError",
        "error": "B1K radius 0.15 has empty clearance-conditioned free space",
    }
    shards = _write_catalog_audit_shards(
        tmp_path, scene_ids, excluded={scene_ids[-1]: [
            typed_failure, typed_failure,
        ]})
    output = tmp_path / "source.json"
    audit_output = tmp_path / "audit.json"

    assert manifest_cli._assemble_catalog_audit(_catalog_audit_args(
        tmp_path, install_path, shards, output, audit_output)) == 0

    manifest = json.loads(output.read_text())
    audit = json.loads(audit_output.read_text())
    catalog = install["installed_scene_ids"]
    excluded_id = scene_ids[-1]
    excluded_attempts = [
        {"shard": directory.name, "attempt_index": index}
        for directory in shards
        for index, attempt in enumerate(json.loads(
            (directory / "shard-progress.json").read_text())["attempts"])
        if attempt["scene_id"] == excluded_id and attempt["status"] == "failed"
    ]
    assert manifest["selection_scope"] == "b1k-catalog-authority-audit.v1"
    assert [row["scene_id"] for row in manifest["scenes"]] == [
        scene_id for scene_id in catalog if scene_id != excluded_id]
    assert audit["schema"] == "b1k-catalog-authority-audit.v1"
    assert audit["installed_scene_ids"] == catalog
    assert [row["scene_id"] for row in audit["accepted"]] == [
        scene_id for scene_id in catalog if scene_id != excluded_id]
    assert audit["excluded"] == [{
        "scene_id": excluded_id,
        "terminal": "excluded_deterministic_failure",
        "failure": {
            "error_type": "ValueError",
            "error": (
                "B1K radius 0.15 has empty clearance-conditioned free space"),
            "category": "empty_clearance",
        },
        "attempts": excluded_attempts,
    }]
    assert audit["source_identities"]["install_manifest"]["sha256"] == \
        hashlib.sha256(install_path.read_bytes()).hexdigest()
    assert len(audit["source_identities"]["shards"]) == 4


@pytest.mark.parametrize("failures", [
    [{
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-50",
        "error_type": "TimeoutError",
        "error": "scene timed out after 1200 seconds",
    }] * 2,
    [{
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-50",
        "error_type": "ValueError",
        "error": "stable failure one",
    }, {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": "scene-50",
        "error_type": "ValueError",
        "error": "stable failure two",
    }],
])
def test_catalog_audit_assembly_rejects_unresolved_failure_histories(
        tmp_path, failures):
    """Catches accepting transient or divergent failures as exclusions."""
    from pipeline import b1k_source_builder
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=51)
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed={
            **b1k_source_builder.PINNED_INSTALL,
            "omnigibson_editable_root": str(
                (tmp_path / "source" / "OmniGibson").resolve()),
        }, scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install, sort_keys=True))
    shards = _write_catalog_audit_shards(
        tmp_path, scene_ids, excluded={scene_ids[-1]: failures})

    with pytest.raises(ValueError, match="unresolved"):
        manifest_cli._assemble_catalog_audit(_catalog_audit_args(
            tmp_path, install_path, shards, tmp_path / "source.json",
            tmp_path / "audit.json"))


def test_catalog_audit_assembly_rejects_legacy_replay_free_fragment(tmp_path):
    """Catches accepting an authority that did not prove state-2--4 replay."""
    from pipeline import b1k_source_builder
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=51)
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed={
            **b1k_source_builder.PINNED_INSTALL,
            "omnigibson_editable_root": str(
                (tmp_path / "source" / "OmniGibson").resolve()),
        }, scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install, sort_keys=True))
    shards = _write_catalog_audit_shards(
        tmp_path, scene_ids, excluded={})
    legacy_scene = scene_ids[0]
    legacy_shard = next(
        directory for directory in shards
        if (directory / f"{legacy_scene}.json").is_file())
    legacy_fragment = legacy_shard / f"{legacy_scene}.json"
    legacy_fragment.write_text(json.dumps(
        _authority_fragment(tmp_path, legacy_scene), sort_keys=True))
    progress_path = legacy_shard / "shard-progress.json"
    progress = json.loads(progress_path.read_text())
    completed_row = next(
        row for row in progress["completed_scenes"]
        if row["scene_id"] == legacy_scene)
    completed_row["fragment"] = manifest_cli._fragment_identity(
        legacy_fragment, legacy_shard)
    progress_path.write_text(json.dumps(progress, sort_keys=True))

    with pytest.raises(ValueError, match="unresolved"):
        manifest_cli._assemble_catalog_audit(_catalog_audit_args(
            tmp_path, install_path, shards, tmp_path / "source.json",
            tmp_path / "audit.json"))


@pytest.mark.parametrize(("error_type", "error"), [
    ("FileNotFoundError", "B1K source input is missing"),
    ("RuntimeError", "CUBLAS_STATUS_ALLOC_FAILED"),
])
def test_catalog_audit_assembly_rejects_unallowlisted_typed_failures(
        tmp_path, error_type, error):
    """Catches turning missing files or allocation failures into exclusions."""
    from pipeline import b1k_source_builder
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=51)
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed={
            **b1k_source_builder.PINNED_INSTALL,
            "omnigibson_editable_root": str(
                (tmp_path / "source" / "OmniGibson").resolve()),
        }, scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install, sort_keys=True))
    failure = {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": scene_ids[-1],
        "error_type": error_type,
        "error": error,
    }
    shards = _write_catalog_audit_shards(
        tmp_path, scene_ids, excluded={scene_ids[-1]: [failure, failure]})

    with pytest.raises(ValueError, match="unresolved"):
        manifest_cli._assemble_catalog_audit(_catalog_audit_args(
            tmp_path, install_path, shards, tmp_path / "source.json",
            tmp_path / "audit.json"))


def test_catalog_audit_assembly_rejects_replayed_failure_attempt(tmp_path):
    """Catches counting one persisted child failure twice as independent work."""
    from pipeline import b1k_source_builder
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=51)
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed={
            **b1k_source_builder.PINNED_INSTALL,
            "omnigibson_editable_root": str(
                (tmp_path / "source" / "OmniGibson").resolve()),
        }, scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install, sort_keys=True))
    failure = {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": scene_ids[-1],
        "error_type": "ValueError",
        "error": "B1K radius 0.15 has empty clearance-conditioned free space",
    }
    shards = _write_catalog_audit_shards(
        tmp_path, scene_ids, excluded={scene_ids[-1]: [failure, failure]})
    progress_path = next(
        directory / "shard-progress.json" for directory in shards
        if any(row["scene_id"] == scene_ids[-1] for row in json.loads(
            (directory / "shard-progress.json").read_text())["attempts"]))
    progress = json.loads(progress_path.read_text())
    failed = [row for row in progress["attempts"]
              if row["scene_id"] == scene_ids[-1]]
    failed[1]["attempt_id"] = failed[0].get("attempt_id")
    progress_path.write_text(json.dumps(progress, sort_keys=True))

    with pytest.raises(ValueError, match="unresolved"):
        manifest_cli._assemble_catalog_audit(_catalog_audit_args(
            tmp_path, install_path, shards, tmp_path / "source.json",
            tmp_path / "audit.json"))


def test_catalog_audit_assembly_rejects_completed_history_without_fragment(
        tmp_path):
    """Catches excluding a scene after a contradictory completed attempt."""
    from pipeline import b1k_source_builder
    from scripts import build_b1k_source_manifest as manifest_cli

    scene_ids = _fake_install(tmp_path, count=51)
    install = b1k_source_builder.build_install_manifest(
        data_root=tmp_path, source_root=tmp_path / "source",
        observed={
            **b1k_source_builder.PINNED_INSTALL,
            "omnigibson_editable_root": str(
                (tmp_path / "source" / "OmniGibson").resolve()),
        }, scene_ids=scene_ids)
    install_path = tmp_path / "install.json"
    install_path.write_text(json.dumps(install, sort_keys=True))
    failure = {
        "schema": "b1k-scene-authority-failure.v1",
        "scene_id": scene_ids[-1],
        "error_type": "ValueError",
        "error": "B1K radius 0.15 has empty clearance-conditioned free space",
    }
    shards = _write_catalog_audit_shards(
        tmp_path, scene_ids, excluded={scene_ids[-1]: [failure, failure]})
    progress_path = next(
        directory / "shard-progress.json" for directory in shards
        if any(row["scene_id"] == scene_ids[-1] for row in json.loads(
            (directory / "shard-progress.json").read_text())["attempts"]))
    progress = json.loads(progress_path.read_text())
    progress["attempts"].append({
        "attempt_id": f"{scene_ids[-1]}:attempt-0002",
        "scene_id": scene_ids[-1],
        "status": "completed",
        "returncode": 0,
    })
    progress_path.write_text(json.dumps(progress, sort_keys=True))

    with pytest.raises(ValueError, match="unresolved"):
        manifest_cli._assemble_catalog_audit(_catalog_audit_args(
            tmp_path, install_path, shards, tmp_path / "source.json",
            tmp_path / "audit.json"))


def test_adapter_probe_records_rgbd_and_source_semantic_diagnostics(tmp_path):
    """Catches a smoke path that bypasses the committed B1K session."""
    from pipeline import b1k_probe, config, scene_pool
    from pipeline.b1k_sim import B1KSimSession

    class ProbeRuntime(_FakeRuntime):
        def render(self, sensor):
            value, info = super().render(sensor)
            width = value["rgb"].shape[1]
            value["rgb"][..., 0] = \
                np.arange(width, dtype=np.uint16) % 256
            return value, info

        def visual_components(self, scene):
            _floor, collisions, _instances = _authority_inputs()
            return collisions

    manifest = Path(__file__).resolve().parent / "unused"
    del manifest
    from tests.test_pl_b1k_runtime import _write_manifest
    scene = scene_pool.discover_b1k_train_scenes(
        tmp_path, _write_manifest(tmp_path))[0]
    with B1KSimSession(
            scene, fovs=config.BENCH_FOVS_DEG,
            runtime=ProbeRuntime()) as session:
        session.id_to_cat[2] = "wall.n.01"
        session.assign_instances = lambda points: np.resize(
            np.array([1, 2], dtype=np.int64), len(points))
        result = b1k_probe.run_adapter_probe(
            session, fovs=config.BENCH_FOVS_DEG, seed=7,
            semantic_sample_count=32)

    assert result["scene_id"] == scene.scene_id
    assert len(result["observations"]) == 2
    assert all(value["rgb_unique_count"] > 1
               for value in result["observations"])
    assert all(value["valid_depth_ratio"] > 0
               for value in result["observations"])
    assert result["smoke_pass"] is True
    assert result["pose_search"]["attempt_count"] == 1
    assert result["pose_search"]["selected_attempt_index"] == 0
    assert result["sample_pose"]["yaw_rad"] == 0.0
    assert all(value["source_instance_id_count"] == 2
               for value in result["observations"])
    assert all(value["source_nonstructural_instance_id_count"] == 1
               for value in result["observations"])
    assert all(value["source_nonstructural_instance_ids"] == [1]
               for value in result["observations"])
    assert all(value["source_semantic_resolution_rate"] == 1.0
               for value in result["observations"])
    assert result["reset"]["hard_gate_pass"] is True
    assert result["reset"]["rgb_psnr_db"] is not None
    assert result["semantic_resolution"]["query_radius_m"] == 0.025
    assert result["semantic_resolution"]["sample_count"] == 32
    assert result["semantic_resolution"]["elapsed_seconds"] >= 0.0
