"""CLI routing tests for explicit record-validation authority."""

import hashlib
import json
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np

from pipeline import perception, record, scene_pool
from pipeline import consensus
from pipeline.consensus import oracle_consensus
from tests._synthetic import clean_record, source_provenance


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_records.py"


def _write_records(path, value):
    path.write_text(json.dumps(value) + "\n")


def _byte_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _strict_r2r_record():
    value = clean_record()
    outcome = value["outcomes"][0]
    contact = outcome["physical"]["contact"]
    center = np.asarray(contact["center_local"], dtype=np.float64)
    radius = outcome["body"]["radius_m"]
    surface = center + np.array([0.0, radius - 0.01])
    boundary = center - np.array([0.0, 0.01])
    contact["world_point"] = perception.world_from_local(
        np.array([[surface[0], 0.0, surface[1]]]),
        value["pose"]["position"], value["pose"]["yaw_rad"])[0].tolist()
    contact["configuration_boundary_world_point"] = \
        perception.world_from_local(
            np.array([[boundary[0], 0.0, boundary[1]]]),
            value["pose"]["position"],
            value["pose"]["yaw_rad"])[0].tolist()
    contact["surface_protocol"] = \
        "radius_extrapolated_navmesh_boundary_v2"
    contact["full_geometry_attribution"] = {
        "instance_id": 5,
        "category": "wall",
        "unattributed": False,
    }
    outcome["oracle_consensus"] = oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(
            outcome["actions"], [{
                "perturbation_id": perturbation["id"],
                "transform": {
                    "x_m": perturbation["x_m"],
                    "z_m": perturbation["z_m"],
                    "yaw_deg": perturbation["yaw_deg"],
                },
                "physical": copy.deepcopy(outcome["physical"]),
                "depth_physical": copy.deepcopy(outcome["depth_physical"]),
                "corridor_coverage": 1.0,
                "exact_contact_identity": {
                    "confirmed": True,
                    "reason": "confirmed",
                    "instance_id": 5,
                    "category": "wall",
                    "instance_triangles_sha256": "d" * 64,
                    "contact_face_distance_m": 0.01,
                    "runner_up_face_distance_m": None,
                    "face_distance_margin_m": None,
                },
            } for perturbation in
                consensus.R2R_A_STABILITY_PERTURBATIONS])
    value["source"] = source_provenance(value["scene_id"], dataset="r2r")
    value["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    value["collection_contract"] = record.r2r_v16_collection_contract(
        value["source"], "main")
    return value


def _legacy_record():
    return clean_record()


def _strict_authorities(tmp_path):
    value = _strict_r2r_record()
    records = tmp_path / "records.jsonl"
    run_meta = tmp_path / "run_meta.json"
    run_contract = tmp_path / "run_contract.json"
    contract = {
        "params": {
            "collection_mode": "main",
            "r2r_train_episodes": str(tmp_path / "episodes.json.gz"),
            "mp3d_root": str(tmp_path / "mp3d"),
        },
        "resolved_scenes": [value["source"]],
    }
    metadata = {
        "record_schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "run_contract_sha256": _canonical_sha256(contract),
        **contract,
    }
    _write_records(records, value)
    run_meta.write_text(json.dumps(metadata))
    run_contract.write_text(json.dumps(contract))
    return records, run_meta, run_contract, contract


def test_check_records_requires_an_explicit_validation_level(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, _legacy_record())

    result = _run(records)

    assert result.returncode == 2
    assert "--validation-level" in result.stderr


def test_check_records_accepts_explicit_local_level(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, _legacy_record())

    result = _run(records, "--validation-level", "local")

    assert result.returncode == 0, result.stdout + result.stderr


def test_check_records_builds_strict_context_from_run_meta(tmp_path):
    records, run_meta, _run_contract, _contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta,
        "--expected-run-meta-sha256", _byte_sha256(run_meta))

    assert result.returncode == 1
    assert "trusted R2R source unavailable" in result.stdout


def test_check_records_routes_b1k_run_meta_to_authenticated_manifest(
        tmp_path):
    """Catches standalone source validation treating B1K as R2R."""
    from tests.test_pl_b1k_runtime import _write_manifest

    manifest = _write_manifest(tmp_path)
    scenes = scene_pool.discover_b1k_train_scenes(tmp_path, manifest)
    records = tmp_path / "records.jsonl"
    records.write_text("")
    run_meta = tmp_path / "run_meta.json"
    run_meta.write_text(json.dumps({
        "record_schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "run_contract_sha256": "c" * 64,
        "params": {
            "backend": "b1k",
            "collection_mode": "main",
            "b1k_data_root": str(tmp_path),
            "b1k_source_manifest": str(manifest),
            "action_mode": "balanced",
            "setting_sampling_policy":
                "one_sensor_one_body_per_pose.v1",
        },
        "resolved_scenes": [{
            **scene.provenance(), "scene_path": scene.scene_path,
        } for scene in scenes],
    }))

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta,
        "--expected-run-meta-sha256", _byte_sha256(run_meta))

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "checked 0 records; 0 violations"


def test_check_records_routes_gs_v18_to_the_registered_source_bundle(
        tmp_path):
    from tests.test_pl_dataset_contracts import _gs_run_meta

    metadata, _source = _gs_run_meta(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text("")
    run_meta = tmp_path / "run_meta.json"
    run_meta.write_text(json.dumps(metadata))

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta,
        "--expected-run-meta-sha256", _byte_sha256(run_meta))

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "checked 0 records; 0 violations"


def test_check_records_rejects_run_contract_as_source_authority(tmp_path):
    records, _run_meta, run_contract, contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source",
        "--run-contract", run_contract,
        "--expected-run-contract-sha256", _canonical_sha256(contract))

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_check_records_run_meta_requires_external_expected_digest(tmp_path):
    records, run_meta, _run_contract, _contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta)

    assert result.returncode == 2
    assert "--expected-run-meta-sha256" in result.stderr


def test_check_records_source_does_not_accept_run_contract(tmp_path):
    records, _run_meta, run_contract, _contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source",
        "--run-contract", run_contract)

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_check_records_rejects_wrong_external_run_meta_digest(tmp_path):
    records, run_meta, _run_contract, _contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta,
        "--expected-run-meta-sha256", "0" * 64)

    assert result.returncode == 2
    assert "run metadata hash" in result.stderr


def test_check_records_rejects_run_contract_digest_option(tmp_path):
    records, _run_meta, run_contract, _contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source",
        "--run-contract", run_contract,
        "--expected-run-contract-sha256", "0" * 64)

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_check_records_rejects_self_consistent_fake_run_meta(tmp_path):
    records, run_meta, _run_contract, trusted_contract = \
        _strict_authorities(tmp_path)
    expected_digest = _byte_sha256(run_meta)
    fake_contract = {**trusted_contract, "attacker_controlled": True}
    fake = json.loads(run_meta.read_text())
    fake["run_contract_sha256"] = _canonical_sha256(fake_contract)
    run_meta.write_text(json.dumps(fake))

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta,
        "--expected-run-meta-sha256", expected_digest)

    assert result.returncode == 2
    assert "run metadata hash" in result.stderr


def test_check_records_rejects_malformed_expected_digest(tmp_path):
    records, run_meta, _run_contract, _contract = \
        _strict_authorities(tmp_path)

    result = _run(
        records, "--validation-level", "source", "--run-meta", run_meta,
        "--expected-run-meta-sha256", "A" * 64)

    assert result.returncode == 2
    assert "lowercase hexadecimal" in result.stderr


def test_check_records_rejects_source_authority_at_local_level(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, _legacy_record())

    result = _run(
        records, "--validation-level", "local",
        "--expected-run-meta-sha256", "0" * 64)

    assert result.returncode == 2
    assert "local validation cannot consume source authority" in result.stderr


def test_check_records_source_requires_the_run_meta_digest_pair(tmp_path):
    records, run_meta, run_contract, _contract = \
        _strict_authorities(tmp_path)

    del run_contract
    result = _run(
        records, "--validation-level", "source",
        "--expected-run-meta-sha256", "0" * 64)

    assert result.returncode == 2
    assert "--run-meta" in result.stderr
