"""Source-authority regressions for exact A3 contact identity."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pipeline import (
    benchmark_tasks, consensus, dataset_contracts, record, scene_pool,
    semantic, source_manifest, validate,
)


def _write_contact_scene(root):
    root.mkdir(parents=True, exist_ok=True)
    scene = root / "TARGET.glb"
    scene.write_bytes(b"")
    vertices = np.array([
        (0.0, 0.0, 0.0, 0, 0, 0),
        (1.0, 0.0, 0.0, 0, 0, 0),
        (0.0, 0.0, 1.0, 0, 0, 0),
        (5.0, 0.0, 0.0, 0, 0, 0),
        (6.0, 0.0, 0.0, 0, 0, 0),
        (5.0, 0.0, 1.0, 0, 0, 0),
    ], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    faces = np.empty(2, dtype=[
        ("vertex_indices", "O"), ("object_id", "i4"),
    ])
    faces[0] = ([0, 1, 2], 0)
    faces[1] = ([3, 4, 5], 1)
    semantic_ply = root / "TARGET_semantic.ply"
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ], text=False).write(semantic_ply)
    house = root / "TARGET.house"
    house.write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "C  1  2 table  3 table  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
        "O  1 0 1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n")
    navmesh = root / "TARGET.navmesh"
    navmesh.write_bytes(b"navmesh")
    scene_config = root / "mp3d_annotated_basis.scene_dataset_config.json"
    scene_config.write_text("{}")
    manifest = root / "train.json"
    manifest.write_text('{"episodes":[{"scene_id":"TARGET.glb"}]}')
    spec = scene_pool.SceneSpec(
        scene_id="TARGET", source_dataset="r2r", official_split="train",
        scene_path=str(scene), navmesh_path=str(navmesh),
        semantic_path=str(semantic_ply),
        semantic_metadata_path=str(house), semantic_format="mp3d_ply",
        scene_dataset_config=str(scene_config),
        provenance_path=str(manifest), provenance_sha256="0" * 64,
    )
    return spec, semantic_ply


def _contact_rows(identity, *, instance_id=1, category="chair"):
    rows = []
    for perturbation in consensus.R2R_A_STABILITY_PERTURBATIONS:
        full_contact = {
            "world_point": [0.2, 0.2, 0.0],
            "full_geometry_attribution": {
                "instance_id": instance_id, "category": category,
                "unattributed": False,
            },
        }
        depth_contact = {
            "depth_mask_attribution": {
                "instance_id": instance_id, "category": category,
                "unattributed": False,
            },
        }
        rows.append({
            "perturbation_id": perturbation["id"],
            "transform": {key: perturbation[key]
                          for key in ("x_m", "z_m", "yaw_deg")},
            "physical": {
                "authority": "navmesh", "collision": True,
                "first_contact_arc_m": 0.5, "contact": full_contact,
            },
            "depth_physical": {
                "authority": "depth", "collision": True,
                "first_contact_arc_m": 0.5, "contact": depth_contact,
            },
            "corridor_coverage": 1.0,
            "exact_contact_identity": copy.deepcopy(identity),
        })
    return rows


def _r2r_binding(source_sha256: str):
    return dataset_contracts.AuthorityBinding(
        source_dataset="r2r",
        identity_schema="mp3d-contact-face-identity.v1",
        authoritative_source_role="semantic",
        source_sha256=source_sha256,
    )


def _source_bound_case(tmp_path, monkeypatch):
    spec, semantic_ply = _write_contact_scene(tmp_path)
    source = spec.provenance()
    semantic_sha = next(
        asset["sha256"] for asset in source["source_assets"]
        if asset["role"] == "semantic")
    house_sha = next(
        asset["sha256"] for asset in source["source_assets"]
        if asset["role"] == "semantic_metadata")
    authority = semantic.load_mp3d_target_authority(
        spec.scene_path, expected_semantic_ply_sha256=semantic_sha,
        expected_house_sha256=house_sha)
    identity = authority.confirm_contact_instance(1, [0.2, 0.2, 0.0])
    actions = [{"type": "forward", "m": 1.0}]
    rows = _contact_rows(identity)
    certificate = consensus.build_a_stability_certificate(
        actions, rows, authority_binding=record.authority_binding(source))
    nominal = certificate["rows"][0]
    rec = {
        "frame_id": "frame", "scene_id": "TARGET", "source": source,
        "collection_contract": record.r2r_v16_collection_contract(
            source, "main"),
        "objects": [
            {"instance_id": 1, "category": "chair",
             "centroid_px": [100, 100]},
            {"instance_id": 2, "category": "table",
             "centroid_px": [200, 100]},
        ],
        "outcomes": [{
            "outcome_id": "outcome", "actions": actions,
            "physical": copy.deepcopy(nominal["physical"]),
            "depth_physical": copy.deepcopy(nominal["depth_physical"]),
            "evidence": {"physical": {"coverage": 1.0}},
            "shared_oracle_stability": certificate,
        }],
    }
    monkeypatch.setattr(
        scene_pool, "_trusted_r2r_scene_for_context",
        lambda _rec, _context: spec)
    context = SimpleNamespace(route="r2r_v16_registered")
    return rec, context, semantic_ply, semantic_sha


def test_certificate_rejects_self_consistent_arbitrary_semantic_digest():
    arbitrary = "a" * 64
    identity = {
        "authority": "mp3d_full_face_universe",
        "schema": "mp3d-contact-face-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 1,
        "category": "chair",
        "streaming_instance_faces_sha256": "d" * 64,
        "contact_face_distance_m": 0.01,
        "runner_up_face_distance_m": 1.0,
        "face_distance_margin_m": 0.99,
        "global_query_protocol":
            "mp3d-complete-face-instance-universe.v1",
        "semantic_ply_sha256": arbitrary,
        "global_universe_sha256": semantic.complete_face_universe_sha256(
            global_query_protocol=
                "mp3d-complete-face-instance-universe.v1",
            semantic_ply_sha256=arbitrary),
        "global_winner_instance_id": 1,
        "global_runner_up_instance_id": 2,
    }

    certificate = consensus.build_a_stability_certificate(
        [{"type": "forward", "m": 1.0}], _contact_rows(identity),
        authority_binding=_r2r_binding("b" * 64))

    assert certificate["summary"]["contact_instance_stable"] is False


def test_source_bound_a3_rejects_substituted_semantic_ply(
        tmp_path, monkeypatch):
    rec, context, semantic_ply, _semantic_sha = _source_bound_case(
        tmp_path, monkeypatch)
    payload = bytearray(semantic_ply.read_bytes())
    payload[-1] ^= 1
    semantic_ply.write_bytes(payload)

    errors = scene_pool.trusted_r2r_a3_source_errors(rec, context)

    assert any("trusted R2R source unavailable" in error for error in errors)


def test_source_bound_rejects_category_eligible_a3_without_exact_proofs(
        tmp_path, monkeypatch):
    rec, context, _semantic_ply, semantic_sha = _source_bound_case(
        tmp_path, monkeypatch)
    outcome = rec["outcomes"][0]
    rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
    for row in rows:
        row["exact_contact_identity"] = None
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(
            outcome["actions"], rows,
            authority_binding=_r2r_binding(semantic_sha))
    assert outcome["shared_oracle_stability"]["summary"][
        "contact_instance_stable"] is False
    assert benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome).eligible is True

    errors = scene_pool.trusted_r2r_a3_source_errors(rec, context)

    assert any("trusted A3 contact identity disagrees" in error
               for error in errors)


def test_source_bound_allows_genuinely_a3_ineligible_collision_record(
        tmp_path, monkeypatch):
    rec, context, _semantic_ply, semantic_sha = _source_bound_case(
        tmp_path, monkeypatch)
    rec["objects"] = rec["objects"][:1]
    outcome = rec["outcomes"][0]
    rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
    for row in rows:
        row["exact_contact_identity"] = None
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(
            outcome["actions"], rows,
            authority_binding=_r2r_binding(semantic_sha))
    eligibility = benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome)

    errors = scene_pool.trusted_r2r_a3_source_errors(rec, context)

    assert eligibility.eligible is False
    assert eligibility.reason == "insufficient_visible_contact_categories"
    assert errors == []


def test_source_bound_record_validation_runs_a3_source_replay(monkeypatch):
    monkeypatch.setattr(
        validate, "validate_record_local", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        scene_pool, "trusted_r2r_source_binding_errors",
        lambda _rec, _context: [])
    monkeypatch.setattr(
        scene_pool, "trusted_r2r_a3_source_errors",
        lambda _rec, _context: ["A3 source replay marker"])
    monkeypatch.setattr(
        scene_pool, "trusted_r2r_b_source_errors",
        lambda _rec, _context: [])
    errors = validate.validate_record_source_bound(
        {"frame_id": "frame"},
        context=source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT)

    assert errors == ["A3 source replay marker"]


@pytest.mark.parametrize("tamper", ["instance", "category"])
def test_source_bound_a3_rejects_wrong_contact_identity(
        tmp_path, monkeypatch, tamper):
    rec, context, _semantic_ply, semantic_sha = _source_bound_case(
        tmp_path, monkeypatch)
    outcome = rec["outcomes"][0]
    rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
    if tamper == "instance":
        for row in rows:
            row["physical"]["contact"]["full_geometry_attribution"].update(
                {"instance_id": 2, "category": "table"})
            row["depth_physical"]["contact"]["depth_mask_attribution"].update(
                {"instance_id": 2, "category": "table"})
            row["exact_contact_identity"].update({
                "instance_id": 2, "category": "table",
                "global_winner_instance_id": 2,
                "global_runner_up_instance_id": 1,
            })
    else:
        for row in rows:
            row["exact_contact_identity"]["category"] = "table"
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(
            outcome["actions"], rows,
            authority_binding=_r2r_binding(semantic_sha))
    nominal = outcome["shared_oracle_stability"]["rows"][0]
    outcome["physical"] = copy.deepcopy(nominal["physical"])
    outcome["depth_physical"] = copy.deepcopy(nominal["depth_physical"])

    errors = scene_pool.trusted_r2r_a3_source_errors(rec, context)

    assert any("trusted A3 contact identity disagrees" in error
               for error in errors)
