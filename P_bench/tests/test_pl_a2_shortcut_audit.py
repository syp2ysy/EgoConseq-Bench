"""Scene-clustered A2 shortcut audits fail closed and are deterministic."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline import (
    a2_shortcut_audit, candidate_preview, consequence, consensus,
    gate_authority, perception, record,
)
from pipeline.actions import Forward, Turn
from pipeline.geometry import Disc
from tests._synthetic import (
    LEVEL_FLOOR_FIT, make_frame, source_provenance,
)
from tests.test_pl_rollout import HalfPlaneNav, _SurfaceIndex
from tests.test_pl_v16_candidate_preview import (
    _png, _refresh_private_manifest, _rewrite_jsonl,
)


def _a2_row(item_id: str, *, answer: str, scene_id: str):
    item = {
        "id": item_id,
        "task_id": "A2_collision_step_grounding",
        "model_input": {"actions": [
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": 30.0},
            {"type": "forward", "m": 1.0},
        ]},
        "choices": [
            {"id": "action_1", "text": "action_1"},
            {"id": "action_3", "text": "action_3"},
        ],
    }
    private = {
        "id": item_id,
        "task_id": "A2_collision_step_grounding",
        "canonical_answer": answer,
        "atom_ref": f"atom-{item_id}",
    }
    return item, private, scene_id


def _heterogeneous_rows():
    patterns = [
        [
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": 30.0},
            {"type": "forward", "m": 1.0},
        ],
        [
            {"type": "forward", "m": 1.0},
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": 30.0},
        ],
        [
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": 30.0},
            {"type": "turn", "deg": 30.0},
            {"type": "forward", "m": 1.0},
        ],
        [
            {"type": "forward", "m": 1.0},
            {"type": "forward", "m": 1.0},
            {"type": "turn", "deg": 30.0},
            {"type": "turn", "deg": 30.0},
        ],
    ]
    rows = []
    for scene_id, answers in (
            ("scene-a", ["action_1", "action_1", "action_1", "action_2"]),
            ("scene-b", ["action_1", "action_2", "action_4", "action_2"])):
        for index, (actions, answer) in enumerate(zip(patterns, answers)):
            item, private, _scene = _a2_row(
                f"{scene_id}-{index}", answer=answer, scene_id=scene_id)
            item["model_input"]["actions"] = actions
            item["choices"] = [
                {"id": f"action_{action_index}",
                 "text": f"action_{action_index}"}
                for action_index, action in enumerate(actions, 1)
                if action["type"] == "forward"
            ]
            rows.append((item, private, scene_id))
    return rows


def test_a2_audit_bootstraps_scene_cluster_max_statistic():
    rows = _heterogeneous_rows()
    report = a2_shortcut_audit.audit_rows(
        [row[0] for row in rows], [row[1] for row in rows],
        scene_by_id={row[0]["id"]: row[2] for row in rows},
        resamples=200, seed=19)

    assert report["item_count"] == 8
    assert report["scene_count"] == 2
    assert report["blind_predictors"] == {
        "constant_forward_ordinal": {"item_count": 8, "accuracy": 0.5},
        "answer_position": {"item_count": 8, "accuracy": 0.5},
        "action_count": {"item_count": 8, "accuracy": 0.75},
        "canonical_sequence_pattern": {
            "item_count": 8, "accuracy": 0.75},
    }
    real_accuracies = [
        value["accuracy"]
        for value in report["blind_predictors"].values()
    ]
    assert report["blind_predictors"][
        "canonical_sequence_pattern"]["accuracy"] == max(real_accuracies)
    # Hand-derived: the familywise point maximum is 6/8. Resampling either
    # scene twice makes the pattern lookup perfect, so the 95th percentile of
    # the scene-clustered max distribution is 1.0 for this fixed seed/sample.
    assert report["max_statistic_cluster_bootstrap"] == {
        "point_estimate": 0.75,
        "resamples": 200,
        "confidence_level": 0.95,
        "upper_confidence_bound": 1.0,
    }


def test_familywise_max_uses_each_resamples_winning_predictor():
    # Synthetic resample scores are necessary here: for the real in-sample
    # predictors, sequence pattern refines every coarser partition and thus
    # cannot lose. These literal vectors isolate the familywise reducer and
    # make averaging or selecting any one fixed coordinate fail.
    synthetic_resample_scores = [
        {
            "constant_forward_ordinal": 0.80,
            "answer_position": 0.20,
            "action_count": 0.40,
            "canonical_sequence_pattern": 0.60,
        },
        {
            "constant_forward_ordinal": 0.10,
            "answer_position": 0.70,
            "action_count": 0.30,
            "canonical_sequence_pattern": 0.50,
        },
        {
            "constant_forward_ordinal": 0.20,
            "answer_position": 0.40,
            "action_count": 0.90,
            "canonical_sequence_pattern": 0.60,
        },
        {
            "constant_forward_ordinal": 0.20,
            "answer_position": 0.30,
            "action_count": 0.40,
            "canonical_sequence_pattern": 0.65,
        },
    ]

    maxima = [
        a2_shortcut_audit._familywise_max(scores)
        for scores in synthetic_resample_scores
    ]

    assert maxima == [0.80, 0.70, 0.90, 0.65]


def test_a2_audit_rejects_a_turn_primitive_as_a_choice():
    item, private, scene_id = _a2_row(
        "bad", answer="action_3", scene_id="scene-a")
    item["choices"].insert(1, {"id": "action_2", "text": "action_2"})

    with pytest.raises(ValueError, match="Forward indexes"):
        a2_shortcut_audit.audit_rows(
            [item], [private], scene_by_id={item["id"]: scene_id})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_strict_a2_source(tmp_path: Path) -> tuple[dict, dict]:
    frame = make_frame()
    frame.objects[0]["mask_area_px"] = 30
    frame.semantic_index = _SurfaceIndex()
    actions = [Forward(0.25), Turn(30.0), Forward(1.5)]
    outcome = consequence.judge(
        frame, Disc(0.25), actions, nav=HalfPlaneNav(0.7), target_ids=[7])
    contact = outcome["physical"]["contact"]
    center_x, center_z = contact["center_local"]
    world_surface = perception.world_from_local(
        [[center_x, 0.0, center_z + 0.24]], frame.position,
        frame.yaw_rad)[0]
    world_boundary = perception.world_from_local(
        [[center_x, 0.0, center_z - 0.01]], frame.position,
        frame.yaw_rad)[0]
    contact.update({
        "world_point": world_surface.tolist(),
        "configuration_boundary_world_point": world_boundary.tolist(),
        "surface_protocol": "radius_extrapolated_navmesh_boundary_v2",
        "full_geometry_attribution": {
            "instance_id": 5, "category": "wall", "unattributed": False},
    })
    outcome["evidence"]["physical"].update({
        "coverage": 1.0, "status": "sufficient"})
    outcome["oracle_consensus"] = consequence.oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    exact_identity = {
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 5,
        "category": "wall",
        "instance_triangles_sha256": "d" * 64,
        "contact_face_distance_m": 0.01,
        "runner_up_face_distance_m": None,
        "face_distance_margin_m": None,
    }
    rows = [{
        "perturbation_id": perturbation["id"],
        "transform": {
            "x_m": perturbation["x_m"],
            "z_m": perturbation["z_m"],
            "yaw_deg": perturbation["yaw_deg"],
        },
        "physical": copy.deepcopy(outcome["physical"]),
        "depth_physical": copy.deepcopy(outcome["depth_physical"]),
        "corridor_coverage": 1.0,
        "exact_contact_identity": copy.deepcopy(exact_identity),
    } for perturbation in consensus.R2R_A_STABILITY_PERTURBATIONS]
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(outcome["actions"], rows)
    outcome["outcome_id"] = "a2-outcome"
    record_value = record.build_record(
        frame, [outcome], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    record_value["source"] = source_provenance(
        record_value["scene_id"], dataset="r2r")
    record_value["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    record_value["collection_contract"] = record.r2r_v16_collection_contract(
        record_value["source"], "main")
    image = tmp_path / record_value["image_path"]
    image.parent.mkdir(parents=True)
    _png(image, size=tuple(record_value["sensor"]["resolution"]))
    (tmp_path / "records.jsonl").write_text(
        json.dumps(record_value, allow_nan=False) + "\n")
    projection = candidate_preview.compile_main_records(
        [record_value], asset_root=tmp_path,
        build_root=tmp_path / "build")
    return projection, record_value


def _write_valid_artifact(tmp_path: Path):
    artifact = tmp_path / "candidate_qa"
    projection, record_value = _write_strict_a2_source(tmp_path)
    run_meta = tmp_path / "run_meta.json"
    run_meta.write_text(json.dumps({
        "record_schema_version": "conseq.v11",
        "oracle_contract_version": "ground-disc-visible-v8",
        "run_contract_sha256": "c" * 64,
        "params": {
            "backend": "r2r",
            "collection_mode": "main",
            "r2r_train_episodes": "/datasets/r2r/train.json",
            "mp3d_root": "/datasets/mp3d",
        },
        "resolved_scenes": [record_value["source"]],
    }))
    manifest = tmp_path / "authority.json"
    manifest.write_text(json.dumps({
        "schema": "egoconseq.golden-manifest.v1",
        "name": "a2-shortcut-test",
        "inputs": [{
            "shard": ".",
            "records.jsonl": _sha256(tmp_path / "records.jsonl"),
            "run_meta.json": _sha256(run_meta),
        }],
    }, sort_keys=True))
    authority = gate_authority.resolve_preview_source_authority_from_manifest(
        manifest, root=tmp_path)
    candidate_preview.write_preview_artifact(
        projection, artifact,
        source_records_path=tmp_path / "records.jsonl",
        source_run_meta_sha256=_sha256(run_meta),
        expected_source_authority=authority)
    return artifact, manifest, authority


def _a2_cli(
        artifact: Path, output: Path, manifest: Path,
        authority_root: Path) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / \
        "scripts" / "audit_a2_shortcuts.py"
    return subprocess.run([
        sys.executable, str(script), "--benchmark", str(artifact),
        "--out", str(output), "--resamples", "200", "--seed", "19",
        "--source-authority-manifest", str(manifest),
        "--source-authority-root", str(authority_root),
    ], capture_output=True, text=True)


def test_a2_artifact_audit_rejects_missing_candidate_manifest(tmp_path):
    artifact, _manifest, authority = _write_valid_artifact(tmp_path)
    (artifact / "private" / "manifest.json").unlink()

    with pytest.raises(FileNotFoundError):
        a2_shortcut_audit.audit_artifact(
            artifact, expected_source_authority=authority)


def test_a2_artifact_audit_requires_external_source_authority(tmp_path):
    artifact, _manifest, _authority = _write_valid_artifact(tmp_path)

    with pytest.raises(ValueError, match="external source authority required"):
        a2_shortcut_audit.audit_artifact(artifact)


def test_a2_shortcut_cli_rejects_rebound_source_atom(tmp_path):
    artifact, manifest, _authority = _write_valid_artifact(tmp_path)
    fake_digest = "f" * 64
    atoms_path = artifact / "private" / "atoms.jsonl"
    atoms = [json.loads(line) for line in atoms_path.read_text().splitlines()]
    atoms[0]["record_sha256"] = fake_digest
    _rewrite_jsonl(atoms_path, atoms)
    contexts_path = artifact / "private" / "record_contexts.jsonl"
    contexts = [
        json.loads(line) for line in contexts_path.read_text().splitlines()]
    contexts[0]["record_sha256"] = fake_digest
    _rewrite_jsonl(contexts_path, contexts)
    _refresh_private_manifest(artifact)

    result = _a2_cli(
        artifact, tmp_path / "audit.json", manifest, tmp_path)

    assert result.returncode != 0
    assert "source atom" in result.stderr


def test_a2_shortcut_cli_is_deterministic(tmp_path):
    artifact, manifest, _authority = _write_valid_artifact(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    for output in (first, second):
        result = _a2_cli(artifact, output, manifest, tmp_path)
        assert result.returncode == 0, result.stderr
    assert json.loads(first.read_text()) == json.loads(second.read_text())
