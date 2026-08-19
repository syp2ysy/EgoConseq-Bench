"""Record round-trip + numpy->python conversion."""

import dataclasses
import copy
import json

import numpy as np
import pytest

from pipeline import (
    consensus, consequence, record, semantic,
)
from pipeline.geometry import Disc
from pipeline.actions import Forward, Turn
from tests._synthetic import (
    LEVEL_FLOOR_FIT, _outcome, make_frame, source_provenance,
)


def _jsonable_recursive(o):
    """Assert no numpy scalars survive serialization."""
    if isinstance(o, dict):
        return all(_jsonable_recursive(v) for v in o.values())
    if isinstance(o, list):
        return all(_jsonable_recursive(v) for v in o)
    return not isinstance(o, (np.generic, np.ndarray))


def test_build_record_roundtrip(tmp_path):
    fr = make_frame()
    ocs = []
    for i, acts in enumerate([[Forward(1.0)], [Turn(90)]]):
        oc = consequence.judge(fr, Disc(0.25), acts, nav=None)
        oc["outcome_id"] = f"o{i}"
        ocs.append(oc)
    rec = record.build_record(fr, ocs, image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    assert rec["schema_version"] == "conseq.v11"
    assert rec["oracle_contract_version"] == "ground-disc-visible-v8"
    assert [
        value["execution"]["execution_regime"]
        for value in rec["outcomes"]
    ] == ["invalid_geometry", "invalid_geometry"]
    assert "semantic_path" not in rec
    assert {
        "review_evidence", "target_reference_sets", "targets",
    }.isdisjoint(rec)
    for outcome in rec["outcomes"]:
        assert {
            "future_state", "terminal_options", "start_terminal_options",
            "object_consequences", "target_projection_keys",
            "target_projections", "probe",
        }.isdisjoint(outcome)
        assert {
            "goal", "terminal_options",
        }.isdisjoint(outcome["evidence"])
    assert rec["sampling_context"] == {
        "clearance": "open", "visible_floor": "medium",
        "object_density": "sparse"}

    assert _jsonable_recursive(rec)                       # numpy fully converted
    assert "_points_xz" not in rec["objects"][0]          # transient stripped
    assert "dist_geodesic_m" not in rec["objects"][0]

    path = tmp_path / "r.jsonl"
    path.write_text("".join(
        json.dumps(record.json_value(rec), allow_nan=False) + "\n"
        for _ in range(2)
    ))
    loaded = list(record.read_records(str(path)))
    assert len(loaded) == 2
    assert loaded[0]["frame_id"] == fr.frame_id
    assert loaded[0]["n_objects"] == len(rec["objects"])
    assert loaded[0]["outcomes"][0]["outcome_id"] == "o0"


def test_decode_records_accepts_bytes_and_text_with_identical_contract_checks():
    value = {
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "frame_id": "snapshot",
    }
    payload = json.dumps(value) + "\n"

    assert record.decode_records(payload) == [value]
    assert record.decode_records(payload.encode("utf-8")) == [value]


def test_read_records_delegates_one_byte_snapshot_to_decoder(
        tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b"opaque snapshot")
    received = []

    def fake_decode(payload):
        received.append(payload)
        return [{"frame_id": "decoded"}]

    monkeypatch.setattr(record, "decode_records", fake_decode)

    assert list(record.read_records(path)) == [{"frame_id": "decoded"}]
    assert received == [b"opaque snapshot"]


def test_build_record_preserves_verified_train_scene_provenance():
    provenance = {
        "scene_id": "00001-AAA",
        "source_dataset": "r2r",
        "official_split": "train",
        "semantic_format": "mp3d_ply",
        "source_manifest": "/datasets/r2r/train.json.gz",
        "source_manifest_sha256": "a" * 64,
    }

    rec = record.build_record(
        make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=provenance)

    assert rec["source"] == provenance


class _ExactContactIndex:
    id_to_cat = {7: "chair"}

    def __init__(self):
        self.confirm_calls = 0

    def confirm_contact_instance(
            self, instance_id, world_point, *, candidate_instance_ids):
        self.confirm_calls += 1
        assert instance_id == 7
        assert world_point == [0.0, 0.15, 1.0]
        assert candidate_instance_ids == [7]
        semantic_sha = next(
            asset["sha256"] for asset in
            source_provenance("synthetic", dataset="r2r")["source_assets"]
            if asset["role"] == "semantic")
        return {
            "authority": "mp3d_full_face_universe",
            "schema": "mp3d-contact-face-identity.v1",
            "confirmed": True,
            "reason": "confirmed",
            "instance_id": 7,
            "category": "chair",
            "streaming_instance_faces_sha256": "d" * 64,
            "contact_face_distance_m": 0.01,
            "runner_up_face_distance_m": 0.5,
            "face_distance_margin_m": 0.49,
            "global_query_protocol":
                "mp3d-complete-face-instance-universe.v1",
            "global_winner_instance_id": 7,
            "global_runner_up_instance_id": 9,
            "semantic_ply_sha256": semantic_sha,
            "global_universe_sha256": semantic.complete_face_universe_sha256(
                global_query_protocol=
                    "mp3d-complete-face-instance-universe.v1",
                semantic_ply_sha256=semantic_sha),
        }


def _strict_collision_record_with_index(index):
    frame = dataclasses.replace(make_frame(), semantic_index=index)
    outcome = _outcome(collision=True, progress=0.4 / 1.5)
    outcome["physical"]["contact"].update({
        "world_point": [0.0, 0.15, 1.0],
        "full_geometry_attribution": {
            "instance_id": 7, "category": "chair", "unattributed": False,
        },
        "depth_mask_attribution": {
            "instance_id": 7, "category": "chair", "unattributed": False,
        },
    })
    outcome["depth_physical"]["contact"] = {
        "depth_mask_attribution": {
            "instance_id": 7, "category": "chair", "unattributed": False,
        },
    }
    outcome["oracle_consensus"] = consensus.oracle_consensus(
        outcome["physical"], outcome["depth_physical"], 1.0,
        require_contact_instance_witness=True)
    exact = index.confirm_contact_instance(
        7, [0.0, 0.15, 1.0], candidate_instance_ids=[7])
    rows = []
    for perturbation in consensus.R2R_A_STABILITY_PERTURBATIONS:
        rows.append({
            "perturbation_id": perturbation["id"],
            "transform": {key: perturbation[key]
                          for key in ("x_m", "z_m", "yaw_deg")},
            "physical": copy.deepcopy(outcome["physical"]),
            "depth_physical": copy.deepcopy(outcome["depth_physical"]),
            "corridor_coverage": 1.0,
            "exact_contact_identity": copy.deepcopy(exact),
        })
    provenance = source_provenance("synthetic", dataset="r2r")
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(
            outcome["actions"], rows,
            authority_binding=record.authority_binding(provenance))
    return record.build_record(
        frame, [outcome], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=provenance,
        collection_contract=record.r2r_v16_collection_contract(
            provenance, "main"))


def test_build_record_persists_full_face_confirmed_contact_identity():
    """Catches A3 reading PLY later or accepting only a sampled witness hash."""
    index = _ExactContactIndex()
    rec = _strict_collision_record_with_index(index)

    semantic_sha = next(
        asset["sha256"] for asset in
        rec["source"]["source_assets"] if asset["role"] == "semantic")
    assert rec["outcomes"][0]["contact_instance_identity"] == {
        "authority": "mp3d_full_face_universe",
        "schema": "mp3d-contact-face-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 7,
        "category": "chair",
        "streaming_instance_faces_sha256": "d" * 64,
        "contact_face_distance_m": 0.01,
        "runner_up_face_distance_m": 0.5,
        "face_distance_margin_m": 0.49,
        "global_query_protocol":
            "mp3d-complete-face-instance-universe.v1",
        "global_winner_instance_id": 7,
        "global_runner_up_instance_id": 9,
        "semantic_ply_sha256": semantic_sha,
        "global_universe_sha256": semantic.complete_face_universe_sha256(
            global_query_protocol=
                "mp3d-complete-face-instance-universe.v1",
            semantic_ply_sha256=semantic_sha),
    }
    assert index.confirm_calls == 1


def test_base_rollout_key_does_not_depend_on_target_identity():
    frame = make_frame()
    outcome = consequence.judge(
        frame, Disc(0.25), [Forward(1.0)], nav=None)

    key = record.base_rollout_key(frame, outcome)

    assert key == record.base_rollout_key(frame, {
        **outcome,
        "target_instance_id": 999,
        "private_note": "not-part-of-the-base-key",
    })


def test_record_json_value_converts_nonfinite_numbers_to_null(tmp_path):
    payload = {
        "finite": np.float32(1.25),
        "nan": float("nan"),
        "positive_infinity": np.float64(float("inf")),
        "nested": [float("-inf")],
    }

    normalized = record.json_value(payload)

    assert normalized == {
        "finite": 1.25,
        "nan": None,
        "positive_infinity": None,
        "nested": [None],
    }
    path = tmp_path / "strict.jsonl"
    path.write_text(
        json.dumps(record.json_value(payload), allow_nan=False) + "\n")
    raw = path.read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == normalized


def test_read_records_rejects_legacy_outcome(tmp_path):
    legacy = {
        "schema_version": "conseq.v1",
        "frame_id": "old",
        "scene_id": "scene",
        "scene_glb": "/tmp/scene.glb",
        "pose": {"position": [0, 0, 0], "yaw_rad": 0},
        "sensor": {"nominal_camera_offset_m": 1.5, "hfov_deg": 79,
                   "vfov_deg": 63.45, "resolution": [640, 480]},
        "objects": [],
        "outcomes": [{
            "outcome_id": "old-o",
            "body": {"shape": "cylinder", "radius_m": 0.2, "height_m": 1.5},
            "actions": [{"type": "forward", "m": 1.0}],
            "total_forward_m": 1.0,
            "collided": False,
            "first_contact_arc_m": None,
            "contact": None,
            "pose_end_full": {"x": 0, "z": 1, "heading_deg": 0},
            "pose_end_exec": {"x": 0, "z": 1, "heading_deg": 0},
            "object_relations": [],
            "view_exit": {},
            "visibility": {},
            "nav_check": None,
        }],
    }
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n")

    with pytest.raises(ValueError, match="recollect"):
        list(record.read_records(path))


def test_read_records_rejects_v5_schema_and_oracle_contract(tmp_path):
    # P0-16: v5 becomes legacy. The reader is the first of three independent
    # trust boundaries; a v5 pool must not be readable at all, because P0-4/P0-5/
    # P0-13 changed what the persisted fields mean.
    schema_stale = record.build_record(
        make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    schema_stale["schema_version"] = "conseq.v5"
    path = tmp_path / "v5-schema.jsonl"
    path.write_text(json.dumps(schema_stale) + "\n")
    with pytest.raises(ValueError, match="recollect"):
        list(record.read_records(path))

    oracle_stale = record.build_record(
        make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    oracle_stale["oracle_contract_version"] = "ground-disc-visible-v3"
    path = tmp_path / "v5-oracle.jsonl"
    path.write_text(json.dumps(oracle_stale) + "\n")
    with pytest.raises(ValueError, match="oracle contract"):
        list(record.read_records(path))


def test_read_records_rejects_pre_tilt_relative_contact_attribution(tmp_path):
    stale = record.build_record(
        make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    stale["oracle_contract_version"] = "ground-disc-visible-v5"
    path = tmp_path / "oracle-v5.jsonl"
    path.write_text(json.dumps(stale) + "\n")

    with pytest.raises(
            ValueError, match="recollect as ground-disc-visible-v8"):
        list(record.read_records(path))


@pytest.mark.parametrize(
    ("field", "stale_value", "current_value"),
    [
        ("schema_version", "conseq.v10", "conseq.v11"),
        (
            "oracle_contract_version",
            "ground-disc-visible-v6",
            "ground-disc-visible-v8",
        ),
    ],
)
def test_read_records_rejects_immediately_preceding_gt_contract(
        tmp_path, field, stale_value, current_value):
    stale = record.build_record(
        make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    stale[field] = stale_value
    path = tmp_path / f"stale-{field}.jsonl"
    path.write_text(json.dumps(stale) + "\n")

    with pytest.raises(ValueError, match=f"recollect as {current_value}"):
        list(record.read_records(path))


def test_read_records_rejects_pre_v131a_v6_contract(tmp_path):
    stale = record.build_record(
        make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    stale["schema_version"] = "conseq.v6"
    stale["oracle_contract_version"] = "ground-disc-visible-v4"
    path = tmp_path / "v6.jsonl"
    path.write_text(json.dumps(stale) + "\n")

    with pytest.raises(ValueError, match="recollect as conseq.v11"):
        list(record.read_records(path))


def test_read_records_rejects_v2_consensus_fields(tmp_path):
    current = record.build_record(make_frame(), [], image_path="img/x.png",
        floor_calibration=LEVEL_FLOOR_FIT)
    current["schema_version"] = "conseq.v2"
    current["outcomes"] = [{
        "body": {"radius_m": 0.2}, "actions": [],
        "execution": {"completed": True},
        "physical": {"authority": "navmesh", "collision": False},
    }]
    path = tmp_path / "v2.jsonl"
    path.write_text(json.dumps(current) + "\n")
    with pytest.raises(ValueError, match="conseq.v2"):
        list(record.read_records(path))
