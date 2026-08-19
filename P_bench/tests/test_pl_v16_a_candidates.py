"""Frozen A1/A2/A3 candidate projections over one shared certificate."""

import copy
import hashlib

import pytest
from PIL import Image

from pipeline import (
    benchmark, benchmark_builders, benchmark_tasks, consensus, record,
)
from tests._synthetic import source_provenance


QUESTIONS = {
    "A1_collision":
        "Will the robot collide with anything while executing all the actions?",
    "A2_collision_step_grounding":
        "The robot will collide while executing the actions. During which "
        "action does the collision occur?",
    "A3_contact_object":
        "The robot will collide while executing the actions. What category "
        "of visible object or surface will it contact first?",
}
_SOURCE = source_provenance("scene", dataset="r2r")


def _row(perturbation, *, collision, arc=None, instance_id=None):
    full_contact = depth_contact = None
    if collision:
        full_contact = {
            "full_geometry_attribution": {
                "instance_id": instance_id, "category": "chair",
                "unattributed": False,
            },
        }
        depth_contact = {
            "depth_mask_attribution": {
                "instance_id": instance_id, "category": "chair",
                "unattributed": False,
            },
        }
    row = {
        "perturbation_id": perturbation["id"],
        "transform": {
            "x_m": perturbation["x_m"], "z_m": perturbation["z_m"],
            "yaw_deg": perturbation["yaw_deg"],
        },
        "physical": {
            "authority": "navmesh", "collision": collision,
            "first_contact_arc_m": arc, "contact": full_contact,
        },
        "depth_physical": {
            "authority": "depth", "collision": collision,
            "first_contact_arc_m": arc, "contact": depth_contact,
        },
        "corridor_coverage": 1.0,
    }
    return row


def _case(*, collision=True, arc=1.5, instance_id=7, duplicate=False):
    actions = [
        {"type": "forward", "m": 1.0},
        {"type": "turn", "deg": 30.0},
        {"type": "forward", "m": 1.0},
    ]
    rows = [
        _row(value, collision=collision,
             arc=arc if collision else None,
             instance_id=instance_id if collision else None)
        for value in consensus.R2R_A_STABILITY_PERTURBATIONS
    ]
    rows[0]["physical"]["contact_action_index"] = 999
    rows[0]["physical"]["simulator_substep"] = 12345
    certificate = consensus.build_a_stability_certificate(actions, rows)
    physical = copy.deepcopy(rows[0]["physical"])
    outcome = {
        "outcome_id": "o-a", "base_rollout_key": "b" * 64,
        "body": {"shape": "disc", "radius_m": 0.2},
        "actions": actions,
        "physical": physical,
        "depth_physical": copy.deepcopy(rows[0]["depth_physical"]),
        "evidence": {"physical": {"coverage": 1.0}},
        "oracle_consensus": copy.deepcopy(rows[0].get("consensus") or
                                            certificate["rows"][0]["consensus"]),
        "shared_oracle_stability": certificate,
    }
    objects = [{
        "instance_id": 7, "category": "chair", "centroid_px": [70, 25],
    }, {
        "instance_id": 11, "category": "table", "centroid_px": [120, 40],
    }]
    if duplicate:
        objects.append({
            "instance_id": 9, "category": "chair", "centroid_px": [90, 20],
        })
    source = copy.deepcopy(_SOURCE)
    rec = {
        "frame_id": "frame", "observation_id": "obs", "objects": objects,
        "source": source,
        "collection_contract": record.r2r_v16_collection_contract(
            source, "main"),
        "sensor": {
            "hfov_deg": 79.0, "vfov_deg": 63.45,
            "resolution": [640, 480],
        },
        "camera_height_above_visible_floor_m": 1.0,
        "image_path": "relocated/by/digest.png",
    }
    return rec, outcome


def test_a_contract_has_only_frozen_task_ids_and_questions():
    """Catches a Q alias, paraphrased template, or diagnostic A task."""
    assert benchmark.A_TASK_QUESTIONS == QUESTIONS


@pytest.mark.parametrize("collision", [True, False])
def test_a1_rejects_certificate_that_disagrees_with_nominal_outcome(collision):
    rec, outcome = _case(collision=collision)
    outcome["physical"]["collision"] = not collision

    eligibility = benchmark_tasks.a_candidate_eligibility(
        "A1_collision", rec, outcome)

    assert eligibility.eligible is False
    assert eligibility.reason == "shared_oracle_stability_invalid"
    with pytest.raises(ValueError, match="shared_oracle_stability_invalid"):
        benchmark_tasks.a_candidate_answer("A1_collision", rec, outcome)


def test_a1_withholds_without_one_complete_stability_certificate():
    rec, outcome = _case(collision=False)
    outcome.pop("shared_oracle_stability")

    result = benchmark_tasks.a_candidate_eligibility(
        "A1_collision", rec, outcome)

    assert result.eligible is False
    assert result.reason == "shared_oracle_stability_missing"


@pytest.mark.parametrize(
    "tamper", ["shallow", "contract_source_hash", "source_hash"])
def test_a_route_rejects_shallow_or_tampered_collection_contract(tamper):
    """Task6 adds external registry validation; projection still rejects shallow data."""
    rec, outcome = _case(collision=False)
    if tamper == "shallow":
        rec["collection_contract"] = {
            "version": record.R2R_V16_COLLECTION_CONTRACT_VERSION,
            "collection_mode": "main",
        }
    elif tamper == "contract_source_hash":
        rec["collection_contract"]["source_manifest_sha256"] = "0" * 64
    else:
        rec["source"]["source_manifest_sha256"] = "0" * 64

    result = benchmark_tasks.a_candidate_eligibility(
        "A1_collision", rec, outcome)

    assert result.eligible is False
    assert result.reason == "strict_r2r_main_contract_required"


@pytest.mark.parametrize("missing", ["source_manifest_sha256", "source_assets_sha256"])
def test_a_route_rejects_missing_required_source_hash(missing):
    rec, outcome = _case(collision=False)
    rec["source"].pop(missing)
    rec["collection_contract"] = record.r2r_v16_collection_contract(
        rec["source"], "main")

    result = benchmark_tasks.a_candidate_eligibility(
        "A1_collision", rec, outcome)

    assert result.eligible is False
    assert result.reason == "strict_r2r_main_contract_required"


def test_a3_requires_a_visible_category_agreed_by_both_oracles():
    rec, outcome = _case(collision=True, instance_id=7)

    eligibility = benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome)
    answer, choices = benchmark_tasks.a_candidate_answer(
        "A3_contact_object", rec, outcome)

    assert eligibility.eligible is True
    assert answer == "chair"
    assert choices == [
        {"id": "chair", "text": "chair"},
        {"id": "table", "text": "table"},
    ]


def test_a_builder_reuses_one_bound_eligibility(tmp_path, monkeypatch):
    rec, outcome = _case(collision=True, instance_id=7)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (640, 480), (1, 2, 3)).save(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    evidence = benchmark_tasks.build_a_candidate_evidence(
        "A3_contact_object", rec, outcome)

    def repeated_validation(*_args, **_kwargs):
        raise AssertionError("A eligibility was evaluated more than once")

    monkeypatch.setattr(
        benchmark_tasks, "a_candidate_eligibility", repeated_validation)
    item, private = benchmark_builders.build_a_candidate(
        task_id="A3_contact_object", record=rec, outcome=outcome,
        image=str(raw), expected_raw_image_sha256=digest,
        a_eligibility_evidence=evidence)

    assert item["task_id"] == "A3_contact_object"
    assert private["canonical_answer"] == "chair"


def test_a_evidence_reuses_the_shared_outcome_certificate(monkeypatch):
    rec, outcome = _case(collision=True, instance_id=7)
    shared = benchmark.build_shared_visible_space_evidence(rec, outcome)

    def repeated_validation(*_args, **_kwargs):
        raise AssertionError("shared certificate was authenticated twice")

    monkeypatch.setattr(
        benchmark_tasks, "_a_shared_certificate", repeated_validation)
    evidence = benchmark_tasks.build_a_candidate_evidence(
        "A3_contact_object", rec, outcome, shared_evidence=shared)

    assert evidence.eligibility.eligible is True


def test_source_validated_shared_evidence_does_not_replay_the_oracle(
        monkeypatch):
    rec, outcome = _case(collision=True, instance_id=7)

    def repeated_validation(*_args, **_kwargs):
        raise AssertionError("source-validated oracle was replayed")

    monkeypatch.setattr(
        benchmark, "shared_visible_space_certificate", repeated_validation)
    shared = benchmark.build_shared_visible_space_evidence(
        rec, outcome, source_validated=True)

    assert shared.certificate == outcome["shared_oracle_stability"]
    assert shared.reason == "eligible"


def test_a3_rejects_a_category_change_across_the_frozen_perturbations():
    rec, outcome = _case(collision=True, instance_id=7)
    rec["objects"].append({
        "instance_id": 13, "category": "sofa", "centroid_px": [160, 60],
    })
    rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
    row = rows[-1]
    row["physical"]["contact"]["full_geometry_attribution"].update(
        {"instance_id": 13, "category": "sofa"})
    row["depth_physical"]["contact"]["depth_mask_attribution"].update(
        {"instance_id": 13, "category": "sofa"})
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(outcome["actions"], rows)

    result = benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome)

    assert result.eligible is False
    assert result.reason == "contact_category_unstable"


def test_a3_rejects_full_depth_instance_disagreement():
    rec, outcome = _case(collision=True, instance_id=7)
    rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
    rows[-1]["depth_physical"]["contact"][
        "depth_mask_attribution"]["instance_id"] = 11
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(outcome["actions"], rows)

    result = benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome)

    assert result.eligible is False
    # The perturbed row's consensus rejected, so no second label was ever
    # produced -- naming this a disagreement would misreport the cause.
    assert result.reason == "stability_evidence_incomplete"


def test_a2_maps_contact_arc_to_original_public_action_not_substep():
    rec, outcome = _case(collision=True, arc=1.5)

    eligibility = benchmark_tasks.a_candidate_eligibility(
        "A2_collision_step_grounding", rec, outcome)
    answer, choices = benchmark_tasks.a_candidate_answer(
        "A2_collision_step_grounding", rec, outcome)

    assert eligibility.eligible is True
    assert answer == "action_3"
    assert choices == ["action_1", "action_3"]


def test_a2_forward_choices_keep_original_indexes_when_sequence_starts_turn():
    """Catches renumbering the filtered Forward subsequence as 1, 2."""
    rec, outcome = _case(collision=True, arc=1.5)
    actions = [
        {"type": "turn", "deg": -30.0},
        {"type": "forward", "m": 1.0},
        {"type": "turn", "deg": 30.0},
        {"type": "forward", "m": 1.0},
    ]
    rows = copy.deepcopy(outcome["shared_oracle_stability"]["rows"])
    outcome["actions"] = actions
    outcome["shared_oracle_stability"] = \
        consensus.build_a_stability_certificate(actions, rows)
    outcome["physical"] = copy.deepcopy(rows[0]["physical"])
    outcome["depth_physical"] = copy.deepcopy(rows[0]["depth_physical"])
    outcome["oracle_consensus"] = copy.deepcopy(
        outcome["shared_oracle_stability"]["rows"][0]["consensus"])

    answer, choices = benchmark_tasks.a_candidate_answer(
        "A2_collision_step_grounding", rec, outcome)

    assert answer == "action_4"
    assert choices == ["action_2", "action_4"]


def test_a2_withholds_contact_near_an_original_action_boundary():
    rec, outcome = _case(collision=True, arc=1.0)

    result = benchmark_tasks.a_candidate_eligibility(
        "A2_collision_step_grounding", rec, outcome)

    assert result.eligible is False
    assert result.reason == "collision_action_index_unstable"


def test_a3_collapses_duplicate_visible_instances_to_one_category_choice():
    rec, outcome = _case(collision=True, instance_id=7, duplicate=True)

    eligibility = benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome)
    answer, choices = benchmark_tasks.a_candidate_answer(
        "A3_contact_object", rec, outcome)

    assert eligibility.eligible is True
    assert answer == "chair"
    assert choices == [
        {"id": "chair", "text": "chair"},
        {"id": "table", "text": "table"},
    ]


def test_a3_rejects_contact_category_absent_from_initial_visible_inventory():
    rec, outcome = _case(collision=True, instance_id=7)
    rec["objects"] = [
        value for value in rec["objects"]
        if value["instance_id"] != 7
    ]

    result = benchmark_tasks.a_candidate_eligibility(
        "A3_contact_object", rec, outcome)

    assert result.eligible is False
    assert result.reason == "contact_instance_not_initially_visible"


def test_closed_exact_a_builder_uses_one_certificate_and_no_legacy_fields(
        tmp_path):
    rec, outcome = _case(collision=True, instance_id=7, duplicate=True)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (640, 480), (20, 30, 40)).save(raw)
    original_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    built = {
        task_id: benchmark_builders.build_a_candidate(
            task_id=task_id, record=rec, outcome=outcome,
            image=str(raw), asset_dir=tmp_path / "assets",
            expected_raw_image_sha256=original_hash)
        for task_id in QUESTIONS
    }

    assert {value[0]["question"] for value in built.values()} == \
        set(QUESTIONS.values())
    assert {value[0]["oracle_ref"]["shared_certificate_sha256"]
            for value in built.values()} == {
                outcome["shared_oracle_stability"]["sha256"]}
    assert built["A1_collision"][1]["canonical_answer"] == "collision"
    assert built["A2_collision_step_grounding"][1][
        "canonical_answer"] == "action_3"
    assert built["A3_contact_object"][1]["canonical_answer"] == "chair"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == original_hash
    assert built["A3_contact_object"][0]["model_input"][
        "initial_rgb_sha256"] == original_hash
    assert built["A3_contact_object"][0]["model_input"][
        "initial_rgb"] == str(raw)
    for item, private in built.values():
        assert item["task_id"] in QUESTIONS
        assert item["answer_format"] == "closed_exact"
        assert "canonical_answer" not in item
        assert not ({"family", "variant", "diagnostic", "response_schema"} &
                    set(item))
        assert set(private) == {
            "id", "task_id", "canonical_answer", "oracle_ref",
            "contact_category", "input_asset",
        }


def test_a3_category_uses_raw_image_without_marker_asset(tmp_path):
    rec, outcome = _case(collision=True, instance_id=7, duplicate=False)
    raw = tmp_path / "raw.png"
    Image.new("RGB", (640, 480), (1, 2, 3)).save(raw)
    item, private = benchmark_builders.build_a_candidate(
        task_id="A3_contact_object", record=rec, outcome=outcome,
        image=str(raw), asset_dir=tmp_path / "assets",
        expected_raw_image_sha256=hashlib.sha256(
            raw.read_bytes()).hexdigest())

    assert item["model_input"]["initial_rgb"] == str(raw)
    assert not (tmp_path / "assets").exists()
    assert private["input_asset"]["marked"] is False


def test_a_builder_rejects_wrong_resolution_mode_or_digest(tmp_path):
    rec, outcome = _case(collision=False)
    rec["sensor"]["resolution"] = [640, 480]
    wrong_size = tmp_path / "wrong-size.png"
    Image.new("RGB", (64, 64)).save(wrong_size)
    gray = tmp_path / "gray.png"
    Image.new("L", (640, 480)).save(gray)

    with pytest.raises(ValueError, match="resolution"):
        benchmark_builders.build_a_candidate(
            task_id="A1_collision", record=rec, outcome=outcome,
            image=str(wrong_size),
            expected_raw_image_sha256=hashlib.sha256(
                wrong_size.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="RGB"):
        benchmark_builders.build_a_candidate(
            task_id="A1_collision", record=rec, outcome=outcome,
            image=str(gray), expected_raw_image_sha256=hashlib.sha256(
                gray.read_bytes()).hexdigest())
    raw = tmp_path / "raw.png"
    Image.new("RGB", (640, 480)).save(raw)
    with pytest.raises(ValueError, match="digest"):
        benchmark_builders.build_a_candidate(
            task_id="A1_collision", record=rec, outcome=outcome,
            image=str(raw), expected_raw_image_sha256="0" * 64)


@pytest.mark.parametrize("bad_source", [[], "not-a-source", 3])
def test_a_eligibility_fails_closed_for_non_dict_source(bad_source):
    rec, outcome = _case(collision=False)
    rec["source"] = bad_source

    result = benchmark_tasks.a_candidate_eligibility(
        "A1_collision", rec, outcome)

    assert result.eligible is False
    assert result.reason == "strict_r2r_main_contract_required"


@pytest.mark.parametrize(
    "bad_sensor", [None, [], {"resolution": None},
                   {"resolution": [640]}, {"resolution": [0, 480]}])
def test_a_builder_rejects_malformed_sensor_resolution(tmp_path, bad_sensor):
    rec, outcome = _case(collision=False)
    rec["sensor"] = bad_sensor
    raw = tmp_path / "raw.png"
    Image.new("RGB", (640, 480)).save(raw)

    with pytest.raises(ValueError, match="sensor resolution"):
        benchmark_builders.build_a_candidate(
            task_id="A1_collision", record=rec, outcome=outcome,
            image=str(raw), expected_raw_image_sha256=hashlib.sha256(
                raw.read_bytes()).hexdigest())


def test_stability_shortfall_separates_missing_evidence_from_disagreement():
    """Catches a thin-evidence row being reported as a label that flipped."""
    accepted = {"consensus": {"accepted": True}}
    rejected = {"consensus": {"accepted": False,
                              "reason": "insufficient_depth_coverage"}}

    assert benchmark._stability_shortfall_reason(
        {"rows": [accepted, rejected]}) == "stability_evidence_incomplete"
    # Every row produced a label, so the only way the summary can be unstable
    # is that the labels themselves disagreed.
    assert benchmark._stability_shortfall_reason(
        {"rows": [accepted, accepted]}) == "stability_label_disagreement"
