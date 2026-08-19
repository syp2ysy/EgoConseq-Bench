"""A1 blind and B2 rotation-only baselines are measured, never enforced."""

from __future__ import annotations

import json

import pytest

from pipeline import action_proposal, blind_baseline_audit


_A1_CHOICES = [
    {"id": "collision", "text": "collision"},
    {"id": "no_collision", "text": "no_collision"},
]
_B2_CHOICES = [
    {"id": value, "text": value}
    for value in ("front", "left", "right", "rear")
]


def _a1_row(item_id: str, *, answer: str, actions: list, radius: float = 0.15):
    item = {
        "id": item_id,
        "task_id": "A1_collision",
        "choices": list(_A1_CHOICES),
        "model_input": {"actions": actions, "body_radius_m": radius},
    }
    answer_row = {
        "id": item_id,
        "task_id": "A1_collision",
        "canonical_answer": answer,
    }
    return item, answer_row


def _b2_row(item_id: str, *, answer: str, actions: list, target: str = "chair",
            bearing_deg: float | None = None):
    item = {
        "id": item_id,
        "task_id": "B2_endpoint_direction",
        "choices": list(_B2_CHOICES),
        "model_input": {
            "actions": actions,
            "body_radius_m": 0.15,
            "target": target,
        },
    }
    answer_row = {
        "id": item_id,
        "task_id": "B2_endpoint_direction",
        "canonical_answer": answer,
        "precise_bearing_deg": bearing_deg,
    }
    return item, answer_row


def _split(rows):
    return [row[0] for row in rows], [row[1] for row in rows]


def test_a1_constant_answer_baseline_is_reported_in_and_out_of_sample():
    forward = [{"type": "forward", "m": 0.5}]
    rows = [
        _a1_row("a1", answer="no_collision", actions=forward),
        _a1_row("a2", answer="no_collision", actions=forward),
        _a1_row("a3", answer="no_collision", actions=forward),
        _a1_row("a4", answer="collision", actions=forward),
    ]
    report = blind_baseline_audit.audit_a1_rows(*_split(rows))

    assert report["schema"] == blind_baseline_audit.A1_BLIND_AUDIT_SCHEMA
    assert report["item_count"] == 4
    assert report["canonical_answer_counts"] == {"collision": 1,
                                                 "no_collision": 3}
    baselines = report["blind_baselines"]
    assert baselines["nominal_uniform_random_accuracy"] == 0.5
    constant = baselines["predictors"]["constant"]
    assert constant["group_count"] == 1
    assert constant["in_sample_accuracy"] == 0.75
    # Held out, the three majority rows still see a majority; the minority row
    # cannot predict itself, so the honest number matches here by coincidence.
    assert constant["leave_one_out_accuracy"] == 0.75


def test_a1_in_sample_inflation_is_visible_through_group_count():
    rows = [
        _a1_row("a1", answer="collision",
                actions=[{"type": "forward", "m": 0.5}]),
        _a1_row("a2", answer="no_collision",
                actions=[{"type": "forward", "m": 1.0}]),
        _a1_row("a3", answer="collision",
                actions=[{"type": "forward", "m": 1.5}]),
        _a1_row("a4", answer="no_collision",
                actions=[{"type": "forward", "m": 2.0}]),
    ]
    predictors = blind_baseline_audit.audit_a1_rows(
        *_split(rows))["blind_baselines"]["predictors"]

    total = predictors["forward_total_m"]
    assert total["group_count"] == 4
    # One row per group memorises the corpus in-sample ...
    assert total["in_sample_accuracy"] == 1.0
    # ... and predicts nothing once it is held out.
    assert total["leave_one_out_accuracy"] == 0.0


def test_a1_audit_rejects_choices_outside_the_frozen_pair():
    item, answer_row = _a1_row(
        "a1", answer="collision", actions=[{"type": "forward", "m": 0.5}])
    item["choices"] = [{"id": "collision"}, {"id": "maybe"}]
    with pytest.raises(ValueError, match="frozen pair"):
        blind_baseline_audit.audit_a1_rows([item], [answer_row])


def test_a1_audit_rejects_mismatched_item_and_answer_ids():
    item, answer_row = _a1_row(
        "a1", answer="collision", actions=[{"type": "forward", "m": 0.5}])
    answer_row = dict(answer_row, id="a2")
    with pytest.raises(ValueError, match="ids do not match"):
        blind_baseline_audit.audit_a1_rows([item], [answer_row])


def test_a1_audit_ignores_other_tasks_and_reports_an_empty_corpus():
    other = {"id": "b1", "task_id": "B1_endpoint_distance"}
    report = blind_baseline_audit.audit_a1_rows([other], [other])
    assert report["item_count"] == 0
    assert report["blind_baselines"]["predictors"] == {}
    assert report["blind_baselines"][
        "familywise_max_leave_one_out_accuracy"] is None


def test_a1_audit_reports_dynamic_and_exact_control_arms_separately():
    rows = [
        _a1_row("n1", answer="collision",
                actions=[{"type": "forward", "m": 1.0}]),
        _a1_row("n2", answer="no_collision",
                actions=[{"type": "forward", "m": 1.5}]),
        _a1_row("c1", answer="collision",
                actions=[{"type": "forward", "m": 2.0}]),
        _a1_row("c2", answer="no_collision",
                actions=[{"type": "forward", "m": 2.0}]),
    ]
    items, answers = _split(rows)
    atoms, contexts = [], []
    for item, answer in zip(items, answers):
        source = ("natural_dynamic" if item["id"].startswith("n")
                  else "a1_control")
        answer["atom_ref"] = f"atom-{item['id']}"
        atoms.append({
            "id": answer["atom_ref"],
            "record_sha256": f"record-{item['id']}",
            "outcome": {"action_group_id": f"group-{item['id']}"},
        })
        contexts.append({
            "record_sha256": f"record-{item['id']}",
            "context": {"selection": {"proposal_provenance": {
                f"group-{item['id']}": {
                    "protocol": action_proposal.PROPOSAL_PROTOCOL_V3,
                    "variant": source,
                },
            }}},
        })

    by_source = blind_baseline_audit.audit_a1_by_source(
        items, answers, atoms, contexts)

    assert set(by_source) == {"a1_control", "natural_dynamic"}
    assert by_source["a1_control"]["item_count"] == 2
    assert by_source["a1_control"]["blind_baselines"]["predictors"][
        "forward_total_m"]["in_sample_accuracy"] == 0.5


def test_b2_rotation_only_rule_is_parameter_free_and_scores_a_right_turn():
    # _fold integrates +deg as an increasing heading, so a dead-ahead target
    # lands at bearing -90 after Turn(+90): the left sector.
    rows = [
        _b2_row("b1", answer="left",
                actions=[{"type": "turn", "deg": 90.0}], bearing_deg=-90.0),
        _b2_row("b2", answer="left",
                actions=[{"type": "turn", "deg": 90.0}], bearing_deg=-88.0),
        _b2_row("b3", answer="front",
                actions=[{"type": "forward", "m": 1.0}], bearing_deg=10.0),
    ]
    report = blind_baseline_audit.audit_b2_rows(*_split(rows))

    assert report["schema"] == blind_baseline_audit.B2_ROTATION_AUDIT_SCHEMA
    rule = report["rotation_only_frontal_prior"]
    assert rule["parameter_free"] is True
    assert rule["accuracy"] == 1.0
    assert rule["predicted_sector_counts"] == {"front": 1, "left": 2}
    assert report["precise_bearing_crosscheck"] == {
        "checked_item_count": 3,
        "missing_bearing_item_count": 0,
        "disagreement_count": 0,
    }


def test_b2_rotation_only_rule_can_be_wrong_without_raising():
    rows = [
        _b2_row("b1", answer="right",
                actions=[{"type": "turn", "deg": 90.0}], bearing_deg=90.0),
        _b2_row("b2", answer="rear",
                actions=[{"type": "turn", "deg": 90.0}], bearing_deg=170.0),
    ]
    report = blind_baseline_audit.audit_b2_rows(*_split(rows))
    assert report["rotation_only_frontal_prior"]["accuracy"] == 0.0
    assert report["rotation_only_frontal_prior"]["confusion"] == {
        "left": {"rear": 1, "right": 1}}


def test_b2_audit_reports_target_name_as_its_own_blind_predictor():
    rows = [
        _b2_row("b1", answer="left", target="cabinet",
                actions=[{"type": "forward", "m": 1.0}], bearing_deg=-90.0),
        _b2_row("b2", answer="left", target="cabinet",
                actions=[{"type": "forward", "m": 1.5}], bearing_deg=-80.0),
        _b2_row("b3", answer="front", target="door",
                actions=[{"type": "forward", "m": 1.0}], bearing_deg=0.0),
        _b2_row("b4", answer="front", target="door",
                actions=[{"type": "forward", "m": 1.5}], bearing_deg=5.0),
    ]
    report = blind_baseline_audit.audit_b2_rows(*_split(rows))

    assert report["target_label_counts"] == {"cabinet": 2, "door": 2}
    target = report["blind_baselines"]["predictors"]["target"]
    assert target["group_count"] == 2
    assert target["in_sample_accuracy"] == 1.0
    assert target["leave_one_out_accuracy"] == 1.0


def test_b2_audit_raises_when_the_private_bearing_contradicts_the_answer():
    item, answer_row = _b2_row(
        "b1", answer="left", actions=[{"type": "turn", "deg": 90.0}],
        bearing_deg=95.0)
    with pytest.raises(ValueError, match="precise bearing disagrees"):
        blind_baseline_audit.audit_b2_rows([item], [answer_row])


def test_b2_audit_counts_a_missing_bearing_instead_of_raising():
    item, answer_row = _b2_row(
        "b1", answer="left", actions=[{"type": "turn", "deg": 90.0}],
        bearing_deg=None)
    report = blind_baseline_audit.audit_b2_rows([item], [answer_row])
    assert report["precise_bearing_crosscheck"] == {
        "checked_item_count": 0,
        "missing_bearing_item_count": 1,
        "disagreement_count": 0,
    }


def test_answer_position_baseline_tracks_the_displayed_order():
    rows = [
        _b2_row("b1", answer="front", actions=[{"type": "forward", "m": 1.0}],
                bearing_deg=0.0),
        _b2_row("b2", answer="front", actions=[{"type": "forward", "m": 1.0}],
                bearing_deg=1.0),
        _b2_row("b3", answer="rear", actions=[{"type": "forward", "m": 1.0}],
                bearing_deg=180.0),
    ]
    report = blind_baseline_audit.audit_b2_rows(*_split(rows))
    assert report["answer_position_counts"] == {"1": 2, "4": 1}
    position = report["blind_baselines"]["predictors"]["answer_position"]
    assert position["in_sample_accuracy"] == pytest.approx(2 / 3)


def _write_artifact(root, rows_a1, rows_b2):
    items, answers = [], []
    for item, answer_row in list(rows_a1) + list(rows_b2):
        items.append(item)
        answers.append(answer_row)
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir(parents=True)
    for path, rows in (
            (root / "public" / "items.jsonl", items),
            (root / "private" / "answers.jsonl", answers)):
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8")


def test_artifact_audit_binds_both_sections_to_the_files_it_read(tmp_path):
    root = tmp_path / "benchmark"
    _write_artifact(
        root,
        [_a1_row("a1", answer="collision",
                 actions=[{"type": "forward", "m": 0.5}])],
        [_b2_row("b1", answer="left",
                 actions=[{"type": "turn", "deg": 90.0}], bearing_deg=-90.0)])

    with pytest.raises(ValueError, match="external source authority"):
        blind_baseline_audit.write_artifact_audit(
            root, tmp_path / "audit.json")


def test_artifact_audit_validates_candidate_source_authority(tmp_path):
    from tests.test_pl_v16_candidate_preview import (
        _PREVIEW_SOURCE_AUTHORITIES, _artifact_key, _one_a1_projection,
        _write_preview_artifact,
    )

    root = tmp_path / "benchmark"
    projection = _one_a1_projection(tmp_path)
    _write_preview_artifact(
        projection, root,
        source_records_path=tmp_path / "records.jsonl")
    authority = _PREVIEW_SOURCE_AUTHORITIES[_artifact_key(root)]

    report = blind_baseline_audit.write_artifact_audit(
        root, tmp_path / "audit.json",
        expected_source_authority=authority)

    assert report["schema"] == (
        blind_baseline_audit.BLIND_BASELINE_AUDIT_SCHEMA)
    assert report["a1_blind"]["item_count"] == 1
    assert report["b2_rotation"]["item_count"] == 0
    assert set(report["source_sha256"]) == {
        "public/items.jsonl", "private/answers.jsonl",
        "private/atoms.jsonl", "private/record_contexts.jsonl"}
    written = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert written == report
    assert not list(root.glob("**/audit.json"))


def test_artifact_audit_refuses_to_write_inside_the_benchmark(tmp_path):
    root = tmp_path / "benchmark"
    _write_artifact(
        root,
        [_a1_row("a1", answer="collision",
                 actions=[{"type": "forward", "m": 0.5}])],
        [])
    with pytest.raises(ValueError, match="outside benchmark"):
        blind_baseline_audit.write_artifact_audit(
            root, root / "private" / "audit.json")
