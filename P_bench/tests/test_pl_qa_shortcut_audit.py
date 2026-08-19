"""B1 shortcut audits stay outside the frozen QA artifact."""

import copy

import pytest

from pipeline import qa_shortcut_audit


def _b1_row(item_id: str, *, displayed: float, truth_rank: int):
    separation = 0.25
    negative_capacity = min(
        3, sum(displayed - step * separation >= 0.0
               for step in range(1, 4)))
    feasible = list(range(1, negative_capacity + 2))
    assert truth_rank in feasible
    values = [
        displayed - step * separation
        for step in range(negative_capacity, 0, -1)
    ]
    values.append(displayed)
    while len(values) < 4:
        values.append(displayed + (len(values) - negative_capacity) * separation)
    choices = [
        {"id": f"choice_{index}", "value_m": value}
        for index, value in enumerate(values, 1)
    ]
    answer = choices[truth_rank - 1]["id"]
    item = {
        "id": item_id,
        "task_id": "B1_endpoint_distance",
        "choices": choices,
    }
    answer_row = {
        "id": item_id,
        "task_id": "B1_endpoint_distance",
        "canonical_answer": answer,
        "choice_certificate": {
            "displayed_distance_m": displayed,
            "precise_distance_m": displayed + 0.001,
            "minimum_separation_m": separation,
            "truth_numeric_rank_1based": truth_rank,
            "feasible_truth_numeric_ranks_1based": feasible,
        },
    }
    return item, answer_row


def test_b1_audit_reports_short_distance_rank_support_separately():
    rows = [
        _b1_row("d010", displayed=0.10, truth_rank=1),
        _b1_row("d030", displayed=0.30, truth_rank=2),
        _b1_row("d055", displayed=0.55, truth_rank=3),
        _b1_row("d080", displayed=0.80, truth_rank=4),
    ]

    report = qa_shortcut_audit.audit_b1_rows(
        [row[0] for row in rows],
        [row[1] for row in rows],
    )

    assert report["item_count"] == 4
    assert report["truth_numeric_rank_counts"] == {
        "1": 1, "2": 1, "3": 1, "4": 1}
    assert report["full_numeric_rank_support"] == {
        "item_count": 1, "fraction": 0.25}
    assert report["feasible_rank_support_counts"] == {
        "1": 1, "1,2": 1, "1,2,3": 1, "1,2,3,4": 1}
    assert report["distance_strata"]["below_1x_separation"][
        "truth_numeric_rank_counts"] == {"1": 1}
    assert report["distance_strata"]["at_least_3x_separation"][
        "truth_numeric_rank_counts"] == {"4": 1}
    assert report["blind_baselines"]["best_constant_numeric_rank_accuracy"] \
        == 0.25
    assert report["blind_baselines"][
        "best_feasible_support_conditioned_rank_accuracy"] == 1.0


def test_b1_audit_recomputes_feasible_rank_support_fail_closed():
    item, answer = _b1_row(
        "bad-support", displayed=0.30, truth_rank=2)
    changed = copy.deepcopy(answer)
    changed["choice_certificate"][
        "feasible_truth_numeric_ranks_1based"] = [1, 2, 3, 4]

    with pytest.raises(ValueError, match="feasible numeric ranks"):
        qa_shortcut_audit.audit_b1_rows(
            [item], [changed])


def test_b1_audit_rejects_a_missing_private_join():
    item, _answer = _b1_row("missing", displayed=0.80, truth_rank=4)

    with pytest.raises(ValueError, match="B1 item/answer ids"):
        qa_shortcut_audit.audit_b1_rows([item], [])
