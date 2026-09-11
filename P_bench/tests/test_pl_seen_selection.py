from collections import Counter
from dataclasses import replace

import pytest

from pipeline.benchmark_candidates import Candidate
from pipeline import a2
from post_QA.seen_build import selection


def _candidate(index, *, starts_with):
    return Candidate(
        source_path="records.jsonl", byte_offset=index,
        record_sha256=f"{index:064x}", record_id=f"gs:f{index}",
        item_id=f"item-{index}", task_id="A1", dataset="gs",
        scene_id=f"scene-{index % 2}", outcome_id=f"out-{index}",
        action_length=2, starts_with=starts_with,
        camera_height_m=1.0, body_radius_m=0.2, hfov_deg=79.0,
        answer_bucket="collision" if index % 2 else "no_collision",
        action_signature=f"action-{index}", action_family="T-F",
        image_path=f"rgb/{index}.png", position=(float(index), 0.0, 0.0),
    )


def test_small_selection_meets_exact_total_and_turn_fraction():
    candidates = [
        _candidate(index, starts_with="turn" if index < 3 else "forward")
        for index in range(10)
    ]

    selected, report = selection.select(
        candidates, seed=5,
        dataset_task_totals={"gs": {"A1": 10}},
        c1_length_totals={}, a2_start_totals={},
        minimum_turn_first_fraction=0.3)

    assert len(selected) == 10
    assert sum(row.starts_with == "turn" for row in selected) == 3
    assert Counter(row.answer_bucket for row in selected) == {
        "collision": 5, "no_collision": 5}
    assert Counter(row.answer_bucket for row in selected
                   if row.starts_with == "turn") == {
        "collision": 1, "no_collision": 2}
    assert report["dataset_task_totals"] == {"gs/A1": 10}
    assert report["turn_first"]["gs/A1"] == {
        "count": 3, "fraction": 0.3}
    selection.validate(
        selected, dataset_task_totals={"gs": {"A1": 10}},
        c1_length_totals={}, a2_start_totals={},
        minimum_turn_first_fraction=0.3)

    broken = [replace(row, starts_with="forward")
              if row.starts_with == "turn" else row for row in selected]
    with pytest.raises(selection.SelectionShortfall):
        selection.validate(
            broken, dataset_task_totals={"gs": {"A1": 10}},
            c1_length_totals={}, a2_start_totals={},
            minimum_turn_first_fraction=0.3)


def test_selection_never_reuses_one_record_for_two_questions():
    first = _candidate(0, starts_with="turn")
    duplicate = replace(first, item_id="second", outcome_id="other")

    try:
        selection.select(
            [first, duplicate], seed=1,
            dataset_task_totals={"gs": {"A1": 2}},
            c1_length_totals={}, a2_start_totals={},
            minimum_turn_first_fraction=0.5)
    except selection.SelectionShortfall as error:
        assert "gs/A1" in str(error)
    else:
        raise AssertionError("duplicate record unexpectedly filled two slots")


def test_exact_slot_matching_moves_c1_off_a_record_needed_by_a2():
    cell = a2.balanced_cells(
        3, 1, starts_with="forward", rotation=4)[0]
    shared_a2 = replace(
        _candidate(20, starts_with="forward"),
        dataset="gs", task_id="A2", record_id="shared",
        action_length=3, answer_bucket=(
            f"o{cell.forward_ordinal_1based}-{cell.distance_rank}"),
        a2_ordinal=cell.forward_ordinal_1based, a2_rank=cell.distance_rank)
    c1_rows = [
        replace(_candidate(index, starts_with="forward"),
                dataset="gs", task_id="C1", action_length=1,
                record_id=record_id)
        for index, record_id in ((21, "shared"), (22, "alternate"))
    ]
    if selection._stable(7, c1_rows[0]) > selection._stable(7, c1_rows[1]):
        c1_rows[0], c1_rows[1] = [
            replace(c1_rows[1], record_id="shared"),
            replace(c1_rows[0], record_id="alternate"),
        ]

    selected, _report = selection.select(
        [shared_a2, *c1_rows], seed=7,
        dataset_task_totals={"gs": {"A2": 1, "C1": 1}},
        c1_length_totals={"gs": (1, 0, 0, 0, 0, 0)},
        a2_start_totals={
            "gs": {3: {"forward": 1, "turn": 0}}},
        minimum_turn_first_fraction=0.0)

    assert {(row.task_id, row.record_id) for row in selected} == {
        ("A2", "shared"), ("C1", "alternate")}


def test_a1_selection_does_not_fill_a_missing_answer_with_the_majority():
    rows = [
        replace(_candidate(index, starts_with="forward"),
                answer_bucket="no_collision" if index == 0 else "collision")
        for index in range(6)
    ]

    with pytest.raises(selection.SelectionShortfall):
        selection.select(
            rows, seed=1, dataset_task_totals={"gs": {"A1": 4}},
            c1_length_totals={}, a2_start_totals={},
            minimum_turn_first_fraction=0.0)


def test_c1_answer_labels_are_balanced_inside_each_dataset():
    rows = []
    index = 0
    for dataset, count in (("b1k", 7), ("gs", 8), ("r2r", 8)):
        for _ in range(count):
            rows.append(replace(
                _candidate(index, starts_with="forward"),
                dataset=dataset, task_id="C1", record_id=f"{dataset}:f{index}"))
            index += 1

    presented = selection._present(rows, seed=11)

    for dataset in ("b1k", "gs", "r2r"):
        counts = Counter(row.c1_label for row in presented
                         if row.dataset == dataset)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_image_matching_reassigns_shared_record_without_accepting_near_duplicate():
    rows = [replace(_candidate(i, starts_with="forward"),
                    task_id=task, record_id=uid)
            for i, task, uid in [(1, "A3", "shared"), (2, "B2", "shared"),
                                 (3, "A3", "alternate"), (4, "B2", "near")]]
    accepted = set()

    def allow(uid):
        return not (uid == "near" and "shared" in accepted)

    selected, missing = selection._match_exact_slots(
        rows, Counter({("gs", "A3", "forward"): 1,
                       ("gs", "B2", "forward"): 1}), seed=7,
        allow_record=allow, accept_record=accepted.add)
    assert not missing
    assert len({row.record_id for row in selected}) == 2
    assert not {"shared", "near"} <= accepted


def test_visual_quotas_balance_lengths_without_requiring_turn_only_l1():
    from post_QA.seen_build import visual_selection
    rows = []
    for length in range(1, 7):
        for start in (("forward",) if length == 1 else ("forward", "turn")):
            for answer in ("collision", "no_collision"):
                for _ in range(3):
                    rows.append(replace(_candidate(len(rows), starts_with=start),
                                        action_length=length, answer_bucket=answer))
    quotas = visual_selection.ordinary_quotas(rows, {"gs": {"A1": 12}})
    lengths = Counter()
    for slot, count in quotas.items():
        lengths[slot[-1]] += count
    assert lengths == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    assert sum(n for slot, n in quotas.items() if slot[2] == "turn") == 4
    assert sum(n for slot, n in quotas.items() if slot[3] == "collision") == 6


def test_a2_capacity_keeps_balanced_remainders_and_sparse_length_coverage():
    from post_QA.seen_build import visual_selection
    assert visual_selection.balanced_capacity([11, 11, 11, 10], rotation=0) == 43
    assert visual_selection.balanced_capacity([1, 1, 0, 1], rotation=0) == 2


def test_length_quotas_use_available_answers_not_an_assumed_per_length_split():
    from post_QA.seen_build import visual_selection
    rows = [replace(_candidate(i, starts_with=start), action_length=length,
                    answer_bucket="collision" if length % 2 else "no_collision")
            for i, (length, start) in enumerate(
                (length, start) for length in range(1, 7)
                for start in (("forward",) if length == 1 else ("forward", "turn"))
                for _ in range(3))]
    quotas = visual_selection.ordinary_quotas(rows, {"gs": {"A1": 12}})
    present = {visual_selection.visual_slot(row) for row in rows}
    assert set(quotas) <= present
    assert sum(quotas.values()) == 12


def test_inventory_keeps_c1_alternatives_until_endpoint_quality(monkeypatch, tmp_path):
    from post_QA.seen_build import catalog
    records = tmp_path / "records.jsonl"
    records.write_text('{}\n')
    source = catalog.Source('gs', records, 'digest', 'meta')
    first = replace(_candidate(0, starts_with='turn'), task_id='C1',
                    terminal_paths=('bad.png',))
    second = replace(first, outcome_id='alternate', item_id='alternate',
                     terminal_paths=('good.png',))
    monkeypatch.setattr(selection.catalog, 'load', lambda root: (source,))
    monkeypatch.setattr(selection.benchmark_candidates, 'project_record',
                        lambda **kwargs: [first, second])
    rows = selection.enumerate_candidates(tmp_path, seed=2, preserve_lengths=True)
    assert {row.outcome_id for row in rows} == {'out-0', 'alternate'}


def test_surface_prefilter_preserves_distinct_directions_from_one_record():
    record = {"schema_version": "abc1.record.v3", "record_uid": "one", "cases": [{
        "case_id": "case", "actions": [{"type": "forward", "m": 1.0}],
        "task_outputs": {"A4": {"checkpoint": {"fraction": 0.5}, "points": [
            {"point_id": "p1", "answer": "front"},
            {"point_id": "p2", "answer": "rear|below"},
            {"point_id": "p3", "answer": "front"},
        ]}},
    }]}
    chosen = selection._selection_facets(record, 7, preserve_lengths=True)
    points = chosen["cases"][0]["task_outputs"]["A4"]["points"]
    assert Counter(p["answer"] for p in points) == {"front": 1, "rear|below": 1}
    assert len(record["cases"][0]["task_outputs"]["A4"]["points"]) == 3


def test_direction_rebalance_preserves_slots_and_other_tasks():
    from post_QA.seen_build import direction_balance

    selected = [replace(_candidate(i, starts_with="forward"), task_id="A4",
                        answer_bucket="front") for i in range(4)]
    unrelated = replace(_candidate(9, starts_with="turn"), task_id="A1")
    alternatives = [replace(selected[i], item_id=f"alt-{i}", answer_bucket=label)
                    for i, label in enumerate(("rear", "left", "right"))]
    # A visually rejected alternative must not replace a valid case.
    rejected = replace(alternatives[0], item_id="rejected", record_id="new",
                       answer_bucket="front|above")
    result = direction_balance.rebalance(
        [*selected, unrelated], [*alternatives, rejected], seed=7,
        allow=lambda new, old: new.item_id != "rejected")
    assert Counter(r.answer_bucket for r in result if r.task_id == "A4") == {
        "front": 1, "rear": 1, "left": 1, "right": 1}
    assert result[-1] == unrelated
    assert Counter((r.dataset, r.task_id, r.starts_with, r.action_length) for r in result) == {
        ("gs", "A4", "forward", 2): 4, ("gs", "A1", "turn", 2): 1}
    assert len({r.record_id for r in result}) == 5


def test_direction_rebalance_improves_available_labels_when_others_are_unavailable():
    from post_QA.seen_build import direction_balance
    rows = [replace(_candidate(i, starts_with="forward"), task_id="A4",
                    answer_bucket="front" if i < 5 else "left") for i in range(8)]
    candidate = replace(rows[0], item_id="alternative", answer_bucket="left")
    result = direction_balance.rebalance(rows, [candidate], seed=3)
    assert Counter(row.answer_bucket for row in result) == {"front": 4, "left": 4}
