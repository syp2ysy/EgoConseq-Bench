"""Frame-first QA diversity selection."""

from collections import Counter

from pipeline import action_proposal, benchmark, diversity_selection


TASKS = tuple(benchmark.ABC_CANDIDATE_TASK_IDS)


def _projection(rows):
    items = []
    answers = []
    atoms = []
    contexts_by_record = {}
    seen_atoms = set()
    seen_contexts = set()
    for index, (dataset, frame_sha, task_id, answer, distance) in enumerate(rows):
        item_id = f"item-{index:03d}"
        atom_id = f"atom-{index:03d}"
        record_sha = f"record-{dataset}-{frame_sha}"
        items.append({
            "id": item_id,
            "task_id": task_id,
            "result_head": task_id[0],
            "model_input": {"actions": [{
                "type": "forward", "m": float(distance or 0.5)}]},
        })
        answer_row = {
            "id": item_id,
            "task_id": task_id,
            "canonical_answer": answer,
            "atom_ref": atom_id,
            "input_asset": {
                "sha256": f"marked-{index}",
                "raw_sha256": frame_sha,
            },
        }
        if distance is not None:
            answer_row["precise_distance_m"] = distance
        answers.append(answer_row)
        if atom_id not in seen_atoms:
            atoms.append({
                "id": atom_id,
                "record_sha256": record_sha,
                "outcome": {"action_group_id": f"tag-{index}"},
            })
            seen_atoms.add(atom_id)
        if record_sha not in seen_contexts:
            contexts_by_record[record_sha] = {
                "record_sha256": record_sha,
                "context": {
                    "source": {"source_dataset": dataset},
                    "selection": {"proposal_provenance": {}},
                },
            }
            seen_contexts.add(record_sha)
        contexts_by_record[record_sha]["context"]["selection"][
            "proposal_provenance"][f"tag-{index}"] = {
                "protocol": action_proposal.PROPOSAL_PROTOCOL_V4,
                "variant": action_proposal.NATURAL_DYNAMIC_VARIANT,
            }
    return {
        "items": items,
        "answers": answers,
        "atoms": atoms,
        "record_contexts": list(contexts_by_record.values()),
        "rejections": {task_id: {} for task_id in TASKS},
        "source_counts": {},
    }


def test_same_raw_frame_merges_marked_inputs_and_keeps_semantic_contrasts():
    rows = [
        ("r2r", "frame-a", "A1_collision", "collision", None),
        ("r2r", "frame-a", "A1_collision", "collision", None),
        ("r2r", "frame-a", "A1_collision", "no_collision", None),
    ]

    selected = diversity_selection.apply_selection(_projection(rows))

    answers = Counter(row["canonical_answer"] for row in selected["answers"])
    assert answers == {"collision": 1, "no_collision": 1}
    assert selected["diversity_selection"]["unique_source_frames"] == 1
    assert selected["diversity_selection"]["a1_contrast_frame_rate"] == 1.0


def test_selection_enforces_task_caps_and_eight_items_per_frame():
    rows = []
    answers = {
        "A1_collision": ["collision", "no_collision"],
        "A2_collision_step_grounding": [
            "action_1", "action_2", "action_3", "action_4"],
        "A3_contact_object": ["chair", "table", "wall", "door"],
        "B1_endpoint_distance": ["c1", "c2", "c3", "c4"],
        "B2_endpoint_direction": ["front", "left", "right", "rear"],
        "C1_future_view_selection": ["image_1", "image_2", "image_3"],
    }
    for task_id, task_answers in answers.items():
        for index, answer in enumerate(task_answers):
            distance = (
                (0.5, 1.5, 2.5, 3.5)[index]
                if task_id == "B1_endpoint_distance" else None)
            rows.append(("gs", "frame-b", task_id, answer, distance))

    selected = diversity_selection.apply_selection(_projection(rows))
    counts = Counter(item["task_id"] for item in selected["items"])

    assert sum(counts.values()) == diversity_selection.MAX_ITEMS_PER_FRAME
    assert all(counts[task_id] <= cap
               for task_id, cap in diversity_selection.TASK_CAPS.items())
    assert set(item["result_head"] for item in selected["items"]) == {
        "A", "B", "C"}
    kept_answers = {
        row["canonical_answer"] for row in selected["answers"]
        if row["task_id"] == "A1_collision"}
    assert kept_answers == {"collision", "no_collision"}


def test_b1_semantic_deduplication_uses_frozen_near_mid_far_bins():
    rows = [
        ("b1k", "frame-c", "B1_endpoint_distance", "choice_1", 0.4),
        ("b1k", "frame-c", "B1_endpoint_distance", "choice_2", 0.8),
        ("b1k", "frame-c", "B1_endpoint_distance", "choice_3", 1.5),
        ("b1k", "frame-c", "B1_endpoint_distance", "choice_4", 2.5),
    ]

    selected = diversity_selection.apply_selection(_projection(rows))

    assert len(selected["items"]) == 3
    assert selected["diversity_selection"]["semantic_answer_counts"][
        "B1_endpoint_distance"] == {"far": 1, "mid": 1, "near": 1}


def test_selection_prunes_unused_atoms_and_record_contexts():
    rows = [
        ("r2r", "frame-d", "A1_collision", "collision", None),
        ("r2r", "frame-d", "A1_collision", "collision", None),
    ]

    selected = diversity_selection.apply_selection(_projection(rows))

    assert len(selected["items"]) == 1
    assert len(selected["answers"]) == 1
    assert len(selected["atoms"]) == 1
    assert len(selected["record_contexts"]) == 1


def test_v3_a1_selection_keeps_only_globally_pairable_action_cells():
    rows = [
        ("r2r", "frame-e1", "A1_collision", "collision", 0.5),
        ("r2r", "frame-e2", "A1_collision", "no_collision", 0.5),
        ("r2r", "frame-e3", "A1_collision", "collision", 1.5),
    ]

    selected = diversity_selection.apply_selection(
        _projection(rows), balance_a1_v3=True)

    assert Counter(row["canonical_answer"] for row in selected["answers"]) == {
        "collision": 1, "no_collision": 1}
