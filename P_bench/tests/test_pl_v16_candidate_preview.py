import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from pipeline import (
    a1_common_support, action_proposal, benchmark, candidate_preview,
    candidate_sources,
    dataset_contracts, gate_authority, gs_semantic, record, scene_pool,
    source_manifest,
)
from tests._synthetic import LEVEL_FLOOR_FIT, make_frame
from tests.test_pl_dataset_contracts import (
    _gs_semantic_a_case, _gs_run_meta,
)
from tests.test_pl_v16_a_candidates import _case as _a_case
from tests.test_pl_v16_b_candidates import _b_case


_PREVIEW_SOURCE_AUTHORITIES = {}


def _artifact_key(path) -> str:
    return str(gate_authority.lexical_absolute_path(path))


def _write_preview_artifact(projection, artifact_root, **kwargs):
    """Give test artifacts authority captured from their fixture inputs."""
    source = kwargs["source_records_path"]
    paths = list(source) if isinstance(source, (list, tuple)) else [source]
    run_meta = kwargs.get("source_run_meta_sha256")
    if run_meta is None:
        run_meta_digests = [None] * len(paths)
    elif isinstance(run_meta, (list, tuple)):
        run_meta_digests = list(run_meta)
    else:
        run_meta_digests = [run_meta]
    authority = gate_authority.resolve_preview_source_authority_from_inputs(
        records_paths=paths,
        expected_run_meta_sha256=run_meta_digests,
        authority_id="test-fixture:preview-sources")
    kwargs["expected_source_authority"] = authority
    result = candidate_preview.write_preview_artifact(
        projection, artifact_root, **kwargs)
    _PREVIEW_SOURCE_AUTHORITIES[_artifact_key(artifact_root)] = authority
    return result


def _validate_preview_artifact(artifact_root):
    return candidate_preview.validate_preview_artifact(
        artifact_root,
        expected_source_authority=
            _PREVIEW_SOURCE_AUTHORITIES[_artifact_key(artifact_root)])


def _evaluate_preview_artifact(artifact_root, **kwargs):
    return candidate_preview.evaluate_preview_artifact(
        artifact_root,
        expected_source_authority=
            _PREVIEW_SOURCE_AUTHORITIES[_artifact_key(artifact_root)],
        **kwargs)


def _render_preview_html(benchmark_path, report_root):
    artifact_root = Path(benchmark_path).parent
    return candidate_preview.render_preview_html(
        benchmark_path, report_root,
        expected_source_authority=
            _PREVIEW_SOURCE_AUTHORITIES[_artifact_key(artifact_root)])


def _png(path: Path, *, size=(8, 8), color=(12, 34, 56)) -> str:
    Image.new("RGB", size, color).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_a1_projection(tmp_path: Path) -> dict:
    image = tmp_path / "real-initial.png"
    _png(image, size=(640, 480))
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    rec["schema_version"] = record.SCHEMA_VERSION
    rec["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    rec["outcomes"] = [copy.deepcopy(outcome)]
    rec["image_path"] = image.name
    (tmp_path / "records.jsonl").write_text(
        json.dumps(record.json_value(rec), sort_keys=True) + "\n")
    return candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build")


def _one_a2_projection(tmp_path: Path) -> dict:
    image = tmp_path / "a2-initial.png"
    _png(image, size=(640, 480))
    rec, outcome = _a_case(collision=True, arc=1.5)
    rec = copy.deepcopy(rec)
    rec.update({
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "outcomes": [copy.deepcopy(outcome)],
        "image_path": image.name,
    })
    _rewrite_jsonl(tmp_path / "records.jsonl", [record.json_value(rec)])
    return candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build")


def _one_b_projection(tmp_path: Path) -> dict:
    image = tmp_path / "b-initial.png"
    _png(image, size=(100, 100))
    rec, outcome = _b_case()
    rec = copy.deepcopy(rec)
    rec.update({
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "outcomes": [copy.deepcopy(outcome)],
        "image_path": image.name,
    })
    _rewrite_jsonl(tmp_path / "records.jsonl", [record.json_value(rec)])
    return candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build")


def _two_a1_projection(tmp_path: Path) -> dict:
    image = tmp_path / "two-records.png"
    _png(image, size=(640, 480))
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    rec.update({
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "outcomes": [copy.deepcopy(outcome)],
        "image_path": image.name,
    })
    second = copy.deepcopy(rec)
    second["frame_id"] = "second-record"
    second["outcomes"][0]["outcome_id"] = "second-outcome"
    _rewrite_jsonl(
        tmp_path / "records.jsonl",
        [record.json_value(rec), record.json_value(second)])
    return candidate_preview.compile_main_records(
        [rec, second], asset_root=tmp_path, build_root=tmp_path / "build")


def test_preview_validation_hashes_each_source_record_once(
        tmp_path, monkeypatch):
    image = tmp_path / "multi-outcome.png"
    _png(image, size=(640, 480))
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    second = copy.deepcopy(outcome)
    second["outcome_id"] = "second-outcome"
    rec.update({
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "outcomes": [copy.deepcopy(outcome), second],
        "image_path": image.name,
    })
    records_path = tmp_path / "records.jsonl"
    _rewrite_jsonl(records_path, [record.json_value(rec)])
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        candidate_preview.compile_main_records(
            [rec], asset_root=tmp_path, build_root=tmp_path / "build"),
        artifact, source_records_path=records_path)
    original = candidate_sources.canonical_sha256
    record_hashes = 0

    def counted(value):
        nonlocal record_hashes
        if isinstance(value, dict) and "outcomes" in value:
            record_hashes += 1
        return original(value)

    monkeypatch.setattr(candidate_sources, "canonical_sha256", counted)

    _validate_preview_artifact(artifact)

    assert record_hashes == 1


def test_preview_validation_does_not_repeat_b_target_authentication(
        tmp_path, monkeypatch):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_b_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    original = record.authenticated_b_target
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(record, "authenticated_b_target", counted)

    _validate_preview_artifact(artifact)

    assert calls == 0


def test_preview_validation_does_not_replay_source_bound_b_relation(
        tmp_path, monkeypatch):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_b_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")

    original = record.b_endpoint_relation_atom_valid
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(record, "b_endpoint_relation_atom_valid", counted)

    _validate_preview_artifact(artifact)

    assert calls == 0


def test_preview_writes_bijective_abc_artifact_and_html(tmp_path):
    artifact = tmp_path / "candidate_qa"
    report = tmp_path / "candidate_qa_report"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl",
        collection_funnel={"status": "completed"})
    benchmark_json = _validate_preview_artifact(artifact)
    _render_preview_html(artifact / "benchmark.json", report)

    assert tuple(benchmark_json["task_catalog"]) == \
        benchmark.ABC_CANDIDATE_TASK_IDS
    assert benchmark_json["task_scope"] == "ABC"
    assert benchmark_json["coverage"]["A1_collision"] == 1
    assert benchmark_json["headline_eligible"] is False
    html = (report / "index.html").read_text()
    assert all(f'data-head="{head}"' in html for head in "ABC")
    assert 'data-head="D"' not in html
    assert 'data-browser-schema="egoconseq.case-browser.v1"' in html
    assert 'data-task-filter="A1_collision"' in html
    assert 'id="case-grid"' in html
    assert '"response_mode":"open_text"' in html
    assert '"camera_height_above_visible_floor_m":1.0' in html
    report_json = json.loads((artifact / "report.json").read_text())
    assert report_json["distributions"]["A1_collision"]["answers"] == {
        "no_collision": 1}
    public_manifest = json.loads(
        (artifact / "public" / "manifest.json").read_text())
    contract = public_manifest["public_input_contract"]
    assert contract == benchmark.PUBLIC_INPUT_CONTRACT
    assert contract["body"]["shape"] == "centered_planar_disc"
    assert contract["camera"]["resolution_px"] == [640, 480]
    assert contract["camera"]["pitch_deg"] == 0.0
    assert contract["action_program"]["positive_turn"] == "right"
    assert contract["relations"]["B1_endpoint_distance"] == \
        ("robot_center_to_target_ground_support_nearest_distance; the item "
         "task_metadata declares the authenticated geometry protocol")


def test_compile_main_records_uses_v16_abc_builders(tmp_path):
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    rec["outcomes"] = [copy.deepcopy(outcome)]
    image = tmp_path / "img" / "frame.png"
    image.parent.mkdir()
    _png(image, size=(640, 480))
    rec["image_path"] = "img/frame.png"

    projection = candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build")

    assert "A1_collision" in {
        item["task_id"] for item in projection["items"]}
    assert set(projection["rejections"]) == set(
        benchmark.ABC_CANDIDATE_TASK_IDS)
    assert projection["rejections"]["A2_collision_step_grounding"] == {
        "collision_required": 1}


def test_candidate_report_rejects_unknown_construction_reason(tmp_path):
    projection = _one_a1_projection(tmp_path)
    projection["rejections"]["A2_collision_step_grounding"] = {
        "invented_reason": 1}

    with pytest.raises(ValueError, match="unknown QA report reason"):
        _write_preview_artifact(
            projection, tmp_path / "candidate_qa",
            source_records_path=tmp_path / "records.jsonl")


def test_preview_validator_rejects_unknown_report_reason(tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text())
    report["rejections"]["A2_collision_step_grounding"] = {
        "invented_reason": 1}
    report_path.write_text(json.dumps(report, sort_keys=True))

    with pytest.raises(ValueError, match="unknown QA report reason"):
        _validate_preview_artifact(artifact)


def test_main_preview_cap_stops_after_requested_real_examples(tmp_path):
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    second = copy.deepcopy(outcome)
    second["outcome_id"] = "second-real-outcome"
    rec["outcomes"] = [copy.deepcopy(outcome), second]
    image = tmp_path / "img" / "frame.png"
    image.parent.mkdir()
    _png(image, size=(640, 480))
    rec["image_path"] = "img/frame.png"

    projection = candidate_preview.compile_main_records(
        [rec], asset_root=tmp_path, build_root=tmp_path / "build",
        max_items_per_task=1)
    assert sum(item["task_id"] == "A1_collision"
               for item in projection["items"]) == 1


def test_compile_contexts_only_for_records_referenced_by_atoms(tmp_path):
    """A capped source record must not bloat the private context table."""
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    rec["schema_version"] = record.SCHEMA_VERSION
    rec["oracle_contract_version"] = record.ORACLE_CONTRACT_VERSION
    rec["outcomes"] = [copy.deepcopy(outcome)]
    rec["image_path"] = "img/frame.png"
    capped = copy.deepcopy(rec)
    capped["frame_id"] = "capped-record"
    capped["outcomes"][0]["outcome_id"] = "capped-outcome"
    image = tmp_path / "img" / "frame.png"
    image.parent.mkdir()
    _png(image, size=(640, 480))

    projection = candidate_preview.compile_main_records(
        [rec, capped], asset_root=tmp_path, build_root=tmp_path / "build",
        max_items_per_task=1)

    atom_digests = {
        atom["record_sha256"] for atom in projection["atoms"]}
    context_digests = {
        row["record_sha256"] for row in projection["record_contexts"]}
    assert context_digests == atom_digests
    assert len(context_digests) == 1

    records_path = tmp_path / "records.jsonl"
    _rewrite_jsonl(records_path, [record.json_value(rec),
                                  record.json_value(capped)])
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        projection, artifact, source_records_path=records_path)
    _validate_preview_artifact(artifact)


def test_a1_common_support_keeps_balanced_cells_and_prunes_private_rows():
    def item(item_id, task_id, actions):
        return {
            "id": item_id,
            "task_id": task_id,
            "model_input": {"actions": actions},
        }

    forward_one = [{"type": "forward", "m": 1.0}]
    rows = [
        (item("a-collision-1", "A1_collision", forward_one), "collision"),
        (item("a-collision-2", "A1_collision", forward_one), "collision"),
        (item("a-safe-1", "A1_collision", forward_one), "no_collision"),
        (item("a-only-collision", "A1_collision",
              [{"type": "forward", "m": 2.0}]), "collision"),
        (item("a-turn-right", "A1_collision", [
            {"type": "turn", "deg": 15.0},
            {"type": "forward", "m": 1.0},
        ]), "collision"),
        (item("a-turn-left", "A1_collision", [
            {"type": "turn", "deg": -15.0},
            {"type": "forward", "m": 1.0},
        ]), "no_collision"),
        (item("b-kept", "B1_endpoint_distance", forward_one), "choice_1"),
    ]
    projection = {
        "items": [value for value, _answer in rows],
        "answers": [{
            "id": value["id"],
            "task_id": value["task_id"],
            "canonical_answer": answer,
            "atom_ref": f"atom-{value['id']}",
        } for value, answer in rows],
        "atoms": [{
            "id": f"atom-{value['id']}",
            "record_sha256": f"record-{value['id']}",
        } for value, _answer in rows],
        "record_contexts": [{
            "record_sha256": f"record-{value['id']}",
            "context": {"frame_id": value["id"]},
        } for value, _answer in rows],
        "rejections": {
            task_id: {} for task_id in benchmark.ABC_CANDIDATE_TASK_IDS},
    }

    selected = a1_common_support.apply_selection(projection)

    assert [value["id"] for value in selected["items"]] == [
        "a-collision-1", "a-safe-1", "a-turn-right", "a-turn-left",
        "b-kept"]
    assert [value["id"] for value in selected["answers"]] == [
        "a-collision-1", "a-safe-1", "a-turn-right", "a-turn-left",
        "b-kept"]
    assert {value["id"] for value in selected["atoms"]} == {
        "atom-a-collision-1", "atom-a-safe-1", "atom-a-turn-right",
        "atom-a-turn-left", "atom-b-kept"}
    assert {value["record_sha256"]
            for value in selected["record_contexts"]} == {
        "record-a-collision-1", "record-a-safe-1",
        "record-a-turn-right", "record-a-turn-left", "record-b-kept"}
    assert selected["rejections"]["A1_collision"] == {
        "a1_common_support_unmatched": 2}
    assert selected["publication_selection"] == {
        "A1_collision": a1_common_support.POLICY}


def test_a1_common_support_default_only_selects_uniform_v2_sources():
    v2 = {
        "selection": {"proposal_provenance": {
            "natural": {
                "protocol": "depth-conditioned-pair-v2",
                "variant": "natural",
            },
            "neighbor": {
                "protocol": "c1-counterfactual-neighbor.v2",
                "variant": "c1_counterfactual",
            },
        }},
    }
    v1 = copy.deepcopy(v2)
    v1["selection"]["proposal_provenance"]["natural"]["protocol"] = \
        "depth-conditioned-pair-v1"
    only_neighbor = copy.deepcopy(v2)
    del only_neighbor["selection"]["proposal_provenance"]["natural"]

    assert a1_common_support.default_for_records([v2]) is True
    assert a1_common_support.default_for_records([v2, v1]) is False
    assert a1_common_support.default_for_records([v1]) is False
    assert a1_common_support.default_for_records([only_neighbor]) is False
    assert a1_common_support.resolve_enabled(
        None, {"depth-conditioned-pair-v2"},
        family_authority_present=True) is False
    assert a1_common_support.resolve_enabled(
        False, {"depth-conditioned-pair-v2"},
        family_authority_present=False) is False
    assert a1_common_support.resolve_enabled(
        True, {"depth-conditioned-pair-v1"},
        family_authority_present=False) is True
    assert a1_common_support.v3_default_for_protocols({
        action_proposal.PROPOSAL_PROTOCOL_V3}) is True
    assert a1_common_support.v3_default_for_protocols({
        action_proposal.PROPOSAL_PROTOCOL_V4}) is True
    assert a1_common_support.v3_default_for_protocols({
        action_proposal.PROPOSAL_PROTOCOL_V2,
        action_proposal.PROPOSAL_PROTOCOL_V3}) is False


def test_a1_common_support_policy_is_bound_in_report_and_source_map(tmp_path):
    projection = a1_common_support.apply_selection(
        _one_a1_projection(tmp_path))
    artifact = tmp_path / "candidate_qa"

    _write_preview_artifact(
        projection, artifact,
        source_records_path=tmp_path / "records.jsonl")

    expected = {
        "A1_collision": a1_common_support.POLICY}
    report = json.loads((artifact / "report.json").read_text())
    source_map = json.loads(
        (artifact / "private" / "source_map.json").read_text())
    assert report["publication_selection"] == expected
    assert source_map["publication_selection"] == expected
    _validate_preview_artifact(artifact)


def test_a1_v3_selection_balances_natural_controls_and_excludes_pairs():
    def row(item_id, answer, group_id, variant, actions, *, task_id=None,
            dataset="r2r"):
        task_id = task_id or "A1_collision"
        return ({
            "id": item_id,
            "task_id": task_id,
            "model_input": {
                "actions": actions,
                "body_radius_m": 0.2,
                "camera_height_above_visible_floor_m": 1.0,
                "hfov_deg": 79.0,
                "vfov_deg": 63.453048,
            },
        }, {
            "id": item_id,
            "task_id": task_id,
            "canonical_answer": answer,
            "atom_ref": f"atom-{item_id}",
        }, {
            "id": f"atom-{item_id}",
            "record_sha256": f"record-{item_id}",
            "outcome": {"action_group_id": group_id},
        }, {
            "record_sha256": f"record-{item_id}",
            "context": {"source": {"source_dataset": dataset},
                        "collection_contract": {"source_dataset": dataset},
                        "selection": {"proposal_provenance": {
                group_id: {
                    "protocol": action_proposal.PROPOSAL_PROTOCOL_V3,
                    "variant": variant,
                },
            }}},
        })

    forward_1 = [{"type": "forward", "m": 1.0}]
    rows = [
        row("n-collision", "collision", "n1", "natural_dynamic", forward_1),
        row("n-safe", "no_collision", "n2", "natural_dynamic",
            [{"type": "forward", "m": 1.5}]),
        row("n-extra", "collision", "n3", "natural_dynamic", forward_1),
        row("c-collision", "collision", "c1", "a1_control",
            [{"type": "forward", "m": 2.0}]),
        row("c-safe", "no_collision", "c2", "a1_control",
            [{"type": "forward", "m": 2.0}]),
        row("paired", "no_collision", "p1", "safe", forward_1),
        row("b-kept", "choice_1", "b1", "safe", forward_1,
            task_id="B1_endpoint_distance"),
    ]
    projection = {
        "items": [value[0] for value in rows],
        "answers": [value[1] for value in rows],
        "atoms": [value[2] for value in rows],
        "record_contexts": [value[3] for value in rows],
        "rejections": {
            task_id: {} for task_id in benchmark.ABC_CANDIDATE_TASK_IDS},
    }

    selected = a1_common_support.apply_v3_selection(projection)

    assert [item["id"] for item in selected["items"]] == [
        "n-collision", "n-safe", "c-collision", "c-safe", "b-kept"]
    assert selected["rejections"]["A1_collision"] == {
        "a1_natural_balance_unmatched": 1,
        "a1_nonpublication_source": 1,
    }
    assert selected["publication_selection"] == {
        "A1_collision": a1_common_support.V3_POLICY}
    a1_common_support.validate_v3_selection(
        selected["items"],
        {value["id"]: value for value in selected["answers"]},
        selected["atoms"], selected["record_contexts"])


def test_a1_v3_never_balances_a_collision_against_another_dataset_safe():
    def row(item_id, answer, dataset):
        return ({
            "id": item_id,
            "task_id": "A1_collision",
            "model_input": {
                "actions": [{"type": "forward", "m": 1.0}],
                "body_radius_m": 0.2,
                "camera_height_above_visible_floor_m": 1.0,
                "hfov_deg": 79.0,
                "vfov_deg": 63.453048,
            },
        }, {
            "id": item_id,
            "task_id": "A1_collision",
            "canonical_answer": answer,
            "atom_ref": f"atom-{item_id}",
        }, {
            "id": f"atom-{item_id}",
            "record_sha256": f"record-{item_id}",
            "outcome": {"action_group_id": item_id},
        }, {
            "record_sha256": f"record-{item_id}",
            "context": {
                "source": {"source_dataset": dataset},
                "collection_contract": {"source_dataset": dataset},
                "selection": {"proposal_provenance": {item_id: {
                    "protocol": action_proposal.PROPOSAL_PROTOCOL_V3,
                    "variant": action_proposal.NATURAL_DYNAMIC_VARIANT,
                }}},
            },
        })

    rows = [
        row("r2r-collision", "collision", "r2r"),
        row("gs-safe", "no_collision", "gs"),
    ]
    projection = {
        "items": [value[0] for value in rows],
        "answers": [value[1] for value in rows],
        "atoms": [value[2] for value in rows],
        "record_contexts": [value[3] for value in rows],
        "rejections": {
            task_id: {} for task_id in benchmark.ABC_CANDIDATE_TASK_IDS},
    }

    selected = a1_common_support.apply_v3_selection(projection)

    assert selected["items"] == []
    assert selected["rejections"]["A1_collision"] == {
        "a1_natural_balance_unmatched": 2}
    a1_common_support.validate_v3_selection(
        selected["items"], {}, selected["atoms"],
        selected["record_contexts"])


def test_a1_v3_reads_gs_dataset_from_v18_source_without_legacy_contract():
    item = {"id": "gs-a1"}
    answers = {"gs-a1": {"atom_ref": "atom-gs-a1"}}
    atoms = {"atom-gs-a1": {"record_sha256": "record-gs"}}
    contexts = {
        "record-gs": {"source": {"source_dataset": "gs"}},
    }

    assert a1_common_support.source_dataset_for_item(
        item, answers, atoms, contexts) == "gs"


@pytest.mark.parametrize("variant,expected", [
    ("natural_dynamic", None),
    ("a1_control", None),
    ("safe", "a1_nonpublication_source"),
    ("collision", "a1_nonpublication_source"),
    ("c1_counterfactual", "a1_nonpublication_source"),
])
def test_v3_a1_source_gate_runs_before_task_caps(variant, expected):
    record_value = {"selection": {"proposal_provenance": {
        "group": {
            "protocol": action_proposal.PROPOSAL_PROTOCOL_V3,
            "variant": variant,
        },
    }}}

    assert a1_common_support.v3_source_rejection(
        record_value, {"action_group_id": "group"}) == expected


@pytest.mark.parametrize("variant,expected", [
    ("natural_dynamic", None),
    ("a1_control", None),
    ("safe", "a1_nonpublication_source"),
    ("c1_counterfactual", "a1_nonpublication_source"),
])
def test_v4_ordinary_provenance_uses_the_a1_v3_publication_path(
        variant, expected):
    record_value = {"selection": {"proposal_provenance": {"group": {
        "protocol": action_proposal.PROPOSAL_PROTOCOL_V4,
        "variant": variant,
    }}}}

    assert a1_common_support.v3_source_rejection(
        record_value, {"action_group_id": "group"}) == expected


def test_preview_validator_detects_public_asset_tamper(tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    item = json.loads(
        (artifact / "public" / "items.jsonl").read_text().strip())
    (artifact / item["model_input"]["initial_rgb"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset"):
        _validate_preview_artifact(artifact)


def test_shared_public_asset_is_verified_once_and_hardlinked(
        tmp_path, monkeypatch):
    source = tmp_path / "shared.png"
    digest = _png(source)
    items = [{
        "model_input": {
            "initial_rgb": str(source),
            "initial_rgb_sha256": digest,
        },
        "choices": [],
    } for _ in range(2)]
    calls = []
    sha256_file = candidate_preview.io_utils.sha256_file

    def counted_sha256(path):
        calls.append(Path(path))
        return sha256_file(path)

    monkeypatch.setattr(
        candidate_preview.io_utils, "sha256_file", counted_sha256)
    artifact = tmp_path / "candidate_qa"

    candidate_preview._relocate_public_assets(items, artifact)

    destination = artifact / items[0]["model_input"]["initial_rgb"]
    assert items[0]["model_input"]["initial_rgb"] == \
        items[1]["model_input"]["initial_rgb"]
    assert calls == [source]
    assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_preview_validation_hashes_each_public_asset_once(
        tmp_path, monkeypatch):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    item = json.loads(
        (artifact / "public" / "items.jsonl").read_text().strip())
    asset = (artifact / item["model_input"]["initial_rgb"]).resolve()
    calls = []
    sha256_file = candidate_preview.io_utils.sha256_file

    def counted_sha256(path):
        if Path(path).resolve() == asset:
            calls.append(asset)
        return sha256_file(path)

    monkeypatch.setattr(
        candidate_preview.io_utils, "sha256_file", counted_sha256)

    _validate_preview_artifact(artifact)

    assert calls == [asset]


def _rewrite_jsonl(path: Path, rows) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _refresh_private_manifest(artifact: Path) -> None:
    path = artifact / "private" / "manifest.json"
    manifest = json.loads(path.read_text())
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256(
            (artifact / "private" / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True))


def _refresh_public_manifest(artifact: Path) -> None:
    path = artifact / "public" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["items.jsonl"] = hashlib.sha256(
        (artifact / "public" / "items.jsonl").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True))


def test_preview_validator_rejects_source_atom_tamper_even_with_new_manifest(
        tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    path = artifact / "private" / "atoms.jsonl"
    atoms = [json.loads(line) for line in path.read_text().splitlines()]
    atoms[0]["record_sha256"] = "f" * 64
    _rewrite_jsonl(path, atoms)
    _refresh_private_manifest(artifact)

    with pytest.raises(ValueError, match="source atom"):
        _validate_preview_artifact(artifact)


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_preview_validator_rejects_record_context_change(
        tmp_path, mutation):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    path = artifact / "private" / "record_contexts.jsonl"
    contexts = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        contexts.clear()
    else:
        contexts[0]["context"]["frame_id"] = "swapped-frame"
    _rewrite_jsonl(path, contexts)
    _refresh_private_manifest(artifact)

    with pytest.raises(ValueError, match="record context content"):
        _validate_preview_artifact(artifact)


@pytest.mark.parametrize("mutation", ["duplicate", "reordered"])
def test_preview_validator_rejects_noncanonical_record_contexts(
        tmp_path, mutation):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _two_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    path = artifact / "private" / "record_contexts.jsonl"
    contexts = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(contexts) == 2
    if mutation == "duplicate":
        contexts.append(copy.deepcopy(contexts[-1]))
    else:
        contexts.reverse()
    _rewrite_jsonl(path, contexts)
    _refresh_private_manifest(artifact)

    with pytest.raises(ValueError, match="canonically ordered"):
        _validate_preview_artifact(artifact)


def test_merge_rejects_conflicting_context_for_one_record_digest(tmp_path):
    projection = _one_a1_projection(tmp_path)
    conflicting = copy.deepcopy(projection)
    conflicting["record_contexts"][0]["context"]["frame_id"] = \
        "conflicting-frame"

    with pytest.raises(ValueError, match="distinct record contexts"):
        candidate_preview.merge_projections(projection, conflicting)


def test_preview_validator_rejects_answer_outside_public_choices(
        tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    path = artifact / "private" / "answers.jsonl"
    answers = [json.loads(line) for line in path.read_text().splitlines()]
    answers[0]["canonical_answer"] = "not_a_public_choice"
    _rewrite_jsonl(path, answers)
    _refresh_private_manifest(artifact)

    with pytest.raises(ValueError, match="public choice"):
        _validate_preview_artifact(artifact)


def test_preview_validator_rejects_turn_index_added_to_a2_choices(tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a2_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    items_path = artifact / "public" / "items.jsonl"
    items = [json.loads(line) for line in items_path.read_text().splitlines()]
    a2 = next(row for row in items
              if row["task_id"] == "A2_collision_step_grounding")
    a2["choices"].insert(1, {"id": "action_2", "text": "action_2"})
    _rewrite_jsonl(items_path, items)
    _refresh_public_manifest(artifact)
    benchmark_path = artifact / "benchmark.json"
    benchmark_json = json.loads(benchmark_path.read_text())
    for case in benchmark_json["cases"]:
        if case["public"]["id"] == a2["id"]:
            case["public"] = a2
    benchmark_path.write_text(json.dumps(benchmark_json, sort_keys=True))

    with pytest.raises(ValueError, match="A answer"):
        _validate_preview_artifact(artifact)


def test_preview_validator_rederives_b1_plausibility_certificate(tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_b_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    answers_path = artifact / "private" / "answers.jsonl"
    answers = [json.loads(line)
               for line in answers_path.read_text().splitlines()]
    b1 = next(row for row in answers
              if row["task_id"] == "B1_endpoint_distance")
    b1["choice_certificate"]["plausible_lower_bound_m"] = 1.0
    _rewrite_jsonl(answers_path, answers)
    _refresh_private_manifest(artifact)

    with pytest.raises(ValueError, match="B choice certificate"):
        _validate_preview_artifact(artifact)


def test_preview_validator_rejects_public_actions_changed_from_source_atom(
        tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    items_path = artifact / "public" / "items.jsonl"
    items = [json.loads(line) for line in items_path.read_text().splitlines()]
    items[0]["model_input"]["actions"] = [{"type": "forward", "m": 99.0}]
    _rewrite_jsonl(items_path, items)
    _refresh_public_manifest(artifact)
    benchmark_path = artifact / "benchmark.json"
    benchmark_json = json.loads(benchmark_path.read_text())
    benchmark_json["cases"][0]["public"] = items[0]
    benchmark_path.write_text(json.dumps(benchmark_json, sort_keys=True))
    answers = [json.loads(line) for line in (
        artifact / "private" / "answers.jsonl").read_text().splitlines()]
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_text())
    report["distributions"] = candidate_preview._candidate_distributions(
        items, {value["id"]: value for value in answers})
    report_path.write_text(json.dumps(report, sort_keys=True))

    with pytest.raises(ValueError, match="public model input"):
        _validate_preview_artifact(artifact)


def test_one_call_pipeline_requires_trusted_run_metadata_digest(tmp_path):
    rec, outcome = _a_case(collision=False)
    rec = copy.deepcopy(rec)
    rec.update({
        "schema_version": record.SCHEMA_VERSION,
        "oracle_contract_version": record.ORACLE_CONTRACT_VERSION,
        "outcomes": [copy.deepcopy(outcome)],
        "image_path": "img/frame.png",
    })
    image = tmp_path / "shard" / "img" / "frame.png"
    image.parent.mkdir(parents=True)
    _png(image, size=(640, 480))
    records_path = tmp_path / "shard" / "records.jsonl"
    records_path.write_text(json.dumps(rec, allow_nan=False) + "\n")

    with pytest.raises(ValueError, match="run metadata digest"):
        candidate_preview.build_main_preview(
            records_path, tmp_path / "candidate_qa",
            tmp_path / "candidate_qa_report")


def test_one_call_pipeline_accepts_authenticated_b1k_main_shard(
        tmp_path, monkeypatch):
    """Catches the preview compiler rejecting the registered B1K route."""
    from tests.test_pl_b1k_runtime import _write_manifest

    manifest = _write_manifest(tmp_path)
    scenes = scene_pool.discover_b1k_train_scenes(tmp_path, manifest)
    shard = tmp_path / "shard"
    shard.mkdir()
    records_path = shard / "records.jsonl"
    records_path.write_text("")
    run_meta = shard / "run_meta.json"
    run_meta_value = {
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
    }
    run_meta.write_text(json.dumps(run_meta_value))
    run_meta_sha256 = hashlib.sha256(run_meta.read_bytes()).hexdigest()

    def source_replay(*_args, **_kwargs):
        raise AssertionError("candidate compilation replayed source assets")

    monkeypatch.setattr(
        candidate_sources.record_validation,
        "validate_file_source_bound", source_replay)
    decode_calls = 0
    decode = candidate_preview.record_fields.decode_records_for_schema

    def counted_decode(payload, schema_version):
        nonlocal decode_calls
        decode_calls += 1
        return decode(payload, schema_version)

    monkeypatch.setattr(
        candidate_preview.record_fields,
        "decode_records_for_schema", counted_decode)

    result = candidate_preview.build_main_preview(
        records_path, tmp_path / "candidate_qa",
        tmp_path / "candidate_qa_report",
        run_meta_sha256=run_meta_sha256)

    assert result["headline_eligible"] is False
    assert result["coverage"] == {
        task_id: 0 for task_id in benchmark.ABC_CANDIDATE_TASK_IDS}
    assert decode_calls == 1
    assert not tuple(tmp_path.glob(".candidate_preview*"))


@pytest.mark.parametrize(
    ("collision", "expected_tasks"),
    [(False, ["A1_collision"]),
     (True, ["A1_collision", "A2_collision_step_grounding"])],
)
def test_gs_compiler_routes_available_tasks_through_evidence_gates(
        tmp_path, collision, expected_tasks):
    record_value, outcome = _gs_semantic_a_case(collision=collision)
    frame = make_frame()
    frame.scene_id = record_value["source"]["scene_id"]
    alignment = record_value["gs_scene_capability"][
        "alignment_certificate"]
    semantic = gs_semantic.BboxSemanticIndex(
        mins=[[-2.0, -1.0, 0.0], [-1.0, -1.0, 1.5]],
        maxs=[[2.0, 1.0, 1.2], [1.0, 1.0, 3.0]],
        ids=[5, 7], id_to_cat=frame.id_to_cat,
        alignment_certificate=alignment)
    binding = dataset_contracts.resolve_gs_collision_binding(
        record_value["source"])
    semantic_sha256 = next(
        row["sha256"] for row in record_value["source"]["source_assets"]
        if row["role"] == "semantic")
    frame.semantic_index = semantic.visible_depth_view(
        frame.pts, frame.pts_sem,
        geometry_authority_sha256=binding.authority_sha256,
        semantic_source_sha256=semantic_sha256)
    envelope = record.build_record_v18(
        frame, [], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=record_value["source"])
    envelope["outcomes"] = [outcome]
    image = tmp_path / "img" / "f.png"
    image.parent.mkdir()
    _png(image, size=(640, 480))

    projection = candidate_preview.compile_main_records(
        [envelope], asset_root=tmp_path, build_root=tmp_path / "build")

    assert [item["task_id"] for item in projection["items"]] == expected_tasks
    for task_id in (
            "A3_contact_object", "B1_endpoint_distance",
            "B2_endpoint_direction"):
        assert projection["rejections"][task_id].get(
            "capability_unavailable", 0) == 0
        assert sum(projection["rejections"][task_id].values()) == 1


def test_one_call_pipeline_accepts_authenticated_gs_v18_main_shard(tmp_path):
    run_meta_value, source = _gs_run_meta(tmp_path)
    # This empty-shard route test exercises authenticated v18 decoding, not
    # the balanced selection envelope covered by collection tests.
    run_meta_value["params"].pop("setting_sampling_policy")
    run_meta_value["params"]["action_mode"] = "empty-test-shard"
    frame = make_frame()
    frame.scene_id = source["scene_id"]
    trusted_context = source_manifest.gs_v18_context_from_run_meta(
        run_meta_value, authority_sha256="d" * 64)
    alignment = trusted_context.gs_scene_capability_resolver(
        source["scene_id"])["alignment_certificate"]
    semantic = gs_semantic.BboxSemanticIndex(
        mins=[[-2.0, -1.0, 0.0], [-1.0, -1.0, 1.5]],
        maxs=[[2.0, 1.0, 1.2], [1.0, 1.0, 3.0]],
        ids=[5, 7], id_to_cat=frame.id_to_cat,
        alignment_certificate=alignment)
    binding = dataset_contracts.resolve_gs_collision_binding(source)
    semantic_sha256 = next(
        row["sha256"] for row in source["source_assets"]
        if row["role"] == "semantic")
    frame.semantic_index = semantic.visible_depth_view(
        frame.pts, frame.pts_sem,
        geometry_authority_sha256=binding.authority_sha256,
        semantic_source_sha256=semantic_sha256)
    shard = tmp_path / "shard"
    image = shard / "img" / "f.png"
    image.parent.mkdir(parents=True)
    _png(image, size=(640, 480))
    record_value = record.build_record_v18(
        frame, [], image_path="img/f.png",
        floor_calibration=LEVEL_FLOOR_FIT,
        source_provenance=source)
    records_path = shard / "records.jsonl"
    records_path.write_text(json.dumps(record_value, allow_nan=False) + "\n")
    run_meta_path = shard / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta_value, allow_nan=False))
    run_meta_sha256 = hashlib.sha256(run_meta_path.read_bytes()).hexdigest()

    result = candidate_preview.build_main_preview(
        records_path, tmp_path / "candidate_qa",
        tmp_path / "candidate_qa_report",
        run_meta_sha256=run_meta_sha256)

    assert result["headline_eligible"] is False
    assert result["coverage"] == {
        task_id: 0 for task_id in benchmark.ABC_CANDIDATE_TASK_IDS}


def test_preview_cli_is_main_records_only_and_repeatable():
    from scripts.build_v16_candidate_preview import build_arg_parser

    args = build_arg_parser().parse_args([
        "--records", "/run/main-r00/records.jsonl",
        "--records", "/run/main-r01/records.jsonl",
        "--run-meta-sha256", "1" * 64,
        "--run-meta-sha256", "2" * 64,
        "--a1-common-support",
        "--output", "/run/candidate_qa",
        "--report", "/run/candidate_qa_report",
    ])
    assert args.records == [
        Path("/run/main-r00/records.jsonl"),
        Path("/run/main-r01/records.jsonl")]
    assert args.run_meta_sha256 == ["1" * 64, "2" * 64]
    assert args.a1_common_support is True
    assert not hasattr(args, "chain_records")

    default_args = build_arg_parser().parse_args([
        "--records", "/run/main-r00/records.jsonl",
        "--run-meta-sha256", "1" * 64,
        "--output", "/run/candidate_qa",
        "--report", "/run/candidate_qa_report",
    ])
    assert default_args.a1_common_support is None

    disabled_args = build_arg_parser().parse_args([
        "--records", "/run/main-r00/records.jsonl",
        "--run-meta-sha256", "1" * 64,
        "--no-a1-common-support",
        "--output", "/run/candidate_qa",
        "--report", "/run/candidate_qa_report",
    ])
    assert disabled_args.a1_common_support is False


@pytest.mark.parametrize(("option", "value"), [
    ("--future-view-gate", "/run/retired-gate.json"),
    ("--future-view-gate-sha256", "0" * 64),
    ("--future-view-gate-authority-id", "retired-gate"),
])
def test_candidate_build_cli_rejects_retired_c1_gate(option, value):
    """Catches exposing the matched/appearance-gated C1 compiler again."""
    from scripts.build_v16_candidate_preview import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([
            option, value,
            "--records", "/run/main-r00/records.jsonl",
            "--run-meta-sha256", "1" * 64,
            "--output", "/run/candidate_qa",
            "--report", "/run/candidate_qa_report",
        ])


def test_preview_cli_runs_directly_from_repository_root():
    script = Path(__file__).resolve().parents[1] / \
        "scripts" / "build_v16_candidate_preview.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=script.parents[1],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--records" in result.stdout
    assert "--chain-records" not in result.stdout


def test_candidate_exact_scorer_uses_equal_abc_heads(tmp_path):
    del tmp_path
    by_task, by_head, overall = candidate_preview._aggregate_candidate_scores({
        task_id: [1.0] for task_id in benchmark.ABC_CANDIDATE_TASK_IDS})
    assert by_task == {
        task_id: 1.0 for task_id in benchmark.ABC_CANDIDATE_TASK_IDS}
    assert by_head == {head: 1.0 for head in "ABC"}
    assert overall == 1.0


def test_candidate_scorer_reports_unweighted_six_task_macro_separately():
    by_task = dict(zip(
        benchmark.ABC_CANDIDATE_TASK_IDS,
        (1.0, 0.0, 0.5, 1.0, 0.5, 0.0)))

    assert candidate_preview._six_task_macro(by_task) == 0.5
    by_task["C1_future_view_selection"] = None
    assert candidate_preview._six_task_macro(by_task) is None


def test_candidate_exact_scorer_does_not_renormalize_missing_task(tmp_path):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    report = _evaluate_preview_artifact(
        artifact, gt_as_pred=True)
    assert report["by_task"]["A1_collision"] == 1.0
    assert report["by_head"] == {head: None for head in "ABC"}
    assert report["overall"] is None
    assert report["six_task_macro"] is None
    assert report["diagnostics"]["chance"]["A1_collision"] == {
        "mean_random_chance": 0.5,
        "chance_normalized_accuracy": 1.0,
    }
    assert report["diagnostics"]["clustered_uncertainty"][
        "A1_collision"]["scene"]["cluster_count"] == 1
    assert report["diagnostics"]["clustered_uncertainty"][
        "A1_collision"]["scene"]["interval_95"] is None


def test_candidate_scorer_reuses_the_build_validation(tmp_path, monkeypatch):
    artifact = tmp_path / "candidate_qa"
    _write_preview_artifact(
        _one_a1_projection(tmp_path), artifact,
        source_records_path=tmp_path / "records.jsonl")
    validated = _validate_preview_artifact(artifact)
    monkeypatch.setattr(
        candidate_preview, "validate_preview_artifact",
        lambda *_args, **_kwargs: pytest.fail(
            "the scorer must not repeat a supplied build validation"))

    report = candidate_preview.evaluate_preview_artifact(
        artifact, gt_as_pred=True, _validated_benchmark=validated)

    assert report["by_task"]["A1_collision"] == 1.0


def test_clustered_bootstrap_resamples_clusters_not_individual_questions():
    rows = [
        {"score": 1.0, "scene_id": "scene-a"},
        {"score": 1.0, "scene_id": "scene-a"},
        {"score": 0.0, "scene_id": "scene-b"},
    ]

    summary = candidate_preview._clustered_bootstrap_summary(
        rows, cluster_field="scene_id", resamples=1000, seed=11)

    assert summary["cluster_count"] == 2
    assert summary["item_count"] == 3
    assert summary["point_estimate"] == pytest.approx(2 / 3)
    assert summary["interval_95"][0] <= summary["point_estimate"]
    assert summary["interval_95"][1] >= summary["point_estimate"]


def test_exact_group_metric_requires_every_member_correct():
    rows = [
        {"group": "f1", "score": 1.0},
        {"group": "f1", "score": 0.0},
        {"group": "f2", "score": 1.0},
        {"group": "f2", "score": 1.0},
    ]

    assert candidate_preview._exact_group_summary(
        rows, group_field="group", required_size=2) == {
            "group_count": 2,
            "exact_accuracy": 0.5,
        }


def test_shortcut_audit_reports_actual_b1_numeric_rank_baselines():
    items = []
    answers = {}
    for truth_rank in range(1, 5):
        item_id = f"b1-{truth_rank}"
        choices = [{
            "id": f"choice_{index}", "value_m": float(index),
        } for index in range(1, 5)]
        items.append({
            "id": item_id, "task_id": "B1_endpoint_distance",
            "choices": choices,
        })
        answers[item_id] = {
            "canonical_answer": f"choice_{truth_rank}",
        }

    audit = candidate_preview._shortcut_audit(items, answers)

    assert audit["B1_numeric_rank"]["counts"] == {
        "1": 1, "2": 1, "3": 1, "4": 1}
    assert audit["B1_numeric_rank"][
        "best_constant_rank_accuracy"] == 0.25
    assert audit["B1_numeric_rank"][
        "best_middle_rank_accuracy"] == 0.25


def test_private_input_asset_identity_is_checkout_independent():
    left = [{
        "id": "a-1",
        "input_asset": {
            "marked": False,
            "path": "/checkout-one/run/img/source.png",
            "raw_path": "/checkout-one/run/img/source.png",
            "record_image_path": "img/source.png",
        },
    }, {
        "id": "b-1",
        "input_asset": {
            "marked": True,
            "path": "/tmp/build-one/marked_inputs/b-1.png",
            "raw_path": "/checkout-one/run/img/source.png",
            "record_image_path": "img/source.png",
        },
    }]
    right = copy.deepcopy(left)
    right[0]["input_asset"].update({
        "path": "/checkout-two/run/img/source.png",
        "raw_path": "/checkout-two/run/img/source.png",
    })
    right[1]["input_asset"].update({
        "path": "/tmp/build-two/marked_inputs/b-1.png",
        "raw_path": "/checkout-two/run/img/source.png",
    })

    candidate_preview._canonicalize_private_input_assets(left)
    candidate_preview._canonicalize_private_input_assets(right)

    assert left == right
    assert left[0]["input_asset"]["path"] == "img/source.png"
    assert left[1]["input_asset"]["path"] == "marked_inputs/b-1.png"
    assert left[1]["input_asset"]["raw_path"] == "img/source.png"
