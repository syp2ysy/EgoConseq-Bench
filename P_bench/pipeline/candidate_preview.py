"""Build an honest ABC candidate benchmark from registered main records.

This module is deliberately narrower than the formal Task-6 compiler.  It
projects only facts already authenticated in ``main`` records, persists the
same public/private/atom split planned for v16, and marks missing task families
as withheld.  It never routes through the legacy Q1--Q10 compiler.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable

from pipeline import action_proposal, benchmark
from pipeline import a1_common_support as a1_common_support_module
from pipeline import benchmark_builders
from pipeline import benchmark_tasks
from pipeline import c1_counterfactual
from pipeline import candidate_sources
from pipeline import diversity_selection
from pipeline.candidate_sources import (
    canonical_record_image_path as _canonical_record_image_path,
    canonical_sha256 as _canonical_sha256,
    canonicalize_private_input_assets as _canonicalize_private_input_assets,
    collect_record_context as _collect_record_context,
    record_context as _record_context,
    resolve_record_asset as _resolve_record_asset,
    source_atom as _source_atom,
    validate_record_contexts as _validate_record_contexts,
)
from pipeline import gate_authority
from pipeline import io_utils
from pipeline import qa_reasons
from pipeline import record as record_fields
from pipeline import viz
from pipeline.evaluation_diagnostics import (
    a1_safety_diagnostics as _a1_safety_diagnostics,
    b1_numeric_diagnostics as _b1_numeric_diagnostics,
    chance_diagnostics as _chance_diagnostics,
    clustered_bootstrap_summary as _clustered_bootstrap_summary,
    exact_group_summary as _exact_group_summary,
    joint_task_summary as _joint_task_summary,
)


PREVIEW_SCHEMA = "egoconseq.qa.v16-candidate-preview"
PREVIEW_ARTIFACT_SCHEMA = "egoconseq.candidate-preview.v2"
_PRIVATE_ARTIFACT_FILES = (
    "answers.jsonl", "atoms.jsonl", "record_contexts.jsonl",
    "source_map.json")
HEAD_TASKS = {
    "A": tuple(benchmark.A_TASK_QUESTIONS),
    "B": tuple(benchmark.B_TASK_QUESTIONS),
    "C": tuple(benchmark.C_TASK_QUESTIONS),
}
TASK_IDS = tuple(
    task_id for head in "ABC" for task_id in HEAD_TASKS[head])
TASK_HEAD = {
    task_id: head for head, task_ids in HEAD_TASKS.items()
    for task_id in task_ids
}
TASK_QUESTIONS = {
    **benchmark.A_TASK_QUESTIONS,
    **benchmark.B_TASK_QUESTIONS,
    **benchmark.C_TASK_QUESTIONS,
}


def _reason_count(rejections: dict, task_id: str, reason: str,
                  count: int = 1) -> None:
    bucket = rejections[task_id]
    frozen_reason = qa_reasons.require_report_reason(reason)
    bucket[frozen_reason] = int(bucket.get(frozen_reason, 0)) + int(count)


def _public_projection(item: dict) -> dict:
    value = copy.deepcopy(item)
    value.pop("oracle_ref", None)
    value.pop("answer_format", None)
    task_metadata = copy.deepcopy(value.get("task_metadata") or {})
    task_metadata["answer_format"] = "closed_exact"
    return {
        "id": value["id"],
        "schema_version": PREVIEW_SCHEMA,
        "task_id": value["task_id"],
        "result_head": value["result_head"],
        "question": value["question"],
        "model_input": value["model_input"],
        "choices": value["choices"],
        "task_metadata": task_metadata,
    }


def compile_main_records(
        records: Iterable[dict], *, asset_root, build_root,
        max_items_per_task=None,
        source_validated_record_sha256=None) -> dict:
    """Project all eligible A/B/C questions from registered main records.

    ``asset_root`` is the record shard root containing ``img/`` and terminal
    assets.

    All task routing follows the active ordinary proposal provenance.
    """
    asset_root = Path(asset_root)
    build_root = Path(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    marker_root = build_root / "marked_inputs"
    items = []
    answers = []
    atoms_by_id = {}
    rejections = {task_id: {} for task_id in TASK_IDS}
    contexts_by_digest = {}
    seen_ids = set()
    record_count = 0
    outcome_count = 0
    built_counts = collections.Counter()

    def has_room(task_id: str) -> bool:
        return (max_items_per_task is None or
                built_counts[task_id] < int(max_items_per_task))

    for record in records:
        record_count += 1
        record_sha256 = _canonical_sha256(record)
        source_validated = source_validated_record_sha256 is not None
        if source_validated and source_validated_record_sha256.get(
                id(record)) != record_sha256:
            raise ValueError("source-validated record changed before projection")
        record_context = record_fields.json_value(_record_context(record))
        record_built = False
        image = _resolve_record_asset(asset_root, record.get("image_path"))
        if not image.is_file():
            raise ValueError(f"record RGB asset is missing: {image}")
        sensor = record.get("sensor")
        if not isinstance(sensor, dict):
            raise ValueError("record sensor resolution is invalid")
        raw_asset = viz.authenticate_raw_rgb_image(
            image, expected_sha256=io_utils.sha256_file(image),
            expected_resolution=sensor.get("resolution"))
        image_sha256 = raw_asset.sha256
        numbered_dot_cache = {}
        outcomes = record.get("outcomes")
        if not isinstance(outcomes, list):
            raise ValueError("record outcomes must be a list")
        if any(not isinstance(outcome, dict) for outcome in outcomes):
            raise ValueError("record outcome must be an object")
        shared_evidence_by_outcome = {
            id(outcome): benchmark.build_shared_visible_space_evidence(
                record, outcome, source_validated=source_validated)
            for outcome in outcomes
        }
        b_record_evidence = None
        c1_selection_index = None
        if has_room("C1_future_view_selection"):
            c1_selection_index = \
                c1_counterfactual.build_record_selection_index(
                    record, asset_root=asset_root,
                    shared_evidence_by_outcome=shared_evidence_by_outcome)
        for outcome in outcomes:
            outcome_count += 1
            atom = None
            built_for_atom = False
            b_outcome_evidence = None

            def source_atom() -> dict:
                nonlocal atom
                if atom is None:
                    atom = _source_atom(
                        record, outcome, record_sha256=record_sha256)
                return atom

            for task_id in benchmark.A_TASK_QUESTIONS:
                if task_id == "A1_collision":
                    source_reason = \
                        a1_common_support_module.v3_source_rejection(
                            record, outcome)
                    if source_reason is not None:
                        _reason_count(rejections, task_id, source_reason)
                        continue
                if not has_room(task_id):
                    continue
                eligibility_evidence = \
                    benchmark_tasks.build_a_candidate_evidence(
                        task_id, record, outcome,
                        shared_evidence=shared_evidence_by_outcome[id(outcome)])
                eligibility = eligibility_evidence.eligibility
                if not eligibility.eligible:
                    _reason_count(
                        rejections, task_id,
                        eligibility.reason or "ineligible")
                    continue
                item, private = benchmark_builders.build_a_candidate(
                    task_id=task_id, record=record, outcome=outcome,
                    image=str(image),
                    expected_raw_image_sha256=image_sha256,
                    asset_dir=marker_root, authenticated_rgb=raw_asset,
                    a_eligibility_evidence=eligibility_evidence)
                built_for_atom = True
                built_counts[task_id] += 1
                _append_projection(
                    items, answers, seen_ids, item, private,
                    source_atom()["id"])

            for task_id in benchmark.B_TASK_QUESTIONS:
                if not has_room(task_id):
                    continue
                if b_outcome_evidence is None:
                    if b_record_evidence is None:
                        b_record_evidence = \
                            benchmark_tasks.build_b_record_evidence(
                                record, source_validated=source_validated)
                    b_outcome_evidence = \
                        benchmark_tasks.build_b_outcome_evidence(
                            record, outcome,
                            shared_evidence=shared_evidence_by_outcome[
                                id(outcome)],
                            record_evidence=b_record_evidence,
                            source_validated=source_validated)
                eligibility = benchmark_tasks.b_candidate_eligibility(
                    task_id, record, outcome,
                    evidence=b_outcome_evidence)
                if not eligibility.eligible:
                    _reason_count(
                        rejections, task_id,
                        eligibility.reason or "ineligible")
                    continue
                item, private = benchmark_builders.build_b_candidate(
                    task_id=task_id, record=record, outcome=outcome,
                    image=str(image),
                    expected_raw_image_sha256=image_sha256,
                    asset_dir=marker_root, authenticated_rgb=raw_asset,
                    numbered_dot_cache=numbered_dot_cache,
                    b_outcome_evidence=b_outcome_evidence)
                built_for_atom = True
                built_counts[task_id] += 1
                _append_projection(
                    items, answers, seen_ids, item, private,
                    source_atom()["id"])

            if has_room("C1_future_view_selection"):
                selection, reason = benchmark_tasks.c_candidate_selection(
                    record, outcome, asset_root=asset_root,
                    selection_index=c1_selection_index)
                if selection is None:
                    _reason_count(
                        rejections, "C1_future_view_selection", reason)
                else:
                    item, private = benchmark_builders.build_c_candidate(
                        record=record, outcome=outcome, image=str(image),
                        expected_raw_image_sha256=image_sha256,
                        asset_root=asset_root, selection=selection,
                        authenticated_rgb=raw_asset)
                    built_for_atom = True
                    built_counts["C1_future_view_selection"] += 1
                    _append_projection(
                        items, answers, seen_ids, item, private,
                        source_atom()["id"])

            if built_for_atom:
                record_built = True
                if atom is None:
                    raise AssertionError("built outcome has no source atom")
                atoms_by_id.setdefault(atom["id"], atom)

        if record_built:
            _collect_record_context(
                contexts_by_digest, record, record_sha256=record_sha256,
                context=record_context)

    return {
        "items": items,
        "answers": answers,
        "atoms": [atoms_by_id[key] for key in sorted(atoms_by_id)],
        "record_contexts": [
            {"record_sha256": digest, "context": contexts_by_digest[digest]}
            for digest in sorted(contexts_by_digest)],
        "rejections": rejections,
        "source_counts": {
            "records": record_count, "outcomes": outcome_count,
        },
    }


def merge_projections(*projections: dict) -> dict:
    """Merge independently authenticated main-record projections."""
    merged = {
        "items": [], "answers": [], "atoms": [], "record_contexts": [],
        "rejections": {task_id: {} for task_id in TASK_IDS},
        "source_counts": {},
    }
    atom_by_id = {}
    context_by_digest = {}
    for projection in projections:
        merged["items"].extend(copy.deepcopy(projection.get("items") or []))
        merged["answers"].extend(
            copy.deepcopy(projection.get("answers") or []))
        for atom in projection.get("atoms") or []:
            atom_by_id.setdefault(atom["id"], copy.deepcopy(atom))
        for row in projection.get("record_contexts") or []:
            digest = str(row["record_sha256"])
            context = copy.deepcopy(row["context"])
            previous = context_by_digest.get(digest)
            if previous is not None and previous != context:
                raise ValueError(
                    f"two distinct record contexts share sha256 {digest}")
            context_by_digest[digest] = context
        for task_id in TASK_IDS:
            for reason, count in (
                    (projection.get("rejections") or {}).get(
                        task_id, {}).items()):
                bucket = merged["rejections"][task_id]
                bucket[reason] = int(bucket.get(reason, 0)) + int(count)
        for key, count in (projection.get("source_counts") or {}).items():
            merged["source_counts"][key] = int(
                merged["source_counts"].get(key, 0)) + int(count)
    merged["atoms"] = [atom_by_id[key] for key in sorted(atom_by_id)]
    merged["record_contexts"] = [
        {"record_sha256": digest, "context": context_by_digest[digest]}
        for digest in sorted(context_by_digest)]
    return merged


def _append_projection(items, answers, seen_ids, item, private, atom_ref):
    item_id = str(item["id"])
    if item_id in seen_ids:
        raise ValueError(f"duplicate candidate item id: {item_id}")
    seen_ids.add(item_id)
    public = _public_projection(item)
    answer = copy.deepcopy(private)
    answer["schema_version"] = PREVIEW_SCHEMA
    answer["atom_ref"] = atom_ref
    items.append(public)
    answers.append(answer)


def _write_jsonl(path: Path, values: Iterable[dict]) -> None:
    text = "".join(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
        for value in values)
    io_utils.atomic_write_text(path, text)


def _copy_public_asset(
        source_value: object, expected_sha256: object, *,
        artifact_root: Path,
        staged_assets: dict[tuple[str, str], str]) -> str:
    source = Path(str(source_value))
    expected = str(expected_sha256)
    cache_key = (str(source), expected)
    if cache_key in staged_assets:
        return staged_assets[cache_key]
    if not source.is_file():
        raise ValueError(f"public image asset is missing: {source}")
    actual = io_utils.sha256_file(source)
    if actual != expected:
        raise ValueError("public image asset digest changed")
    relative = Path("public") / "images" / f"{actual}.png"
    destination = artifact_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if io_utils.sha256_file(destination) != actual:
            raise ValueError("content-addressed public asset is inconsistent")
    else:
        linked = io_utils.link_or_copy_file(source, destination)
        if not linked and io_utils.sha256_file(destination) != actual:
            destination.unlink(missing_ok=True)
            raise ValueError("copied public image asset digest changed")
    staged_assets[cache_key] = relative.as_posix()
    return staged_assets[cache_key]


def _relocate_public_assets(items: list[dict], artifact_root: Path) -> None:
    staged_assets: dict[tuple[str, str], str] = {}
    for item in items:
        model_input = item["model_input"]
        model_input["initial_rgb"] = _copy_public_asset(
            model_input["initial_rgb"],
            model_input["initial_rgb_sha256"],
            artifact_root=artifact_root, staged_assets=staged_assets)
        for choice in item.get("choices") or []:
            if "image" in choice:
                choice["image"] = _copy_public_asset(
                    choice["image"], choice["image_sha256"],
                    artifact_root=artifact_root,
                    staged_assets=staged_assets)
            for frame in choice.get("frames") or []:
                frame["image"] = _copy_public_asset(
                    frame["image"], frame["image_sha256"],
                    artifact_root=artifact_root,
                    staged_assets=staged_assets)


def _coverage(items: Iterable[dict], task_ids=TASK_IDS) -> dict[str, int]:
    counts = collections.Counter(value["task_id"] for value in items)
    return {task_id: int(counts.get(task_id, 0)) for task_id in task_ids}


def _task_status(
        coverage: dict[str, int],
        rejections: dict | None = None, *, task_ids=None) -> dict[str, str]:
    rejections = rejections or {}
    task_ids = tuple(task_ids or coverage)
    return {
        task_id: (
            "available_real" if coverage[task_id] else
            "withheld_no_eligible_real_case")
        for task_id in task_ids
    }


def _candidate_distributions(
        items: Iterable[dict], answers_by_id: dict[str, dict]) -> dict:
    fields = collections.defaultdict(
        lambda: {
            "answers": collections.Counter(),
            "action_counts": collections.Counter(),
            "body_radius_m": collections.Counter(),
            "camera_height_m": collections.Counter(),
            "fov_deg": collections.Counter(),
        })
    for item in items:
        task_id = item["task_id"]
        model_input = item.get("model_input") or {}
        row = fields[task_id]
        row["answers"][str(
            answers_by_id[item["id"]]["canonical_answer"])] += 1
        row["action_counts"][str(len(model_input.get("actions") or []))] += 1
        row["body_radius_m"][str(float(
            model_input["body_radius_m"]))] += 1
        row["camera_height_m"][str(float(
            model_input["camera_height_above_visible_floor_m"]))] += 1
        row["fov_deg"][
            f"{float(model_input['hfov_deg'])}x"
            f"{float(model_input['vfov_deg'])}"] += 1
    return {
        task_id: {
            name: dict(sorted(counter.items()))
            for name, counter in values.items()
        }
        for task_id, values in sorted(fields.items())
    }


def _shortcut_audit(items: Iterable[dict],
                    answers_by_id: dict[str, dict]) -> dict:
    """Report empirical blind baselines from the compiled test artifact."""
    items = list(items)
    answers_by_task = collections.defaultdict(collections.Counter)
    ranks = collections.Counter()
    answer_positions = collections.Counter()
    b1_count = 0
    for item in items:
        answer = answers_by_id[item["id"]]["canonical_answer"]
        task_id = item["task_id"]
        answers_by_task[task_id][str(answer)] += 1
        if task_id != "B1_endpoint_distance":
            continue
        choices = item.get("choices") or []
        by_value = sorted(
            choices, key=lambda value: float(value["value_m"]))
        rank = next((index for index, choice in enumerate(by_value, 1)
                     if choice.get("id") == answer), None)
        position = next((index for index, choice in enumerate(choices, 1)
                         if choice.get("id") == answer), None)
        if rank is None or position is None:
            raise ValueError("B1 answer is absent from its metric choices")
        ranks[str(rank)] += 1
        answer_positions[str(position)] += 1
        b1_count += 1

    majority = {}
    for task_id, counts in sorted(answers_by_task.items()):
        total = sum(counts.values())
        majority[task_id] = {
            "item_count": total,
            "answer_counts": dict(sorted(counts.items())),
            "majority_answer_accuracy": (
                max(counts.values()) / total if total else None),
        }
    return {
        "schema": "egoconseq.shortcut-audit.v1",
        "task_answer_frequency": majority,
        "B1_numeric_rank": {
            "item_count": b1_count,
            "counts": dict(sorted(ranks.items())),
            "best_constant_rank_accuracy": (
                max(ranks.values()) / b1_count if b1_count else None),
            "best_middle_rank_accuracy": (
                max(ranks.get("2", 0), ranks.get("3", 0)) / b1_count
                if b1_count else None),
            "answer_position_counts": dict(sorted(answer_positions.items())),
            "best_constant_answer_position_accuracy": (
                max(answer_positions.values()) / b1_count
                if b1_count else None),
        },
    }


def write_preview_artifact(
        projection: dict, artifact_root, *, source_records_path,
        source_run_meta_sha256=None, collection_funnel=None,
        expected_source_authority=None) -> dict:
    """Persist an explicitly non-headline preview bound to local source rows."""
    artifact_root = Path(artifact_root)
    if artifact_root.exists():
        raise FileExistsError(f"preview output already exists: {artifact_root}")
    (artifact_root / "public").mkdir(parents=True)
    (artifact_root / "private").mkdir(parents=True)
    items = [copy.deepcopy(value) for value in projection.get("items", [])]
    answers = [copy.deepcopy(value) for value in projection.get("answers", [])]
    atoms = [copy.deepcopy(value) for value in projection.get("atoms", [])]
    task_ids = TASK_IDS
    order = {task_id: index for index, task_id in enumerate(task_ids)}
    items.sort(key=lambda value: (order[value["task_id"]], value["id"]))
    answers_by_id = {value["id"]: value for value in answers}
    if len(answers_by_id) != len(answers):
        raise ValueError("private answer ids are not unique")
    if set(answers_by_id) != {value["id"] for value in items}:
        raise ValueError("public/private candidate ids are not bijective")
    atoms_by_id = {value["id"]: value for value in atoms}
    if len(atoms_by_id) != len(atoms):
        raise ValueError("source atom ids are not unique")
    for item in items:
        task_id = item.get("task_id")
        if task_id not in task_ids or item.get("result_head") != TASK_HEAD[task_id]:
            raise ValueError("candidate task catalog binding is invalid")
        answer = answers_by_id[item["id"]]
        if answer.get("task_id") != task_id:
            raise ValueError("public/private task binding is invalid")
        if answer.get("atom_ref") not in atoms_by_id:
            raise ValueError("private answer source atom is missing")
    _canonicalize_private_input_assets(answers)
    _relocate_public_assets(items, artifact_root)
    ordered_answers = [answers_by_id[item["id"]] for item in items]
    ordered_atoms = [atoms_by_id[key] for key in sorted(atoms_by_id)]
    _write_jsonl(artifact_root / "public" / "items.jsonl", items)
    _write_jsonl(
        artifact_root / "private" / "answers.jsonl", ordered_answers)
    _write_jsonl(artifact_root / "private" / "atoms.jsonl", ordered_atoms)
    _write_jsonl(
        artifact_root / "private" / "record_contexts.jsonl",
        projection.get("record_contexts") or [])

    if isinstance(source_records_path, (list, tuple)):
        source_paths = [Path(value) for value in source_records_path]
    else:
        source_paths = [Path(source_records_path)]
    if source_run_meta_sha256 is None:
        run_meta_digests = [None] * len(source_paths)
    elif isinstance(source_run_meta_sha256, (list, tuple)):
        run_meta_digests = [str(value) for value in source_run_meta_sha256]
    else:
        run_meta_digests = [str(source_run_meta_sha256)]
    if len(run_meta_digests) != len(source_paths):
        raise ValueError("source records/run metadata authority count differs")
    if not isinstance(
            expected_source_authority,
            gate_authority.ResolvedPreviewSourceAuthority):
        raise ValueError("external source authority required")
    source_binding = expected_source_authority.binding()
    publication_selection = dict(
        projection.get("publication_selection") or {})
    if publication_selection not in (
            {}, {"A1_collision": a1_common_support_module.POLICY},
            {"A1_collision": a1_common_support_module.V3_POLICY}):
        raise ValueError("candidate publication selection policy is invalid")
    selection_marker = (
        {"publication_selection": publication_selection}
        if publication_selection else {})
    diversity_report = dict(projection.get("diversity_selection") or {})
    diversity_policy = diversity_report.get("policy")
    if diversity_policy not in (None, diversity_selection.POLICY):
        raise ValueError("candidate diversity selection policy is invalid")
    diversity_marker = (
        {"diversity_selection": {"policy": diversity_policy}}
        if diversity_policy else {})
    expected_paths = [str(gate_authority.lexical_absolute_path(path))
                      for path in source_paths]
    resolved_sources = expected_source_authority.sources
    if ([source["path"] for source in resolved_sources] != expected_paths or
            [source["run_meta_sha256"] for source in resolved_sources] !=
            run_meta_digests):
        raise ValueError(
            "source artifact inputs differ from external authority")
    source_map = {
        "schema": PREVIEW_ARTIFACT_SCHEMA,
        "source_authority": source_binding,
        "source_counts": dict(projection.get("source_counts") or {}),
        **selection_marker,
        **diversity_marker,
    }
    io_utils.atomic_write_json(
        artifact_root / "private" / "source_map.json", source_map,
        allow_nan=False)
    coverage = _coverage(items, task_ids)
    rejections = {
        task_id: dict((projection.get("rejections") or {}).get(
            task_id, {})) for task_id in task_ids
    }
    qa_reasons.validate_report_rejections(
        rejections, task_ids=tuple(task_ids))
    status = _task_status(coverage, rejections, task_ids=task_ids)
    cases = [{
        "public": item,
        "canonical_answer": answers_by_id[item["id"]]["canonical_answer"],
        "oracle_ref": answers_by_id[item["id"]].get("oracle_ref"),
        "atom_ref": answers_by_id[item["id"]]["atom_ref"],
    } for item in items]
    benchmark_json = {
        "schema": PREVIEW_ARTIFACT_SCHEMA,
        "candidate_only": True,
        "headline_eligible": False,
        "task_scope": "ABC",
        "task_catalog": list(task_ids),
        "task_definitions": {
            task_id: {
                "result_head": TASK_HEAD[task_id],
                "question": TASK_QUESTIONS[task_id],
            } for task_id in task_ids
        },
        "coverage": coverage,
        "task_status": status,
        "cases": cases,
    }
    io_utils.atomic_write_json(
        artifact_root / "benchmark.json", benchmark_json, allow_nan=False)
    report = {
        "schema": PREVIEW_ARTIFACT_SCHEMA,
        "candidate_only": True,
        "headline_eligible": False,
        "coverage": coverage,
        "task_status": status,
        "source_counts": dict(projection.get("source_counts") or {}),
        **selection_marker,
        **({"diversity_selection": diversity_report}
           if diversity_report else {}),
        "distributions": _candidate_distributions(items, answers_by_id),
        "shortcut_audit": _shortcut_audit(items, answers_by_id),
        "rejections": rejections,
        "collection_funnel": collection_funnel,
    }
    io_utils.atomic_write_json(
        artifact_root / "report.json", report, allow_nan=False)
    public_manifest = {
        "schema": PREVIEW_ARTIFACT_SCHEMA,
        "candidate_only": True,
        "headline_eligible": False,
        "public_input_contract": benchmark.PUBLIC_INPUT_CONTRACT,
        "num_items": len(items),
        "files": {
            "items.jsonl": io_utils.sha256_file(
                artifact_root / "public" / "items.jsonl"),
        },
        "assets": {
            path.relative_to(artifact_root).as_posix(): io_utils.sha256_file(path)
            for path in sorted((artifact_root / "public" / "images").glob("*"))
        },
    }
    private_manifest = {
        "schema": PREVIEW_ARTIFACT_SCHEMA,
        "candidate_only": True,
        "files": {
            name: io_utils.sha256_file(artifact_root / "private" / name)
            for name in _PRIVATE_ARTIFACT_FILES
        },
    }
    io_utils.atomic_write_json(
        artifact_root / "public" / "manifest.json", public_manifest,
        allow_nan=False)
    io_utils.atomic_write_json(
        artifact_root / "private" / "manifest.json", private_manifest,
        allow_nan=False)
    return benchmark_json


def _read_jsonl(path: Path) -> list[dict]:
    return io_utils.read_jsonl(path, require_dict=True)


def _validate_preview_manifests(artifact_root: Path) -> dict[str, str]:
    public = json.loads((artifact_root / "public" / "manifest.json").read_text())
    private = json.loads(
        (artifact_root / "private" / "manifest.json").read_text())
    if public.get("schema") != PREVIEW_ARTIFACT_SCHEMA or \
            private.get("schema") != PREVIEW_ARTIFACT_SCHEMA:
        raise ValueError("preview manifest schema changed")
    if public.get("public_input_contract") != benchmark.PUBLIC_INPUT_CONTRACT:
        raise ValueError("public preview input contract changed")
    expected_public = {"items.jsonl": io_utils.sha256_file(
        artifact_root / "public" / "items.jsonl")}
    if public.get("files") != expected_public:
        raise ValueError("public preview manifest digest changed")
    image_root = artifact_root / "public" / "images"
    expected_assets = {
        path.relative_to(artifact_root).as_posix(): io_utils.sha256_file(path)
        for path in sorted(image_root.glob("*")) if path.is_file()
    }
    if public.get("assets") != expected_assets:
        raise ValueError("public preview asset manifest changed")
    expected_private = {
        name: io_utils.sha256_file(artifact_root / "private" / name)
        for name in _PRIVATE_ARTIFACT_FILES
    }
    if private.get("files") != expected_private:
        raise ValueError("private preview manifest digest changed")
    return expected_assets


def _source_records(
        source_map: dict, expected_source_authority, *, source_snapshots=None,
        ) -> dict[tuple[str, str], tuple[dict, dict, Path]]:
    if source_map.get("schema") != PREVIEW_ARTIFACT_SCHEMA:
        raise ValueError("preview source map schema changed")
    source_authority = source_map.get("source_authority")
    if not isinstance(source_authority, dict):
        raise ValueError("preview source authority binding is missing")
    by_key = {}
    records_seen = outcomes_seen = 0
    persisted_sources = source_authority.get("sources") or []
    resolved_sources = expected_source_authority.sources
    if len(persisted_sources) != len(resolved_sources):
        raise ValueError("preview source authority count changed")
    snapshots = None
    if source_snapshots is not None:
        snapshots = tuple(source_snapshots)
        if len(snapshots) != len(resolved_sources):
            raise ValueError("preview source snapshot count changed")
    for source_index, (binding, source) in enumerate(zip(
            persisted_sources, resolved_sources)):
        path = Path(str(source.get("path")))
        if not path.is_absolute() or not path.is_file():
            raise ValueError("preview source records are unavailable")
        run_meta_digest = binding.get("run_meta_sha256")
        records = None
        if run_meta_digest is not None:
            snapshot = (
                snapshots[source_index] if snapshots is not None else
                candidate_sources.load_candidate_source(
                    path, expected_run_meta_sha256=run_meta_digest,
                    source_authority=expected_source_authority,
                    source_index=source_index))
            if (snapshot.source_index != source_index or
                    snapshot.records_path != path or
                    snapshot.records_sha256 != binding.get("records_sha256") or
                    snapshot.run_meta_sha256 != run_meta_digest):
                raise ValueError("preview source snapshot binding changed")
            records = snapshot.records
        else:
            if io_utils.sha256_file(path) != binding.get("records_sha256"):
                raise ValueError("preview source records digest changed")
            records = record_fields.decode_records_for_schema(
                path.read_bytes(), record_fields.SCHEMA_VERSION)
        for record in records:
            records_seen += 1
            record_sha256 = candidate_sources.canonical_sha256(record)
            for outcome in record.get("outcomes") or []:
                outcomes_seen += 1
                key = (str(record.get("frame_id")),
                       str(outcome.get("outcome_id")))
                if key in by_key:
                    raise ValueError("preview source outcome identity repeats")
                by_key[key] = (
                    record, outcome, path.parent, record_sha256)
    expected_counts = {"records": records_seen, "outcomes": outcomes_seen}
    if source_map.get("source_counts") != expected_counts:
        raise ValueError("preview source counts changed")
    return by_key


def _validate_source_atoms(atoms: list[dict], source_rows: dict) -> dict:
    atom_by_id = {}
    for atom in atoms:
        key = (str(atom.get("frame_id")), str(atom.get("outcome_id")))
        source = source_rows.get(key)
        if source is None:
            raise ValueError("preview source atom has no source outcome")
        expected = _source_atom(
            source[0], source[1], record_sha256=source[3])
        if atom != expected:
            raise ValueError("preview source atom content changed")
        atom_id = atom.get("id")
        if atom_id in atom_by_id:
            raise ValueError("preview source atom identity repeats")
        atom_by_id[atom_id] = atom
    return atom_by_id


def _choice_ids(item: dict) -> list[str]:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("preview public choices are empty")
    ids = [str(value.get("id")) for value in choices
           if isinstance(value, dict)]
    if len(ids) != len(choices) or len(set(ids)) != len(ids):
        raise ValueError("preview public choice ids are invalid")
    return ids


def _validate_public_from_atom(
        item: dict, answer: dict, atom: dict, contexts: dict, *,
        b_record_evidence=None) -> None:
    record = contexts[atom["record_sha256"]]
    outcome = atom["outcome"]
    sensor = record.get("sensor") or {}
    expected = {
        "camera_height_above_visible_floor_m": round(
            float(record["camera_height_above_visible_floor_m"]),
            benchmark.PUBLIC_HEIGHT_DECIMALS),
        "hfov_deg": float(sensor["hfov_deg"]),
        "vfov_deg": float(sensor["vfov_deg"]),
        "body_radius_m": float(outcome["body"]["radius_m"]),
        "actions": record_fields.json_value(outcome.get("actions") or []),
    }
    task_id = item["task_id"]
    if task_id in benchmark.B_TASK_QUESTIONS:
        target, reason = b_record_evidence.require(record)
        if target is None:
            raise ValueError(f"public B target is not authenticated: {reason}")
        expected["target"] = target["selection"]["name"]
        expected_question = benchmark.B_TASK_QUESTIONS[task_id].format(
            target=expected["target"])
    else:
        expected_question = TASK_QUESTIONS[task_id]
    model_input = copy.deepcopy(item.get("model_input") or {})
    initial_sha256 = model_input.pop("initial_rgb_sha256", None)
    model_input.pop("initial_rgb", None)
    if model_input != expected or item.get("question") != expected_question:
        raise ValueError("public model input differs from its source atom")
    input_asset = answer.get("input_asset") or {}
    if (input_asset.get("sha256") != initial_sha256 or
            input_asset.get("record_image_path") != record.get("image_path")):
        raise ValueError("public initial image differs from its source atom")
    record_image_path = _canonical_record_image_path(record.get("image_path"))
    expected_input_path = (
        f"marked_inputs/{item['id']}.png"
        if input_asset.get("marked") is True else record_image_path)
    if input_asset.get("path") != expected_input_path:
        raise ValueError("private input asset path is not canonical")
    if ("raw_path" in input_asset and
            input_asset.get("raw_path") != record_image_path):
        raise ValueError("private raw input asset path is not canonical")
    task_metadata = {"answer_format": "closed_exact"}
    if task_id in benchmark.B_TASK_QUESTIONS:
        geometry = record["b_target"]["geometry"]
        task_metadata["target_geometry_protocol"] = str(
            geometry[
                "ground_support" if task_id == "B1_endpoint_distance"
                else "reference_centroid"]["protocol"])
    if item.get("task_metadata") != task_metadata:
        raise ValueError("public response format changed")


def _validate_answer_from_atom(
        item: dict, answer: dict, atom: dict, source_record: dict,
        contexts: dict, *, source_outcome: dict,
        source_asset_root: Path, shared_evidence=None,
        b_outcome_evidence=None) -> None:
    task_id = item["task_id"]
    canonical = str(answer.get("canonical_answer"))
    choice_ids = _choice_ids(item)
    if canonical not in choice_ids:
        raise ValueError("canonical answer is not a public choice")
    if answer.get("task_id") != task_id:
        raise ValueError("preview public/private task binding changed")
    oracle = answer.get("oracle_ref") or {}
    if (str(oracle.get("frame_id")) != str(atom.get("frame_id")) or
            str(oracle.get("outcome_id")) != str(atom.get("outcome_id"))):
        raise ValueError("preview oracle/source atom binding changed")
    record = contexts[atom["record_sha256"]]
    outcome = atom["outcome"]
    if task_id in benchmark.A_TASK_QUESTIONS:
        evidence = benchmark_tasks.build_a_candidate_evidence(
            task_id, record, outcome, shared_evidence=shared_evidence)
        expected, raw_choices = benchmark_tasks.a_candidate_answer(
            task_id, record, outcome, evidence=evidence)
        expected_ids = [str(value.get("id")) if isinstance(value, dict)
                        else str(value) for value in raw_choices]
        if canonical != expected or choice_ids != expected_ids:
            raise ValueError("A answer differs from its authenticated atom")
        expected_oracle = {
            "frame_id": record["frame_id"],
            "outcome_id": outcome.get("outcome_id"),
            "base_rollout_key": outcome.get("base_rollout_key"),
            "shared_certificate_sha256":
                outcome["shared_oracle_stability"]["sha256"],
        }
    elif task_id in benchmark.B_TASK_QUESTIONS:
        expected, raw_choices, expected_certificate = \
            benchmark_tasks.b_candidate_answer(
                task_id, record, outcome, evidence=b_outcome_evidence)
        expected_ids = [str(value["id"]) for value in raw_choices]
        if (canonical != expected or choice_ids != expected_ids or
                item.get("choices") != raw_choices):
            raise ValueError("B answer differs from its authenticated atom")
        if answer.get("choice_certificate") != expected_certificate:
            raise ValueError(
                "B choice certificate differs from its authenticated atom")
        expected_oracle = {
            "frame_id": record["frame_id"],
            "outcome_id": outcome.get("outcome_id"),
            "base_rollout_key": outcome.get("base_rollout_key"),
            "shared_certificate_sha256":
                outcome["shared_oracle_stability"]["sha256"],
            "target_sha256": record["b_target"]["sha256"],
            "target_geometry_sha256":
                record["b_target"]["geometry"]["sha256"],
            "endpoint_relation_sha256":
                outcome["b_endpoint_relation"]["sha256"],
        }
    else:
        certificate = answer.get("selection_certificate") or {}
        if certificate.get("schema") != c1_counterfactual.SELECTION_SCHEMA:
            raise ValueError("C selection is not counterfactual")
        terminal = outcome.get("terminal_rgb_asset") or {}
        terminal_sha256 = terminal.get("png_sha256")
        matches = [
            str(choice["id"]) for choice in item["choices"]
            if choice.get("image_sha256") == terminal_sha256
        ]
        if matches != [canonical]:
            raise ValueError("C answer differs from its authenticated atom")
        certificate_payload = {
            key: value for key, value in certificate.items()
            if key != "sha256"
        }
        if (certificate.get("sha256") !=
                record_fields.canonical_atom_sha256(certificate_payload) or
                certificate.get("canonical_answer") != canonical or
                certificate.get("correct_outcome_id") != outcome.get(
                    "outcome_id")):
            raise ValueError("C selection certificate changed")
        if certificate.get("headline_eligible") is not False:
            raise ValueError(
                "counterfactual C selection claims headline eligibility")
        c1_counterfactual.validate_counterfactual_selection(
            source_record, source_outcome, certificate,
            asset_root=source_asset_root)
        source_outcomes = {
            value.get("outcome_id"): value
            for value in source_record.get("outcomes") or []
        }
        certified_choices = certificate.get("choices") or []
        if len(certified_choices) != len(item["choices"]):
            raise ValueError("C selection choices changed")
        for public_choice, certified in zip(
                item["choices"], certified_choices):
            sibling = source_outcomes.get(certified.get("outcome_id")) or {}
            sibling_terminal = sibling.get("terminal_rgb_asset") or {}
            if (certified.get("id") != public_choice.get("id") or
                    certified.get("terminal_rgb_sha256") !=
                    public_choice.get("image_sha256") or
                    sibling_terminal.get("png_sha256") !=
                    public_choice.get("image_sha256")):
                raise ValueError("C public choice differs from its source outcome")
        expected_oracle = {
            "frame_id": record["frame_id"],
            "outcome_id": outcome.get("outcome_id"),
            "base_rollout_key": outcome.get("base_rollout_key"),
            "shared_certificate_sha256":
                outcome["shared_oracle_stability"]["sha256"],
            "terminal_rgb_atom_sha256": terminal.get("sha256"),
            "future_view_selection_sha256": certificate.get("sha256"),
        }
    if oracle != expected_oracle:
        raise ValueError("preview oracle reference differs from source atom")


def validate_preview_artifact(
        artifact_root, *, expected_source_authority=None,
        source_snapshots=None) -> dict:
    """Validate every preview binding and return the joined benchmark JSON."""
    if not isinstance(
            expected_source_authority,
            gate_authority.ResolvedPreviewSourceAuthority):
        raise ValueError("external source authority required")
    artifact_root = Path(artifact_root)
    public_assets = _validate_preview_manifests(artifact_root)
    items = _read_jsonl(artifact_root / "public" / "items.jsonl")
    answers = _read_jsonl(artifact_root / "private" / "answers.jsonl")
    atoms = _read_jsonl(artifact_root / "private" / "atoms.jsonl")
    context_rows = _read_jsonl(
        artifact_root / "private" / "record_contexts.jsonl")
    benchmark_json = json.loads(
        (artifact_root / "benchmark.json").read_text(encoding="utf-8"))
    report = json.loads(
        (artifact_root / "report.json").read_text(encoding="utf-8"))
    qa_reasons.validate_report_rejections(
        report.get("rejections"), task_ids=TASK_IDS)
    source_map = json.loads(
        (artifact_root / "private" / "source_map.json").read_text(
            encoding="utf-8"))
    if source_map.get("source_authority") != \
            expected_source_authority.binding():
        raise ValueError("preview source authority binding changed")
    publication_selection = source_map.get("publication_selection") or {}
    if publication_selection not in (
            {}, {"A1_collision": a1_common_support_module.POLICY},
            {"A1_collision": a1_common_support_module.V3_POLICY}):
        raise ValueError("preview publication selection policy changed")
    if (report.get("publication_selection") or {}) != \
            publication_selection:
        raise ValueError("preview publication selection marker disagrees")
    diversity_marker = source_map.get("diversity_selection") or {}
    if diversity_marker not in (
            {}, {"policy": diversity_selection.POLICY}):
        raise ValueError("preview diversity selection marker changed")
    diversity_report = report.get("diversity_selection") or {}
    if bool(diversity_marker) != bool(diversity_report) or (
            diversity_report and
            diversity_report.get("policy") != diversity_selection.POLICY):
        raise ValueError("preview diversity selection report disagrees")
    task_ids = tuple(benchmark_json.get("task_catalog") or ())
    if task_ids != TASK_IDS:
        raise ValueError("preview task catalog is not exact ABC")
    if benchmark_json.get("task_scope") != "ABC":
        raise ValueError("preview task scope changed")
    if benchmark_json.get("headline_eligible") is not False:
        raise ValueError("candidate preview must not be headline eligible")
    answer_by_id = {value.get("id"): value for value in answers}
    if publication_selection == {
            "A1_collision": a1_common_support_module.POLICY}:
        a1_common_support_module.validate_selection(items, answer_by_id)
    elif publication_selection == {
            "A1_collision": a1_common_support_module.V3_POLICY}:
        a1_common_support_module.validate_v3_selection(
            items, answer_by_id, atoms, context_rows)
    source_rows = _source_records(
        source_map, expected_source_authority,
        source_snapshots=source_snapshots)
    atom_by_id = _validate_source_atoms(atoms, source_rows)
    contexts = _validate_record_contexts(
        context_rows, atom_by_id, source_rows)
    if (len(answer_by_id) != len(answers) or
            set(answer_by_id) != {value.get("id") for value in items}):
        raise ValueError("public/private candidate ids are not bijective")
    expected_cases = []
    shared_evidence_by_atom = {}
    b_record_evidence_by_digest = {}
    b_outcome_evidence_by_atom = {}
    for item in items:
        task_id = item.get("task_id")
        if task_id not in task_ids or item.get("result_head") != TASK_HEAD[task_id]:
            raise ValueError("preview task/head binding changed")
        answer = answer_by_id[item["id"]]
        atom = atom_by_id.get(answer.get("atom_ref"))
        if atom is None:
            raise ValueError("preview source atom binding changed")
        digest = atom["record_sha256"]
        record_context = contexts[digest]
        outcome = atom["outcome"]
        if atom["id"] not in shared_evidence_by_atom:
            shared_evidence_by_atom[atom["id"]] = \
                benchmark.build_shared_visible_space_evidence(
                    record_context, outcome, source_validated=True)
        shared_evidence = shared_evidence_by_atom[atom["id"]]
        b_record_evidence = None
        b_outcome_evidence = None
        if task_id in benchmark.B_TASK_QUESTIONS:
            if digest not in b_record_evidence_by_digest:
                b_record_evidence_by_digest[digest] = \
                    benchmark_tasks.build_b_record_evidence(
                        record_context, source_validated=True)
            b_record_evidence = b_record_evidence_by_digest[digest]
            if atom["id"] not in b_outcome_evidence_by_atom:
                b_outcome_evidence_by_atom[atom["id"]] = \
                    benchmark_tasks.build_b_outcome_evidence(
                        record_context, outcome, shared_evidence=shared_evidence,
                        record_evidence=b_record_evidence,
                        source_validated=True)
            b_outcome_evidence = b_outcome_evidence_by_atom[atom["id"]]
        _validate_public_from_atom(
            item, answer, atom, contexts,
            b_record_evidence=b_record_evidence)
        source_record, source_outcome, source_asset_root, _record_sha256 = \
            source_rows[
            (str(atom["frame_id"]), str(atom["outcome_id"]))]
        _validate_answer_from_atom(
            item, answer, atom, source_record, contexts,
            source_outcome=source_outcome,
            source_asset_root=source_asset_root,
            shared_evidence=shared_evidence,
            b_outcome_evidence=b_outcome_evidence)
        model_input = item.get("model_input") or {}
        _validate_asset_binding(
            public_assets, model_input.get("initial_rgb"),
            model_input.get("initial_rgb_sha256"))
        for choice in item.get("choices") or []:
            if "image" in choice:
                _validate_asset_binding(
                    public_assets, choice.get("image"),
                    choice.get("image_sha256"))
            for frame in choice.get("frames") or []:
                _validate_asset_binding(
                    public_assets, frame.get("image"),
                    frame.get("image_sha256"))
        expected_cases.append({
            "public": item,
            "canonical_answer": answer["canonical_answer"],
            "oracle_ref": answer.get("oracle_ref"),
            "atom_ref": answer["atom_ref"],
        })
    coverage = _coverage(items, task_ids)
    if benchmark_json.get("coverage") != coverage:
        raise ValueError("benchmark coverage changed")
    if benchmark_json.get("task_status") != _task_status(
            coverage, report.get("rejections") or {}, task_ids=task_ids):
        raise ValueError("benchmark task status changed")
    if benchmark_json.get("cases") != expected_cases:
        raise ValueError("benchmark JSON is not a mechanical public/private join")
    if report.get("coverage") != coverage or \
            report.get("headline_eligible") is not False:
        raise ValueError("preview report changed")
    if report.get("distributions") != _candidate_distributions(
            items, answer_by_id):
        raise ValueError("preview distributions changed")
    if report.get("shortcut_audit") != _shortcut_audit(
            items, answer_by_id):
        raise ValueError("preview shortcut audit changed")
    if diversity_report:
        expected_diversity = dict(diversity_report)
        expected_diversity.update(
            diversity_selection.summarize_selection({
                "items": items,
                "answers": answers,
                "atoms": atoms,
                "record_contexts": context_rows,
            }))
        if diversity_report != expected_diversity:
            raise ValueError("preview diversity summary changed")
    return benchmark_json


def _optional_mean(values) -> float | None:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else None


def _aggregate_candidate_scores(
        task_cells: dict, active_task_ids=TASK_IDS) -> tuple[dict, dict, float | None]:
    """Apply the frozen task-within-head and equal-head aggregation."""
    active_task_ids = tuple(active_task_ids)
    by_task = {
        task_id: _optional_mean(task_cells.get(task_id, ()))
        for task_id in active_task_ids
    }
    by_head = {}
    for head, task_ids in HEAD_TASKS.items():
        if not set(task_ids) <= set(active_task_ids):
            continue
        values = [by_task[task_id] for task_id in task_ids]
        by_head[head] = (
            _optional_mean(values)
            if all(value is not None for value in values) else None)
    overall = (
        _optional_mean(by_head.values())
        if by_head and all(value is not None for value in by_head.values())
        else None)
    return by_task, by_head, overall


def _six_task_macro(by_task: dict) -> float | None:
    """Return the official equal-task aggregate without renormalizing gaps."""
    values = [by_task.get(task_id) for task_id in TASK_IDS]
    return (
        _optional_mean(values)
        if all(value is not None for value in values) else None)


def evaluate_preview_artifact(
        artifact_root, *, predictions: dict[str, object] | None = None,
        gt_as_pred: bool = False, allow_partial: bool = False,
        expected_source_authority=None,
        _validated_benchmark: dict | None = None) -> dict:
    """Score v16 Closed Exact answers with equal A/B/C head weights.

    Predictions map item IDs either directly to a choice ID or to an object
    containing ``answer``.  Missing tasks make their head and the ABC overall
    undefined; the scorer never renormalizes over surviving tasks or heads.
    """
    artifact_root = Path(artifact_root)
    benchmark_json = _validated_benchmark
    if benchmark_json is None:
        benchmark_json = validate_preview_artifact(
            artifact_root,
            expected_source_authority=expected_source_authority)
    items = _read_jsonl(artifact_root / "public" / "items.jsonl")
    answers = _read_jsonl(artifact_root / "private" / "answers.jsonl")
    atoms = _read_jsonl(artifact_root / "private" / "atoms.jsonl")
    context_rows = _read_jsonl(
        artifact_root / "private" / "record_contexts.jsonl")
    answer_by_id = {value["id"]: value for value in answers}
    atom_by_id = {value["id"]: value for value in atoms}
    context_by_digest = {
        value["record_sha256"]: value["context"]
        for value in context_rows
    }
    if gt_as_pred and predictions is not None:
        raise ValueError("GT replay cannot be combined with predictions")
    if gt_as_pred:
        submitted = {
            item["id"]: answer_by_id[item["id"]]["canonical_answer"]
            for item in items
        }
    else:
        submitted = dict(predictions or {})
        unknown = set(submitted) - {item["id"] for item in items}
        if unknown:
            raise ValueError("predictions contain unknown candidate IDs")
        if not allow_partial and set(submitted) != {item["id"] for item in items}:
            raise ValueError("prediction coverage is incomplete")

    item_scores = {}
    task_cells = collections.defaultdict(list)
    evaluation_rows = []
    for item in items:
        value = submitted.get(item["id"])
        if isinstance(value, dict):
            value = value.get("answer")
        score = float(
            value == answer_by_id[item["id"]]["canonical_answer"])
        item_scores[item["id"]] = score
        task_id = item["task_id"]
        task_cells[task_id].append(score)
        answer = answer_by_id[item["id"]]
        atom = atom_by_id[answer["atom_ref"]]
        context = context_by_digest[atom["record_sha256"]]
        choices = item.get("choices") or []
        choice_by_id = {
            choice.get("id"): choice for choice in choices
            if isinstance(choice, dict)
        }
        selected_choice = choice_by_id.get(value) or {}
        evaluation_rows.append({
            "id": item["id"],
            "task_id": task_id,
            "score": score,
            "canonical_answer": answer["canonical_answer"],
            "submitted_answer": value,
            "random_chance": (
                1.0 / len(choices) if choices else 0.0),
            "atom_ref": answer["atom_ref"],
            "scene_id": (
                context.get("scene_id") or
                (context.get("source") or {}).get("scene_id") or
                atom.get("frame_id")),
            "s0_id": context.get("observation_id") or context.get("frame_id"),
            "precise_distance_m": answer.get("precise_distance_m"),
            "submitted_choice_value_m": selected_choice.get("value_m"),
        })

    active_task_ids = tuple(benchmark_json["task_catalog"])
    by_task, by_head, overall = _aggregate_candidate_scores(
        task_cells, active_task_ids)
    rows_by_task = collections.defaultdict(list)
    for row in evaluation_rows:
        rows_by_task[row["task_id"]].append(row)
    clustered = {}
    for task_id, rows in rows_by_task.items():
        seed = int.from_bytes(hashlib.sha256(
            task_id.encode("utf-8")).digest()[:8], "big")
        clustered[task_id] = {
            "scene": _clustered_bootstrap_summary(
                rows, cluster_field="scene_id", seed=seed),
            "s0": _clustered_bootstrap_summary(
                rows, cluster_field="s0_id", seed=seed + 1),
        }
    diagnostics = {
        "chance": _chance_diagnostics(rows_by_task, by_task),
        "A1_safety": _a1_safety_diagnostics(
            rows_by_task.get("A1_collision", [])),
        "B1_numeric": _b1_numeric_diagnostics(
            rows_by_task.get("B1_endpoint_distance", [])),
        "joint_accuracy": {
            "A1_and_A2": _joint_task_summary(
                evaluation_rows,
                ("A1_collision", "A2_collision_step_grounding")),
            "A1_and_A3": _joint_task_summary(
                evaluation_rows,
                ("A1_collision", "A3_contact_object")),
            "B1_and_B2": _joint_task_summary(
                evaluation_rows,
                ("B1_endpoint_distance", "B2_endpoint_direction")),
            "B1_B2_and_C1": _joint_task_summary(
                evaluation_rows,
                ("B1_endpoint_distance", "B2_endpoint_direction",
                 "C1_future_view_selection")),
        },
        "rollout_exact": _exact_group_summary(
            [{**row, "rollout_group": row["atom_ref"]}
             for row in evaluation_rows],
            group_field="rollout_group"),
        "clustered_uncertainty": clustered,
    }
    return {
        "schema": "egoconseq.candidate-evaluation.v1",
        "candidate_only": True,
        "headline_eligible": False,
        "role": (
            "scorer_integrity_check" if gt_as_pred else
            "model_evaluation"),
        "coverage": dict(benchmark_json["coverage"]),
        "task_scope": benchmark_json["task_scope"],
        "num_items": len(items),
        "num_predictions": len(submitted),
        "item_scores": item_scores,
        "by_task": by_task,
        "by_head": by_head,
        "overall": overall,
        "six_task_macro": _six_task_macro(by_task),
        # Diagnostics never feed the frozen equal-head candidate headline.
        "diagnostics": diagnostics,
    }


def _validate_asset_binding(
        assets: dict[str, str], relative: object,
        expected_sha256: object) -> None:
    value = Path(str(relative))
    if value.is_absolute():
        raise ValueError("public preview asset must be relative")
    name = value.as_posix()
    if ".." in value.parts or assets.get(name) != expected_sha256:
        raise ValueError("public asset digest changed")


def _render_preview_html_validated(
        benchmark_path, report_root, validated_benchmark) -> Path:
    """Render a benchmark value returned by preview validation."""
    from pipeline import case_browser

    benchmark_path = Path(benchmark_path)
    source = case_browser.browser_source_from_validated_artifact(
        "Candidate", benchmark_path.parent, validated_benchmark)
    return case_browser.render_case_browser([source], Path(report_root))


def render_preview_html(
        benchmark_path, report_root, *, expected_source_authority=None) -> Path:
    """Validate external source authority, then render one candidate artifact."""
    benchmark_path = Path(benchmark_path)
    benchmark_json = validate_preview_artifact(
        benchmark_path.parent,
        expected_source_authority=expected_source_authority)
    return _render_preview_html_validated(
        benchmark_path, report_root, benchmark_json)


def apply_main_publication_selection(
        projection: dict, *, ordinary_proposal_protocols: Iterable[str],
        a1_common_support: bool | None) -> dict:
    """Apply the production A1 selector from authenticated source facts.

    Keeping this decision in one pure helper prevents validators from trusting
    an artifact's publication marker to choose a weaker selection policy.
    """
    use_v3_selection = (
        a1_common_support is not False and
        a1_common_support_module.v3_default_for_protocols(
            ordinary_proposal_protocols))
    use_legacy_selection = (
        not use_v3_selection and
        a1_common_support_module.resolve_enabled(
            a1_common_support, ordinary_proposal_protocols,
            family_authority_present=False))
    projection = diversity_selection.apply_selection(
        projection, balance_a1_v3=use_v3_selection)
    if use_v3_selection:
        projection = a1_common_support_module.apply_v3_selection(projection)
    elif use_legacy_selection:
        projection = a1_common_support_module.apply_selection(projection)
    return diversity_selection.refresh_report(projection)


def build_main_preview(
        records_path, output_root, report_root, *, run_meta_sha256=None,
        expected_source_authority=None, collection_funnel_path=None,
        main_max_items_per_task=None,
        a1_common_support: bool | None = None) -> dict:
    """One-call authenticated main records -> ABC artifact -> HTML pipeline."""
    records_paths = (
        [Path(value) for value in records_path]
        if isinstance(records_path, (list, tuple)) else [Path(records_path)])
    if run_meta_sha256 is None:
        raise ValueError("trusted run metadata digest is required")
    run_meta_digests = (
        [str(value) for value in run_meta_sha256]
        if isinstance(run_meta_sha256, (list, tuple)) else
        [str(run_meta_sha256)])
    if len(run_meta_digests) != len(records_paths):
        raise ValueError("records/run metadata digest count differs")
    if expected_source_authority is None:
        expected_source_authority = \
            gate_authority.resolve_preview_source_authority_from_inputs(
                records_paths=records_paths,
                expected_run_meta_sha256=run_meta_digests)
    output_root = Path(output_root)
    projections = []
    source_paths = []
    source_snapshots = []
    for index, (main_path, expected_meta_sha256) in enumerate(zip(
            records_paths, run_meta_digests)):
        snapshot = candidate_sources.load_candidate_source(
            main_path, expected_run_meta_sha256=expected_meta_sha256,
            source_authority=expected_source_authority,
            source_index=index)
        source_snapshots.append(snapshot)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".candidate_preview_build.",
            dir=output_root.parent) as temporary_build:
        build_root = Path(temporary_build)
        ordinary_proposal_protocols = set()
        for snapshot in source_snapshots:
            index = snapshot.source_index
            main_path = snapshot.records_path
            records = snapshot.records
            ordinary_proposal_protocols.update(
                a1_common_support_module.ordinary_proposal_protocols(records))
            projections.append(compile_main_records(
                records, asset_root=main_path.parent,
                build_root=build_root / f"main_{index:02d}",
                max_items_per_task=main_max_items_per_task,
                source_validated_record_sha256=
                    snapshot.record_sha256_by_identity))
            source_paths.append(main_path)
        projection = apply_main_publication_selection(
            merge_projections(*projections),
            ordinary_proposal_protocols=ordinary_proposal_protocols,
            a1_common_support=a1_common_support)
        funnel = None
        if collection_funnel_path is not None:
            funnel = json.loads(Path(collection_funnel_path).read_text(
                encoding="utf-8"))
        write_preview_artifact(
            projection, output_root, source_records_path=source_paths,
            source_run_meta_sha256=run_meta_digests,
            collection_funnel=funnel,
            expected_source_authority=expected_source_authority)
    benchmark_json = validate_preview_artifact(
        output_root, expected_source_authority=expected_source_authority,
        source_snapshots=source_snapshots)
    _render_preview_html_validated(
        output_root / "benchmark.json", report_root, benchmark_json)
    return benchmark_json
