"""Deterministic publication balance for A1 public-action nuisance cells."""

from __future__ import annotations

import collections
import copy
import json
import math
from typing import Iterable

from pipeline import action_proposal, actions as action_geometry
from pipeline import c1_counterfactual, qa_reasons


POLICY = "a1-action-common-support.v2"
V3_POLICY = "a1-three-source-balance.v1"
V3_PUBLICATION_PROTOCOLS = frozenset({
    action_proposal.PROPOSAL_PROTOCOL_V3,
    action_proposal.PROPOSAL_PROTOCOL_V4,
    action_proposal.PROPOSAL_PROTOCOL_V5,
})


def v3_source_rejection_for(
        protocol: str | None, variant: str | None) -> str | None:
    """Reject proposal sources whose generator encodes an A1 label."""
    protocol = str(protocol or "")
    variant = str(variant or "")
    if variant == c1_counterfactual.VARIANT:
        return "a1_nonpublication_source"
    if protocol not in V3_PUBLICATION_PROTOCOLS:
        return None
    if variant in {
            action_proposal.NATURAL_DYNAMIC_VARIANT,
            action_proposal.A1_CONTROL_VARIANT}:
        return None
    return "a1_nonpublication_source"


def v3_source_rejection(record: dict, outcome: dict) -> str | None:
    """Read one legacy provenance row and apply the publication policy."""
    tag = str(outcome.get("action_group_id") or "")
    row = ((record.get("selection") or {}).get(
        "proposal_provenance") or {}).get(tag)
    if not isinstance(row, dict):
        return None
    return v3_source_rejection_for(row.get("protocol"), row.get("variant"))


def ordinary_proposal_protocols(records: Iterable[dict]) -> set[str]:
    """Return sampler protocols, excluding auxiliary C1 neighbours."""
    protocols = set()
    for record in records:
        provenance = (record.get("selection") or {}).get(
            "proposal_provenance") or {}
        for row in provenance.values():
            if not isinstance(row, dict):
                continue
            if (row.get("variant") == c1_counterfactual.VARIANT or
                    row.get("protocol") ==
                    c1_counterfactual.NEIGHBOR_PROTOCOL):
                continue
            if row.get("protocol") is not None:
                protocols.add(str(row["protocol"]))
    return protocols


def has_uniform_v3_ordinary_provenance(record: dict) -> bool:
    """Require every ordinary proposal row, including missing protocols."""
    provenance = (record.get("selection") or {}).get(
        "proposal_provenance")
    if not isinstance(provenance, dict):
        return False
    ordinary = []
    for row in provenance.values():
        if not isinstance(row, dict):
            return False
        if (row.get("variant") == c1_counterfactual.VARIANT or
                row.get("protocol") == c1_counterfactual.NEIGHBOR_PROTOCOL):
            continue
        ordinary.append(row)
    protocols = {row.get("protocol") for row in ordinary}
    return bool(ordinary) and len(protocols) == 1 and \
        protocols <= V3_PUBLICATION_PROTOCOLS


def default_for_protocols(protocols: Iterable[str]) -> bool:
    """Require one uniform v2 sampler population for default selection."""
    return set(protocols) == {action_proposal.PROPOSAL_PROTOCOL_V2}


def v3_default_for_protocols(protocols: Iterable[str]) -> bool:
    """Select v3 balance for any supported action-bank population."""
    values = set(protocols)
    return bool(values) and values <= V3_PUBLICATION_PROTOCOLS


def default_for_records(records: Iterable[dict]) -> bool:
    """Enable by default only for a uniform ordinary v2 population."""
    return default_for_protocols(ordinary_proposal_protocols(records))


def resolve_enabled(
        requested: bool | None, protocols: Iterable[str], *,
        family_authority_present: bool) -> bool:
    """Resolve an explicit switch or the v2-only default."""
    if requested is not None:
        return bool(requested)
    return (not family_authority_present and
            default_for_protocols(protocols))


def cell_key(item: dict) -> tuple:
    """Length, exact total Forward grid, and primitive-kind pattern.

    Turn magnitudes and signs are deliberately not cell dimensions.  Requiring
    an exact signed Turn program discarded balanced examples that share every
    coarse public-action nuisance a blind model can exploit.  The retained
    ``F``/``T`` pattern still balances action count, Forward-leg count, and the
    placement of turns without paying that unnecessary sample cost.
    """
    item_id = str(item.get("id") or "")
    try:
        parsed = action_geometry.parse_actions(
            (item.get("model_input") or {})["actions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"A1 common-support actions are invalid for {item_id}") from error
    if not parsed:
        raise ValueError(
            f"A1 common-support actions are empty for {item_id}")
    primitive_pattern = tuple(
        "F" if isinstance(action, action_geometry.Forward)
        else "T"
        for action in parsed)
    return (
        len(parsed),
        round(float(action_geometry.total_forward_m(parsed)), 6),
        primitive_pattern,
    )


def apply_selection(projection: dict) -> dict:
    """Keep equal A1 labels per cell and prune unreferenced private rows."""
    selected = copy.deepcopy(projection)
    items = list(selected.get("items") or [])
    answers = list(selected.get("answers") or [])
    answers_by_id = {str(value.get("id")): value for value in answers}
    if (len(answers_by_id) != len(answers) or
            set(answers_by_id) != {str(value.get("id")) for value in items}):
        raise ValueError(
            "A1 common-support public/private ids are not bijective")

    cells = collections.defaultdict(
        lambda: collections.defaultdict(list))
    a1_ids = set()
    for item in items:
        if item.get("task_id") != "A1_collision":
            continue
        item_id = str(item["id"])
        answer = str(answers_by_id[item_id].get("canonical_answer") or "")
        if answer not in {"collision", "no_collision"}:
            raise ValueError(
                f"A1 common-support answer is invalid for {item_id}")
        cells[cell_key(item)][answer].append(item_id)
        a1_ids.add(item_id)

    keep_a1_ids = set()
    for labels in cells.values():
        pair_count = min(
            len(labels.get("collision", ())),
            len(labels.get("no_collision", ())))
        for answer in ("collision", "no_collision"):
            keep_a1_ids.update(sorted(labels.get(answer, ()))[:pair_count])
    keep_ids = {
        str(item["id"]) for item in items
        if item.get("task_id") != "A1_collision"
    } | keep_a1_ids
    selected["items"] = [
        item for item in items if str(item["id"]) in keep_ids]
    selected["answers"] = [
        answer for answer in answers if str(answer["id"]) in keep_ids]

    used_atom_ids = {
        str(answer["atom_ref"]) for answer in selected["answers"]}
    selected["atoms"] = [
        atom for atom in selected.get("atoms") or []
        if str(atom.get("id")) in used_atom_ids]
    used_record_digests = {
        str(atom["record_sha256"]) for atom in selected["atoms"]}
    selected["record_contexts"] = [
        row for row in selected.get("record_contexts") or []
        if str(row.get("record_sha256")) in used_record_digests]

    dropped = len(a1_ids - keep_a1_ids)
    if dropped:
        reason = qa_reasons.require_report_reason(
            "a1_common_support_unmatched")
        bucket = selected["rejections"]["A1_collision"]
        bucket[reason] = int(bucket.get(reason, 0)) + dropped
    selected["publication_selection"] = {"A1_collision": POLICY}
    return selected


def validate_selection(items: list[dict], answers_by_id: dict) -> None:
    """Require both A1 labels at equal count inside every retained cell."""
    cells = collections.defaultdict(collections.Counter)
    for item in items:
        if item.get("task_id") != "A1_collision":
            continue
        item_id = str(item["id"])
        cells[cell_key(item)][str(
            answers_by_id[item_id].get("canonical_answer"))] += 1
    for labels in cells.values():
        if (set(labels) != {"collision", "no_collision"} or
                labels["collision"] != labels["no_collision"]):
            raise ValueError("A1 common-support publication is unbalanced")


def _projection_source_maps(
        answers: list[dict], atoms: list[dict],
        record_contexts: list[dict]) -> tuple[dict, dict, dict]:
    answers_by_id = {str(row.get("id")): row for row in answers}
    atoms_by_id = {str(row.get("id")): row for row in atoms}
    contexts_by_digest = {
        str(row.get("record_sha256")): row.get("context")
        for row in record_contexts
    }
    if (len(answers_by_id) != len(answers) or
            len(atoms_by_id) != len(atoms) or
            len(contexts_by_digest) != len(record_contexts)):
        raise ValueError("A1 v3 source tables are not unique")
    return answers_by_id, atoms_by_id, contexts_by_digest


def proposal_source_for_item(
        item: dict, answers_by_id: dict, atoms_by_id: dict,
        contexts_by_digest: dict) -> tuple[str, str]:
    item_id = str(item.get("id") or "")
    try:
        answer = answers_by_id[item_id]
        atom = atoms_by_id[str(answer["atom_ref"])]
        context = contexts_by_digest[str(atom["record_sha256"])]
        outcome = atom["outcome"]
        tag = str(outcome["action_group_id"])
        row = context["selection"]["proposal_provenance"][tag]
        return str(row["protocol"]), str(row["variant"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"A1 v3 source provenance is missing for {item_id}") from error


def source_dataset_for_item(
        item: dict, answers_by_id: dict, atoms_by_id: dict,
        contexts_by_digest: dict) -> str:
    """Return the authenticated dataset shared by both source contracts."""
    item_id = str(item.get("id") or "")
    try:
        answer = answers_by_id[item_id]
        atom = atoms_by_id[str(answer["atom_ref"])]
        context = contexts_by_digest[str(atom["record_sha256"])]
        containers = [context["source"]]
        legacy_contract = context.get("collection_contract")
        if legacy_contract is not None:
            containers.append(legacy_contract)
        values = {
            str(container.get("source_dataset"))
            for container in containers
            if container.get("source_dataset")
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"A1 v3 dataset provenance is missing for {item_id}") from error
    if len(values) != 1:
        raise ValueError(
            f"A1 v3 dataset provenance is invalid for {item_id}")
    return values.pop()


def natural_v3_cell_key(item: dict) -> tuple[int, int]:
    """The preregistered dynamic-natural nuisance cell."""
    parsed = action_geometry.parse_actions(
        (item.get("model_input") or {}).get("actions") or [])
    if not parsed:
        raise ValueError("A1 v3 natural action is empty")
    return (
        len(parsed),
        int(math.floor(
            float(action_geometry.total_forward_m(parsed)) / 1.0 + 1e-9)),
    )


def control_v3_cell_key(item: dict) -> tuple:
    """All public non-visual inputs of one exact control intervention."""
    model_input = item.get("model_input") or {}
    parsed = action_geometry.parse_actions(model_input.get("actions") or [])
    if not parsed:
        raise ValueError("A1 v3 control action is empty")
    public_setting = {
        key: model_input.get(key)
        for key in (
            "body_radius_m",
            "camera_height_above_visible_floor_m",
            "hfov_deg",
            "vfov_deg",
        )
    }
    if any(value is None for value in public_setting.values()):
        raise ValueError("A1 v3 control setting is incomplete")
    return (
        action_geometry.canonical_actions_sha256(parsed),
        json.dumps(
            public_setting, sort_keys=True, separators=(",", ":"),
            allow_nan=False),
    )


def _balanced_ids(cells: dict) -> tuple[set[str], int]:
    keep = set()
    offered = 0
    for labels in cells.values():
        offered += sum(len(ids) for ids in labels.values())
        pair_count = min(
            len(labels.get("collision", ())),
            len(labels.get("no_collision", ())))
        for answer in ("collision", "no_collision"):
            keep.update(sorted(labels.get(answer, ()))[:pair_count])
    return keep, offered - len(keep)


def _prune_projection(selected: dict, keep_ids: set[str]) -> None:
    selected["items"] = [
        item for item in selected.get("items") or []
        if str(item.get("id")) in keep_ids]
    selected["answers"] = [
        answer for answer in selected.get("answers") or []
        if str(answer.get("id")) in keep_ids]
    used_atom_ids = {
        str(answer["atom_ref"]) for answer in selected["answers"]}
    selected["atoms"] = [
        atom for atom in selected.get("atoms") or []
        if str(atom.get("id")) in used_atom_ids]
    used_record_digests = {
        str(atom["record_sha256"]) for atom in selected["atoms"]}
    selected["record_contexts"] = [
        row for row in selected.get("record_contexts") or []
        if str(row.get("record_sha256")) in used_record_digests]


def apply_v3_selection(projection: dict) -> dict:
    """Publish natural A1 plus a small exact control, never paired labels."""
    selected = copy.deepcopy(projection)
    items = list(selected.get("items") or [])
    answers = list(selected.get("answers") or [])
    answers_by_id, atoms_by_id, contexts = _projection_source_maps(
        answers, list(selected.get("atoms") or []),
        list(selected.get("record_contexts") or []))
    natural_cells = collections.defaultdict(
        lambda: collections.defaultdict(list))
    control_cells = collections.defaultdict(
        lambda: collections.defaultdict(list))
    keep_non_a1 = {
        str(item["id"]) for item in items
        if item.get("task_id") != "A1_collision"}
    nonpublication = 0
    for item in items:
        if item.get("task_id") != "A1_collision":
            continue
        item_id = str(item["id"])
        answer = str(answers_by_id[item_id].get("canonical_answer") or "")
        if answer not in {"collision", "no_collision"}:
            raise ValueError(f"A1 v3 answer is invalid for {item_id}")
        protocol, variant = proposal_source_for_item(
            item, answers_by_id, atoms_by_id, contexts)
        if protocol not in V3_PUBLICATION_PROTOCOLS:
            nonpublication += 1
        elif variant == action_proposal.NATURAL_DYNAMIC_VARIANT:
            dataset = source_dataset_for_item(
                item, answers_by_id, atoms_by_id, contexts)
            natural_cells[(dataset, *natural_v3_cell_key(item))][answer].append(
                item_id)
        elif variant == action_proposal.A1_CONTROL_VARIANT:
            dataset = source_dataset_for_item(
                item, answers_by_id, atoms_by_id, contexts)
            control_cells[(dataset, *control_v3_cell_key(item))][answer].append(
                item_id)
        else:
            nonpublication += 1

    natural_keep, natural_dropped = _balanced_ids(natural_cells)
    control_keep, control_dropped = _balanced_ids(control_cells)
    _prune_projection(
        selected, keep_non_a1 | natural_keep | control_keep)
    bucket = selected["rejections"]["A1_collision"]
    for reason, count in (
            ("a1_natural_balance_unmatched", natural_dropped),
            ("a1_control_exact_unmatched", control_dropped),
            ("a1_nonpublication_source", nonpublication)):
        if count:
            frozen = qa_reasons.require_report_reason(reason)
            bucket[frozen] = int(bucket.get(frozen, 0)) + int(count)
    selected["publication_selection"] = {"A1_collision": V3_POLICY}
    return selected


def validate_v3_selection(
        items: list[dict], answers_by_id: dict,
        atoms: list[dict], record_contexts: list[dict]) -> None:
    """Require exact balance and allowed provenance in a v3 artifact."""
    _answers, atoms_by_id, contexts = _projection_source_maps(
        list(answers_by_id.values()), atoms, record_contexts)
    natural = collections.defaultdict(collections.Counter)
    controls = collections.defaultdict(collections.Counter)
    for item in items:
        if item.get("task_id") != "A1_collision":
            continue
        answer = str(answers_by_id[str(item["id"])]["canonical_answer"])
        protocol, variant = proposal_source_for_item(
            item, answers_by_id, atoms_by_id, contexts)
        if protocol not in V3_PUBLICATION_PROTOCOLS:
            raise ValueError("A1 v3 publication contains a legacy source")
        dataset = source_dataset_for_item(
            item, answers_by_id, atoms_by_id, contexts)
        if variant == action_proposal.NATURAL_DYNAMIC_VARIANT:
            natural[(dataset, *natural_v3_cell_key(item))][answer] += 1
        elif variant == action_proposal.A1_CONTROL_VARIANT:
            controls[(dataset, *control_v3_cell_key(item))][answer] += 1
        else:
            raise ValueError("A1 v3 publication contains a paired source")
    for labels in (*natural.values(), *controls.values()):
        if (set(labels) != {"collision", "no_collision"} or
                labels["collision"] != labels["no_collision"]):
            raise ValueError("A1 v3 publication is unbalanced")
