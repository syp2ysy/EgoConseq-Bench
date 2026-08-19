"""Counterfactual C1 neighbours for the ordinary collection route.

This module asks what the nearby action programs would have looked like: one
primitive shorter, one turn further, half a metre longer or shorter, or the
last turn mirrored.

The neighbour bank is a pure function of the query program.  Collection first
certifies the ordinary label-blind shortlist, chooses a query that really
completed clear, then spends a bounded second pass on its neighbours.
Compilation recovers exactly the same set from a published record without a
seed, a sidecar or a new schema field.  Anything that varied per run -- a pose
seed, a wall clock, a bank ordering -- would have to be carried in the record
for the validator to re-derive the family, so none of it is used here.

Candidate-only by construction: ``headline_eligible`` is always false and the
formal visual-distinguishability gate stays pending.  What this layer does
certify is cheap and total -- the correct image is the query's own terminal
frame, the three distractors are distinct programs, the four images are
pixelwise distinct, every program sits on the frozen action grid, and the option
order is a deterministic function of the family identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from typing import Mapping

import numpy as np

from pipeline import (
    action_proposal, action_sampling, actions as A, config,
    future_view_selection, outcome, record,
)


NEIGHBOR_PROTOCOL = "c1-counterfactual-neighbor.v2"
SELECTION_SCHEMA = "c1-counterfactual-selection.v2"
SELECTOR_PROTOCOL = "counterfactual-neighbors.v2"
VARIANT = "c1_counterfactual"
DISTRACTOR_COUNT = 3
NEIGHBOR_GENERATORS = (
    "prefix_drop",
    "extend_turn",
    "last_forward_delta",
    "reverse_last_turn",
    "extend_turn_forward",
)
#: Turns offered by the extending families, nearest deviation first.
EXTEND_TURNS_DEG = (-15.0, 15.0, -30.0, 30.0)
EXTEND_FORWARD_M = 0.5
FORWARD_DELTA_M = 0.5


@dataclass(frozen=True)
class Neighbor:
    """One counterfactual program derived from a query program."""

    tag: str
    actions: tuple[A.Action, ...]
    generator_id: str


@dataclass(frozen=True)
class C1Family:
    """Private collection/render authority for one selected C1 query."""

    query_tag: str
    neighbor_tags: tuple[str, ...]
    duplicate_program_count: int = 0


@dataclass(frozen=True)
class RecordSelectionIndex:
    """Authenticated C1 metadata plus a selected-image decode cache."""

    record_outcomes: Mapping[str, dict]
    descriptors: Mapping[str, dict]
    pixels: dict[str, np.ndarray]


def _parsed(actions) -> list:
    values = list(actions or [])
    if values and isinstance(values[0], Mapping):
        values = A.parse_actions(values)
    return values


def action_tag(actions) -> str:
    """Return the opaque candidate identifier shared with the action bank."""
    return action_proposal.candidate_tag(_parsed(actions))


def action_sha256(actions) -> str:
    """Return the record-domain action digest used by terminal RGB atoms."""
    return record.action_program_sha256(A.actions_to_dicts(_parsed(actions)))


def is_query_program(actions) -> bool:
    """Return whether a program may act as a counterfactual C1 query.

    Ending on a Forward is what makes the item worth asking: the last thing the
    body did was move, so the four options differ in where it ended up and not
    only in where it is looking.
    """
    try:
        values = _parsed(actions)
    except (KeyError, TypeError, ValueError):
        return False
    return (len(values) in config.GEN_LENGTHS and
            bool(values) and isinstance(values[-1], A.Forward))


def _valid(actions) -> tuple[A.Action, ...] | None:
    values = tuple(actions)
    if len(values) not in config.GEN_LENGTHS:
        return None
    try:
        A.validate_physics_actions(values)
    except (TypeError, ValueError):
        return None
    return values


def _raw_proposals(query: tuple[A.Action, ...]):
    """Yield generator IDs and programs before query-conditioned ordering."""
    yield "prefix_drop", query[:-1]
    for deg in EXTEND_TURNS_DEG:
        yield "extend_turn", query + (A.Turn(deg),)
    last = query[-1]
    if isinstance(last, A.Forward):
        for delta in (FORWARD_DELTA_M, -FORWARD_DELTA_M):
            yield "last_forward_delta", query[:-1] + (
                A.Forward(round(float(last.m) + delta, 9)),)
    for index in range(len(query) - 1, -1, -1):
        value = query[index]
        if isinstance(value, A.Turn):
            yield "reverse_last_turn", (
                query[:index] + (A.Turn(-float(value.deg)),) +
                query[index + 1:])
            break
    for deg in EXTEND_TURNS_DEG:
        yield "extend_turn_forward", query + (
            A.Turn(deg), A.Forward(EXTEND_FORWARD_M))


def neighbor_generator_priority(query_actions) -> tuple[str, ...]:
    """Return a query-conditioned generator priority with no label input.

    A fixed ``prefix_drop -> extend_turn -> ...`` priority creates an option
    shortcut because the generators have visibly different motion scales.
    The query action digest is present at both collection and compilation, so
    it can randomize that priority without introducing an unrecoverable pose
    seed or consulting any outcome label.
    """
    query = tuple(_parsed(query_actions))
    if not is_query_program(query):
        raise ValueError("C1 counterfactual query program is invalid")
    query_sha = action_sha256(query)
    return tuple(sorted(
        NEIGHBOR_GENERATORS,
        key=lambda generator_id: hashlib.sha256(
            f"{query_sha}:{generator_id}".encode("ascii")).hexdigest()))


def _proposals(query: tuple[A.Action, ...]):
    """Yield programs in the frozen, query-conditioned generator order."""
    priority = {
        generator_id: index for index, generator_id in enumerate(
            neighbor_generator_priority(query))
    }
    rows = list(enumerate(_raw_proposals(query)))
    rows.sort(key=lambda row: (priority[row[1][0]], row[0]))
    for _source_index, proposal in rows:
        yield proposal


def _counterfactual_neighbor_offer_audit(
        query_actions) -> tuple[list[Neighbor], int]:
    """Return unique offers and the number of duplicate programs removed."""
    query = tuple(_parsed(query_actions))
    if not is_query_program(query):
        raise ValueError("C1 counterfactual query program is invalid")
    seen = {action_sha256(query)}
    neighbors = []
    duplicates = 0
    for generator_id, proposal in _proposals(query):
        actions = _valid(proposal)
        if actions is None:
            continue
        digest = action_sha256(actions)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        neighbors.append(Neighbor(
            tag=action_tag(actions), actions=actions,
            generator_id=generator_id))
    return neighbors, duplicates


def counterfactual_neighbors(query_actions) -> list[Neighbor]:
    """Return the deterministic neighbour bank for one query program.

    The offer order is frozen and the result is deduplicated against the query
    and against itself, so a caller that takes the first *n* members always
    takes the same *n* members.  A proposal that leaves the published vocabulary
    -- an empty prefix, a seventh primitive, a forward off the half-metre grid --
    is dropped here rather than rejected later, because an invalid program must
    never consume one of the pose's bounded second-pass slots.
    """
    neighbors, _duplicates = _counterfactual_neighbor_offer_audit(
        query_actions)
    return neighbors


def neighbor_provenance(neighbor: Neighbor, query_actions, *,
                        index: int) -> dict:
    """Return the bank provenance row for one counterfactual neighbour.

    *index* is the neighbour's position in the frozen bank order.  It is in the
    template id so that two members of one family at one length still sort
    apart in the bank manifest, whose canonical order would otherwise depend on
    the order they happened to be appended in.
    """
    return {
        "protocol": NEIGHBOR_PROTOCOL,
        "template_id": (
            f"c1-counterfactual-{int(index):02d}-"
            f"{neighbor.generator_id}"),
        "variant": VARIANT,
        "base_action_sha256": action_sha256(query_actions),
        "neighbor_generator_id": neighbor.generator_id,
    }


# A collision program is proposed in order to hit something, so it almost never
# completes clear and would waste one of the pose's bounded query slots.
QUERY_VARIANTS = ("safe", action_proposal.NATURAL_DYNAMIC_VARIANT)


def reserve_pose_slots(pools, proposal_provenance, bank_manifest, *,
                       pose_seed, stats, variant_of, group_labels):
    """Bank up to two certified-clear query families for the second pass.

    The first, label-blind shortlist is certified before this function runs.
    C1 itself is defined only for a program that completed clear, so choosing
    the query from that certified set is task eligibility, not proposal-label
    conditioning.  Its neighbours are then added for a bounded second
    certification pass. This avoids spending either query slot on a depth-side
    ``safe`` proposal that the full oracle later rejects.

    Returns immutable query-family authority; an empty tuple just means this
    pose has no program worth asking about, which costs nothing.
    """
    existing_in_pools = {
        tag for values in pools.values() for tag, _actions in values}
    existing_in_manifest = {
        str(row.get("tag")) for row in bank_manifest
        if isinstance(row, dict) and row.get("tag") is not None}
    queries = sorted(
        ((tag, actions) for length in sorted(pools)
         for tag, actions in pools[length]
         if variant_of(proposal_provenance, tag) in QUERY_VARIANTS and
         group_labels.get(str(tag)) == "safe" and
         is_query_program(actions)),
        key=lambda item: (
            QUERY_VARIANTS.index(variant_of(proposal_provenance, item[0])),
            action_sampling.candidate_sort_key(pose_seed, item[0], item[1])))
    selected_queries = queries[:config.C1_QUERIES_PER_POSE]
    stats["c1_counterfactual_query_shortfall"] += (
        config.C1_QUERIES_PER_POSE - len(selected_queries))
    query_tags = {str(tag) for tag, _actions in selected_queries}
    globally_claimed = set(query_tags)
    families = []
    for query_tag, query_actions in selected_queries:
        neighbors, duplicate_count = \
            _counterfactual_neighbor_offer_audit(query_actions)
        stats["c1_counterfactual_neighbors_offered"] += len(neighbors)
        reserved = []
        for index, neighbor in enumerate(neighbors):
            if neighbor.tag in globally_claimed:
                duplicate_count += 1
                continue
            globally_claimed.add(neighbor.tag)
            reserved.append(neighbor.tag)
            if neighbor.tag in existing_in_pools:
                # The shortlist already certified this program on its own
                # merits; it keeps ordinary provenance and serves both roles.
                stats["c1_counterfactual_neighbors_already_banked"] += 1
            else:
                existing_in_pools.add(neighbor.tag)
                provenance = neighbor_provenance(
                    neighbor, query_actions, index=index)
                proposal_provenance[neighbor.tag] = provenance
                if neighbor.tag in existing_in_manifest:
                    # A recalled row becomes C1-only because its admission is
                    # conditioned on the selected query's safe certificate.
                    manifest_row = next(
                        row for row in bank_manifest
                        if str(row.get("tag")) == neighbor.tag)
                    manifest_row.update({
                        "variant": VARIANT,
                        "template_id": provenance["template_id"],
                    })
                    stats["c1_counterfactual_neighbors_recalled"] += 1
                else:
                    bank_manifest.append({
                        "length": len(neighbor.actions),
                        "tag": neighbor.tag,
                        "variant": VARIANT,
                        "template_id": provenance["template_id"],
                        "actions": A.actions_to_dicts(neighbor.actions),
                    })
                    existing_in_manifest.add(neighbor.tag)
                    stats["c1_counterfactual_neighbors_banked"] += 1
                pools.setdefault(len(neighbor.actions), []).append(
                    (neighbor.tag, list(neighbor.actions)))
            if len(reserved) == config.C1_NEIGHBORS_PER_QUERY:
                break
        if duplicate_count:
            stats["c1_neighbor_failure.duplicate_program"] += duplicate_count
            stats[
                f"c1_neighbor_failure.{query_tag}.duplicate_program"
            ] += duplicate_count
        families.append(C1Family(
            query_tag=str(query_tag), neighbor_tags=tuple(reserved),
            duplicate_program_count=duplicate_count))
        stats["c1_counterfactual_queries_banked"] += 1
    for length in pools:
        pools[length] = sorted(pools[length], key=lambda item: item[0])
    bank_manifest.sort(key=action_proposal.manifest_order_key)
    return tuple(families)


_NEIGHBOR_FAILURE_KEYS = (
    "physical_rejection",
    "render_rejection",
    "duplicate_program",
    "duplicate_image",
    "insufficient_unique_distractors",
)


def neighbor_failure_counts(
        families, accepted_outcomes, *, frame_id: str) -> dict[str, dict]:
    """Classify C1 family shortfalls from authoritative outcomes/assets."""

    def completed(tag):
        by_frame = (accepted_outcomes.get(str(tag)) or {}).get(frame_id) or []
        return next((value for value in by_frame
                     if outcome.is_completed_clear(value)), None)

    def asset_digests(value):
        asset = (value or {}).get("terminal_rgb_asset")
        if not isinstance(asset, dict):
            return None
        pixel = asset.get("pixel_sha256")
        png = asset.get("png_sha256")
        if not isinstance(pixel, str) or not pixel or \
                not isinstance(png, str) or not png:
            return None
        return pixel, png

    result = {}
    for family in families:
        counts = {key: 0 for key in _NEIGHBOR_FAILURE_KEYS}
        counts["duplicate_program"] = int(
            family.duplicate_program_count)
        seen_pixels = set()
        seen_pngs = set()
        query_digests = asset_digests(completed(family.query_tag))
        if query_digests is not None:
            seen_pixels.add(query_digests[0])
            seen_pngs.add(query_digests[1])
        viable = 0
        for tag in family.neighbor_tags:
            value = completed(tag)
            if value is None:
                counts["physical_rejection"] += 1
                continue
            digests = asset_digests(value)
            if digests is None:
                counts["render_rejection"] += 1
                continue
            pixel, png = digests
            if pixel in seen_pixels or png in seen_pngs:
                counts["duplicate_image"] += 1
                continue
            seen_pixels.add(pixel)
            seen_pngs.add(png)
            viable += 1
        if viable < DISTRACTOR_COUNT:
            counts["insufficient_unique_distractors"] = 1
        result[str(family.query_tag)] = counts
    return result


def tally_neighbor_outcome_failures(
        stats, families, accepted_outcomes, *, frame_id: str) -> None:
    """Add post-certification C1 failure causes to private funnel counters."""
    diagnostics = neighbor_failure_counts(
        families, accepted_outcomes, frame_id=frame_id)
    for query_tag, counts in diagnostics.items():
        for cause, count in counts.items():
            # Reservation already records duplicate programs once per pose;
            # the remaining causes are frame/render specific.
            if cause == "duplicate_program" or not count:
                continue
            stats[f"c1_neighbor_failure.{cause}"] += int(count)
            stats[f"c1_neighbor_failure.{query_tag}.{cause}"] += int(count)


def _sha256_values(values) -> str:
    return record.canonical_atom_sha256({"values": [str(v) for v in values]})


def build_record_selection_index(
        rec: dict, *, asset_root,
        shared_evidence_by_outcome: Mapping[int, object] | None = None
        ) -> RecordSelectionIndex:
    """Validate the complete C1 metadata pool without decoding its PNGs."""
    outcomes = list(rec.get("outcomes") or [])
    record_outcomes = {
        str(value.get("outcome_id") or ""): value for value in outcomes
    }
    if len(record_outcomes) != len(outcomes) or "" in record_outcomes:
        raise ValueError("counterfactual record outcome ids are invalid")
    descriptors = {}
    pixels = {}
    for outcome_value in outcomes:
        eligible, _reason = \
            future_view_selection._completed_clear_c_eligibility(
                rec, outcome_value,
                shared_evidence=(
                    None if shared_evidence_by_outcome is None else
                    shared_evidence_by_outcome.get(id(outcome_value))))
        if not eligible:
            continue
        withhold = outcome_value.get("terminal_rgb_asset_withhold")
        if withhold in future_view_selection.TERMINAL_ASSET_WITHHOLD_REASONS:
            continue
        try:
            atom = future_view_selection.validate_terminal_rgb_asset_metadata(
                rec, outcome_value)
        except (KeyError, TypeError, ValueError):
            continue
        descriptor = \
            future_view_selection.selection_descriptor_from_terminal_atom(
                rec, outcome_value, atom)
        digest = descriptor["action_sha256"]
        previous = descriptors.get(digest)
        if previous is None or descriptor["sha256"] < previous["sha256"]:
            descriptors[digest] = descriptor
    return RecordSelectionIndex(
        record_outcomes=record_outcomes,
        descriptors=descriptors,
        pixels=pixels)


def counterfactual_choices(
        rec: dict, primary: dict, *, asset_root,
        selection_index: RecordSelectionIndex | None = None) -> dict:
    """Select one query plus three counterfactual siblings from a record.

    Selection is by action digest, never by image similarity: the neighbour bank
    is rebuilt from the query's own program and matched against whatever the
    collector managed to certify.  A pose that lost some neighbours to the
    oracle still produces an item as long as three survived, which is the point
    of reserving more slots than the item needs.
    """
    record_outcomes = {
        str(value.get("outcome_id") or ""): value
        for value in rec.get("outcomes") or []
    }
    primary_id = str(primary.get("outcome_id") or "")
    if (not primary_id or record_outcomes.get(primary_id) != primary or
            len(record_outcomes) != len(rec.get("outcomes") or [])):
        raise ValueError("counterfactual primary outcome is not record-bound")
    query_actions = _parsed(primary.get("actions") or [])
    if not is_query_program(query_actions):
        raise ValueError("counterfactual_query_not_eligible")
    record_index = selection_index or build_record_selection_index(
        rec, asset_root=asset_root)
    if record_index.record_outcomes != record_outcomes:
        raise ValueError("counterfactual selection index is not record-bound")
    descriptors = record_index.descriptors
    query_sha = action_sha256(query_actions)
    query_descriptor = descriptors.get(query_sha)
    if query_descriptor is None or \
            query_descriptor["outcome_id"] != primary_id:
        raise ValueError("counterfactual_query_not_eligible")
    bank = counterfactual_neighbors(query_actions)
    chosen = []
    images = {query_descriptor["terminal_pixel_sha256"]}
    pngs = {query_descriptor["terminal_rgb_sha256"]}
    for neighbor in bank:
        descriptor = descriptors.get(action_sha256(neighbor.actions))
        if descriptor is None:
            continue
        if (descriptor["terminal_pixel_sha256"] in images or
                descriptor["terminal_rgb_sha256"] in pngs):
            continue
        images.add(descriptor["terminal_pixel_sha256"])
        pngs.add(descriptor["terminal_rgb_sha256"])
        chosen.append((neighbor, descriptor))
        if len(chosen) == DISTRACTOR_COUNT:
            break
    if len(chosen) < DISTRACTOR_COUNT:
        raise ValueError("counterfactual_distractor_shortfall")
    members = [(None, query_descriptor)] + chosen
    if len({value["action_sha256"] for _n, value in members}) != len(members):
        raise ValueError("counterfactual options repeat an action program")
    pixel_cache = record_index.pixels
    for _neighbor, descriptor in members:
        descriptor_sha256 = descriptor["sha256"]
        if descriptor_sha256 in pixel_cache:
            continue
        selected_outcome = record_index.record_outcomes[
            descriptor["outcome_id"]]
        snapshot, native_pixels = \
            future_view_selection.validate_terminal_rgb_asset_with_pixels(
                rec, selected_outcome, asset_root=asset_root)
        rebuilt = future_view_selection._selection_descriptor_from_snapshot(
            rec, selected_outcome, snapshot)
        if rebuilt != descriptor:
            raise ValueError("counterfactual terminal RGB metadata changed")
        pixel_cache[descriptor_sha256] = native_pixels
    identity = {
        "schema": SELECTION_SCHEMA,
        "selector_protocol": SELECTOR_PROTOCOL,
        "neighbor_protocol": NEIGHBOR_PROTOCOL,
        "record_id": str(rec.get("observation_id") or rec.get("frame_id")),
        "correct_outcome_id": primary_id,
        "query_action_sha256": query_sha,
        "neighbor_bank_action_sha256": [
            action_sha256(value.actions) for value in bank],
        "member_descriptor_sha256": sorted(
            value["sha256"] for _n, value in members),
    }
    family_id = record.canonical_atom_sha256(identity)
    ordered = sorted(members, key=lambda member: _sha256_values(
        [family_id, "choice-position", member[1]["sha256"]]))
    choices = []
    canonical_answer = None
    for index, (neighbor, descriptor) in enumerate(ordered, 1):
        choice_id = f"image_{index}"
        choices.append({
            "id": choice_id,
            "outcome_id": descriptor["outcome_id"],
            "base_rollout_key": descriptor["base_rollout_key"],
            "action_sha256": descriptor["action_sha256"],
            "neighbor_generator_id": (
                "query" if neighbor is None else neighbor.generator_id),
            "terminal_rgb_atom_sha256":
                descriptor["terminal_rgb_atom_sha256"],
            "terminal_rgb_path": descriptor["terminal_rgb_path"],
            "terminal_rgb_sha256": descriptor["terminal_rgb_sha256"],
            "terminal_pixel_sha256": descriptor["terminal_pixel_sha256"],
            "terminal_pose": descriptor["terminal_pose"],
            "action_summary": descriptor["action_summary"],
            "descriptor_sha256": descriptor["sha256"],
        })
        if descriptor["outcome_id"] == primary_id:
            canonical_answer = choice_id
    if canonical_answer is None:
        raise ValueError("counterfactual canonical answer is missing")
    block_features = {
        descriptor["sha256"]: future_view_selection.block_l1_features(
            pixel_cache[descriptor["sha256"]])
        for _neighbor, descriptor in ordered
    }
    pairwise_block_l1 = []
    for left_index, right_index in itertools.combinations(range(4), 2):
        left_descriptor = ordered[left_index][1]
        right_descriptor = ordered[right_index][1]
        left_pixels = pixel_cache[left_descriptor["sha256"]]
        right_pixels = pixel_cache[right_descriptor["sha256"]]
        pairwise_block_l1.append({
            "choice_ids": [
                choices[left_index]["id"], choices[right_index]["id"]],
            "outcome_ids": [
                left_descriptor["outcome_id"],
                right_descriptor["outcome_id"],
            ],
            "certificate":
                future_view_selection._block_l1_certificate_from_features(
                    left_pixels,
                    right_pixels,
                    block_features[left_descriptor["sha256"]],
                    block_features[right_descriptor["sha256"]]),
        })
    value = {
        **identity,
        "family_id": family_id,
        "canonical_answer": canonical_answer,
        "eligible_pool_count": len(descriptors),
        "eligible_pool_sha256": _sha256_values(
            sorted(row["sha256"] for row in descriptors.values())),
        "neighbor_bank_count": len(bank),
        "neighbor_generator_priority": list(
            neighbor_generator_priority(query_actions)),
        "selected_neighbor_generator_ids": [
            row.generator_id for row, _descriptor in chosen],
        "permutation_outcome_ids": [
            row["outcome_id"] for _n, row in ordered],
        "choices": choices,
        # Diagnostic only: these six exact distances are never read by the
        # selector and carry no acceptance threshold.
        "pairwise_block_l1": pairwise_block_l1,
        # The formal visual-distinguishability gate is still pending, so this
        # certificate never claims headline eligibility no matter how clean the
        # geometry underneath it is.
        "headline_eligible": False,
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def candidate_selection_or_reason(
        rec: dict, primary: dict, *, asset_root,
        selection_index: RecordSelectionIndex | None = None
        ) -> tuple[dict | None, str]:
    """Return the counterfactual selection, or the typed reason it is absent."""
    try:
        return counterfactual_choices(
            rec, primary, asset_root=asset_root,
            selection_index=selection_index), "eligible"
    except (KeyError, TypeError, ValueError) as error:
        reason = str(error)
        if reason not in {
                "counterfactual_distractor_shortfall",
                "counterfactual_query_not_eligible"}:
            reason = "counterfactual_selection_invalid"
        return None, reason


def validate_counterfactual_selection(
        rec: dict, primary: dict, certificate: dict, *, asset_root) -> dict:
    """Mechanically reselect from the record and compare atoms."""
    rebuilt = counterfactual_choices(rec, primary, asset_root=asset_root)
    if rebuilt != certificate:
        raise ValueError("counterfactual selection certificate changed")
    return rebuilt
