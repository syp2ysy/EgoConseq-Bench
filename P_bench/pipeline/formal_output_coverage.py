"""Run-final action-length coverage over published collection records."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Iterable

from pipeline import config, io_utils


def _general_ab_summary(records: Iterable[dict], requested: int) -> dict:
    """Count general A/B units from one or more source shards."""
    keep = int(requested)
    if keep < 1:
        raise ValueError("formal output keep count must be positive")
    required_lengths = list(config.GEN_LENGTHS)
    units = {}
    for rec in records:
        selection = rec.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("formal output record selection is invalid")
        intervention = rec.get("intervention") or {}
        intervention_id = intervention.get("group_id")
        selected_ids = selection.get("action_group_ids")
        if (not isinstance(intervention_id, str) or not intervention_id or
                not isinstance(selected_ids, list) or
                any(not isinstance(value, str) or not value
                    for value in selected_ids)):
            raise ValueError("formal A/B selection binding is invalid")
        lengths_by_group = collections.defaultdict(set)
        for outcome in rec.get("outcomes") or []:
            if not isinstance(outcome, dict):
                raise ValueError("formal output outcome is invalid")
            group_id = outcome.get("action_group_id")
            if group_id not in selected_ids:
                continue
            actions = outcome.get("actions")
            if not isinstance(actions, list):
                raise ValueError("formal output actions are invalid")
            lengths_by_group[group_id].add(len(actions))
        for group_id in selected_ids:
            observed = lengths_by_group.get(group_id, set())
            if len(observed) != 1:
                raise ValueError("formal A/B published group length is invalid")
            length = next(iter(observed))
            if length not in required_lengths:
                raise ValueError("formal A/B published length is out of scope")
            unit_id = (intervention_id, group_id)
            previous = units.setdefault(unit_id, int(length))
            if previous != int(length):
                raise ValueError("formal A/B group length changed")

    counts = collections.Counter(units.values())
    counts_by_length = {
        f"L{length}": int(counts.get(length, 0))
        for length in required_lengths
    }
    shortfall = {
        f"L{length}": keep - counts_by_length[f"L{length}"]
        for length in required_lengths
        if counts_by_length[f"L{length}"] < keep
    }
    return {
        "schema": "egoconseq.formal-action-length-coverage.v1",
        "scope": "general_ab",
        "exempt_reason": None,
        "required_lengths": required_lengths,
        "requested_keep_per_length": keep,
        "counts_by_length": counts_by_length,
        "shortfall": shortfall,
        "complete": not shortfall,
    }


def summarize_paths(records_paths, args) -> dict:
    """Count general A/B units across independent source-bound shards."""
    requested = getattr(args, "keep_per_length", None)
    if requested is None:
        raise ValueError("cross-shard coverage requires keep_per_length")

    def records():
        for records_path in records_paths:
            yield from io_utils.read_jsonl(
                Path(records_path), require_dict=True)

    return _general_ab_summary(records(), int(requested))


def summarize(records_path, args) -> dict:
    """Count final published units without sensor/body/quartet inflation."""
    requested = getattr(args, "keep_per_length", None)
    exempt_reason = None
    if requested is None:
        # Direct unit callers that do not carry the collection CLI contract
        # are not final collection outputs.
        exempt_reason = "non_final_caller"
    if exempt_reason is not None:
        return {
            "schema": "egoconseq.formal-action-length-coverage.v1",
            "scope": "exempt",
            "exempt_reason": exempt_reason,
            "required_lengths": [],
            "requested_keep_per_length": requested,
            "counts_by_length": {},
            "shortfall": {},
            "complete": True,
        }

    return _general_ab_summary(
        io_utils.read_jsonl(Path(records_path), require_dict=True),
        int(requested))
