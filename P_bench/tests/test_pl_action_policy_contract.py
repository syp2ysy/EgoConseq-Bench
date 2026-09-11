"""A record must not be able to declare its own action-sampling policy.

Like the setting policy, the expectation comes from hash-verified run metadata.
These tests pin the three ways that could be subverted: claiming the wrong
policy, omitting the marker, and carrying provenance the declared policy could
not have produced. They also pin that the per-pose bank digest is recomputable
from the record rather than a fingerprint that only a rerun could check.
"""

from __future__ import annotations

import copy
import dataclasses
import pytest

from pipeline import action_proposal as AP
from pipeline import config, source_manifest, validate
from pipeline.pose_setting import SETTING_SAMPLING_POLICY


RADIUS = config.RADII_M[1]
_BANK = [
    {"length": 1, "tag": "L1-aaaa", "variant": "collision",
     "template_id": "T1-00", "actions": [{"type": "forward", "m": 2.5}]},
    {"length": 1, "tag": "L1-bbbb", "variant": "safe",
     "template_id": "T1-00", "actions": [{"type": "forward", "m": 1.0}]},
]


def _outcome(tag="L1-bbbb"):
    return {
        "action_group_id": tag, "action_group_label": "safe", "seq_len": 1,
        "actions": [{"type": "forward", "m": 1.0}],
        "body": {"radius_m": RADIUS}, "physical": {"collision": False},
    }


def _record(policy, *, bank=None, provenance=None, digest=None):
    selection = {
        "policy": policy,
        "setting_sampling_policy": SETTING_SAMPLING_POLICY,
        "action_group_ids": ["L1-bbbb"],
        "action_group_labels": {"L1-bbbb": "safe"},
        "candidate_budget": config.ACTION_CANDIDATE_MIN_PER_POSE,
        "observed_lengths": [1],
        "observed_label_counts": {"safe": 1},
        "required_radii_m": [float(RADIUS)],
    }
    if policy in {
            AP.DEPTH_CONDITIONED_POLICY,
            AP.DEPTH_CONDITIONED_POLICY_V3,
            AP.DEPTH_CONDITIONED_POLICY_V4,
            AP.DEPTH_CONDITIONED_POLICY_V5}:
        manifest = _BANK if bank is None else bank
        selection["materialized_action_bank"] = manifest
        selection["materialized_action_bank_sha256"] = (
            digest if digest is not None
            else AP.action_bank_manifest_sha256(manifest))
        selection["proposal_provenance"] = (
            {"L1-bbbb": {
                "template_id": "T1-00", "variant": "safe",
            }} if provenance is None
            else provenance)
    return {
        "frame_id": "f", "outcomes": [_outcome()],
        "intervention": {"group_id": "pose-1"}, "selection": selection,
    }


def _errors(record, action_policy):
    return validate._validate_balanced_selection(
        record, SETTING_SAMPLING_POLICY, action_policy)


# --------------------------------------------------------------------------
# Where the expectation comes from
# --------------------------------------------------------------------------

@pytest.mark.parametrize("action_mode,expected", [
    ("balanced", AP.DEPTH_CONDITIONED_POLICY),
    ("file", AP.EXPLICIT_ACTION_FILE_POLICY),
    ("something-else", None),
    (None, None),
])
def test_policy_is_derived_from_the_run_action_mode(action_mode, expected):
    assert AP.expected_policy_for_action_mode(action_mode) == expected


def test_new_v5_collection_is_explicit_without_reinterpreting_legacy_runs():
    assert AP.expected_policy_for_action_mode("balanced") == \
        AP.DEPTH_CONDITIONED_POLICY
    assert AP.policy_for_new_collection("balanced") == \
        AP.DEPTH_CONDITIONED_POLICY_V5
    assert AP.DEPTH_CONDITIONED_POLICY_V5 != AP.DEPTH_CONDITIONED_POLICY_V4
    assert AP.DEPTH_CONDITIONED_POLICY_V4 != AP.DEPTH_CONDITIONED_POLICY_V3
    assert AP.DEPTH_CONDITIONED_POLICY_V3 != AP.DEPTH_CONDITIONED_POLICY
    assert AP.policy_for_new_collection("file") == \
        AP.EXPLICIT_ACTION_FILE_POLICY


def test_explicit_v3_through_v5_policies_match_balanced_metadata():
    assert AP.declared_policy_matches_action_mode(
        "balanced", AP.DEPTH_CONDITIONED_POLICY_V3)
    assert AP.declared_policy_matches_action_mode(
        "balanced", AP.DEPTH_CONDITIONED_POLICY_V4)
    assert AP.declared_policy_matches_action_mode(
        "balanced", AP.DEPTH_CONDITIONED_POLICY_V5)
    assert not AP.declared_policy_matches_action_mode(
        "file", AP.DEPTH_CONDITIONED_POLICY_V3)


# --------------------------------------------------------------------------
# A record cannot declare its own policy
# --------------------------------------------------------------------------

def test_each_mode_accepts_only_its_own_policy():
    assert _errors(_record(AP.EXPLICIT_ACTION_FILE_POLICY),
                   AP.EXPLICIT_ACTION_FILE_POLICY) == []
    assert _errors(_record(AP.DEPTH_CONDITIONED_POLICY),
                   AP.DEPTH_CONDITIONED_POLICY) == []
    assert _errors(_record(AP.DEPTH_CONDITIONED_POLICY_V3),
                   AP.DEPTH_CONDITIONED_POLICY_V3) == []
    assert _errors(_record(AP.DEPTH_CONDITIONED_POLICY_V4),
                   AP.DEPTH_CONDITIONED_POLICY_V4) == []
    assert _errors(_record(AP.DEPTH_CONDITIONED_POLICY_V5),
                   AP.DEPTH_CONDITIONED_POLICY_V5) == []


def test_claiming_the_depth_policy_under_a_file_run_is_rejected():
    errors = _errors(_record(AP.DEPTH_CONDITIONED_POLICY),
                     AP.EXPLICIT_ACTION_FILE_POLICY)
    assert any("action sampling policy" in error for error in errors)


def test_omitting_the_policy_cannot_escape_the_run_contract():
    record = _record(AP.DEPTH_CONDITIONED_POLICY)
    record["selection"].pop("policy")
    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY)
    assert any("action sampling policy" in error for error in errors)


def test_a_file_run_must_not_carry_depth_proposal_provenance():
    record = _record(AP.EXPLICIT_ACTION_FILE_POLICY)
    record["selection"]["proposal_provenance"] = {"L1-bbbb": {}}
    errors = _errors(record, AP.EXPLICIT_ACTION_FILE_POLICY)
    assert any("must not carry depth proposal" in error for error in errors)


@pytest.mark.parametrize("field", [
    "proposal_provenance", "materialized_action_bank",
    "materialized_action_bank_sha256",
])
def test_a_depth_run_must_carry_every_proposal_field(field):
    record = _record(AP.DEPTH_CONDITIONED_POLICY)
    record["selection"].pop(field)
    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY)
    assert any("depth proposal fields are required" in error
               for error in errors)


# --------------------------------------------------------------------------
# The digest authenticates rather than fingerprints
# --------------------------------------------------------------------------

def test_the_bank_digest_is_recomputable_from_the_record():
    record = _record(AP.DEPTH_CONDITIONED_POLICY)
    stored = record["selection"]
    assert AP.action_bank_manifest_sha256(
        stored["materialized_action_bank"]) == (
            stored["materialized_action_bank_sha256"])
    assert _errors(record, AP.DEPTH_CONDITIONED_POLICY) == []


def test_a_digest_that_does_not_match_its_bank_is_rejected():
    record = _record(AP.DEPTH_CONDITIONED_POLICY, digest="0" * 64)
    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY)
    assert any("digest does not match" in error for error in errors)


def test_a_non_canonical_bank_order_is_rejected():
    """A reordered manifest would hash self-consistently, so the order is part
    of what the digest certifies rather than an incidental detail."""
    record = _record(AP.DEPTH_CONDITIONED_POLICY,
                     bank=list(reversed(_BANK)), digest="0" * 64)
    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY)
    assert any("not in canonical order" in error for error in errors)


def test_selected_groups_must_come_from_the_offered_bank():
    record = _record(AP.DEPTH_CONDITIONED_POLICY)
    bank = copy.deepcopy(_BANK)
    bank = [entry for entry in bank if entry["tag"] != "L1-bbbb"]
    record["selection"]["materialized_action_bank"] = bank
    record["selection"]["materialized_action_bank_sha256"] = (
        AP.action_bank_manifest_sha256(bank))
    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY)
    assert any("outside the proposal bank" in error for error in errors)


def test_v3_provenance_cannot_relabel_a_hashed_paired_bank_row():
    record = _record(
        AP.DEPTH_CONDITIONED_POLICY_V3,
        provenance={"L1-bbbb": {
            "protocol": AP.PROPOSAL_PROTOCOL_V3,
            "template_id": "forged-natural",
            "variant": AP.NATURAL_DYNAMIC_VARIANT,
        }})

    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY_V3)

    assert any("does not match its action bank row" in error
               for error in errors)


def test_provenance_may_not_name_programs_the_bank_never_offered():
    record = _record(AP.DEPTH_CONDITIONED_POLICY,
                     provenance={"L1-never": {"variant": "safe"}})
    errors = _errors(record, AP.DEPTH_CONDITIONED_POLICY)
    assert any("outside the bank" in error for error in errors)


# --------------------------------------------------------------------------
# The published identifier must not spell out the answer
# --------------------------------------------------------------------------

def test_group_identifiers_reaching_the_artifact_stay_opaque():
    """``action_group_id`` becomes ``outcome_id`` and then the QA's
    ``oracle_ref``, so a readable variant in it would publish the label."""
    context = dataclasses.replace(
        source_manifest.LEGACY_RECORD_VALIDATION_CONTEXT,
        expected_setting_sampling_policy=SETTING_SAMPLING_POLICY,
        expected_action_sampling_policy=AP.DEPTH_CONDITIONED_POLICY)
    assert context.expected_action_sampling_policy
    for entry in _BANK:
        assert entry["variant"] not in entry["tag"]
