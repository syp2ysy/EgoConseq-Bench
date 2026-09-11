"""Characterisation lock for the error inventory of validate_record.

`_validate_outcome_execution` is 300 lines and is about to be split by
responsibility. A split must not change which errors are reported, how they are
worded, or the order they arrive in, and the module lost its dedicated suite
when the legacy stack was removed -- so "no test asserts this" cannot be used
to justify dropping a check.

The synthetic record below is deliberately not publishable: it trips several
validators at once, which is what makes it useful. Each test pins either the
whole ordered inventory or the exact delta one corruption introduces.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline import record as record_fields
from pipeline import source_manifest, validate
from pipeline.pose_setting import SETTING_SAMPLING_POLICY
from tests._synthetic import _outcome, _record, source_provenance


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_MANIFEST = ROOT / "tests" / "fixtures" / "abc_golden.json"


BASELINE_ERRORS = [
    "[f] f: floor calibration is missing; cannot rebuild this frame",
    "[f] sensor profile violates formal FOV calibration",
    "[f:o1] physical authority 'test_geometry' does not match source dataset "
    "'r2r'",
    "[f:o1] strict shared-oracle consensus marker is missing",
    "[f:o1] stored shared-oracle consensus differs from reconstruction: "
    "contact_instance_witness_required",
    "[f:o1] A stability certificate is missing",
]


def _context(
        source, collection_mode="main", setting_sampling_policy=None):
    return source_manifest.r2r_v16_registered_validation_context(
        [source],
        collection_mode=collection_mode,
        expected_schema_version=record_fields.SCHEMA_VERSION,
        expected_oracle_contract_version=(
            record_fields.ORACLE_CONTRACT_VERSION),
        authority_sha256="a" * 64,
        setting_sampling_policy=setting_sampling_policy,
    )


def _base_record():
    rec = _record(_outcome())
    rec["oracle_contract_version"] = record_fields.ORACLE_CONTRACT_VERSION
    rec["source"] = source_provenance("s", dataset="r2r")
    rec["collection_contract"] = record_fields.r2r_v16_collection_contract(
        rec["source"], "main")
    return rec


def _validate(rec, source):
    return validate.validate_record_local(rec, context=_context(source))


def _delta(mutate):
    base = _base_record()
    source = base["source"]
    baseline = _validate(base, source)
    corrupted = copy.deepcopy(base)
    mutate(corrupted)
    errors = _validate(corrupted, source)
    added = [error for error in errors if error not in baseline]
    removed = [error for error in baseline if error not in errors]
    return added, removed, [errors.index(error) for error in added]


def test_baseline_error_inventory_and_order_are_stable():
    base = _base_record()

    assert _validate(base, base["source"]) == BASELINE_ERRORS


def _set_coverage_protocol(rec, value):
    rec["outcomes"][0]["evidence"]["physical"]["coverage_protocol"] = value


def test_frozen_v5_evidence_protocol_still_validates():
    """v5 artefacts are read-only history and must stay verifiable.

    The v6 near-field wedge changed the geometry, not the record shape, so a
    record collected under v5 carries its own protocol and is accepted against
    that. Rejecting it would strand every frozen Golden shard.
    """
    added, removed, _ = _delta(
        lambda rec: _set_coverage_protocol(rec, "swept_floor_v5"))
    assert (added, removed) == ([], [])


def test_unknown_evidence_protocol_is_still_rejected():
    added, removed, _ = _delta(
        lambda rec: _set_coverage_protocol(rec, "swept_floor_v99"))
    assert added == [
        "[f:o1] unsupported physical evidence protocol 'swept_floor_v99'"]
    assert removed == []


def test_validate_record_enforces_the_authenticated_setting_policy():
    rec = _base_record()
    rec["selection"] = {
        "setting_sampling_policy": SETTING_SAMPLING_POLICY,
        "required_radii_m": [0.2],
    }
    context = _context(
        rec["source"], setting_sampling_policy=SETTING_SAMPLING_POLICY)

    errors = validate.validate_record_local(rec, context=context)

    assert not [
        error for error in errors if "setting_sampling_policy" in error]


def test_v10_rejects_every_retired_v9_branch():
    rec = _base_record()
    outcome = rec["outcomes"][0]
    rec["review_evidence"] = {}
    outcome["future_state"] = {}
    outcome["target_projections"] = {}
    outcome["evidence"]["goal"] = {}

    errors = _validate(rec, rec["source"])

    assert "[f] retired v9 record fields are forbidden: ['review_evidence']" \
        in errors
    assert (
        "[f:o1] retired v9 outcome fields are forbidden: "
        "['future_state', 'target_projections']") in errors
    assert "[f:o1] retired v9 evidence fields are forbidden: ['goal']" \
        in errors


def test_missing_body_radius_adds_exactly_one_error_in_place():
    added, removed, indexes = _delta(
        lambda rec: rec["outcomes"][0]["body"].pop("radius_m", None))

    assert added == [
        "[f:o1] base rollout key disagrees with record inputs",
        "[f:o1] body must be a radius-only disc",
    ]
    assert removed == []
    assert indexes == [2, 3]


def test_wrong_source_binding_reports_before_any_outcome_error():
    added, removed, indexes = _delta(
        lambda rec: rec["source"].__setitem__("scene_id", "other-scene"))

    assert added == ["[f] source scene_id does not match record"]
    assert removed == []
    assert indexes == [0]


def test_dropped_shared_certificate_reports_reconstruction_and_consensus():
    added, removed, indexes = _delta(
        lambda rec: rec["outcomes"][0].pop("oracle_consensus", None))

    assert added == [
        "[f:o1] stored shared-oracle consensus differs from reconstruction: "
        "accepted, verdict, reason, full_authority, depth_authority, "
        "full_collision, depth_collision, full_contact_arc_m, "
        "depth_contact_arc_m, contact_arc_difference_m, contact_tolerance_m, "
        "corridor_coverage, coverage_min, contact_instance_witness_required",
        "[f:o1] non-consensus outcome lacks accepted consensus",
    ]
    assert removed == [
        "[f:o1] stored shared-oracle consensus differs from reconstruction: "
        "contact_instance_witness_required",
    ]
    assert indexes == [4, 6]


def test_flipped_collision_flag_reports_the_full_execution_inconsistency():
    added, removed, indexes = _delta(
        lambda rec: rec["outcomes"][0]["physical"].__setitem__(
            "collision", True))

    assert added == [
        "[f:o1] execution regime cannot be derived: collision execution state "
        "is inconsistent",
        "[f:o1] navmesh configuration boundary point is missing",
        "[f:o1] reconstructed shared-oracle consensus rejected: "
        "collision_state_mismatch",
        "[f:o1] stored shared-oracle consensus differs from reconstruction: "
        "accepted, verdict, reason, full_collision, "
        "contact_instance_witness_required",
        "[f:o1] collision execution state is inconsistent",
        "[f:o1] collision lacks stop location",
        "[f:o1] collision execution lacks stop_arc_m",
    ]
    assert removed == [
        "[f:o1] stored shared-oracle consensus differs from reconstruction: "
        "contact_instance_witness_required",
    ]
    assert indexes == [2, 4, 6, 7, 9, 10, 11]


def _golden_shards():
    if not GOLDEN_MANIFEST.exists():
        return []
    manifest = json.loads(GOLDEN_MANIFEST.read_text())
    shards = []
    for entry in manifest["inputs"]:
        shard = ROOT / entry["shard"]
        run_meta = shard / "run_meta.json"
        if (not (shard / "records.jsonl").exists() or
                not run_meta.exists() or
                json.loads(run_meta.read_text()).get(
                    "record_schema_version") != record_fields.SCHEMA_VERSION):
            continue
        if (shard / "records.jsonl").exists():
            shards.append((shard, entry["run_meta.json"]))
    return shards


@pytest.mark.parametrize("shard,run_meta_sha256", _golden_shards())
def test_golden_shards_validate_without_violations(shard, run_meta_sha256):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_records.py"),
         str(shard / "records.jsonl"),
         "--validation-level", "source",
         "--run-meta", str(shard / "run_meta.json"),
         "--expected-run-meta-sha256", run_meta_sha256],
        cwd=ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "0 violations" in completed.stdout
