"""The one-sensor-one-body contract must be enforced, not just written down.

`setting_sampling_policy` is a claim the collector makes about a record. These
tests pin the two checks that can falsify it: a per-record check that the body
grid really collapsed to one radius, and a cross-record check that the pose
published a single sensor sibling. The second one cannot live in
validate_record, which only ever sees one record at a time.
"""

from pipeline import config
from pipeline.intervention_validation import validate_intervention_groups
from pipeline.action_proposal import EXPLICIT_ACTION_FILE_POLICY
from pipeline.pose_setting import SETTING_SAMPLING_POLICY
from pipeline import validate


RADIUS = config.RADII_M[1]


def _errors(record, expected=SETTING_SAMPLING_POLICY, action_policy=None):
    """Validate a record the way a hash-verified run route would."""
    return validate._validate_balanced_selection(
        record, expected, action_policy)


def _outcome(radius=RADIUS, collision=False):
    return {
        "action_group_id": "g",
        "action_group_label": "collision" if collision else "safe",
        "seq_len": 1,
        "actions": [{"type": "forward", "m": 2.0}],
        "body": {"radius_m": radius},
        "physical": {"collision": collision},
    }


def _record(*, outcomes=None, required_radii=(RADIUS,), frame="f"):
    return {
        "frame_id": frame,
        "outcomes": list(outcomes if outcomes is not None else [_outcome()]),
        "intervention": {"group_id": "pose-1"},
        "selection": {
            "policy": EXPLICIT_ACTION_FILE_POLICY,
            "setting_sampling_policy": SETTING_SAMPLING_POLICY,
            "action_group_ids": ["g"],
            "action_group_labels": {"g": "safe"},
            "candidate_budget": config.ACTION_CANDIDATE_MIN_PER_POSE,
            "observed_lengths": [1],
            "observed_label_counts": {"safe": 1},
            "required_radii_m": [float(value) for value in required_radii],
        },
    }


def test_single_setting_record_is_accepted():
    assert _errors(_record()) == []


def test_dropping_the_marker_cannot_buy_the_weaker_legacy_contract():
    # The expectation comes from hash-verified run metadata, so a record
    # that simply omits the marker must fail rather than fall back to the
    # older multi-radius contract.
    record = _record(required_radii=config.RADII_M)
    record["selection"].pop("setting_sampling_policy")

    errors = _errors(record)

    assert any("setting_sampling_policy is None" in error
               for error in errors), errors


def test_a_record_without_any_selection_still_owes_the_policy():
    record = _record()
    record["selection"] = None

    assert _errors(record) != []


def test_claiming_a_policy_the_run_never_used_is_rejected():
    assert any("setting_sampling_policy" in error
               for error in _errors(_record(), expected=None))


def test_single_setting_record_must_declare_exactly_one_radius():
    record = _record(required_radii=config.RADII_M)

    errors = _errors(record)

    assert any("one body radius" in error for error in errors), errors


def test_single_setting_record_outcomes_must_use_the_declared_radius():
    other = [radius for radius in config.RADII_M if radius != RADIUS][0]
    record = _record(outcomes=[_outcome(), _outcome(radius=other)])

    errors = _errors(record)

    assert any("does not use the published body radius" in error
               for error in errors), errors


def test_a_pose_may_publish_only_one_sensor_sibling():
    # Two records sharing an intervention group is exactly the shape the old
    # height x FOV cross product produced, so this is the check that tells the
    # two sampling regimes apart.
    siblings = [_record(frame="f-a"), _record(frame="f-b")]

    errors = validate_intervention_groups(siblings)

    assert any("publishes 2 sensor siblings" in error
               for error in errors), errors


def test_legacy_records_without_the_policy_keep_the_old_contract():
    legacy = _record(required_radii=config.RADII_M)
    legacy["selection"].pop("setting_sampling_policy")
    legacy["outcomes"] = [
        _outcome(radius=radius) for radius in config.RADII_M]
    legacy["selection"]["observed_label_counts"] = {"safe": 1}

    assert _errors(legacy, expected=None) == []
    assert validate_intervention_groups(
        [legacy, dict(legacy, frame_id="f-b")]) == []
