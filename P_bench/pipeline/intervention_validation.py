"""Cross-record intervention-group validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from pipeline import pose_setting
from pipeline import record as R

_POS = 1e-3
_FLOOR = 1e-9


def _physical_signature(outcome: dict):
    physical = outcome.get("physical", {})
    return (physical.get("collision"), physical.get("first_contact_arc_m"),
            physical.get("minimum_clearance_m"))


def _same_signature(a, b) -> bool:
    if a[0] != b[0]:
        return False
    for x, y in zip(a[1:], b[1:]):
        if x is None or y is None:
            if x != y:
                return False
        elif abs(float(x) - float(y)) > _POS:
            return False
    return True


def validate_intervention_groups(records: Iterable[dict]) -> List[str]:
    """Physical GT is invariant across every camera height and FOV sibling."""
    errors: List[str] = []
    groups = defaultdict(list)
    for record in records:
        group_id = (record.get("intervention") or {}).get("group_id")
        if group_id:
            groups[group_id].append(record)
    for group_id, siblings in groups.items():
        published = [
            sibling for sibling in siblings
            if (sibling.get("selection") or {}).get("setting_sampling_policy")
            == pose_setting.SETTING_SAMPLING_POLICY
        ]
        if published and len(siblings) != 1:
            errors.append(
                f"[{group_id}] one-setting pose publishes {len(siblings)} "
                "sensor siblings")
        # All sibling frames must carry the same action/body outcome membership
        # (the same action set is selected across every camera profile for the
        # sensor-consistency evaluation protocol).
        def membership(record):
            return sorted(
                (outcome.get("outcome_id"),
                 round(float(outcome.get("body", {}).get("radius_m", -1.0)), 6))
                for outcome in record.get("outcomes", []))
        baseline_membership = membership(siblings[0])
        if any(membership(sibling) != baseline_membership
               for sibling in siblings[1:]):
            errors.append(
                f"[{group_id}] action/body membership differs across siblings")
        baseline = {
            outcome.get("outcome_id"): _physical_signature(outcome)
            for outcome in siblings[0].get("outcomes", [])
        }
        for sibling in siblings[1:]:
            current = {
                outcome.get("outcome_id"): _physical_signature(outcome)
                for outcome in sibling.get("outcomes", [])
            }
            if (set(current) != set(baseline) or any(
                    not _same_signature(baseline[key], current[key])
                    for key in baseline)):
                errors.append(
                    f"[{group_id}] sensor-profile physical invariance violated")
                break
        errors.extend(_validate_group_floor_calibration(group_id, siblings))
    return errors


def _validate_group_floor_calibration(group_id: str, siblings) -> List[str]:
    """Siblings of one physical pose share one floor, exactly.
    Two rules, and they are not the same rule. The plane must be *bit-identical*
    across the group -- a per-profile fit may verify the canonical plane, never
    replace it, so "close enough" is a failure. The published height then must
    track the mount offset through that shared plane:
        h_eff_2 - h_eff_1 = n_y * (nominal_offset_2 - nominal_offset_1)
    which is why siblings differing only in FOV must publish the *same* height
    and siblings differing in mount height must publish different ones. The
    per-image floor estimate broke exactly this.
    """
    errors: List[str] = []
    def _number(value):
        return (float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None)
    # Baseline is the first sibling that has a usable calibration, not simply
    # the first sibling: anchoring on siblings[0] would let one corrupt record
    # silently switch off the whole group's floor checks. Records without one
    # are already reported individually.
    usable = []
    for sibling in siblings:
        plane, _reason = R.stored_floor_plane(sibling)
        offset = _number((sibling.get("sensor") or {}).get(
            "nominal_camera_offset_m"))
        height = _number(sibling.get("camera_height_above_visible_floor_m"))
        if plane is not None and offset is not None and height is not None:
            usable.append((plane, offset, height))
    if len(usable) < 2:
        return errors
    baseline_plane, reference_offset, reference_height = usable[0]
    for plane, offset, height in usable[1:]:
        if (plane.normal_local != baseline_plane.normal_local or
                plane.offset_m != baseline_plane.offset_m):
            errors.append(
                f"[{group_id}] siblings do not share one canonical floor plane")
            continue
        expected = baseline_plane.normal_local[1] * (offset - reference_offset)
        if abs((height - reference_height) - expected) > _FLOOR:
            errors.append(
                f"[{group_id}] effective height difference "
                f"{height - reference_height} does not match "
                f"n_y * offset difference {expected}")
    return errors
