"""
egoconseq.eval.metrics
======================
Evaluation metrics for O5 single-robot binary contact groups.

O5 now uses one robot per question:
    label = "contact" iff d_safe(radius, action) < H else "no_contact"

Counterfactual sensitivity is measured across cases sharing a group_id:
same RGB / pose / action horizon, different chassis diameters.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def _nan() -> float:
    return float("nan")


def o5_metrics(groups: Sequence[Sequence[Mapping[str, object]]]) -> dict:
    """Compute O5 binary-contact and group-flip metrics.

    Parameters
    ----------
    groups:
        Sequence of counterfactual groups. Each group is a sequence of dicts
        containing:
            radius_m: float
            gt: "contact" | "no_contact"
            pred: "contact" | "no_contact"

    Returns
    -------
    dict
        case_accuracy:
            Fraction of individual cases whose prediction matches GT.
        false_safe_rate:
            Among GT contact cases, fraction predicted no_contact.
        flip_groups:
            Number of groups where GT labels differ across radii.
        correct_flip_rate:
            Among flip groups, fraction whose every case is predicted
            correctly.
        invariance_error:
            Among flip groups, fraction where all predictions are identical,
            i.e. the model did not react to chassis diameter.

    Compatibility aliases are included for older reporting code:
        accuracy = case_accuracy
        narrow_band_flip_acc = correct_flip_rate
        embodiment_sensitivity = 1 - invariance_error when defined
    """
    flat = [case for group in groups for case in group]

    if not flat:
        case_accuracy = _nan()
    else:
        case_accuracy = sum(c.get("gt") == c.get("pred") for c in flat) / len(flat)

    contact_cases = [c for c in flat if c.get("gt") == "contact"]
    if not contact_cases:
        false_safe_rate = _nan()
    else:
        false_safe_rate = (
            sum(c.get("pred") == "no_contact" for c in contact_cases)
            / len(contact_cases)
        )

    flip_groups = []
    for group in groups:
        gt_labels = {c.get("gt") for c in group}
        if len(gt_labels) >= 2:
            flip_groups.append(group)

    if not flip_groups:
        correct_flip_rate = _nan()
        invariance_error = _nan()
        embodiment_sensitivity = _nan()
    else:
        correct_flip_rate = (
            sum(all(c.get("gt") == c.get("pred") for c in group) for group in flip_groups)
            / len(flip_groups)
        )
        invariance_error = (
            sum(len({c.get("pred") for c in group}) == 1 for group in flip_groups)
            / len(flip_groups)
        )
        embodiment_sensitivity = 1.0 - invariance_error

    return {
        "case_accuracy": case_accuracy,
        "false_safe_rate": false_safe_rate,
        "flip_groups": len(flip_groups),
        "correct_flip_rate": correct_flip_rate,
        "invariance_error": invariance_error,
        # Backward-compatible report aliases.
        "accuracy": case_accuracy,
        "narrow_band_flip_acc": correct_flip_rate,
        "embodiment_sensitivity": embodiment_sensitivity,
    }
