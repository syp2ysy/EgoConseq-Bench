"""
egoconseq.eval.baselines
========================
Non-vision shortcut baselines for O5 single-robot binary contact groups.

These baselines intentionally ignore RGB. They exist to check whether the O5
cases can be solved from label priors or diameter alone.
"""

from __future__ import annotations

import random as _random
from collections import Counter
from typing import Mapping, Optional, Sequence

from egoconseq.eval.metrics import o5_metrics

O5_LABELS = ("contact", "no_contact")


def random_baseline(
    n: int,
    labels: Sequence[str] = O5_LABELS,
    seed: Optional[int] = None,
) -> list[str]:
    """Return n uniformly random binary contact predictions."""
    rng = _random.Random(seed)
    return [rng.choice(list(labels)) for _ in range(n)]


def majority_baseline(gt_labels: Sequence[str]) -> list[str]:
    """Predict the most frequent binary label for every case."""
    if not gt_labels:
        return []
    counts = Counter(gt_labels)
    majority = max(counts, key=lambda k: (counts[k], k))
    return [majority] * len(gt_labels)


def blind_text_only(n: int, fixed_label: str = "no_contact") -> list[str]:
    """Always output fixed_label regardless of image/question content."""
    return [fixed_label] * n


def radius_only(
    cases: Sequence[Mapping[str, object]],
    threshold_radius_m: float = 0.25,
) -> list[str]:
    """Predict contact from diameter/radius alone.

    This baseline says wide chassis contacts and narrow chassis does not,
    without looking at the scene. It can look strong on true flip groups, but
    should fail on groups where all diameters share the same outcome.
    """
    preds = []
    for case in cases:
        radius = float(case.get("radius_m", 0.0))
        preds.append("contact" if radius >= threshold_radius_m else "no_contact")
    return preds


def _flatten_groups(groups: Sequence[Sequence[Mapping[str, object]]]) -> list[dict]:
    return [dict(case) for group in groups for case in group]


def _with_predictions(
    groups: Sequence[Sequence[Mapping[str, object]]],
    preds: Sequence[str],
) -> list[list[dict]]:
    out: list[list[dict]] = []
    idx = 0
    for group in groups:
        out_group: list[dict] = []
        for case in group:
            updated = dict(case)
            updated["pred"] = preds[idx]
            out_group.append(updated)
            idx += 1
        out.append(out_group)
    return out


def evaluate_all_baselines(
    groups: Sequence[Sequence[Mapping[str, object]]],
    seed: int = 42,
) -> dict:
    """Run all non-vision O5 baselines and return metric dicts."""
    flat = _flatten_groups(groups)
    gt_labels = [str(c.get("gt")) for c in flat]
    n = len(flat)

    results: dict = {}

    random_preds = random_baseline(n, seed=seed)
    results["random"] = o5_metrics(_with_predictions(groups, random_preds))

    majority_preds = majority_baseline(gt_labels)
    results["majority"] = o5_metrics(_with_predictions(groups, majority_preds))

    blind_preds = blind_text_only(n)
    results["blind_text_only"] = o5_metrics(_with_predictions(groups, blind_preds))

    radius_preds = radius_only(flat)
    results["radius_only"] = o5_metrics(_with_predictions(groups, radius_preds))

    results["geometry_oracle"] = o5_metrics(_with_predictions(groups, gt_labels))

    return results
