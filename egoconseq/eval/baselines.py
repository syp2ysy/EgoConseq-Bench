"""
egoconseq.eval.baselines
========================
Non-vision shortcut baselines for EgoConseq-Bench O5 (footprint counterfactual).

These baselines intentionally IGNORE the RGB image.  Their purpose is to
confirm that the O5 benchmark cannot be solved without vision:

  random_baseline
      Picks uniformly at random from {"both", "small_only", "neither"}.
      Expected accuracy = 1/3 ≈ 0.33.  Narrow-band flip accuracy ≈ 1/3.

  majority_baseline
      Always predicts the most frequent label in the *provided* label list.
      When called with gt_labels, it returns the mode; when called with
      "both" as the majority label (unknown at test time), accuracy = label
      fraction of "both".  Upper bound for a label-frequency shortcut.

  blind_text_only
      Reads only the question text (no image) and always outputs a FIXED
      predicted label regardless of content.  Default fixed label = "both"
      (the most common label in typical generation).  This models a VLM that
      has learned to ignore image content and always says "both pass".

  radius_only
      Decides based solely on the two radii (r_small, r_large) provided in
      the question text — cannot know the gap in the scene, so effectively
      picks a fixed output.  Here implemented as always predicting "small_only"
      (the hypothesis that *body width always matters*), which is incorrect
      whenever both/neither is the true label.

NOTE: "geometry_oracle" trivially scores 100% since labels are derived from
geometry; it is computed externally by comparing gt_labels with themselves.
"""

from __future__ import annotations

import random as _random
from collections import Counter
from typing import List, Optional, Sequence

from egoconseq.eval.metrics import o5_metrics


# ---------------------------------------------------------------------------
# Individual prediction functions
# ---------------------------------------------------------------------------

def random_baseline(
    n: int,
    labels: Sequence[str] = ("both", "small_only", "neither"),
    seed: Optional[int] = None,
) -> List[str]:
    """Return *n* labels sampled uniformly at random from *labels*.

    Parameters
    ----------
    n : int
        Number of predictions to generate.
    labels : sequence of str
        The label vocabulary.  Default: {"both", "small_only", "neither"}.
    seed : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    list of str
    """
    rng = _random.Random(seed)
    return [rng.choice(list(labels)) for _ in range(n)]


def majority_baseline(
    gt_labels: Sequence[str],
) -> List[str]:
    """Predict the most frequent label in *gt_labels* for every sample.

    Parameters
    ----------
    gt_labels : sequence of str
        Ground-truth labels used to determine the majority class.
        At test-time this represents the distribution we compute from
        a development set; here we compute it from gt_labels directly
        (upper-bound shortcut).

    Returns
    -------
    list of str
        A list of length ``len(gt_labels)`` where every element is the
        majority label.
    """
    if not gt_labels:
        return []
    counts = Counter(gt_labels)
    # Break ties consistently (alphabetical, since Counter.most_common is stable
    # in insertion order only for equal counts in Python 3.7+; sort explicitly).
    majority = max(counts, key=lambda k: (counts[k], k))
    return [majority] * len(gt_labels)


def blind_text_only(
    n: int,
    fixed_label: str = "both",
) -> List[str]:
    """Always output *fixed_label* regardless of image or question content.

    This models a model that ignores visual evidence and always predicts the
    most "obvious" label ("both" — both robots pass).

    Parameters
    ----------
    n : int
        Number of predictions.
    fixed_label : str
        The fixed predicted label for every case.

    Returns
    -------
    list of str
    """
    return [fixed_label] * n


def radius_only(
    n: int,
    fixed_label: str = "small_only",
) -> List[str]:
    """Always predict *fixed_label* regardless of scene geometry.

    This models a reasoner that infers "the larger body always gets stuck in
    narrow gaps" but cannot determine whether the *current scene* actually has
    a narrow gap that matters.  Implemented as always predicting "small_only".

    Parameters
    ----------
    n : int
        Number of predictions.
    fixed_label : str
        The fixed predicted label.  Default "small_only" (the hypothesis that
        body width always distinguishes outcomes — true only in narrow-gap cases).

    Returns
    -------
    list of str
    """
    return [fixed_label] * n


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def evaluate_all_baselines(
    gt_labels: Sequence[str],
    seed: int = 42,
) -> dict:
    """Run all non-vision baselines against *gt_labels* and return a results dict.

    Parameters
    ----------
    gt_labels : sequence of str
        Ground-truth labels (the test set).
    seed : int
        RNG seed for the random baseline.

    Returns
    -------
    dict mapping baseline_name -> o5_metrics output dict.
    """
    n = len(gt_labels)
    results: dict = {}

    # 1. random
    random_preds = random_baseline(n, seed=seed)
    results["random"] = o5_metrics(gt_labels, random_preds)

    # 2. majority
    majority_preds = majority_baseline(gt_labels)
    results["majority"] = o5_metrics(gt_labels, majority_preds)

    # 3. blind_text_only (always "both")
    blind_preds = blind_text_only(n, fixed_label="both")
    results["blind_text_only"] = o5_metrics(gt_labels, blind_preds)

    # 4. radius_only (always "small_only")
    radius_preds = radius_only(n, fixed_label="small_only")
    results["radius_only"] = o5_metrics(gt_labels, radius_preds)

    # 5. geometry oracle (trivially 100% — labels come from geometry)
    results["geometry_oracle"] = o5_metrics(gt_labels, list(gt_labels))

    return results
