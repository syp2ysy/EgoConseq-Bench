"""
egoconseq.eval.metrics
======================
Evaluation metrics for EgoConseq-Bench O5 (footprint counterfactual).

The key discriminative band is the ``small_only`` GT class — cases where the
narrow robot passes but the wide robot contacts an obstacle.  A model that
always outputs the same label ("both" or "neither") exhibits *embodiment
invariance* and fails precisely these cases.

Operationalization
------------------
All band metrics are computed over the ``small_only`` GT subset only.

narrow_band_flip_acc / correct_flip_rate:
    Fraction of GT ``small_only`` cases predicted exactly as ``small_only``.
    These two names refer to the same quantity; both are returned so callers
    can use whichever is more readable in context.

embodiment_sensitivity:
    Fraction of GT ``small_only`` cases where the model acknowledges a
    width-dependent difference.  Operationally identical to
    ``correct_flip_rate``: the model is *sensitive* to embodiment only when
    it outputs ``small_only``.  A model that always says "both" or "neither"
    has sensitivity = 0.

invariance_error:
    Fraction of GT ``small_only`` cases where the model outputs "both" or
    "neither" — i.e. treats the two bodies as indistinguishable.
    By construction: ``invariance_error + embodiment_sensitivity == 1.0``.

accuracy:
    Standard 3-way overall accuracy over all GT labels (reference metric).

Division-by-zero handling
-------------------------
When there are no ``small_only`` GT cases (including the empty-list case),
band metrics (``narrow_band_flip_acc``, ``embodiment_sensitivity``,
``correct_flip_rate``, ``invariance_error``) are set to ``float('nan')``.
Overall ``accuracy`` is set to ``float('nan')`` when the list is empty.
These values are *never* ``None`` — callers can use ``math.isnan()`` to
detect the undefined case.
"""

from __future__ import annotations

import math
from typing import List, Sequence


def o5_metrics(
    gt_labels: Sequence[str],
    pred_labels: Sequence[str],
) -> dict:
    """Compute O5 flip-discrimination metrics.

    Parameters
    ----------
    gt_labels:
        Ground-truth labels.  Each element must be one of
        ``{"both", "small_only", "neither"}``.
    pred_labels:
        Predicted labels from the model, same length as ``gt_labels``.

    Returns
    -------
    dict with keys:
        ``accuracy``, ``narrow_band_flip_acc``, ``correct_flip_rate``,
        ``embodiment_sensitivity``, ``invariance_error``.

    See module docstring for full operationalization.

    Raises
    ------
    ValueError
        If ``gt_labels`` and ``pred_labels`` have different lengths.
    """
    if len(gt_labels) != len(pred_labels):
        raise ValueError(
            f"gt_labels and pred_labels must have the same length; "
            f"got {len(gt_labels)} vs {len(pred_labels)}"
        )

    n_total = len(gt_labels)

    # ── overall accuracy ──────────────────────────────────────────────────────
    if n_total == 0:
        accuracy = float("nan")
    else:
        accuracy = sum(g == p for g, p in zip(gt_labels, pred_labels)) / n_total

    # ── band metrics (restricted to GT == "small_only") ───────────────────────
    small_only_indices = [i for i, g in enumerate(gt_labels) if g == "small_only"]
    n_band = len(small_only_indices)

    if n_band == 0:
        # No discriminative cases → all band metrics undefined
        nan = float("nan")
        return {
            "accuracy": accuracy,
            "narrow_band_flip_acc": nan,
            "correct_flip_rate": nan,
            "embodiment_sensitivity": nan,
            "invariance_error": nan,
        }

    # Counts over the small_only GT subset
    n_correct_flip = 0   # pred == "small_only"
    n_invariant = 0      # pred in {"both", "neither"}

    for i in small_only_indices:
        p = pred_labels[i]
        if p == "small_only":
            n_correct_flip += 1
        else:
            # "both" or "neither" → model treats bodies identically
            n_invariant += 1

    correct_flip_rate = n_correct_flip / n_band
    invariance_error = n_invariant / n_band

    # narrow_band_flip_acc and embodiment_sensitivity are the same quantity
    # as correct_flip_rate (see docstring); returned under their own keys for
    # readability.
    narrow_band_flip_acc = correct_flip_rate
    embodiment_sensitivity = correct_flip_rate

    return {
        "accuracy": accuracy,
        "narrow_band_flip_acc": narrow_band_flip_acc,
        "correct_flip_rate": correct_flip_rate,
        "embodiment_sensitivity": embodiment_sensitivity,
        "invariance_error": invariance_error,
    }
