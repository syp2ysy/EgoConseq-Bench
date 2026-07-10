"""InteriorGS 3D-bbox semantics for GS scenes (replaces the HM3D texture decode).

`labels.json` = list of {ins_id, label, bounding_box: [8 x {x,y,z}]} in the
InteriorGS **Z-up** frame. We rotate to the Habitat **Y-up** gaussian/navmesh
frame with (x,y,z)->(x, z, -y) (calibrated: same rotation wins the axis-permutation
coverage search on every scene, scale = 1), then translate to align the bbox union
to the gaussian cloud. `assign(world_points)` returns, per point, the instance id of
the **smallest containing AABB** (most specific object), or 0 if none.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np


def _rot_ig_to_habitat(p: np.ndarray) -> np.ndarray:
    """InteriorGS Z-up -> Habitat Y-up: (x, y, z) -> (x, z, -y)."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


class BboxSemanticIndex:
    """Per-point instance assignment by point-in-AABB (smallest box wins)."""

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, ids: np.ndarray,
                 id_to_cat: Dict[int, str]):
        self._mins = np.asarray(mins, np.float64)
        self._maxs = np.asarray(maxs, np.float64)
        self._ids = np.asarray(ids, np.int64)
        vol = np.prod(np.maximum(self._maxs - self._mins, 1e-6), axis=1)
        self._order = np.argsort(-vol)          # large -> small; small overwrites
        self.id_to_cat = id_to_cat

    def assign(self, world_points: np.ndarray) -> np.ndarray:
        P = np.asarray(world_points, np.float64)
        out = np.zeros(P.shape[0], np.int64)
        for bi in self._order:
            lo, hi = self._mins[bi], self._maxs[bi]
            m = ((P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) &
                 (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1]) &
                 (P[:, 2] >= lo[2]) & (P[:, 2] <= hi[2]))
            if m.any():
                out[m] = self._ids[bi]
        return out


def load_bbox_index(labels_path: str, align_points: Optional[np.ndarray] = None
                    ) -> BboxSemanticIndex:
    """Build a BboxSemanticIndex from an InteriorGS labels.json.

    `align_points` (the scene's gaussian means, Habitat frame) is used to estimate
    the per-scene translation by aligning the bbox-union centre to the gaussian
    robust (1-99 pct) centre. Instance ids are 1..N (0 = unlabelled).
    """
    lab = json.load(open(labels_path))
    mins, maxs, ids = [], [], []
    id_to_cat: Dict[int, str] = {}
    for i, o in enumerate(lab):
        bb = o.get("bounding_box")
        if not bb:
            continue
        c = _rot_ig_to_habitat(np.array([[q["x"], q["y"], q["z"]] for q in bb], float))
        iid = i + 1
        mins.append(c.min(0)); maxs.append(c.max(0)); ids.append(iid)
        id_to_cat[iid] = o["label"]
    mins = np.array(mins, float); maxs = np.array(maxs, float)
    ids = np.array(ids, np.int64)

    if align_points is not None and len(mins):
        g = np.asarray(align_points, float)
        gc = (np.percentile(g, 1, 0) + np.percentile(g, 99, 0)) / 2.0
        bc = (mins.min(0) + maxs.max(0)) / 2.0
        t = gc - bc
        mins = mins + t
        maxs = maxs + t

    return BboxSemanticIndex(mins, maxs, ids, id_to_cat)
