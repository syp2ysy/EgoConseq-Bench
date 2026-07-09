"""Offline HM3D semantic decode + runtime per-point instance assignment.

WHY: this habitat-sim 0.2.4 build cannot render HM3D-v0.2 texture semantics
(the SEMANTIC sensor returns all zeros and obj.aabb is empty), even though the
semantic mesh loads. The instance labels live only in the semantic.glb texture.

We therefore decode instances OFFLINE, once per scene:
  1. parse `<scene>.semantic.txt`  -> palette (colour -> instance id, category)
  2. load `<scene>.semantic.glb`, per face sample the texture at its UV centroid
     -> nearest palette colour -> instance id (~94% face coverage)
  3. area-weighted surface-sample the labelled faces -> labelled points, and
     transform mesh frame (Z-up) -> Habitat world frame: world = (x, z, -y)
     (verified empirically: median depth<->semantic nearest distance ~0.16 m)
  4. cache the labelled world points + instance ids to an .npz

At runtime, SemanticIndex assigns each depth-derived world point to the nearest
labelled surface point within a tolerance (0 = unlabelled), reproducing a
depth-aligned instance mask without the broken sensor.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from pipeline import config


# --------------------------------------------------------------------------
# semantic.txt palette
# --------------------------------------------------------------------------

def _paths(scene_glb: str):
    stem = scene_glb.replace(".basis.glb", "")
    return stem + ".semantic.glb", stem + ".semantic.txt"


def parse_palette(txt_path: str):
    """Return (ids (N,), colours (N,3) float, id_to_cat)."""
    ids, cols, id_to_cat = [], [], {}
    with open(txt_path) as f:
        for ln in f:
            p = ln.strip().split(",")
            if len(p) >= 3 and p[0].isdigit():
                iid = int(p[0]); h = p[1]
                ids.append(iid)
                cols.append([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)])
                id_to_cat[iid] = p[2].strip().strip('"')
    return np.array(ids), np.array(cols, dtype=np.float64), id_to_cat


# --------------------------------------------------------------------------
# offline decode
# --------------------------------------------------------------------------

def _decode_labelled_points(semantic_glb: str, ids, cols,
                            sample_count: int, match_tol: float, seed: int = 0):
    """Decode per-face instances from the texture, then surface-sample.

    Returns (points_world (M,3) float32, inst (M,) int32).
    """
    import trimesh

    scene = trimesh.load(semantic_glb, process=False)
    geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
    paltree = cKDTree(cols)

    face_inst_all = []
    meshes = []
    for g in geoms:
        F = np.asarray(g.faces)
        if F.shape[0] == 0:
            continue
        uv = getattr(g.visual, "uv", None)
        mat = getattr(g.visual, "material", None)
        tex = getattr(mat, "baseColorTexture", None) if mat is not None else None
        if uv is None or tex is None:
            face_inst_all.append(np.zeros(F.shape[0], dtype=np.int64))
            meshes.append(g)
            continue
        uv = np.asarray(uv)
        tex = np.asarray(tex)[..., :3]
        th, tw = tex.shape[:2]
        fuv = uv[F].mean(axis=1)                       # (nf, 2)
        tx = np.clip((fuv[:, 0] * tw).astype(int), 0, tw - 1)
        ty = np.clip(((1 - fuv[:, 1]) * th).astype(int), 0, th - 1)
        fc = tex[ty, tx].astype(np.float64)            # (nf, 3)
        d, idx = paltree.query(fc, k=1)
        inst = np.where((d < match_tol) & ~np.all(fc < 8, axis=1), ids[idx], 0)
        face_inst_all.append(inst.astype(np.int64))
        meshes.append(g)

    mesh = trimesh.util.concatenate(meshes)
    face_inst = np.concatenate(face_inst_all)

    pts, face_idx = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    pts = np.asarray(pts); inst = face_inst[face_idx]
    keep = inst != 0
    pts, inst = pts[keep], inst[keep]

    # mesh frame (Z-up, front=+Y) -> Habitat world (Y-up): world = (x, z, -y)
    world = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1).astype(np.float32)
    return world, inst.astype(np.int32)


# --------------------------------------------------------------------------
# public: build (cached) + runtime index
# --------------------------------------------------------------------------

class SemanticIndex:
    """Nearest-labelled-surface instance assigner for a scene."""

    def __init__(self, points_world: np.ndarray, inst: np.ndarray,
                 id_to_cat: Dict[int, str]):
        self.id_to_cat = id_to_cat
        self._inst = inst
        self._tree = cKDTree(points_world) if points_world.shape[0] else None

    def assign(self, world_points: np.ndarray,
               tol: float = config.SEMANTIC_ASSIGN_TOL_M) -> np.ndarray:
        """world_points (N,3) -> instance id per point (0 if none within tol)."""
        if self._tree is None:
            return np.zeros(world_points.shape[0], dtype=np.int64)
        d, idx = self._tree.query(np.asarray(world_points), k=1)
        return np.where(d <= tol, self._inst[idx], 0).astype(np.int64)


def load_semantic_index(scene_glb: str,
                        cache_dir: str = config.SEMANTIC_CACHE_DIR,
                        sample_count: int = config.SEMANTIC_SAMPLE_COUNT,
                        match_tol: float = config.SEMANTIC_MATCH_TOL,
                        rebuild: bool = False) -> SemanticIndex:
    """Load (or build+cache) the semantic index for a scene."""
    sem_glb, sem_txt = _paths(scene_glb)
    ids, cols, id_to_cat = parse_palette(sem_txt)

    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.basename(scene_glb).replace(".basis.glb", "")
    cache = os.path.join(cache_dir, stem + ".sem.npz")

    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        return SemanticIndex(z["points"], z["inst"], id_to_cat)

    points, inst = _decode_labelled_points(sem_glb, ids, cols, sample_count, match_tol)
    np.savez_compressed(cache, points=points, inst=inst)
    return SemanticIndex(points, inst, id_to_cat)
