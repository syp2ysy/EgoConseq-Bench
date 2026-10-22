"""A3: the instance a geometric first contact touched, certified only when unique and visibly contacted.

Every contacted face point of a first contact (co-contacts within the geometry contact window) is
resolved by the dataset's instance authority: for R2R the exact MP3D semantic-PLY face universe,
independent of the collision mesh; for B1K the object that owns the contacted collision mesh. A3 is certified when all points resolve to one labelled
object instance and the initial image shows at least MIN_POINTS depth points of that instance within
CONTACT_REGION_M of the contact, inside the body band. GS has no instance geometry and never certifies
A3. Otherwise A3 is withheld with its reason; A1/A2 of the execution stay valid.
"""

from __future__ import annotations

import numpy as np

from pipeline import perception

CONTACT_REGION_M = .10
CONTACT_FACE_MAX_M = .05  # a contact point on the scene mesh must lie this close to its semantic face
MIN_POINTS = 10
NON_OBJECT_CATEGORIES = {"", "misc", "void", "unlabeled", "objects", "floor", "floors", "ceiling"}


def resolve(session, dataset: str, points, owners=None) -> list:
    """Instance evidence of every contact point of one observation (one batched authority query).

    B1K faces belong to one object's collision mesh, so ``owners`` (the object of every contacted face)
    is exact; R2R points are resolved against the semantic PLY.
    """
    if owners is not None:
        return [{"confirmed": True, "reason": "collision_mesh_owner", "instance_id": int(owner),
                 "contact_face_distance_m": 0.} for owner in owners]
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if dataset != "r2r" or not len(points):
        return [{"confirmed": False, "reason": f"{dataset}_has_no_instance_authority", "instance_id": None}
                for _ in points]
    witness = [int(w) for w in session.assign_instances(points)]
    labelled = [i for i, w in enumerate(witness) if w in session.id_to_cat]  # 0: no semantic point near the contact
    results = [{"confirmed": False, "reason": "no_semantic_instance_near_contact", "instance_id": None}
               for _ in points]
    for i, result in zip(labelled, session._sem.confirm_contact_instances([(witness[i], points[i]) for i in labelled])
                         if labelled else []):
        results[i] = result
    return results


def identify(dataset: str, frame, points, resolutions, body: dict, floor_y: float, id_to_cat) -> dict:
    """Return {'status': 'certified'|'withheld', ...}; never guesses a category."""
    if dataset == "gs":
        return {"status": "withheld", "reason": "gs_has_no_instance_geometry"}
    evidence = {"contact_points": len(points),
                "resolutions": sorted({r["reason"] for r in resolutions})}
    if not resolutions or any(not r["confirmed"] or r["contact_face_distance_m"] > CONTACT_FACE_MAX_M
                              for r in resolutions):
        return {"status": "withheld", "reason": "contact_instance_unconfirmed", "evidence": evidence}
    instances = sorted({int(r["instance_id"]) for r in resolutions})
    evidence["instances"] = instances
    if len(instances) != 1:
        return {"status": "withheld", "reason": "contact_ambiguous", "evidence": evidence}
    instance = instances[0]
    category = str(id_to_cat.get(instance, "") or "")
    if instance <= 0 or category.casefold() in NON_OBJECT_CATEGORIES:
        return {"status": "withheld", "reason": "contact_unlabelled", "evidence": evidence}
    mine = np.flatnonzero(np.asarray(frame.pts_sem) == instance)
    world = perception.world_from_local(frame.pts[mine], frame.position, frame.yaw_rad)
    height = world[:, 1] - floor_y
    band = (height > body["ground_clearance_m"]) & (height < body["top_m"])
    gap = np.min(np.linalg.norm(world[:, None, :] - np.asarray(points, dtype=float)[None], axis=2), axis=1) \
        if len(world) else np.zeros(0)
    visible = int(np.sum(band & (gap <= CONTACT_REGION_M)))
    evidence["visible_points"] = visible
    if visible < MIN_POINTS:
        return {"status": "withheld", "reason": "contact_region_not_visible", "evidence": evidence}
    return {"status": "certified", "instance_id": instance, "category": category, "evidence": evidence}
