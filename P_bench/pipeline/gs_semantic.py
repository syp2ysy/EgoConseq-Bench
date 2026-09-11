"""Source-bound InteriorGS bbox semantics in the Habitat world frame.

InteriorGS labels are Z-up.  The official Habitat-GS conversion maps
``(x, y, z) -> (x, z, -y)`` with zero translation.  A visible depth point
belongs to an instance only when exactly one transformed bbox contains it;
overlaps and gaps remain unlabelled.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from pipeline import config, dataset_contracts


AXIS_TRANSFORM_PROTOCOL = "interiorgs-zup-to-habitat-yup.v1"
ALIGNMENT_CERTIFICATE_SCHEMA = \
    dataset_contracts.GS_OFFICIAL_COORDINATE_BINDING_SCHEMA
SCENE_CAPABILITY_SCHEMA = "egoconseq.gs-scene-capability.v1"
VISIBLE_CONTACT_IDENTITY_SCHEMA = \
    "gs-visible-contact-instance-identity.v1"
VISIBLE_INSTANCE_PROTOCOL = "initial-visible-depth-instance-points.v1"
_FROZEN_AXIS_MATRIX = np.asarray([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.int64)
_POSITIVE_INSTANCE_ID = re.compile(r"[1-9][0-9]*\Z")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def complete_visible_instance_universe_sha256(
        *, geometry_authority_sha256: str, semantic_source_sha256: str,
        alignment_certificate_sha256: str) -> str:
    """Hash the source-bound universe used by one visible A3 proof."""
    return _canonical_sha256({
        "schema": "gs-visible-instance-universe-proof.v1",
        "global_query_protocol": VISIBLE_INSTANCE_PROTOCOL,
        "geometry_authority_sha256": str(geometry_authority_sha256),
        "semantic_source_sha256": str(semantic_source_sha256),
        "alignment_certificate_sha256": str(
            alignment_certificate_sha256),
    })


def _rot_ig_to_habitat(points: np.ndarray) -> np.ndarray:
    """InteriorGS Z-up -> Habitat Y-up under the frozen axis protocol."""
    values = np.asarray(points, dtype=np.float64)
    return values @ _FROZEN_AXIS_MATRIX.T


def _source_instance_id(value: object) -> int:
    """Parse one positive, source-authored InteriorGS instance identifier."""
    if isinstance(value, bool):
        raise ValueError("GS label instance ids are invalid")
    if isinstance(value, int):
        instance_id = value
    elif isinstance(value, str) and _POSITIVE_INSTANCE_ID.fullmatch(value):
        instance_id = int(value)
    else:
        raise ValueError("GS label instance ids are invalid")
    if instance_id <= 0:
        raise ValueError("GS label instance ids are invalid")
    return instance_id


def _unique_bbox_assignments(
        points: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
        ids: np.ndarray) -> np.ndarray:
    """Assign points contained by exactly one AABB; zero means unlabelled."""
    points = np.asarray(points, dtype=np.float64)
    output = np.zeros(len(points), dtype=np.int64)
    hits = np.zeros(len(points), dtype=np.uint16)
    if not len(points) or not len(ids):
        return output
    tree = cKDTree(points)
    centers = (mins + maxs) / 2.0
    radii = np.linalg.norm((maxs - mins) / 2.0, axis=1)
    for index, instance_id in enumerate(ids):
        candidates = np.asarray(
            tree.query_ball_point(centers[index], radii[index]),
            dtype=np.int64)
        if not len(candidates):
            continue
        values = points[candidates]
        inside = np.all(
            (values >= mins[index]) & (values <= maxs[index]), axis=1)
        selected = candidates[inside]
        first = selected[hits[selected] == 0]
        output[first] = int(instance_id)
        hits[selected] += 1
        output[selected[hits[selected] > 1]] = 0
    return output


def build_alignment_certificate(
        label_boxes_z_up: np.ndarray, instance_ids: np.ndarray) -> dict:
    """Bind official InteriorGS labels to Habitat's published frame.

    Habitat-GS's official InteriorGS ObjectNav implementation defines the
    source conversion directly as ``(x, y, z) -> (x, z, -y)`` and applies no
    fitted translation.  The previous implementation guessed among signed
    axes using Gaussian occupancy; that heuristic rejected valid official
    scenes and was not an authority.  Source hashes bind the labels, render
    PLY, navmesh, and official collision artifact as one scene bundle.
    """
    boxes = np.asarray(label_boxes_z_up, dtype=np.float64)
    ids = np.asarray(instance_ids, dtype=np.int64)
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError("GS alignment boxes must have shape (B, 8, 3)")
    if (not len(boxes) or len(boxes) != len(ids) or
            len(set(ids.tolist())) != len(ids) or
            np.any(ids <= 0) or not np.isfinite(boxes).all()):
        raise ValueError("GS alignment instance ids are invalid")
    transformed = _rot_ig_to_habitat(boxes)
    certificate = {
        "schema": ALIGNMENT_CERTIFICATE_SCHEMA,
        "axis_transform_protocol": AXIS_TRANSFORM_PROTOCOL,
        "coordinate_authority": "habitat-gs-interiorgs-objectnav.v1",
        "matrix": _FROZEN_AXIS_MATRIX.tolist(),
        "translation_m": [0.0, 0.0, 0.0],
        "instance_count": int(len(ids)),
        "habitat_bounds_m": {
            "minimum": [float(value) for value in transformed.min(axis=(0, 1))],
            "maximum": [float(value) for value in transformed.max(axis=(0, 1))],
        },
        "accepted": True,
    }
    certificate["sha256"] = _canonical_sha256(certificate)
    return certificate


def validate_alignment_certificate(
        label_boxes_z_up: np.ndarray, instance_ids: np.ndarray,
        certificate: dict) -> None:
    """Rebuild and compare one per-scene GS alignment certificate."""
    expected = build_alignment_certificate(label_boxes_z_up, instance_ids)
    if certificate != expected:
        raise ValueError("GS alignment certificate does not match its source")


def _validated_stored_alignment_certificate(certificate) -> dict | None:
    if certificate is None:
        return None
    if (not isinstance(certificate, dict) or
            certificate.get("schema") != ALIGNMENT_CERTIFICATE_SCHEMA or
            not isinstance(certificate.get("accepted"), bool)):
        raise ValueError("GS alignment certificate is invalid")
    body = {key: value for key, value in certificate.items()
            if key != "sha256"}
    if certificate.get("sha256") != _canonical_sha256(body):
        raise ValueError("GS alignment certificate digest is invalid")
    return json.loads(json.dumps(certificate, sort_keys=True))


def scene_capability_atom(source: dict, alignment_certificate) -> dict:
    """Bind per-scene semantic task availability to the GS source bundle."""
    if (not isinstance(source, dict) or
            source.get("source_dataset") != "gs" or
            not isinstance(source.get("scene_id"), str) or
            not source.get("scene_id") or
            not isinstance(source.get("source_assets_sha256"), str) or
            re.fullmatch(
                r"[0-9a-f]{64}", source["source_assets_sha256"]) is None):
        raise ValueError("GS scene capability source is invalid")
    certificate = _validated_stored_alignment_certificate(
        alignment_certificate)
    accepted = certificate is not None and certificate["accepted"] is True
    semantic_status = (
        {"status": "available"} if accepted else {
            "status": "unavailable",
            "reason": (
                "semantic_alignment_missing" if certificate is None else
                "semantic_alignment_rejected"),
        })
    value = {
        "schema": SCENE_CAPABILITY_SCHEMA,
        "scene_id": source["scene_id"],
        "source_assets_sha256": source["source_assets_sha256"],
        "alignment_certificate": certificate,
        "task_statuses": {
            "A1": {"status": "available"},
            "A2": {"status": "available"},
            "A3": dict(semantic_status),
            "B1": dict(semantic_status),
            "B2": dict(semantic_status),
            "C1": {"status": "available"},
        },
    }
    return {**value, "sha256": _canonical_sha256(value)}


def validate_scene_capability_atom(atom: dict, source: dict) -> None:
    """Validate one record-local atom; source-bound validation rederives it."""
    if not isinstance(atom, dict):
        raise ValueError("GS scene capability is missing")
    expected = scene_capability_atom(
        source, atom.get("alignment_certificate"))
    if atom != expected:
        raise ValueError("GS scene capability differs from recomputation")


def scene_task_available(atom: dict, source: dict, task_id: str) -> bool:
    validate_scene_capability_atom(atom, source)
    status = atom["task_statuses"].get(str(task_id))
    if not isinstance(status, dict):
        raise ValueError("GS scene capability task is missing")
    return status.get("status") == "available"


class BboxSemanticIndex:
    """Unique per-point instance ownership by source-bound transformed AABBs."""

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, ids: np.ndarray,
                 id_to_cat: Dict[int, str], surface_points: np.ndarray = None,
                 *, alignment_certificate: dict | None = None):
        self._mins = np.asarray(mins, np.float64)
        self._maxs = np.asarray(maxs, np.float64)
        self._ids = np.asarray(ids, np.int64)
        self.id_to_cat = id_to_cat
        self.alignment_certificate = alignment_certificate
        self.semantic_certified = (
            alignment_certificate is not None and
            alignment_certificate.get("accepted") is True)
        self._instance_points = {}
        if surface_points is not None:
            points = np.asarray(surface_points, np.float64)
            assignments = self._assign_spatial(points)
            for instance_id in self._ids:
                selected = points[assignments == int(instance_id)]
                if len(selected) > config.GS_INSTANCE_MAX_POINTS:
                    take = np.linspace(
                        0, len(selected) - 1,
                        config.GS_INSTANCE_MAX_POINTS).astype(int)
                    selected = selected[take]
                self._instance_points[int(instance_id)] = selected.copy()

    def _assign_spatial(self, world_points: np.ndarray) -> np.ndarray:
        return _unique_bbox_assignments(
            world_points, self._mins, self._maxs, self._ids)

    def assign(self, world_points: np.ndarray) -> np.ndarray:
        return self._assign_spatial(world_points)

    def instance_points(self, instance_id: int) -> np.ndarray:
        cached = self._instance_points.get(int(instance_id))
        if cached is not None and len(cached):
            return cached.copy()
        # A bbox corner is annotation metadata, not an observed object-surface
        # point. Targets without Gaussian surface support fail closed.
        return np.empty((0, 3), np.float64)

    def instance_bbox_centroid(self, instance_id: int) -> np.ndarray:
        """Return the centroid of one official transformed InteriorGS bbox."""
        rows = np.flatnonzero(self._ids == int(instance_id))
        if len(rows) != 1:
            raise KeyError(f"GS bbox instance {int(instance_id)} is unknown")
        row = int(rows[0])
        return ((self._mins[row] + self._maxs[row]) * 0.5).copy()

    def visible_depth_view(
            self, world_points: np.ndarray, instance_ids: np.ndarray, *,
            geometry_authority_sha256: str,
            semantic_source_sha256: str):
        """Bind this label index to one frame's actually observed surfaces."""
        return VisibleDepthSemanticIndex(
            self, world_points, instance_ids,
            geometry_authority_sha256=geometry_authority_sha256,
            semantic_source_sha256=semantic_source_sha256)

class VisibleDepthSemanticIndex:
    """Frame-local GS semantics backed only by initial visible RGB-D points."""

    def __init__(
            self, base: BboxSemanticIndex, world_points: np.ndarray,
            instance_ids: np.ndarray, *, geometry_authority_sha256: str,
            semantic_source_sha256: str):
        points = np.asarray(world_points, dtype=np.float64)
        ids = np.asarray(instance_ids, dtype=np.int64)
        if (points.ndim != 2 or points.shape[1] != 3 or
                ids.shape != (len(points),) or
                not np.isfinite(points).all()):
            raise ValueError("GS visible semantic points are invalid")
        for digest in (
                geometry_authority_sha256, semantic_source_sha256):
            if not isinstance(digest, str) or \
                    re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("GS visible semantic source digest is invalid")
        self._base = base
        self._points = np.ascontiguousarray(points)
        self._ids = np.ascontiguousarray(ids)
        self.id_to_cat = base.id_to_cat
        self.alignment_certificate = base.alignment_certificate
        self.semantic_certified = base.semantic_certified
        self.geometry_authority_sha256 = geometry_authority_sha256
        self.semantic_source_sha256 = semantic_source_sha256

    def assign(self, world_points: np.ndarray) -> np.ndarray:
        return self._base.assign(world_points)

    def instance_points(self, instance_id: int) -> np.ndarray:
        return self._points[self._ids == int(instance_id)].copy()

    @staticmethod
    def _points_sha256(points: np.ndarray) -> str:
        ordered = np.asarray(points, dtype=np.float64)
        if len(ordered):
            order = np.lexsort((
                ordered[:, 2], ordered[:, 1], ordered[:, 0]))
            ordered = ordered[order]
        return hashlib.sha256(np.ascontiguousarray(
            ordered, dtype="<f8").tobytes()).hexdigest()

    def confirm_contact_instances(self, requests) -> list[dict]:
        """Confirm A3 only when one initial-visible instance wins uniquely."""
        results = []
        for request in requests:
            if not isinstance(request, (tuple, list)) or len(request) < 2:
                results.append({
                    "confirmed": False,
                    "reason": "visible_contact_request_invalid",
                    "instance_id": None,
                })
                continue
            try:
                instance_id = _source_instance_id(request[0])
                point = np.asarray(request[1], dtype=np.float64)
            except (TypeError, ValueError):
                point = np.empty(0)
            if (point.shape != (3,) or not np.isfinite(point).all() or
                    not self.semantic_certified):
                results.append({
                    "confirmed": False,
                    "reason": "visible_contact_authority_unavailable",
                    "instance_id": None,
                })
                continue
            distances = []
            for candidate_id in sorted(set(self._ids.tolist()) - {0}):
                visible = self.instance_points(candidate_id)
                if len(visible):
                    distances.append((
                        float(np.min(np.linalg.norm(
                            visible - point[None, :], axis=1))),
                        int(candidate_id), visible,
                    ))
            distances.sort(key=lambda value: (value[0], value[1]))
            winner = distances[0] if distances else None
            runner = distances[1] if len(distances) > 1 else None
            if winner is None or winner[1] != instance_id:
                results.append({
                    "confirmed": False,
                    "reason": "visible_contact_winner_mismatch",
                    "instance_id": None,
                })
                continue
            if runner is None:
                results.append({
                    "confirmed": False,
                    "reason": "visible_contact_runner_missing",
                    "instance_id": None,
                })
                continue
            margin = (
                None if runner is None else float(runner[0] - winner[0]))
            if winner[0] > config.A3_CONTACT_FACE_MAX_DISTANCE_M + 1e-9:
                reason = "visible_contact_too_far"
            elif (margin is not None and margin <=
                  config.A3_CONTACT_FACE_TIE_MARGIN_M + 1e-9):
                reason = "visible_contact_near_tie"
            else:
                reason = None
            if reason is not None:
                results.append({
                    "confirmed": False, "reason": reason,
                    "instance_id": None,
                })
                continue
            alignment_sha256 = str(
                self.alignment_certificate["sha256"])
            body = {
                "authority": "gs_initial_visible_depth_instance",
                "schema": VISIBLE_CONTACT_IDENTITY_SCHEMA,
                "instance_id": int(instance_id),
                "category": self.id_to_cat[int(instance_id)],
                "contact_visible_distance_m": float(winner[0]),
                "runner_up_visible_distance_m": (
                    None if runner is None else float(runner[0])),
                "visible_distance_margin_m": margin,
                "visible_points_sha256": self._points_sha256(winner[2]),
                "geometry_authority_sha256":
                    self.geometry_authority_sha256,
                "semantic_source_sha256": self.semantic_source_sha256,
                "alignment_certificate_sha256": alignment_sha256,
                "global_query_protocol": VISIBLE_INSTANCE_PROTOCOL,
                "global_universe_sha256":
                    complete_visible_instance_universe_sha256(
                        geometry_authority_sha256=
                            self.geometry_authority_sha256,
                        semantic_source_sha256=self.semantic_source_sha256,
                        alignment_certificate_sha256=alignment_sha256),
                "global_winner_instance_id": int(winner[1]),
                "global_runner_up_instance_id": int(runner[1]),
            }
            results.append({
                "confirmed": True, "reason": "confirmed", **body,
            })
        return results

def load_bbox_index(labels_path: str) -> BboxSemanticIndex:
    """Load source labels under the official fixed coordinate contract."""
    boxes_value, ids_value, id_to_cat = read_bbox_source(labels_path)
    certificate = build_alignment_certificate(boxes_value, ids_value)
    transformed = _rot_ig_to_habitat(boxes_value)
    mins = transformed.min(axis=1)
    maxs = transformed.max(axis=1)
    return BboxSemanticIndex(
        mins, maxs, ids_value, id_to_cat,
        alignment_certificate=certificate)


def read_bbox_source(labels_path: str):
    """Read canonical InteriorGS boxes without constructing an index."""
    with open(labels_path, encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        raise ValueError("GS labels must contain a list")
    boxes, ids = [], []
    id_to_cat: Dict[int, str] = {}
    seen_ids: set[int] = set()
    for value in labels:
        bbox = value.get("bounding_box")
        if not bbox:
            continue
        instance_id = _source_instance_id(value.get("ins_id"))
        if instance_id in seen_ids:
            raise ValueError("GS label instance ids are invalid")
        seen_ids.add(instance_id)
        boxes.append(np.asarray([
            [point["x"], point["y"], point["z"]] for point in bbox
        ], dtype=np.float64))
        ids.append(instance_id)
        id_to_cat[instance_id] = value["label"]
    return (
        np.asarray(boxes, dtype=np.float64),
        np.asarray(ids, dtype=np.int64),
        id_to_cat,
    )


def derive_scene_capability_from_sources(
        source: dict, labels_path: str) -> dict:
    """Rederive the official coordinate binding from trusted labels."""
    boxes, ids, _categories = read_bbox_source(labels_path)
    certificate = build_alignment_certificate(boxes, ids)
    return scene_capability_atom(source, certificate)


def scene_diagnostics(
        source: dict, gaussian_path: str, labels_path: str) -> dict:
    """Measure source-bound scene capacity without changing eligibility."""
    from pipeline import gs_source

    gaussian = gs_source.read_gaussian_source(gaussian_path)
    boxes, ids, _categories = read_bbox_source(labels_path)
    certificate = build_alignment_certificate(boxes, ids)
    support_radii = float(config.GS_SIGMA) * np.max(
        gaussian["scales"], axis=1)
    if not len(support_radii):
        raise ValueError("GS scene has no Gaussian elements")
    capability = scene_capability_atom(source, certificate)
    return {
        "schema": "egoconseq.gs-scene-diagnostics.v1",
        "scene_id": source["scene_id"],
        "gaussian_count": int(len(support_radii)),
        "support_radius_p50_m": float(np.quantile(support_radii, 0.50)),
        "support_radius_p90_m": float(np.quantile(support_radii, 0.90)),
        "fraction_above_quality_band": float(np.mean(
            support_radii > float(config.GS_SUPPORT_RADIUS_QUALITY_M))),
        "alignment_accepted": bool(certificate["accepted"]),
        "gs_scene_capability": capability,
    }
