"""Offline MP3D semantic geometry and runtime instance assignment."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import tempfile
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path
import re
import stat
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from pipeline import config, mp3d_contact_query
from pipeline.io_utils import sha256_file


SEMANTIC_CACHE_SCHEMA = "semantic-surface.v3"
MP3D_INSTANCE_TRIANGLES_SCHEMA = "mp3d-instance-triangles.v1"
MP3D_COMPLETE_FACE_UNIVERSE_PROTOCOL = \
    "mp3d-complete-face-instance-universe.v1"
MP3D_STREAMING_INSTANCE_FACES_PROTOCOL = \
    "mp3d-source-order-canonical-vertex-faces.v1"
B_TARGET_GEOMETRY_SCHEMA = "b-target-geometry.v1"
B_GROUND_SUPPORT_PROTOCOL = "full-triangle-floor-slab-xz-union.v1"
B_REFERENCE_CENTROID_PROTOCOL = "full-triangle-area-weighted-centroid.v1"
MP3D_TARGET_FACE_INDEX_PROTOCOL = "mp3d-authenticated-face-index.v1"
_MP3D_TARGET_FACE_INDEX_CACHE = OrderedDict()
# One decode seed shared by the decoder default and the cache identity: if
# they ever diverged, a changed decode would silently reuse a stale cache.
SEMANTIC_DECODE_SEED = 0


def _input_identity(role: str, path: os.PathLike) -> dict:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {
        "role": str(role),
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def _cache_identity(
        backend: str,
        inputs: list,
        *,
        sample_count: int,
        match_tol: Optional[float] = None,
        seed: Optional[int] = None) -> tuple:
    parameters = {"sample_count": int(sample_count)}
    if match_tol is not None:
        parameters["match_tol"] = float(match_tol)
    if seed is not None:
        parameters["seed"] = int(seed)
    payload = {
        "backend": str(backend),
        "cache_schema": SEMANTIC_CACHE_SCHEMA,
        "inputs": list(inputs),
        "parameters": parameters,
    }
    metadata = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(metadata.encode("utf-8")).hexdigest()
    return metadata, digest


def _load_cache(path: Path, expected_metadata: str):
    """Load a cache only when its embedded provenance matches exactly."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            if "cache_metadata_json" not in values:
                return None
            actual = str(values["cache_metadata_json"].item())
            if actual != expected_metadata:
                return None
            return (
                np.asarray(values["points"]).copy(),
                np.asarray(values["inst"]).copy(),
            )
    except (OSError, ValueError, KeyError, EOFError,
            zipfile.BadZipFile, zlib.error):
        return None


def _write_cache_atomic(
        path: Path,
        *,
        points: np.ndarray,
        instances: np.ndarray,
        metadata: str) -> None:
    """Write a self-describing NPZ via fsync + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            np.savez_compressed(
                handle,
                points=np.asarray(points),
                inst=np.asarray(instances),
                cache_metadata_json=np.asarray(metadata),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class SemanticIndex:
    """Nearest-labelled-surface instance assigner for a scene."""
    def __init__(self, points_world: np.ndarray, inst: np.ndarray,
                 id_to_cat: Dict[int, str], *, query_workers: int = 1):
        query_workers = int(query_workers)
        if query_workers <= 0:
            raise ValueError("semantic query workers must be positive")
        self.id_to_cat = id_to_cat
        self._points = np.asarray(points_world)
        self._inst = inst
        self._tree = cKDTree(points_world) if points_world.shape[0] else None
        self._query_workers = query_workers
        self._by_instance = None
    def assign(self, world_points: np.ndarray,
               tol: float = config.SEMANTIC_ASSIGN_TOL_M) -> np.ndarray:
        """world_points (N,3) -> instance id per point (0 if none within tol)."""
        if self._tree is None:
            return np.zeros(world_points.shape[0], dtype=np.int64)
        points = np.asarray(world_points)
        workers = (
            self._query_workers
            if len(points) >= config.SEMANTIC_ASSIGN_PARALLEL_MIN_POINTS
            else 1)
        d, idx = self._tree.query(points, k=1, workers=workers)
        return np.where(d <= tol, self._inst[idx], 0).astype(np.int64)
    def instance_points(self, instance_id: int) -> np.ndarray:
        if self._by_instance is None:
            self._by_instance = {
                int(iid): np.flatnonzero(self._inst == iid)
                for iid in np.unique(self._inst) if int(iid) != 0
            }
        idx = self._by_instance.get(int(instance_id))
        return self._points[idx] if idx is not None else np.empty((0, 3), np.float32)


class MP3DSemanticIndex(SemanticIndex):
    """MP3D sampled lookup plus exact, per-instance face geometry authority."""

    def __init__(self, points_world: np.ndarray, inst: np.ndarray,
                 id_to_cat: Dict[int, str], semantic_ply: os.PathLike,
                 semantic_ply_sha256: str | None = None, *,
                 query_workers: int = 1):
        super().__init__(
            points_world, inst, id_to_cat, query_workers=query_workers)
        self._semantic_ply = Path(semantic_ply)
        self._semantic_ply_sha256 = (
            str(semantic_ply_sha256) if semantic_ply_sha256 is not None
            else sha256_file(self._semantic_ply))
        self._instance_triangles = {}
        self._instance_triangle_hashes = {}
        self._streaming_instance_face_hashes = {}

    def instance_triangles(self, instance_id: int) -> np.ndarray:
        """Return canonical full triangles in Habitat world XYZ, shape F×3×3.

        This exact face query is deliberately separate from the sampled
        nearest-surface index. It supplies semantic identity/target geometry;
        it is not a collision oracle.
        """
        instance_id = int(instance_id)
        if instance_id not in self.id_to_cat:
            raise KeyError(f"MP3D semantic instance {instance_id} is unknown")
        if instance_id not in self._instance_triangles:
            triangles = _decode_mp3d_instance_triangles(
                self._semantic_ply, instance_id)
            if not len(triangles):
                raise KeyError(
                    f"MP3D semantic instance {instance_id} has no triangles")
            self._instance_triangles[instance_id] = triangles
        return self._instance_triangles[instance_id].copy()

    def instance_triangles_sha256(self, instance_id: int) -> str:
        """Hash the instance identity and canonical full-triangle geometry."""
        instance_id = int(instance_id)
        if instance_id not in self._instance_triangle_hashes:
            triangles = self.instance_triangles(instance_id)
            metadata = json.dumps({
                "schema": MP3D_INSTANCE_TRIANGLES_SCHEMA,
                "frame": "habitat_world_xyz",
                "instance_id": instance_id,
                "category": self.id_to_cat[instance_id],
                "triangle_count": int(len(triangles)),
                "dtype": "float32-le",
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            digest = hashlib.sha256()
            digest.update(metadata.encode("utf-8"))
            digest.update(b"\n")
            digest.update(np.asarray(
                triangles, dtype="<f4", order="C").tobytes(order="C"))
            self._instance_triangle_hashes[instance_id] = digest.hexdigest()
        return self._instance_triangle_hashes[instance_id]

    def target_geometry_atom(
            self, instance_id: int, floor_plane, *,
            expected_semantic_ply_sha256: str, pose: dict | None = None) -> dict:
        """Materialize B geometry from one source-pinned complete-face read."""
        instance_id = int(instance_id)
        if instance_id not in self.id_to_cat:
            raise KeyError(f"MP3D semantic instance {instance_id} is unknown")
        expected = str(expected_semantic_ply_sha256)
        if expected != self._semantic_ply_sha256:
            raise ValueError("semantic PLY digest does not match source authority")
        triangles = _pinned_instance_triangles(
            self._semantic_ply, instance_id,
            expected_semantic_ply_sha256=expected)
        if not len(triangles):
            raise KeyError(
                f"MP3D semantic instance {instance_id} has no triangles")
        triangle_sha = _instance_triangles_sha256(
            triangles, instance_id=instance_id,
            category=self.id_to_cat[instance_id])
        support = clip_target_ground_support(
            triangles, floor_plane, pose=pose)
        centroid = area_weighted_surface_centroid(triangles)
        support_value = {
            "protocol": B_GROUND_SUPPORT_PROTOCOL,
            "frame": "habitat_world_xz",
            "ground_band_m": [
                float(config.GROUND_OBSTACLE_BAND_M[0]),
                float(config.GROUND_OBSTACLE_BAND_M[1]),
            ],
            **support,
        }
        support_value["sha256"] = _json_sha256(support_value)
        centroid_value = {
            "protocol": B_REFERENCE_CENTROID_PROTOCOL,
            "frame": "habitat_world_xyz",
            "world_xyz_m": centroid.tolist(),
            "world_xz_m": centroid[[0, 2]].tolist(),
        }
        centroid_value["sha256"] = _json_sha256(centroid_value)
        value = {
            "schema": B_TARGET_GEOMETRY_SCHEMA,
            "instance_id": instance_id,
            "category": self.id_to_cat[instance_id],
            "semantic_ply_sha256": expected,
            "full_triangle_protocol": MP3D_INSTANCE_TRIANGLES_SCHEMA,
            "full_triangle_count": int(len(triangles)),
            "full_triangles_sha256": triangle_sha,
            "ground_support": support_value,
            "reference_centroid": centroid_value,
        }
        return {**value, "sha256": _json_sha256(value)}

    def confirm_contact_instance(
            self, instance_id: int, world_point,
            *, candidate_instance_ids=None) -> dict:
        """Confirm a sampled contact witness against exact MP3D faces.

        ``world_point`` is the navmesh contact surface in Habitat world XYZ.
        Winner and runner-up are resolved over every face-instance ID in the
        semantic PLY. ``candidate_instance_ids`` is retained only as caller
        audit context; it never restricts this authority query. Visibility is
        enforced separately by the record/candidate projection.
        """
        return self.confirm_contact_instances([(instance_id, world_point)])[0]

    def confirm_contact_instances(self, requests) -> list[dict]:
        """Resolve all contact points in one bounded complete-PLY stream."""
        parsed = []
        results = [None] * len(requests)
        for index, (instance_id, world_point) in enumerate(requests):
            instance_id = int(instance_id)
            point = np.asarray(world_point, dtype=np.float64)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                results[index] = {
                    "confirmed": False,
                    "reason": "contact_face_point_invalid",
                    "instance_id": None,
                }
            else:
                parsed.append((index, instance_id, point))
        if not parsed:
            return results
        batch = _query_mp3d_complete_face_universe_batch(
            self._semantic_ply,
            np.stack([value[2] for value in parsed]),
            witness_instance_ids=[value[1] for value in parsed],
            instance_categories=self.id_to_cat,
            expected_semantic_ply_sha256=self._semantic_ply_sha256,
            query_workers=self._query_workers)
        self._streaming_instance_face_hashes.update(
            batch["streaming_instance_faces_sha256"])
        universe = {
            "global_query_protocol": MP3D_COMPLETE_FACE_UNIVERSE_PROTOCOL,
            "semantic_ply_sha256": self._semantic_ply_sha256,
        }
        universe["global_universe_sha256"] = \
            complete_face_universe_sha256(**universe)
        for (output_index, instance_id, _point), query in zip(
                parsed, batch["queries"]):
            results[output_index] = self._contact_identity_from_query(
                instance_id, query, universe)
        return results

    def _contact_identity_from_query(
            self, instance_id: int, query: dict, universe: dict) -> dict:
        winner_id = query["global_winner_instance_id"]
        runner_id = query["global_runner_up_instance_id"]
        witness_distance = query["witness_face_distance_m"]
        winner_distance = query["global_winner_face_distance_m"]
        runner_up = query["global_runner_up_face_distance_m"]
        margin = (
            float(runner_up - winner_distance)
            if runner_up is not None else None)
        common = {
            "authority": "mp3d_full_face_universe",
            "schema": "mp3d-contact-face-identity.v1",
            "contact_face_distance_m": witness_distance,
            "runner_up_face_distance_m": runner_up,
            "face_distance_margin_m": margin,
            "global_winner_instance_id": winner_id,
            "global_runner_up_instance_id": runner_id,
            **universe,
        }
        if runner_id is None or runner_up is None or margin is None:
            return {
                "confirmed": False,
                "reason": "global_runner_evidence_missing",
                "instance_id": None,
                **common,
            }
        if (margin is not None and
                margin <= config.A3_CONTACT_FACE_TIE_MARGIN_M + 1e-9):
            return {
                "confirmed": False,
                "reason": "contact_face_identity_near_tie",
                "instance_id": None,
                **common,
            }
        if winner_id != instance_id:
            return {
                "confirmed": False,
                "reason": "global_contact_winner_mismatch",
                "instance_id": None,
                **common,
            }
        if witness_distance > config.A3_CONTACT_FACE_MAX_DISTANCE_M + 1e-9:
            return {
                "confirmed": False,
                "reason": "contact_face_too_far",
                "instance_id": None,
                **common,
            }
        return {
            "confirmed": True,
            "reason": "confirmed",
            "instance_id": instance_id,
            "category": self.id_to_cat[instance_id],
            "streaming_instance_faces_sha256":
                self._streaming_instance_face_hashes[instance_id],
            **common,
        }


def _json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _instance_triangles_sha256(
        triangles: np.ndarray, *, instance_id: int, category: str) -> str:
    canonical = _canonical_triangles(triangles)
    metadata = json.dumps({
        "schema": MP3D_INSTANCE_TRIANGLES_SCHEMA,
        "frame": "habitat_world_xyz",
        "instance_id": int(instance_id),
        "category": str(category),
        "triangle_count": int(len(canonical)),
        "dtype": "float32-le",
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256()
    digest.update(metadata.encode("utf-8"))
    digest.update(b"\n")
    digest.update(np.asarray(
        canonical, dtype="<f4", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _world_triangles_to_floor_local(
        triangles: np.ndarray, pose: dict | None) -> np.ndarray:
    values = np.asarray(triangles, dtype=np.float64)
    if pose is None:
        return values
    position = np.asarray(pose.get("position"), dtype=np.float64)
    try:
        yaw = float(pose.get("yaw_rad"))
    except (TypeError, ValueError):
        raise ValueError("target geometry pose is invalid") from None
    if (position.shape != (3,) or not np.all(np.isfinite(position)) or
            not np.isfinite(yaw)):
        raise ValueError("target geometry pose is invalid")
    shifted = values - position
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local = np.empty_like(shifted)
    local[:, :, 0] = shifted[:, :, 0] * cosine - shifted[:, :, 2] * sine
    local[:, :, 1] = shifted[:, :, 1]
    local[:, :, 2] = -shifted[:, :, 0] * sine - shifted[:, :, 2] * cosine
    return local


def _clip_polygon_by_height(
        world_polygon: list[np.ndarray], heights: list[float], *,
        boundary: float, keep_above: bool) -> tuple[list, list]:
    output_points = []
    output_heights = []
    for index, current in enumerate(world_polygon):
        previous = world_polygon[index - 1]
        current_height = float(heights[index])
        previous_height = float(heights[index - 1])
        current_inside = (
            current_height >= boundary if keep_above
            else current_height <= boundary)
        previous_inside = (
            previous_height >= boundary if keep_above
            else previous_height <= boundary)
        if current_inside != previous_inside:
            denominator = current_height - previous_height
            if denominator == 0.0:
                continue
            fraction = (boundary - previous_height) / denominator
            output_points.append(previous + fraction * (current - previous))
            output_heights.append(float(boundary))
        if current_inside:
            output_points.append(current)
            output_heights.append(current_height)
    return output_points, output_heights


def _canonical_rows(values: list[np.ndarray], width: int) -> list:
    if not values:
        return []
    unique = np.unique(np.stack(values).reshape(len(values), width), axis=0)
    shaped = unique.reshape(len(unique), width // 2, 2)
    return (shaped[:, 0, :].tolist() if width == 2 else shaped.tolist())


def _canonical_support_components(polygons: list[list[np.ndarray]]) -> dict:
    triangles = []
    segments = []
    points = []
    for raw_polygon in polygons:
        polygon = []
        for raw_point in raw_polygon:
            point = np.asarray(raw_point, dtype=np.float64)
            if not polygon or not np.array_equal(point, polygon[-1]):
                polygon.append(point)
        if len(polygon) > 1 and np.array_equal(polygon[0], polygon[-1]):
            polygon.pop()
        unique = np.unique(np.stack(polygon), axis=0) if polygon else \
            np.empty((0, 2), dtype=np.float64)
        if len(unique) == 1:
            points.append(unique)
            continue
        if len(unique) == 2:
            segments.append(unique)
            continue
        coordinates = np.stack(polygon)
        signed_twice_area = math.fsum(
            float(coordinates[index, 0] * coordinates[(index + 1) % len(
                coordinates), 1] - coordinates[(index + 1) % len(
                coordinates), 0] * coordinates[index, 1])
            for index in range(len(coordinates)))
        if signed_twice_area == 0.0:
            distances = np.sum(
                (unique[:, None, :] - unique[None, :, :]) ** 2, axis=2)
            left, right = np.unravel_index(np.argmax(distances), distances.shape)
            if distances[left, right] == 0.0:
                points.append(unique[:1])
            else:
                segments.append(np.stack([unique[left], unique[right]]))
            continue
        for index in range(1, len(coordinates) - 1):
            triangle = np.stack([
                coordinates[0], coordinates[index], coordinates[index + 1]])
            area = float(np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]))
            if area == 0.0:
                continue
            order = np.lexsort((triangle[:, 1], triangle[:, 0]))
            triangles.append(triangle[order])
    if not triangles and not segments and not points:
        raise ValueError("target ground support is empty")
    return {
        "triangles_xz_m": _canonical_rows(triangles, 6),
        "segments_xz_m": _canonical_rows(segments, 4),
        "points_xz_m": _canonical_rows(points, 2),
    }


def clip_target_ground_support(
        triangles_world: np.ndarray, floor_plane, *, pose: dict | None = None,
        band_m=config.GROUND_OBSTACLE_BAND_M) -> dict:
    """Clip complete target faces to the floor slab and project world XZ."""
    triangles = np.asarray(triangles_world, dtype=np.float64)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or
            not len(triangles) or not np.all(np.isfinite(triangles))):
        raise ValueError("target triangles must be finite Fx3x3")
    low, high = (float(value) for value in band_m)
    if not (np.isfinite(low) and np.isfinite(high) and 0 <= low <= high):
        raise ValueError("target ground band is invalid")
    local = _world_triangles_to_floor_local(triangles, pose)
    heights = np.asarray(
        floor_plane.height_above_points(local.reshape(-1, 3)),
        dtype=np.float64).reshape(-1, 3)
    projected_polygons = []
    for world_triangle, triangle_heights in zip(triangles, heights):
        polygon, polygon_heights = _clip_polygon_by_height(
            [point for point in world_triangle], triangle_heights.tolist(),
            boundary=low, keep_above=True)
        if not polygon:
            continue
        polygon, polygon_heights = _clip_polygon_by_height(
            polygon, polygon_heights, boundary=high, keep_above=False)
        if not polygon:
            continue
        projected_polygons.append([point[[0, 2]] for point in polygon])
    support = _canonical_support_components(projected_polygons)
    if not support["triangles_xz_m"]:
        raise ValueError(
            "target ground support has no nondegenerate 2-D triangle")
    return support


def point_to_ground_support_nearest(
        point_xz, support: dict) -> tuple[float, list[float]]:
    """Return exact distance and a deterministic witness on ground support.

    Coordinates are Habitat world ``(x, z)`` in metres.  Canonical component
    order breaks equidistant ties, so a rebuilt certificate selects the same
    witness without depending on set or dictionary iteration order.
    """
    point = np.asarray(point_xz, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("support query point must be finite XZ")
    if not isinstance(support, dict):
        raise ValueError("ground support must be an object")
    try:
        triangles = np.asarray(support["triangles_xz_m"], dtype=np.float64)
        segments = np.asarray(support["segments_xz_m"], dtype=np.float64)
        points = np.asarray(support["points_xz_m"], dtype=np.float64)
        triangles = triangles.reshape(-1, 3, 2)
        segments = segments.reshape(-1, 2, 2)
        points = points.reshape(-1, 2)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ground support components are invalid") from error
    if not all(np.all(np.isfinite(value))
               for value in (triangles, segments, points)):
        raise ValueError("ground support components must be finite")
    best: tuple[float, int, int, int, np.ndarray] | None = None

    def consider(distance: float, kind: int, component_index: int,
                 part_index: int, witness: np.ndarray) -> None:
        nonlocal best
        candidate = (
            float(distance), int(kind), int(component_index), int(part_index),
            np.asarray(witness, dtype=np.float64),
        )
        if best is None or candidate[:4] < best[:4]:
            best = candidate

    for triangle_index, triangle in enumerate(triangles):
        edges = [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ]
        area = float(np.cross(edges[0], triangle[2] - triangle[0]))
        if area == 0.0:
            continue
        signs = [
            float(np.cross(edge, point - triangle[index]))
            for index, edge in enumerate(edges)
        ]
        if (all(value >= 0.0 for value in signs) or
                all(value <= 0.0 for value in signs)):
            return 0.0, [float(value) for value in point]
        for index, edge in enumerate(edges):
            length_sq = float(np.dot(edge, edge))
            fraction = min(1.0, max(0.0, float(
                np.dot(point - triangle[index], edge) / length_sq)))
            nearest = triangle[index] + fraction * edge
            consider(
                float(np.linalg.norm(point - nearest)), 0,
                triangle_index, index, nearest)
    for segment_index, segment in enumerate(segments):
        edge = segment[1] - segment[0]
        length_sq = float(np.dot(edge, edge))
        if length_sq == 0.0:
            consider(
                float(np.linalg.norm(point - segment[0])), 1,
                segment_index, 0, segment[0])
            continue
        fraction = min(1.0, max(0.0, float(
            np.dot(point - segment[0], edge) / length_sq)))
        nearest = segment[0] + fraction * edge
        consider(
            float(np.linalg.norm(point - nearest)), 1,
            segment_index, 0, nearest)
    for point_index, support_point in enumerate(points):
        consider(
            float(np.linalg.norm(point - support_point)), 2,
            point_index, 0, support_point)
    if best is None or not math.isfinite(best[0]):
        raise ValueError("ground support is empty")
    return float(best[0]), [float(value) for value in best[4]]


def point_to_ground_support_distance_m(point_xz, support: dict) -> float:
    """Distance to the dataset-typed target ground-support union."""
    return point_to_ground_support_nearest(point_xz, support)[0]


def area_weighted_surface_centroid(triangles: np.ndarray) -> np.ndarray:
    """Return the full-surface centroid using 3D triangle area measure."""
    values = np.asarray(triangles, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (3, 3) or
            not len(values) or not np.all(np.isfinite(values))):
        raise ValueError("surface triangles must be finite Fx3x3")
    first_edges = values[:, 1] - values[:, 0]
    second_edges = values[:, 2] - values[:, 0]
    scales = np.maximum(
        np.max(np.abs(first_edges), axis=1),
        np.max(np.abs(second_edges), axis=1))
    normalized_area_twice = np.zeros(len(values), dtype=np.float64)
    nonzero_scale = scales > 0.0
    normalized_area_twice[nonzero_scale] = np.linalg.norm(np.cross(
        first_edges[nonzero_scale] / scales[nonzero_scale, None],
        second_edges[nonzero_scale] / scales[nonzero_scale, None]), axis=1)
    keep = normalized_area_twice > 0.0
    if not np.any(keep):
        raise ValueError("surface has no nondegenerate triangles")
    log_weights = (
        np.log(normalized_area_twice[keep]) + 2.0 * np.log(scales[keep]))
    weights = np.exp(log_weights - np.max(log_weights))
    centers = values[keep].mean(axis=1)
    total = math.fsum(float(value) for value in weights)
    return np.array([
        math.fsum(float(weight * center[axis])
                  for weight, center in zip(weights, centers)) / total
        for axis in range(3)
    ], dtype=np.float64)


def complete_face_universe_sha256(
        *, global_query_protocol: str, semantic_ply_sha256: str) -> str:
    """Bind one complete face-universe description to its PLY bytes."""
    value = {
        "schema": "mp3d-complete-face-universe-proof.v1",
        "global_query_protocol": str(global_query_protocol),
        "semantic_ply_sha256": str(semantic_ply_sha256),
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _reset_mp3d_query_diagnostics() -> None:
    mp3d_contact_query.reset_diagnostics()


def _mp3d_query_diagnostics() -> dict:
    return mp3d_contact_query.diagnostics()


def _mp3d_query_profile_rows() -> list[dict]:
    """Return execution-only A3 pruning rows; these never enter a record."""
    return mp3d_contact_query.profile_rows()


def _branch_and_bound_instance_top_two(
        point: np.ndarray, instance_ids: np.ndarray,
        aabb_minima: np.ndarray, aabb_maxima: np.ndarray,
        exact_distance, *,
        slack_m: float = config.A3_FACE_AABB_PRUNE_SLACK_M
        ) -> tuple[list[tuple[float, int]], list[int]]:
    return mp3d_contact_query.branch_and_bound_instance_top_two(
        point, instance_ids, aabb_minima, aabb_maxima, exact_distance,
        slack_m=slack_m)


def _query_mp3d_complete_face_universe_batch(
        semantic_ply: os.PathLike, points: np.ndarray,
        *, witness_instance_ids,
        instance_categories: dict[int, str],
        expected_semantic_ply_sha256: str,
        query_workers: int = 1) -> dict:
    """Resolve several contact points in one complete-universe PLY stream."""
    import trimesh

    query_points = np.asarray(points, dtype=np.float64)
    if (query_points.ndim != 2 or query_points.shape[1] != 3 or
            not len(query_points) or not np.all(np.isfinite(query_points))):
        raise ValueError("MP3D contact query points must be finite Kx3")
    witness_ids = [int(value) for value in witness_instance_ids]
    if len(witness_ids) != len(query_points):
        raise ValueError("MP3D witness IDs must align with query points")
    if any(instance_id not in instance_categories
           for instance_id in witness_ids):
        raise ValueError(
            "MP3D requested instance has no house category")
    descriptor = os.open(
        semantic_ply, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("MP3D semantic PLY must be a regular file")
        file_size = source_stat.st_size
        if file_size <= 0:
            raise ValueError("MP3D semantic PLY is empty")
        pinned = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(descriptor)
    try:
        completed = _capture_mp3d_pinned_query(
            pinned, file_size, semantic_ply, query_points, witness_ids,
            instance_categories, expected_semantic_ply_sha256,
            int(query_workers))
    finally:
        pinned.close()
    if completed[0] == "error":
        raise completed[1](*completed[2])
    return completed[1]


def _capture_mp3d_pinned_query(*args):
    """Drop exception tracebacks (and their mmap views) before buffer close."""
    try:
        return "ok", _query_mp3d_complete_face_universe_buffer(*args)
    except BaseException as error:
        return "error", type(error), tuple(error.args)


def _query_mp3d_complete_face_universe_buffer(
        pinned, file_size: int, semantic_ply: os.PathLike,
        query_points: np.ndarray, witness_ids: list[int],
        instance_categories: dict[int, str],
        expected_semantic_ply_sha256: str,
        query_workers: int = 1) -> dict:
    """Run the bounded query, falling back to the frozen exhaustive body."""
    try:
        return _query_mp3d_complete_face_universe_buffer_bounded(
            pinned, file_size, semantic_ply, query_points, witness_ids,
            instance_categories, expected_semantic_ply_sha256,
            query_workers=query_workers)
    except (KeyError, TypeError, ValueError):
        mp3d_contact_query.record_fallback()
        return _query_mp3d_complete_face_universe_buffer_exhaustive(
            pinned, file_size, semantic_ply, query_points, witness_ids,
            instance_categories, expected_semantic_ply_sha256,
            query_workers=query_workers)


def _mp3d_contact_query_backend() -> dict:
    return {
        "binary_ply_layout_from_buffer":
            _binary_ply_layout_from_buffer,
        "vertex_dtype": _MP3D_VERTEX_DTYPE,
        "triangle_dtype": _MP3D_TRIANGLE_DTYPE,
        "source_triangles": _mp3d_source_triangles,
        "authenticated_face_index": _authenticated_face_index,
        "ensure_instance_aabbs":
            _ensure_authenticated_instance_aabbs,
        "streaming_protocol":
            MP3D_STREAMING_INSTANCE_FACES_PROTOCOL,
    }


def _query_mp3d_complete_face_universe_buffer_exhaustive(
        pinned, file_size: int, semantic_ply: os.PathLike,
        query_points: np.ndarray, witness_ids: list[int],
        instance_categories: dict[int, str],
        expected_semantic_ply_sha256: str,
        query_workers: int = 1) -> dict:
    return mp3d_contact_query.query_exhaustive(
        pinned, file_size, semantic_ply, query_points, witness_ids,
        instance_categories, expected_semantic_ply_sha256,
        backend=_mp3d_contact_query_backend(),
        query_workers=query_workers)


def _query_mp3d_complete_face_universe_buffer_bounded(
        pinned, file_size: int, semantic_ply: os.PathLike,
        query_points: np.ndarray, witness_ids: list[int],
        instance_categories: dict[int, str],
        expected_semantic_ply_sha256: str,
        query_workers: int = 1) -> dict:
    return mp3d_contact_query.query_bounded(
        pinned, file_size, semantic_ply, query_points, witness_ids,
        instance_categories, expected_semantic_ply_sha256,
        backend=_mp3d_contact_query_backend(),
        query_workers=query_workers)


# --------------------------------------------------------------------------
# MP3D semantic PLY + house descriptor
# --------------------------------------------------------------------------

_MP3D_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
_MP3D_TRIANGLE_DTYPE = np.dtype([
    ("count", "u1"), ("vertex_indices", "<i4", (3,)),
    ("object_id", "<i4"),
])


def _mp3d_paths(scene_glb: os.PathLike) -> tuple:
    scene = Path(scene_glb)
    stem = scene.stem
    return (
        scene.with_name(stem + "_semantic.ply"),
        scene.with_name(stem + ".house"),
    )


def _read_immutable_regular_file(
        path: os.PathLike, *, expected_sha256: str | None = None) -> bytes:
    """Copy one regular file once; hash and consumers share those bytes."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"source authority must be a regular file: {path}")
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            immutable = bytes(chunk)
            chunks.append(immutable)
            digest.update(immutable)
        if expected_sha256 is not None and digest.hexdigest() != \
                str(expected_sha256):
            raise ValueError(f"source authority digest mismatch: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_mp3d_house_bytes(payload: bytes, path: os.PathLike) -> Dict[int, str]:
    categories = {}
    objects = {}
    for line in payload.decode("utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "C" and len(parts) >= 6:
            categories[int(parts[1])] = str(parts[5])
        elif parts[0] == "O" and len(parts) >= 4:
            objects[int(parts[1])] = int(parts[3])
    # Instance zero is reserved for "unlabelled" throughout this pipeline.
    return {
        int(object_id) + 1: categories[category_id]
        for object_id, category_id in objects.items()
        if category_id in categories
    }


def _parse_mp3d_house(
        path: os.PathLike, *, expected_sha256: str | None = None
        ) -> Dict[int, str]:
    payload = _read_immutable_regular_file(
        path, expected_sha256=expected_sha256)
    return _parse_mp3d_house_bytes(payload, path)


def _binary_ply_layout(path: os.PathLike) -> tuple:
    header = bytearray()
    with open(path, "rb") as handle:
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            header.extend(line)
    return _parse_binary_ply_header(bytes(header), path)


def _binary_ply_layout_from_buffer(buffer, path: os.PathLike) -> tuple:
    marker = b"end_header\n"
    end = buffer.find(marker)
    if end < 0:
        raise ValueError(f"unterminated PLY header: {path}")
    return _parse_binary_ply_header(buffer[:end + len(marker)], path)


def _parse_binary_ply_header(header: bytes, path: os.PathLike) -> tuple:
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("MP3D semantic PLY must be binary little-endian")
    vertex_match = re.search(r"element vertex (\d+)", text)
    face_match = re.search(r"element face (\d+)", text)
    if not vertex_match or not face_match:
        raise ValueError("MP3D semantic PLY is missing vertex/face counts")
    expected_vertex = (
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n")
    expected_face = (
        "property list uchar int vertex_indices\nproperty int object_id\n")
    if expected_vertex not in text or expected_face not in text:
        raise ValueError("unsupported MP3D semantic PLY property layout")
    return len(header), int(vertex_match.group(1)), int(face_match.group(1))


def _mp3d_face_sample_indices(
        object_ids: np.ndarray, sample_count: int) -> np.ndarray:
    """Global coverage plus a small deterministic quota for every instance."""
    face_count = len(object_ids)
    if not face_count:
        return np.empty(0, dtype=np.int64)
    global_count = min(face_count, max(1, int(sample_count)))
    selected = set(np.linspace(
        0, face_count - 1, global_count, dtype=np.int64).tolist())
    first_by_object = {}
    chunk_size = 1_000_000
    for start in range(0, face_count, chunk_size):
        chunk = np.asarray(
            object_ids[start:start + chunk_size], dtype=np.int64)
        unique, first = np.unique(chunk, return_index=True)
        for object_id, local_index in zip(unique.tolist(), first.tolist()):
            first_by_object.setdefault(int(object_id), start + int(local_index))
    for object_id, start in first_by_object.items():
        for index in range(start, min(start + 16, face_count)):
            if int(object_ids[index]) != object_id:
                break
            selected.add(index)
    return np.asarray(sorted(selected), dtype=np.int64)


def _validated_mp3d_face_indices(
        raw_indices: np.ndarray, vertex_count: int) -> np.ndarray:
    """Validate triangle vertex references before NumPy advanced indexing."""
    indices = np.asarray(raw_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("MP3D semantic faces must contain three vertex indices")
    if np.any(indices < 0) or np.any(indices >= int(vertex_count)):
        raise ValueError("MP3D semantic face vertex index is out of range")
    return indices


def _mp3d_source_triangles(
        vertices: np.ndarray, raw_indices: np.ndarray,
        vertex_count: int) -> np.ndarray:
    indices = _validated_mp3d_face_indices(raw_indices, vertex_count)
    triangles = np.stack([
        vertices["x"][indices],
        vertices["y"][indices],
        vertices["z"][indices],
    ], axis=-1)
    if not np.all(np.isfinite(triangles)):
        raise ValueError(
            "MP3D semantic faces must reference finite vertex coordinates")
    return triangles


def _decode_mp3d_labelled_points(
        semantic_ply: os.PathLike, sample_count: int) -> tuple:
    """Decode face instance ids without materializing the multi-GB mesh."""
    header_size, vertex_count, face_count = _binary_ply_layout(semantic_ply)
    vertices = np.memmap(
        semantic_ply, mode="r", dtype=_MP3D_VERTEX_DTYPE,
        offset=header_size, shape=(vertex_count,))
    face_offset = header_size + vertex_count * _MP3D_VERTEX_DTYPE.itemsize
    faces = np.memmap(
        semantic_ply, mode="r", dtype=_MP3D_TRIANGLE_DTYPE,
        offset=face_offset, shape=(face_count,))
    if face_count and not np.all(faces["count"] == 3):
        raise ValueError("MP3D semantic PLY contains non-triangle faces")
    selected = _mp3d_face_sample_indices(
        faces["object_id"], sample_count)
    triangles = _mp3d_source_triangles(
        vertices, faces["vertex_indices"][selected], vertex_count)
    centroids = np.asarray(triangles.mean(axis=1), dtype=np.float32)
    # MP3D descriptor is Z-up/front=+Y. Habitat's scene frame is Y-up/-Z front.
    points_world = np.stack([
        centroids[:, 0], centroids[:, 2], -centroids[:, 1],
    ], axis=1).astype(np.float32)
    instance_ids = (
        np.asarray(faces["object_id"][selected], dtype=np.int32) + 1)
    return points_world, instance_ids


def _canonical_triangles(triangles: np.ndarray) -> np.ndarray:
    """Canonicalize vertex and face order without changing float32 geometry."""
    values = np.asarray(triangles, dtype="<f4")
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("semantic triangles must have shape (F, 3, 3)")
    ordered = np.empty_like(values)
    for index, triangle in enumerate(values):
        vertex_order = np.lexsort((
            triangle[:, 2], triangle[:, 1], triangle[:, 0]))
        ordered[index] = triangle[vertex_order]
    flattened = ordered.reshape(len(ordered), 9)
    face_order = np.lexsort(tuple(
        flattened[:, column]
        for column in range(flattened.shape[1] - 1, -1, -1)))
    return np.ascontiguousarray(ordered[face_order], dtype="<f4")


def _decode_mp3d_instance_triangles(
        semantic_ply: os.PathLike, instance_id: int) -> np.ndarray:
    """Read every face for one MP3D instance through the shared PLY layout."""
    header_size, vertex_count, face_count = _binary_ply_layout(semantic_ply)
    vertices = np.memmap(
        semantic_ply, mode="r", dtype=_MP3D_VERTEX_DTYPE,
        offset=header_size, shape=(vertex_count,))
    face_offset = header_size + vertex_count * _MP3D_VERTEX_DTYPE.itemsize
    faces = np.memmap(
        semantic_ply, mode="r", dtype=_MP3D_TRIANGLE_DTYPE,
        offset=face_offset, shape=(face_count,))
    raw_object_id = int(instance_id) - 1
    chunks = []
    chunk_size = 1_000_000
    for start in range(0, face_count, chunk_size):
        chunk = faces[start:start + chunk_size]
        if len(chunk) and not np.all(chunk["count"] == 3):
            raise ValueError("MP3D semantic PLY contains non-triangle faces")
        selected = np.flatnonzero(chunk["object_id"] == raw_object_id)
        if not len(selected):
            continue
        source = _mp3d_source_triangles(
            vertices, chunk["vertex_indices"][selected], vertex_count)
        # MP3D descriptor is Z-up/front=+Y. Habitat is Y-up/-Z front.
        chunks.append(np.stack([
            source[:, :, 0], source[:, :, 2], -source[:, :, 1],
        ], axis=-1))
    if not chunks:
        return np.empty((0, 3, 3), dtype=np.float32)
    return _canonical_triangles(np.concatenate(chunks, axis=0))


def _read_exact(descriptor: int, byte_count: int) -> bytes:
    chunks = []
    remaining = int(byte_count)
    while remaining:
        chunk = os.read(descriptor, min(
            remaining, config.MP3D_TARGET_FACE_INDEX_STREAM_BYTES))
        if not chunk:
            raise ValueError("MP3D semantic PLY ended before parsed endpoint")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_binary_ply_header(descriptor: int, path: os.PathLike) -> bytes:
    header = bytearray()
    marker = b"end_header\n"
    while not header.endswith(marker):
        value = os.read(descriptor, 1)
        if not value:
            raise ValueError(f"unterminated PLY header: {path}")
        header.extend(value)
        if len(header) > 1024 * 1024:
            raise ValueError("MP3D semantic PLY header exceeds 1 MiB")
    return bytes(header)


def _minimum_unsigned_dtype(maximum: int) -> np.dtype:
    if maximum <= np.iinfo(np.uint8).max:
        return np.dtype("u1")
    if maximum <= np.iinfo(np.uint16).max:
        return np.dtype("<u2")
    return np.dtype("<u4")


def _minimum_signed_dtype(minimum: int, maximum: int) -> np.dtype:
    for dtype in (np.dtype("i1"), np.dtype("<i2"), np.dtype("<i4")):
        limits = np.iinfo(dtype)
        if limits.min <= minimum and maximum <= limits.max:
            return dtype
    raise ValueError("MP3D semantic object ID is outside int32 range")


def _cache_authenticated_face_index(key: tuple, entry: dict) -> dict:
    maximum_bytes = int(config.MP3D_TARGET_FACE_INDEX_CACHE_MAX_BYTES)
    maximum_scenes = int(config.MP3D_TARGET_FACE_INDEX_CACHE_MAX_SCENES)
    if maximum_bytes <= 0 or maximum_scenes <= 0:
        raise MemoryError("MP3D face-index cache budget is disabled")
    if entry["cache_bytes"] > maximum_bytes:
        raise MemoryError("MP3D face-index cache byte budget exceeded")
    current_bytes = sum(
        value["cache_bytes"]
        for value in _MP3D_TARGET_FACE_INDEX_CACHE.values())
    while _MP3D_TARGET_FACE_INDEX_CACHE and (
            len(_MP3D_TARGET_FACE_INDEX_CACHE) >= maximum_scenes or
            current_bytes + entry["cache_bytes"] > maximum_bytes):
        _old_key, old = _MP3D_TARGET_FACE_INDEX_CACHE.popitem(last=False)
        current_bytes -= old["cache_bytes"]
    _MP3D_TARGET_FACE_INDEX_CACHE[key] = entry
    return entry


def _build_authenticated_face_index(
        semantic_ply: os.PathLike, expected_sha256: str) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(semantic_ply, flags)
    try:
        initial_stat = os.fstat(descriptor)
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError("MP3D semantic PLY must be a regular file")
        header = _read_binary_ply_header(descriptor, semantic_ply)
        header_size, vertex_count, face_count = _parse_binary_ply_header(
            header, semantic_ply)
        vertex_bytes = vertex_count * _MP3D_VERTEX_DTYPE.itemsize
        face_bytes = face_count * _MP3D_TRIANGLE_DTYPE.itemsize
        if header_size + vertex_bytes + face_bytes != initial_stat.st_size:
            raise ValueError("MP3D semantic PLY parsed endpoint is not file size")
        index_dtype = _minimum_unsigned_dtype(max(0, vertex_count - 1))
        worst_cache_bytes = face_count * (4 + 3 * index_dtype.itemsize)
        if worst_cache_bytes > \
                config.MP3D_TARGET_FACE_INDEX_CACHE_MAX_BYTES:
            raise MemoryError("MP3D face-index cache byte budget exceeded")
        object_ids = np.empty(face_count, dtype="<i4")
        vertex_indices = np.empty((face_count, 3), dtype=index_dtype)
        digest = hashlib.sha256(header)
        remaining_vertices = vertex_bytes
        while remaining_vertices:
            chunk = _read_exact(descriptor, min(
                remaining_vertices,
                config.MP3D_TARGET_FACE_INDEX_STREAM_BYTES))
            digest.update(chunk)
            remaining_vertices -= len(chunk)
        faces_per_chunk = max(
            1, config.MP3D_TARGET_FACE_INDEX_STREAM_BYTES //
            _MP3D_TRIANGLE_DTYPE.itemsize)
        for start in range(0, face_count, faces_per_chunk):
            count = min(faces_per_chunk, face_count - start)
            payload = _read_exact(
                descriptor, count * _MP3D_TRIANGLE_DTYPE.itemsize)
            digest.update(payload)
            faces = np.frombuffer(
                payload, dtype=_MP3D_TRIANGLE_DTYPE, count=count)
            if not np.all(faces["count"] == 3):
                raise ValueError(
                    "MP3D semantic PLY contains non-triangle faces")
            raw_ids = np.asarray(faces["object_id"], dtype=np.int64)
            if np.any(raw_ids < -1):
                raise ValueError(
                    "MP3D semantic PLY contains invalid object IDs")
            indices = _validated_mp3d_face_indices(
                faces["vertex_indices"], vertex_count)
            endpoint = start + count
            object_ids[start:endpoint] = raw_ids
            vertex_indices[start:endpoint] = indices
        if digest.hexdigest() != expected_sha256:
            raise ValueError("semantic PLY digest does not match source authority")
    finally:
        os.close(descriptor)
    if len(object_ids):
        object_dtype = _minimum_signed_dtype(
            int(np.min(object_ids)), int(np.max(object_ids)))
        object_ids = object_ids.astype(object_dtype, copy=False)
    else:
        object_dtype = np.dtype("i1")
        object_ids = object_ids.astype(object_dtype, copy=False)
    entry = {
        "protocol": (
            f"{MP3D_TARGET_FACE_INDEX_PROTOCOL}:"
            f"objects={object_dtype.str}:indices={index_dtype.str}"),
        "header_size": int(header_size),
        "vertex_count": int(vertex_count),
        "face_count": int(face_count),
        "face_offset": int(header_size + vertex_bytes),
        "object_ids": object_ids,
        "vertex_indices": vertex_indices,
        "cache_bytes": int(object_ids.nbytes + vertex_indices.nbytes),
    }
    return entry


def _authenticated_face_index(
        semantic_ply: os.PathLike, expected_sha256: str) -> dict:
    key = (str(Path(semantic_ply).absolute()), str(expected_sha256))
    cached = _MP3D_TARGET_FACE_INDEX_CACHE.get(key)
    if cached is not None:
        _MP3D_TARGET_FACE_INDEX_CACHE.move_to_end(key)
        return cached
    return _cache_authenticated_face_index(
        key, _build_authenticated_face_index(semantic_ply, expected_sha256))


def _extend_authenticated_face_index_cache(
        entry: dict, arrays: dict[str, np.ndarray]) -> None:
    """Attach small derived arrays while preserving the global byte budget."""
    additional = int(sum(value.nbytes for value in arrays.values()))
    maximum = int(config.MP3D_TARGET_FACE_INDEX_CACHE_MAX_BYTES)
    if entry["cache_bytes"] + additional > maximum:
        raise MemoryError("MP3D face-index cache byte budget exceeded")
    current = int(sum(
        value["cache_bytes"]
        for value in _MP3D_TARGET_FACE_INDEX_CACHE.values()))
    for key, cached in list(_MP3D_TARGET_FACE_INDEX_CACHE.items()):
        if current + additional <= maximum:
            break
        if cached is entry:
            continue
        current -= int(cached["cache_bytes"])
        del _MP3D_TARGET_FACE_INDEX_CACHE[key]
    if current + additional > maximum:
        raise MemoryError("MP3D face-index cache byte budget exceeded")
    entry.update(arrays)
    entry["cache_bytes"] = int(entry["cache_bytes"] + additional)


def _ensure_authenticated_instance_aabbs(
        pinned, index: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact world-space instance AABBs by chunked scatter-reduce."""
    keys = (
        "instance_aabb_instance_ids",
        "instance_aabb_min_world_xyz",
        "instance_aabb_max_world_xyz",
    )
    if all(key in index for key in keys):
        return tuple(index[key] for key in keys)
    object_ids = index["object_ids"]
    vertex_indices = index["vertex_indices"]
    raw_instance_ids = set()
    item_bytes = int(
        object_ids.dtype.itemsize + 3 * vertex_indices.dtype.itemsize)
    chunk_faces = max(
        1, int(config.MP3D_TARGET_FACE_INDEX_STREAM_BYTES) // item_bytes)
    for start in range(0, len(object_ids), chunk_faces):
        values = np.asarray(
            object_ids[start:start + chunk_faces], dtype=np.int64)
        values = values[values >= 0]
        if len(values):
            raw_instance_ids.update(int(value) for value in np.unique(values))
    if not raw_instance_ids:
        raise ValueError("MP3D semantic PLY contains no face-instance universe")
    raw_ids = np.asarray(sorted(raw_instance_ids), dtype=np.int64)
    minima = np.full((len(raw_ids), 3), np.inf, dtype=np.float64)
    maxima = np.full((len(raw_ids), 3), -np.inf, dtype=np.float64)
    vertices = np.frombuffer(
        pinned, dtype=_MP3D_VERTEX_DTYPE,
        count=index["vertex_count"], offset=index["header_size"])
    try:
        for chunk_index, start in enumerate(
                range(0, len(object_ids), chunk_faces)):
            endpoint = min(start + chunk_faces, len(object_ids))
            chunk_ids = np.asarray(
                object_ids[start:endpoint], dtype=np.int64)
            keep = chunk_ids >= 0
            if not np.any(keep):
                continue
            selected_ids = chunk_ids[keep]
            compact_ids = np.searchsorted(raw_ids, selected_ids)
            source = _mp3d_source_triangles(
                vertices, vertex_indices[start:endpoint][keep],
                index["vertex_count"])
            face_minima = np.stack([
                np.min(source[:, :, 0], axis=1),
                np.min(source[:, :, 2], axis=1),
                -np.max(source[:, :, 1], axis=1),
            ], axis=1).astype(np.float64, copy=False)
            face_maxima = np.stack([
                np.max(source[:, :, 0], axis=1),
                np.max(source[:, :, 2], axis=1),
                -np.min(source[:, :, 1], axis=1),
            ], axis=1).astype(np.float64, copy=False)
            np.minimum.at(minima, compact_ids, face_minima)
            np.maximum.at(maxima, compact_ids, face_maxima)
            advice = getattr(mmap, "MADV_DONTNEED", None)
            interval = int(config.MP3D_AABB_MADVISE_INTERVAL_CHUNKS)
            if (advice is not None and interval > 0 and
                    (chunk_index + 1) % interval == 0):
                # NumPy owns no vertex copy here. Discarding clean read-only
                # mmap pages caps RSS without changing the authenticated bytes
                # or any reduction result; later references fault them back.
                pinned.madvise(advice, 0, index["face_offset"])
    finally:
        del vertices
    if (not np.all(np.isfinite(minima)) or
            not np.all(np.isfinite(maxima))):
        raise ValueError("MP3D instance AABB construction was incomplete")
    arrays = {
        keys[0]: np.asarray(raw_ids + 1, dtype=np.int64),
        keys[1]: minima,
        keys[2]: maxima,
    }
    _extend_authenticated_face_index_cache(index, arrays)
    return tuple(index[key] for key in keys)


def _pinned_instance_triangles(
        semantic_ply: os.PathLike, instance_id: int, *,
        expected_semantic_ply_sha256: str) -> np.ndarray:
    index = _authenticated_face_index(
        semantic_ply, str(expected_semantic_ply_sha256))
    raw_object_id = int(instance_id) - 1
    face_rows = np.flatnonzero(index["object_ids"] == raw_object_id)
    expected_indices = np.asarray(
        index["vertex_indices"][face_rows], dtype=np.int64)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(semantic_ply, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("MP3D semantic PLY must be a regular file")
        pinned = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        all_faces = all_vertices = None
        try:
            all_faces = np.frombuffer(
                pinned, dtype=_MP3D_TRIANGLE_DTYPE,
                count=index["face_count"], offset=index["face_offset"])
            faces = all_faces[face_rows].copy()
            del all_faces
            if (not np.all(faces["count"] == 3) or
                    not np.all(faces["object_id"] == raw_object_id)):
                raise ValueError("authenticated MP3D face membership changed")
            actual_indices = _validated_mp3d_face_indices(
                faces["vertex_indices"], index["vertex_count"])
            if not np.array_equal(actual_indices, expected_indices):
                raise ValueError("authenticated MP3D face indices changed")
            if not len(expected_indices):
                vertices = np.empty(0, dtype=_MP3D_VERTEX_DTYPE)
                inverse = np.empty(0, dtype=np.int64)
            else:
                unique_vertices, inverse = np.unique(
                    expected_indices.reshape(-1), return_inverse=True)
                all_vertices = np.frombuffer(
                    pinned, dtype=_MP3D_VERTEX_DTYPE,
                    count=index["vertex_count"], offset=index["header_size"])
                vertices = all_vertices[unique_vertices].copy()
                del all_vertices
                all_vertices = None
        finally:
            all_faces = None
            all_vertices = None
            pinned.close()
    finally:
        os.close(descriptor)
    if not len(expected_indices):
        return np.empty((0, 3, 3), dtype=np.float32)
    selected = vertices[inverse].reshape(len(expected_indices), 3)
    source = np.stack([
        selected["x"], selected["y"], selected["z"],
    ], axis=-1)
    if not np.all(np.isfinite(source)):
        raise ValueError(
            "MP3D semantic faces must reference finite vertex coordinates")
    world = np.stack([
        source[:, :, 0], source[:, :, 2], -source[:, :, 1],
    ], axis=-1).astype(np.float32)
    return _canonical_triangles(world)


def load_mp3d_semantic_index(
        scene_glb: os.PathLike,
        cache_dir: os.PathLike = config.SEMANTIC_CACHE_DIR,
        sample_count: int = config.SEMANTIC_SAMPLE_COUNT,
        rebuild: bool = False, *,
        query_workers: int = 1,
        expected_semantic_ply_sha256: str | None = None,
        expected_house_sha256: str | None = None) -> SemanticIndex:
    """Load or build an MP3D face-instance surface index."""
    semantic_ply, house = _mp3d_paths(scene_glb)
    id_to_cat = _parse_mp3d_house(
        house, expected_sha256=expected_house_sha256)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    semantic_identity = _input_identity("semantic_ply", semantic_ply)
    if (expected_semantic_ply_sha256 is not None and
            semantic_identity["sha256"] != expected_semantic_ply_sha256):
        raise ValueError("semantic PLY digest does not match source authority")
    metadata, digest = _cache_identity(
        "mp3d_ply",
        [
            semantic_identity,
            _input_identity("house_descriptor", house),
        ],
        sample_count=sample_count,
    )
    cache = (
        cache_dir /
        f"mp3d-{Path(scene_glb).stem}.{digest[:16]}.sem.npz"
    )
    if not rebuild:
        cached = _load_cache(cache, metadata)
        if cached is not None:
            return MP3DSemanticIndex(
                cached[0], cached[1], id_to_cat, semantic_ply,
                semantic_identity["sha256"],
                query_workers=query_workers)
    points, instances = _decode_mp3d_labelled_points(
        semantic_ply, sample_count)
    labelled = np.isin(instances, np.asarray(
        sorted(id_to_cat), dtype=np.int32))
    points, instances = points[labelled], instances[labelled]
    _write_cache_atomic(
        cache, points=points, instances=instances, metadata=metadata)
    return MP3DSemanticIndex(
        points, instances, id_to_cat, semantic_ply,
        semantic_identity["sha256"], query_workers=query_workers)


def load_mp3d_target_authority(
        scene_glb: os.PathLike, *, expected_semantic_ply_sha256: str,
        expected_house_sha256: str) -> MP3DSemanticIndex:
    """Load immutable MP3D category and geometry validation authorities."""
    semantic_ply, house = _mp3d_paths(scene_glb)
    id_to_cat = _parse_mp3d_house(
        house, expected_sha256=expected_house_sha256)
    return MP3DSemanticIndex(
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.int32), id_to_cat, semantic_ply,
        str(expected_semantic_ply_sha256))
