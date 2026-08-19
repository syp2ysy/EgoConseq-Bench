"""Pure runtime-triangle semantic authority for BEHAVIOR-1K.

The OmniGibson session extracts runtime prim metadata and decrypted collision
triangles, converts them to the PBench frame, and then constructs this module's
value objects.  No simulator package is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import re
from typing import Iterable, Mapping

import numpy as np

from pipeline import b1k_geometry, config, record, semantic


SCENE_AUTHORITY_SCHEMA = "b1k-derived-scene-authority.v1"
B1K_CANONICAL_REPLAY_PROTOCOL = \
    "b1k-canonical-fixed-snapshot-replay.v2"
B1K_CONTACT_IDENTITY_SCHEMA = "b1k-contact-triangle-identity.v1"
B1K_COMPLETE_TRIANGLE_UNIVERSE_PROTOCOL = \
    "b1k-runtime-instance-triangle-universe.v1"
B1K_RENDERER_INSTANCE_PROTOCOL = record.B1K_RENDERER_INSTANCE_PROTOCOL
B1K_TRIANGLE_FRAME = "pbench_world_xyz"
_CANONICAL_REPLAY_AUTHORITY_FIELDS = frozenset({
    "canonical_replay_protocol", "canonical_state_sha256",
})
_DERIVED_SCENE_BINDING_TOKEN = object()
_ASSIGN_POINT_CHUNK_SIZE = 256
_ASSIGN_TRIANGLE_CHUNK_SIZE = 512


def _canonical_json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_string(value, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"B1K runtime instance {field} must be nonempty")
    return normalized


def canonical_replay_binding(authority: Mapping) -> dict | None:
    """Validate the exact optional replay-field pair on an authority atom."""
    present = _CANONICAL_REPLAY_AUTHORITY_FIELDS.intersection(authority)
    if not present:
        return None
    if present != _CANONICAL_REPLAY_AUTHORITY_FIELDS:
        raise ValueError(
            "B1K canonical replay fields must be both absent or present")
    protocol = authority["canonical_replay_protocol"]
    state_digest = authority["canonical_state_sha256"]
    if protocol is None or state_digest is None:
        raise ValueError("B1K canonical replay fields must be non-null")
    if protocol != B1K_CANONICAL_REPLAY_PROTOCOL:
        raise ValueError("B1K canonical replay protocol is unsupported")
    if (not isinstance(state_digest, str) or
            re.fullmatch(r"[0-9a-f]{64}", state_digest) is None):
        raise ValueError("B1K canonical state digest is invalid")
    return {
        "canonical_replay_protocol": protocol,
        "canonical_state_sha256": state_digest,
    }


def _positive_instance_id(value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("B1K instance ID must be a positive integral non-bool")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("B1K instance ID must be positive")
    return normalized


@dataclass(frozen=True)
class RuntimeInstanceSpec:
    """Private runtime identity and exact PBench triangle geometry."""

    prim_identity: str
    raw_category: str
    model: str
    synset_label: str
    triangles: np.ndarray

    def __post_init__(self) -> None:
        for field in (
                "prim_identity", "raw_category", "model", "synset_label"):
            object.__setattr__(
                self, field, _nonempty_string(getattr(self, field), field))
        object.__setattr__(
            self, "triangles", b1k_geometry.canonical_triangles(
                self.triangles))


def _normalized_instances(instances: Iterable[RuntimeInstanceSpec]
                          ) -> tuple[RuntimeInstanceSpec, ...]:
    parsed = []
    for value in instances:
        if isinstance(value, RuntimeInstanceSpec):
            instance = value
        elif isinstance(value, Mapping):
            instance = RuntimeInstanceSpec(
                prim_identity=value.get(
                    "prim_identity", value.get("prim_path")),
                raw_category=value.get("raw_category", value.get("category")),
                model=value.get("model"),
                synset_label=value.get(
                    "synset_label", value.get("synset")),
                triangles=value.get("triangles"),
            )
        else:
            instance = RuntimeInstanceSpec(*value)
        parsed.append(instance)
    parsed.sort(key=lambda value: value.prim_identity)
    identities = [value.prim_identity for value in parsed]
    if len(set(identities)) != len(identities):
        raise ValueError("B1K runtime prim identities must be unique")
    return tuple(parsed)


def _triangle_bytes_sha256(triangles: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(
        triangles, dtype="<f4", order="C").tobytes(order="C")).hexdigest()


def _component_authority_value(component: b1k_geometry.TriangleComponent
                               ) -> dict:
    return {
        "identity_sha256": hashlib.sha256(
            component.identity.encode("utf-8")).hexdigest(),
        "triangle_count": int(len(component.triangles)),
        "triangles_sha256": _triangle_bytes_sha256(component.triangles),
    }


def _instance_authority_value(instance_id: int,
                              instance: RuntimeInstanceSpec) -> dict:
    private_identity = {
        "prim_identity": instance.prim_identity,
        "raw_category": instance.raw_category,
        "model": instance.model,
    }
    return {
        "instance_id": int(instance_id),
        "private_runtime_identity_sha256": _canonical_json_sha256(
            private_identity),
        "synset_label": instance.synset_label,
        "triangle_count": int(len(instance.triangles)),
        "triangles_sha256": _triangle_bytes_sha256(instance.triangles),
    }


def derive_scene_authority_atom(
        *, floor_components, collision_components, instances,
        canonical_replay_protocol: str | None = None,
        canonical_state_sha256: str | None = None) -> dict:
    """Bind all runtime collision and semantic triangles to one digest."""
    if (canonical_replay_protocol is None) != \
            (canonical_state_sha256 is None):
        raise ValueError(
            "B1K canonical replay authority fields must be provided together")
    if canonical_replay_protocol is not None:
        if canonical_replay_protocol != B1K_CANONICAL_REPLAY_PROTOCOL:
            raise ValueError("B1K canonical replay protocol is unsupported")
        if re.fullmatch(r"[0-9a-f]{64}", canonical_state_sha256 or "") \
                is None:
            raise ValueError("B1K canonical state digest must be lowercase sha256")
    floors = b1k_geometry._normalized_components(floor_components)
    collisions = b1k_geometry._normalized_components(collision_components)
    runtime_instances = _normalized_instances(instances)
    value = {
        "schema": SCENE_AUTHORITY_SCHEMA,
        "frame": B1K_TRIANGLE_FRAME,
        "ground_obstacle_band_m": [
            float(config.GROUND_OBSTACLE_BAND_M[0]),
            float(config.GROUND_OBSTACLE_BAND_M[1]),
        ],
        "floor_components": [
            _component_authority_value(value) for value in floors],
        "collision_components": [
            _component_authority_value(value) for value in collisions],
        "runtime_instances": [
            _instance_authority_value(index, value)
            for index, value in enumerate(runtime_instances, start=1)],
    }
    if canonical_replay_protocol is not None:
        value.update({
            "canonical_replay_protocol": canonical_replay_protocol,
            "canonical_state_sha256": canonical_state_sha256,
        })
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def complete_triangle_universe_sha256(*, global_query_protocol: str,
                                      scene_authority_sha256: str) -> str:
    value = {
        "schema": "b1k-complete-triangle-universe-proof.v1",
        "global_query_protocol": str(global_query_protocol),
        "scene_authority_sha256": str(scene_authority_sha256),
    }
    return _canonical_json_sha256(value)


def _instance_triangles_sha256(triangles: np.ndarray, *, instance_id: int,
                               category: str) -> str:
    metadata = json.dumps({
        "schema": record.B1K_INSTANCE_TRIANGLES_SCHEMA,
        "frame": B1K_TRIANGLE_FRAME,
        "instance_id": int(instance_id),
        "category": str(category),
        "triangle_count": int(len(triangles)),
        "dtype": "float32-le",
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256()
    digest.update(metadata.encode("ascii"))
    digest.update(b"\n")
    digest.update(np.asarray(
        triangles, dtype="<f4", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _segment_distances(point: np.ndarray, first: np.ndarray,
                       second: np.ndarray) -> np.ndarray:
    edges = second - first
    lengths = np.einsum("ij,ij->i", edges, edges)
    fractions = np.divide(
        np.einsum("ij,ij->i", point - first, edges), lengths,
        out=np.zeros_like(lengths), where=lengths > 0.0)
    fractions = np.clip(fractions, 0.0, 1.0)
    nearest = first + fractions[:, None] * edges
    return np.linalg.norm(nearest - point, axis=1)


def _point_triangle_distance_m(point, triangles: np.ndarray) -> float:
    """Exact Euclidean point-to-triangle distance in bounded NumPy chunks."""
    query = np.asarray(point, dtype=np.float64)
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("B1K triangle query point must be finite XYZ")
    if not len(triangles):
        return float("inf")
    best = float("inf")
    for start in range(0, len(triangles), 65_536):
        values = triangles[start:start + 65_536]
        first, second, third = values[:, 0], values[:, 1], values[:, 2]
        edge_0 = second - first
        edge_1 = third - first
        normals = np.cross(edge_0, edge_1)
        norm_sq = np.einsum("ij,ij->i", normals, normals)
        signed = np.einsum("ij,ij->i", query - first, normals)
        signed_scale = np.divide(
            signed, norm_sq, out=np.zeros_like(signed), where=norm_sq > 0.0)
        projection = query - signed_scale[:, None] * normals
        relative = projection - first
        dot_00 = np.einsum("ij,ij->i", edge_0, edge_0)
        dot_01 = np.einsum("ij,ij->i", edge_0, edge_1)
        dot_11 = np.einsum("ij,ij->i", edge_1, edge_1)
        dot_20 = np.einsum("ij,ij->i", relative, edge_0)
        dot_21 = np.einsum("ij,ij->i", relative, edge_1)
        denominator = dot_00 * dot_11 - dot_01 * dot_01
        bary_0 = np.divide(
            dot_11 * dot_20 - dot_01 * dot_21, denominator,
            out=np.zeros_like(denominator), where=denominator > 0.0)
        bary_1 = np.divide(
            dot_00 * dot_21 - dot_01 * dot_20, denominator,
            out=np.zeros_like(denominator), where=denominator > 0.0)
        inside = (
            (denominator > 0.0) &
            (bary_0 >= 0.0) & (bary_1 >= 0.0) &
            (bary_0 + bary_1 <= 1.0))
        plane = np.divide(
            np.abs(signed), np.sqrt(norm_sq),
            out=np.full_like(signed, np.inf), where=norm_sq > 0.0)
        edges = np.minimum.reduce([
            _segment_distances(query, first, second),
            _segment_distances(query, second, third),
            _segment_distances(query, third, first),
        ])
        distances = np.where(inside, plane, edges)
        best = min(best, float(np.min(distances)))
    return best


def _point_segment_distance_matrix(
        points: np.ndarray, first: np.ndarray,
        second: np.ndarray) -> np.ndarray:
    """Return exact distances for every point/segment pair, shape PxT."""
    edges = second - first
    lengths = np.einsum("tj,tj->t", edges, edges)
    relative = points[:, None, :] - first[None, :, :]
    numerator = np.einsum("ptj,tj->pt", relative, edges)
    fractions = np.divide(
        numerator, lengths[None, :], out=np.zeros_like(numerator),
        where=lengths[None, :] > 0.0)
    fractions = np.clip(fractions, 0.0, 1.0)
    residual = relative - fractions[..., None] * edges[None, :, :]
    return np.sqrt(np.einsum("ptj,ptj->pt", residual, residual))


def _paired_point_segment_distances_m(
        points: np.ndarray, first: np.ndarray,
        second: np.ndarray) -> np.ndarray:
    """Return exact distances for aligned point/segment pairs, shape P."""
    edges = second - first
    lengths = np.einsum("ij,ij->i", edges, edges)
    relative = points - first
    numerator = np.einsum("ij,ij->i", relative, edges)
    fractions = np.divide(
        numerator, lengths, out=np.zeros_like(numerator),
        where=lengths > 0.0)
    fractions = np.clip(fractions, 0.0, 1.0)
    residual = relative - fractions[:, None] * edges
    return np.sqrt(np.einsum("ij,ij->i", residual, residual))


def _paired_point_triangle_distances_m(
        points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return exact distances for aligned point/triangle pairs, shape P."""
    first = triangles[:, 0]
    second = triangles[:, 1]
    third = triangles[:, 2]
    edge_0 = second - first
    edge_1 = third - first
    normals = np.cross(edge_0, edge_1)
    norm_sq = np.einsum("ij,ij->i", normals, normals)
    relative = points - first
    signed = np.einsum("ij,ij->i", relative, normals)
    signed_scale = np.divide(
        signed, norm_sq, out=np.zeros_like(signed), where=norm_sq > 0.0)
    projection_relative = relative - signed_scale[:, None] * normals
    dot_00 = np.einsum("ij,ij->i", edge_0, edge_0)
    dot_01 = np.einsum("ij,ij->i", edge_0, edge_1)
    dot_11 = np.einsum("ij,ij->i", edge_1, edge_1)
    dot_20 = np.einsum("ij,ij->i", projection_relative, edge_0)
    dot_21 = np.einsum("ij,ij->i", projection_relative, edge_1)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    bary_0 = np.divide(
        dot_11 * dot_20 - dot_01 * dot_21, denominator,
        out=np.zeros_like(denominator), where=denominator > 0.0)
    bary_1 = np.divide(
        dot_00 * dot_21 - dot_01 * dot_20, denominator,
        out=np.zeros_like(denominator), where=denominator > 0.0)
    inside = (
        (denominator > 0.0) &
        (bary_0 >= 0.0) & (bary_1 >= 0.0) &
        (bary_0 + bary_1 <= 1.0))
    plane = np.divide(
        np.abs(signed), np.sqrt(norm_sq),
        out=np.full_like(signed, np.inf), where=norm_sq > 0.0)
    edges = np.minimum.reduce([
        _paired_point_segment_distances_m(points, first, second),
        _paired_point_segment_distances_m(points, second, third),
        _paired_point_segment_distances_m(points, third, first),
    ])
    return np.where(inside, plane, edges)


def _point_triangle_distances_m(
        points, triangles: np.ndarray, *,
        point_chunk_size: int = _ASSIGN_POINT_CHUNK_SIZE,
        triangle_chunk_size: int = _ASSIGN_TRIANGLE_CHUNK_SIZE,
        max_distance_m: float | None = None,
) -> np.ndarray:
    """Exact minimum triangle distance per point in bounded PxT chunks.

    When ``max_distance_m`` is provided, pairs whose expanded triangle AABB
    cannot contain the point are returned as infinity.  Every distance at or
    below the threshold remains exact; semantic assignment only consumes that
    certified tolerance neighborhood.
    """
    queries = np.asarray(points, dtype=np.float64)
    values = np.asarray(triangles, dtype=np.float64)
    if (queries.ndim != 2 or queries.shape[1:] != (3,) or
            not np.isfinite(queries).all()):
        raise ValueError("B1K triangle query points must be finite Nx3")
    if (values.ndim != 3 or values.shape[1:] != (3, 3) or
            not np.isfinite(values).all()):
        raise ValueError("B1K triangle query geometry must be finite Tx3x3")
    point_chunk = int(point_chunk_size)
    triangle_chunk = int(triangle_chunk_size)
    if point_chunk <= 0 or triangle_chunk <= 0:
        raise ValueError("B1K triangle query chunks must be positive")
    max_distance = None
    if max_distance_m is not None:
        max_distance = float(max_distance_m)
        if not math.isfinite(max_distance) or max_distance < 0.0:
            raise ValueError(
                "B1K triangle maximum distance must be finite and nonnegative")
    result = np.full(len(queries), np.inf, dtype=np.float64)
    if not len(values):
        return result
    for point_start in range(0, len(queries), point_chunk):
        point_values = queries[point_start:point_start + point_chunk]
        best = np.full(len(point_values), np.inf, dtype=np.float64)
        for triangle_start in range(0, len(values), triangle_chunk):
            triangle_values = values[
                triangle_start:triangle_start + triangle_chunk]
            if max_distance is not None:
                minimum = np.min(triangle_values, axis=1)
                maximum = np.max(triangle_values, axis=1)
                candidate_pairs = np.all(
                    (point_values[:, None, :] >=
                     minimum[None, :, :] - max_distance) &
                    (point_values[:, None, :] <=
                     maximum[None, :, :] + max_distance),
                    axis=2,
                )
                point_rows, triangle_rows = np.nonzero(candidate_pairs)
                if not len(point_rows):
                    continue
                pair_distances = _paired_point_triangle_distances_m(
                    point_values[point_rows], triangle_values[triangle_rows])
                np.minimum.at(best, point_rows, pair_distances)
                continue
            first = triangle_values[:, 0]
            second = triangle_values[:, 1]
            third = triangle_values[:, 2]
            edge_0 = second - first
            edge_1 = third - first
            normals = np.cross(edge_0, edge_1)
            norm_sq = np.einsum("tj,tj->t", normals, normals)
            relative = point_values[:, None, :] - first[None, :, :]
            signed = np.einsum("ptj,tj->pt", relative, normals)
            signed_scale = np.divide(
                signed, norm_sq[None, :], out=np.zeros_like(signed),
                where=norm_sq[None, :] > 0.0)
            projection_relative = relative - signed_scale[..., None] * \
                normals[None, :, :]
            dot_00 = np.einsum("tj,tj->t", edge_0, edge_0)
            dot_01 = np.einsum("tj,tj->t", edge_0, edge_1)
            dot_11 = np.einsum("tj,tj->t", edge_1, edge_1)
            dot_20 = np.einsum(
                "ptj,tj->pt", projection_relative, edge_0)
            dot_21 = np.einsum(
                "ptj,tj->pt", projection_relative, edge_1)
            denominator = dot_00 * dot_11 - dot_01 * dot_01
            numerator_0 = (
                dot_11[None, :] * dot_20 -
                dot_01[None, :] * dot_21
            )
            numerator_1 = (
                dot_00[None, :] * dot_21 -
                dot_01[None, :] * dot_20
            )
            bary_0 = np.divide(
                numerator_0, denominator[None, :],
                out=np.zeros_like(numerator_0),
                where=denominator[None, :] > 0.0)
            bary_1 = np.divide(
                numerator_1, denominator[None, :],
                out=np.zeros_like(numerator_1),
                where=denominator[None, :] > 0.0)
            inside = (
                (denominator[None, :] > 0.0) &
                (bary_0 >= 0.0) & (bary_1 >= 0.0) &
                (bary_0 + bary_1 <= 1.0))
            plane = np.divide(
                np.abs(signed), np.sqrt(norm_sq)[None, :],
                out=np.full_like(signed, np.inf),
                where=norm_sq[None, :] > 0.0)
            edges = np.minimum.reduce([
                _point_segment_distance_matrix(
                    point_values, first, second),
                _point_segment_distance_matrix(
                    point_values, second, third),
                _point_segment_distance_matrix(
                    point_values, third, first),
            ])
            distances = np.where(inside, plane, edges)
            best = np.minimum(best, np.min(distances, axis=1))
        result[point_start:point_start + len(point_values)] = best
    return result


class B1KSemanticAuthority:
    """Stable synset labels and exact runtime-instance triangle queries."""

    def __init__(self, instances, *, scene_authority_sha256: str,
                 _derived_binding_token=None):
        if _derived_binding_token is not _DERIVED_SCENE_BINDING_TOKEN:
            raise ValueError(
                "bound B1K semantics must be built by build_b1k_authorities")
        digest = str(scene_authority_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("B1K scene authority digest must be lowercase sha256")
        values = _normalized_instances(instances)
        self._scene_authority_sha256 = digest
        self._triangles_by_id = {
            index: value.triangles.copy()
            for index, value in enumerate(values, start=1)
        }
        self._aabb_by_id = {
            instance_id: (
                np.min(triangles.reshape(-1, 3), axis=0),
                np.max(triangles.reshape(-1, 3), axis=0),
            )
            for instance_id, triangles in self._triangles_by_id.items()
            if len(triangles)
        }
        self._runtime_metadata_by_id = {
            index: {
                "prim_identity": value.prim_identity,
                "raw_category": value.raw_category,
                "model": value.model,
            }
            for index, value in enumerate(values, start=1)
        }
        self._prim_to_id = {
            value.prim_identity: index
            for index, value in enumerate(values, start=1)
        }
        self.id_to_cat = {
            index: value.synset_label
            for index, value in enumerate(values, start=1)
        }
        self.id_to_predicate_cat = {
            index: value.raw_category
            for index, value in enumerate(values, start=1)
        }
        import trimesh
        self._surface_query_by_id = {
            instance_id: trimesh.proximity.ProximityQuery(trimesh.Trimesh(
                vertices=triangles.reshape(-1, 3),
                faces=np.arange(3 * len(triangles), dtype=np.int64).reshape(
                    -1, 3),
                process=False,
                validate=False,
            ))
            for instance_id, triangles in self._triangles_by_id.items()
            if len(triangles)
        }
        self._triangle_hashes = {}

    @property
    def scene_authority_sha256(self) -> str:
        return self._scene_authority_sha256

    def public_instances(self) -> list[dict]:
        return [
            {"instance_id": instance_id, "category": self.id_to_cat[instance_id]}
            for instance_id in sorted(self.id_to_cat)
        ]

    def category_layers(self, instance_id: int) -> dict[str, str]:
        """Return machine, raw, and display labels without changing identity."""
        instance_id = _positive_instance_id(instance_id)
        if instance_id not in self.id_to_cat:
            raise KeyError(f"B1K semantic instance {instance_id} is unknown")
        raw = self.id_to_predicate_cat[instance_id]
        return {
            "machine": self.id_to_cat[instance_id],
            "raw": raw,
            "display": " ".join(raw.replace("_", " ").split()),
        }

    def instance_id_for_prim(self, prim_identity: str) -> int:
        try:
            return self._prim_to_id[str(prim_identity)]
        except KeyError:
            raise KeyError(
                f"B1K runtime prim {prim_identity!r} is unknown") from None

    def instance_triangles(self, instance_id: int) -> np.ndarray:
        instance_id = _positive_instance_id(instance_id)
        if instance_id not in self._triangles_by_id:
            raise KeyError(f"B1K semantic instance {instance_id} is unknown")
        triangles = self._triangles_by_id[instance_id]
        if not len(triangles):
            raise KeyError(
                f"B1K semantic instance {instance_id} has no triangles")
        return triangles.copy()

    def instance_points(self, instance_id: int) -> np.ndarray:
        """Return canonical exact surface samples, shape ``(N, 3)`` metres.

        Samples are the lexicographically sorted unique union of runtime
        triangle vertices and centroids.  They remain bound to the same exact
        triangles as B1/B2 and never fall back to proxy bbox corners.
        """
        triangles = self.instance_triangles(instance_id)
        points = np.concatenate([
            triangles.reshape(-1, 3),
            triangles.mean(axis=1),
        ], axis=0)
        points = np.unique(points, axis=0)
        if not len(points):
            raise KeyError(
                f"B1K semantic instance {int(instance_id)} has no surface "
                "points")
        return points.copy()

    def instance_triangles_sha256(self, instance_id: int) -> str:
        instance_id = _positive_instance_id(instance_id)
        if instance_id not in self._triangle_hashes:
            self._triangle_hashes[instance_id] = _instance_triangles_sha256(
                self.instance_triangles(instance_id),
                instance_id=instance_id,
                category=self.id_to_cat[instance_id])
        return self._triangle_hashes[instance_id]

    def assign(self, world_points: np.ndarray,
               tol: float = config.SEMANTIC_ASSIGN_TOL_M) -> np.ndarray:
        points = np.asarray(world_points, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1:] != (3,) or
                not np.isfinite(points).all()):
            raise ValueError("B1K semantic query points must be finite Nx3")
        tolerance = float(tol)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("B1K semantic tolerance must be finite and nonnegative")
        assigned = np.zeros(len(points), dtype=np.int64)
        best_distances = np.full(len(points), np.inf, dtype=np.float64)
        aabb_candidate_count = 0
        indexed_surface_query_count = 0
        for instance_id in sorted(self._aabb_by_id):
            minimum, maximum = self._aabb_by_id[instance_id]
            candidate_rows = np.flatnonzero(np.all(
                (points >= minimum - tolerance) &
                (points <= maximum + tolerance), axis=1))
            candidate_count = len(candidate_rows)
            if not candidate_count:
                continue
            _closest, distances, _triangle_ids = self._surface_query_by_id[
                instance_id].on_surface(points[candidate_rows])
            distances = np.asarray(distances, dtype=np.float64)
            if (distances.shape != (candidate_count,) or
                    not np.isfinite(distances).all()):
                raise RuntimeError(
                    "B1K indexed surface query returned invalid distances")
            aabb_candidate_count += candidate_count
            indexed_surface_query_count += candidate_count
            wins = distances < best_distances[candidate_rows]
            winning_rows = candidate_rows[wins]
            best_distances[winning_rows] = distances[wins]
            assigned[winning_rows] = instance_id
        assigned[best_distances > tolerance] = 0
        self.last_assign_diagnostics = {
            "query_point_count": len(points),
            "aabb_candidate_count": aabb_candidate_count,
            "indexed_surface_query_count": indexed_surface_query_count,
        }
        return assigned

    def confirm_contact_instance(self, instance_id: int, world_point, *,
                                 candidate_instance_ids=None) -> dict:
        del candidate_instance_ids
        return self.confirm_contact_instances([
            (instance_id, world_point)])[0]

    def confirm_contact_instances(self, requests) -> list[dict]:
        results = []
        for raw_instance_id, raw_point in requests:
            instance_id = _positive_instance_id(raw_instance_id)
            if instance_id not in self.id_to_cat:
                raise KeyError(
                    f"B1K semantic instance {instance_id} is unknown")
            point = np.asarray(raw_point, dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                results.append({
                    "confirmed": False,
                    "reason": "contact_triangle_point_invalid",
                    "instance_id": None,
                })
                continue
            candidates = sorted(
                (_point_triangle_distance_m(point, triangles), candidate_id)
                for candidate_id, triangles in self._triangles_by_id.items()
                if len(triangles))
            if not candidates:
                raise ValueError("B1K triangle universe is empty")
            winner_distance, winner_id = candidates[0]
            runner_distance, runner_id = (
                candidates[1] if len(candidates) > 1 else (None, None))
            witness_distance = _point_triangle_distance_m(
                point, self.instance_triangles(instance_id))
            margin = (
                float(runner_distance - winner_distance)
                if runner_distance is not None else None)
            common = {
                "authority": "b1k_runtime_triangle_universe",
                "schema": B1K_CONTACT_IDENTITY_SCHEMA,
                "contact_triangle_distance_m": float(witness_distance),
                "runner_up_triangle_distance_m": (
                    float(runner_distance)
                    if runner_distance is not None else None),
                "triangle_distance_margin_m": margin,
                "global_query_protocol":
                    B1K_COMPLETE_TRIANGLE_UNIVERSE_PROTOCOL,
                "scene_authority_sha256": self._scene_authority_sha256,
                "global_universe_sha256":
                    complete_triangle_universe_sha256(
                        global_query_protocol=
                            B1K_COMPLETE_TRIANGLE_UNIVERSE_PROTOCOL,
                        scene_authority_sha256=
                            self._scene_authority_sha256),
                "global_winner_instance_id": int(winner_id),
                "global_runner_up_instance_id": (
                    int(runner_id) if runner_id is not None else None),
            }
            if runner_id is None or margin is None:
                results.append({
                    "confirmed": False,
                    "reason": "global_runner_evidence_missing",
                    "instance_id": None,
                    **common,
                })
            elif margin <= config.A3_CONTACT_FACE_TIE_MARGIN_M + 1e-9:
                results.append({
                    "confirmed": False,
                    "reason": "contact_triangle_identity_near_tie",
                    "instance_id": None,
                    **common,
                })
            elif winner_id != instance_id:
                results.append({
                    "confirmed": False,
                    "reason": "global_contact_winner_mismatch",
                    "instance_id": None,
                    **common,
                })
            elif witness_distance > config.A3_CONTACT_FACE_MAX_DISTANCE_M + 1e-9:
                results.append({
                    "confirmed": False,
                    "reason": "contact_triangle_too_far",
                    "instance_id": None,
                    **common,
                })
            else:
                results.append({
                    "confirmed": True,
                    "reason": "confirmed",
                    "instance_id": int(instance_id),
                    "category": self.id_to_cat[instance_id],
                    "runtime_instance_triangles_sha256":
                        self.instance_triangles_sha256(instance_id),
                    **common,
                })
        return results

    def target_geometry_atom(
            self, instance_id: int, floor_plane, *,
            expected_scene_authority_sha256: str,
            pose: dict | None = None) -> dict:
        instance_id = _positive_instance_id(instance_id)
        expected = str(expected_scene_authority_sha256)
        if expected != self._scene_authority_sha256:
            raise ValueError(
                "scene authority digest does not match B1K runtime authority")
        triangles = self.instance_triangles(instance_id)
        support = semantic.clip_target_ground_support(
            triangles, floor_plane, pose=pose)
        centroid = semantic.area_weighted_surface_centroid(triangles)
        support_value = {
            "protocol": semantic.B_GROUND_SUPPORT_PROTOCOL,
            "frame": "pbench_world_xz",
            "ground_band_m": [
                float(config.GROUND_OBSTACLE_BAND_M[0]),
                float(config.GROUND_OBSTACLE_BAND_M[1]),
            ],
            **support,
        }
        support_value["sha256"] = record.canonical_atom_sha256(support_value)
        centroid_value = {
            "protocol": semantic.B_REFERENCE_CENTROID_PROTOCOL,
            "frame": B1K_TRIANGLE_FRAME,
            "world_xyz_m": centroid.tolist(),
            "world_xz_m": centroid[[0, 2]].tolist(),
        }
        centroid_value["sha256"] = record.canonical_atom_sha256(
            centroid_value)
        value = {
            "schema": record.B1K_B_TARGET_GEOMETRY_SCHEMA,
            "instance_id": instance_id,
            "category": self.id_to_cat[instance_id],
            "scene_authority_sha256": expected,
            "full_triangle_protocol": record.B1K_INSTANCE_TRIANGLES_SCHEMA,
            "full_triangle_count": int(len(triangles)),
            "full_triangles_sha256":
                self.instance_triangles_sha256(instance_id),
            "ground_support": support_value,
            "reference_centroid": centroid_value,
        }
        return {**value, "sha256": record.canonical_atom_sha256(value)}


def build_b1k_authorities(
        *, floor_components, collision_components, instances,
        canonical_replay_protocol: str | None = None,
        canonical_state_sha256: str | None = None):
    """Construct geometry and semantics bound to the same derived atom."""
    atom = derive_scene_authority_atom(
        floor_components=floor_components,
        collision_components=collision_components,
        instances=instances,
        canonical_replay_protocol=canonical_replay_protocol,
        canonical_state_sha256=canonical_state_sha256)
    geometry = b1k_geometry.B1KGeometryAuthority(
        floor_components=floor_components,
        collision_components=collision_components,
        scene_authority_sha256=atom["sha256"],
        _derived_binding_token=
            b1k_geometry._DERIVED_SCENE_BINDING_TOKEN)
    semantics = B1KSemanticAuthority(
        instances, scene_authority_sha256=atom["sha256"],
        _derived_binding_token=_DERIVED_SCENE_BINDING_TOKEN)
    return geometry, semantics, atom
