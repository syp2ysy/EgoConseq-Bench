"""Exact MP3D contact-face queries with conservative instance pruning.

This module owns execution mechanics only.  The caller supplies authenticated
PLY layout/index callbacks, so source binding and cache policy remain in
``pipeline.semantic`` while this implementation stays independently testable
and free of import cycles.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import threading

import numpy as np

from pipeline import config


_QUERY_DIAGNOSTICS = Counter()
_QUERY_PROFILE_ROWS = []


def reset_diagnostics() -> None:
    _QUERY_DIAGNOSTICS.clear()
    _QUERY_PROFILE_ROWS.clear()


def diagnostics() -> dict:
    return dict(_QUERY_DIAGNOSTICS)


def profile_rows() -> list[dict]:
    """Return execution-only pruning rows; these never enter a record."""
    return [dict(row) for row in _QUERY_PROFILE_ROWS]


def record_fallback() -> None:
    _QUERY_DIAGNOSTICS["fallback_invocations"] += 1


def branch_and_bound_instance_top_two(
        point: np.ndarray, instance_ids: np.ndarray,
        aabb_minima: np.ndarray, aabb_maxima: np.ndarray,
        exact_distance, *,
        slack_m: float = config.A3_FACE_AABB_PRUNE_SLACK_M
        ) -> tuple[list[tuple[float, int]], list[int]]:
    """Return exact ``(distance, instance_id)`` top two with AABB pruning."""
    query = np.asarray(point, dtype=np.float64)
    ids = np.asarray(instance_ids, dtype=np.int64)
    minima = np.asarray(aabb_minima, dtype=np.float64)
    maxima = np.asarray(aabb_maxima, dtype=np.float64)
    if (query.shape != (3,) or minima.shape != maxima.shape or
            minima.shape != (len(ids), 3) or
            not np.all(np.isfinite(query)) or
            not np.all(np.isfinite(minima)) or
            not np.all(np.isfinite(maxima)) or
            np.any(maxima < minima)):
        raise ValueError("instance AABB query inputs are invalid")
    slack = float(slack_m)
    if not math.isfinite(slack) or slack < 0.0:
        raise ValueError("instance AABB prune slack must be nonnegative")
    outside = np.maximum(np.maximum(minima - query, query - maxima), 0.0)
    lower_bounds = np.linalg.norm(outside, axis=1)
    order = np.lexsort((ids, lower_bounds))
    ranked: list[tuple[float, int]] = []
    evaluated = []
    for raw_index in order:
        index = int(raw_index)
        if (len(ranked) >= 2 and
                float(lower_bounds[index]) - slack > ranked[1][0]):
            break
        instance_id = int(ids[index])
        distance = float(exact_distance(instance_id))
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("exact instance face distance is invalid")
        evaluated.append(instance_id)
        ranked.append((distance, instance_id))
        ranked.sort()
        del ranked[2:]
    return ranked, evaluated


def query_exhaustive(
        pinned, file_size: int, semantic_ply,
        query_points: np.ndarray, witness_ids: list[int],
        instance_categories: dict[int, str],
        expected_semantic_ply_sha256: str, *, backend: dict,
        query_workers: int = 1) -> dict:
    """Frozen exhaustive reference over one pinned read-only PLY buffer."""
    del query_workers
    import trimesh

    header_size, vertex_count, face_count = \
        backend["binary_ply_layout_from_buffer"](pinned, semantic_ply)
    vertex_dtype = backend["vertex_dtype"]
    triangle_dtype = backend["triangle_dtype"]
    face_offset = header_size + vertex_count * vertex_dtype.itemsize
    parsed_endpoint = face_offset + face_count * triangle_dtype.itemsize
    if parsed_endpoint != file_size:
        raise ValueError("MP3D semantic PLY parsed endpoint is not file size")
    raw_file = np.frombuffer(pinned, dtype=np.uint8, count=file_size)
    source_hasher = hashlib.sha256()
    source_hasher.update(memoryview(raw_file[:face_offset]))
    vertices = np.frombuffer(
        pinned, dtype=vertex_dtype, count=vertex_count, offset=header_size)
    faces = np.frombuffer(
        pinned, dtype=triangle_dtype, count=face_count, offset=face_offset)
    minima = [{} for _point in query_points]
    requested_ids = set(witness_ids)
    streaming_hashers = {
        instance_id: hashlib.sha256() for instance_id in requested_ids}
    streaming_face_counts = dict.fromkeys(requested_ids, 0)
    for start in range(0, face_count, 200_000):
        chunk = faces[start:start + 200_000]
        byte_start = face_offset + start * triangle_dtype.itemsize
        byte_end = byte_start + len(chunk) * triangle_dtype.itemsize
        source_hasher.update(memoryview(raw_file[byte_start:byte_end]))
        if len(chunk) and not np.all(chunk["count"] == 3):
            raise ValueError("MP3D semantic PLY contains non-triangle faces")
        raw_ids = np.asarray(chunk["object_id"], dtype=np.int64)
        if np.any(raw_ids < -1):
            raise ValueError("MP3D semantic PLY contains invalid object IDs")
        keep = raw_ids >= 0
        if not np.any(keep):
            continue
        raw_ids = raw_ids[keep]
        source = backend["source_triangles"](
            vertices, chunk["vertex_indices"][keep], vertex_count)
        world = np.stack([
            source[:, :, 0], source[:, :, 2], -source[:, :, 1],
        ], axis=-1).astype(np.float64)
        instance_ids = raw_ids + 1
        unique_ids, inverse_ids = np.unique(
            instance_ids, return_inverse=True)
        for instance_id in requested_ids & set(
                int(value) for value in unique_ids):
            selected = world[instance_ids == instance_id].astype(np.float32)
            order = np.lexsort(
                (selected[:, :, 2], selected[:, :, 1], selected[:, :, 0]),
                axis=1)
            canonical_vertices = np.take_along_axis(
                selected, order[:, :, None], axis=1).astype("<f4", copy=False)
            streaming_hashers[instance_id].update(
                canonical_vertices.tobytes(order="C"))
            streaming_face_counts[instance_id] += len(selected)
        for query_index, point in enumerate(query_points):
            repeated = np.repeat(point[None, :], len(world), axis=0)
            nearest = trimesh.triangles.closest_point(world, repeated)
            distances = np.linalg.norm(nearest - repeated, axis=1)
            per_instance = np.full(len(unique_ids), np.inf, dtype=np.float64)
            np.minimum.at(per_instance, inverse_ids, distances)
            for instance_id, minimum in zip(unique_ids, per_instance):
                normalized = int(instance_id)
                minima[query_index][normalized] = min(
                    minima[query_index].get(normalized, float("inf")),
                    float(minimum))
    if not minima[0]:
        raise ValueError("MP3D semantic PLY contains no face-instance universe")
    actual_source_sha256 = source_hasher.hexdigest()
    if actual_source_sha256 != expected_semantic_ply_sha256:
        raise ValueError("MP3D semantic PLY source digest changed")
    queries = []
    for query_minima, witness_instance_id in zip(minima, witness_ids):
        ranked = sorted(
            query_minima.items(), key=lambda value: (value[1], value[0]))
        witness_distance = query_minima.get(witness_instance_id)
        if witness_distance is None:
            raise KeyError(
                f"MP3D semantic instance {witness_instance_id} has no triangles")
        runner = ranked[1] if len(ranked) > 1 else (None, None)
        queries.append({
            "global_winner_instance_id": int(ranked[0][0]),
            "global_winner_face_distance_m": float(ranked[0][1]),
            "global_runner_up_instance_id": (
                int(runner[0]) if runner[0] is not None else None),
            "global_runner_up_face_distance_m": (
                float(runner[1]) if runner[1] is not None else None),
            "witness_face_distance_m": float(witness_distance),
        })
    streaming_digests = {}
    for instance_id in requested_ids:
        if not streaming_face_counts[instance_id]:
            continue
        value = {
            "schema": "mp3d-streaming-instance-faces.v1",
            "protocol": backend["streaming_protocol"],
            "semantic_ply_sha256": actual_source_sha256,
            "instance_id": instance_id,
            "category": instance_categories[instance_id],
            "face_count": streaming_face_counts[instance_id],
            "geometry_payload_sha256":
                streaming_hashers[instance_id].hexdigest(),
        }
        streaming_digests[instance_id] = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")).hexdigest()
    return {
        "queries": queries,
        "streaming_instance_faces_sha256": streaming_digests,
    }


def query_bounded(
        pinned, file_size: int, semantic_ply,
        query_points: np.ndarray, witness_ids: list[int],
        instance_categories: dict[int, str],
        expected_semantic_ply_sha256: str, *, backend: dict,
        query_workers: int = 1) -> dict:
    """Resolve exact top-two instances after conservative AABB pruning."""
    import trimesh

    raw_view = memoryview(pinned)[:file_size]
    try:
        actual_source_sha256 = hashlib.sha256(raw_view).hexdigest()
    finally:
        raw_view.release()
    if actual_source_sha256 != expected_semantic_ply_sha256:
        raise ValueError("MP3D semantic PLY source digest changed")
    header_size, vertex_count, face_count = \
        backend["binary_ply_layout_from_buffer"](pinned, semantic_ply)
    vertex_dtype = backend["vertex_dtype"]
    triangle_dtype = backend["triangle_dtype"]
    face_offset = header_size + vertex_count * vertex_dtype.itemsize
    if face_offset + face_count * triangle_dtype.itemsize != file_size:
        raise ValueError("MP3D semantic PLY parsed endpoint is not file size")
    index = backend["authenticated_face_index"](
        semantic_ply, expected_semantic_ply_sha256)
    if (index["header_size"] != header_size or
            index["vertex_count"] != vertex_count or
            index["face_count"] != face_count or
            index["face_offset"] != face_offset):
        raise ValueError("authenticated MP3D face-index layout changed")
    instance_ids, aabb_minima, aabb_maxima = \
        backend["ensure_instance_aabbs"](pinned, index)
    vertices = np.frombuffer(
        pinned, dtype=vertex_dtype,
        count=vertex_count, offset=header_size)
    face_rows_cache: dict[int, np.ndarray] = {}
    triangles_cache: dict[int, np.ndarray] = {}
    triangle_cache_lock = threading.Lock()

    def instance_triangles(instance_id: int) -> np.ndarray:
        normalized = int(instance_id)
        with triangle_cache_lock:
            cached = triangles_cache.get(normalized)
            if cached is not None:
                return cached
            raw_object_id = normalized - 1
            rows = face_rows_cache.get(normalized)
            if rows is None:
                rows = np.flatnonzero(index["object_ids"] == raw_object_id)
                face_rows_cache[normalized] = rows
            if not len(rows):
                triangles = np.empty((0, 3, 3), dtype=np.float64)
            else:
                source = backend["source_triangles"](
                    vertices, index["vertex_indices"][rows], vertex_count)
                triangles = np.stack([
                    source[:, :, 0], source[:, :, 2], -source[:, :, 1],
                ], axis=-1).astype(np.float64)
            triangles_cache[normalized] = triangles
            return triangles

    queries = [None] * len(query_points)
    point_groups = {}
    for query_index, (point, witness_instance_id) in enumerate(
            zip(query_points, witness_ids)):
        normalized_point = np.ascontiguousarray(point, dtype="<f8")
        key = normalized_point.tobytes(order="C")
        group = point_groups.setdefault(key, {
            "point": normalized_point,
            "requests": [],
        })
        group["requests"].append((query_index, int(witness_instance_id)))
    _QUERY_DIAGNOSTICS["bounded_request_points"] += len(query_points)
    _QUERY_DIAGNOSTICS[
        "bounded_unique_query_points"] += len(point_groups)
    try:
        def resolve_group(group):
            point = group["point"]
            distance_cache = {}
            query_faces_evaluated = 0

            def exact_distance(instance_id: int) -> float:
                nonlocal query_faces_evaluated
                normalized = int(instance_id)
                if normalized in distance_cache:
                    return distance_cache[normalized]
                triangles = instance_triangles(normalized)
                if not len(triangles):
                    raise KeyError(
                        f"MP3D semantic instance {normalized} has no triangles")
                repeated = np.repeat(
                    np.asarray(point, dtype=np.float64)[None, :],
                    len(triangles), axis=0)
                nearest = trimesh.triangles.closest_point(
                    triangles, repeated)
                distance = float(np.min(np.linalg.norm(
                    nearest - repeated, axis=1)))
                distance_cache[normalized] = distance
                query_faces_evaluated += int(len(triangles))
                return distance

            ranked, evaluated = branch_and_bound_instance_top_two(
                point, instance_ids, aabb_minima, aabb_maxima,
                exact_distance)
            if not ranked:
                raise ValueError(
                    "MP3D semantic PLY contains no face-instance universe")
            runner = ranked[1] if len(ranked) > 1 else (None, None)
            query_values = []
            for query_index, witness_instance_id in group["requests"]:
                witness_distance = exact_distance(witness_instance_id)
                query_values.append((query_index, {
                    "global_winner_instance_id": int(ranked[0][1]),
                    "global_winner_face_distance_m": float(ranked[0][0]),
                    "global_runner_up_instance_id": (
                        int(runner[1]) if runner[1] is not None else None),
                    "global_runner_up_face_distance_m": (
                        float(runner[0]) if runner[0] is not None else None),
                    "witness_face_distance_m": float(witness_distance),
                }))
            witness_categories = sorted({
                str(instance_categories[witness_instance_id])
                for _query_index, witness_instance_id in group["requests"]
            })
            return {
                "queries": query_values,
                "profile": {
                    "semantic_ply_sha256": actual_source_sha256,
                    "scene_face_count": int(face_count),
                    "scene_instance_count": int(len(instance_ids)),
                    "request_count": int(len(group["requests"])),
                    "unique_witness_instance_count": int(len({
                        witness_instance_id
                        for _query_index, witness_instance_id
                        in group["requests"]
                    })),
                    "witness_categories": witness_categories,
                    "shortlisted_instance_count": int(len(distance_cache)),
                    "shortlisted_face_count": int(query_faces_evaluated),
                    "shortlisted_face_ratio": (
                        float(query_faces_evaluated / face_count)
                        if face_count else 0.0),
                    "global_winner_instance_id": int(ranked[0][1]),
                    "global_runner_up_instance_id": (
                        int(runner[1]) if runner[1] is not None else None),
                },
                "faces_evaluated": int(query_faces_evaluated),
                "instances_evaluated": int(len(evaluated)),
            }

        groups = list(point_groups.values())
        workers = min(max(1, int(query_workers)), len(groups))
        if workers > 1:
            _QUERY_DIAGNOSTICS["bounded_parallel_batches"] += 1
            _QUERY_DIAGNOSTICS[
                "bounded_parallel_query_points"] += len(groups)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                resolved_groups = list(executor.map(resolve_group, groups))
        else:
            resolved_groups = [resolve_group(group) for group in groups]
        for resolved in resolved_groups:
            for query_index, query in resolved["queries"]:
                queries[query_index] = query
            _QUERY_PROFILE_ROWS.append(resolved["profile"])
            _QUERY_DIAGNOSTICS["bounded_query_points"] += 1
            _QUERY_DIAGNOSTICS[
                "bounded_faces_evaluated"] += resolved["faces_evaluated"]
            _QUERY_DIAGNOSTICS[
                "bounded_instances_evaluated"] += (
                    resolved["instances_evaluated"])

        streaming_digests = {}
        for instance_id in set(int(value) for value in witness_ids):
            selected = instance_triangles(instance_id).astype(np.float32)
            if not len(selected):
                continue
            order = np.lexsort(
                (selected[:, :, 2], selected[:, :, 1], selected[:, :, 0]),
                axis=1)
            canonical_vertices = np.take_along_axis(
                selected, order[:, :, None], axis=1).astype(
                    "<f4", copy=False)
            geometry_digest = hashlib.sha256(
                canonical_vertices.tobytes(order="C")).hexdigest()
            value = {
                "schema": "mp3d-streaming-instance-faces.v1",
                "protocol": backend["streaming_protocol"],
                "semantic_ply_sha256": actual_source_sha256,
                "instance_id": instance_id,
                "category": instance_categories[instance_id],
                "face_count": int(len(selected)),
                "geometry_payload_sha256": geometry_digest,
            }
            streaming_digests[instance_id] = hashlib.sha256(json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True).encode("ascii")).hexdigest()
        return {
            "queries": queries,
            "streaming_instance_faces_sha256": streaming_digests,
        }
    finally:
        face_rows_cache.clear()
        triangles_cache.clear()
        del vertices
