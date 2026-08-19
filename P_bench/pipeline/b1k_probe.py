"""Pure diagnostic probes over loaded B1K adapter inputs."""

from __future__ import annotations

from collections.abc import Iterable
import math
import time

import numpy as np
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

from pipeline import b1k_geometry, b1k_semantic, config, perception


_POSE_SEARCH_POSITION_COUNT = 4
_POSE_SEARCH_YAWS_RAD = tuple(
    math.radians(value) for value in range(0, 360, 45))
_POSE_CONTENT_SEMANTIC_SAMPLE_COUNT = 2048


class B1KProbeBlockingError(RuntimeError):
    """A required geometry or C-space operation cannot be trusted."""


def _slice_footprint(components, floor_height_m: float):
    shapes = []
    for raw in components:
        component = (
            raw if isinstance(raw, b1k_geometry.TriangleComponent)
            else b1k_geometry.TriangleComponent(*raw))
        footprint = b1k_geometry._obstacle_footprint(
            component, float(floor_height_m))
        if not footprint.is_empty:
            shapes.append(footprint)
    return unary_union(shapes) if shapes else GeometryCollection()


def _coordinate_rows(geometry) -> Iterable[tuple[float, float]]:
    kind = geometry.geom_type
    if kind == "Polygon":
        yield from geometry.exterior.coords
        for ring in geometry.interiors:
            yield from ring.coords
    elif kind in {"LineString", "LinearRing"}:
        yield from geometry.coords
    elif kind == "Point":
        yield (float(geometry.x), float(geometry.y))
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _coordinate_rows(child)


def _directed_vertex_sampled_hausdorff_approximation(
        source, target) -> float | None:
    """Max source-boundary-vertex distance, not continuous Hausdorff."""
    if source.is_empty or target.is_empty:
        return None
    from shapely.geometry import Point

    distances = [Point(value).distance(target)
                 for value in _coordinate_rows(source)]
    return float(max(distances, default=0.0))


def collision_visual_slice_probe(
        collision_components, visual_components, *,
        floor_height_m: float) -> dict:
    """Compare real triangle-slice footprints without a mesh convex hull."""
    collision = _slice_footprint(collision_components, floor_height_m)
    visual = _slice_footprint(visual_components, floor_height_m)
    return {
        "schema": "b1k-collision-visual-slice-probe.v1",
        "floor_height_m": float(floor_height_m),
        "collision_slice_area_m2": float(collision.area),
        "visual_slice_area_m2": float(visual.area),
        "collision_minus_visual_area_m2": float(
            collision.difference(visual).area),
        "collision_to_visual_vertex_sampled_directed_hausdorff_approx_m":
            _directed_vertex_sampled_hausdorff_approximation(
                collision, visual),
    }


def _psnr_and_change(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    difference = (first.astype(np.float64) - second.astype(np.float64))
    mean_square = float(np.mean(difference * difference))
    psnr = (100.0 if mean_square == 0.0 else
            20.0 * math.log10(255.0 / math.sqrt(mean_square)))
    change = float(np.mean(np.any(first != second, axis=2)))
    return float(psnr), change


def _observation_diagnostics(
        observation, source_semantic, *, hfov, vfov) -> dict:
    depth = np.asarray(observation.depth)
    valid = np.isfinite(depth) & (depth > 0.0)
    return {
        "hfov_deg": float(hfov),
        "vfov_deg": float(vfov),
        "rgb_shape": list(observation.rgb.shape),
        "rgb_unique_count": int(np.unique(observation.rgb).size),
        "rgb_standard_deviation": float(np.std(observation.rgb)),
        "depth_shape": list(depth.shape),
        "valid_depth_ratio": float(np.mean(valid)),
        **source_semantic,
    }


def _source_semantic_diagnostics(
        session, observation, position, yaw: float) -> dict:
    points_camera, _pixels = perception.unproject(
        observation.depth, observation.K)
    points_local = perception.to_agent_ground(
        points_camera,
        camera_height=observation.sensor.nominal_camera_offset_m)
    points_world = perception.world_from_local(
        points_local, position, yaw)
    count = min(_POSE_CONTENT_SEMANTIC_SAMPLE_COUNT, len(points_world))
    if count:
        selected = np.linspace(
            0, len(points_world) - 1, count).astype(np.int64)
        assigned = session.assign_instances(points_world[selected])
    else:
        assigned = np.empty(0, dtype=np.int64)
    resolved = assigned[assigned != 0]
    resolved_ids = sorted(int(value) for value in np.unique(resolved))
    predicate_categories = getattr(
        session, "id_to_predicate_cat", session.id_to_cat)
    nonstructural_ids = [
        instance_id for instance_id in resolved_ids
        if session.id_to_cat.get(instance_id) and
        not config.is_structural(predicate_categories.get(
            instance_id, session.id_to_cat[instance_id]))
    ]
    return {
        "source_semantic_assign_tolerance_m":
            float(config.SEMANTIC_ASSIGN_TOL_M),
        "source_semantic_sample_count": int(count),
        "source_semantic_resolved_count": int(len(resolved)),
        "source_semantic_resolution_rate": (
            float(len(resolved) / count) if count else 0.0),
        "source_instance_id_count": len(resolved_ids),
        "source_nonstructural_instance_id_count": len(nonstructural_ids),
        "source_nonstructural_instance_ids": nonstructural_ids,
    }


def _content_rejections(diagnostics: dict) -> list[str]:
    reasons = []
    if diagnostics["rgb_unique_count"] <= 1:
        reasons.append("constant_rgb")
    if diagnostics["valid_depth_ratio"] <= 0.0:
        reasons.append("no_valid_depth")
    if diagnostics["source_instance_id_count"] <= 1:
        reasons.append("insufficient_source_instance_labels")
    return reasons


def run_adapter_probe(
        session, *, fovs, seed: int = 0,
        semantic_sample_count: int = 256) -> dict:
    """Run non-gating diagnostics through one source-bound B1K session.

    Geometry construction and C-space sampling/querying are deliberately not
    caught: those are the Task-4 execution blockers.  Visual comparison is a
    diagnostic and records its own error instead.
    """
    selected_fovs = [tuple(map(float, value)) for value in fovs]
    if not selected_fovs:
        raise ValueError("B1K probe requires at least one FOV")
    radius = max(float(value) for value in config.RADII_M)
    try:
        session.recompute_navmesh(radius)
        session.pathfinder.seed(int(seed))
    except Exception as error:
        raise B1KProbeBlockingError(
            "B1K probe could not initialize its C-space authority") from error
    attempts = []
    selected = None
    last_candidate = None
    for position_index in range(_POSE_SEARCH_POSITION_COUNT):
        try:
            position = session.pathfinder.get_random_navigable_point()
            cspace_query = session.nav(
                position, 0.0).query_pose((0.0, 0.0, 0.0))
        except Exception as error:
            raise B1KProbeBlockingError(
                "B1K probe C-space sampling or query failed") from error
        if not cspace_query.navigable:
            raise B1KProbeBlockingError(
                "B1K sampled probe pose failed C-space query")
        for yaw_index, yaw in enumerate(_POSE_SEARCH_YAWS_RAD):
            frames = []
            observations = []
            rejections = []
            for fov_index, (hfov, vfov) in enumerate(selected_fovs):
                rendered = session.render(
                    position, yaw, config.CAMERA_HEIGHT_M, hfov, vfov)
                source_semantic = _source_semantic_diagnostics(
                    session, rendered, position, yaw)
                diagnostics = _observation_diagnostics(
                    rendered, source_semantic,
                    hfov=hfov, vfov=vfov)
                reasons = _content_rejections(diagnostics)
                if reasons:
                    rejections.append({
                        "fov_index": fov_index,
                        "hfov_deg": hfov,
                        "vfov_deg": vfov,
                        "reasons": reasons,
                    })
                frames.append(rendered)
                observations.append(diagnostics)
            attempt = {
                "attempt_index": len(attempts),
                "position_index": position_index,
                "yaw_index": yaw_index,
                "position_pbench_xyz_m": np.asarray(position).tolist(),
                "yaw_rad": float(yaw),
                "rejections": rejections,
            }
            attempts.append(attempt)
            last_candidate = (
                np.asarray(position).copy(), float(yaw),
                frames, observations, attempt)
            if not rejections:
                selected = last_candidate
                break
        if selected is not None:
            break
    if last_candidate is None:
        raise RuntimeError("B1K pose search produced no candidates")
    position, selected_yaw, frames, observations, selected_attempt = (
        selected if selected is not None else last_candidate)
    first = frames[0]
    first_hfov, first_vfov = selected_fovs[0]
    repeated = session.render(
        position, selected_yaw, config.CAMERA_HEIGHT_M,
        first_hfov, first_vfov)
    psnr, change = _psnr_and_change(first.rgb, repeated.rgb)

    points_camera, pixels = perception.unproject(first.depth, first.K)
    points_local = perception.to_agent_ground(
        points_camera, camera_height=first.sensor.nominal_camera_offset_m)
    points_world = perception.world_from_local(
        points_local, position, selected_yaw)
    count = min(int(semantic_sample_count), len(points_world))
    if count:
        selected = np.linspace(0, len(points_world) - 1, count).astype(int)
        sampled_world = points_world[selected]
        sampled_pixels = pixels[selected]
    else:
        sampled_world = np.empty((0, 3), dtype=np.float64)
        sampled_pixels = np.empty((0, 2), dtype=np.int64)
    started = time.perf_counter()
    assigned = session.semantic_index.assign(sampled_world, tol=0.025)
    elapsed = time.perf_counter() - started
    unresolved = np.flatnonzero(assigned == 0)
    semantic_resolution = {
        "schema": "b1k-semantic-resolution-probe.v1",
        "query_radius_m": 0.025,
        "sample_count": int(count),
        "resolved_count": int(np.count_nonzero(assigned)),
        "resolution_rate": (
            float(np.mean(assigned != 0)) if count else 0.0),
        "elapsed_seconds": float(elapsed),
        "aabb_prefilter": dict(
            getattr(session.semantic_index, "last_assign_diagnostics", {})),
        "unresolved_examples": [
            {
                "world_xyz_m": sampled_world[index].tolist(),
                "pixel_uv": sampled_pixels[index].tolist(),
            }
            for index in unresolved[:5]
        ],
    }

    try:
        components = session.diagnostic_geometry_components()
        levels = b1k_geometry._floor_level_groups(components["floor"])
        slice_rows = [
            collision_visual_slice_probe(
                components["collision"], components["visual"],
                floor_height_m=ground_y)
            for ground_y, _triangles in levels
        ]
        collision_visual = {
            "status": "computed",
            "visual_component_count": len(components["visual"]),
            "levels": slice_rows,
        }
    except Exception as error:
        collision_visual = {
            "status": "diagnostic_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }

    reset_errors = dict(session.last_reset_pose_errors)
    reset_gate = max(reset_errors.values(), default=0.0) <= 1e-6
    smoke_pass = all(
        value["rgb_unique_count"] > 1 and
        value["valid_depth_ratio"] > 0.0 and
        value["source_instance_id_count"] > 1
        for value in observations)
    return {
        "schema": "b1k-adapter-probe.v1",
        "scene_id": session.scene_id,
        "scene_authority_sha256": session.scene_authority_sha256,
        "sample_pose": {
            "position_pbench_xyz_m": np.asarray(position).tolist(),
            "yaw_rad": selected_yaw,
            "sampling_authority": "b1k_geometry",
            "radius_m": radius,
        },
        "pose_search": {
            "schema": "b1k-probe-pose-search.v1",
            "max_position_count": _POSE_SEARCH_POSITION_COUNT,
            "yaw_candidates_rad": list(_POSE_SEARCH_YAWS_RAD),
            "attempt_count": len(attempts),
            "selected_attempt_index": (
                selected_attempt["attempt_index"]
                if selected is not None else None),
            "content_gate_pass": selected is not None,
            "attempts": attempts,
        },
        "observations": observations,
        "reset": {
            **reset_errors,
            "hard_gate_tolerance": 1e-6,
            "hard_gate_pass": bool(reset_gate),
            "rgb_psnr_db": psnr,
            "rgb_change_ratio": change,
            "image_metrics_are_diagnostic": True,
        },
        "semantic_resolution": semantic_resolution,
        "collision_visual": collision_visual,
        "smoke_pass": bool(
            smoke_pass and reset_gate and selected is not None),
    }
