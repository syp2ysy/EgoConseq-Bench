"""Source-bound planar collision authority extracted from official SAGE-3D USD."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import shapely
from shapely.geometry import GeometryCollection

from pipeline import b1k_geometry, config


ARTIFACT_SCHEMA = "gs-collision-footprint-authority.v3"
AUTHORITY_NAME = "gs_collision_mesh"
CONTACT_SURFACE_PROTOCOL = "sage3d-source-collision-footprint.v1"
_SOURCE_ROLES = ("scene", "navmesh", "semantic", "collision_mesh")
_LEVEL_MERGE_TOLERANCE_M = 1e-6


@dataclass(frozen=True)
class SourceCollisionComponent:
    """One official USD mesh in deterministic source order.

    Unlike the B1K runtime component, this offline representation deliberately
    does not lexicographically sort millions of source triangles.  Planar
    union is independent of face order, while the source-file SHA binds the
    exact USD bytes used to derive the authority.
    """

    identity: str
    triangles: np.ndarray

    def __post_init__(self) -> None:
        identity = str(self.identity).strip()
        triangles = np.asarray(self.triangles, dtype=np.float64)
        if (not identity or triangles.ndim != 3 or
                triangles.shape[1:] != (3, 3) or
                not np.isfinite(triangles).all()):
            raise ValueError("GS collision component is invalid")
        if len(triangles):
            area_twice = np.linalg.norm(np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0]), axis=1)
            triangles = triangles[area_twice > 0.0]
        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self, "triangles", np.ascontiguousarray(triangles))


def _source_components(values) -> tuple[SourceCollisionComponent, ...]:
    components = []
    for value in values:
        if isinstance(value, SourceCollisionComponent):
            component = value
        elif hasattr(value, "identity") and hasattr(value, "triangles"):
            component = SourceCollisionComponent(
                value.identity, value.triangles)
        else:
            identity, triangles = value
            component = SourceCollisionComponent(identity, triangles)
        components.append(component)
    components.sort(key=lambda component: component.identity)
    identities = [component.identity for component in components]
    if not components or len(identities) != len(set(identities)):
        raise ValueError("GS collision component identities are invalid")
    return tuple(components)


def canonical_planar_union(geometries: Iterable[object]):
    """Union source projections after deterministic nanometre repair.

    Some official USD meshes contain coincident slivers whose independently
    valid projections trigger GEOS overlay topology errors.  This is offline
    numerical canonicalization, not a physical margin: coordinates are snapped
    at one nanometre, nine orders below the published 0.5 m action grid and far
    below the source mesh precision.  No source component is silently dropped.
    """
    grid = float(config.GS_COLLISION_PLANAR_GRID_M)
    values = [geometry for geometry in geometries if geometry is not None]
    if not values:
        return GeometryCollection()
    array = np.empty(len(values), dtype=object)
    array[:] = values
    array = array[~shapely.is_empty(array)]
    if not len(array):
        return GeometryCollection()
    invalid = ~shapely.is_valid(array)
    if np.any(invalid):
        array[invalid] = shapely.make_valid(array[invalid])
    array = shapely.set_precision(array, grid, mode="valid_output")
    array = array[~shapely.is_empty(array)]
    if not len(array):
        return GeometryCollection()
    merged = shapely.union_all(array, grid_size=grid)
    if not merged.is_valid:
        merged = shapely.make_valid(merged)
    return shapely.normalize(merged)


def planar_overlap_area(first, second) -> float:
    """Return deterministic overlap for already-canonical source surfaces."""
    grid = float(config.GS_COLLISION_PLANAR_GRID_M)
    values = np.empty(2, dtype=object)
    values[:] = (first, second)
    invalid = ~shapely.is_valid(values)
    if np.any(invalid):
        values[invalid] = shapely.make_valid(values[invalid])
    values = shapely.set_precision(values, grid, mode="valid_output")
    overlap = shapely.intersection(
        values[0], values[1], grid_size=grid)
    return float(shapely.area(overlap))


def clipped_triangle_projections(
        values, low: float, high: float) -> tuple[object, ...]:
    """Project triangle/slab intersections without per-face GEOS calls.

    The intersection of a triangle and a horizontal slab is convex.  Its
    vertices are exactly the original vertices inside the slab plus edge-plane
    intersections.  Batched convex hull therefore preserves the scalar frozen
    clipping contract for polygons, vertical lines, and point contacts.
    """
    triangles = np.asarray(values, dtype=np.float64)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or
            not np.isfinite(triangles).all()):
        raise ValueError("GS slab projection requires finite Fx3x3 triangles")
    lower, upper = float(low), float(high)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("GS slab projection bounds are invalid")
    if not len(triangles):
        return ()
    heights = triangles[:, :, 1]
    intersects = (
        np.max(heights, axis=1) >= lower) & (
        np.min(heights, axis=1) <= upper)
    triangles = triangles[intersects]
    if not len(triangles):
        return ()

    count = len(triangles)
    slots = np.zeros((count, 9, 2), dtype=np.float64)
    present = np.zeros((count, 9), dtype=bool)
    heights = triangles[:, :, 1]
    inside = (heights >= lower) & (heights <= upper)
    for vertex in range(3):
        slots[:, vertex] = triangles[:, vertex][:, [0, 2]]
        present[:, vertex] = inside[:, vertex]

    slot = 3
    for start, end in ((0, 1), (1, 2), (2, 0)):
        first = triangles[:, start]
        second = triangles[:, end]
        first_y = first[:, 1]
        second_y = second[:, 1]
        for boundary in (lower, upper):
            crosses = ((first_y < boundary) & (second_y > boundary)) | (
                (first_y > boundary) & (second_y < boundary))
            if np.any(crosses):
                fraction = (
                    (boundary - first_y[crosses]) /
                    (second_y[crosses] - first_y[crosses]))
                points = first[crosses] + fraction[:, None] * (
                    second[crosses] - first[crosses])
                slots[crosses, slot] = points[:, [0, 2]]
                present[crosses, slot] = True
            slot += 1

    flat_present = present.reshape(-1)
    coordinates = slots.reshape(-1, 2)[flat_present]
    indices = np.repeat(np.arange(count, dtype=np.int64), 9)[flat_present]
    multipoints = shapely.multipoints(coordinates, indices=indices)
    hulls = shapely.convex_hull(multipoints)
    hulls = hulls[~shapely.is_empty(hulls)]
    return tuple(hulls.tolist())


def _floor_level_groups(components: tuple[SourceCollisionComponent, ...]
                        ) -> list[tuple[float, np.ndarray]]:
    """Vectorized equivalent of the shared scalar floor grouping contract."""
    height_rows = []
    triangle_rows = []
    for component in components:
        triangles = component.triangles
        if not len(triangles):
            continue
        heights = triangles[:, :, 1]
        horizontal = np.ptp(heights, axis=1) <= _LEVEL_MERGE_TOLERANCE_M
        if np.any(horizontal):
            height_rows.append(np.mean(heights[horizontal], axis=1))
            triangle_rows.append(triangles[horizontal])
    if not height_rows:
        raise ValueError("GS floor collision support is empty")
    heights = np.concatenate(height_rows)
    triangles = np.concatenate(triangle_rows)
    order = np.argsort(heights, kind="stable")
    heights = heights[order]
    triangles = triangles[order]
    groups = []
    start = 0
    while start < len(heights):
        stop = int(np.searchsorted(
            heights, heights[start] + _LEVEL_MERGE_TOLERANCE_M,
            side="right"))
        groups.append((
            float(np.mean(heights[start:stop])), triangles[start:stop]))
        start = stop
    return groups


def _triangle_polygons(triangles: np.ndarray) -> tuple[object, ...]:
    coordinates = np.asarray(triangles, dtype=np.float64)[:, :, [0, 2]]
    polygons = shapely.polygons(coordinates)
    polygons = polygons[shapely.area(polygons) > 0.0]
    return tuple(polygons.tolist())


def interiorgs_zup_to_pbench_xyz(values) -> np.ndarray:
    """Map official InteriorGS collision XYZ (Z-up) to PBench Y-up XYZ."""
    points = np.asarray(values, dtype=np.float64)
    if (points.ndim < 1 or points.shape[-1] != 3 or
            not np.isfinite(points).all()):
        raise ValueError("InteriorGS collision coordinates must be finite XYZ")
    return np.stack(
        [points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def build_planar_levels(*, collision_components,
                        navmesh_triangles) -> tuple["FrozenPlanarLevel", ...]:
    """Project official collision components onto navmesh-supported floors.

    The navmesh identifies which horizontal source surfaces are walkable; it
    never supplies obstacle labels.  Obstacle support is clipped exclusively
    from the official collision USD in the frozen ground-height band.
    """
    components = _source_components(collision_components)
    navmesh = b1k_geometry.canonical_triangles(navmesh_triangles)
    if not components or not len(navmesh):
        raise ValueError("GS collision conversion requires source and navmesh")
    nav_support = canonical_planar_union(_triangle_polygons(navmesh))
    if nav_support.is_empty or nav_support.area <= 0.0:
        raise ValueError("GS navmesh has no planar support")
    nav_heights = np.mean(navmesh[:, :, 1], axis=1)
    floor_groups = _floor_level_groups(components)
    levels = []
    identities = tuple(component.identity for component in components)
    for ground_y, triangles in floor_groups:
        if float(np.min(np.abs(nav_heights - ground_y))) > \
                float(config.NAV_Y_DELTA_M) + 1e-9:
            continue
        floor_shapes = _triangle_polygons(triangles)
        if not floor_shapes:
            continue
        support = canonical_planar_union(floor_shapes)
        if planar_overlap_area(support, nav_support) <= 0.0:
            continue
        low, high = (
            float(ground_y) + float(value)
            for value in config.GROUND_OBSTACLE_BAND_M)
        footprints = tuple(canonical_planar_union(
            clipped_triangle_projections(component.triangles, low, high)
        ) for component in components)
        levels.append(FrozenPlanarLevel(
            ground_y_m=ground_y,
            support=support,
            obstacle_identities=identities,
            obstacle_footprints=footprints,
        ))
    if not levels:
        raise ValueError(
            "GS collision source has no navmesh-supported floor level")
    return tuple(sorted(levels, key=lambda level: level.ground_y_m))


@dataclass(frozen=True)
class FrozenPlanarLevel:
    """One floor level plus source-component obstacle footprints in metres."""

    ground_y_m: float
    support: object
    obstacle_identities: tuple[str, ...]
    obstacle_footprints: tuple[object, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.ground_y_m)):
            raise ValueError("GS collision ground height must be finite")
        if self.support.is_empty or self.support.area <= 0.0:
            raise ValueError("GS collision support must have positive area")
        if len(self.obstacle_identities) != len(self.obstacle_footprints):
            raise ValueError("GS collision obstacle identities are incomplete")
        if len(set(self.obstacle_identities)) != len(
                self.obstacle_identities):
            raise ValueError("GS collision obstacle identities are duplicated")
        for identity, footprint in zip(
                self.obstacle_identities, self.obstacle_footprints):
            if not isinstance(identity, str) or not identity:
                raise ValueError("GS collision obstacle identity is invalid")


def _canonical_sha256(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii"))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _encode_geometries(geometries: Iterable[object]) -> dict[str, np.ndarray]:
    payload = bytearray()
    offsets = [0]
    for geometry in geometries:
        if geometry is None:
            raise ValueError("GS collision geometry is missing")
        normalized = shapely.normalize(geometry)
        coordinates = shapely.get_coordinates(normalized)
        if not np.isfinite(coordinates).all():
            raise ValueError("GS collision geometry coordinates are invalid")
        encoded = shapely.to_wkb(
            normalized, byte_order=1, include_srid=False,
            output_dimension=2)
        payload.extend(encoded)
        offsets.append(len(payload))
    return {
        "geometry_wkb_bytes": np.frombuffer(
            bytes(payload), dtype=np.uint8).copy(),
        "geometry_wkb_offsets": np.asarray(offsets, dtype=np.int64),
    }


def _decode_geometries(arrays: dict[str, np.ndarray]) -> tuple[object, ...]:
    payload = np.asarray(arrays["geometry_wkb_bytes"], dtype=np.uint8)
    offsets = np.asarray(arrays["geometry_wkb_offsets"], dtype=np.int64)
    if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or
            offsets[-1] != len(payload) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("GS collision geometry offsets are invalid")
    output = []
    try:
        for index in range(len(offsets) - 1):
            encoded = payload[
                int(offsets[index]):int(offsets[index + 1])].tobytes()
            output.append(shapely.from_wkb(encoded))
    except Exception as error:
        raise ValueError("GS collision geometry payload is invalid") from error
    return tuple(output)


def _normalized_source_assets(source_assets) -> tuple[tuple[str, int, str], ...]:
    values = tuple(
        (str(role), int(byte_size), str(digest))
        for role, byte_size, digest in source_assets)
    if tuple(value[0] for value in values) != _SOURCE_ROLES:
        raise ValueError("GS collision source asset roles are invalid")
    for _role, byte_size, digest in values:
        if byte_size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("GS collision source asset identity is invalid")
    return values


def write_collision_artifact(path, *, scene_id: str, source_assets,
                             levels: Iterable[FrozenPlanarLevel]) -> dict:
    """Write one deterministic, non-pickle authority artifact."""
    scene = str(scene_id)
    if not scene:
        raise ValueError("GS collision scene id is empty")
    assets = _normalized_source_assets(source_assets)
    levels = tuple(levels)
    if not levels:
        raise ValueError("GS collision artifact has no levels")
    identities = levels[0].obstacle_identities
    if any(level.obstacle_identities != identities for level in levels):
        raise ValueError("GS collision levels disagree on component identity")
    geometries = []
    level_rows = []
    for level in levels:
        support_index = len(geometries)
        geometries.append(level.support)
        obstacle_start = len(geometries)
        geometries.extend(level.obstacle_footprints)
        radius_rows = []
        for radius in config.RADII_M:
            supported = level.support.buffer(
                -float(radius),
                quad_segs=b1k_geometry._BUFFER_QUAD_SEGS)
            supported_index = len(geometries)
            geometries.append(supported)
            radius_rows.append({
                "radius_m": float(radius),
                "supported_geometry_index": supported_index,
            })
        level_rows.append({
            "ground_y_m": float(level.ground_y_m),
            "support_geometry_index": support_index,
            "obstacle_geometry_start": obstacle_start,
            "obstacle_geometry_count": len(level.obstacle_footprints),
            "radius_support": radius_rows,
        })
    arrays = _encode_geometries(geometries)
    metadata = {
        "schema": ARTIFACT_SCHEMA,
        "scene_id": scene,
        "source_assets": [
            {"role": role, "bytes": byte_size, "sha256": digest}
            for role, byte_size, digest in assets],
        "axis_protocol": "interiorgs-zup-to-pbench-yup.v1",
        "contact_surface_protocol": CONTACT_SURFACE_PROTOCOL,
        "collision_query_mode": "nearest_obstacle_distance",
        "obstacle_identities": list(identities),
        "levels": level_rows,
    }
    authority_sha256 = _canonical_sha256(metadata, arrays)
    metadata["authority_sha256"] = authority_sha256
    arrays = {
        **arrays,
        "metadata_json": np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True)),
    }
    np.savez_compressed(Path(path), **arrays)
    return {**metadata, "sha256": authority_sha256}


class GsCollisionAuthority:
    """Thin typed facade over the shared indexed planar geometry engine."""

    authority = AUTHORITY_NAME

    def __init__(self, *, metadata: dict, geometries: tuple[object, ...],
                 source_file_sha256: str):
        self._metadata = metadata
        self.scene_id = metadata["scene_id"]
        self.content_authority_sha256 = metadata["authority_sha256"]
        self.geometry_authority_sha256 = str(source_file_sha256)
        identities = tuple(metadata["obstacle_identities"])
        levels = []
        prepared_supported_centers = []
        if metadata.get("collision_query_mode") != \
                "nearest_obstacle_distance":
            raise ValueError("GS collision query mode is invalid")
        for row in metadata["levels"]:
            start = int(row["obstacle_geometry_start"])
            count = int(row["obstacle_geometry_count"])
            levels.append((
                float(row["ground_y_m"]),
                geometries[int(row["support_geometry_index"])],
                geometries[start:start + count],
            ))
            supported_centers = {}
            for radius_row in row.get("radius_support") or ():
                radius = float(radius_row["radius_m"])
                supported_centers[radius] = geometries[
                    int(radius_row["supported_geometry_index"])]
            prepared_supported_centers.append(supported_centers)
        self._engine = b1k_geometry.B1KGeometryAuthority.from_planar_levels(
            levels=levels, obstacle_identities=identities,
            authority_name=AUTHORITY_NAME,
            contact_surface_protocol=CONTACT_SURFACE_PROTOCOL,
            scene_authority_sha256=self.geometry_authority_sha256,
            collision_query_mode="distance",
            prepared_supported_centers=prepared_supported_centers)
        clusters = []
        for level in self._engine._levels:
            if (clusters and abs(
                    float(level.ground_y_m) -
                    float(clusters[-1][0].ground_y_m)) <=
                    float(config.GS_COLLISION_COPLANAR_FLOOR_M) + 1e-9):
                clusters[-1].append(level)
            else:
                clusters.append([level])
        self._ground_levels = tuple(
            max(cluster, key=lambda level: (
                float(shapely.area(level.support)),
                -float(level.ground_y_m),
            ))
            for cluster in clusters
        )

    def bind(self, position, yaw: float, *, radius_m: float):
        return self._engine.bind(position, yaw, radius_m=radius_m)

    def binding_atom(self) -> dict:
        return {
            "schema": ARTIFACT_SCHEMA,
            "scene_id": self.scene_id,
            "source_assets": list(self._metadata["source_assets"]),
            "protocol": CONTACT_SURFACE_PROTOCOL,
            "content_sha256": self.content_authority_sha256,
            "sha256": self.geometry_authority_sha256,
        }

    def ground_y(self, position) -> float | None:
        """Return the authenticated source floor supporting a navmesh pose.

        InteriorGS navmesh points sit above the collision floor.  The source
        USD can contain several millimetre-separated horizontal shells in the
        same stratum, so pure nearest-height selection can mistake a small
        furniture bottom for the room floor.  Collapse each near-coplanar
        stratum to its dominant source support before applying the normal nav
        Y gate; distinct storeys remain separate.
        """
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("GS collision ground query requires finite XYZ")
        candidates = [
            level for level in self._ground_levels
            if abs(float(point[1]) - float(level.ground_y_m)) <=
            float(config.NAV_Y_DELTA_M) + 1e-9
        ]
        if not candidates:
            return None
        level = min(candidates, key=lambda value: (
            abs(float(point[1]) - float(value.ground_y_m)),
            float(value.ground_y_m),
        ))
        return float(level.ground_y_m)


def load_collision_artifact(
        path, *, expected_file_sha256: str = None,
        expected_source_assets=None) -> GsCollisionAuthority:
    """Load and authenticate one preprocessed collision authority."""
    resolved = Path(path)
    payload = resolved.read_bytes()
    file_sha256 = hashlib.sha256(payload).hexdigest()
    if (expected_file_sha256 is not None and
            file_sha256 != expected_file_sha256):
        raise ValueError("GS collision artifact file digest is invalid")
    with np.load(resolved, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    try:
        metadata = json.loads(str(arrays.pop("metadata_json").item()))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GS collision artifact metadata is invalid") from error
    if metadata.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("GS collision artifact schema is invalid")
    authority_sha256 = metadata.pop("authority_sha256", None)
    rebuilt = _canonical_sha256(metadata, arrays)
    metadata["authority_sha256"] = authority_sha256
    if authority_sha256 != rebuilt:
        raise ValueError("GS collision authority digest is invalid")
    observed_assets = _normalized_source_assets(tuple(
        (row["role"], row["bytes"], row["sha256"])
        for row in metadata.get("source_assets") or ()))
    if expected_source_assets is not None and observed_assets != \
            _normalized_source_assets(expected_source_assets):
        raise ValueError("GS collision source assets do not match")
    geometries = _decode_geometries(arrays)
    return GsCollisionAuthority(
        metadata=metadata, geometries=geometries,
        source_file_sha256=file_sha256)
