"""Pure coordinate and collision geometry for BEHAVIOR-1K scenes.

This module consumes NumPy values extracted by the runtime adapter.  It must
remain importable without OmniGibson, Isaac Sim, or Habitat.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from shapely import STRtree, covers as shapely_covers, points as shapely_points
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from pipeline import config
from pipeline.geometry import GeometryQuery


B1K_CONTACT_SURFACE_PROTOCOL = "b1k-closest-collision-footprint.v1"
_BUFFER_QUAD_SEGS = 16
_LEVEL_MERGE_TOLERANCE_M = 1e-6
_DERIVED_SCENE_BINDING_TOKEN = object()


def _finite_xyz(values) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("B1K coordinates must be finite with final dimension 3")
    return points


def og_to_pbench_xyz(values) -> np.ndarray:
    """Map OmniGibson Z-up XYZ to PBench Y-up XYZ, preserving shape."""
    points = _finite_xyz(values)
    return np.stack(
        [points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def pbench_to_og_xyz(values) -> np.ndarray:
    """Map PBench Y-up XYZ to OmniGibson Z-up XYZ, preserving shape."""
    points = _finite_xyz(values)
    return np.stack(
        [points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def og_yaw_to_pbench_yaw_rad(yaw_rad: float) -> float:
    """Return the PBench yaw for an OmniGibson yaw about world up.

    Under the frozen axis mapping both conventions increase from forward
    toward world -X, so the numeric angle is unchanged.
    """
    yaw = float(yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("B1K yaw must be finite")
    return yaw


def canonical_triangles(values) -> np.ndarray:
    """Return finite, nondegenerate triangles in canonical source order.

    Vertices and faces are lexicographically sorted and duplicate faces are
    removed.  Winding is intentionally discarded: this authority uses closed
    collision support, not renderer normals.
    """
    triangles = np.asarray(values, dtype=np.float64)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or
            not np.isfinite(triangles).all()):
        raise ValueError("B1K triangles must be finite Fx3x3")
    if not len(triangles):
        return np.empty((0, 3, 3), dtype=np.float64)
    area_twice = np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]), axis=1)
    triangles = triangles[area_twice > 0.0]
    if not len(triangles):
        return np.empty((0, 3, 3), dtype=np.float64)
    canonical = []
    for triangle in triangles:
        order = np.lexsort((triangle[:, 2], triangle[:, 1], triangle[:, 0]))
        canonical.append(triangle[order])
    flat = np.stack(canonical).reshape(-1, 9)
    flat = np.unique(flat, axis=0)
    order = np.lexsort(tuple(flat[:, column]
                             for column in reversed(range(flat.shape[1]))))
    return np.ascontiguousarray(flat[order].reshape(-1, 3, 3))


@dataclass(frozen=True)
class TriangleComponent:
    """One real runtime collision component in PBench world coordinates."""

    identity: str
    triangles: np.ndarray

    def __post_init__(self) -> None:
        identity = str(self.identity).strip()
        if not identity:
            raise ValueError("B1K component identity must be nonempty")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "triangles", canonical_triangles(
            self.triangles))


def _normalized_components(components: Iterable[TriangleComponent]
                           ) -> tuple[TriangleComponent, ...]:
    parsed = []
    for value in components:
        if isinstance(value, TriangleComponent):
            component = value
        elif isinstance(value, Mapping):
            component = TriangleComponent(
                value.get("identity", value.get("prim_identity")),
                value.get("triangles"))
        else:
            identity, triangles = value
            component = TriangleComponent(identity, triangles)
        parsed.append(component)
    parsed.sort(key=lambda component: component.identity)
    identities = [component.identity for component in parsed]
    if len(set(identities)) != len(identities):
        raise ValueError("B1K component identities must be unique")
    return tuple(parsed)


def _triangle_projection(triangle: np.ndarray):
    points = triangle[:, [0, 2]]
    polygon = Polygon(points)
    if polygon.area > 0.0:
        return polygon
    unique = np.unique(points, axis=0)
    if len(unique) >= 2:
        return LineString(unique)
    if len(unique) == 1:
        return Point(unique[0])
    return GeometryCollection()


def _clip_polygon_y(points: Sequence[np.ndarray], boundary: float, *,
                    keep_above: bool) -> list[np.ndarray]:
    output = []
    for index, current in enumerate(points):
        previous = points[index - 1]
        current_inside = (
            current[1] >= boundary if keep_above else current[1] <= boundary)
        previous_inside = (
            previous[1] >= boundary if keep_above else previous[1] <= boundary)
        if current_inside != previous_inside:
            fraction = ((boundary - previous[1]) /
                        (current[1] - previous[1]))
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
    return output


def _clipped_triangle_projection(triangle: np.ndarray, low: float,
                                 high: float):
    polygon = _clip_polygon_y(
        [point for point in triangle], low, keep_above=True)
    if not polygon:
        return GeometryCollection()
    polygon = _clip_polygon_y(polygon, high, keep_above=False)
    if not polygon:
        return GeometryCollection()
    points = np.asarray(polygon, dtype=np.float64)[:, [0, 2]]
    unique = np.unique(points, axis=0)
    if len(unique) >= 3:
        shape = Polygon(points)
        if shape.area > 0.0:
            return shape if shape.is_valid else shape.buffer(0)
    if len(unique) >= 2:
        return LineString(unique)
    if len(unique) == 1:
        return Point(unique[0])
    return GeometryCollection()


def _ground_support(triangles: np.ndarray):
    shapes = [
        _triangle_projection(triangle)
        for triangle in triangles
    ]
    polygons = [shape for shape in shapes
                if not shape.is_empty and shape.area > 0.0]
    support = unary_union(polygons) if polygons else GeometryCollection()
    if support.is_empty or support.area <= 0.0:
        raise ValueError("B1K floor collision support is empty")
    return support


def _obstacle_footprint(component: TriangleComponent, floor_height_m: float):
    low, high = (
        float(floor_height_m) + float(value)
        for value in config.GROUND_OBSTACLE_BAND_M)
    triangles = component.triangles
    if len(triangles):
        heights = triangles[:, :, 1]
        triangles = triangles[
            (np.max(heights, axis=1) >= low) &
            (np.min(heights, axis=1) <= high)]
    shapes = [
        _clipped_triangle_projection(triangle, low, high)
        for triangle in triangles
    ]
    return unary_union([shape for shape in shapes if not shape.is_empty])


@dataclass(frozen=True)
class _GroundLevel:
    ground_y_m: float
    support: object
    obstacle_footprints: tuple
    obstacle_tree: STRtree | None
    obstacle_tree_component_indices: tuple[int, ...]
    obstacles: object
    supported_centers: dict
    configuration_obstacles: dict
    free_spaces: dict
    distance_conditioned: bool


def _floor_level_groups(components: Sequence[TriangleComponent]
                        ) -> list[tuple[float, np.ndarray]]:
    measured = []
    for component in components:
        for triangle_index, triangle in enumerate(component.triangles):
            heights = triangle[:, 1]
            if float(np.ptp(heights)) > _LEVEL_MERGE_TOLERANCE_M:
                continue
            measured.append((
                float(np.mean(heights)), component.identity,
                triangle_index, triangle))
    measured.sort(key=lambda value: value[:3])
    groups: list[list] = []
    for ground_y, _identity, _triangle_index, triangle in measured:
        if (groups and abs(ground_y - groups[-1][0]) <=
                _LEVEL_MERGE_TOLERANCE_M):
            groups[-1][1].append(triangle)
        else:
            groups.append([ground_y, [triangle]])
    if not groups:
        raise ValueError("B1K floor collision support is empty")
    return [
        (float(np.mean([
            float(np.mean(triangle[:, 1])) for triangle in triangles
        ])), np.stack(triangles))
        for _ground_y, triangles in groups
    ]


class B1KGeometryAuthority:
    """Floor-supported, radius-conditioned planar collision authority."""

    authority = "b1k_geometry"

    def __init__(self, *, floor_components, collision_components,
                 scene_authority_sha256: str | None = None,
                 _derived_binding_token=None, authority_name: str = None,
                 contact_surface_protocol: str = None):
        if (scene_authority_sha256 is not None and
                _derived_binding_token is not _DERIVED_SCENE_BINDING_TOKEN):
            raise ValueError(
                "bound B1K geometry must be built by build_b1k_authorities")
        self.authority = str(authority_name or self.authority)
        self.contact_surface_protocol = str(
            contact_surface_protocol or B1K_CONTACT_SURFACE_PROTOCOL)
        self._floor_components = _normalized_components(floor_components)
        self._collision_components = _normalized_components(
            collision_components)
        self.radii_m = tuple(float(value) for value in config.RADII_M)
        levels = []
        for ground_y, floor_triangles in _floor_level_groups(
                self._floor_components):
            support = _ground_support(floor_triangles)
            footprints = tuple(
                _obstacle_footprint(component, ground_y)
                for component in self._collision_components)
            indexed_footprints = tuple(
                (index, footprint)
                for index, footprint in enumerate(footprints)
                if not footprint.is_empty)
            nonempty = [
                footprint for _index, footprint in indexed_footprints]
            obstacle_tree = STRtree(nonempty) if nonempty else None
            obstacles = (
                unary_union(nonempty) if nonempty else GeometryCollection())
            supported_centers = {}
            configuration_obstacles = {}
            free_spaces = {}
            for radius in self.radii_m:
                supported = support.buffer(
                    -radius, quad_segs=_BUFFER_QUAD_SEGS)
                expanded = (
                    obstacles.buffer(radius, quad_segs=_BUFFER_QUAD_SEGS)
                    if not obstacles.is_empty else GeometryCollection())
                supported_centers[radius] = supported
                configuration_obstacles[radius] = expanded
                free_spaces[radius] = supported.difference(expanded)
            levels.append(_GroundLevel(
                ground_y_m=ground_y,
                support=support,
                obstacle_footprints=footprints,
                obstacle_tree=obstacle_tree,
                obstacle_tree_component_indices=tuple(
                    index for index, _footprint in indexed_footprints),
                obstacles=obstacles,
                supported_centers=supported_centers,
                configuration_obstacles=configuration_obstacles,
                free_spaces=free_spaces,
                distance_conditioned=False,
            ))
        self._levels = tuple(levels)
        self._scene_authority_sha256 = (
            None if scene_authority_sha256 is None
            else str(scene_authority_sha256))

    @classmethod
    def from_planar_levels(
            cls, *, levels, obstacle_identities, authority_name: str,
            contact_surface_protocol: str,
            scene_authority_sha256: str | None = None,
            collision_query_mode: str = "buffered",
            prepared_supported_centers=None):
        """Build the indexed engine from source-preprocessed 2-D geometry.

        GS uses exact nearest-footprint distance for disc collision instead of
        materializing enormous Minkowski buffers.  B1K retains its existing
        buffered configuration-space path.  ``prepared_supported_centers``
        moves the much smaller floor-boundary erosion to offline preprocessing.
        """
        value = cls.__new__(cls)
        value.authority = str(authority_name)
        value.contact_surface_protocol = str(contact_surface_protocol)
        value._floor_components = ()
        value._collision_components = _normalized_components(
            TriangleComponent(identity, np.empty((0, 3, 3)))
            for identity in obstacle_identities)
        value.radii_m = tuple(float(radius) for radius in config.RADII_M)
        source_levels = tuple(levels)
        mode = str(collision_query_mode)
        if mode not in {"buffered", "distance"}:
            raise ValueError("planar collision query mode is invalid")
        prepared_support = (
            None if prepared_supported_centers is None else
            tuple(prepared_supported_centers))
        if prepared_support is not None and \
                len(prepared_support) != len(source_levels):
            raise ValueError(
                "prepared planar support count is invalid")
        prepared = []
        for level_index, (ground_y, support, footprints) in enumerate(
                source_levels):
            footprints = tuple(footprints)
            if len(footprints) != len(value._collision_components):
                raise ValueError(
                    "planar level obstacle count disagrees with identities")
            indexed = tuple(
                (index, footprint)
                for index, footprint in enumerate(footprints)
                if not footprint.is_empty)
            nonempty = [footprint for _index, footprint in indexed]
            obstacle_tree = STRtree(nonempty) if nonempty else None
            if mode == "buffered":
                obstacles = (
                    unary_union(nonempty)
                    if nonempty else GeometryCollection())
                supported_centers = {}
                configuration_obstacles = {}
                free_spaces = {}
                for radius in value.radii_m:
                    supported = support.buffer(
                        -radius, quad_segs=_BUFFER_QUAD_SEGS)
                    expanded = (
                        obstacles.buffer(
                            radius, quad_segs=_BUFFER_QUAD_SEGS)
                        if not obstacles.is_empty else GeometryCollection())
                    supported_centers[radius] = supported
                    configuration_obstacles[radius] = expanded
                    free_spaces[radius] = supported.difference(expanded)
            else:
                obstacles = GeometryCollection()
                supported_centers = (
                    {float(radius): support.buffer(
                        -float(radius), quad_segs=_BUFFER_QUAD_SEGS)
                     for radius in value.radii_m}
                    if prepared_support is None else
                    dict(prepared_support[level_index]))
                expected_radii = set(value.radii_m)
                if set(supported_centers) != expected_radii:
                    raise ValueError(
                        "prepared planar support radii are invalid")
                configuration_obstacles = {}
                free_spaces = {}
            prepared.append(_GroundLevel(
                ground_y_m=float(ground_y),
                support=support,
                obstacle_footprints=footprints,
                obstacle_tree=obstacle_tree,
                obstacle_tree_component_indices=tuple(
                    index for index, _footprint in indexed),
                obstacles=obstacles,
                supported_centers=supported_centers,
                configuration_obstacles=configuration_obstacles,
                free_spaces=free_spaces,
                distance_conditioned=mode == "distance",
            ))
        if not prepared:
            raise ValueError("planar collision authority has no ground levels")
        value._levels = tuple(prepared)
        value._scene_authority_sha256 = (
            None if scene_authority_sha256 is None
            else str(scene_authority_sha256))
        return value

    @property
    def scene_authority_sha256(self) -> str | None:
        return self._scene_authority_sha256

    def _radius(self, radius_m: float) -> float:
        radius = float(radius_m)
        for frozen in self.radii_m:
            if radius == frozen:
                return frozen
        raise ValueError(
            f"B1K radius {radius!r} is not one of the frozen radii")

    def bind(self, position, yaw: float, *, radius_m: float):
        """Bind rollout-local SE(2) coordinates to this scene authority."""
        return B1KGeometryNav(
            self, position, yaw, radius_m=self._radius(radius_m))

    def sample_position(self, rng, *, radius_m: float,
                        minimum_clearance_m: float = 0.0,
                        max_tries: int = 10_000) -> np.ndarray:
        """Sample a PBench root from clearance-conditioned free space.

        ``minimum_clearance_m`` is the additional obstacle clearance after
        accounting for the body radius.  The radius-conditioned free space and
        obstacle index are immutable scene data.  Rejection against their
        exact nearest distance samples the same conditioned domain without
        rebuilding an expensive polygon buffer for every pose attempt.
        """
        radius = self._radius(radius_m)
        clearance = float(minimum_clearance_m)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "B1K minimum clearance must be finite and nonnegative")
        candidates = [(level, (
            level.supported_centers[radius]
            if level.distance_conditioned else level.free_spaces[radius]))
            for level in self._levels]
        candidates = [
            (level, free) for level, free in candidates
            if not free.is_empty and free.area > 0.0
        ]
        if not candidates:
            raise ValueError(
                f"B1K radius {radius} has empty clearance-conditioned free space")
        areas = np.asarray(
            [free.area for _level, free in candidates], dtype=np.float64)
        cumulative = np.cumsum(areas)
        for _ in range(int(max_tries)):
            draw = float(rng.uniform(0.0, float(areas.sum())))
            selected = min(
                int(np.searchsorted(cumulative, draw, side="right")),
                len(candidates) - 1)
            level, free = candidates[selected]
            min_x, min_z, max_x, max_z = free.bounds
            x = float(rng.uniform(min_x, max_x))
            z = float(rng.uniform(min_z, max_z))
            point = Point(x, z)
            if not free.covers(point):
                continue
            if level.distance_conditioned or clearance > 0.0:
                distance, obstacle_index = self._nearest_component(
                    level, point)
                if obstacle_index is not None:
                    if level.distance_conditioned and distance <= radius:
                        continue
                    if distance - radius < clearance - 1e-9:
                        continue
            return np.array(
                [x, level.ground_y_m, z], dtype=np.float64)
        raise RuntimeError("B1K free-space rejection sampler exhausted")

    def _select_level(self, world_y: float, max_y_delta: float
                      ) -> _GroundLevel | None:
        candidates = sorted(
            (abs(float(world_y) - level.ground_y_m), level.ground_y_m, level)
            for level in self._levels)
        if (not candidates or candidates[0][0] >
                float(max_y_delta) + 1e-9):
            return None
        return candidates[0][2]

    @staticmethod
    def _nearest_component(level: _GroundLevel, point: Point
                           ) -> tuple[float, int | None]:
        if level.obstacle_tree is None:
            return float(config.D_MAX_M), None
        tree_positions = np.asarray(
            level.obstacle_tree.query_nearest(point, all_matches=True),
            dtype=np.int64,
        ).reshape(-1)
        if not len(tree_positions):
            return float(config.D_MAX_M), None
        candidates = []
        for tree_position in tree_positions:
            component_index = level.obstacle_tree_component_indices[
                int(tree_position)]
            footprint = level.obstacle_footprints[component_index]
            candidates.append((
                float(point.distance(footprint)), component_index))
        return min(candidates)


class B1KGeometryNav:
    """Rollout-compatible nav view over one frozen B1K C-space."""

    authority = "b1k_geometry"

    def __init__(self, field: B1KGeometryAuthority, position, yaw: float, *,
                 radius_m: float):
        self._field = field
        self.authority = field.authority
        self.geometry_authority_sha256 = field.scene_authority_sha256
        self._position = np.asarray(position, dtype=np.float64)
        self._yaw = float(yaw)
        if (self._position.shape != (3,) or
                not np.isfinite(self._position).all() or
                not math.isfinite(self._yaw)):
            raise ValueError("B1K nav pose must be finite PBench XYZ plus yaw")
        self.radius_m = float(radius_m)
        self._level = field._select_level(
            self._position[1], config.NAV_Y_DELTA_M)

    def _world(self, pose) -> np.ndarray:
        x = float(pose[0])
        z = float(pose[1])
        cosine, sine = math.cos(self._yaw), math.sin(self._yaw)
        return np.array([
            self._position[0] + x * cosine - z * sine,
            self._position[1],
            self._position[2] - x * sine - z * cosine,
        ], dtype=np.float64)

    def query_pose(self, pose, max_y_delta=config.NAV_Y_DELTA_M
                   ) -> GeometryQuery:
        world = self._world(pose)
        if (self._level is None or
                abs(float(world[1]) - self._level.ground_y_m) >
                float(max_y_delta) + 1e-9):
            return GeometryQuery(False, 0.0, None, "unsupported_floor")
        point = Point(float(world[0]), float(world[2]))
        supported = self._level.supported_centers[self.radius_m]
        if not supported.covers(point):
            return GeometryQuery(False, 0.0, None, "unsupported_floor")
        distance, index = self._field._nearest_component(self._level, point)
        clearance = distance - self.radius_m if index is not None else distance
        collision = (
            index is not None and distance <= self.radius_m
            if self._level.distance_conditioned else
            self._level.configuration_obstacles[
                self.radius_m].covers(point))
        return GeometryQuery(
            navigable=not collision,
            clearance_m=float(clearance),
            obstacle_index=index,
            geometry_source=self.authority if collision else None,
        )

    def query_many(self, poses, max_y_delta=config.NAV_Y_DELTA_M
                   ) -> list[GeometryQuery]:
        values = list(poses)
        if not values:
            return []
        if (self._level is None or
                abs(float(self._position[1]) - self._level.ground_y_m) >
                float(max_y_delta) + 1e-9):
            return [GeometryQuery(
                False, 0.0, None, "unsupported_floor") for _ in values]
        local = np.asarray(values, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] < 2 or \
                not np.isfinite(local[:, :2]).all():
            raise ValueError("B1K query poses must contain finite x/z")
        cosine, sine = math.cos(self._yaw), math.sin(self._yaw)
        world_x = self._position[0] + \
            local[:, 0] * cosine - local[:, 1] * sine
        world_z = self._position[2] - \
            local[:, 0] * sine - local[:, 1] * cosine
        query_points = shapely_points(np.column_stack([world_x, world_z]))
        supported = np.asarray(shapely_covers(
            self._level.supported_centers[self.radius_m], query_points),
            dtype=bool)
        collision = (
            None if self._level.distance_conditioned else
            np.asarray(shapely_covers(
                self._level.configuration_obstacles[self.radius_m],
                query_points), dtype=bool))
        nearest = {}
        supported_indices = np.flatnonzero(supported)
        tree = self._level.obstacle_tree
        if tree is not None and len(supported_indices):
            tree_pairs, distances = tree.query_nearest(
                query_points[supported_indices], all_matches=True,
                return_distance=True)
            for local_index, tree_position, distance in zip(
                    tree_pairs[0], tree_pairs[1], distances):
                output_index = int(supported_indices[int(local_index)])
                component_index = int(
                    self._level.obstacle_tree_component_indices[
                        int(tree_position)])
                candidate = (float(distance), component_index)
                if candidate < nearest.get(
                        output_index, (float("inf"), component_index)):
                    nearest[output_index] = candidate
        results = []
        for index in range(len(values)):
            if not supported[index]:
                results.append(GeometryQuery(
                    False, 0.0, None, "unsupported_floor"))
                continue
            distance, component_index = nearest.get(
                index, (float(config.D_MAX_M), None))
            clearance = (
                distance - self.radius_m
                if component_index is not None else distance)
            collides = (
                component_index is not None and distance <= self.radius_m
                if self._level.distance_conditioned else
                bool(collision[index]))
            results.append(GeometryQuery(
                navigable=not collides,
                clearance_m=float(clearance),
                obstacle_index=component_index,
                geometry_source=(
                    self.authority if collides else None),
            ))
        return results

    def is_navigable(self, pose,
                     max_y_delta=config.NAV_Y_DELTA_M) -> bool:
        return self.query_pose(
            pose, max_y_delta=max_y_delta).navigable

    def clearance(self, pose) -> float:
        return max(0.0, float(self.query_pose(pose).clearance_m))

    def closest_obstacle(self, pose) -> dict:
        if self._level is None:
            return {}
        world = self._world(pose)
        point = Point(float(world[0]), float(world[2]))
        distance, index = self._field._nearest_component(self._level, point)
        if index is None:
            return {}
        footprint = self._level.obstacle_footprints[index]
        target = footprint.boundary if footprint.covers(point) else footprint
        _query, surface = nearest_points(point, target)
        delta = np.array([
            float(point.x - surface.x), float(point.y - surface.y)],
            dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        unit = delta / norm if norm > 1e-12 else np.array([0.0, 1.0])
        if self._level.distance_conditioned:
            configuration_point = Point(
                float(surface.x) + float(unit[0]) * self.radius_m,
                float(surface.y) + float(unit[1]) * self.radius_m)
        else:
            expanded = self._level.configuration_obstacles[self.radius_m]
            boundary = expanded.boundary
            _query, configuration_point = nearest_points(point, boundary)
        component = self._field._collision_components[index]
        return {
            "world_point": [
                float(surface.x), float(
                    self._level.ground_y_m +
                    config.CONTACT_ATTRIBUTION_HEIGHT_M), float(surface.y)],
            "world_normal": [float(unit[0]), 0.0, float(unit[1])],
            "distance_m": float(distance),
            "configuration_boundary_world_point": [
                float(configuration_point.x), self._level.ground_y_m,
                float(configuration_point.y)],
            "configuration_boundary_distance_m": float(
                point.distance(configuration_point)),
            "surface_protocol": self._field.contact_surface_protocol,
            "obstacle_index": int(index),
            "obstacle_identity": component.identity,
        }

    def rebase(self, pose):
        return B1KGeometryNav(
            self._field, self._world(pose),
            self._yaw - math.radians(float(pose[2])),
            radius_m=self.radius_m)
