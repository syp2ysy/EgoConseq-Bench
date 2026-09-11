"""Structured future-consequence rollout tests without Habitat."""

import copy
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import (
    config,
    consequence,
    outcome as outcome_fields,
    perception,
    rollout,
)
from pipeline.actions import Forward, Turn, pose_at_arc
from pipeline.geometry import Disc
from pipeline.geometry import GeometryQuery
from pipeline.floor_plane import FloorPlaneEstimate
from pipeline import sim as sim_module
from pipeline.sim import Nav
from tests._synthetic import LEVEL_FLOOR, make_frame, make_overhang_frame


class _SurfaceIndex:
    id_to_cat = {7: "chair"}

    def instance_points(self, instance_id):
        if instance_id != 7:
            return np.empty((0, 3), float)
        # Local (0, 0.15, 2) lies in the target ground-support band.
        return np.repeat(
            np.array([[0.0, 0.15, -2.0]]),
            5,
            axis=0,
        )


class HalfPlaneNav:
    """Navigable while global z <= limit, with a configurable local base."""

    authority = "navmesh"

    def __init__(self, limit=0.8, base=(0.0, 0.0, 0.0)):
        self.limit = float(limit)
        self.base = tuple(float(v) for v in base)

    def _global(self, pose):
        x, z, heading = pose
        bx, bz, bh = self.base
        h = math.radians(bh)
        return (bx + x * math.cos(h) + z * math.sin(h),
                bz - x * math.sin(h) + z * math.cos(h),
                bh + heading)

    def is_navigable(self, pose):
        return self._global(pose)[1] <= self.limit + 1e-9

    def clearance(self, pose):
        return max(0.0, self.limit - self._global(pose)[1])

    def closest_obstacle(self, pose):
        gx, _, _ = self._global(pose)
        return {"world_point": [gx, 0.0, self.limit],
                "world_normal": [0.0, 0.0, -1.0],
                "distance_m": self.clearance(pose)}

    def rebase(self, pose):
        return HalfPlaneNav(self.limit, self._global(pose))


class _BatchCountingNav:
    authority = "batch-test"

    def __init__(self):
        self.batch_calls = 0
        self.scalar_calls = 0

    def query_many(self, poses, max_y_delta=0.5):
        del max_y_delta
        self.batch_calls += 1
        return [
            SimpleNamespace(
                navigable=True,
                clearance_m=0.4,
                obstacle_index=None,
            )
            for _pose in poses
        ]

    def query_pose(self, pose, max_y_delta=0.5):
        del pose, max_y_delta
        self.scalar_calls += 1
        return SimpleNamespace(
            navigable=True,
            clearance_m=0.4,
            obstacle_index=None,
        )

    def is_navigable(self, _pose):
        raise AssertionError("rollout must use the unified query API")

    def clearance(self, _pose):
        raise AssertionError("rollout must use the unified query API")


def test_rollout_retains_the_actual_collision_component():
    class ComponentNav(HalfPlaneNav):
        def closest_obstacle(self, pose):
            return {**super().closest_obstacle(pose),
                    "obstacle_identity": "/World/chair/base_link/collision"}

    result = rollout.physical_rollout(ComponentNav(), [Forward(2.0)])
    assert result["physical"]["contact"].get("obstacle_identity") == \
        "/World/chair/base_link/collision"


class _ExcludedGeometryNav:
    authority = "gs_collision_mesh"
    geometry_authority_sha256 = "b" * 64

    def __init__(self, source):
        self.source = source

    def query_many(self, poses, max_y_delta=0.5):
        del max_y_delta
        return [
            GeometryQuery(False, 0.0, None, self.source)
            for _pose in poses
        ]

    def query_pose(self, pose, max_y_delta=0.5):
        del pose, max_y_delta
        return GeometryQuery(False, 0.0, None, self.source)


class _MixedBatchNav:
    authority = "b1k_geometry"

    def __init__(self):
        self.batch_calls = 0
        self.scalar_calls = 0

    @staticmethod
    def _query(pose):
        x, z = float(pose[0]), float(pose[1])
        if x > 0.25:
            return GeometryQuery(False, 0.0, None, "unsupported_floor")
        navigable = z <= 0.75
        return GeometryQuery(
            navigable, max(0.0, 0.75 - z), None,
            None if navigable else "b1k_geometry")

    def query_many(self, poses, max_y_delta=0.5):
        del max_y_delta
        self.batch_calls += 1
        return [self._query(pose) for pose in poses]

    def query_pose(self, pose, max_y_delta=0.5):
        del max_y_delta
        self.scalar_calls += 1
        return self._query(pose)


class _Hit:
    hit_pos = np.array([1.0, 0.0, -3.0])
    hit_normal = np.array([0.0, 0.0, 1.0])
    hit_dist = 0.4


class _PathFinder:
    def is_navigable(self, point, max_y_delta=0.5):
        return float(point[2]) >= -3.0

    def distance_to_closest_obstacle(self, point, max_search_radius=2.0):
        return 0.4

    def closest_obstacle_surface_point(self, point, max_search_radius=2.0):
        return _Hit()


def test_habitat_nav_adapter_rebases_local_pose():
    nav = Nav(_PathFinder(), np.array([1.0, 0.0, -2.0]), 0.0,
              radius_m=0.25, authority="navmesh")
    assert nav.is_navigable((0.0, 0.5, 0.0)) is True
    assert nav.clearance((0.0, 0.5, 0.0)) == pytest.approx(0.4)
    query = nav.query_pose((0.0, 0.5, 0.0))
    assert query.navigable is True
    assert query.clearance_m == pytest.approx(0.4)
    assert [
        value.navigable
        for value in nav.query_many([(0.0, 0.5, 0.0), (0.0, 1.5, 0.0)])
    ] == [True, False]
    contact = nav.closest_obstacle((0.0, 1.01, 0.0))
    center_world = nav._world(0.0, 1.01)
    boundary_world = np.asarray(
        contact["configuration_boundary_world_point"])[[0, 2]]
    crossed = float(np.linalg.norm(boundary_world - center_world[[0, 2]]))
    assert np.linalg.norm(
        np.asarray(contact["world_point"])[[0, 2]] -
        center_world[[0, 2]]) == pytest.approx(0.25 - crossed)
    assert contact["configuration_boundary_world_point"] == [
        1.0, 0.0, -3.0]
    assert contact["surface_protocol"] == \
        "radius_extrapolated_navmesh_boundary_v2"

    endpoint = nav.rebase((0.0, 0.5, 90.0))
    # After turning right, local forward points toward world +X.
    world = endpoint._world(0.0, 1.0)
    assert world[[0, 2]] == pytest.approx((2.0, -2.5))


def test_habitat_nav_contact_surface_points_away_from_configuration_boundary():
    nav = Nav(_PathFinder(), np.array([1.0, 0.0, -2.0]), 0.0,
              radius_m=0.25, authority="navmesh")
    contact_pose = (0.0, 1.01, 0.0)
    contact = nav.closest_obstacle(contact_pose)
    center = nav._world(contact_pose[0], contact_pose[1])[[0, 2]]
    boundary = np.asarray(
        contact["configuration_boundary_world_point"])[[0, 2]]
    surface = np.asarray(contact["world_point"])[[0, 2]]
    boundary_direction = boundary - center
    surface_direction = surface - center

    crossed = float(np.linalg.norm(boundary_direction))
    assert surface_direction == pytest.approx(
        -boundary_direction / crossed * (0.25 - crossed))
    assert np.dot(surface_direction, boundary_direction) < 0.0


def _visible_frame(visible):
    fr = make_frame()
    sem = np.zeros(len(fr.pts_sem), np.int64)
    uv = np.zeros((len(sem), 2), np.int64)
    for i in range(len(sem)):
        uv[i] = [i % 20, i // 20]
    if visible:
        sem[:30] = 7
    return dataclasses.replace(fr, pts_sem=sem, pts_uv=uv,
                               semantic_index=_SurfaceIndex())


def test_navmesh_is_physical_authority_and_collision_stops_checkpoints():
    fr = dataclasses.replace(make_frame(), semantic_index=_SurfaceIndex())
    out = consequence.judge(fr, Disc(0.25), [Forward(2.0)],
                            nav=HalfPlaneNav(0.8))

    assert out["physical"]["authority"] == "navmesh"
    assert out["physical"]["collision"] is True
    assert out["physical"]["first_contact_arc_m"] == pytest.approx(0.8, abs=0.01)
    assert out["execution"]["executed_forward_fraction"] == pytest.approx(0.4, abs=0.01)
    assert [c["requested_progress"] for c in out["checkpoints"]] == [0, .25, .5, .75, 1]
    assert out["checkpoints"][-1]["pose"]["z"] == pytest.approx(0.8, abs=0.01)
    assert out["checkpoints"][-1]["realized_progress"] == pytest.approx(0.4, abs=0.01)


def test_physical_precheck_batches_the_coarse_path_without_double_queries():
    nav = _BatchCountingNav()
    actions = [Forward(1.0), Turn(30.0), Forward(0.5)]

    result = rollout.physical_collision_precheck(nav, actions)

    assert result["collision"] is False
    assert result["minimum_clearance_m"] == pytest.approx(0.4)
    assert nav.batch_calls == 1
    assert nav.scalar_calls == 0


@pytest.mark.parametrize(
    "nav_factory",
    [lambda: HalfPlaneNav(10.0), lambda: HalfPlaneNav(0.8),
     lambda: _ExcludedGeometryNav("unsupported_floor")],
)
def test_physical_path_trace_preserves_safe_collision_and_excluded_rollouts(
        nav_factory):
    actions = [Turn(15.0), Forward(2.0)]
    nav = nav_factory()

    path_trace = rollout.physical_path_trace(nav, actions)
    precheck = rollout.physical_collision_precheck(
        nav, actions, path_trace=path_trace)
    cached = rollout.physical_rollout(
        nav, actions, path_trace=path_trace)
    fresh = rollout.physical_rollout(nav_factory(), actions)

    assert precheck == rollout.physical_collision_precheck(
        nav_factory(), actions)
    assert cached == fresh


def test_physical_path_traces_batch_actions_without_changing_results():
    programs = [
        [Forward(0.5)],
        [Forward(1.0)],
        [Turn(90.0), Forward(0.5)],
    ]
    expected = [
        rollout.physical_path_trace(_MixedBatchNav(), actions)
        for actions in programs
    ]
    nav = _MixedBatchNav()

    observed = rollout.physical_path_traces(nav, programs)

    assert observed == expected
    assert [trace.collision for trace in observed] == [False, True, None]
    assert nav.batch_calls == 1


def test_cached_safe_path_materializes_only_checkpoint_queries():
    nav = _BatchCountingNav()
    actions = [Forward(1.0), Turn(30.0), Forward(0.5)]

    path_trace = rollout.physical_path_trace(nav, actions)
    assert (nav.batch_calls, nav.scalar_calls) == (1, 0)

    rollout.physical_rollout(nav, actions, path_trace=path_trace)

    assert (nav.batch_calls, nav.scalar_calls) == (2, 0)


def test_judge_uses_execution_endpoint_without_continuation_oracle():
    assert not hasattr(consequence.rollout, "terminal_options")
    out = consequence.judge(
        make_frame(), Disc(0.25), [Forward(0.5)],
        nav=HalfPlaneNav(1.2),
    )

    assert out["execution"]["realized_pose"] == {
        "x": pytest.approx(0.0),
        "z": pytest.approx(0.5),
        "heading_deg": pytest.approx(0.0),
    }
    assert not {
        "future_state", "terminal_options", "start_terminal_options",
        "object_consequences",
    } & set(out)
    assert set(out["evidence"]) == {"physical"}


def test_collision_stops_at_last_navigable_arc_before_first_contact():
    actions = [Forward(2.0)]
    nav = HalfPlaneNav(0.803)

    out = rollout.physical_rollout(nav, actions)
    execution = out["execution"]
    physical = out["physical"]
    stop_arc = execution["stop_arc_m"]
    contact_arc = physical["first_contact_arc_m"]
    stop_pose = (
        execution["realized_pose"]["x"],
        execution["realized_pose"]["z"],
        execution["realized_pose"]["heading_deg"],
    )
    contact_pose = pose_at_arc(actions, contact_arc)

    assert stop_arc < contact_arc
    assert contact_arc - stop_arc <= (
        config.MARCH_STEP_M / (2 ** config.CONTACT_REFINE_ITERS) + 1e-12)
    assert nav.is_navigable(stop_pose) is True
    assert nav.is_navigable(contact_pose) is False
    assert execution["executed_forward_m"] == pytest.approx(stop_arc)
    assert execution["executed_forward_fraction"] == pytest.approx(stop_arc / 2.0)
    assert execution["animation_stop_time_fraction"] == pytest.approx(
        stop_arc / 2.0)
    assert out["checkpoints"][-1]["arc_m"] == pytest.approx(stop_arc)
    assert out["checkpoints"][-1]["pose"] == execution["realized_pose"]
    assert physical["contact_action_local_arc_m"] == pytest.approx(contact_arc)
    assert physical["contact"]["center_local"] == pytest.approx(
        contact_pose[:2])


def test_summarize_execution_safe_counts_trailing_turn():
    # Regression: a Turn issued after the final Forward must count toward the
    # executed net turn. The pre-fix rollout stopped accumulating once the
    # forward arc budget hit zero, dropping the trailing 45-degree turn.
    actions = [Turn(30.0), Forward(1.0), Turn(45.0)]
    summary = rollout.summarize_execution(actions, collision=False, stop_arc_m=None)
    assert summary["executed_turn_deg"] == pytest.approx(75.0)
    assert summary["executed_forward_after_turn_m"] == pytest.approx(1.0)
    # Full-execution pose includes the trailing turn (heading = 30 + 45).
    assert summary["realized_pose"][2] == pytest.approx(75.0)


def test_physical_rollout_safe_reports_full_net_turn():
    actions = [Turn(30.0), Forward(1.0), Turn(45.0)]
    execution = rollout.physical_rollout(HalfPlaneNav(10.0), actions)["execution"]
    assert execution["completed"] is True
    assert execution["executed_turn_deg"] == pytest.approx(75.0)
    assert execution["realized_pose"]["heading_deg"] == pytest.approx(75.0)


def test_summarize_execution_collision_drops_post_stop_turn():
    # A forward-leg collision stops inside the forward; any Turn after that leg
    # is never executed and must not be counted.
    actions = [Turn(30.0), Forward(2.0), Turn(45.0)]
    summary = rollout.summarize_execution(actions, collision=True, stop_arc_m=0.5)
    assert summary["executed_turn_deg"] == pytest.approx(30.0)
    assert summary["executed_forward_after_turn_m"] == pytest.approx(0.5)
    # Heading reflects only the pre-collision turn.
    assert summary["realized_pose"][2] == pytest.approx(30.0)
    assert summary["realized_pose"] == pytest.approx(pose_at_arc(actions, 0.5))


def test_summarize_execution_stop_at_forward_boundary_excludes_next_turn():
    # stop_arc exactly at a forward-leg boundary: the leg is complete, execution
    # stops, and the following Turn is not counted (no over-run, no double-count).
    actions = [Turn(30.0), Forward(1.0), Turn(45.0), Forward(2.0)]
    summary = rollout.summarize_execution(actions, collision=True, stop_arc_m=1.0)
    assert summary["executed_turn_deg"] == pytest.approx(30.0)
    assert summary["executed_forward_after_turn_m"] == pytest.approx(1.0)
    assert summary["realized_pose"][2] == pytest.approx(30.0)


def test_summarize_execution_multi_leg_collision_counts_intermediate_turn():
    # Collision on a later forward: turns before the stopping leg all count.
    actions = [Turn(30.0), Forward(1.0), Turn(45.0), Forward(2.0), Turn(15.0)]
    summary = rollout.summarize_execution(actions, collision=True, stop_arc_m=1.5)
    assert summary["executed_turn_deg"] == pytest.approx(75.0)
    assert summary["executed_forward_after_turn_m"] == pytest.approx(1.5)


@pytest.mark.parametrize("wall_z", [0.8, 10.0])
def test_physical_precheck_matches_full_collision_and_contact_arc(wall_z):
    actions = [Turn(15), Forward(2.0)]
    nav = HalfPlaneNav(wall_z)
    precheck = rollout.physical_collision_precheck(nav, actions)
    full = rollout.physical_rollout(nav, actions)["physical"]
    assert precheck["authority"] == full["authority"]
    assert precheck["collision"] is full["collision"]
    assert precheck["minimum_clearance_m"] == pytest.approx(
        full["minimum_clearance_m"])
    if full["first_contact_arc_m"] is None:
        assert precheck["first_contact_arc_m"] is None
    else:
        assert precheck["first_contact_arc_m"] == pytest.approx(
            full["first_contact_arc_m"])


def test_physical_rollout_accepts_gs_collision_authority():
    nav = HalfPlaneNav(0.4)
    nav.authority = "gs_collision_mesh"
    nav.geometry_authority_sha256 = "a" * 64
    out = consequence.judge(make_frame(), Disc(0.25), [Forward(1.0)], nav=nav)
    assert out["physical"]["authority"] == "gs_collision_mesh"
    assert out["physical"]["collision"] is True
    assert out["physical"]["geometry_authority_sha256"] == "a" * 64


def test_physical_rollout_rejects_unbound_gs_collision_authority():
    nav = HalfPlaneNav(0.4)
    nav.authority = "gs_collision_mesh"

    with pytest.raises(ValueError, match="geometry authority binding"):
        rollout.physical_rollout(nav, [Forward(1.0)])


@pytest.mark.parametrize(
    "source", ["navmesh_boundary", "unsupported_floor"])
def test_gs_support_failures_are_invalid_geometry_not_collisions(source):
    nav = _ExcludedGeometryNav(source)
    actions = [Forward(1.0)]

    precheck = rollout.physical_collision_precheck(nav, actions)
    full = rollout.physical_rollout(nav, actions)

    assert precheck == {
        "authority": "gs_collision_mesh",
        "geometry_authority_sha256": "b" * 64,
        "collision": None,
        "collision_source": source,
        "first_contact_arc_m": None,
        "minimum_clearance_m": None,
    }
    assert full["physical"]["collision"] is None
    assert full["physical"]["collision_source"] == source
    assert outcome_fields.derive_execution_regime(full) == "invalid_geometry"


def test_visible_depth_collision_does_not_override_safe_navmesh():
    fr = dataclasses.replace(make_frame(), semantic_index=_SurfaceIndex())
    out = consequence.judge(fr, Disc(0.25), [Forward(2.0)],
                            nav=HalfPlaneNav(10.0))
    assert out["physical"]["collision"] is False
    assert out["evidence"]["physical"]["view_collision_estimate"] is True


def test_judge_reuses_cached_depth_coverage_and_consensus():
    frame = dataclasses.replace(make_frame(), semantic_index=_SurfaceIndex())
    actions = [Forward(0.2)]
    physical = rollout.physical_rollout(HalfPlaneNav(10.0), actions)
    cached_depth = {
        "authority": "depth", "collision": False,
        "first_contact_arc_m": None, "realized_pose": {
            "x": 0.0, "z": 0.2, "heading_deg": 0.0},
    }
    cached_consensus = {
        "accepted": True, "reason": "accepted", "verdict": "agree_safe",
        "corridor_coverage": 0.93,
    }

    out = consequence.judge(
        frame, Disc(0.25), actions, nav=HalfPlaneNav(10.0),
        cached_physical=physical,
        cached_depth_physical=cached_depth, cached_corridor_coverage=0.93,
        cached_oracle_consensus=cached_consensus)

    assert out["depth_physical"] == {**cached_depth, "contact": None}
    assert out["oracle_consensus"] == cached_consensus
    assert out["evidence"]["physical"]["coverage"] == pytest.approx(0.93)


def test_contact_semantics_keep_depth_and_full_attribution_independent():
    class FloorHitSurfaceIndex:
        id_to_cat = {5: "floor", 7: "chair"}

        def assign(self, _points):
            return np.array([5])

        def instance_points(self, instance_id):
            if instance_id == 7:
                return np.array([[0.0, 0.5, -1.0]])
            return np.empty((0, 3))

    frame = make_frame()
    frame = dataclasses.replace(
        frame, pts_sem=np.full_like(frame.pts_sem, 7),
        id_to_cat={5: "floor", 7: "chair"},
        semantic_index=FloorHitSurfaceIndex())
    physical = {"contact": {
        "center_local": [0.0, 0.8],
        "world_point": [0.0, 0.0, -0.8],
    }}
    consequence._attribute_full_contact(frame, physical)
    assert physical["contact"]["full_geometry_attribution"] == {
        "instance_id": None, "category": None, "unattributed": True}


def test_world_and_pose_local_coordinates_are_exact_inverses():
    local = np.array([
        [-0.7, 0.13, 2.4],
        [0.0, -0.2, 0.0],
        [3.2, 1.1, -0.4],
    ], dtype=np.float64)
    position = (4.5, -1.2, 8.0)
    yaw_rad = math.radians(37.0)

    world = perception.world_from_local(local, position, yaw_rad)
    recovered = perception.local_from_world(world, position, yaw_rad)

    np.testing.assert_allclose(recovered, local, rtol=0.0, atol=1e-12)


def test_full_contact_probes_follow_the_canonical_tilted_floor():
    class CapturingSurfaceIndex:
        def __init__(self):
            self.points = None

        def assign(self, points):
            self.points = np.asarray(points, dtype=np.float64)
            return np.full(len(self.points), 7, dtype=np.int64)

    normal = np.array([0.08, 0.995, -0.06], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    plane = FloorPlaneEstimate(tuple(normal), offset_m=-0.04)
    position = (3.0, 1.2, -4.0)
    yaw_rad = math.radians(-31.0)
    contact_xz = (0.65, 1.35)
    contact_local = np.array([[
        contact_xz[0],
        plane.y_at(*contact_xz),
        contact_xz[1],
    ]])
    world_point = perception.world_from_local(
        contact_local, position, yaw_rad)[0]
    semantic_index = CapturingSurfaceIndex()
    frame = dataclasses.replace(
        make_frame(),
        position=position,
        yaw_rad=yaw_rad,
        floor_plane=plane,
        semantic_index=semantic_index,
        id_to_cat={7: "chair"},
    )
    # This stored field is intentionally inconsistent. Full attribution must
    # derive its pose-local coordinates from the trusted world contact point.
    physical = {"contact": {
        "center_local": [-9.0, -8.0],
        "world_point": world_point.tolist(),
    }}

    consequence._attribute_full_contact(frame, physical)

    probes_local = perception.local_from_world(
        semantic_index.points, position, yaw_rad)
    expected_heights = np.linspace(
        config.GROUND_OBSTACLE_BAND_M[0],
        config.GROUND_OBSTACLE_BAND_M[1],
        config.CONTACT_ATTRIBUTION_PROBE_COUNT,
    )
    np.testing.assert_allclose(
        probes_local[:, 0], contact_xz[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        probes_local[:, 2], contact_xz[1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        plane.height_above_points(probes_local),
        expected_heights,
        rtol=0.0,
        atol=1e-12,
    )
    assert physical["contact"]["instance_id"] == 7


def test_depth_contact_projection_uses_the_attribution_band_midpoint():
    normal = np.array([0.06, 0.996, 0.07], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    plane = FloorPlaneEstimate(tuple(normal), offset_m=0.03)
    center = [0.4, 1.1]
    frame = dataclasses.replace(make_frame(), floor_plane=plane)
    estimate = {"center_local": center}

    consequence._attribute_depth_contact(frame, Disc(0.25), estimate)

    # The representative point sits at the band midpoint measured along the
    # plane normal — the same signed-normal convention the full-geometry
    # probes use — not at a vertical offset above the floor.
    contact_point = np.asarray([estimate["contact"]["point_3d"]])
    assert plane.height_above_points(contact_point)[0] == pytest.approx(
        config.CONTACT_ATTRIBUTION_HEIGHT_M)


def test_contact_xy_is_the_disc_center_for_both_attribution_paths():
    class SurfaceIndex:
        def assign(self, points):
            return np.full(len(np.asarray(points)), 7, dtype=np.int64)

    plane = LEVEL_FLOOR
    position = (2.0, 1.0, -3.0)
    yaw_rad = math.radians(20.0)
    disc_center_local = [0.4, 1.1]
    obstacle_local = np.array([[0.62, 0.05, 1.28]])
    obstacle_world = perception.world_from_local(
        obstacle_local, position, yaw_rad)[0]
    frame = dataclasses.replace(
        make_frame(),
        position=position,
        yaw_rad=yaw_rad,
        floor_plane=plane,
        semantic_index=SurfaceIndex(),
        id_to_cat={7: "chair"},
    )
    physical = {"contact": {
        "center_local": list(disc_center_local),
        "world_point": obstacle_world.tolist(),
    }}

    consequence._attribute_full_contact(frame, physical)
    estimate = {"center_local": list(disc_center_local)}
    consequence._attribute_depth_contact(frame, Disc(0.25), estimate)

    # Both oracles report the same contact anchor: the disc centre at the
    # moment of contact. The obstacle surface point stays in point_3d.
    assert physical["contact"]["xy"] == pytest.approx(disc_center_local)
    assert estimate["contact"]["xy"] == pytest.approx(disc_center_local)
    assert physical["contact"]["point_3d"] == pytest.approx(
        obstacle_world.tolist())


def test_full_contact_attribution_uses_the_radius_extrapolated_surface():
    class DirectionalSurfaceIndex:
        def __init__(self, expected_xz):
            self.expected_xz = np.asarray(expected_xz, dtype=np.float64)
            self.points = None

        def assign(self, points):
            self.points = np.asarray(points, dtype=np.float64)
            local = perception.local_from_world(
                self.points, frame.position, frame.yaw_rad)
            assert local[:, [0, 2]] == pytest.approx(
                np.repeat(self.expected_xz[None, :], len(local), axis=0))
            return np.full(len(local), 7, dtype=np.int64)

    body = Disc(0.25)
    center = np.array([0.4, 1.1], dtype=np.float64)
    direction = np.array([0.6, 0.8], dtype=np.float64)
    surface_xz = center + body.radius_m * direction
    frame = dataclasses.replace(
        make_frame(),
        semantic_index=DirectionalSurfaceIndex(surface_xz),
        id_to_cat={7: "chair"},
    )
    physical = {"contact": {
        "center_local": center.tolist(),
        "world_point": perception.world_from_local(
            np.array([[surface_xz[0], 0.0, surface_xz[1]]]),
            frame.position, frame.yaw_rad)[0].tolist(),
        "surface_protocol": "radius_extrapolated_navmesh_boundary_v2",
    }}

    consequence._attribute_full_contact(frame, physical)

    assert physical["contact"]["instance_id"] == 7
    assert np.linalg.norm(
        surface_xz - center) == pytest.approx(body.radius_m)


def test_full_contact_without_world_point_is_explicitly_unattributed():
    class SurfaceIndex:
        def assign(self, points):
            return np.full(len(np.asarray(points)), 7, dtype=np.int64)

    frame = dataclasses.replace(
        make_frame(),
        semantic_index=SurfaceIndex(),
        id_to_cat={7: "chair"},
    )
    physical = {"contact": {
        "center_local": [0.4, 1.1],
        "world_point": None,
    }}

    consequence._attribute_full_contact(frame, physical)

    contact = physical["contact"]
    assert contact["unattributed"] is True
    assert contact["instance_id"] is None
    assert contact["category"] is None
    assert contact["full_geometry_attribution"] == {
        "instance_id": None, "category": None, "unattributed": True}
    assert contact["xy"] == pytest.approx([0.4, 1.1])
    assert contact["point_3d"] is None


def test_near_field_wedge_matches_the_v5_formula_straight_ahead():
    """Straight ahead the orientation-aware wedge reduces to radius/tan(hfov/2)."""
    half_hfov = math.radians(config.HFOV_DEG) / 2.0
    radius = 0.20
    assert rollout.near_field_blind_distance_m(
        radius=radius, half_hfov_rad=half_hfov, bearing_rad=0.0,
        certified_near_field_m=10.0,
    ) == pytest.approx(radius / math.tan(half_hfov))


def test_near_field_wedge_grows_with_bearing():
    """A corridor centred off-axis leaves the view cone farther out."""
    half_hfov = math.radians(config.HFOV_DEG) / 2.0
    previous = -1.0
    for deg in (0.0, 15.0, 30.0):
        value = rollout.near_field_blind_distance_m(
            radius=0.20, half_hfov_rad=half_hfov,
            bearing_rad=math.radians(deg), certified_near_field_m=10.0)
        assert value > previous
        previous = value
    assert previous > 0.20 / math.tan(half_hfov)


def test_near_field_wedge_never_passes_the_certified_radius():
    """The exemption may not reach past the radius the collision gate guards."""
    half_hfov = math.radians(config.HFOV_DEG) / 2.0
    certified = 2.447
    for deg in (0.0, 15.0, 30.0, 39.4, 45.0, 90.0):
        assert rollout.near_field_blind_distance_m(
            radius=0.25, half_hfov_rad=half_hfov,
            bearing_rad=math.radians(deg),
            certified_near_field_m=certified) <= certified + 1e-12


def test_near_field_wedge_is_inert_on_a_straight_corridor():
    """Every sample of a straight path sits at bearing 0, so v6 cannot move it."""
    frame = make_frame()
    details = rollout.corridor_coverage_details(frame, [Forward(0.5)], 0.2)
    assert details["horizontal_near_field_m"] == pytest.approx(
        0.2 / math.tan(math.radians(frame.sensor.hfov_deg) / 2.0))
    assert details["out_of_frame_samples"] == 0


def test_near_field_wedge_exempts_a_turned_near_corridor():
    """A body turned 30 deg pushes its own flank out of frame within the strip.

    Under v5 the flank counted as out-of-frame evidence and voided the whole
    corridor; the cross-section never fit inside the cone at that range, so the
    penalty measured the formula rather than the scene.
    """
    frame = make_frame()
    turned = rollout.corridor_coverage_details(
        frame, [Turn(30.0), Forward(0.5)], 0.25)
    assert turned["out_of_frame_samples"] == 0


def test_view_collision_rollout_ignores_camera_height_for_overhangs():
    actions = [Forward(1.5)]
    tall = rollout.view_collision_rollout(make_overhang_frame(1.5), actions, 0.2)
    short = rollout.view_collision_rollout(make_overhang_frame(0.5), actions, 0.2)
    assert tall["collision"] is False
    assert short["collision"] is False


def test_full_attribution_rejects_ground_fallback():
    # No in-band surface confirms the depth id and the world_point sits at floor
    # height: the assign() fallback must NOT label the ground as the contact.
    class GroundAssignIndex:
        id_to_cat = {3: "floor", 7: "chair"}

        def assign(self, points):
            return np.full(len(points), 3)      # everything nearest the floor

        def instance_points(self, _instance_id):
            return np.empty((0, 3))             # no in-band confirmation

    base = make_frame()
    frame = dataclasses.replace(
        base, pts_sem=np.full_like(base.pts_sem, 7),
        id_to_cat={3: "floor", 7: "chair"}, semantic_index=GroundAssignIndex())
    physical = {"contact": {"center_local": [0.0, 0.8], "world_point": [0.0, 0.0, -0.8]}}
    consequence._attribute_full_contact(frame, physical)
    assert physical["contact"]["full_geometry_attribution"]["unattributed"] is True


def test_corridor_coverage_measures_swept_body_depth_support():
    frame = make_frame()
    action = [Forward(rollout.certified_near_field_m(frame, 0.25) + 0.5)]
    details = rollout.corridor_coverage_details(
        frame, action, radius=0.25)
    assert details["protocol"] == rollout.EVIDENCE_PROTOCOL_VERSION
    assert details["lateral_sample_count"] == 5
    assert details["coverage"] == 1.0

    depth = frame.depth.copy()
    depth[:, 350:] = 0.0
    partially_visible = dataclasses.replace(frame, depth=depth)
    swept = rollout.corridor_coverage(
        partially_visible, action, radius=0.25)
    centerline = rollout.corridor_coverage(
        partially_visible, action, radius=0.0)
    assert swept < centerline == 1.0


def test_certified_near_field_tracks_camera_height_and_vertical_fov():
    frame = make_frame()
    profile_type = type(frame.sensor)
    low = dataclasses.replace(
        frame, sensor=profile_type.from_values(0.8, 79.0, 70.0))
    high = dataclasses.replace(
        frame, sensor=profile_type.from_values(1.5, 79.0, 70.0))
    narrow = dataclasses.replace(
        frame, sensor=profile_type.from_values(0.8, 79.0, 42.0))

    low_distance = rollout.certified_near_field_m(low, radius=0.25)
    high_distance = rollout.certified_near_field_m(high, radius=0.25)
    narrow_distance = rollout.certified_near_field_m(narrow, radius=0.25)

    assert low_distance == pytest.approx(
        0.8 / math.tan(math.radians(70.0) / 2.0), rel=0.01)
    assert high_distance > low_distance
    assert narrow_distance > low_distance


def test_corridor_coverage_certifies_the_unshown_near_floor_strip():
    frame = make_frame()
    near_field = rollout.certified_near_field_m(frame, radius=0.2)
    no_depth = dataclasses.replace(frame, depth=np.zeros_like(frame.depth))

    coverage = rollout.corridor_coverage(
        no_depth, [Forward(near_field * 0.9)], radius=0.2)

    assert coverage == 1.0


def test_near_floor_certificate_does_not_cover_out_of_hfov_path():
    frame = make_frame()
    near_field = rollout.certified_near_field_m(frame, radius=0.2)
    action = [Turn(45.0), Forward(min(2.0, near_field * 0.9))]

    details = rollout.corridor_coverage_details(
        frame, action, radius=0.2)

    assert details["total_samples"] > 0
    assert details["coverage"] == 0.0
    assert details["meets_visibility_contract"] is False


def test_low_obstacle_occlusion_reduces_ground_not_horizon_coverage():
    frame = make_frame()
    near_field = rollout.certified_near_field_m(frame, radius=0.0)
    depth = np.full_like(frame.depth, 5.0)
    horizon_row = int(round(frame.K[1, 2]))
    depth[horizon_row + 1:, :] = 1.0
    low_obstacle = dataclasses.replace(frame, depth=depth)

    coverage = rollout.corridor_coverage(
        low_obstacle, [Forward(near_field + 0.6)], radius=0.0)

    assert coverage < 0.2


def _corridor_protocol_frame(depth):
    frame = make_frame()
    profile = type(frame.sensor).from_values(
        0.5, config.HFOV_DEG, config.VFOV_DEG)
    return dataclasses.replace(frame, depth=depth, sensor=profile)


def _grid_project_ground(point, _K, camera_height):
    del camera_height
    x, _y, z = point
    return 320.0 + 100.0 * float(x), 50.0 * float(z)


def test_corridor_terminal_strip_requires_all_lateral_samples(monkeypatch):
    monkeypatch.setattr(rollout.perception, "project_ground", _grid_project_ground)
    clear = np.full_like(make_frame().depth, 10.0)

    one_missing = clear.copy()
    one_missing[125:151, 295] = 0.0
    rejected = rollout.corridor_coverage_details(
        _corridor_protocol_frame(one_missing), [Forward(3.0)], radius=0.25)
    assert rejected["terminal_coverage"] == pytest.approx(0.8)
    assert rejected["terminal_invalid_depth_samples"] > 0
    assert rejected["meets_visibility_contract"] is False
    assert rejected["coverage"] == 0.0


def test_corridor_allows_one_nonterminal_invalid_depth_sample(monkeypatch):
    monkeypatch.setattr(rollout.perception, "project_ground", _grid_project_ground)
    clear = np.full_like(make_frame().depth, 10.0)

    one_hole = clear.copy()
    one_hole[100, 295] = 0.0
    accepted = rollout.corridor_coverage_details(
        _corridor_protocol_frame(one_hole), [Forward(3.0)], radius=0.25)
    assert accepted["invalid_depth_samples"] == 1
    assert accepted["terminal_invalid_depth_samples"] == 0
    assert accepted["meets_visibility_contract"] is True


def test_corridor_rejects_two_nonterminal_invalid_depth_samples(monkeypatch):
    monkeypatch.setattr(rollout.perception, "project_ground", _grid_project_ground)
    depth = np.full_like(make_frame().depth, 10.0)
    depth[100:102, 295] = 0.0

    rejected = rollout.corridor_coverage_details(
        _corridor_protocol_frame(depth), [Forward(3.0)], radius=0.25)
    assert rejected["invalid_depth_samples"] == 2
    assert rejected["meets_visibility_contract"] is False
    assert rejected["coverage"] == 0.0


def test_corridor_rejects_even_one_occluded_sample(monkeypatch):
    monkeypatch.setattr(rollout.perception, "project_ground", _grid_project_ground)
    depth = np.full_like(make_frame().depth, 10.0)
    depth[100, 295] = 0.5

    rejected = rollout.corridor_coverage_details(
        _corridor_protocol_frame(depth), [Forward(3.0)], radius=0.25)
    assert rejected["occluded_samples"] == 1
    assert rejected["meets_visibility_contract"] is False
    assert rejected["coverage"] == 0.0


def test_corridor_rejects_even_one_out_of_frame_sample(monkeypatch):
    def one_sample_out_of_frame(point, K, camera_height):
        u, v = _grid_project_ground(point, K, camera_height)
        x, _y, z = point
        if abs(float(x) + 0.25) < 1e-9 and abs(float(z) - 2.0) < 1e-9:
            u = -1.0
        return u, v

    monkeypatch.setattr(
        rollout.perception, "project_ground", one_sample_out_of_frame)
    depth = np.full_like(make_frame().depth, 10.0)

    rejected = rollout.corridor_coverage_details(
        _corridor_protocol_frame(depth), [Forward(3.0)], radius=0.25)
    assert rejected["out_of_frame_samples"] == 1
    assert rejected["meets_visibility_contract"] is False
    assert rejected["coverage"] == 0.0


def test_corridor_coverage_can_stop_at_realized_contact_arc():
    frame = make_frame()
    depth = np.full_like(frame.depth, 0.35)
    limited = dataclasses.replace(frame, depth=depth)
    near_field = rollout.certified_near_field_m(frame, radius=0.2)

    full = rollout.corridor_coverage(
        limited, [Forward(near_field + 0.6)], radius=0.2)
    executed = rollout.corridor_coverage(
        limited, [Forward(near_field + 0.6)], radius=0.2,
        max_arc_m=near_field * 0.9)

    assert full < 1.0
    assert executed == 1.0


class _StairPathFinder(_PathFinder):
    """Non-navigable because the walkable surface below sits far lower."""

    def is_navigable(self, point, max_y_delta=0.5):
        return float(point[2]) >= -3.0

    def snap_point(self, point):
        snapped = np.asarray(point, dtype=np.float64).copy()
        if float(snapped[2]) < -3.0:
            snapped[1] -= 0.8
        return snapped


class _WallPathFinder(_PathFinder):
    """Non-navigable because the disc centre lies outside the eroded polygon."""

    def snap_point(self, point):
        snapped = np.asarray(point, dtype=np.float64).copy()
        if float(snapped[2]) < -3.0:
            snapped[2] = -3.0
        return snapped


class _VoidPathFinder(_PathFinder):
    def snap_point(self, point):
        return np.full(3, np.nan)


class _UnexplainedPathFinder(_PathFinder):
    """Rejects a point that snaps onto the polygon at the same height."""

    def is_navigable(self, point, max_y_delta=0.5):
        return False

    def snap_point(self, point):
        return np.asarray(point, dtype=np.float64).copy()


def _nav(pathfinder):
    return Nav(pathfinder, np.array([1.0, 0.0, -2.0]), 0.0,
               radius_m=0.25, authority="navmesh")


def test_navmesh_vertical_discontinuity_is_unsupported_floor():
    query = _nav(_StairPathFinder()).query_pose((0.0, 1.5, 0.0))
    assert query.navigable is False
    assert query.geometry_source == "unsupported_floor"


def test_navmesh_lateral_overlap_keeps_an_unnamed_collision_source():
    """The radius-eroded navmesh is the collision oracle.

    A centre that projects more than habitat's own horizontal tolerance away
    from the polygon overlaps an obstacle at this radius. Naming that a
    geometry failure would route the benchmark's primary signal into the
    invalid-geometry bucket, so this assertion must never be relaxed.
    """
    query = _nav(_WallPathFinder()).query_pose((0.0, 1.5, 0.0))
    assert query.navigable is False
    assert query.geometry_source is None


def test_navmesh_missing_projection_is_unsupported_floor():
    query = _nav(_VoidPathFinder()).query_pose((0.0, 1.5, 0.0))
    assert query.geometry_source == "unsupported_floor"


def test_navmesh_rejection_neither_clause_explains_is_excluded():
    query = _nav(_UnexplainedPathFinder()).query_pose((0.0, 0.5, 0.0))
    assert query.navigable is False
    assert query.geometry_source == "navmesh_boundary"


@pytest.mark.parametrize(
    ("offset", "expected"),
    (
        # Clearly inside the polygon: only the vertical clause can have failed.
        (sim_module.NAVMESH_LATERAL_SNAP_MAX_M * 0.5, "unsupported_floor"),
        # Within a float32 tie band of the tolerance the two clauses cannot be
        # told apart, and the tie resolves toward contact so that a genuine
        # collision is never renamed a geometry failure.
        (sim_module.NAVMESH_LATERAL_SNAP_MAX_M -
         float(np.nextafter(np.float32(0.0), np.float32(1.0))), None),
        (sim_module.NAVMESH_LATERAL_SNAP_MAX_M * 1.5, None),
    ),
)
def test_navmesh_lateral_split_resolves_ties_toward_contact(offset, expected):
    class _OffsetPathFinder(_PathFinder):
        def is_navigable(self, point, max_y_delta=0.5):
            return False

        def snap_point(self, point):
            snapped = np.asarray(point, dtype=np.float64).copy()
            snapped[0] += offset
            snapped[1] += 5.0
            return snapped

    query = _nav(_OffsetPathFinder()).query_pose((0.0, 0.5, 0.0))
    assert query.geometry_source == expected


def test_navmesh_query_feeds_one_float32_point_to_both_habitat_calls():
    """A dtype mismatch would let the two clauses disagree about the point."""
    seen = []

    class _RecordingPathFinder(_PathFinder):
        def is_navigable(self, point, max_y_delta=0.5):
            seen.append(("is_navigable", np.asarray(point)))
            return False

        def snap_point(self, point):
            seen.append(("snap_point", np.asarray(point)))
            return np.asarray(point, dtype=np.float64).copy()

    _nav(_RecordingPathFinder()).query_pose((0.0, 0.5, 0.0))
    assert [name for name, _ in seen] == ["is_navigable", "snap_point"]
    assert all(value.dtype == np.float32 for _, value in seen)
    assert np.array_equal(seen[0][1], seen[1][1])


def test_navmesh_query_without_snap_point_reports_no_geometry_source():
    query = _nav(_PathFinder()).query_pose((0.0, 1.5, 0.0))
    assert query.navigable is False
    assert query.geometry_source is None


def test_navigable_poses_never_pay_for_a_snap_lookup():
    calls = []

    class _CountingPathFinder(_StairPathFinder):
        def snap_point(self, point):
            calls.append(tuple(np.asarray(point, dtype=np.float64)))
            return super().snap_point(point)

    query = _nav(_CountingPathFinder()).query_pose((0.0, 0.5, 0.0))
    assert query.navigable is True
    assert query.geometry_source is None
    assert calls == []


def test_navmesh_settings_quantise_to_the_ground_band_in_float32():
    """Assert against habitat's own float32 settings, not float64 arithmetic.

    Recast voxelises before eroding and its settings are float32, so a value
    meant to be an exact multiple reads a shade high and gains a whole cell.
    Checked in float64 this passes while the real navmesh enforces 0.35 m of
    clearance and erodes 0.15 m and 0.20 m bodies by the same four cells --
    one navmesh for two radii, which would erase the body axis silently.
    """
    import habitat_sim

    low, high = config.GROUND_OBSTACLE_BAND_M
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.cell_size = config.NAVMESH_CELL_SIZE_M
    settings.cell_height = config.NAVMESH_CELL_HEIGHT_M
    settings.agent_max_climb = config.NAVMESH_MAX_CLIMB_M
    settings.agent_height = config.navmesh_agent_height(
        config.GROUND_ORACLE_HEIGHT_M)

    clearance = math.ceil(
        settings.agent_height / settings.cell_height) * settings.cell_height
    climb = math.floor(
        settings.agent_max_climb / settings.cell_height) * settings.cell_height
    assert clearance == pytest.approx(high, abs=1e-6)
    assert climb == pytest.approx(low, abs=1e-6)

    eroded = {}
    for radius in (*config.RADII_M, 0.30):
        settings.agent_radius = config.navmesh_agent_radius(radius)
        cells = math.ceil(settings.agent_radius / settings.cell_size)
        eroded[radius] = cells
        assert cells * settings.cell_size == pytest.approx(radius, abs=1e-6)
    assert len(set(eroded.values())) == len(eroded)
