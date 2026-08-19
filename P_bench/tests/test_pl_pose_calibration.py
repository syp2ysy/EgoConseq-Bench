"""Canonical floor calibration at a fixed reference profile (P0-4 stage 1).

``sample_pose`` is duck-typed on its pathfinder and renderer, so the whole
calibration path is exercised here with fakes -- no Habitat, no GPU.
"""

import math
from pathlib import Path
import tempfile

import numpy as np
import pytest

from pipeline import config, perception
from pipeline.floor_plane import fit_floor_plane
from pipeline.frame import RenderObservation, SensorProfile, build_frame
from pipeline.pose_calibration import PoseCalibration
from pipeline import sim as sim_module
from pipeline.sim import sample_pose


class _FakePathfinder:
    """Navigable everywhere, one fixed point, so pose choice is deterministic."""

    def __init__(self, position=(1.0, 0.0, 2.0), clearance=5.0):
        self.position = np.array(position, dtype=np.float64)
        self.clearance = float(clearance)

    def get_random_navigable_point(self):
        return self.position.copy()

    def distance_to_closest_obstacle(self, _position):
        return self.clearance


def _floor_depth(sensor: SensorProfile, K, *, floor_y=0.0, tilt_deg=0.0):
    """Depth of a plane ``y = floor_y + tan(tilt)*z`` seen by this sensor.

    Rendering the geometry rather than hand-writing a cloud keeps the test
    honest about the projection the pipeline actually inverts.
    """
    height, width = sensor.height_px, sensor.width_px
    _us, vs = np.meshgrid(np.arange(width), np.arange(height))
    # perception.unproject uses y = (cy - v)/fy * d, so camera y is up-positive
    # and the ground frame adds the camera height:
    #   -y_over_z * d + h = floor_y + tan * d
    y_over_z = (vs - K[1, 2]) / K[1, 1]
    denominator = y_over_z + math.tan(math.radians(tilt_deg))
    safe = np.abs(denominator) > 1e-6
    depth = np.where(
        safe,
        (sensor.nominal_camera_offset_m - floor_y) / np.where(safe, denominator, 1.0),
        np.nan)
    return np.where(np.isfinite(depth) & (depth > 0.05) & (depth < 8.0),
                    depth, np.nan).astype(np.float64)


class _Renderer:
    """Records every render request so the tests can count and inspect them."""

    def __init__(self, *, floor_y=0.0, tilt_deg=0.0):
        self.requests = []
        self.floor_y = floor_y
        self.tilt_deg = tilt_deg

    def __call__(self, position, yaw, cam_h, hfov, vfov):
        self.requests.append(
            (tuple(np.asarray(position, dtype=float)), float(yaw),
             float(cam_h), float(hfov), float(vfov)))
        sensor = SensorProfile.from_values(cam_h, hfov, vfov)
        K = config.intrinsics(hfov, vfov)
        depth = _floor_depth(
            sensor, K, floor_y=self.floor_y, tilt_deg=self.tilt_deg)
        rgb = np.zeros((sensor.height_px, sensor.width_px, 3), dtype=np.uint8)
        return RenderObservation(
            rgb=rgb, depth=depth, K=K, sensor=sensor,
            position=np.asarray(position, dtype=np.float64),
            yaw_rad=float(yaw))


def _sample(renderer, **kwargs):
    # A small max_tries keeps a broken fixture failing fast instead of spinning
    # through the production retry budget.
    kwargs.setdefault("max_tries", 4)
    return sample_pose(
        _FakePathfinder(), renderer, np.random.default_rng(0), [0.25],
        **kwargs)


def test_calibration_always_uses_the_frozen_reference_profile():
    renderer = _Renderer()

    calibration = _sample(renderer)

    height, hfov, vfov = config.calibration_profile()
    assert (height, hfov, vfov) == (
        config.CAMERA_HEIGHT_M, config.HFOV_DEG, config.VFOV_DEG)
    assert [request[2:] for request in renderer.requests] == [
        (height, hfov, vfov)]
    assert calibration.reference_profile.nominal_camera_offset_m == height
    assert calibration.reference_profile.hfov_deg == hfov


def test_publishing_more_profiles_changes_neither_rng_nor_the_plane():
    first = _Renderer()
    second = _Renderer()

    one = sample_pose(_FakePathfinder(), first, np.random.default_rng(0),
                      [0.25])
    other = sample_pose(_FakePathfinder(), second, np.random.default_rng(0),
                        [0.15, 0.25, 0.5])

    assert one.yaw_rad == other.yaw_rad
    assert one.canonical_plane == other.canonical_plane
    assert len(first.requests) == len(second.requests) == 1


def test_the_calibration_observation_is_reused_rather_than_re_rendered():
    renderer = _Renderer()
    calibration = _sample(renderer)

    class _Sim:
        render = staticmethod(renderer)
        id_to_cat = {}

        @staticmethod
        def assign_instances(points):
            return np.zeros(len(points), dtype=np.int64)

        @staticmethod
        def dist_to_obstacle(_position):
            return 5.0

    before = len(renderer.requests)
    frame = build_frame(
        _Sim(), calibration.position, calibration.yaw_rad,
        frame_id="f", scene_id="s", scene_glb="s.glb",
        floor_plane=calibration.canonical_plane,
        cam_h=calibration.reference_profile.nominal_camera_offset_m,
        hfov=calibration.reference_profile.hfov_deg,
        vfov=calibration.reference_profile.vfov_deg,
        rendered=calibration.reference_observation)

    assert len(renderer.requests) == before, "reuse must not re-render"
    assert frame.rgb is calibration.reference_observation.rgb


def test_a_reused_observation_still_passes_the_input_contract():
    # The cache must not become a way around the profile and shape checks.
    renderer = _Renderer()
    calibration = _sample(renderer)

    class _Sim:
        render = staticmethod(renderer)
        id_to_cat = {}

        @staticmethod
        def assign_instances(points):
            return np.zeros(len(points), dtype=np.int64)

        @staticmethod
        def dist_to_obstacle(_position):
            return 5.0

    with pytest.raises(ValueError, match="camera height mismatch"):
        build_frame(
            _Sim(), calibration.position, calibration.yaw_rad,
            frame_id="f", scene_id="s", scene_glb="s.glb",
            floor_plane=calibration.canonical_plane,
            cam_h=config.CAMERA_HEIGHT_M + 0.5,
            hfov=config.HFOV_DEG, vfov=config.VFOV_DEG,
            rendered=calibration.reference_observation)

    with pytest.raises(ValueError, match="FOV mismatch"):
        build_frame(
            _Sim(), calibration.position, calibration.yaw_rad,
            frame_id="f", scene_id="s", scene_glb="s.glb",
            floor_plane=calibration.canonical_plane,
            cam_h=config.CAMERA_HEIGHT_M, hfov=110.0,
            vfov=config.vfov_for_hfov(110.0),
            rendered=calibration.reference_observation)


def test_a_rejected_floor_fit_advances_to_the_next_pose_with_a_reason():
    class _MovingPathfinder(_FakePathfinder):
        def __init__(self):
            super().__init__()
            self.draws = 0

        def get_random_navigable_point(self):
            self.draws += 1
            return self.position.copy()

    class _SlopeThenFloor(_Renderer):
        def __call__(self, position, yaw, cam_h, hfov, vfov):
            # Reject the first pose with an unusable slope, then a good floor.
            self.tilt_deg = 30.0 if not self.requests else 0.0
            return super().__call__(position, yaw, cam_h, hfov, vfov)

    pathfinder = _MovingPathfinder()
    renderer = _SlopeThenFloor()
    reasons = []

    calibration = sample_pose(
        pathfinder, renderer, np.random.default_rng(0), [0.25],
        on_reject=reasons.append)

    assert calibration is not None
    assert pathfinder.draws >= 2
    assert reasons, "a rejected calibration must say why"
    assert all(isinstance(reason, str) and reason for reason in reasons)


def test_sample_pose_no_longer_uses_the_scalar_floor_estimate():
    assert not hasattr(perception, "estimate_floor_height")

    calibration = _sample(_Renderer(floor_y=0.06))

    assert calibration.canonical_plane.y_at(0.0, 0.0) == pytest.approx(
        0.06, abs=0.01)


def test_the_plane_stays_pose_local_and_the_fit_must_have_succeeded():
    calibration = _sample(_Renderer(floor_y=-0.08))

    # Pose-local: the plane is expressed under the camera at the origin, with no
    # scene-coordinate magnitude, even though the pose sits away from origin.
    assert np.linalg.norm(calibration.position) > 1.0
    assert abs(calibration.canonical_plane.y_at(0.0, 0.0)) < 0.2
    assert calibration.canonical_floor_fit.fitted

    rejected = fit_floor_plane(np.empty((0, 3)))
    with pytest.raises(ValueError, match="fitted"):
        PoseCalibration(
            position=calibration.position, yaw_rad=calibration.yaw_rad,
            reference_profile=calibration.reference_profile,
            reference_observation=calibration.reference_observation,
            canonical_floor_fit=rejected)


def test_calibration_rejects_an_observation_from_another_profile():
    calibration = _sample(_Renderer())
    other = _Renderer()(calibration.position, 0.0, 0.5, 110.0,
                        config.vfov_for_hfov(110.0))

    with pytest.raises(ValueError, match="calibration profile"):
        PoseCalibration(
            position=calibration.position, yaw_rad=calibration.yaw_rad,
            reference_profile=other.sensor,
            reference_observation=other,
            canonical_floor_fit=calibration.canonical_floor_fit)


def test_the_calibration_profile_is_registered_even_when_unpublished():
    # Habitat/R2R can only render profiles it registered sensors for, and
    # sample_pose always asks for the calibration one. A run publishing only
    # 110 degrees would otherwise raise on every pose.
    published_heights = [0.5, 1.0]
    published_fovs = [(110.0, config.vfov_for_hfov(110.0))]

    profiles = sim_module.render_profile_set(published_heights, published_fovs)

    assert config.calibration_profile() in profiles
    # The published sets are untouched: siblings must not gain a profile.
    assert published_heights == [0.5, 1.0]
    assert published_fovs == [(110.0, config.vfov_for_hfov(110.0))]


def test_registering_the_calibration_profile_does_not_duplicate_it():
    profiles = sim_module.render_profile_set(
        [config.CAMERA_HEIGHT_M], [(config.HFOV_DEG, config.VFOV_DEG)])

    assert profiles == ((config.CAMERA_HEIGHT_M, config.HFOV_DEG,
                         config.VFOV_DEG),)


@pytest.mark.parametrize("heights, hfovs", [
    ([0.5, 1.0], [110.0]),
    ([0.8, 1.2, 1.6], [110.0, 60.0]),
    ([1.5], [79.0]),
])
def test_the_render_set_is_a_union_of_profiles_not_a_product(heights, hfovs):
    # Merging the height list and the FOV list separately and letting the
    # session cross them registers sensors nobody renders: two published
    # heights and one FOV asked for six Habitat sensors where three suffice,
    # and each one costs resolution-sized colour and depth buffers.
    fovs = [(value, config.vfov_for_hfov(value)) for value in hfovs]

    profiles = sim_module.render_profile_set(heights, fovs)

    expected = {(height, hfov, vfov)
                for hfov, vfov in fovs for height in heights}
    expected.add(config.calibration_profile())
    assert set(profiles) == expected
    assert len(profiles) == len(expected)
    # Deterministic order: sensor registration must not depend on set iteration.
    assert list(profiles) == sorted(profiles)


def test_a_session_renders_the_calibration_profile_it_does_not_publish():
    session = object.__new__(sim_module.SimSession)
    published_fovs = [(110.0, config.vfov_for_hfov(110.0))]
    session._heights = [0.5, 1.0]
    session._fovs = published_fovs
    session._render_profiles = sim_module.render_profile_set(
        session._heights, session._fovs)
    session._hfov, session._vfov = published_fovs[0]

    height, hfov, vfov = config.calibration_profile()
    # Reaching the agent-state call means the profile guard let it through.
    with pytest.raises(AttributeError):
        session.render((0.0, 0.0, 0.0), 0.0, height, hfov, vfov)
    with pytest.raises(ValueError, match="not configured"):
        session.render((0.0, 0.0, 0.0), 0.0, 0.9, hfov, vfov)
    # The cross terms were never registered, so asking for one must fail
    # rather than silently render a profile the run does not publish.
    with pytest.raises(ValueError, match="not configured"):
        session.render((0.0, 0.0, 0.0), 0.0, height, *published_fovs[0])
    with pytest.raises(ValueError, match="not configured"):
        session.render((0.0, 0.0, 0.0), 0.0, 0.5, hfov, vfov)


def test_session_reuses_radius_conditioned_navmesh_within_a_scene():
    class Pathfinder:
        def __init__(self):
            self.saved = []
            self.loaded = []

        def save_nav_mesh(self, path):
            self.saved.append(str(path))
            Path(path).write_bytes(b"navmesh")
            return True

        def load_nav_mesh(self, path):
            self.loaded.append(str(path))
            return True

    class Simulator:
        def __init__(self):
            self.pathfinder = Pathfinder()
            self.recomputed = 0

        def recompute_navmesh(self, _pathfinder, _settings):
            self.recomputed += 1
            return True

    with tempfile.TemporaryDirectory(
            prefix="egoconseq-test-navmesh-") as temp_root:
        session = object.__new__(sim_module.SimSession)
        session._sim = Simulator()
        session._navmesh_cache_dir = Path(temp_root)
        session._navmesh_cache = {}
        session._active_navmesh_key = None

        session.recompute_navmesh(0.15)
        session.recompute_navmesh(0.25)
        session.recompute_navmesh(0.15)

        assert session._sim.recomputed == 2
        assert len(session._sim.pathfinder.saved) == 2
        assert len(session._sim.pathfinder.loaded) == 1
        assert {
            Path(path).parent for path in session._sim.pathfinder.saved
        } == {Path(temp_root)}


def test_a_cached_observation_from_another_pose_is_refused():
    # The cache is keyed by nothing today: an observation with the right
    # profile but the wrong pose would be accepted and silently describe a
    # different place.
    renderer = _Renderer()
    calibration = _sample(renderer)

    class _Sim:
        render = staticmethod(renderer)
        id_to_cat = {}

        @staticmethod
        def assign_instances(points):
            return np.zeros(len(points), dtype=np.int64)

        @staticmethod
        def dist_to_obstacle(_position):
            return 5.0

    height, hfov, vfov = config.calibration_profile()
    with pytest.raises(ValueError, match="pose"):
        build_frame(
            _Sim(), calibration.position + np.array([1.0, 0.0, 0.0]),
            calibration.yaw_rad, frame_id="f", scene_id="s", scene_glb="s.glb",
            floor_plane=calibration.canonical_plane,
            cam_h=height, hfov=hfov, vfov=vfov,
            rendered=calibration.reference_observation)
    with pytest.raises(ValueError, match="pose"):
        build_frame(
            _Sim(), calibration.position, calibration.yaw_rad + 0.5,
            frame_id="f", scene_id="s", scene_glb="s.glb",
            floor_plane=calibration.canonical_plane,
            cam_h=height, hfov=hfov, vfov=vfov,
            rendered=calibration.reference_observation)


def test_an_observation_whose_K_contradicts_its_profile_is_refused():
    # sample_pose unprojects with K before anything validates it, so a renderer
    # whose sensor fields are right but whose K is wrong would fit a plausible
    # plane from the wrong geometry.
    height, hfov, vfov = config.calibration_profile()
    good = _Renderer()((0.0, 0.0, 0.0), 0.0, height, hfov, vfov)
    wrong_k = RenderObservation(
        rgb=good.rgb, depth=good.depth,
        K=config.intrinsics(110.0, config.vfov_for_hfov(110.0)),
        sensor=good.sensor, position=good.position, yaw_rad=good.yaw_rad)

    with pytest.raises(ValueError, match="intrinsics"):
        sim_module.validate_render_observation(
            wrong_k, position=(0.0, 0.0, 0.0), yaw=0.0,
            cam_h=height, hfov=hfov, vfov=vfov)

    non_finite = RenderObservation(
        rgb=good.rgb, depth=good.depth, K=np.full((3, 3), np.nan),
        sensor=good.sensor, position=good.position, yaw_rad=good.yaw_rad)
    with pytest.raises(ValueError, match="intrinsics"):
        sim_module.validate_render_observation(
            non_finite, position=(0.0, 0.0, 0.0), yaw=0.0,
            cam_h=height, hfov=hfov, vfov=vfov)


def _reference_observation():
    height, hfov, vfov = config.calibration_profile()
    return _Renderer()((0.0, 0.0, 0.0), 0.0, height, hfov, vfov)


def test_an_observation_must_carry_the_pose_it_was_taken_from():
    # A pose-less observation makes the cache guard unfalsifiable. While the
    # pose was optional the validator skipped it entirely, so an observation
    # with the right profile and no pose was accepted at any position at all.
    good = _reference_observation()

    with pytest.raises(TypeError):
        RenderObservation(rgb=good.rgb, depth=good.depth, K=good.K,
                          sensor=good.sensor)


@pytest.mark.parametrize("position, yaw", [
    ((0.0, 0.0), 0.0),                        # too few components
    ((0.0, 0.0, 0.0, 0.0), 0.0),              # too many
    (0.0, 0.0),                               # a scalar broadcasts under allclose
    ((float("nan"), 0.0, 0.0), 0.0),
    ((float("inf"), 0.0, 0.0), 0.0),
    ((0.0, 0.0, 0.0), float("nan")),
    ((0.0, 0.0, 0.0), float("inf")),
])
def test_an_observation_with_a_malformed_pose_is_refused(position, yaw):
    good = _reference_observation()

    with pytest.raises(ValueError):
        RenderObservation(rgb=good.rgb, depth=good.depth, K=good.K,
                          sensor=good.sensor, position=position, yaw_rad=yaw)


def test_a_pose_drift_at_scene_scale_is_refused():
    # A relative tolerance grows with the coordinate magnitude. Large scenes
    # reach tens of metres from the origin and the comparison ran at the numpy
    # default rtol=1e-5, so far from the origin millimetres of drift passed --
    # and a millimetre of drift is a different place, not a rounding artefact.
    height, hfov, vfov = config.calibration_profile()
    far = np.array([500.0, 0.0, 0.0])
    good = _Renderer()(far, 0.0, height, hfov, vfov)

    with pytest.raises(ValueError, match="pose position"):
        sim_module.validate_render_observation(
            good, position=far + np.array([4e-3, 0.0, 0.0]), yaw=0.0,
            cam_h=height, hfov=hfov, vfov=vfov)


@pytest.mark.parametrize("turns", [1, -1, 2])
def test_a_yaw_that_differs_by_whole_turns_is_the_same_heading(turns):
    # Habitat yaw arrives from a quaternion and a caller may hold the
    # unwrapped angle, so comparing raw radians rejects the same heading.
    good = _reference_observation()
    height, hfov, vfov = config.calibration_profile()

    sim_module.validate_render_observation(
        good, position=(0.0, 0.0, 0.0), yaw=turns * 2.0 * math.pi,
        cam_h=height, hfov=hfov, vfov=vfov)


def test_a_yaw_half_a_turn_away_is_still_refused():
    good = _reference_observation()
    height, hfov, vfov = config.calibration_profile()

    with pytest.raises(ValueError, match="pose yaw"):
        sim_module.validate_render_observation(
            good, position=(0.0, 0.0, 0.0), yaw=math.pi,
            cam_h=height, hfov=hfov, vfov=vfov)


@pytest.mark.parametrize("position, yaw", [
    ((0.0, 0.0), 0.0),
    ((float("nan"), 0.0, 0.0), 0.0),
    ((0.0, 0.0, 0.0), float("nan")),
])
def test_a_malformed_requested_pose_is_refused(position, yaw):
    # The request side is checked too: allclose(nan, nan) is False, so a
    # non-finite request would otherwise be reported as a pose mismatch --
    # a true rejection with a misleading reason.
    good = _reference_observation()
    height, hfov, vfov = config.calibration_profile()

    with pytest.raises(ValueError, match="requested pose"):
        sim_module.validate_render_observation(
            good, position=position, yaw=yaw,
            cam_h=height, hfov=hfov, vfov=vfov)
