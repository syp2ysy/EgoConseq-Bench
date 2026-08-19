"""Independent HFOV/VFOV knobs and exact dynamic sensor provenance."""

import math
import inspect

import numpy as np
import pytest

from pipeline import config, sim as pipeline_sim
from pipeline.frame import RenderObservation, SensorProfile, build_frame, build_future_frame
from pipeline.collection_cli import validate_formal_fov_profiles, sensor_intervention
from tests._synthetic import LEVEL_FLOOR


# --- fixed-resolution monocular camera ------------------------------------

def test_resolution_default_640x480():
    assert config.resolution() == (640, 480)
    assert config.hw() == [480, 640]


def test_intrinsics_default_matches_old_formula():
    fx_old = (640 / 2.0) / math.tan(math.radians(79.0) / 2.0)
    fy_old = (480 / 2.0) / math.tan(math.radians(config.VFOV_DEG) / 2.0)
    K_old = np.array([[fx_old, 0, 320.0], [0, fy_old, 240.0], [0, 0, 1.0]])
    assert config.intrinsics() == pytest.approx(K_old)


# --- the two knobs are independent ----------------------------------------

def test_fov_changes_intrinsics_not_resolution():
    assert config.resolution() == (640, 480)
    assert inspect.signature(config.resolution).parameters == {}
    assert "focal" not in inspect.signature(config.intrinsics).parameters
    narrow = config.intrinsics(69, 42)
    wide = config.intrinsics(110, 70)
    assert narrow[0, 0] > wide[0, 0]
    assert narrow[1, 1] > wide[1, 1]


def test_intrinsics_use_independent_horizontal_and_vertical_fov():
    K = config.intrinsics(110, 70)
    assert K[0, 0] == pytest.approx(320 / math.tan(math.radians(55)))
    assert K[1, 1] == pytest.approx(240 / math.tan(math.radians(35)))
    assert (K[0, 2], K[1, 2]) == (320, 240)


def test_sensor_profile_records_actual_dynamic_camera():
    p = SensorProfile.from_values(nominal_camera_offset_m=0.8, hfov_deg=110,
                                 vfov_deg=70)
    assert p.nominal_camera_offset_m == pytest.approx(0.8)
    assert (p.width_px, p.height_px) == (640, 480)
    assert p.focal_x_px == pytest.approx(config.intrinsics(110, 70)[0, 0])
    assert p.focal_y_px == pytest.approx(config.intrinsics(110, 70)[1, 1])


def test_reference_focal_is_named_as_intermediate_render_setting():
    assert config.RENDER_REFERENCE_FOCAL_PX == pytest.approx(388.19, abs=0.01)
    assert not hasattr(config, "FOCAL_PX")


class _ProfileSim:
    id_to_cat = {}

    def __init__(self, actual_height):
        self.actual_height = actual_height
        self.last_pose = None

    def render(self, position, yaw, cam_h, hfov, vfov):
        self.last_pose = (np.asarray(position), float(yaw))
        profile = SensorProfile.from_values(self.actual_height, hfov, vfov)
        h, w = profile.height_px, profile.width_px
        return RenderObservation(
            rgb=np.zeros((h, w, 3), np.uint8),
            depth=np.ones((h, w), np.float32),
            K=config.intrinsics(hfov, vfov),
            sensor=profile,
            position=np.asarray(position, dtype=np.float64),
            yaw_rad=float(yaw),
        )

    def assign_instances(self, points):
        return np.zeros(len(points), np.int64)

    def dist_to_obstacle(self, position):
        return 1.0


def test_build_frame_rejects_silent_camera_height_snap():
    with pytest.raises(ValueError, match="camera height"):
        build_frame(
            _ProfileSim(actual_height=1.5), np.zeros(3), 0.0,
            frame_id="f", scene_id="s", scene_glb="/tmp/s.glb",
            floor_plane=LEVEL_FLOOR,
            cam_h=0.8, hfov=79, vfov=config.VFOV_DEG,
        )


def test_future_frame_uses_base_pose_and_same_sensor_profile():
    sim = _ProfileSim(actual_height=0.8)
    base = build_frame(
        sim, np.array([1.0, 0.0, -2.0]), 0.0,
        frame_id="base", scene_id="s", scene_glb="/tmp/s.glb",
        floor_plane=LEVEL_FLOOR,
        cam_h=0.8, hfov=110, vfov=70,
    )
    future = build_future_frame(sim, base, (1.0, 2.0, 90.0), checkpoint_index=3)
    position, yaw = sim.last_pose
    assert position == pytest.approx([2.0, 0.0, -4.0])
    assert yaw == pytest.approx(-math.pi / 2)
    assert future.sensor == base.sensor
    assert future.frame_id == "base-k3"


def test_sensor_intervention_taxonomy_tracks_controlled_axes():
    # Height and FOV are both observation-only sensor axes.
    assert sensor_intervention([1.5], [(79, 63.45)]) == ("base", [])
    assert sensor_intervention([0.5, 1.0, 1.5], [(79, 63.45)]) == (
        "sensor_profile", ["nominal_camera_offset_m"])
    assert sensor_intervention([1.5], [(79, 63.45), (110, 70)]) == (
        "sensor_profile", ["hfov_deg", "vfov_deg"])
    assert sensor_intervention(
        [0.5, 1.5], [(79, 63.45), (110, 70)])[0] == "sensor_profile"


def test_formal_collection_accepts_dynamic_but_calibrated_fov_profiles():
    custom_hfov = 90.0
    profiles = validate_formal_fov_profiles([
        (custom_hfov, config.vfov_for_hfov(custom_hfov)),
        config.BENCH_FOVS_DEG[1],
    ])
    assert profiles[0][0] == custom_hfov
    assert profiles[0][1] == pytest.approx(config.vfov_for_hfov(custom_hfov))

    with pytest.raises(ValueError, match="requires VFOV"):
        validate_formal_fov_profiles([(110.0, 70.0)])


def test_habitat_sensor_uuid_distinguishes_height_and_fov():
    values = {
        pipeline_sim.sensor_uuid(kind, height, hfov, vfov)
        for kind in ("color", "depth")
        for height in (0.8, 1.5)
        for hfov, vfov in ((79, 63.453), (110, 70))
    }
    assert len(values) == 8
    assert pipeline_sim.sensor_uuid("color", 0.8, 79, 63.453) \
        == "color_080_f0790x0635"


def test_habitat_pose_sampling_does_not_double_count_navmesh_radius(monkeypatch):
    captured = {}

    def fake_sample_pose(_pf, _render, _rng, radii, **kwargs):
        captured["radii"] = list(radii)
        captured.update(kwargs)
        return "pose"

    session = object.__new__(pipeline_sim.SimSession)
    session._heights = [0.8]
    session._hfov = 79.0
    session._vfov = 63.453
    session._sim = type("FakeHabitat", (), {"pathfinder": object()})()
    session.render = lambda *_args, **_kwargs: None
    monkeypatch.setattr(pipeline_sim, "sample_pose", fake_sample_pose)

    result = session.sample_random_pose(np.random.default_rng(0), [0.15, 0.5])

    assert result == "pose"
    assert captured["radii"] == [0.15, 0.5]
    assert captured["obstacle_margin_m"] == pytest.approx(
        config.BENCH_SAFE_CLEARANCE_M)
