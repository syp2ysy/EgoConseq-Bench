"""Point rendering and camera-relative geometry."""

import numpy as np
import pytest
from PIL import Image
from pipeline import io_utils, surface_points, viz

@pytest.mark.parametrize("background", [(0, 0, 0), (255, 255, 255),
                                        (220, 24, 24), (12, 34, 56)])
def test_target_dot_has_contrasting_rings_without_text_or_pixel_shift(tmp_path, background):
    raw = tmp_path / "raw.png"
    Image.new("RGB", (64, 64), tuple(background)).save(raw)
    original = raw.read_bytes()
    authenticated = viz.authenticate_raw_rgb_image(
        raw, expected_sha256=io_utils.sha256_file(raw),
        expected_resolution=(64, 64))
    destination = tmp_path / "marked.png"

    result = viz.materialize_target_point_image(
        authenticated,
        {"pixel_xy_px": [32, 32], "initial_robot_xyz_m": [0.1, 0.2, 2.0]},
        destination)

    with Image.open(destination) as image:
        pixels = np.asarray(image)
    assert result["marker"]["center_xy"] == [32, 32]
    assert "number" not in result["marker"]
    assert np.all(pixels[30:35, 30:35] == [220, 24, 24])
    assert pixels[32, 39].tolist() == [255, 255, 255]
    assert pixels[32, 41].tolist() == [0, 0, 0]
    yy, xx = np.indices(pixels.shape[:2])
    outside = (xx - 32) ** 2 + (yy - 32) ** 2 > 11 ** 2
    assert np.all(pixels[outside] == background)
    assert raw.read_bytes() == original


def test_surface_point_relation_measures_camera_to_fixed_point_in_3d():
    relation = surface_points.relation(
        [1.0, 0.2, 3.0], {"x": 0.0, "z": 1.0, "heading_deg": 0.0}, 1.2)

    assert relation["endpoint_distance_m"] == pytest.approx(6 ** 0.5)
    assert relation["horizontal_direction"] == "front-right"
    assert relation["vertical_direction"] == "below"
    assert "sector_boundary_margin_deg" not in relation


@pytest.mark.parametrize(("camera_height_m", "expected"), (
    (0.5, "above"),
    (1.0, "level"),
    (1.5, "below"),
))
def test_surface_point_direction_uses_camera_optical_center_height(
        camera_height_m, expected):
    relation = surface_points.relation(
        [0.0, 1.0, 2.0], {"x": 0.0, "z": 0.0, "heading_deg": 0.0},
        camera_height_m)

    assert relation["horizontal_direction"] == "front"
    assert relation["vertical_direction"] == expected
