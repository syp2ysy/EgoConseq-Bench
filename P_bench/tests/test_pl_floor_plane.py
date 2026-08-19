"""Deterministic robust floor-plane estimation (P0-4 stage 0).

Pure NumPy: no Habitat, no renderer. Every case here is a synthetic cloud whose
true plane is known, so the assertions are about the estimator itself rather
than about any scene.
"""

import json
import math

import numpy as np
import pytest

from pipeline import config
from pipeline.floor_plane import FloorPlaneEstimate, fit_floor_plane


def _plane_points(*, y0=0.0, tilt_deg=0.0, extent=2.0, step=0.05,
                  noise_m=0.0, seed=0):
    """A rectangular patch of floor, optionally tilted about the x axis."""
    axis = np.arange(-extent / 2.0, extent / 2.0 + 1e-9, step)
    xs, zs = np.meshgrid(axis, axis)
    xs, zs = xs.ravel(), zs.ravel()
    ys = y0 + math.tan(math.radians(tilt_deg)) * zs
    if noise_m:
        ys = ys + np.random.default_rng(seed).normal(0.0, noise_m, ys.shape)
    return np.column_stack([xs, ys, zs])


def _fit(points, **kwargs):
    return fit_floor_plane(points, **kwargs)


def test_exact_horizontal_plane_is_recovered_without_residual():
    result = _fit(_plane_points(y0=-0.12))

    assert result.rejection_reasons == ()
    estimate = result.estimate
    assert isinstance(estimate, FloorPlaneEstimate)
    assert estimate.normal_local == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert estimate.offset_m == pytest.approx(0.12, abs=1e-9)
    assert result.tilt_deg == pytest.approx(0.0, abs=1e-9)
    assert result.residual_rmse_m == pytest.approx(0.0, abs=1e-9)


def test_public_height_degenerates_to_nominal_offset_minus_floor_y():
    # Convention A must reduce to the familiar scalar form on a level floor,
    # otherwise the migration would silently change every published height by
    # more than the correction it is supposed to apply.
    floor_y = -0.075
    estimate = _fit(_plane_points(y0=floor_y)).estimate

    assert estimate.height_above((0.0, 1.5, 0.0)) == pytest.approx(
        1.5 - floor_y, abs=1e-9)
    assert estimate.y_at(0.0, 0.0) == pytest.approx(floor_y, abs=1e-9)


def test_height_above_is_the_signed_normal_distance_on_a_tilted_plane():
    estimate = _fit(_plane_points(y0=-0.05, tilt_deg=3.0)).estimate
    normal_y = estimate.normal_local[1]

    # n . c + d, expressed through the plane's own height under the camera.
    assert estimate.height_above((0.0, 1.5, 0.0)) == pytest.approx(
        normal_y * (1.5 - estimate.y_at(0.0, 0.0)), abs=1e-9)
    # A tilted floor is nearer in normal distance than the naive vertical drop.
    assert estimate.height_above((0.0, 1.5, 0.0)) < 1.5 - estimate.y_at(0.0, 0.0)


def test_the_histogram_only_seeds_the_band_and_never_quantizes_the_result():
    # D3: the 0.05 m bin width must not reach the answer. A floor at 0.037 m
    # sits mid-bin, so a histogram-quantized estimator would land on 0.025 or
    # 0.075 and be wrong by more than a centimetre.
    floor_y = 0.037
    estimate = _fit(_plane_points(y0=floor_y)).estimate

    assert estimate.y_at(0.0, 0.0) == pytest.approx(floor_y, abs=1e-6)
    assert abs(estimate.y_at(0.0, 0.0) - floor_y) < config.FLOOR_HISTOGRAM_BIN_M / 10.0


def test_low_clutter_inside_the_candidate_band_does_not_drag_the_plane_up():
    # Rug edges, thresholds and cable runs sit within centimetres of the floor,
    # so they survive the seed band and a plain least-squares fit would tilt and
    # lift towards them. They must be trimmed, not averaged in.
    floor = _plane_points(y0=0.0, extent=2.0)
    rng = np.random.default_rng(3)
    clutter = np.column_stack([
        rng.uniform(-1.0, 1.0, 700),
        rng.uniform(0.03, 0.09, 700),      # inside the candidate band
        rng.uniform(0.2, 1.0, 700),        # one-sided, so it would also tilt
    ])

    result = _fit(np.vstack([floor, clutter]))

    assert result.estimate.y_at(0.0, 0.0) == pytest.approx(0.0, abs=0.005)
    assert result.estimate.normal_local[1] > 0.9999
    assert result.tilt_deg < 0.5
    # The clutter is visible in the diagnostics rather than absorbed silently.
    assert result.inlier_ratio < 0.95


def test_tilt_gate_accepts_near_horizontal_and_rejects_a_slope():
    gentle = _fit(_plane_points(tilt_deg=3.0))
    steep = _fit(_plane_points(tilt_deg=12.0))

    assert gentle.estimate is not None
    assert gentle.tilt_deg == pytest.approx(3.0, abs=0.05)
    assert steep.estimate is None
    assert "floor_tilt_too_large" in steep.rejection_reasons
    # The measurement that caused the rejection is still reported.
    assert steep.tilt_deg == pytest.approx(12.0, abs=0.05)


def test_thin_support_is_rejected_and_never_falls_back_to_zero():
    result = _fit(_plane_points(extent=0.2, step=0.05))

    assert result.estimate is None
    assert "insufficient_floor_support" in result.rejection_reasons
    assert result.support_count < config.FLOOR_MIN_SUPPORT


def test_a_long_narrow_band_is_rejected_despite_a_large_extent():
    # A one-cell-wide strip can be metres long. Extent alone would pass it; the
    # plane it defines is unconstrained across the narrow axis.
    xs = np.arange(-3.0, 3.0, 0.005)
    points = np.column_stack([
        xs, np.zeros_like(xs), np.full_like(xs, 0.01)])

    result = _fit(points)

    assert result.estimate is None
    assert "floor_support_too_narrow" in result.rejection_reasons
    assert result.support_extent_x_m > config.FLOOR_MIN_SUPPORT_EXTENT_M
    assert result.support_extent_z_m < config.FLOOR_MIN_SUPPORT_EXTENT_M


def test_collinear_support_is_rejected_rather_than_producing_nan():
    zs = np.arange(-3.0, 3.0, 0.005)
    points = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])

    result = _fit(points)

    assert result.estimate is None
    assert result.rejection_reasons
    assert "nan" not in json.dumps(result.to_json()).lower()


def test_row_permutation_gives_bit_identical_coefficients():
    # Determinism is a contract: the same pose must calibrate identically on a
    # rerun, and point order is not part of the pose.
    points = _plane_points(y0=-0.08, tilt_deg=2.0, noise_m=0.004, seed=11)
    shuffled = points[np.random.default_rng(5).permutation(len(points))]

    first = _fit(points).estimate
    second = _fit(shuffled).estimate

    assert first.normal_local == second.normal_local
    assert first.offset_m == second.offset_m


def test_json_form_is_finite_for_both_outcomes():
    fitted = _fit(_plane_points(y0=0.01)).to_json()
    rejected = _fit(_plane_points(extent=0.2, step=0.05)).to_json()

    for payload in (fitted, rejected):
        text = json.dumps(payload)
        assert "NaN" not in text
        assert "Infinity" not in text
    assert fitted["estimate"] is not None
    assert rejected["estimate"] is None
    assert rejected["rejection_reasons"]


def test_every_rejection_carries_a_reason_and_no_estimate():
    for points in (_plane_points(extent=0.2, step=0.05),
                   _plane_points(tilt_deg=20.0),
                   np.empty((0, 3))):
        result = _fit(points)
        assert result.estimate is None
        assert result.rejection_reasons, "a rejection must say why"


def _leveraged_ledge(*, ledge_y, count=500, corner=0.5, seed=7):
    """A floor with a dense low platform packed into one corner.

    The cluster is a minority of the points but sits at the edge of the support,
    so it has high leverage: a plane tilted towards it fits the corner well and
    the far side badly, which a single all-point start can prefer.
    """
    axis = np.linspace(-1.0, 1.0, 31)
    xs, zs = np.meshgrid(axis, axis)
    floor = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()])
    rng = np.random.default_rng(seed)
    ledge = np.column_stack([
        rng.uniform(corner, 1.0, count),
        np.full(count, ledge_y),
        rng.uniform(corner, 1.0, count),
    ])
    return np.vstack([floor, ledge])


@pytest.mark.parametrize("ledge_y", [-0.09, -0.06, -0.04, 0.04, 0.06, 0.09])
@pytest.mark.parametrize("count", [450, 500, 550, 650, 800, 960])
@pytest.mark.parametrize("corner", [0.5, 0.7])
def test_a_leveraged_cluster_is_survived_or_rejected_but_never_published_wrong(
        ledge_y, count, corner):
    """Fail correct or fail loud, across the whole contamination range.

    A single all-point start settles into the tilted basin these clusters create
    and reports it as a clean fit. Once the cluster exceeds the trim fraction the
    trimmed objective itself prefers the tilted plane -- every candidate subset
    is forced to contain non-floor points -- so no tie-break can recover it, and
    the estimator must decline instead.
    """
    result = _fit(_leveraged_ledge(
        ledge_y=ledge_y, count=count, corner=corner))

    if result.fitted:
        assert result.estimate.y_at(0.0, 0.0) == pytest.approx(0.0, abs=0.002)
        assert result.tilt_deg < 0.5
    else:
        assert result.rejection_reasons


def test_residual_gate_is_reachable_within_the_inlier_tolerance():
    # Inliers are selected by residual <= FLOOR_INLIER_TOL_M and the RMSE is then
    # measured over exactly those points, so an RMSE gate above the tolerance is
    # unreachable arithmetic. The two constants must stay ordered.
    assert config.FLOOR_MAX_INLIER_RMSE_M < config.FLOOR_INLIER_TOL_M

    axis = np.linspace(-1.0, 1.0, 41)
    xs, zs = np.meshgrid(axis, axis)
    rng = np.random.default_rng(19)
    spread = 1.5 * config.FLOOR_INLIER_TOL_M
    noisy = np.column_stack([
        xs.ravel(),
        rng.uniform(-spread, spread, xs.size),
        zs.ravel(),
    ])

    result = _fit(noisy)

    # The noise sits inside the tolerance, so the inlier ratio passes its gate
    # and only the residual gate can reject this floor.
    assert result.inlier_ratio >= config.FLOOR_MIN_INLIER_RATIO
    assert result.estimate is None
    assert "floor_residual_too_large" in result.rejection_reasons
    assert result.residual_rmse_m > config.FLOOR_MAX_INLIER_RMSE_M


@pytest.mark.parametrize("normal, offset", [
    ((0.0, 0.0, 0.0), 0.0),                      # zero normal -> y_at divides by 0
    ((0.0, 0.0, 1.0), 0.0),                      # vertical plane, n_y == 0
    ((0.0, -1.0, 0.0), 0.0),                     # downward normal breaks the sign
    ((0.0, 2.0, 0.0), 0.0),                      # not unit length
    ((float("nan"), 1.0, 0.0), 0.0),             # NaN component
    ((0.0, 1.0, 0.0), float("inf")),             # non-finite offset
])
def test_estimate_rejects_a_geometrically_impossible_plane(normal, offset):
    # Stage 1 serializes, validates and reconstructs these, so the constructor is
    # a contract entry point -- not merely whatever the fitter happens to emit.
    with pytest.raises(ValueError):
        FloorPlaneEstimate(normal_local=normal, offset_m=offset)


def test_estimate_round_trips_through_json_under_the_same_validation():
    original = _fit(_plane_points(y0=-0.05, tilt_deg=2.0)).estimate

    restored = FloorPlaneEstimate.from_json(original.to_json())

    assert restored == original
    with pytest.raises(ValueError):
        FloorPlaneEstimate.from_json({"normal_local": [0.0, 0.0, 0.0],
                                      "offset_m": 0.0})


def _tilted_floor_with_low_platform(*, tilt_deg=3.0, platform_y=-0.13,
                                    count=550, corner=0.5, seed=7):
    """A tilted floor whose points spread across bins, plus a flat platform.

    The floor is the larger surface but a tilt spreads its points over several
    height bins, while the platform concentrates all of its points in one. A seed
    chosen by raw bin count therefore picks the platform, and the candidate band
    around it excludes most of the real floor -- so no amount of robust fitting
    downstream can recover points that were filtered out before it ran.
    """
    axis = np.linspace(-1.0, 1.0, 31)
    xs, zs = np.meshgrid(axis, axis)
    floor = np.column_stack([
        xs.ravel(), math.tan(math.radians(tilt_deg)) * zs.ravel(), zs.ravel()])
    rng = np.random.default_rng(seed)
    platform = np.column_stack([
        rng.uniform(corner, 1.0, count),
        np.full(count, platform_y),
        rng.uniform(corner, 1.0, count),
    ])
    return np.vstack([floor, platform])


@pytest.mark.parametrize("platform_y", [-0.13, -0.16, 0.14, 0.18])
@pytest.mark.parametrize("count", [500, 550, 700])
def test_a_dense_platform_cannot_capture_the_seed_from_a_tilted_floor(
        platform_y, count):
    result = _fit(_tilted_floor_with_low_platform(
        platform_y=platform_y, count=count))

    if result.fitted:
        # The real floor passes through the origin; the platform does not.
        assert result.estimate.y_at(0.0, 0.0) == pytest.approx(0.0, abs=0.01)
        assert result.estimate.y_at(0.0, 0.0) != pytest.approx(
            platform_y, abs=0.02)
    else:
        assert result.rejection_reasons


def test_seed_scoring_uses_spatial_occupancy_not_raw_point_count():
    # The defect is upstream of the fit: a small dense patch outvotes a large
    # sparse floor on raw counts, so bins are scored by how much ground they
    # cover instead.
    result = _fit(_tilted_floor_with_low_platform())

    assert result.fitted
    assert result.estimate.y_at(0.0, 0.0) == pytest.approx(0.0, abs=0.01)
    assert result.tilt_deg == pytest.approx(3.0, abs=0.1)


def test_two_equally_supported_floors_are_rejected_as_ambiguous():
    # A mezzanine or a stair landing can present two surfaces of the same extent.
    # Silently choosing one publishes a height that is wrong by the storey
    # separation, so the pose is declined instead.
    axis = np.linspace(-1.0, 1.0, 31)
    xs, zs = np.meshgrid(axis, axis)
    upper = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()])
    lower = np.column_stack([
        xs.ravel(), np.full(xs.size, -0.15), zs.ravel()])

    result = _fit(np.vstack([upper, lower]))

    assert result.estimate is None
    assert "ambiguous_floor_planes" in result.rejection_reasons


@pytest.mark.parametrize("payload", [
    {"normal_local": [0, True, 0], "offset_m": 0.0},
    {"normal_local": [0.0, 1.0, 0.0], "offset_m": "0.1"},
    {"normal_local": ["0.0", 1.0, 0.0], "offset_m": 0.0},
    {"normal_local": [0.0, 1.0, 0.0], "offset_m": True},
])
def test_from_json_rejects_values_that_are_not_json_numbers(payload):
    # From stage 1 this is a persistence trust boundary, and float() happily
    # coerces "0.1" and True into a plane nobody measured.
    with pytest.raises(ValueError):
        FloorPlaneEstimate.from_json(payload)


@pytest.mark.parametrize("count", [1100, 1300])
@pytest.mark.parametrize("ledge_y", [-0.06, -0.04, 0.04, 0.06])
def test_a_contaminant_larger_than_the_floor_is_declined_not_split(
        count, ledge_y):
    """Above the 50% breakdown point the fit must decline, not compromise.

    Scoring seeds by coverage correctly centres the band on the floor, which
    then puts a majority contaminant *inside* the band. The trimmed objective
    settles on a plane that fits part of the floor and part of the ledge: a
    couple of degrees of tilt and a centimetre of height, with a healthy inlier
    ratio. What gives it away is that whole regions of the band are left
    unexplained.
    """
    result = _fit(_leveraged_ledge(ledge_y=ledge_y, count=count))

    if result.fitted:
        assert result.estimate.y_at(0.0, 0.0) == pytest.approx(0.0, abs=0.002)
    else:
        assert "floor_plane_explains_too_little_ground" in \
            result.rejection_reasons
        assert result.inlier_cell_count < result.support_cell_count
