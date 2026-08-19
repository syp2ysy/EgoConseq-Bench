"""Deterministic robust floor-plane estimation for the canonical calibration.

P0-4 stage 0. The published camera height is the signed normal distance from the
camera centre to the canonical floor plane, so a scalar histogram mode cannot
supply it: there is no normal, no tilt and no residual to gate on. This module
fits a near-horizontal plane by a **deterministic multi-start concentration fit**
and reports enough statistics for a pose to be rejected with a stated reason.

That name is deliberate. Concentration steps converge to a local optimum of the
trimmed objective, and a global LTS search over subsets is not performed, so this
does not carry the strict LTS breakdown guarantee. What it does carry is measured
behaviour: across a sweep of leveraged-cluster contaminations it either recovers
the floor or rejects, and never publishes a wrong plane.

Two rules shape the API:

* A rejected fit has no plane. ``FloorPlaneEstimate`` therefore exists only for a
  successful fit, and ``FloorPlaneFitResult.estimate`` is ``None`` otherwise --
  there is no state in which ``y_at()`` returns a pseudo-zero that a caller could
  mistake for a measurement.
* The result is bit-reproducible. Candidate points are sorted lexicographically
  before any reduction, so float summation order does not depend on the order the
  renderer happened to emit points in.

The physics model remains a horizontal SE(2) disc. The normal exists for the
quality gate and for exact calibration; formal data accepts near-horizontal
floors only.

Frame: the agent pose-local frame produced by ``perception.to_agent_ground``,
where the camera centre is ``(0, nominal_camera_offset_m, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from pipeline import config


def _finite(value) -> Optional[float]:
    """JSON-safe float: NaN and infinity become ``None``, never 0.0."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _local_to_world_rotation(yaw_rad: float) -> np.ndarray:
    """The ``R`` of ``perception.world_from_local``: ``w = p + R(yaw) l``.
    Local axes are +x right, +y up, +z forward with yaw=0 facing world -Z, so
    this is a reflection (det = -1), symmetric and equal to its own inverse.
    """
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    return np.array([
        [cosine, 0.0, -sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, -cosine],
    ], dtype=np.float64)


def _json_number(value, label: str) -> float:
    """Accept a real JSON number only -- not a bool, not a numeric string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class FloorPlaneEstimate:
    """A fitted floor plane ``n . p + d = 0`` with ``n_y > 0``.
    Geometry only. Every fit statistic lives on :class:`FloorPlaneFitResult`, so
    there is exactly one place to read a metric and no pair of fields that can
    drift apart.
    """
    normal_local: Tuple[float, float, float]
    offset_m: float
    def __post_init__(self):
        # Stage 1 serializes, validates and reconstructs planes, so this is a
        # contract entry point rather than only whatever the fitter emits. A
        # zero or vertical normal would make y_at() divide by zero, and a
        # downward one silently flips the sign of every published height.
        normal = tuple(float(value) for value in self.normal_local)
        if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
            raise ValueError(
                f"floor plane normal must be three finite numbers: "
                f"{self.normal_local!r}")
        if not math.isfinite(float(self.offset_m)):
            raise ValueError(
                f"floor plane offset must be finite: {self.offset_m!r}")
        norm = math.sqrt(sum(value * value for value in normal))
        if abs(norm - 1.0) > 1e-9:
            raise ValueError(
                f"floor plane normal must be unit length, got {norm!r}")
        if normal[1] <= 0.0:
            raise ValueError(
                f"floor plane normal must point upwards (n_y > 0), got "
                f"{normal[1]!r}")
        object.__setattr__(self, "normal_local", normal)
        object.__setattr__(self, "offset_m", float(self.offset_m))
    @classmethod
    def from_json(cls, payload: dict) -> "FloorPlaneEstimate":
        """Rebuild from a persisted plane under the same invariants.
        Only real JSON numbers are accepted. ``float()`` would coerce ``"0.1"``
        and ``True`` into a plane nobody measured, and from stage 1 this is a
        persistence trust boundary.
        """
        normal = payload.get("normal_local")
        if not isinstance(normal, (list, tuple)) or len(normal) != 3:
            raise ValueError(f"floor plane normal is malformed: {normal!r}")
        return cls(
            normal_local=tuple(
                _json_number(value, "floor plane normal component")
                for value in normal),
            offset_m=_json_number(payload.get("offset_m"), "floor plane offset"))
    def height_above(self, point: Sequence[float]) -> float:
        """Signed normal distance from ``point`` to the plane (Convention A)."""
        x, y, z = (float(value) for value in point)
        n_x, n_y, n_z = self.normal_local
        return n_x * x + n_y * y + n_z * z + self.offset_m
    def height_above_points(self, points) -> np.ndarray:
        """Vectorised :meth:`height_above` over an ``(N, 3)`` point array.
        Every band test in the pipeline -- obstacle band, ground support,
        contact attribution -- asks this one question, so they all share this
        one definition and cannot drift into a vertical-difference convention
        that would disagree with the published height.
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"expected an (N, 3) point array, got {array.shape}")
        return array @ np.asarray(self.normal_local, dtype=np.float64) \
            + self.offset_m
    def y_at(self, x: float, z: float) -> float:
        """Height of the plane under ``(x, z)`` -- where a ground sample lies.
        Distinct from :meth:`height_above`, deliberately: this answers *where
        the surface is* (for reprojecting a ground sample), that one answers
        *how far above it* a point sits (for a band test). They coincide only
        on a level floor.
        """
        n_x, n_y, n_z = self.normal_local
        return -(n_x * float(x) + n_z * float(z) + self.offset_m) / n_y
    def transformed(self, *, from_position, from_yaw_rad: float,
                    to_position, to_yaw_rad: float) -> "FloorPlaneEstimate":
        """Re-express this pose-local plane in another pose's local frame.
        A checkpoint is a different physical pose, so its observation cannot
        simply reuse these coefficients -- but it must not re-fit either, or the
        same action would be judged against two different floors and the future
        view GT would depend on whatever floor the checkpoint happened to see.
        The reference surface is decided once, at the base pose, and carried.
        ``perception.world_from_local`` maps ``w = p + R(yaw) l`` with a
        symmetric, involutive ``R``, so the round trip is ``n' = R' R n`` and the
        y component of the normal -- and therefore the tilt -- is preserved
        exactly.
        """
        normal = np.asarray(self.normal_local, dtype=np.float64)
        source = np.asarray(from_position, dtype=np.float64)
        target = np.asarray(to_position, dtype=np.float64)
        if source.shape != (3,) or target.shape != (3,) or \
                not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("plane transform needs two finite (3,) positions")
        if not (math.isfinite(float(from_yaw_rad)) and
                math.isfinite(float(to_yaw_rad))):
            raise ValueError("plane transform needs two finite yaws")
        world_normal = _local_to_world_rotation(from_yaw_rad) @ normal
        world_offset = self.offset_m - float(world_normal @ source)
        target_normal = _local_to_world_rotation(to_yaw_rad) @ world_normal
        target_offset = world_offset + float(world_normal @ target)
        return FloorPlaneEstimate(
            normal_local=tuple(float(value) for value in target_normal),
            offset_m=float(target_offset))
    def to_json(self) -> dict:
        return {
            "normal_local": [_finite(value) for value in self.normal_local],
            "offset_m": _finite(self.offset_m),
        }


@dataclass(frozen=True)
class FloorPlaneFitResult:
    """Outcome of one fit: the plane if it passed every gate, plus diagnostics.
    Diagnostics are populated as far as the fit progressed, so a pose rejected
    for tilt still reports the tilt that rejected it.
    """
    estimate: Optional[FloorPlaneEstimate]
    rejection_reasons: Tuple[str, ...]
    seed_y_m: Optional[float]
    candidate_band_m: Optional[Tuple[float, float]]
    support_count: int
    support_extent_x_m: float
    support_extent_z_m: float
    support_cell_count: int
    support_area_m2: float
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    inlier_cell_count: Optional[int] = None
    inlier_area_m2: Optional[float] = None
    residual_rmse_m: Optional[float] = None
    residual_p95_m: Optional[float] = None
    tilt_deg: Optional[float] = None
    # Where the fitted plane sat under the camera, kept even when the quality
    # gates withheld it. A scalar, deliberately: a threshold pilot needs to
    # replay hypothesis selection offline, and it must not be able to reach a
    # plane the gates refused. ``estimate`` remains the only usable plane.
    plane_y_at_origin_m: Optional[float] = None
    @property
    def fitted(self) -> bool:
        return self.estimate is not None
    def to_json(self) -> dict:
        return {
            "estimate": self.estimate.to_json() if self.estimate else None,
            "rejection_reasons": list(self.rejection_reasons),
            "seed_y_m": _finite(self.seed_y_m),
            "candidate_band_m": (
                [_finite(value) for value in self.candidate_band_m]
                if self.candidate_band_m is not None else None),
            "support_count": int(self.support_count),
            "support_extent_x_m": _finite(self.support_extent_x_m),
            "support_extent_z_m": _finite(self.support_extent_z_m),
            "support_cell_count": int(self.support_cell_count),
            "support_area_m2": _finite(self.support_area_m2),
            "inlier_count": (int(self.inlier_count)
                             if self.inlier_count is not None else None),
            "inlier_ratio": _finite(self.inlier_ratio),
            "inlier_cell_count": (int(self.inlier_cell_count)
                                  if self.inlier_cell_count is not None else None),
            "inlier_area_m2": _finite(self.inlier_area_m2),
            "residual_rmse_m": _finite(self.residual_rmse_m),
            "residual_p95_m": _finite(self.residual_p95_m),
            "tilt_deg": _finite(self.tilt_deg),
            "plane_y_at_origin_m": _finite(self.plane_y_at_origin_m),
        }


def _rejected(reasons, **fields) -> FloorPlaneFitResult:
    defaults = {
        "seed_y_m": None, "candidate_band_m": None, "support_count": 0,
        "support_extent_x_m": 0.0, "support_extent_z_m": 0.0,
        "support_cell_count": 0, "support_area_m2": 0.0,
    }
    defaults.update(fields)
    return FloorPlaneFitResult(
        estimate=None, rejection_reasons=tuple(reasons), **defaults)


def _seed_bands(points: np.ndarray):
    """Top-K candidate bands, scored by ground coverage rather than point count.
    Raw counts are the defect this replaces: a small dense platform concentrates
    every one of its points into one height bin, while a tilted floor spreads its
    points across several. The platform then wins the seed and the band drawn
    around it filters the real floor out *before* the robust fit runs, so nothing
    downstream can recover it. Scoring a bin by how many distinct ground cells it
    occupies makes a large sparse surface outrank a small dense one.
    Several bands are returned because one is a guess. Each is fitted
    independently and the results are compared.
    """
    low, high = config.FLOOR_SEED_WINDOW_M
    inside = points[(points[:, 1] >= low) & (points[:, 1] <= high)]
    if len(inside) < config.FLOOR_MIN_SUPPORT:
        return [], int(len(inside))
    edges = np.arange(low, high + config.FLOOR_HISTOGRAM_BIN_M,
                      config.FLOOR_HISTOGRAM_BIN_M)
    bins = np.clip(np.digitize(inside[:, 1], edges) - 1, 0, len(edges) - 2)
    cell = config.FLOOR_SUPPORT_CELL_M
    keys = np.floor(inside[:, [0, 2]] / cell).astype(np.int64)
    scored = []
    for index in range(len(edges) - 1):
        selected = bins == index
        if not selected.any():
            continue
        coverage = len(np.unique(keys[selected], axis=0))
        centre = float((edges[index] + edges[index + 1]) / 2.0)
        # Descending coverage, then ascending height: fully deterministic.
        scored.append((-coverage, centre))
    scored.sort()
    half = config.FLOOR_CANDIDATE_HALF_BAND_M
    chosen = []
    for _score, centre in scored:
        if any(abs(centre - taken) < config.FLOOR_SEED_MIN_SEPARATION_M
               for taken in chosen):
            continue
        chosen.append(centre)
        if len(chosen) >= config.FLOOR_SEED_HYPOTHESES:
            break
    return [(centre, (centre - half, centre + half)) for centre in chosen], \
        int(len(inside))


def _support_geometry(points: np.ndarray):
    """Extent per axis plus occupancy, so a one-cell-wide band cannot pass."""
    if not len(points):
        return 0.0, 0.0, 0, 0.0
    extent_x = float(points[:, 0].max() - points[:, 0].min())
    extent_z = float(points[:, 2].max() - points[:, 2].min())
    cell = config.FLOOR_SUPPORT_CELL_M
    cells = np.unique(
        np.floor(points[:, [0, 2]] / cell).astype(np.int64), axis=0)
    return extent_x, extent_z, int(len(cells)), float(len(cells) * cell * cell)


def _solve(points: np.ndarray):
    """Least-squares ``y = a*x + b*z + c``; ``None`` if the support is degenerate."""
    if len(points) < 3:
        return None
    design = np.column_stack([points[:, 0], points[:, 2], np.ones(len(points))])
    solution, _residuals, rank, _singular = np.linalg.lstsq(
        design, points[:, 1], rcond=None)
    if rank < 3 or not np.all(np.isfinite(solution)):
        return None
    return solution


def _residuals(points: np.ndarray, coefficients) -> np.ndarray:
    a, b, c = coefficients
    return np.abs(points[:, 1] - (a * points[:, 0] + b * points[:, 2] + c))


def _cell_median_start(points: np.ndarray):
    """Fit one median height per occupancy cell, then a plane through those.
    Each cell contributes once regardless of how many points it holds, which is
    the specific defence against a dense cluster: a corner packed with hundreds
    of platform points becomes a handful of cells with no more leverage than the
    floor around it.
    """
    cell = config.FLOOR_SUPPORT_CELL_M
    keys = np.floor(points[:, [0, 2]] / cell).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    if len(unique) < 3:
        return None
    medians = np.array([
        np.median(points[inverse == index, 1]) for index in range(len(unique))])
    centres = (unique.astype(np.float64) + 0.5) * cell
    return _solve(np.column_stack([centres[:, 0], medians, centres[:, 1]]))


def _concentrate(points: np.ndarray, coefficients, keep: int):
    """Fixed-count C-steps, each reselecting from the FULL candidate set.
    Reselecting from the previous subset would compound to
    ``(1 - trim)**iterations`` and could never recover a point an early bad fit
    misjudged.
    """
    for _ in range(config.FLOOR_FIT_ITERATIONS):
        order = np.argsort(_residuals(points, coefficients), kind="stable")
        refit = _solve(points[np.sort(order[:keep])])
        if refit is None:
            return None
        coefficients = refit
    return coefficients


def _trimmed_least_squares(points: np.ndarray, seed_y: float):
    """Deterministic multi-start concentration fit, then an inlier refit.
    Concentration steps converge to a *local* optimum of the trimmed objective,
    so a single all-point start is not enough: an ordinary least-squares start is
    pulled by a high-leverage cluster, and the C-steps then happily converge
    inside that wrong basin and report a clean fit. Several fixed starts are run
    and the best trimmed objective wins.
    This is not a claim of the strict LTS breakdown point, which would require a
    global search over subsets. It is a deterministic multi-start approximation:
    no random seed, no convergence test, ties broken by coefficient order.
    """
    keep = max(3, int(math.ceil(
        len(points) * (1.0 - config.FLOOR_TRIM_FRACTION))))
    starts = [
        _solve(points),                                  # all-point OLS
        np.array([0.0, 0.0, float(seed_y)]),             # horizontal at the seed
        _cell_median_start(points),                      # leverage-flattened
    ]
    best = None
    for start in starts:
        if start is None:
            continue
        coefficients = _concentrate(points, start, keep)
        if coefficients is None:
            continue
        squared = np.sort(
            _residuals(points, coefficients) ** 2, kind="stable")
        score = float(squared[:keep].sum())
        key = (score, tuple(float(value) for value in coefficients))
        if best is None or key < best[0]:
            best = (key, coefficients)
    if best is None:
        return None, None
    coefficients = best[1]
    # Inliers are defined by residual, not by the concentration count, so the
    # ratio stays informative: a clean floor approaches 1.0 while a cluttered
    # band does not.
    inliers = points[
        _residuals(points, coefficients) <= config.FLOOR_INLIER_TOL_M]
    if len(inliers) < 3:
        return None, None
    final = _solve(inliers)
    if final is None:
        return None, None
    return final, inliers


def _fit_in_band(points: np.ndarray, seed_y: float, band) -> FloorPlaneFitResult:
    candidates = points[(points[:, 1] >= band[0]) & (points[:, 1] <= band[1])]
    # Deterministic order before any float reduction, so a permuted input cloud
    # produces bit-identical coefficients.
    candidates = candidates[np.lexsort(
        (candidates[:, 1], candidates[:, 2], candidates[:, 0]))]
    extent_x, extent_z, cell_count, area = _support_geometry(candidates)
    diagnostics = {
        "seed_y_m": seed_y, "candidate_band_m": band,
        "support_count": int(len(candidates)),
        "support_extent_x_m": extent_x, "support_extent_z_m": extent_z,
        "support_cell_count": cell_count, "support_area_m2": area,
    }
    reasons = []
    if len(candidates) < config.FLOOR_MIN_SUPPORT:
        reasons.append("insufficient_floor_support")
    if (extent_x < config.FLOOR_MIN_SUPPORT_EXTENT_M or
            extent_z < config.FLOOR_MIN_SUPPORT_EXTENT_M):
        reasons.append("floor_support_too_narrow")
    if cell_count < config.FLOOR_MIN_SUPPORT_CELLS:
        reasons.append("floor_support_too_sparse")
    if reasons:
        return _rejected(reasons, **diagnostics)
    coefficients, inliers = _trimmed_least_squares(candidates, seed_y)
    if coefficients is None:
        return _rejected(("degenerate_floor_support",), **diagnostics)
    a, b, c = (float(value) for value in coefficients)
    normal = np.array([-a, 1.0, -b], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    offset = -c * float(normal[1])
    residual = np.abs(
        inliers[:, 1] - (a * inliers[:, 0] + b * inliers[:, 2] + c))
    tilt_deg = math.degrees(math.acos(min(1.0, max(-1.0, float(normal[1])))))
    _extent_x, _extent_z, inlier_cells, inlier_area = _support_geometry(inliers)
    diagnostics.update({
        "inlier_count": int(len(inliers)),
        "inlier_ratio": float(len(inliers)) / float(len(candidates)),
        "inlier_cell_count": inlier_cells,
        "inlier_area_m2": inlier_area,
        "residual_rmse_m": float(np.sqrt(np.mean(residual ** 2))),
        "residual_p95_m": float(np.percentile(residual, 95)),
        "tilt_deg": tilt_deg,
        "plane_y_at_origin_m": float(-offset / normal[1]),
    })
    if diagnostics["inlier_ratio"] < config.FLOOR_MIN_INLIER_RATIO:
        reasons.append("floor_inlier_ratio_too_low")
    if (inlier_cells / max(cell_count, 1)) < \
            config.FLOOR_MIN_INLIER_COVERAGE_RATIO:
        reasons.append("floor_plane_explains_too_little_ground")
    if diagnostics["residual_rmse_m"] > config.FLOOR_MAX_INLIER_RMSE_M:
        reasons.append("floor_residual_too_large")
    if tilt_deg > config.FLOOR_MAX_TILT_DEG:
        reasons.append("floor_tilt_too_large")
    if reasons:
        return _rejected(reasons, **diagnostics)
    return FloorPlaneFitResult(
        estimate=FloorPlaneEstimate(
            normal_local=(float(normal[0]), float(normal[1]), float(normal[2])),
            offset_m=offset),
        rejection_reasons=(), **diagnostics)


def _selection_key(result: FloorPlaneFitResult):
    """Prefer the hypothesis that explains the most ground, then the cleanest.
    Coverage leads because the competing hypothesis is a small dense surface: it
    can beat the floor on point count, ratio and residual all at once, and only
    "how much ground does this plane actually account for" separates them.
    """
    return (-int(result.inlier_cell_count or 0),
            -float(result.inlier_area_m2 or 0.0),
            -float(result.inlier_ratio or 0.0),
            float(result.residual_rmse_m or 0.0),
            result.estimate.normal_local, result.estimate.offset_m)


def _ambiguous(first: FloorPlaneFitResult, second: FloorPlaneFitResult) -> bool:
    leader = int(first.inlier_cell_count or 0)
    runner_up = int(second.inlier_cell_count or 0)
    if leader <= 0:
        return False
    if runner_up / leader < config.FLOOR_AMBIGUOUS_COVERAGE_RATIO:
        return False
    separation = abs(first.estimate.y_at(0.0, 0.0) -
                     second.estimate.y_at(0.0, 0.0))
    return separation > config.FLOOR_AMBIGUOUS_SEPARATION_M


def _as_rejected(result: FloorPlaneFitResult, reason: str) -> FloorPlaneFitResult:
    payload = {key: value for key, value in vars(result).items()
               if key not in {"estimate", "rejection_reasons"}}
    return FloorPlaneFitResult(
        estimate=None, rejection_reasons=(reason,), **payload)


def fit_floor_plane(points_local) -> FloorPlaneFitResult:
    """Fit the canonical floor plane from an agent pose-local point cloud."""
    return fit_floor_plane_with_attempts(points_local)[0]


def fit_floor_plane_with_attempts(points_local):
    """Return ``(selected, per-band attempts)``.
    The attempt list supports threshold audits: re-thresholding honestly means
    replaying hypothesis selection over every band, not just re-judging the one
    that happened to win under the frozen constants.
    """
    points = np.asarray(points_local, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return _rejected(("no_points",)), []
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return _rejected(("no_finite_points",)), []
    bands, window_count = _seed_bands(points)
    if not bands:
        return _rejected(("insufficient_floor_support",),
                         support_count=window_count), []
    attempts = [_fit_in_band(points, seed_y, band) for seed_y, band in bands]
    viable = sorted((value for value in attempts if value.fitted),
                    key=_selection_key)
    if not viable:
        # Report the hypothesis that came closest, so the reason names the floor
        # the pose actually has rather than the first band tried.
        return max(attempts, key=lambda value: value.support_count), attempts
    if len(viable) >= 2 and _ambiguous(viable[0], viable[1]):
        return _as_rejected(viable[0], "ambiguous_floor_planes"), attempts
    return viable[0], attempts
