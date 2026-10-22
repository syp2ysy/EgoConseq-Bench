"""Frozen constants — single source of truth for the consequence pipeline.

Do NOT scatter these literals elsewhere; import from here.
Values mirror the validated egoconseq design plus structured rollout knobs.
"""

import math
import os
from pathlib import Path

# --- data / scene pool ---
_repository_root = Path(__file__).resolve().parents[1]
_data_root = Path(os.environ.get(
    "EGOCONSEQ_DATA_ROOT", _repository_root / "data" / "sources"))
R2R_TRAIN_EPISODES = os.environ.get(
    "EGOCONSEQ_R2R_TRAIN_EPISODES",
    str(_data_root / "r2r_vlnce_v1-3" / "train" / "train.json.gz"))
MP3D_ROOT = os.environ.get(
    "EGOCONSEQ_MP3D_ROOT",
    str(_data_root / "scene_datasets" / "mp3d"))

# GS backend (Habitat-GS 3DGS scenes + InteriorGS bbox labels), one dir per scene:
#   <GS_ROOT>/<scene>/{scene.gs.ply, scene.navmesh, labels.json}
GS_ROOT = os.environ.get(
    "EGOCONSEQ_GS_ROOT", str(_data_root / "gs"))
B1K_ROOT = os.environ.get("B1K_DATA_ROOT", str(_data_root / "behavior-1k-v3.9.1"))
B1K_SOURCE_MANIFEST = os.environ.get("B1K_SOURCE_MANIFEST", os.path.join(
    B1K_ROOT, "pbench-abc1-task4", "catalog-authority-audit-v2-20260812", "audit-derived-source-manifest.json"))

# --- sensor (Habitat defaults) ---
# Calibrated monocular pinhole output. Formal profiles use square pixels and the
# same 4:3 raster; changing HFOV therefore changes VFOV by calibration, never by
# anisotropically stretching a rendered image.
HFOV_DEG = 79.0                   # horizontal FOV -> image width
_BASE_WIDTH, _BASE_HEIGHT = 640, 480   # calibration reference resolution
RENDER_REFERENCE_FOCAL_PX = (
    (_BASE_WIDTH / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0))  # ≈388.19
VFOV_DEG = math.degrees(2.0 * math.atan(
    (_BASE_HEIGHT / 2.0) / RENDER_REFERENCE_FOCAL_PX))  # ≈63.45
CAMERA_HEIGHT_M = 1.5
OUTPUT_RESOLUTION = (_BASE_WIDTH, _BASE_HEIGHT)
# Lossless PNG encoding for QA-only numbered target overlays.  Level 1 keeps
# identical pixels while avoiding the large CPU cost of maximum compression.
QA_MARKED_PNG_COMPRESSION_LEVEL = 1


def vfov_for_hfov(hfov_deg: float, *, width: int = _BASE_WIDTH,
                  height: int = _BASE_HEIGHT) -> float:
    """Vertical FOV for a square-pixel pinhole camera at ``width:height``."""
    return math.degrees(2.0 * math.atan(
        math.tan(math.radians(float(hfov_deg)) / 2.0) *
        float(height) / float(width)))

# Formal benchmark observation profiles.  Camera parameters are independent
# sensor variables; they never alter the ground-disc physical oracle.
BENCH_FOVS_DEG = tuple(
    (hfov, vfov_for_hfov(hfov)) for hfov in (79.0, 110.0))

# --- body (robot chassis footprint radii; edit here to change the tested bodies) ---
# Real home chassis robots span ~0.12-0.25 m radius (Ø24-50 cm):
# robot vacuum ~0.17, Amazon Astro ~0.21, Temi ~0.22, Pepper ~0.24.
RADII_M = (0.15, 0.20, 0.25)

# Physical consequences are defined for a 2D circular chassis.  Both full and
# depth oracles use the same thin, fixed ground-support slab to reject floor
# points and ignore overhead geometry.  This is an oracle rasterization rule,
# not a robot-height parameter; camera height never changes physical labels.
GROUND_OBSTACLE_BAND_M = (0.05, 0.30)
B1K_RESET_POSITION_TOL_M = 1e-6
B1K_RESET_ORIENTATION_TOL_RAD = 1e-6
# Runtime floor meshes are closed solids.  Only upward-facing support faces
# are walkable floor authority; bottom and side faces remain collision input.
B1K_FLOOR_UP_NORMAL_MIN = 0.5
GROUND_ORACLE_HEIGHT_M = GROUND_OBSTACLE_BAND_M[1]
# Fine-collision v4 body (declared configuration, validated by pilot tests):
# the camera is mounted at the top of the robot, the body top is 0.10 m above
# the camera optical centre, and the body clears the floor by 0.05 m.
FINE_BODY_TOP_ABOVE_CAMERA_M = 0.10
FINE_BODY_GROUND_CLEARANCE_M = 0.05


def fine_body(camera_height_m: float) -> dict:
    """Explicit collision envelope for one camera height (metres above floor)."""
    top = round(float(camera_height_m) + FINE_BODY_TOP_ABOVE_CAMERA_M, 6)
    clearance = FINE_BODY_GROUND_CLEARANCE_M
    return {"camera_height_m": float(camera_height_m), "top_m": top,
            "ground_clearance_m": clearance, "cylinder_length_m": round(top - clearance, 6),
            "habitat_navmesh_agent_height_m": top, "habitat_max_climb_m": globals()["NAVMESH_MAX_CLIMB_M"]}
# Recast voxelises before it erodes, so leaving these at habitat's defaults
# silently rounds the band: a 0.2 m cell height turns a 0.30 m agent into
# ceil(0.30/0.2)=2 cells = 0.40 m, and a 0.2 m climb lets the navmesh step over
# obstacles the depth and Gaussian oracles both call solid. At 0.05 m both ends
# of the band land on exact cell boundaries. This aligns the three oracles'
# discretisation; it does not make them the same body.
NAVMESH_CELL_SIZE_M = 0.05
NAVMESH_CELL_HEIGHT_M = 0.05
NAVMESH_MAX_CLIMB_M = GROUND_OBSTACLE_BAND_M[0]


def navmesh_cell_clearance_m(height_m: float) -> float:
    """Return the clearance Recast actually enforces for ``height_m``."""
    cells = math.ceil(
        round(float(height_m) / NAVMESH_CELL_HEIGHT_M, 6))
    return cells * NAVMESH_CELL_HEIGHT_M


def navmesh_agent_height(height_m: float) -> float:
    """Aim at the middle of the intended cell so float32 cannot round up.

    Recast takes ``ceil(agent_height / cell_height)`` in float32, where a value
    meant to be an exact multiple can read a shade high and gain a whole cell.
    """
    return navmesh_cell_clearance_m(height_m) - NAVMESH_CELL_HEIGHT_M * 0.5


def navmesh_erosion_m(radius_m: float) -> float:
    """Return the footprint Recast actually erodes by for ``radius_m``."""
    cells = round(float(radius_m) / NAVMESH_CELL_SIZE_M)
    if cells < 1:
        raise ValueError("body radius is smaller than one navmesh cell")
    return cells * NAVMESH_CELL_SIZE_M


def navmesh_agent_radius(radius_m: float) -> float:
    """Aim at the middle of the intended cell, as ``navmesh_agent_height`` does.

    Left uncorrected the same float32 ceiling collapses neighbouring bodies:
    0.15 m and 0.20 m both erode by four cells and become one navmesh, which
    would silently erase the body axis this benchmark is built on.
    """
    return navmesh_erosion_m(radius_m) - NAVMESH_CELL_SIZE_M * 0.5

# --- perception / occupancy ---
# Compatibility is intentionally not provided for the former camera-coupled
# occupancy band.  All current code must use GROUND_OBSTACLE_BAND_M.
VOXEL_SIZE_M = 0.05
VOXEL_DILATION = 1                # dilation iterations
MIN_SUPPORT_VOXELS = 3            # occupied voxels within footprint -> contact

# --- action geometry / march ---
MARCH_STEP_M = 0.02               # path sample spacing
D_MAX_M = 5.0                     # cap when marching for "no contact" (nav/collision)
# --- attribution ---
NAV_Y_DELTA_M = 0.5               # footprint query Y-snap tolerance

# --- current-view evidence ---
DEPTH_SUPPORT_TOL_M = 0.3         # projected-pixel depth slack for corridor support

# --- GS runtime geometry and rendering ---
GS_INSTANCE_MAX_POINTS = 10_000      # visible semantic point sample cap
# Official InteriorGS navmeshes report the agent surface above the collision
# floor.  Millimetre-raised horizontal shells in the USD are the same floor
# stratum for pose grounding, not independent walkable levels.  This matches
# the frozen floor inlier tolerance used by RGB-D calibration below.
GS_COLLISION_COPLANAR_FLOOR_M = 0.02
GS_RENDER_NEAR_M = 0.05
GS_RENDER_FAR_M = 50.0
GS_COLLISION_PLANAR_GRID_M = 1e-9  # numerical USD projection canonicalization

# --- object extraction filters ---
OBJ_MIN_AREA_PX = 400
OBJ_MIN_VALID_DEPTH = 20
OBJ_MAX_POINTS = 2000             # transient subsample cap for points_xz
# A4/B1/B2 use exact depth-backed instance pixels, without object-size gates.
SURFACE_TARGET_EXCLUDED_MATERIAL_TOKENS = frozenset({
    "glass", "mirror", "reflective", "transparent",
})

# The terminal checkpoint and independently accumulated realized endpoint are
# the same analytic pose reached through different floating-point paths.  The
# dimensioned tolerances accept only arithmetic dust; neither is a rendering,
# collision, or family-diversity threshold.
TARGET_DIRECTION_MIN_RANGE_M = 0.05
# MP3D A3 complete-face queries keep a small authenticated index cache.
MP3D_TARGET_FACE_INDEX_CACHE_MAX_SCENES = 2
MP3D_TARGET_FACE_INDEX_CACHE_MAX_BYTES = 512 * 1024 * 1024
MP3D_TARGET_FACE_INDEX_STREAM_BYTES = 1024 * 1024
MP3D_AABB_MADVISE_INTERVAL_CHUNKS = 8
# Conservative absolute slack used only by the AABB pruning decision.  Exact
# triangle distances, not this value, determine the published top two.
A3_FACE_AABB_PRUNE_SLACK_M = 1e-9
# Public metric-distance contract, shared by B1 and B3.
B1_OPEN_DISPLAY_DECIMALS = 2
EVIDENCE_COVERAGE_MIN = 0.90
EVIDENCE_CORRIDOR_LATERAL_SAMPLES = 5
EVIDENCE_MIN_LATERAL_FRACTION = 1.00
EVIDENCE_TERMINAL_LENGTH_M = 0.50
EVIDENCE_MAX_UNSUPPORTED_RUN_M = 0.00
EVIDENCE_MAX_INVALID_DEPTH_SAMPLES = 1

# --- offline semantic geometry and runtime assignment ---
SEMANTIC_CACHE_DIR = ".cache/pbench/semantic"
SEMANTIC_SAMPLE_COUNT = 1_000_000  # surface samples per scene (labelled subset kept)
SEMANTIC_ASSIGN_TOL_M = 0.20       # depth point -> nearest labelled surface cutoff
# Full-frame depth assignment is large enough to amortize scipy's worker-pool
# startup; contact probes remain single-threaded for lower latency.
SEMANTIC_ASSIGN_PARALLEL_MIN_POINTS = 20_000
# A3's exact face confirmation is a semantic identity gate, not a second
# collision oracle.  The full-mesh witness must be no farther away than the
# existing semantic assignment cutoff, and must beat every face-instance in
# the complete semantic PLY by at least one physical march step.
A3_CONTACT_FACE_MAX_DISTANCE_M = SEMANTIC_ASSIGN_TOL_M
A3_CONTACT_FACE_TIE_MARGIN_M = MARCH_STEP_M
TARGET_POINT_MARKER_RADIUS_PX = 10

# Free-text category substrings flagged as structural (flag only, not dropped).
STRUCTURAL_CATEGORIES = frozenset({
    "wall", "floor", "ceiling", "door", "window", "stairs", "staircase",
    "handrail", "railing", "beam", "column", "pillar", "unknown",
})

# Exact source-taxonomy sentinels that do not identify a recognizable public
# target or contact class. Keep this separate from STRUCTURAL_CATEGORIES:
# structural matching is intentionally substring-based, while this contract
# must not reject a concrete label merely because it contains a generic token.
NON_SPECIFIC_SEMANTIC_CATEGORIES = frozenset({
    "misc", "miscellaneous", "unknown", "unlabeled", "unlabelled",
    "other", "others", "object", "objects", "background", "void", "none",
})


def render_resolution(hfov=HFOV_DEG, vfov=VFOV_DEG,
                      reference_focal_px=RENDER_REFERENCE_FOCAL_PX):
    """Output raster for an aspect-consistent square-pixel FOV profile."""
    del reference_focal_px
    expected = vfov_for_hfov(float(hfov))
    if abs(float(vfov) - expected) > 1e-3:
        raise ValueError(
            f"HFOV {float(hfov):g} requires VFOV {expected:.6g} at 4:3; "
            f"got {float(vfov):g}")
    return OUTPUT_RESOLUTION


def resolution():
    """Normalized monocular output size shared by every calibrated profile."""
    return OUTPUT_RESOLUTION


def hw():
    """Normalized output resolution [H, W]."""
    w, h = resolution()
    return [h, w]


def render_hw(hfov=HFOV_DEG, vfov=VFOV_DEG,
              reference_focal_px=RENDER_REFERENCE_FOCAL_PX):
    """Habitat intermediate sensor resolution [H, W]."""
    w, h = render_resolution(hfov, vfov, reference_focal_px)
    return [h, w]


def height_tag(h: float) -> str:
    """Compact tag for a camera height, for sensor uuids / filenames (0.4 -> '040')."""
    return f"{int(round(h * 100)):03d}"


def intrinsics(hfov=HFOV_DEG, vfov=VFOV_DEG):
    """3x3 pinhole K for a fixed-size raster and independently specified FOV."""
    import numpy as np
    w, h = resolution()
    fx = (w / 2.0) / math.tan(math.radians(float(hfov)) / 2.0)
    fy = (h / 2.0) / math.tan(math.radians(float(vfov)) / 2.0)
    return np.array([[fx, 0.0, w / 2.0],
                     [0.0, fy, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def is_structural(category: str) -> bool:
    """True if a raw category string names a structural surface."""
    c = (category or "").strip().lower()
    return any(tok in c for tok in STRUCTURAL_CATEGORIES)


def is_specific_semantic_category(category: str) -> bool:
    """True when ``category`` names a publishable, recognizable class."""
    normalized = str(category or "").strip().casefold()
    return bool(normalized) and normalized not in NON_SPECIFIC_SEMANTIC_CATEGORIES


