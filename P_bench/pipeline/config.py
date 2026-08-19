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
GS_TRAIN_MANIFEST = os.environ.get(
    "EGOCONSEQ_GS_TRAIN_MANIFEST",
    os.path.join(GS_ROOT, "splits", "train.json"))
# Offline source-quality ranking only. These values never certify or reject
# an action and deliberately do not reuse a physical-oracle tolerance.
GS_SUPPORT_RADIUS_QUALITY_M = 0.30
GS_SUPPORT_RADIUS_OUTLIER_FRACTION = 0.05

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
# Persisted height and floor calibration are two serializations of the same
# already-computed identity, so only float round-off is allowed between them.
CAMERA_HEIGHT_CALIBRATION_TOLERANCE_M = 1e-9


def vfov_for_hfov(hfov_deg: float, *, width: int = _BASE_WIDTH,
                  height: int = _BASE_HEIGHT) -> float:
    """Vertical FOV for a square-pixel pinhole camera at ``width:height``."""
    return math.degrees(2.0 * math.atan(
        math.tan(math.radians(float(hfov_deg)) / 2.0) *
        float(height) / float(width)))

# Formal benchmark observation profiles.  Camera parameters are independent
# sensor variables; they never alter the ground-disc physical oracle.
BENCH_CAMERA_HEIGHTS_M = (0.5, 1.0, 1.5)
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
# Deterministic compression for per-entity initial-depth ground support.  This
# is a record representation parameter, not the navigation voxel resolution,
# even though both currently use five-centimetre cells.
INITIAL_VISIBLE_SUPPORT_GRID_CELL_M = 0.05
GROUND_ORACLE_HEIGHT_M = GROUND_OBSTACLE_BAND_M[1]
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
# Contact attribution samples the same signed-normal height band as obstacle
# extraction. The representative projected point is the band's midpoint, not
# half of the full geometry oracle height.
CONTACT_ATTRIBUTION_PROBE_COUNT = 7
CONTACT_ATTRIBUTION_HEIGHT_M = sum(GROUND_OBSTACLE_BAND_M) / 2.0
CONTACT_SURFACE_RADIUS_TOL_M = 1e-4
# Habitat's own horizontal tolerance inside isNavigable and snapPoint,
# mirrored rather than chosen. Raising it reclassifies genuine lateral
# collisions -- this benchmark's primary signal -- as invalid geometry.
NAVMESH_LATERAL_SNAP_MAX_M = 0.01
NAVMESH_SNAP_EPSILON_M = 1e-6

# --- perception / occupancy ---
# Compatibility is intentionally not provided for the former camera-coupled
# occupancy band.  All current code must use GROUND_OBSTACLE_BAND_M.
VOXEL_SIZE_M = 0.05
VOXEL_DILATION = 1                # dilation iterations
MIN_SUPPORT_VOXELS = 3            # occupied voxels within footprint -> contact

# --- action geometry / march ---
MARCH_STEP_M = 0.02               # path sample spacing
D_MAX_M = 5.0                     # cap when marching for "no contact" (nav/collision)
CHECKPOINT_PROGRESS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTACT_REFINE_ITERS = 14
# A point may sit up to NAVMESH_LATERAL_SNAP_MAX_M outside the eroded polygon
# and still read as navigable, so the bisection converges on that tolerance
# contour rather than on the polygon edge. The centre-to-boundary distance at
# first contact is therefore about one tolerance plus the bisection residual;
# a threshold near zero would withhold almost every reconstructed contact surface.
CONTACT_NAVMESH_BOUNDARY_MAX_M = (
    NAVMESH_LATERAL_SNAP_MAX_M
    + MARCH_STEP_M / 2 ** CONTACT_REFINE_ITERS
    + CONTACT_SURFACE_RADIUS_TOL_M)
LINEAR_SPEED_M_S = 0.5
ANGULAR_SPEED_DEG_S = 45.0
# --- attribution ---
ATTR_MARGIN_M = 0.10              # extra radius around contact for semantic vote
NAV_Y_DELTA_M = 0.5               # is_navigable Y-snap tolerance

# --- current-view evidence ---
DEPTH_SUPPORT_TOL_M = 0.3         # projected-pixel depth slack for corridor support

# --- balanced action-combination generation ---
GEN_TURNS_DEG = (-45, -30, -15, 15, 30, 45)
# Forward legs are 0.5 m multiples only, so every action is a clean metric value
# (no arbitrary/near-boundary distances). Turn/forward strictly alternate, so no
# two consecutive same-direction primitives can occur (enforced in actions.py).
#
# v1 stopped at 3.0 m, which conflated "the grid step is 0.5 m" with "the
# longest publishable leg is 3 m". Only the first is a task requirement.
# Measured on R2R: depth boundaries land at 3.66-5.02 m often enough that the
# cap -- not the scene -- accounted for 41% of rejected candidate pairs
# (bracket_grid_exhausted_collision), and lifting it tripled the pair yield.
# v2 keeps the 0.5 m multiple and raises the ceiling to the sensing horizon.
# Which subset of it a pose may actually use is decided per prefix from that
# prefix's RGB-D, FOV and body radius -- never from this table. v1 is a subset
# of v2, so frozen v1 programs stay inside the published vocabulary.
GEN_FORWARDS_M_V1 = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
GEN_FORWARDS_M_V2 = tuple(round(0.5 * (step + 1), 3) for step in range(12))
GEN_FORWARDS_M = GEN_FORWARDS_M_V2
GEN_LENGTHS = (1, 2, 3, 4, 5, 6)

# --- background collection capacity, quota, and controller closeout
BACKGROUND_MIN_TOTAL_ITEMS = 6000
BACKGROUND_MIN_ITEMS_PER_TASK = 1000
BACKGROUND_MAX_SCENE_WALLCLOCK_S = 21600
BACKGROUND_CANARY_ACCEPTED_RECORDS = 20
BACKGROUND_CANARY_POSE_ATTEMPT_CAP_BY_DATASET = {
    "r2r": 800,
    "gs": 800,
    "b1k": 6000,
}
BACKGROUND_SCENE_WALLCLOCK_S_BY_DATASET = {
    "r2r": 600,
    "gs": 600,
    "b1k": 900,
}
BACKGROUND_RECORD_IDLE_STOP_S = 120.0
BACKGROUND_SIGINT_GRACE_S = 30.0
BACKGROUND_SIGTERM_GRACE_S = 30.0
BACKGROUND_CAPACITY_HANDOFF_GRACE_S = 120.0
BACKGROUND_FINALIZATION_GRACE_S = 30.0
BACKGROUND_HEARTBEAT_INTERVAL_S = 30
BACKGROUND_INITIALIZATION_DEADLINE_S = 600
BACKGROUND_FIRST_RECORD_DEADLINE_S = BACKGROUND_RECORD_IDLE_STOP_S
BACKGROUND_RETRYABLE_INITIALIZATION_ATTEMPTS = 1
BACKGROUND_MAX_IDLE_GPU_MEMORY_MIB = 500
BACKGROUND_MIN_FREE_STORAGE_BYTES = 120 * 1024 ** 3
BACKGROUND_MIN_ITEMS_PER_LENGTH = 1
BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET = {
    "r2r": 15000,
    "gs": 15000,
    "b1k": 6000,
}
BACKGROUND_MIN_HEADLINE_ITEMS_GLOBAL = 160000
BACKGROUND_OVERFLOW_DATASET = "r2r"
BACKGROUND_MAX_A4_ITEMS = 60000
BACKGROUND_MIN_SCENE_FAMILIES_BY_DATASET = {
    "r2r": 12,
    "gs": 12,
    "b1k": 12,
}
BACKGROUND_MAX_SCENE_FAMILY_FRACTION = {
    "r2r": 0.10,
    "gs": 0.10,
    "b1k": 0.10,
}
BACKGROUND_GS_CATALOG_EXCLUSIONS = {
    "interior_0505_839970": "missing_authenticated_collision_artifact",
}
BACKGROUND_GS_CATALOG_SCENE_COUNT = 54

# Natural per-pose candidate selection and private do(action) matching.
# Candidate selection is deliberately label-blind; balance belongs to the
# artifact compiler, never to a physical pose.
ACTION_CANDIDATE_MIN_PER_POSE = 2
# The total action bank that may pay for full geometry, so it also bounds the
# published ``candidate_budget``. Proposal v4 divides it into the ordinary
# shortlist and the bounded C1 second pass below.
ACTION_CANDIDATE_MAX_PER_POSE = 48
# Certify a reserve wider than the published ordinary arm. Stability is known
# only after the seven frozen perturbations, so trying exactly 36 would make
# every unstable group reduce the record instead of letting a later,
# label-blind shortlist member take its place.
ACTION_CANDIDATE_ORDINARY_ATTEMPTS_PER_POSE = 48
ACTION_CANDIDATE_ORDINARY_PER_POSE = 36
ACTION_CANDIDATE_ORDINARY_LADDER = (18, 24, 36)
# C1 counterfactual slots are taken from that same total budget rather than
# added on top. Two queries come from the ordinary shortlist itself; only their
# neighbours consume additional certification and terminal-rendering work.
C1_QUERIES_PER_POSE = 2
C1_NEIGHBORS_PER_QUERY = 6
C1_NEIGHBOR_SLOTS_PER_POSE = (
    C1_QUERIES_PER_POSE * C1_NEIGHBORS_PER_QUERY)
ACTION_CANDIDATE_CERTIFICATION_MAX_PER_POSE = (
    ACTION_CANDIDATE_ORDINARY_ATTEMPTS_PER_POSE +
    C1_NEIGHBOR_SLOTS_PER_POSE)
ACTION_MATCH_DISTANCE_BUCKET_M = 0.25
ACTION_MATCH_TURN_BUCKET_DEG = 15.0
# Label-blind natural-arm allocation within each action length. The distance
# bands are thirds of the per-pose depth-supported Forward grid; the weights
# produce 10 short, 10 mid, and 20 near-reach offers at the default budget 40.
NATURAL_DYNAMIC_STRATUM_WEIGHTS = {"short": 1, "mid": 1, "near": 2}

COLLECTION_MODES = ("main",)
CLI_COLLECTION_MODES = COLLECTION_MODES

KEEP_PER_LENGTH = 1                  # formal group per (frame, length)
POOL_FACTOR = 40                     # strict consensus needs a broad pre-screened pool
MAIN_ACTION_PROPOSAL_PER_LENGTH = 40
ACTION_REJECTION_FACTOR = 200        # random draws allowed per requested candidate
ORACLE_CONTACT_TOL_M = 0.30          # max full/depth first-contact disagreement
# A2 deliberately reuses the dual-oracle arc tolerance as its stricter
# publication margin around cumulative Forward-action boundaries.  A contact
# inside this band could change public action index while still satisfying the
# physical oracle's frozen arc agreement gate.
A2_ACTION_BOUNDARY_MARGIN_M = ORACLE_CONTACT_TOL_M
# --- GS runtime geometry and rendering ---
GS_SIGMA = 2.0                       # conservative ellipsoid support radius
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
TARGET_VISIBLE_MIN_PX = 20
TARGET_GROUND_SUPPORT_MIN_POINTS = 5
# v16 B selects exactly one target from initial depth-backed object facts.
B_TARGET_SELECTION_PROTOCOL = "b-s0-target-selection.v1"
B_TARGET_MIN_DEPTH_BACKED_AREA_RATIO = 0.01
B_TARGET_MIN_BBOX_WIDTH_RATIO = 0.08
B_TARGET_MIN_BBOX_HEIGHT_RATIO = 0.08
B_TARGET_VISIBLE_DISTANCE_RANGE_M = (1.0, 5.0)
B_TARGET_EXCLUDED_MATERIAL_TOKENS = frozenset({
    "glass", "mirror", "reflective", "transparent",
})
B_ENDPOINT_DISTANCE_MIN_M = 0.05

# The terminal checkpoint and independently accumulated realized endpoint are
# the same analytic pose reached through different floating-point paths.  The
# dimensioned tolerances accept only arithmetic dust; neither is a rendering,
# collision, or family-diversity threshold.
C1_TERMINAL_POSITION_TOL_M = 1e-9
C1_TERMINAL_HEADING_TOL_DEG = 1e-9
B_DISTANCE_CHANGE_MIN_M = 0.30
B_DISTANCE_CHANGE_MIN_RATIO = 0.10
B_DIRECTION_BOUNDARY_MARGIN_DEG = 15.0
B_TARGET_CENTROID_MIN_RANGE_M = 0.05
# Task-GT diagnostics are measured before QA publication policy is chosen.
# This threshold has its own semantics: it limits how far the complete-mesh
# B1 nearest witness may lie from the initial visible support.  Its numerical
# equality to SEMANTIC_ASSIGN_TOL_M is intentional (both are seeded at the
# current depth/semantic uncertainty scale), but neither constant aliases the
# other and future calibration may move them independently.
MP3D_TARGET_FACE_INDEX_CACHE_MAX_SCENES = 2
MP3D_TARGET_FACE_INDEX_CACHE_MAX_BYTES = 512 * 1024 * 1024
MP3D_TARGET_FACE_INDEX_STREAM_BYTES = 1024 * 1024
MP3D_AABB_MADVISE_INTERVAL_CHUNKS = 8
# Conservative absolute slack used only by the AABB pruning decision.  Exact
# triangle distances, not this value, determine the published top two.
A3_FACE_AABB_PRUNE_SLACK_M = 1e-9
TRUSTED_R2R_SCENE_CACHE_MAX_SCENES = 4
B_DISTANCE_CHOICE_PROTOCOL = "b1-metric-choices-rank-balanced.v3"
B_DISTANCE_CHOICE_DECIMALS = 2
B_DISTANCE_CHOICE_MIN_SEPARATION_M = 0.25
B_DISTANCE_DISPLAY_ERROR_TOLERANCE_M = 0.006
# Publication-only bins used to avoid keeping many numerically different B1
# questions with the same coarse near/mid/far meaning from one source frame.
DIVERSITY_B1_DISTANCE_BINS_M = (1.0, 2.0)
EVIDENCE_COVERAGE_MIN = 0.90
EVIDENCE_CORRIDOR_LATERAL_SAMPLES = 5
EVIDENCE_MIN_LATERAL_FRACTION = 1.00
EVIDENCE_TERMINAL_LENGTH_M = 0.50
EVIDENCE_MAX_UNSUPPORTED_RUN_M = 0.00
EVIDENCE_MAX_INVALID_DEPTH_SAMPLES = 1
# P0-4 canonical floor plane. The histogram window and bin width seed the
# candidate band ONLY; the published plane comes from the continuous fit, so no
# 0.05 m quantum reaches a published height.
FLOOR_SEED_WINDOW_M = (-0.25, 0.30)
FLOOR_HISTOGRAM_BIN_M = 0.05
FLOOR_CANDIDATE_HALF_BAND_M = 0.10
# Seeds are multi-hypothesis and scored by ground COVERAGE, not raw point count:
# a small dense platform holds more points than a large sparse floor, and a band
# chosen around it filters the real floor out before any robust fit can run.
FLOOR_SEED_HYPOTHESES = 3
FLOOR_SEED_MIN_SEPARATION_M = 0.10
# Two hypotheses of comparable coverage at clearly separated heights are declined
# rather than silently resolved -- a mezzanine or stair landing looks like this.
FLOOR_AMBIGUOUS_COVERAGE_RATIO = 0.90
FLOOR_AMBIGUOUS_SEPARATION_M = 0.03
# Deterministic multi-start concentration fit. The subset is a fixed share of the
# FULL candidate set, reselected each step; several fixed starts are run and the
# best LTS objective wins, because one all-point start can be captured by a
# high-leverage cluster.
# 0.50 is the maximum-breakdown LTS choice, and it is needed: at 0.35 a corner
# platform holding 36% of the band forces every candidate subset to include
# non-floor points, so the trimmed objective genuinely prefers a tilted plane and
# no tie-break can recover the floor.
FLOOR_TRIM_FRACTION = 0.50        # share the concentration subset may exclude
FLOOR_FIT_ITERATIONS = 5          # fixed count -> deterministic, no convergence race
FLOOR_INLIER_TOL_M = 0.02         # residual within which a point supports the plane
FLOOR_SUPPORT_CELL_M = 0.10
# Frozen after the 2026-07-25 multi-dataset 15-scene x 40-pose pilot and
# Gaussian-splat scenes (code revision 57ece20). The pilot measured yield rather
# than wrong-fit labels, so it did not justify relaxing the conservative values
# that passed the synthetic contamination sweeps with zero wrong fits.
FLOOR_MIN_SUPPORT = 200
FLOOR_MIN_INLIER_RATIO = 0.60
# Share of the candidate band's ground cells the accepted plane must explain.
# Above the 50% breakdown point the fit is dragged into a compromise that fits
# part of the floor and part of the contaminant, which leaves whole regions
# unexplained -- the only signal that separates it from a correct fit whose band
# merely happens to contain clutter.
FLOOR_MIN_INLIER_COVERAGE_RATIO = 0.85
# RMSE is measured over the inliers, which are selected at FLOOR_INLIER_TOL_M, so
# this gate is arithmetically unreachable unless it stays BELOW that tolerance.
FLOOR_MAX_INLIER_RMSE_M = 0.010
FLOOR_MAX_TILT_DEG = 5.0
FLOOR_MIN_SUPPORT_EXTENT_M = 0.60
FLOOR_MIN_SUPPORT_CELLS = 40
# Frame-composition gates use geometry only. Texture or blur heuristics would
# introduce a backend-dependent appearance bias into the benchmark.
COLLECT_MIN_VALID_DEPTH_RATIO = 0.95
COLLECT_MIN_VISIBLE_FLOOR_RATIO = 0.08
COLLECT_VISIBLE_FLOOR_REFERENCE_HEIGHT_M = 0.80
# Pose discovery is an input-population budget, not an answer or witness
# quota. Each scene must expose enough independently sampled positions and
# headings before a rare task-specific witness may be declared unavailable.
POSE_CANDIDATES_PER_SCENE = 100


def collect_min_visible_floor_ratio(camera_height_above_floor_m: float) -> float:
    """Scale the composition gate without penalizing taller camera views."""
    height = max(float(camera_height_above_floor_m), 1e-6)
    return (COLLECT_MIN_VISIBLE_FLOOR_RATIO *
            COLLECT_VISIBLE_FLOOR_REFERENCE_HEIGHT_M / height)

# --- benchmark publication margins ---
# These define the release set, not the geometric oracle resolution. Samples
# closer to a decision boundary remain valid records but are not turned into
# benchmark questions.
BENCH_COLLISION_REMAINING_M = 0.50
BENCH_SAFE_CLEARANCE_M = 0.30
# A3's diagnostic asks whether at least one *wrong category* is close to the
# swept prefix in the initial visible support.  This is an independent seed,
# not the natural safe-publication margin despite their current equal values.

# The body-intervention slice is deliberately a separate near-boundary track.
# A 5 cm radius change can never flip a natural safe item whose configuration-
# space clearance is at least 30 cm.  Two centimetres is one path-march step;
# the frozen A perturbation certificate remains the actual robustness gate.

# Two poses in one scene count as the same observation when they are within
# BOTH of these. One source, because the collector and the threshold pilot must
# reproduce each other's population from the same exclusion payload: while the
# pilot held its own 1.0 m / 30 deg against the collector's 1.5 / 45, a
# candidate 1.2 m from an excluded pose was kept by one and dropped by the other.
POSE_DIVERSITY_POSITION_M = 1.5
POSE_DIVERSITY_YAW_DEG = 45.0
BENCH_RADIUS_CLEARANCE_MONOTONIC_TOL_M = 0.02

# --- offline semantic geometry and runtime assignment ---
SEMANTIC_CACHE_DIR = "data/conseq/semantic_cache"
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
A3_MARKER_RADIUS_PX = 8
A3_MARKER_MIN_SEPARATION_PX = 20

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
NON_CONTACT_GROUND_CATEGORIES = frozenset({
    "floor", "flooring", "ground", "rug", "area rug", "carpet",
    "carpeting", "mat", "floor mat", "doormat", "ceiling",
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


def calibration_profile() -> tuple:
    """The one sensor profile the canonical floor plane is ever fitted from.
    Fixed, and deliberately independent of which profiles a run publishes:
    deriving it from the published list would recalibrate every scene whenever
    the published set changed, so the same pose would report a different
    height in two runs.
    """
    return (CAMERA_HEIGHT_M, HFOV_DEG, VFOV_DEG)


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


def is_contact_obstacle_category(category: str) -> bool:
    """Whether a semantic class can physically block the planar chassis."""
    normalized = " ".join(str(category or "").strip().casefold().split())
    return (
        is_specific_semantic_category(normalized) and
        normalized not in NON_CONTACT_GROUND_CATEGORIES
    )
