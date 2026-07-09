"""Frozen constants — single source of truth for the consequence pipeline.

Do NOT scatter these literals elsewhere; import from here.
Values mirror the validated egoconseq design (camera, voxel, march) plus the
new consequence-pipeline knobs (FOV half-angle, geodesic, view-exit).
"""

import math

# --- data / scene pool ---
HM3D_ROOT = "/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d"
HM3D_VAL_DIR = HM3D_ROOT + "/val"
HM3D_SCENE_DATASET_CFG = HM3D_ROOT + "/hm3d_annotated_basis.scene_dataset_config.json"

# --- sensor (Habitat defaults) ---
HFOV_DEG = 79.0
RESOLUTION = (640, 480)           # (W, H)
CAMERA_HEIGHT_M = 1.5
FOV_HALF_DEG = HFOV_DEG / 2.0     # 39.5 deg — view-cone half-angle

# --- body (robot chassis footprint radii; edit here to change the tested bodies) ---
# Real home chassis robots span ~0.12-0.25 m radius (Ø24-50 cm):
# robot vacuum ~0.17, Amazon Astro ~0.21, Temi ~0.22, Pepper ~0.24.
CYLINDER_HEIGHT_M = 1.5
RADII_M = (0.15, 0.20, 0.25)

# --- perception / occupancy ---
OBSTACLE_BAND_M = (0.05, 1.5)     # height-above-floor band counted as obstacle
VOXEL_SIZE_M = 0.05
VOXEL_DILATION = 1                # dilation iterations
MIN_SUPPORT_VOXELS = 3            # occupied voxels within footprint -> contact

# --- action geometry / march ---
MARCH_STEP_M = 0.02               # path sample spacing
D_MAX_M = 5.0                     # cap when marching for "no contact" (nav/collision)

# --- attribution ---
ATTR_MARGIN_M = 0.10              # extra radius around contact for semantic vote
NAV_Y_DELTA_M = 0.5               # is_navigable Y-snap tolerance
NAV_AGREE_TOL_M = 0.3             # depth-vs-navmesh agreement tolerance

# --- view exit ---
END_VISIBLE_DEPTH_TOL_M = 0.3     # projected-pixel depth slack for end_visible

# --- action-combination generation (length-bucketed, in-FOV) ---
GEN_TURNS_DEG = (-30, -15, 15, 30)   # bounded turns: one turn+forward stays in 39.5 cone
GEN_FORWARDS_M = (0.5, 1.0, 1.5)     # forward step lengths (m)
GEN_LENGTHS = (1, 2, 3, 4, 5, 6)     # number of primitive actions per sequence
KEEP_PER_LENGTH = 5                  # kept sequences per (frame, length); CLI override
POOL_FACTOR = 3                      # candidate pool per length = KEEP_PER_LENGTH * this

# --- refined nav_check (depth-vs-navmesh, artifact-free) ---
FWD_COVER_MIN = 0.90                 # forward-cone valid-depth ratio below which -> review
NAV_STRONG_SUPPORT = 2 * MIN_SUPPORT_VOXELS  # solid depth obstacle (keep_visible) vs noise

# --- object extraction filters ---
OBJ_MIN_AREA_PX = 400
OBJ_MIN_VALID_DEPTH = 20
OBJ_MAX_POINTS = 2000             # transient subsample cap for points_xz

# --- semantic decode (this habitat build cannot render HM3D texture semantics,
#     so we decode instance ids offline from semantic.glb; see pipeline/semantic.py)
SEMANTIC_CACHE_DIR = "data/conseq/semantic_cache"
SEMANTIC_SAMPLE_COUNT = 1_000_000  # surface samples per scene (labelled subset kept)
SEMANTIC_MATCH_TOL = 12.0          # face-colour vs palette L2 tolerance (0-255)
SEMANTIC_ASSIGN_TOL_M = 0.20       # depth point -> nearest labelled surface cutoff

# HM3D free-text category substrings flagged as structural (flag only, not dropped).
STRUCTURAL_CATEGORIES = frozenset({
    "wall", "floor", "ceiling", "door", "window", "stairs", "staircase",
    "handrail", "railing", "beam", "column", "pillar", "unknown",
})


def hw():
    """Habitat sensor resolution [H, W] from RESOLUTION=(W, H)."""
    w, h = RESOLUTION
    return [h, w]


def intrinsics():
    """3x3 pinhole K from HFOV and RESOLUTION (level camera)."""
    import numpy as np
    w, h = RESOLUTION
    fx = (w / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    return np.array([[fx, 0.0, w / 2.0],
                     [0.0, fx, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def is_structural(category: str) -> bool:
    """True if the (raw HM3D) category string names a structural surface."""
    c = (category or "").strip().lower()
    return any(tok in c for tok in STRUCTURAL_CATEGORIES)
