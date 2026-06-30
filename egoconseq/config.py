"""Frozen constants — single source mirrors docs/2026-06-30-egoconseq-bench-v1-design.md.
Do NOT scatter these literals elsewhere; import from here."""

# --- data ---
HM3D_ROOT = "/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d"
HM3D_VAL_DIR = HM3D_ROOT + "/val"
HM3D_SCENE_DATASET_CFG = HM3D_ROOT + "/hm3d_annotated_basis.scene_dataset_config.json"

# --- sensor (design §3) ---
HFOV_DEG = 79
RESOLUTION = (640, 480)          # (W, H)
CAMERA_HEIGHT_M = 1.5            # Habitat default agent sensor height

# --- body (design §3) ---
CYLINDER_HEIGHT_M = 1.5
RADII_M = (0.10, 0.25, 0.40)
OBSTACLE_BAND_M = (0.05, 1.5)    # height-above-floor band counted as obstacle

# --- actions / oracle (design §3/§4) ---
FORWARD_STEP_M = 0.25
MARCH_STEP_M = 0.02
TURN_ANGLES_DEG = (-15, 0, 15)   # ±30 optional, not in core demo
H_BODYWIDTHS = (1, 2, 4, 6)      # O1/O3 horizon unit
D_MAX_M = 5.0                    # cap for "no contact within visible local space"

# --- voxel oracle (tuned in P1.5; placeholders) ---
VOXEL_SIZE_M = 0.05
VOXEL_DILATION = 1               # voxels
MIN_SUPPORT_VOXELS = 3

# --- gates (design §5) ---
GATE_VISIBLE_SWEEP_RATIO = 0.7
GATE_VALID_DEPTH_RATIO = 0.9
GATE_DEPTH_HOLE_RATIO = 0.05
MARGIN_BODYWIDTHS = 0.5
STEP_SIZE_STABILITY = 0.95

# --- categories (design §2) ---
OPERATIONS = ("O1", "O2", "O3", "O4", "O5", "O6")
HEADLINE_OPS = ("O4", "O5")

def hw():
    """Habitat sensor resolution [H, W] from RESOLUTION=(W,H)."""
    w, h = RESOLUTION
    return [h, w]
