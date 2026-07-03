"""Generate O5 (footprint counterfactual) dataset cases.

For each accepted pose across 3 HM3D val scenes:
  1. Render RGB + depth.
  2. Build obstacle VoxelField (backproject → to_agent_ground → estimate_floor_height
     → remove_floor → VoxelField).
  3. Compute d_safe_visible for r=0.10 and r=0.40.
  4. Compute d_safe_navmesh for both radii (recompute_navmesh per radius).
  5. Apply disagreement.classify for BOTH radii — keep only if BOTH verdict.keep=True.
  6. Choose horizon H to target flip / all_no_contact / all_contact groups.
  7. Emit one binary-contact case per radius, sharing group_id.
  8. Save:  data/demo/o5/img/{case_id}.png
            data/demo/o5/topdown/{case_id}.png
  9. Write  data/demo/o5/manifest.jsonl

Usage:
    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \\
        python scripts/generate_o5.py [--max-poses N] [--scenes S1 S2 S3]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import uuid

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq import config
from egoconseq.sim.habitat_env import EgoConseqSim
from egoconseq.sim.navmesh import recompute_navmesh, d_safe_navmesh
from egoconseq.oracle.pointcloud import (
    backproject,
    to_agent_ground,
    estimate_floor_height,
    remove_floor,
)
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible
from egoconseq.oracle.disagreement import classify as classify_disagreement
from egoconseq.pipeline.sample_poses import is_valid_start, MIN_CLEARANCE_M
from egoconseq.tasks.instantiate import make_o5_case
from egoconseq.manifest import Case, write_jsonl
from egoconseq.geometry import swept_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_SMALL = 0.10  # narrow body radius
R_LARGE = 0.40  # wide body radius
W_SMALL = 2 * R_SMALL  # width 0.20 m
W_LARGE = 2 * R_LARGE  # width 0.80 m

# Discriminative band condition for a GT flip group:
#   d_safe_large + R_LARGE <= H <= d_safe_small - R_SMALL
#   i.e. gap >= R_SMALL + R_LARGE = 0.50 m
MIN_GAP_FOR_FLIP = R_SMALL + R_LARGE  # 0.50 m

# H offsets outside the band for non-flip control groups.
ALL_NO_CONTACT_H_FRACTION = 0.80
ALL_CONTACT_MARGIN_M = 0.10

# Default 3 scenes from HM3D val
DEFAULT_SCENES = [
    "/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
    "/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb",
    "/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb",
]

OUT_DIR = os.path.join(_REPO_ROOT, "data", "demo", "o5")
IMG_DIR = os.path.join(OUT_DIR, "img")
TOPDOWN_DIR = os.path.join(OUT_DIR, "topdown")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.jsonl")


def clear_o5_output_dirs(img_dir: str = IMG_DIR, topdown_dir: str = TOPDOWN_DIR) -> None:
    """Remove stale generated PNGs before a fresh O5 run."""
    for directory in (img_dir, topdown_dir):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.lower().endswith(".png"):
                os.remove(os.path.join(directory, name))


# ---------------------------------------------------------------------------
# Top-down plot helper (O5 version: one robot + H marker)
# ---------------------------------------------------------------------------

def save_topdown_o5(
    out_path: str,
    obstacle_pts: np.ndarray,
    path_samples: list,
    d_safe_m: float,
    horizon_m: float,
    radius_m: float,
    gt_label: str,
    title: str = "",
    step: float = config.MARCH_STEP_M,
) -> None:
    """Save a top-down plot showing evidence for one single-robot O5 case."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Obstacle cloud (x, z)
    obstacle_pts = np.asarray(obstacle_pts)
    if obstacle_pts.size > 0 and obstacle_pts.shape[0] > 0:
        n = obstacle_pts.shape[0]
        idx = np.random.choice(n, min(n, 8000), replace=False)
        ax.scatter(
            obstacle_pts[idx, 0], obstacle_pts[idx, 2],
            s=1, c="steelblue", alpha=0.4, label="obstacle pts",
        )

    path_limit = min(d_safe_m, horizon_m)
    disk_label = (
        f"first contact (d={d_safe_m:.2f}m)"
        if gt_label == "contact"
        else f"clear endpoint (H={horizon_m:.2f}m)"
    )
    _plot_body_path(
        ax,
        path_samples,
        path_limit,
        radius_m,
        path_color="tomato" if gt_label == "contact" else "limegreen",
        disk_color="tomato" if gt_label == "contact" else "limegreen",
        path_label=f"robot path r={radius_m:.2f}m",
        disk_label=disk_label,
        step=step,
    )

    # Horizon H marker (horizontal dashed line at z = horizon_m)
    ax.axhline(y=horizon_m, color="purple", linestyle="--", linewidth=2,
               label=f"H = {horizon_m:.2f}m", zorder=5)

    # Agent at origin
    ax.plot(0, 0, "k^", markersize=10, label="agent")

    ax.set_xlabel("x (right)  [m]")
    ax.set_ylabel("z (forward)  [m]")
    ax.set_title(
        f"{title}\nGT={gt_label}  radius={radius_m:.2f}  "
        f"d_safe={d_safe_m:.2f}  H={horizon_m:.2f}"
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_body_path(
    ax,
    path_samples: list,
    d_safe: float,
    radius: float,
    path_color: str,
    disk_color: str,
    path_label: str,
    disk_label: str,
    step: float,
) -> None:
    """Draw path centerline up to d_safe and a disk at the contact point."""
    cx_list, cz_list = [], []
    for i, (x, z, _h) in enumerate(path_samples):
        arc = i * step
        if arc > d_safe + 1e-9:
            break
        cx_list.append(x)
        cz_list.append(z)

    if cx_list:
        ax.plot(cx_list, cz_list, color=path_color, linewidth=1.5,
                linestyle="-", label=path_label)

    # Contact disk at last point
    if cx_list:
        cx, cz = cx_list[-1], cz_list[-1]
        circle = plt.Circle(
            (cx, cz), radius,
            color=disk_color, fill=False, linewidth=2, label=disk_label,
        )
        ax.add_patch(circle)
        ax.plot(cx, cz, "x", color=disk_color, markersize=8, markeredgewidth=2)


# ---------------------------------------------------------------------------
# Per-pose processing
# ---------------------------------------------------------------------------

def process_pose(
    sim: EgoConseqSim,
    pos: np.ndarray,
    yaw: float,
    scene_id: str,
    case_id: str,
    turn_deg: float = 0.0,
) -> tuple:
    """Run full oracle pipeline for one pose.

    Returns (rgb, result_small, result_large, d_nav_small, d_nav_large,
             pts_obs, floor_y, K) or raises ValueError on rejection.
    """
    # Render
    rgb, depth, K, _ = sim.render(pos, yaw)

    # Validity check
    valid_mask = np.isfinite(depth) & (depth > 0)
    vdr = float(valid_mask.mean())
    floor_mask = valid_mask & (depth < 4.0)
    vfr = float(floor_mask.mean())
    dist = sim.pathfinder.distance_to_closest_obstacle(pos)

    ok, reason = is_valid_start(
        valid_depth_ratio=vdr,
        dist_to_obstacle=dist,
        visible_floor_ratio=vfr,
    )
    if not ok:
        raise ValueError(f"invalid start: {reason}")

    # Point cloud → obstacle VoxelField
    pts_cam = backproject(depth, K)
    pts_ground = to_agent_ground(pts_cam, camera_height=config.CAMERA_HEIGHT_M)
    floor_y = estimate_floor_height(pts_ground)
    pts_obs = remove_floor(pts_ground, floor_y, band=config.OBSTACLE_BAND_M)
    vf = VoxelField(pts_obs)

    # d_safe_visible for both radii
    result_small = d_safe_visible(vf, radius=R_SMALL, turn_deg=turn_deg)
    result_large = d_safe_visible(vf, radius=R_LARGE, turn_deg=turn_deg)

    # d_safe_navmesh for both radii — recompute navmesh per radius
    recompute_navmesh(sim.sim, radius=R_SMALL)
    d_nav_small = d_safe_navmesh(sim.pathfinder, pos, yaw, turn_deg=turn_deg)

    recompute_navmesh(sim.sim, radius=R_LARGE)
    d_nav_large = d_safe_navmesh(sim.pathfinder, pos, yaw, turn_deg=turn_deg)

    # Restore navmesh to default radius for next sampling
    recompute_navmesh(sim.sim, radius=max(config.RADII_M))

    return rgb, result_small, result_large, d_nav_small, d_nav_large, pts_obs, floor_y, K


def disagree_check(
    d_depth: float,
    d_nav: float,
    vf: VoxelField,
    r: float,
    d_nav_val: float,
) -> bool:
    """Return True if this radius-specific observation should be kept."""
    # contact_visible: is the depth contact point within visible range?
    # Simple proxy: if d_safe_visible < D_MAX, contact was seen in the frame
    contact_visible = (d_depth < config.D_MAX_M)
    verdict = classify_disagreement(
        d_depth=d_depth,
        d_nav=d_nav,
        contact_visible=contact_visible,
    )
    return verdict.keep


# ---------------------------------------------------------------------------
# Horizon selection logic
# ---------------------------------------------------------------------------

def choose_horizon(
    d_safe_small: float,
    d_safe_large: float,
    target_counts: dict,
) -> tuple | None:
    """Choose (horizon_m, group_kind) to balance O5 group targets.

    Returns None if we cannot generate a valid case for any needed label.

    Args:
        target_counts: dict mapping group_kind -> (target, current_count).
            We try to fill the group kind that is most behind target.
    """
    # --- flip candidate ---
    # Need: d_safe_large + R_LARGE <= H <= d_safe_small - R_SMALL
    lo_flip = d_safe_large + R_LARGE
    hi_flip = d_safe_small - R_SMALL
    has_flip_band = (hi_flip - lo_flip) >= 0.01

    # --- all_no_contact control ---
    # H well below the large-body contact boundary, so both radii clear H.
    h_all_no_contact = ALL_NO_CONTACT_H_FRACTION * (d_safe_large - R_LARGE)
    has_all_no_contact = h_all_no_contact > 0.05

    # --- all_contact control ---
    # H beyond the contact boundary for both radii with each radius margin.
    h_all_contact = max(
        d_safe_small + R_SMALL,
        d_safe_large + R_LARGE,
    ) + ALL_CONTACT_MARGIN_M
    has_all_contact = h_all_contact < config.D_MAX_M

    # Priority: fill the label furthest below its target
    label_priority = []
    for lbl, (target, current) in target_counts.items():
        deficit = target - current
        label_priority.append((deficit, lbl))
    label_priority.sort(reverse=True)  # largest deficit first

    for _, lbl in label_priority:
        target_val, current_val = target_counts[lbl]
        if current_val >= target_val:
            continue  # already met target for this label

        if lbl == "flip" and has_flip_band:
            H = (lo_flip + hi_flip) / 2.0
            return H, "flip"
        elif lbl == "all_no_contact" and has_all_no_contact:
            return h_all_no_contact, "all_no_contact"
        elif lbl == "all_contact" and has_all_contact:
            return h_all_contact, "all_contact"

    # If all targets met, prefer more flip groups.
    if has_flip_band:
        H = (lo_flip + hi_flip) / 2.0
        return H, "flip"
    if has_all_no_contact:
        return h_all_no_contact, "all_no_contact"
    if has_all_contact:
        return h_all_contact, "all_contact"

    return None


def build_o5_group_cases(
    group_id: str,
    case_prefix: str,
    scene_id: str,
    pose: list[float],
    horizon_m: float,
    group_kind: str,
    d_vis_small: float,
    d_vis_large: float,
    d_nav_small: float,
    d_nav_large: float,
) -> list[Case] | None:
    """Build the two single-robot binary-contact cases for one O5 group.

    The group shares one image/pose/action horizon.  Each returned case
    describes exactly one cylindrical-chassis robot; no prompt compares the
    two body sizes.
    """
    geometry_tag = "narrow-gap" if group_kind == "flip" else group_kind
    specs = [
        ("r010", R_SMALL, d_vis_small, d_nav_small),
        ("r040", R_LARGE, d_vis_large, d_nav_large),
    ]

    cases: list[Case] = []
    for suffix, radius_m, d_vis, d_nav in specs:
        case = make_o5_case(
            d_safe_m=d_vis,
            radius_m=radius_m,
            horizon_m=horizon_m,
            group_id=group_id,
            case_id=f"{case_prefix}-{suffix}",
            scene_id=scene_id,
            pose=list(pose),
            geometry_tag=geometry_tag,
        )
        if case is None:
            return None

        case.d_safe_navmesh_m = d_nav
        case.tags.update({
            "group_kind": group_kind,
            "d_safe_small_m": d_vis_small,
            "d_safe_large_m": d_vis_large,
            "d_safe_navmesh_small_m": d_nav_small,
            "d_safe_navmesh_large_m": d_nav_large,
            "radii_in_group_m": [R_SMALL, R_LARGE],
        })
        cases.append(case)

    return cases


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_o5_cases(
    scenes: list[str],
    max_poses_per_scene: int = 400,
    target_flip: int = 40,
    target_all_no_contact: int = 20,
    target_all_contact: int = 20,
    seed: int = 42,
) -> list[Case]:
    """Generate O5 cases across multiple scenes."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(TOPDOWN_DIR, exist_ok=True)
    clear_o5_output_dirs()

    cases: list[Case] = []
    # Track how many counterfactual groups of each kind we have.
    target_counts = {
        "flip": [target_flip, 0],
        "all_no_contact": [target_all_no_contact, 0],
        "all_contact": [target_all_contact, 0],
    }

    total_poses_tried = 0
    total_kept_after_disagree = 0
    group_counts = {"flip": 0, "all_no_contact": 0, "all_contact": 0}
    case_label_counts = {"contact": 0, "no_contact": 0}
    scene_stats = []

    for scene_glb in scenes:
        scene_name = os.path.basename(os.path.dirname(scene_glb))
        print(f"\n[O5-gen] === Scene: {scene_name} ===")

        scene_poses_tried = 0
        scene_kept = 0
        scene_groups = {"flip": 0, "all_no_contact": 0, "all_contact": 0}
        scene_labels = {"contact": 0, "no_contact": 0}

        try:
            sim = EgoConseqSim(scene_glb)
        except Exception as e:
            print(f"  [WARN] Failed to load scene: {e}")
            continue

        try:
            # Recompute navmesh with largest radius for pose sampling
            recompute_navmesh(sim.sim, radius=max(config.RADII_M))

            for attempt in range(max_poses_per_scene):
                # Check if we have enough of everything
                total_needed = sum(
                    max(0, t - c)
                    for t, c in target_counts.values()
                )
                if total_needed == 0:
                    # All targets met — still try to add extra flip groups if possible.
                    if group_counts["flip"] >= target_flip + 10:
                        break

                # Sample a random pose
                pos = sim.pathfinder.get_random_navigable_point()
                yaw = random.uniform(0, 2 * math.pi)
                scene_poses_tried += 1
                total_poses_tried += 1

                # Process the pose
                try:
                    (rgb, res_small, res_large,
                     d_nav_small, d_nav_large,
                     pts_obs, floor_y, K) = process_pose(
                        sim, pos, yaw, scene_name, "", turn_deg=0.0
                    )
                except ValueError:
                    continue

                d_vis_small = res_small.d_safe
                d_vis_large = res_large.d_safe

                # Disagreement check for BOTH radii
                keep_small = disagree_check(d_vis_small, d_nav_small, None, R_SMALL, d_nav_small)
                keep_large = disagree_check(d_vis_large, d_nav_large, None, R_LARGE, d_nav_large)

                if not (keep_small and keep_large):
                    continue

                # Monotonicity check
                if d_vis_large > d_vis_small + 0.01:
                    # Physics violation — skip
                    continue

                total_kept_after_disagree += 1
                scene_kept += 1

                # Choose horizon
                tc = {k: (v[0], v[1]) for k, v in target_counts.items()}
                result = choose_horizon(d_vis_small, d_vis_large, tc)
                if result is None:
                    continue

                horizon_m, group_kind = result

                group_uid = uuid.uuid4().hex[:8]
                group_id = f"O5G-{group_uid}"
                case_prefix = f"O5-{group_uid}"
                group_cases = build_o5_group_cases(
                    group_id=group_id,
                    case_prefix=case_prefix,
                    horizon_m=horizon_m,
                    scene_id=scene_name,
                    pose=[float(pos[0]), float(pos[1]), float(pos[2]), float(yaw)],
                    group_kind=group_kind,
                    d_vis_small=d_vis_small,
                    d_vis_large=d_vis_large,
                    d_nav_small=d_nav_small,
                    d_nav_large=d_nav_large,
                )

                if group_cases is None:
                    # Margin check failed for at least one radius.
                    continue

                for case in group_cases:
                    # Save the same RGB image under each case id so model payloads stay per-case.
                    img_path = os.path.join(IMG_DIR, f"{case.case_id}.png")
                    Image.fromarray(rgb).save(img_path)
                    case.image_path = img_path

                    # Save one single-robot evidence plot for this case.
                    td_path = os.path.join(TOPDOWN_DIR, f"{case.case_id}.png")
                    case_d_safe = float(case.d_safe_visible_m)
                    case_radius = float(case.body["radius_m"])
                    case_label = case.answer["label"]
                    path_samples = swept_path(
                        0.0,
                        forward_m=min(case_d_safe, horizon_m),
                        step=config.MARCH_STEP_M,
                    )
                    save_topdown_o5(
                        td_path,
                        obstacle_pts=pts_obs,
                        path_samples=path_samples,
                        d_safe_m=case_d_safe,
                        horizon_m=horizon_m,
                        radius_m=case_radius,
                        gt_label=case_label,
                        title=f"{scene_name[:20]} | {case.case_id}",
                    )

                    actual_label = case_label
                    case_label_counts[actual_label] = case_label_counts.get(actual_label, 0) + 1
                    scene_labels[actual_label] = scene_labels.get(actual_label, 0) + 1

                group_counts[group_kind] = group_counts.get(group_kind, 0) + 1
                scene_groups[group_kind] = scene_groups.get(group_kind, 0) + 1
                if group_kind in target_counts:
                    target_counts[group_kind][1] += 1

                cases.extend(group_cases)

                if (scene_kept % 5 == 0) or scene_kept <= 3:
                    print(
                        f"  [kept {scene_kept:3d}] attempt={attempt+1:4d}"
                        f"  d_small={d_vis_small:.2f}  d_large={d_vis_large:.2f}"
                        f"  H={horizon_m:.2f}  group={group_kind}"
                    )

        finally:
            sim.close()

        scene_stats.append({
            "scene": scene_name,
            "poses_tried": scene_poses_tried,
            "kept_after_disagree": scene_kept,
            "groups": scene_groups,
            "labels": scene_labels,
        })
        print(
            f"  Scene summary: tried={scene_poses_tried}  kept={scene_kept}"
            f"  groups={scene_groups}  case_labels={scene_labels}"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Total poses tried       : {total_poses_tried}")
    print(f"  Kept after disagree     : {total_kept_after_disagree}")
    print(f"  Cases generated         : {len(cases)}")
    print(f"  Groups generated        : {sum(group_counts.values())}")
    print(f"  flip groups             : {group_counts['flip']}")
    print(f"  all_no_contact groups   : {group_counts['all_no_contact']}")
    print(f"  all_contact groups      : {group_counts['all_contact']}")
    print(f"  contact cases           : {case_label_counts['contact']}")
    print(f"  no_contact cases        : {case_label_counts['no_contact']}")
    print(f"  flip groups >= 30?      : {'YES' if group_counts['flip'] >= 30 else 'NO'}")
    print("=" * 60)
    for ss in scene_stats:
        print(
            f"  {ss['scene']}: tried={ss['poses_tried']}"
            f"  kept={ss['kept_after_disagree']}  groups={ss['groups']}  labels={ss['labels']}"
        )
    print("=" * 60)

    return cases


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate O5 dataset cases")
    p.add_argument("--max-poses", type=int, default=500,
                   help="Max pose attempts per scene (default 500)")
    p.add_argument("--target-flip", type=int, default=40,
                   help="Target counterfactual flip group count (default 40)")
    p.add_argument("--target-all-no-contact", type=int, default=20,
                   help="Target control groups where both radii do not contact")
    p.add_argument("--target-all-contact", type=int, default=20,
                   help="Target control groups where both radii contact")
    # Backward-compatible aliases from the previous three-way O5 schema.
    p.add_argument("--target-small-only", dest="target_flip", type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--target-both", dest="target_all_no_contact", type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--target-neither", dest="target_all_contact", type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("[O5-gen] Starting O5 dataset generation")
    print(f"  Scenes : {len(DEFAULT_SCENES)}")
    print(f"  Max poses per scene : {args.max_poses}")
    print(
        f"  Targets: flip={args.target_flip}"
        f"  all_no_contact={args.target_all_no_contact}"
        f"  all_contact={args.target_all_contact}"
    )

    cases = generate_o5_cases(
        scenes=DEFAULT_SCENES,
        max_poses_per_scene=args.max_poses,
        target_flip=args.target_flip,
        target_all_no_contact=args.target_all_no_contact,
        target_all_contact=args.target_all_contact,
        seed=args.seed,
    )

    if not cases:
        print("[O5-gen] ERROR: no cases generated!")
        sys.exit(1)

    write_jsonl(cases, MANIFEST_PATH)
    print(f"\n[O5-gen] manifest written: {MANIFEST_PATH}  ({len(cases)} cases)")


if __name__ == "__main__":
    main()
