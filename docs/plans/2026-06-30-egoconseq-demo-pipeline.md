# EgoConseq-Bench Demo Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the EgoConseq-Bench data-generation + evaluation vertical slice (P0→P5) that produces a demo dataset and runs the four validity gates, yielding the go/no-go decision for the full benchmark.

**Architecture:** Deterministic geometry pipeline on top of Habitat-Sim. A scene is loaded, a pose is sampled, RGB+depth rendered; depth is back-projected to a point cloud, floor-removed, voxelized → obstacle field; a swept-cylinder oracle marches the footprint along piecewise action paths to get `d_safe_visible` + contact point; per-radius navmesh gives `d_safe_navmesh` for a disagreement-taxonomy cross-check; visibility/sanity gates filter cases; surviving `d_safe` tables instantiate O1/O3/O4/O5/O6 QA; baselines + per-operation metrics run the validity gates. The model only ever sees `RGB + question`.

**Tech Stack:** Python 3 (conda env `qwen3vl_habitat`), habitat-sim 0.2.4 (`PathFinder`, `NavMeshSettings`, RGB/Depth sensors), numpy, scipy (KDTree/binary_dilation), pytest. VLM baseline via the existing qwen3vl serving (deferred to P3+).

**Design source of truth (FROZEN v1.0):** `P_bench/docs/2026-06-30-egoconseq-bench-v1-design.md`. Do NOT hardcode any constant that lives in `egoconseq/config.py`; all category/threshold/sensor/body values derive from there.

**Key frozen constants (from design §3/§5):**
- HM3D val: `/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/`, dataset cfg `.../hm3d/hm3d_annotated_basis.scene_dataset_config.json`
- Sensor: 640×480, HFOV **79°**, camera height **1.5m** (Habitat default agent), depth same res (offline only)
- Body: fixed-height **visible cylinder**, `cylinder_height = 1.5m`; radii `{0.10, 0.25, 0.40} m` (width = 2r)
- Actions: `forward`, `turn ±15° then forward` (±30° optional tag); forward step 0.25m semantic, **march step 0.02m**
- Horizon H: O1/O3 in body-widths `{1,2,4,6}`; **O5 horizon in fixed meters** (red line — same physical path across bodies)
- Vertical obstacle band: height-above-floor `[0.05, 1.5] m`; min support `N` voxels (tune in P1.5)
- Gates: `visible_sweep_ratio ≥ 0.7`, `valid_depth_ratio ≥ 0.9`, `depth_hole_ratio ≤ 0.05`, margin `|d_safe−H| ≥ 0.5 bw`, ranking `top1−top2 ≥ 0.5 bw`, monotonicity, step-size stability ≥ 95%

---

## Conventions (READ FIRST — resolves cross-module frame/shape bugs)

**Canonical coordinate frame** — every obstacle point, floor estimate, voxel, and swept path MUST live in ONE frame:
- **Agent-local ground frame**: origin at the agent's base footprint center on the floor; **+z = forward, +x = right, +y = up**; units = meters.
- Habitat camera (`render`) returns depth in a camera frame. `pointcloud.backproject` produces camera-frame points; `pointcloud.to_agent_ground(pts, camera_height=1.5, pitch=0)` converts them into the agent-local ground frame above (translate down by camera height, apply camera→agent rotation; demo cameras are level so pitch=0). **All downstream code (floor removal, VoxelField, sweep) consumes only agent-ground-frame points.** `VoxelField` and `swept_path` operate on the 2D `(x, z)` ground projection; the obstacle band filter uses `y` (height above floor).
- Floor height is estimated in this frame (≈ y=0); `OBSTACLE_BAND_M = (0.05, 1.5)` is height-above-floor in `y`.

**Resolution shape**: `config.RESOLUTION = (640, 480)` is `(W, H)`. Habitat `CameraSensorSpec.resolution` wants `[H, W]`. Use the helper `config.hw()` → `[480, 640]` when configuring sensors and when building intrinsics. NEVER pass `RESOLUTION` straight into Habitat (silent transpose → rotated FOV + wrong intrinsics).

**D_MAX semantics**: `d_safe == D_MAX_M` means "no contact within the visible local horizon" (treated as `> D_max`). Thresholds (`d_safe < H`) and capped-MAE treat the capped value as the upper bin edge; never compare two capped values as if precise.

**VLM is mandatory for the P2/P3 gate decisions** (it is the content of the life-or-death gates), accessed via `eval/baselines.py::vlm_answer(image, question)` wrapping the repo's qwen3vl serving. Generation tasks stay decoupled from serving details, but a gate is NOT "passed" until the VLM has been run.

---

## File Structure

```
P_bench/
  egoconseq/
    __init__.py
    config.py                 # frozen constants mirrored from design doc (single import point)
    geometry.py               # pure math: poses, headings, piecewise path, projection helpers
    manifest.py               # Case dataclass + jsonl read/write + answer schema
    sim/
      habitat_env.py          # scene load, agent+sensor config, render(pose)->(rgb,depth,K,T)
      navmesh.py              # per-radius navmesh build + march -> d_safe_navmesh
    oracle/
      pointcloud.py           # depth->cloud, floor estimate+removal
      voxel.py                # voxelize + dilation + support query
      sweep.py                # swept-cylinder march -> d_safe_visible + contact 3d/pixel
      disagreement.py         # depth-vs-navmesh taxonomy -> keep/discard/review
    gates/
      visibility.py           # G1 visible_sweep_ratio / valid_depth_ratio / depth_hole_ratio
      sanity.py               # G3 margin, G4 monotonicity, step-size stability
    tasks/
      prompts.py              # width-only prompt templates (no height)
      instantiate.py          # d_safe table -> O1/O3/O4/O5/O6 cases
    eval/
      baselines.py            # random/majority/blind/radius-only/center-ray-depth
      metrics.py              # capped-MAE, bin-acc, False-Safe Rate, ranking, O5 flip metrics
    pipeline/
      sample_poses.py         # stratified pose sampling + per-pose validity
      generate.py             # end-to-end orchestration -> manifest + overlays
  tests/                      # pytest unit tests mirroring egoconseq/
  data/demo/                  # output: images, overlays, manifest.jsonl, reports
  docs/plans/                 # this file
```

Each oracle/gate/eval module is pure-Python and unit-tested in isolation (no Habitat). `sim/` and `pipeline/generate.py` are integration glue verified by smoke gates (the design's "每加一层一个 smoke gate").

---

## Phase Map (maps to design appendix P0→P5)

- **P0** — Tasks 1–6: project skeleton, config, Habitat render, point cloud + floor removal. Smoke: one RGB + cloud looks right.
- **P1** — Tasks 7–11: voxel field, swept oracle `d_safe(forward)`, overlay. Smoke: `d_safe` matches a hand-checked image.
- **P1.5** — Tasks 12–14: navmesh + disagreement taxonomy + sanity (monotonicity, step-size, voxel-param tuning).
- **P2** — Tasks 15–18: **O5 counterfactual (life-or-death gate 1)** — narrow-gap pairs, fixed-meter horizon, flip metrics.
- **P3** — Tasks 19–21: **O4 directional ranking (life-or-death gate 2)** + center-ray depth baseline.
- **P4** — Tasks 22–24: O1 magnitude + O3 turn-then-forward.
- **P5** — Tasks 25–28: scale to ≥5 scenes, full baseline suite, four validity gates, demo report + decision.

---

## P0 — Skeleton, Config, Render, Point Cloud

### Task 1: Project skeleton + pytest

**Files:**
- Create: `P_bench/egoconseq/__init__.py` (empty), `P_bench/tests/__init__.py` (empty)
- Create: `P_bench/pytest.ini`, `P_bench/README.md`

- [ ] **Step 1: Create package dirs and `__init__.py` files**

```bash
cd /home/zhangshan/syp/myvln/P_bench
mkdir -p egoconseq/{sim,oracle,gates,tasks,eval,pipeline} tests data/demo
touch egoconseq/__init__.py egoconseq/sim/__init__.py egoconseq/oracle/__init__.py \
      egoconseq/gates/__init__.py egoconseq/tasks/__init__.py egoconseq/eval/__init__.py \
      egoconseq/pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Sanity test that the package imports**

```python
# tests/test_smoke_import.py
def test_import_package():
    import egoconseq  # noqa: F401
```

- [ ] **Step 4: Run**

Run: `cd /home/zhangshan/syp/myvln/P_bench && /home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python -m pytest tests/test_smoke_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit** (init git first — P_bench is not yet a repo)

```bash
cd /home/zhangshan/syp/myvln/P_bench && git init -q && \
printf "data/demo/\n__pycache__/\n*.pyc\n.pytest_cache/\n" > .gitignore && \
git add -A && git commit -q -m "chore: egoconseq package skeleton + pytest"
```

---

### Task 2: Frozen config module

**Files:**
- Create: `egoconseq/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from egoconseq import config as C

def test_frozen_constants_match_design():
    assert C.HFOV_DEG == 79
    assert C.RESOLUTION == (640, 480)
    assert C.CAMERA_HEIGHT_M == 1.5
    assert C.CYLINDER_HEIGHT_M == 1.5
    assert C.RADII_M == (0.10, 0.25, 0.40)
    assert C.MARCH_STEP_M == 0.02
    assert C.OBSTACLE_BAND_M == (0.05, 1.5)
    assert C.GATE_VISIBLE_SWEEP_RATIO == 0.7
    assert C.GATE_VALID_DEPTH_RATIO == 0.9
    assert C.GATE_DEPTH_HOLE_RATIO == 0.05
    assert C.MARGIN_BODYWIDTHS == 0.5
    assert C.HM3D_VAL_DIR.endswith("hm3d-0.2/hm3d/val")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.../python -m pytest tests/test_config.py -v` → FAIL (no module)

- [ ] **Step 3: Implement `egoconseq/config.py`**

```python
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
```

Also add `assert C.hw() == [480, 640]` to `test_config.py`.

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: frozen config mirroring design v1.0"`

---

### Task 3: Pure geometry helpers

**Files:**
- Create: `egoconseq/geometry.py`
- Test: `tests/test_geometry.py`

Responsibility: heading math + piecewise swept path generation (NO Habitat, NO depth). A path is a list of `(x, z, heading)` centerline samples spaced `MARCH_STEP_M`, in the agent's local ground frame (agent at origin, +z forward, +x right).

- [ ] **Step 1: Failing test**

```python
# tests/test_geometry.py
import numpy as np
from egoconseq.geometry import swept_path

def test_forward_path_length_and_heading():
    pts = swept_path(turn_deg=0, forward_m=1.0, step=0.02)
    assert abs(pts[0][0]) < 1e-9 and abs(pts[0][1]) < 1e-9      # starts at origin
    assert abs(pts[-1][1] - 1.0) < 0.02                         # ends ~1m forward (+z)
    assert all(abs(p[2] - 0.0) < 1e-9 for p in pts)            # heading unchanged

def test_turn_then_forward_heading():
    pts = swept_path(turn_deg=15, forward_m=1.0, step=0.02)
    h = np.deg2rad(15)
    assert all(abs(p[2] - h) < 1e-9 for p in pts)              # all samples carry new heading
    # final point displaced along rotated heading
    assert pts[-1][0] > 0 and pts[-1][1] > 0
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement `swept_path`**

```python
import numpy as np

def swept_path(turn_deg: float, forward_m: float, step: float):
    """Return list of (x, z, heading_rad) centerline samples in agent-local ground frame.
    Turn is in-place (footprint unchanged), then translate forward along new heading.
    +z = forward, +x = right."""
    h = np.deg2rad(turn_deg)
    n = max(1, int(round(forward_m / step)))
    out = [(0.0, 0.0, h)]
    for i in range(1, n + 1):
        d = i * step
        out.append((d * np.sin(h), d * np.cos(h), h))
    return out
```

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: swept_path geometry helper"`

---

### Task 4: Manifest schema

**Files:**
- Create: `egoconseq/manifest.py`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_manifest.py
from egoconseq.manifest import Case, write_jsonl, read_jsonl

def test_roundtrip(tmp_path):
    c = Case(case_id="x1", operation_id="O5", readout_tag="pair_flip",
             scene_id="s", pose=[0,0,0,0], body={"radius_m":0.25},
             action={"type":"forward","horizon_m":1.0,"horizon_reference":"metric_fixed"},
             d_safe_visible_m=1.4, d_safe_navmesh_m=1.5, oracle_agreement="agree",
             gates={"visible_sweep_ratio":0.82}, answer={"answer_type":"pair_flip","label":"small_only"},
             tags={"geometry_tag":"narrow-gap"}, group_id="g1")
    p = tmp_path/"m.jsonl"
    write_jsonl([c], p)
    got = read_jsonl(p)
    assert got[0].operation_id == "O5" and got[0].answer["label"] == "small_only"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** `Case` as a `@dataclass` with `to_dict`/`from_dict`, `write_jsonl`/`read_jsonl` using `json`. Include every field in design §8.2 metadata example. Model-visible payload = `{image_path, question}` only; everything else is offline.
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: Case manifest schema + jsonl io"`

---

### Task 5: Habitat env — render(pose) (INTEGRATION, smoke-gated)

**Files:**
- Create: `egoconseq/sim/habitat_env.py`
- Create: `scripts/smoke_render.py`

Responsibility: configure a `habitat_sim.Simulator` with the **default agent (height 1.5m, radius 0.1m)** + RGB & Depth `CameraSensor` at 1.5m, HFOV 79°, 640×480; `render(position, yaw)` returns `rgb (H,W,3) uint8`, `depth (H,W) float32 meters`, intrinsics `K (3,3)`, and world→camera pose.

- [ ] **Step 1: Implement `habitat_env.py`** — `EgoConseqSim(scene_glb)` with `.render(pos, yaw)`, `.pathfinder`. Set `CameraSensorSpec.hfov = 79`, `resolution=[480,640]`, `position=[0,1.5,0]`. Build intrinsics from hfov+resolution.

- [ ] **Step 2: Smoke script** `scripts/smoke_render.py`: load first val scene, sample a navigable point via `pathfinder.get_random_navigable_point()`, render, save `data/demo/smoke_rgb.png` + `data/demo/smoke_depth.png` (normalized), print depth min/max/valid-ratio.

- [ ] **Step 3: Run the smoke gate**

Run:
```bash
cd /home/zhangshan/syp/myvln/P_bench && \
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/smoke_render.py \
  --scene /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
```
Expected: writes two PNGs; depth valid-ratio > 0.8; **human eyeballs `smoke_rgb.png` — it is a plausible indoor first-person view, floor visible ahead.**

- [ ] **Step 4: Commit** `git commit -am "feat: habitat env render + smoke gate (P0)"`

> ⚠ GATE P0a: do not proceed until the rendered RGB looks correct and depth is metric (meters, not normalized). This is also where we confirm 1.5m camera gives enough forward floor visibility (design open-risk note).

---

### Task 6: Depth → point cloud + floor removal

**Files:**
- Create: `egoconseq/oracle/pointcloud.py`
- Test: `tests/test_pointcloud.py`

- [ ] **Step 1: Failing test (synthetic depth, no Habitat)**

```python
# tests/test_pointcloud.py
import numpy as np
from egoconseq.oracle.pointcloud import backproject, estimate_floor_height, remove_floor

def test_backproject_center_ray_distance():
    H, W = 4, 4
    depth = np.full((H, W), 2.0, np.float32)
    K = np.array([[100,0,W/2],[0,100,H/2],[0,0,1]], float)
    pts = backproject(depth, K)                       # (N,3) camera frame, +z forward
    assert np.isclose(pts[:,2].min(), 2.0, atol=1e-5)

def test_floor_removed_keeps_wall():
    # floor points at y≈0, wall points at y in [0.05,1.5]
    floor = np.array([[x*0.1, 0.0, 1.0] for x in range(20)])
    wall  = np.array([[0.0, h, 2.0] for h in np.linspace(0.1,1.4,20)])
    pts = np.vstack([floor, wall])
    fh = estimate_floor_height(pts)
    kept = remove_floor(pts, fh, band=(0.05,1.5))
    assert len(kept) == len(wall)                     # floor gone, wall kept
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** (in the canonical frame from Conventions):
  - `backproject(depth, K) -> (N,3)` camera-frame points.
  - `to_agent_ground(pts_cam, camera_height=config.CAMERA_HEIGHT_M, pitch=0.0) -> (N,3)` agent-local ground frame (+z fwd, +x right, +y up): apply camera→agent rotation (level cam ⇒ identity up to axis relabel) then translate `y -= camera_height`. **Add a test** asserting a camera point straight ahead at depth d maps to agent-ground `(x≈0, z≈d, y≈0)` (floor level) for a level camera.
  - `estimate_floor_height(pts)` = robust low-percentile / histogram mode of `y`.
  - `remove_floor(pts, floor_y, band)` keeps points with `band[0] ≤ (y-floor_y) ≤ band[1]` (default band = `config.OBSTACLE_BAND_M`).
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: pointcloud backproject + floor removal"`

---

## P1 — Voxel Field + Swept Oracle

### Task 7: Obstacle voxel field

**Files:** Create `egoconseq/oracle/voxel.py`; Test `tests/test_voxel.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np
from egoconseq.oracle.voxel import VoxelField

def test_support_query_dilation_and_min_support():
    pts = np.array([[0.5,0.2,1.0]]*5)                 # 5 coincident points -> support
    vf = VoxelField(pts, voxel=0.05, dilation=1)
    assert vf.support_count(center=(0.5,1.0), radius=0.1) >= 1
    assert vf.support_count(center=(3.0,3.0), radius=0.1) == 0
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** `VoxelField`: voxelize ground-plane projection (x,z) of obstacle pts; `scipy.ndimage.binary_dilation` by `dilation`; `support_count(center,radius)` = occupied voxels whose center lies within `radius` of `center`. (Keep a KDTree of occupied voxel centers for the radius query.)
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: obstacle VoxelField with dilation + support query"`

---

### Task 8: Swept-cylinder oracle d_safe_visible

**Files:** Create `egoconseq/oracle/sweep.py`; Test `tests/test_sweep.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np
from egoconseq.oracle.voxel import VoxelField
from egoconseq.oracle.sweep import d_safe_visible

def test_wall_ahead_gives_finite_contact():
    wall = np.array([[x, 0.3, 1.0] for x in np.linspace(-1,1,200)])   # wall at z=1.0
    vf = VoxelField(wall, voxel=0.05, dilation=1)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0,
                         step=0.02, min_support=1)
    assert 0.7 < res.d_safe < 1.1
    assert res.contact_xy is not None

def test_open_space_returns_dmax():
    vf = VoxelField(np.zeros((0,3)), voxel=0.05, dilation=0)
    res = d_safe_visible(vf, radius=0.10, turn_deg=0, d_max=5.0, step=0.02, min_support=1)
    assert res.d_safe >= 5.0 and res.contact_xy is None

def test_larger_radius_contacts_no_later():
    # monotonicity sanity at the unit level
    wall = np.array([[x, 0.3, 1.0] for x in np.linspace(-1,1,200)])
    vf = VoxelField(wall, voxel=0.05, dilation=1)
    small = d_safe_visible(vf, 0.10, 0, 5.0, 0.02, 1).d_safe
    large = d_safe_visible(vf, 0.40, 0, 5.0, 0.02, 1).d_safe
    assert large <= small + 1e-6
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** `d_safe_visible(...)`: build path via `geometry.swept_path`; for each centerline sample, `vf.support_count(center, radius) ≥ min_support` → contact at that arc length, record `contact_xy`; else `d_safe = d_max`. Return dataclass `SweepResult(d_safe, contact_xy)`.
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: swept-cylinder d_safe_visible oracle"`

---

### Task 9: Contact point → 3D + pixel projection

**Files:** Modify `egoconseq/oracle/sweep.py`; add `project_contact` to `geometry.py`; Test extend `tests/test_geometry.py`

- [ ] **Step 1: Failing test** for `project_contact(contact_xy_local, floor_y, K, agent_pose)` → returns pixel `(u,v)` and 3D; assert a contact directly ahead projects near image horizontal center.
- [ ] **Step 2-4: Implement + run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: contact 3d + pixel projection (O6 substrate)"`

> Note: this builds the O6 substrate only. O6 (first-contact grounding) is "诊断（可选）" in the spec — **no O6 cases are generated in the demo**; deferred to the full benchmark plan.

---

### Task 10: Overlay renderer (debug/diagnostic)

**Files:** Create `egoconseq/oracle/overlay.py`; Create `scripts/smoke_oracle.py`

- [ ] **Step 1: Implement** `draw_sweep_overlay(rgb, path_pixels, contact_pixel)` → draws the swept corridor footprint + contact marker on the RGB.
- [ ] **Step 2: Smoke** `scripts/smoke_oracle.py`: render a real HM3D pose, run full chain (cloud→floor→voxel→d_safe forward r=0.25), save `data/demo/oracle_overlay.png` + print `d_safe`.
- [ ] **Step 3: Run smoke gate** → human checks overlay: corridor lies on the floor, contact marker sits on the first real obstacle, `d_safe` number matches what the eye sees.
- [ ] **Step 4: Commit** `git commit -am "feat: sweep overlay + P1 oracle smoke gate"`

> ⚠ GATE P1: overlay + d_safe must visually agree on a hand-picked frame before building tasks on top.

---

### Task 11: Stratified pose sampling

**Files:** Create `egoconseq/pipeline/sample_poses.py`; Test `tests/test_sample_poses.py` (pure-logic parts)

- [ ] **Step 1: Failing test** for per-pose validity predicate `is_valid_start(pathfinder_stub, depth, ...)`: rejects vanishing valid-depth, rejects too-close-to-wall (use a fake pathfinder with `distance_to_closest_obstacle`).
- [ ] **Step 2-4: Implement + run** → PASS. Sampler yields poses navigable for **all radii** (needed for counterfactual), with enough visible floor and valid depth.
- [ ] **Step 5: Commit** `git commit -am "feat: stratified pose sampling + validity predicate"`

---

## P1.5 — Navmesh Cross-Check + Sanity

### Task 12: Per-radius navmesh d_safe

**Files:** Create `egoconseq/sim/navmesh.py`; Create `scripts/smoke_navmesh.py`

- [ ] **Step 1: Implement** `recompute_navmesh(sim, radius, height=config.CYLINDER_HEIGHT_M)` via `NavMeshSettings(agent_radius=radius, agent_height=height)` + `sim.recompute_navmesh(...)` (pull height from config, do not hardcode 1.5); `d_safe_navmesh(pathfinder, pos, yaw, turn_deg, d_max, step)` marches centerline, stops at first `not is_navigable`.
- [ ] **Step 2: Smoke** on a real pose: print `d_safe_navmesh` for r∈{0.10,0.25,0.40}; assert monotonic non-increasing in r.
- [ ] **Step 3: Run** → monotonic; values plausible.
- [ ] **Step 4: Commit** `git commit -am "feat: per-radius navmesh d_safe cross-check"`

---

### Task 13: Disagreement taxonomy

**Files:** Create `egoconseq/oracle/disagreement.py`; Test `tests/test_disagreement.py`

- [ ] **Step 1: Failing test** encoding design §4 taxonomy:

```python
from egoconseq.oracle.disagreement import classify

def test_taxonomy():
    # navmesh earlier + contact NOT visible -> hidden geometry, discard
    assert classify(d_depth=5.0, d_nav=1.0, contact_visible=False).verdict == "discard_hidden"
    # navmesh earlier + visible -> depth miss
    assert classify(5.0, 1.0, contact_visible=True).verdict == "review_depth_miss"
    # depth earlier + no clear visible obstacle -> noise
    assert classify(1.0, 5.0, contact_visible=False).verdict == "discard_noise"
    # depth earlier + clear visible obstacle -> keep visible obstacle
    assert classify(1.0, 5.0, contact_visible=True).verdict == "keep_visible"
    # agree -> high confidence
    assert classify(1.45, 1.5, contact_visible=True).verdict == "keep_agree"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** `classify(d_depth,d_nav,contact_visible,tol=0.3)` returning `Verdict(verdict, keep: bool)`.
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: depth-vs-navmesh disagreement taxonomy"`

---

### Task 14: Sanity gates + voxel-param tuning

**Files:** Create `egoconseq/gates/sanity.py`; Create `scripts/tune_voxel_params.py`; Test `tests/test_sanity.py`

- [ ] **Step 1: Failing test** for `step_size_stable(labels_a, labels_b, thresh=0.95)` and `monotonic_in_radius(dsafe_by_radius)`.
- [ ] **Step 2-4: Implement + run** → PASS.
- [ ] **Step 5: Tuning script** `scripts/tune_voxel_params.py`: on ~5 poses, sweep `(voxel_size, dilation, min_support)`; report step-size stability (0.02 vs 0.01) and false-contact rate on known-open corridors; pick values; **write chosen values back into `config.py`** and commit.
- [ ] **Step 6: Commit** `git commit -am "feat: sanity gates + tuned voxel params"`

> ⚠ GATE P1.5: monotonicity holds on sampled poses AND step-size stability ≥ 95% before mass generation.

---

### Task 14b: Visibility gates G1/G2 (the most label-corrupting gate)

**Files:** Create `egoconseq/gates/visibility.py`; Test `tests/test_visibility.py`

> Spec §5 calls G2 the hard gate "最易造假标签" — it MUST exist before any O5/O4/O1/O3 case is accepted in P2+. Without it, no-contact labels are silently corrupted by occlusion ("看不见 ≠ 没东西").

- [ ] **Step 1: Failing test** (synthetic depth + projected corridor, no Habitat)

```python
import numpy as np
from egoconseq.gates.visibility import corridor_visibility, passes_g2

def test_ratios_on_clean_corridor():
    depth = np.full((480,640), 3.0, np.float32)          # all-valid depth
    corridor_px = [(320, v) for v in range(240, 460)]    # column of pixels inside frame
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert r["valid_depth_ratio"] > 0.95
    assert r["depth_hole_ratio"] < 0.05
    assert r["visible_sweep_ratio"] > 0.95

def test_g2_rejects_holey_no_contact():
    depth = np.full((480,640), 3.0, np.float32)
    depth[240:460, 300:340] = 0.0                        # invalid/hole down the corridor
    corridor_px = [(320, v) for v in range(240, 460)]
    r = corridor_visibility(depth, corridor_px, d_max=5.0)
    assert not passes_g2(r)                               # no-contact label must be rejected
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
  - `corridor_visibility(depth, corridor_pixels, d_max)` → dict with:
    - `visible_sweep_ratio` = fraction of swept-corridor footprint sample points that project inside the image frame,
    - `valid_depth_ratio` = fraction of in-frame corridor pixels with valid (nonzero, finite, < d_max+margin) depth,
    - `depth_hole_ratio` = fraction of in-frame corridor pixels that are invalid,
    - `occlusion_free_ratio` = fraction of corridor centerline samples whose **rendered depth ≥ the sample's along-ray distance** (i.e. nothing nearer occludes the queried point); a sample whose pixel depth is much smaller than its 3D distance is occluded.
  - `passes_g1(r)` = `visible_sweep_ratio ≥ config.GATE_VISIBLE_SWEEP_RATIO`.
  - `passes_g2(r)` = `valid_depth_ratio ≥ 0.9 AND depth_hole_ratio ≤ 0.05 AND occlusion_free_ratio ≥ threshold AND endpoint in-frame`. **Apply G2 only to no-contact labels; for contact labels require the contact pixel in-frame + depth-consistent (Task 9).**
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: G1/G2 visibility + no-contact evidence gate"`

> ⚠ Wire G1/G2 into every `generate_*` (Tasks 18/21/24): a case is accepted only if disagreement-keep AND G1 AND (G2 for no-contact / contact-visible for contact).

---

## P2 — O5 Counterfactual (LIFE-OR-DEATH GATE 1)

### Task 15: O5 prompt + answer schema

**Files:** Create `egoconseq/tasks/prompts.py`; Test `tests/test_prompts.py`

- [ ] **Step 1: Failing test**: `o5_prompt(width_small_m, width_large_m, horizon_m)` mentions both widths + a **fixed metric** horizon, mentions "身体接触"/"宽", and does **NOT** mention height. Assert `"高" not in prompt and "米" in prompt`.
- [ ] **Step 2-4: Implement + run** → PASS. Answer schema `{answer_type:"pair_flip", options:[both,small_only,neither], label, group_id}`.
- [ ] **Step 5: Commit** `git commit -am "feat: O5 width-only prompt + pair-flip schema"`

---

### Task 16: O5 instantiation (fixed-meter horizon)

**Files:** Create `egoconseq/tasks/instantiate.py` (`make_o5`); Test `tests/test_instantiate_o5.py`

- [ ] **Step 1: Failing test** — the red line:

```python
from egoconseq.tasks.instantiate import make_o5, o5_label

def test_o5_uses_fixed_metric_horizon_not_bodywidth():
    # same physical path for both bodies
    case = make_o5(d_safe_small=1.2, d_safe_large=0.6, horizon_m=1.0,
                   r_small=0.10, r_large=0.40)
    assert case.action["horizon_reference"] == "metric_fixed"
    assert case.action["horizon_m"] == 1.0
    assert o5_label(1.2, 0.6, 1.0) == "small_only"     # small passes (1.2>1.0), large contacts (0.6<1.0)
    assert o5_label(1.2, 1.1, 1.0) == "both"
    assert o5_label(0.5, 0.4, 1.0) == "neither"
```

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** `o5_label(d_small,d_large,H)` (compare each `d_safe` to the **same** fixed-meter `H`) and `make_o5(...)` → `Case` with `operation_id="O5"`, group_id, narrow-gap tag.
  - **Margin rule (resolves ambiguity):** reject (return None) unless **each body independently clears its own-width margin**: `|d_small − H| ≥ 0.5*(2*r_small)` AND `|d_large − H| ≥ 0.5*(2*r_large)`. The horizon `H` is the single shared physical path (red line); only the margin threshold is per-body (each body must be unambiguously on its side of `H` relative to its own footprint).
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -am "feat: O5 instantiation with fixed-metric horizon (red line)"`

---

### Task 17: O5 flip metrics

**Files:** Create `egoconseq/eval/metrics.py` (`o5_metrics`); Test `tests/test_metrics_o5.py`

- [ ] **Step 1: Failing test**

```python
from egoconseq.eval.metrics import o5_metrics

def test_o5_flip_metrics():
    # groups with GT flip (small_only). model preds vary or not.
    gt   = ["small_only","small_only","both"]
    pred = ["small_only","both",      "both"]
    m = o5_metrics(gt, pred)
    assert 0 <= m["narrow_band_flip_acc"] <= 1
    assert "embodiment_sensitivity" in m and "correct_flip_rate" in m and "invariance_error" in m
```

- [ ] **Step 2-4: Implement + run** → PASS. Primary = `narrow_band_flip_acc` over `small_only` groups; plus Embodiment Sensitivity / Correct Flip Rate / Invariance Error (design §6).
- [ ] **Step 5: Commit** `git commit -am "feat: O5 flip metrics"`

---

### Task 18: O5 demo generation + run (GATE 1)

**Files:** Create `egoconseq/pipeline/generate.py` (`generate_o5`); Create `scripts/run_o5_demo.py`

- [ ] **Step 1: Implement** `generate_o5(scenes, n_pairs)`: sample poses valid for all radii → forward `d_safe` for `r_small,r_large` (depth + navmesh) → disagreement keep → narrow-band O5 pairs → write `data/demo/o5/manifest.jsonl` + overlays.
- [ ] **Step 2: Run generation** on 2–3 scenes, target ≥30 pairs; print kept/discarded by taxonomy.
- [ ] **Step 3: Baselines + oracle + VLM (VLM MANDATORY)**: run `radius-only`, `image-only-no-body`, oracle, and the qwen3vl VLM via `eval/baselines.py::vlm_answer`; compute `o5_metrics`. The gate cannot be evaluated without the VLM run.
- [ ] **Step 4: GATE 1 decision** — write `data/demo/o5/report.md`. Two conditions, BOTH required:
  - **(a) GT-discriminator exists** (necessary): ≥30 `small_only` pairs; oracle flip_acc ≈ 1.0; radius-only/blind ≈ chance.
  - **(b) VLM gap confirmed** (also required, per spec §7): VLM fails to flip reliably (low `correct_flip_rate` / high `invariance_error`).
  - Pass only if (a) AND (b). If VLM flips perfectly → O5 not discriminative for this VLM → reconsider difficulty/headline before scaling.
- [ ] **Step 5: Commit** `git commit -am "feat: O5 demo generation + GATE 1 report"`

> ⚠ GATE P2 (LIFE-OR-DEATH 1): a real GT flip set exists and behaves as a discriminator. This is the primary justification for the whole benchmark.

---

## P3 — O4 Directional Ranking (LIFE-OR-DEATH GATE 2)

### Task 19: Center-ray depth heuristic baseline

**Files:** Create `egoconseq/eval/baselines.py`; Test `tests/test_baselines.py`

- [ ] **Step 1: Failing test** for `center_ray_depth_clearance(depth, K)` and `three_ray_depth(depth,K)` returning per-direction distances; assert on synthetic depth a closer left wall → left has smaller clearance.
- [ ] **Step 2-4: Implement + run** → PASS. Also `random_baseline`, `majority_baseline`, `blind_text_only` stubs.
- [ ] **Step 5: Commit** `git commit -am "feat: center-ray depth + shortcut baselines"`

---

### Task 20: O4 ranking instantiation + metrics

**Files:** Modify `instantiate.py` (`make_o4`), `metrics.py` (`ranking_metrics`); Tests

- [ ] **Step 1: Failing tests**: `make_o4` ranks `{-15,0,+15}` by `d_safe`, rejects if `top1-top2 < 0.5*width` (tie). `ranking_metrics` → top-1 acc + Kendall tau.
- [ ] **Step 2-4: Implement + run** → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: O4 ranking instantiation + metrics"`

---

### Task 21: O4 demo generation + run (GATE 2)

**Files:** Modify `generate.py` (`generate_o4`); Create `scripts/run_o4_demo.py`

- [ ] **Step 1-2: Implement + generate** ≥30 O4 cases over 2–3 scenes (depth + navmesh per direction, disagreement keep, tie-reject).
- [ ] **Step 3: Run** baselines (center-ray, three-ray, random) + oracle + VLM; compute ranking metrics.
- [ ] **Step 4: GATE 2 decision** — `data/demo/o4/report.md`: **is center-ray/three-ray depth heuristic ≪ VLM, and is VLM above center-bias (not always "straight")?** If depth heuristic already solves O4 → O4 is just depth, weaken headline.
- [ ] **Step 5: Commit** `git commit -am "feat: O4 demo generation + GATE 2 report"`

> ⚠ GATE P3 (LIFE-OR-DEATH 2): O4 is not solved by a depth heuristic and the VLM shows non-trivial directional comparison.

---

## P4 — O1 / O3 Foundation Tasks

> **O2 (Critical Width) is intentionally deferred past the demo.** Spec Demo DoD §11 prioritizes O5→O4→O1/O3 and lists O2 as a body-width-family continuous diagnostic, not a demo gate. O2 (binary-search `r_crit` over the same `d_safe` machinery) is added in the full-benchmark plan, not here. This is a deliberate scope cut, not an omission.

### Task 22: O1 forward clearance (magnitude)

**Files:** `instantiate.py` (`make_o1`), `prompts.py`, `metrics.py` (`capped_mae`, `bin_acc`); Tests

- [ ] **Step 1: Failing tests**: `make_o1` answer in **body-widths**, capped at visible region; `capped_mae` caps error beyond D_max bin; `bin_acc` over `{<2,2-4,4-8,>8}`; numeric tolerance 0.75 bw.
- [ ] **Step 2-4: Implement + run** → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: O1 magnitude task + capped MAE/bin metrics"`

---

### Task 23: O3 turn-then-forward contact

**Files:** `instantiate.py` (`make_o3`), `metrics.py` (`false_safe_rate`); Tests

- [ ] **Step 1: Failing tests**: `make_o3` threshold `d_safe(turn±15→forward) < H`, body-width horizon; `false_safe_rate` = GT-contact-but-pred-no-contact fraction.
- [ ] **Step 2-4: Implement + run** → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: O3 turn-contact task + False-Safe Rate"`

---

### Task 24: O1/O3 generation (foundation, not a gate)

**Files:** `generate.py` (`generate_o1`, `generate_o3`); `scripts/run_o1o3_demo.py`

- [ ] **Step 1-3: Generate + run** ≥30 each; report VLM **and** center-ray depth side-by-side.
- [ ] **Step 4: Note in report**: O1/O3 being depth-solvable is expected (foundation), NOT a failure (design §11). Do not gate on it.
- [ ] **Step 5: Commit** `git commit -am "feat: O1/O3 foundation generation"`

---

## P5 — Scale + Full Validity Gates + Decision

### Task 25: Dataset balance + dedup

**Files:** Create `egoconseq/pipeline/balance.py`; Test `tests/test_balance.py`

- [ ] **Step 1: Failing tests** (design §8): per-class single-answer cap, per-scene cap, same-RGB not reused across operations (except counterfactual group_id), one non-counterfactual case per trajectory.
- [ ] **Step 2-4: Implement + run** → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat: dataset balance + dedup gates"`

---

### Task 26: Full baseline suite runner

**Files:** Create `scripts/run_all_baselines.py`

- [ ] **Step 1: Implement** runs random / majority / blind text-only / radius-only / action-only / center-ray depth / floor-width heuristic / geometry oracle / VLM(RGB) / VLM(RGB+sweep overlay) / (human placeholder) across all generated operations.
- [ ] **Step 2: Run** over ≥5 scenes dataset; dump `data/demo/baselines.json`.
- [ ] **Step 3: Commit** `git commit -am "feat: full baseline suite runner"`

---

### Task 27: Four validity gates + per-operation report

**Files:** Create `egoconseq/eval/validity.py`; Create `scripts/build_demo_report.py`; Test `tests/test_validity.py`

- [ ] **Step 1: Failing test** for `validity_gates(results)` returning pass/fail on: oracle self-consistency; blind/majority ≈ chance; **center-ray depth ≪ VLM on O4/O5**; human > VLM (placeholder until human run).
- [ ] **Step 2-4: Implement + run** → PASS.
- [ ] **Step 5: Build report** `scripts/build_demo_report.py` → `data/demo/REPORT.md` with per-operation metrics table (design §6) + gate verdicts.
- [ ] **Step 6: Commit** `git commit -am "feat: validity gates + demo report"`

---

### Task 28: Demo go/no-go decision writeup

**Files:** Create `P_bench/docs/2026-XX-demo-findings.md`

- [ ] **Step 1: Summarize** which operations passed validity, the O4/O5 VLM gaps, false-safe rates, and the decision: finding-paper vs +method, or iterate design.
- [ ] **Step 2: Update memory** `project_egoconseq_bench.md` with demo outcome.
- [ ] **Step 3: Commit** `git commit -am "docs: demo findings + go/no-go decision"`

> ⚠ GATE P5 (DEMO DECISION): all four validity gates evaluated; O4/O5 show VLM≪oracle/human gaps. Only then scale to the full ≥100-case-per-class benchmark (separate plan).

---

## Notes for the implementer

- Always run python via `/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python`.
- Habitat headless: if X errors occur, set `export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet` and use the EGL/headless build (already in this env).
- Pure modules (`geometry`, `oracle/*`, `gates/*`, `tasks/*`, `eval/*`) must have **no Habitat import** so their unit tests run without a GPU/scene.
- The model NEVER receives depth/pose/navmesh — only `image_path + question`. Enforce this in `manifest.Case` (model payload is a separate method).
- VLM hookup (P2+) reuses the existing qwen3vl serving in this repo; wrap behind `eval/baselines.py::vlm_answer(image, question)` so tasks stay decoupled from the serving details.
- Defer to the design doc on every constant; if something's missing there, stop and ask rather than inventing.
```
