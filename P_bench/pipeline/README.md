# EgoConseq Consequence Pipeline

A self-contained GT-extraction pipeline for the **EgoConseq** consequence benchmark.

> Given one first-person frame, a robot **body** (cylinder radius + camera height),
> and an **action list** (turns + forwards, exact values), compute — by pure
> geometry — every ground-truth element the benchmark needs, and dump it raw.

The task it feeds: *can an MLLM, seeing a single egocentric image and told its own
body + a candidate action, mentally sweep the body through the visible space and
predict the physical consequence?* — collision, how far it can go, which action it
collides during, resulting heading/bearings, and view-exit. This package produces the
**GT**; QA question generation is a separate read-only layer on the records.

**Two scene backends** (same geometric core, swap only the render+semantic modules):

| backend | scenes | look | semantics | camera height |
|---|---|---|---|---|
| **`hm3d`** | HM3D v0.2 val (real photogrammetry scans) | blurry | offline texture-decode → per-pixel masks | discrete sensors |
| **`gs`** | Habitat-GS 3DGS + InteriorGS labels | **crisp / photoreal** | InteriorGS 3D bboxes (755 cats) | continuous (gsplat) |

---

## Table of contents

1. [Environment setup](#environment-setup)
2. [Full pipeline: download → collect → validate → review](#full-pipeline-download--collect--validate--review)
3. [HM3D backend: offline semantic texture decode](#hm3d-backend-offline-semantic-texture-decode)
4. [GS backend: gsplat render + InteriorGS bbox semantics](#gs-backend-gsplat-render--interiorgs-bbox-semantics)
5. [Concepts & data flow](#concepts--data-flow)
6. [Coordinate conventions](#coordinate-conventions)
7. [What `judge` computes](#what-judge-computes)
8. [Review app: robot-body panel & variable height](#review-app-robot-body-panel--variable-height)
9. [Package layout](#package-layout)
10. [Record schema](#record-schema)
11. [Invariants (V1–V9)](#invariants-v1v9)
12. [Scripts & CLI](#scripts--cli)
13. [Config constants](#config-constants)
14. [Testing & caveats](#testing--caveats)

---

## Environment setup

**Runtime.** Conda env `qwen3vl_habitat` — **Python 3.9**, **habitat-sim 0.2.4**,
**torch 2.5.1+cu124** (CUDA), **gsplat 1.5.3** + plyfile (GS backend), plus numpy,
scipy, trimesh, Pillow, matplotlib.

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
cd /home/zhangshan/syp/myvln/P_bench

# GS render dependency (one-time; gsplat JIT-compiles a CUDA ext on first use)
$PY -m pip install gsplat plyfile
```

Rules that bite:
- **Always** prefix Habitat runs with `MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet`.
- **GS backend needs the CUDA toolchain on PATH** (gsplat's ninja + nvcc):
  ```bash
  ENV=/home/zhangshan/miniconda3/envs/qwen3vl_habitat
  export PATH=$ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  ```
- Habitat is **single-threaded / not thread-safe** — one scene session at a time.
- **Only `pipeline/sim.py` (HM3D) and `pipeline/gs_*.py` (GS) touch heavy deps.**
  Everything else is pure numpy and unit-tested without Habitat/CUDA.

---

## Full pipeline: download → collect → validate → review

### Stage 1 — download scene data

**HM3D** — HM3D v0.2, `val` split, the 36 scenes shipping a `*.semantic.glb`. Roots
in `config.py` (`HM3D_ROOT`, `HM3D_VAL_DIR`). Layout:
```
<HM3D_ROOT>/val/<scene>/<id>.basis.glb          # render mesh
                        <id>.semantic.glb        # per-instance colour texture
                        <id>.semantic.txt        # palette: id, hex, "category", region
```

**GS** — two Hugging Face sources, aligned by scene id (`interior_0007_840137`↔`0007_840137`):
```bash
# gate: accept the InteriorGS licence once (browser): huggingface.co/datasets/spatialverse/InteriorGS
TOK=$(cat ~/.cache/huggingface/token)
GS=https://huggingface.co/datasets/RukawaY/gs_scenes/resolve/main
IG=https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/main
ROOT=/home/zhangshan/syp/datasets/gs                       # = config.GS_ROOT
for id in interior_0733_841584 interior_0651_841463; do    # a few scenes suffice
  num=${id#interior_}; d=$ROOT/$id; mkdir -p "$d"
  curl -sL -H "Authorization: Bearer $TOK" "$GS/val/$id/$id.gs.ply"  -o "$d/scene.gs.ply"
  curl -sL -H "Authorization: Bearer $TOK" "$GS/val/$id/$id.navmesh" -o "$d/scene.navmesh"
  curl -sL -H "Authorization: Bearer $TOK" "$IG/$num/labels.json"    -o "$d/labels.json"   # gated
done
```
Each GS scene dir holds `{scene.gs.ply, scene.navmesh, labels.json}`. `RukawaY/gs_scenes`
is public; **`spatialverse/InteriorGS` is gated** (must accept the licence on HF first —
it carries the 3D bboxes, 755 categories).

### Stage 2 — collect → `records.jsonl`

```bash
Q="MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet"
# HM3D (default backend)
$Q $PY scripts/collect.py --auto-scenes --poses-per-scene 20 \
    --radii 0.15 0.20 0.25 --seed 42 --out data/conseq/v2

# GS (needs the CUDA env vars above)
$Q $PY scripts/collect.py --backend gs --auto-scenes \
    --poses-per-scene 20 --radii 0.15 0.20 0.25 --out data/conseq/gs
```
Per scene: sample quality-filtered poses → `build_frame` → generate an **in-FOV,
length-1..6 action pool** (no adjacent forwards; consecutive turns alternate direction)
→ `judge` each × radius, keep 5 clean outcomes per (frame,length), drop red / flag
yellow → one jsonl line per frame + `img/<frame>.png` + `run_meta.json`. Auto-validates.

### Stage 3 — validate

`collect.py` runs it automatically (exit 1 on any violation); or standalone:
```bash
$PY scripts/check_records.py data/conseq/v2/records.jsonl
```

### Stage 4 — review (interactive browser app)

```bash
$Q PATH=$ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
  $PY scripts/serve_viz.py --port 8767 \
    --datasets gs=data/conseq/gs:gs v2=data/conseq/v2:hm3d
# open http://localhost:8767  (remote: ssh -N -L 8767:localhost:8767 user@host)
```
Browse scene→frame→outcome; the **Robot body** panel (camera height + chassis radius)
re-renders the egocentric image and re-judges the frame live; switch datasets (GS/HM3D)
from the top dropdown. See [Review app](#review-app-robot-body-panel--variable-height).

---

## HM3D backend: offline semantic texture decode

**The key HM3D fact.** This habitat-sim 0.2.4 build **cannot deliver HM3D-v0.2
semantics** through its API: the `SEMANTIC` sensor renders all zeros; `semantic_scene`
AABBs are all zero (no bboxes); HM3D ships no bboxes. Instance labels live **only in
the `semantic.glb` texture**, so we decode them offline (`pipeline/semantic.py`):

```
1. parse <scene>.semantic.txt   -> palette: colour -> instance id + category
2. load  <scene>.semantic.glb (trimesh); per face sample the texture at its UV
   centroid -> nearest palette colour -> instance id            (~94% face coverage)
3. area-weighted surface-sample labelled faces (~1e6 pts); mesh Z-up -> Habitat
   Y-up: world = (x, z, -y)
4. cache to data/conseq/semantic_cache/<scene>.sem.npz          (~one-time 20-30 s/scene)
```
At runtime every depth-derived world point is assigned to the nearest labelled surface
point within a tolerance → **100 % per-pixel masks** without the broken sensor. No new
dependency, no Habitat rebuild. (Confirmed by the SEMNAV paper: the built-in sensor does
not deliver true HM3D segmentation.)

---

## GS backend: gsplat render + InteriorGS bbox semantics

- **Render** (`pipeline/gs_render.py`): `plyfile` loads the `.gs.ply` gaussians;
  `gsplat.rasterization` renders **RGB + depth** at any camera pose/height. The
  camera extrinsics match `perception`'s convention (Y-up world, yaw about +Y).
- **Navmesh** (`pipeline/gs_sim.py`): the scene's pre-baked `.navmesh` loads directly
  via habitat 0.2.4 `PathFinder.load_nav_mesh` — **no fork needed**. (Single baked
  navmesh; collision GT is depth-based so multi-radius still works.)
- **Semantics** (`pipeline/gs_semantic.py`): InteriorGS `labels.json` gives 8-corner
  3D bboxes (Z-up) with 755 categories. We rotate them to the gaussian/navmesh frame
  with **`(x, y, z) → (x, z, -y)`** (calibrated: same rotation maximises gaussian-in-bbox
  coverage on every scene; scale = 1) and assign each point to the **smallest containing
  bbox**. Structural surfaces (wall/floor/ceiling) may be unlabelled in a scene →
  those contacts read `unattributed`, which is fine.

`GsSimSession` exposes the **same interface** as HM3D `SimSession` (`render`,
`assign_instances`, `nav`, `sample_random_pose`, ...), so `build_frame`/`judge`/
`record`/`validate`/`viz` are backend-agnostic — the pure-numpy core is unchanged.

---

## Concepts & data flow

```
SimSession(scene) | GsSimSession(scene)   # RGB+Depth render + pathfinder + semantics
   │
   ├─ sample_random_pose(rng, radii)      # quality-filtered (pos, yaw)
build_frame(sim, pos, yaw, cam_h) ───────▶ Frame   (perception evidence, built ONCE)
   │                                         rgb / depth / K / floor_y
   │                                         pts (N,3) ground cloud + pts_uv + pts_sem
   │                                         vf (obstacle VoxelField) + objects[]
Cylinder(radius_m, height_m) ────────────▶ Body
[Turn(-30), Forward(1.2), ...] ──────────▶ ActionSeq   (length 1..6; exact values)
judge(frame, body, acts, nav) ───────────▶ Consequence (one dict of all GT elements)
FrameRecord(frame, [outcome, ...]) ──────▶ one jsonl line (1 frame × many body×action)
```

**Single unproject, everything downstream slices it.** `build_frame` backprojects the
depth **once** into a ground cloud carrying per-point `(pixel, instance id)` provenance;
the obstacle field, object table, and contact attribution are all mask-slices of that
same cloud. `cam_h` threads through render + the ground transform, so a different camera
height yields a genuinely different image **and** different visible geometry (→ GT).

---

## Coordinate conventions

- **Ground frame** (agent-local): origin = footprint on floor, `+z` fwd, `+x` right,
  `+y` up. Metres. Camera at height `cam_h` ⇒ `y_ground = y_cam + cam_h`.
- **Heading** `h`: forward = `(sin h, cos h)`; `h=0` faces `+z`. **positive `Turn` = right**.
- **Bearing** to `(x,z)` = `deg(atan2(x,z))`; **view cone** = `z>0 ∧ |bearing| ≤ 39.5°`.
- **World** (Habitat): `+Y` up, yaw about `+Y`, `yaw=0` looks `-Z`.
- **Semantic mesh / InteriorGS bbox** (Z-up) → world: `(x, y, z) → (x, z, -y)`.

Turns are in-place (zero arc). Only forward legs advance arc length.

---

## What `judge` computes

All deterministic, no learned components.

| Element | How | Record fields |
|---|---|---|
| **Path** | `sample_path`: turn updates heading; forward emits `k·2cm` + endpoint | — |
| **Collision** | footprint disk marches the path; first sample with obstacle-voxel support ≥ 3; also **which action** it lands in | `collided`, `first_contact_arc_m`, `contact_action_index` |
| **Contact class** | majority vote of instance ids within `radius+0.1 m` of the contact | `contact.category/instance_id/vote_fraction/unattributed` |
| **Heading / poses** | `pose_end_full` (executed) & `pose_end_exec` (truncated at contact) | `pose_end_full/exec`, `net_turn_deg` |
| **Object relations** | per object, transform to the end pose; bearing, in-FOV, distances + Δ | `object_relations[].exec/full/delta_full` |
| **View exit** | first arc the path leaves the cone; end in-cone? | `view_exit.*` |
| **Nav cross-check** | navmesh vs depth agreement via the frame's voxel field + fwd coverage | `nav_check.*` |
| **Visibility** | fraction of swept samples that project in-frame with valid depth | `visibility.*` |

---

## Review app: robot-body panel & variable height

`serve_viz.py` is a 3-column browser app (needs Habitat/CUDA — it rebuilds frames):
- **left** scene→frame thumbnails; **middle** RGB (object + contact overlay) + a
  filterable outcome table; **right** an obstacle-accurate **top-down** with a
  body-circle **step-through** (play/scrub, red past the contact arc), a step-by-step
  **reasoning trace**, and a distance-to-objects table.
- **Robot body panel** (top): pick a **camera height** (`config.RENDER_HEIGHTS`, e.g.
  0.4/0.8/1.2/1.5/1.7 m) and a **chassis radius**; on change the app **re-renders the
  egocentric image at that height and re-judges the frame** (`/api/ds/rejudge`,
  `/api/ds/render`). A lower camera sees more floor / under furniture and yields
  different visible geometry → different collisions. The current camera height is shown.
- **Dataset dropdown** switches between all `--datasets` (GS + HM3D) — one Habitat
  session at a time. `Live` tab samples an ad-hoc pose + actions through the same view.

Note: in v1 the collision oracle is a **2D footprint** (radius-dependent); height
changes the GT only through *what the camera sees*. True height-dependent 3D body
collision (under-table) is v2.

---

## Package layout

```
pipeline/
  config.py        constants — single source of truth (camera, voxel, thresholds, paths,
                   RADII_M, RENDER_HEIGHTS, GEN_*, height_tag())
  actions.py       Turn/Forward + ActionSeq; sample_path / pose_* / contact_action_index (pure)
  body.py          Cylinder(radius_m, height_m)
  perception.py    unproject / to_agent_ground(cam_h) / world_from_local / VoxelField /
                   in_cone / bearing_dist                              (pure)
  objects.py       extract_objects / attribute_contact                (pure)
  consequence.py   judge(frame, body, acts, nav) -> Consequence       (pure geometry)
  frame.py         Frame dataclass + build_frame(sim, pos, yaw, cam_h=)
  record.py        FrameRecord schema + jsonl I/O
  validate.py      invariants V1–V9                                   (pure)
  action_gen.py    fov_bucketed_pool (length 1..6, in-FOV, no-degenerate) + path_stays_in_fov
  scene_pool.py    discover_semantic_scenes                           (pure)
  sim.py           HM3D SimSession + Nav + sample_pose (the shared pose sampler)
  semantic.py      HM3D offline texture decode -> SemanticIndex
  gs_render.py     gsplat RGB+depth from a .gs.ply
  gs_semantic.py   InteriorGS 3D-bbox -> per-point instance
  gs_sim.py        GS GsSimSession (same interface as SimSession)
  viz.py           reasoning_trace + top-down evidence (matplotlib, for query/debug)

scripts/
  collect.py       batch collection -> records.jsonl (+ auto-validate)   --backend {hm3d,gs}
  serve_viz.py     interactive review app (+ serve_viz.html)             --datasets / --backend
  query.py         one manual query -> consequences + evidence.png
  smoke_frame.py   HM3D P0 mask eye-check
  gs_smoke.py      GS render smoke (pose -> gsplat RGB+depth)
  check_records.py validator CLI
```

Everything except `sim.py` / `gs_*.py` (and the Habitat/CUDA-facing bits of scripts) is
pure numpy and unit-tested without Habitat.

---

## Record schema

`records.jsonl`, one `FrameRecord` per line (all raw continuous quantities):

```jsonc
{ "schema_version": "conseq.v1",
  "frame_id": "F-...-p003", "scene_id": "...", "scene_glb": "/abs/....",
  "pose": {"position": [x,y,z], "yaw_rad": ..},
  "sensor": {"camera_height_m": 1.5, "hfov_deg": 79, "resolution": [640,480]},
  "floor_y": .., "image_path": "img/F-....png",
  "n_objects": .., "n_nonstructural": .., "category_inventory": {..},
  "objects": [ { "instance_id": .., "category": .., "is_structural": false,
      "centroid_px": [u,v], "ground_xy_centroid": [x,z], "ground_xy_nearest": [x,z],
      "bearing_deg": .., "dist_centroid_m": .., "dist_nearest_m": .., "dist_geodesic_m": null } ],
  "outcomes": [
    { "outcome_id": "F-...-b015-L3-02", "seq_len": 3,
      "body": {"shape":"cylinder","radius_m":0.15,"height_m":1.5},
      "actions": [{"type":"turn","deg":15.0},{"type":"forward","m":1.5},{"type":"turn","deg":-15.0}],
      "total_forward_m": 1.5, "net_turn_deg": 0.0,
      "pose_end_full": {"x":..,"z":..,"heading_deg":..},
      "collided": true, "first_contact_arc_m": 1.5,
      "contact_action_index": 1, "contact_action_local_arc_m": 1.0,   // which action collided
      "pose_end_exec": {..},
      "contact": {"xy":[x,z],"point_3d":[x,y,z],"pixel":[u,v],"pixel_in_frame":false,
                  "instance_id":275,"category":"couch","vote_fraction":1.0,"unattributed":false},
      "view_exit": {"path_exit_arc_m":null,"end_in_fov":true, ..},
      "nav_check": {"d_nav_m":..,"d_depth_m":..,"diff_m":..,"fwd_cover":..,"verdict":"keep_agree","keep":true},
      "visibility": {..},
      "object_relations": [ {"instance_id":275,"category":"couch",
          "exec": {"bearing_deg":..,"dist_centroid_m":..,"dist_nearest_m":..,"in_fov":false},
          "full": {..}, "delta_full": {"d_centroid_m":-0.95,..} } ],
      "human_check": true } ]   // only on yellow review_depth_hole
] }
```
Notes: `fov_len` outcomes carry `seq_len`; red `discard_noise` outcomes are dropped;
`exec` = truncated at contact (realised), `full` = fully executed (counterfactual);
`dist_geodesic_m` is null unless `--geodesic`.

---

## Invariants (V1–V9)

`validate.py` re-derives poses/bearings from stored actions and checks physical bounds.

| # | Check |
|---|---|
| V1 | triangle: `\|dist_after − dist_before\| ≤ total_forward + ε` |
| V2 | pure turn ⇒ distances unchanged, `bearing_after = wrap(bearing_before − net_turn)`, no collision |
| V3 | `collided ⇔ contact ⇔ arc`; `pose_end_exec == pose_at_arc(arc)`; else exec == full |
| V4 | `pose_end_full == pose_after(acts)` |
| V5 | frame object bearing/dist recompute from `ground_xy_centroid` |
| V6 | `in_fov ⇒ \|bearing\| ≤ 39.5°` |
| V7 | schema ranges (bearing, distances ≥ 0, ratios ∈ [0,1], contact consistency) |
| V8 | view-exit consistency (end_in_fov, heading offset, path_exit ≤ total_forward, ...) |
| V9 | `contact_action_index` recomputed matches; points at a Forward; null iff not collided |

The v2 run (710 records, 63,900 outcomes) passes with **0 violations**.

---

## Scripts & CLI

### `collect.py`
```
--backend {hm3d, gs}                   # hm3d = HM3D scans; gs = 3DGS (needs CUDA env)
--auto-scenes | --scenes ...           # auto = all discoverable scenes for the backend
--gs-root DIR                          # gs scene root (default config.GS_ROOT)
--max-scenes N  --poses-per-scene 20  --seed 42
--radii 0.15 0.20 0.25                 # multiple -> per-frame body counterfactuals
--action-mode {fov_len, file}          # default fov_len
    fov_len: length-1..6 in-FOV seqs; keep --keep-per-length clean/frame/length
             (no adjacent forwards; consecutive turns alternate direction)
    file   : --action-file JSON {"sequences":[{"actions":[...]}]}
--lengths 1 2 3 4 5 6  --keep-per-length 5  --pool-factor 3
--min-objects 1  --geodesic  --save-arrays  --debug-images  --out DIR  --no-validate
```

### `serve_viz.py`
```
--datasets name=dir:backend ...        # e.g. gs=data/conseq/gs:gs v2=data/conseq/v2:hm3d
--dataset DIR --backend {hm3d,gs}      # single-dataset shorthand
--port 8767  --img-dir DIR
```

### others
```
query.py         --scene GLB --actions '[...]' --radius 0.25 [--seed N] --out DIR
smoke_frame.py   --scene GLB --poses 2 --out DIR         # HM3D mask eye-check
gs_smoke.py      --scene <gs-scene-dir> --out DIR        # GS render eye-check
check_records.py RECORDS.jsonl
```

---

## Config constants

All in `config.py` (import, never hard-code):
```
camera      HFOV 79°, 640×480, height 1.5 m, FOV half 39.5°
body        RADII (0.15,0.20,0.25) m; RENDER_HEIGHTS (0.4,0.8,1.2,1.5,1.7) m
gen         turns ±{15,30}, forwards {0.5,1,1.5}, lengths 1..6, keep 5/len, pool ×3
perception  obstacle band [0.05,1.5] m, voxel 5 cm, dilation 1, min support 3
march       step 2 cm, d_max 5 m       attribution margin 0.10 m
nav         agree tol 0.3 m, fwd-cover min 0.90, strong support 6
semantic    (HM3D) 1e6 samples, palette tol 12, assign tol 0.20 m
paths       HM3D_ROOT ...; GS_ROOT /home/zhangshan/syp/datasets/gs
```

---

## Testing & caveats

```bash
$PY -m pytest tests/ -q     # pure-numpy pipeline tests, no Habitat/CUDA
```

Caveats:
- **`nav_check` is QA metadata; the depth voxel field defines the collision label.**
  Verdicts: `keep_agree`/`keep_depth`/`keep_visible` (green), `review_depth_hole`
  (yellow → `human_check`), `discard_noise` (red, dropped by the `fov_len` collector).
- **GS depth is see-through at windows/glass** (renders the geometry beyond) → a wall
  opening reads far and won't register a collision there. Solid walls are fine.
- **GS semantics are 3D bboxes (AABB)** — coarser than HM3D per-pixel masks; unlabelled
  structure ⇒ `unattributed` contacts.
- **Variable height** re-renders + re-judges (v1 collision still 2D footprint; only the
  *visible geometry* changes with height). HM3D height is discrete (one sensor per
  `RENDER_HEIGHTS`); GS is continuous.
- **Record size**: `object_relations` are stored for every object × outcome (v2 ≈ 0.9 GB
  / 63,900 outcomes). Restrict to non-structural or gzip to shrink.
- **Action model**: hard-stop at first contact, no sliding; single-frame perception only
  sees the current frustum.
