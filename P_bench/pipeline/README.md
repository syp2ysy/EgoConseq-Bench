# EgoConseq Consequence Pipeline

A self-contained data source for the **EgoConseq** consequence benchmark.
Built on **habitat-sim 0.2.4** + **HM3D v0.2** (val split, 36 semantic scenes).

> Given one first-person Habitat frame, a **body** (cylinder), and an arbitrary
> **action list** (turns + forwards with exact values), compute — by pure
> geometry — every ground-truth element the benchmark needs, and dump it raw.

The task it feeds: *can a VLM, seeing a single egocentric image and told its own
body + a sequence of actions, predict the physical consequences?* — collision,
leaving the field of view, resulting heading, and which target it ends up closer
to. This package produces the **GT** for all of those; QA question generation is
a separate read-only layer on top of the records.

Zero dependency on the old `egoconseq/` package. All validated math (backproject,
floor estimation, voxel occupancy, swept march, projection) was copied in and
re-organised.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Environment & data](#environment--data)
3. [⚠️ Semantic masks: offline texture decode](#-semantic-masks-offline-texture-decode)
4. [Concepts & data flow](#concepts--data-flow)
5. [Coordinate conventions](#coordinate-conventions)
6. [What `judge` computes](#what-judge-computes)
7. [Package layout](#package-layout)
8. [Record schema](#record-schema)
9. [Invariants (V1–V9)](#invariants-v1v9)
10. [Scripts & CLI](#scripts--cli)
11. [Datasets](#datasets)
12. [How the QA layer consumes records](#how-the-qa-layer-consumes-records)
13. [Config / tunable constants](#config--tunable-constants)
14. [Testing](#testing)
15. [Caveats & known issues](#caveats--known-issues)

---

## Quick start

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
Q="MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet"
cd /home/zhangshan/syp/myvln/P_bench

# 1. unit tests (pure numpy, no Habitat)
$PY -m pytest tests/test_pl_*.py -q

# 2. eye-check semantic masks on one scene
$Q $PY scripts/smoke_frame.py \
    --scene $HM3D/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb --poses 2 --out data/conseq/smoke_p0

# 3. one manual query -> prints all consequences + evidence.png
$Q $PY scripts/query.py --scene .../TEEsavR23oF.basis.glb --seed 7 --radius 0.25 \
    --actions '[{"type":"turn","deg":-30},{"type":"forward","m":1.2}]' --out data/conseq/query_demo

# 4. batch collect + auto-validate (length-1..6 in-FOV, 3 chassis radii)
$Q $PY scripts/collect.py --auto-scenes --poses-per-scene 20 --action-mode fov_len \
    --radii 0.15 0.20 0.25 --seed 42 --out data/conseq/v2

# 5. interactive review app (browse the dataset in a browser)
$Q $PY scripts/serve_viz.py --port 8767 --dataset data/conseq/v2
```

Always prefix Habitat runs with `MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet`.

---

## Environment & data

**Runtime.** Conda env `qwen3vl_habitat` — **Python 3.9**, **habitat-sim 0.2.4**
(the only heavy dependency; plus numpy, scipy, trimesh, Pillow, matplotlib).
```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
```
Habitat is **single-threaded / not thread-safe** — one `SimSession` (one scene) is
held at a time. **Only `pipeline/sim.py` imports `habitat_sim`;** everything else is
pure numpy and unit-tested without Habitat.

**Scenes.** **HM3D v0.2** (Habitat-Matterport 3D), **`val` split**, restricted to the
**36 scenes that ship a `*.semantic.glb`**:
```
/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/<scene>/
    <id>.basis.glb        # render mesh
    <id>.semantic.glb     # per-instance colour texture (labels live HERE)
    <id>.semantic.txt     # palette: instance-id, hex colour, "category", region
```
`scene_pool.discover_semantic_scenes()` globs exactly those 36 (a `.basis.glb` with
a sibling `.semantic.glb`). Roots are in `config.py` (`HM3D_ROOT`, `HM3D_VAL_DIR`,
`HM3D_SCENE_DATASET_CFG` = `hm3d_annotated_basis.scene_dataset_config.json`); edit
there to point at another HM3D checkout.

**Semantics decoded offline.** This build's `SEMANTIC` sensor is broken (next
section), so instance masks are decoded from `semantic.glb` into
`data/conseq/semantic_cache/<scene>.sem.npz` (~428 MB for all 36) — **required** to
build frames / re-collect; do not delete.

---

## ⚠️ Semantic masks: offline texture decode

**The single most important implementation fact.** This habitat-sim 0.2.4 build
**cannot deliver HM3D-v0.2 semantics** through its normal API:

- the `SEMANTIC` sensor renders **all zeros** (verified over 200+ frames), even
  with the annotated `scene_dataset_config`, `load_semantic_mesh=True`, and the
  semantic mesh loading successfully (`Semantic Txtrs: True` in the log);
- `sim.semantic_scene.objects[*].aabb` / `.obb` are **all zero** (0 / 661) — no
  per-object bounding boxes either;
- the HM3D dataset ships **no bounding boxes**: `semantic.txt` is only
  `id, hex_colour, "category", region`.

The instance labels live **only in the `semantic.glb` texture** (each object's
surface is painted its palette colour). So we decode them ourselves, offline:

```
pipeline/semantic.py
  1. parse  <scene>.semantic.txt          -> palette: colour -> instance id + category
  2. load   <scene>.semantic.glb (trimesh); per face sample the texture at its
     UV centroid -> nearest palette colour -> instance id      (~94% face coverage)
  3. area-weighted surface-sample the labelled faces (~1e6 pts); transform mesh
     frame (Z-up) -> Habitat world (Y-up):  world = (x, z, -y)  (empirically verified)
  4. cache the labelled world points + ids to data/conseq/semantic_cache/<scene>.sem.npz
```

At runtime (`build_frame`), every depth-derived world point is assigned to the
**nearest labelled surface point within a tolerance** (`SEMANTIC_ASSIGN_TOL_M`,
0 = unlabelled) — reproducing a **depth-aligned per-pixel instance mask** without
the broken sensor. On real scenes this yields **100 % coverage** of valid-depth
pixels with masks that track object silhouettes (eye-checked). **No new
dependency, no Habitat rebuild.**

Cost: ~one-time 20–30 s per scene to build the cache; negligible at runtime.

---

## Concepts & data flow

Four cleanly-layered concepts:

```
SimSession(scene)                     # RGB+Depth render + pathfinder + semantic index
   │
   ├─ sample_random_pose(rng, radii)  # quality-filtered (pos, yaw)
   │
build_frame(sim, pos, yaw)  ─────────▶ Frame      (perception evidence, built ONCE)
   │                                     rgb / depth / K / floor_y
   │                                     pts (N,3) ground cloud + pts_uv + pts_sem
   │                                     vf  (obstacle VoxelField)
   │                                     objects[]  (visible instances)
   │
Cylinder(radius_m, height_m)  ───────▶ Body
[Turn(-30), Forward(1.2), ...]  ─────▶ ActionSeq   (any length; exact values)
   │
judge(frame, body, acts, nav)  ──────▶ Consequence (one dict of all GT elements)
   │
FrameRecord(frame, [outcome, ...])  ─▶ one jsonl line   (1 frame × many body×action)
```

**Single unproject, everything downstream slices it.** `build_frame` backprojects
the depth **once** into a ground point cloud carrying per-point `(pixel, instance
id)` provenance. The obstacle field, the visible-object table, and contact
attribution are all boolean-mask slices of that same cloud — no second
backprojection.

**End-to-end run (`collect.py`), per scene:**

1. `SimSession(scene)` — open the render mesh + pathfinder + offline `SemanticIndex`.
2. `recompute_navmesh(max radius)`, then `sample_random_pose × N` — random navigable
   poses with obstacle clearance ≥ `max(radii)+0.1 m`, ≥ 85 % valid depth, and some
   visible floor (so "no obstacle underfoot" holds for every tested body).
3. `build_frame` at each pose → `Frame` (render RGB-D once → unproject → assign
   instances → obstacle `VoxelField` → `extract_objects`). Frames with fewer than
   `--min-objects` non-structural objects are skipped.
4. `fov_bucketed_pool(rng)` — rejection-sample **in-FOV** action sequences bucketed by
   primitive length **1..6** (no adjacent forwards; consecutive turns alternate
   direction). Frame-independent, so built once per scene.
5. For each **radius × frame × length**: `judge` pool candidates until
   `--keep-per-length` (=5) clean outcomes are kept — **drop** red `discard_noise`,
   **flag** yellow `review_depth_hole` (`human_check: true`), keep the greens.
6. `build_record` → append one jsonl line per frame (frame + all its outcomes);
   `validate_file` runs at the end (exit 1 on any violation).

The result is `data/conseq/v2/`: `records.jsonl` + `img/<frame>.png` + `run_meta.json`.

---

## Coordinate conventions

Used everywhere in the package:

- **Ground frame** (agent-local): origin = footprint centre **on the floor**,
  `+z` = forward, `+x` = right, `+y` = up. Metres.
- **Heading** `h` (rad): forward direction at heading `h` is `(sin h, cos h)` in
  `(x, z)`; `h = 0` faces `+z`. A **positive `Turn` (deg > 0) = right turn**.
- **Bearing** to a point `(x, z)`: `deg(atan2(x, z))` — `0` straight ahead,
  positive = right.
- **View cone** (current FOV): `z > 0 ∧ |atan2(x, z)| ≤ 39.5°` (HFOV 79°/2).
- **World frame** (Habitat): `+Y` up, agent root at floor level, yaw about `+Y`,
  `yaw = 0` looks along `-Z`.
- **Camera frame**: `+z` forward (= depth), `+x` right, `+y` up; level camera at
  1.5 m ⇒ `y_ground = y_cam + 1.5`.
- **Semantic mesh** (Z-up) → world: `world = (x, z, -y)`.

Turns are **in-place** (zero arc length; the footprint disk is rotation-invariant
so a turn adds no swept area). Forward legs march along the current heading.

---

## What `judge` computes

All deterministic, no learned components.

| Element | How | Record fields |
|---|---|---|
| **Path** | `sample_path(acts, 2 cm)`: turn updates heading only; forward emits `k·step` + exact endpoint | — |
| **Collision** | footprint disk marches the path; first sample where the obstacle `VoxelField` support ≥ 3 voxels; also which action it lands in | `collided`, `first_contact_arc_m`, `contact_action_index`, `contact.xy/point_3d/pixel` |
| **Contact class** | majority vote of raw instance ids within `radius+0.10 m` of the contact (relax ×1, else `unattributed`) | `contact.category/instance_id/vote_fraction/votes/unattributed` |
| **Heading / poses** | `pose_end_full` (executed fully) and `pose_end_exec` (truncated at first contact) | `pose_end_full`, `pose_end_exec`, `net_turn_deg` |
| **Object relations** | per visible object, transform its centroid/points into the end pose frame; bearing, in-FOV, euclidean (centroid + nearest surface) + geodesic distances, and Δ vs before | `object_relations[].exec/full/delta_full` |
| **View exit** | first arc the path leaves the view cone; end pose in-cone?; endpoint projection + occlusion; end heading offset | `view_exit.*` |
| **Nav cross-check** | march the same path on the per-radius navmesh; classify depth-vs-navmesh agreement via the frame's own obstacle voxel field + forward-cone depth coverage (artifact-free) | `nav_check.*` |
| **Visibility** | fraction of swept-corridor samples that project in-frame with valid depth | `visibility.*` |

---

## Package layout

```
pipeline/
  config.py        constants — single source of truth (camera, voxel, thresholds, paths)
  actions.py       Turn/Forward + ActionSeq; sample_path / pose_after / pose_at_arc /
                   net_turn_deg / total_forward_m / wrap_deg           (pure)
  body.py          Cylinder(radius_m, height_m)
  perception.py    unproject / to_agent_ground / world_from_local / estimate_floor /
                   obstacle_mask / project_ground / VoxelField         (pure)
  objects.py       extract_objects / attribute_contact                 (pure)
  semantic.py      offline HM3D texture decode + runtime SemanticIndex.assign
  consequence.py   judge(frame, body, acts, nav) -> Consequence        (pure geometry)
  frame.py         Frame dataclass + build_frame(sim, pos, yaw)
  record.py        FrameRecord schema + jsonl I/O (append-style; numpy -> python)
  validate.py      invariants V1–V9 (pure functions over serialized records)
  scene_pool.py    discover_semantic_scenes                            (pure)
  sim.py           SimSession + Nav — the ONLY Habitat-dependent module
  viz.py           object overlay (RGB) + top-down evidence map
  action_gen.py    action-sequence generators: grid / random +
                   fov_bucketed_pool (length 1..6, in-FOV rejection sampling;
                   no adjacent forwards, consecutive turns alternate direction) +
                   path_stays_in_fov (frame-free FOV-cone filter)
  assets/          hm3d_val_annotated.scene_dataset_config.json (val-path config)

scripts/
  smoke_frame.py   P0: build frames on a real scene, eye-check masks
  query.py         manual single query (exact actions) -> consequences + evidence.png
  collect.py       batch collection -> records.jsonl (+ auto-validate)
  check_records.py validator CLI (exit 1 on any violation)
  build_viz.py     static HTML review page from a collection dir
  serve_viz.py     interactive test page (stdlib http.server; single Habitat session)

tests/
  test_pl_*.py     pure numpy, no Habitat (actions, perception, objects, consequence,
                   record, validate, scene_pool) + tests/_synthetic.py (synthetic Frame)
```

Everything except `sim.py` (and the Habitat-facing bits of the scripts) is pure
numpy and unit-tested without Habitat.

---

## Record schema

`records.jsonl`, one `FrameRecord` per line (all raw continuous quantities):

```jsonc
{ "schema_version": "conseq.v1",
  "frame_id": "F-00800-TEEsavR23oF-p003", "scene_id": "00800-TEEsavR23oF",
  "scene_glb": "/abs/.../TEEsavR23oF.basis.glb",
  "pose": {"position": [x,y,z], "yaw_rad": ..},
  "sensor": {"camera_height_m": 1.5, "hfov_deg": 79, "resolution": [640,480]},
  "floor_y": 0.02,
  "image_path": "img/F-....png", "depth_path": null, "semantic_path": null,
  "quality": {"valid_depth_ratio": .., "dist_to_obstacle_m": .., "visible_floor_ratio": ..},
  "n_objects": 31, "n_nonstructural": 12,
  "category_inventory": {"couch": 2, "table": 2, "wall": 14, ...},   // full mask inventory
  "objects": [                                                       // one per visible instance
    { "instance_id": 275, "category": "couch", "is_structural": false,
      "mask_area_px": 9216, "centroid_px": [u,v],
      "ground_xy_centroid": [x,z], "ground_xy_nearest": [x,z],
      "bearing_deg": -32.0, "dist_centroid_m": 2.21, "dist_nearest_m": 2.02,
      "n_points_raw": 9216, "dist_geodesic_m": null }
  ],
  "outcomes": [                                                      // one per (body, action seq)
    { "outcome_id": "F-...-b015-L3-02", "seq_len": 3,                // b{r*100}-L{len}-{k}
      "body": {"shape": "cylinder", "radius_m": 0.15, "height_m": 1.5},
      "actions": [{"type":"turn","deg":15.0}, {"type":"forward","m":1.5}, {"type":"turn","deg":-15.0}],
      "total_forward_m": 1.5, "net_turn_deg": 0.0,
      "pose_end_full":  {"x":0.0, "z":1.5, "heading_deg":0.0},
      "collided": true, "first_contact_arc_m": 1.5,
      "contact_action_index": 1, "contact_action_local_arc_m": 1.0,  // which action collided (0-based) + m into it; null if no collision
      "pose_end_exec":  {"x":0.0, "z":1.5, "heading_deg":0.0},       // == full if no collision
      "contact": {                                                   // null if no collision
        "xy": [x,z], "point_3d": [x,y,z], "pixel": [u,v], "pixel_in_frame": false,
        "instance_id": 275, "category": "couch",
        "vote_fraction": 1.0, "votes": {"275": 46}, "unattributed": false },
      "view_exit": {
        "path_exit_arc_m": null, "end_in_fov": true,
        "end_bearing_deg": 0.0, "end_dist_m": 1.5,
        "end_pixel": [u,v], "end_pixel_in_frame": false, "end_visible": false,
        "end_heading_offset_deg": 0.0 },
      "nav_check": {                                                 // null if nav not supplied
        "d_nav_m": 0.88, "d_depth_m": 1.05, "diff_m": 0.17, "fwd_cover": 0.98,
        "support_at_hit": null, "verdict": "keep_agree", "keep": true },
      "visibility": {"visible_sweep_ratio": .., "valid_depth_ratio": ..,
                     "depth_hole_ratio": .., "occlusion_free_ratio": ..},
      "object_relations": [                                          // one per visible object
        { "instance_id": 275, "category": "couch",
          "exec": {"bearing_deg":63.1,"dist_centroid_m":1.62,"dist_nearest_m":0.95,
                   "dist_geodesic_m":null,"in_fov":false},
          "full": {"bearing_deg":78.4,"dist_centroid_m":1.31,"dist_nearest_m":0.61,
                   "dist_geodesic_m":null,"in_fov":false},
          "delta_full": {"d_centroid_m":-0.95,"d_nearest_m":-1.06,"d_geodesic_m":null} }
      ] }
  ] }
```

Notes:
- `fov_len` outcomes carry `seq_len` (primitive count 1..6) and, on yellow
  `review_depth_hole`, `human_check: true`; red `discard_noise` outcomes are dropped
  (never written). `outcome_id` is `{frame}-b{r*100}-L{len}-{k}`.
- `dist_geodesic_m` is `null` unless collection ran with `--geodesic` (opt-in; slow).
- `contact.pixel` uses a mid-band display height (`floor_y + 0.5`); at close range
  it can project below the image (`pixel_in_frame: false`) — the class/arc are
  still correct.
- `exec` = truncated at first contact; `full` = as if fully executed. On a
  collision use `exec` for the *realised* consequence, `full` for the counterfactual.

---

## Invariants (V1–V9)

`validate.py` re-derives poses/bearings from the stored actions and checks
physical inequalities. `scripts/check_records.py` exits 1 on any violation;
`collect.py` runs it automatically unless `--no-validate`.

| # | Check |
|---|---|
| V1 | triangle inequality `|dist_after − dist_before| ≤ total_forward + ε` (all 3 distances; geodesic lower-bounded) |
| V2 | pure turn ⇒ distances unchanged, `bearing_after = wrap(bearing_before − net_turn)`, `collided = false` |
| V3 | `collided ⇔ contact ⇔ arc`; `pose_end_exec == pose_at_arc(acts, arc)`; else `exec == full` |
| V4 | `pose_end_full == pose_after(acts)` (recomputed) |
| V5 | frame object bearing/dist recomputed from `ground_xy_centroid` matches stored |
| V6 | `in_fov ⇒ |bearing| ≤ 39.5° + ε` |
| V7 | schema ranges: bearing ∈ (−180,180], distances ≥ 0, ratios ∈ [0,1], `d_nav ≤ total + D_MAX`, contact-field consistency |
| V8 | view-exit consistency: `end_in_fov ⇔ (z>0 ∧ |end_bearing| ≤ 39.5°)`, `end_heading_offset == wrap(net_turn)`, `path_exit_arc ≤ total_forward`, `end_visible ⇒ end_pixel_in_frame`, recomputed end bearing/dist match |
| V9 | `contact_action_index` recomputed from `actions`+`arc` matches; points at a Forward; `local_arc ∈ [0, leg]`; null iff not collided (new-schema outcomes only) |

The full v2 run (710 records, 63,900 outcomes) passes with **0 violations**.

---

## Scripts & CLI

### `collect.py` — batch collection
```
--scenes GLB... | --auto-scenes        # auto = all 36 semantic val scenes
--max-scenes N   --poses-per-scene 20  --seed 42
--radii 0.15 0.20 0.25                 # default = config.RADII_M (home-robot chassis);
                                       #   multiple -> same-frame body counterfactuals (Q3)
--action-mode {fov_len, grid, random, file}   # default fov_len
    fov_len: length-1..6 in-FOV sequences; per (frame,length) keep --keep-per-length
             clean outcomes (drop red discard_noise, flag yellow review_depth_hole)
    grid   : turns{0,±15,±30,±45,±90} × forwards{0.5,1,1.5,2} + pure turns{±30,±45,±90}
    random : --actions-per-frame K random seqs (1–2 legs)
    file   : --action-file JSON {"sequences":[{"actions":[...]}]}
--lengths 1 2 3 4 5 6                  # fov_len: primitive counts to bucket
--keep-per-length 5  --pool-factor 3   # fov_len: kept per (frame,length); pool = keep*factor
--min-objects 1                        # skip frames with too few non-structural objects
--geodesic                             # per-relation geodesic (slow; default off)
--save-arrays                          # also dump depth .npy
--debug-images  --debug-outcomes-per-frame 4   # save overlay + N evidence PNGs/frame
--out DIR  --no-validate
```
Writes `records.jsonl`, `img/*.png`, `run_meta.json`, optional `debug/`, then
auto-validates (exit 1 on violation).

### `query.py` — manual single query
```
--scene GLB  --actions '[{"type":"turn","deg":-30},{"type":"forward","m":1.2}]'
--radius 0.25  [--seed N | --pose x y z --yaw rad]  --target INSTANCE  --geodesic  --out DIR
```
Prints collision / contact / heading / view-exit / nav-check / closest-targets and
writes `evidence.png` (RGB overlay + top-down) and `outcome.json`.

### others
```
smoke_frame.py   --scene GLB --poses 2 --seed 0 --out DIR         # P0 mask eye-check
check_records.py RECORDS.jsonl [--max-print 40]                   # validator CLI
serve_viz.py     --port 8767 --dataset data/conseq/v2            # interactive REVIEW app
build_viz.py     OUT_DIR                                          # legacy static review.html
```

**Review app (`serve_viz.py --dataset`).** A 3-column browser page (needs Habitat —
it rebuilds each frame to recover obstacle geometry):

- **left** scene → frame thumbnails; **middle** the frame RGB (object + contact
  overlay) + a filterable outcome table (length / radius / collided / verdict /
  human-check); **right** an obstacle-accurate **top-down** with a body-circle
  **step-through** (scrub / ▶ play, turns red past the contact arc), a
  step-by-step **reasoning trace**, and a distance-to-every-object table.
- The `Live` tab samples a fresh pose and judges an ad-hoc action sequence through
  the same view.
- Remote host: forward the port, e.g. `ssh -N -L 8767:localhost:8767 user@host`,
  then open `http://localhost:8767`. Restart the process + hard-refresh after code
  or dataset changes.

---

## Datasets

### v2 (current) — `data/conseq/v2/`

`collect.py --auto-scenes --poses-per-scene 20 --action-mode fov_len --radii 0.15 0.20 0.25 --seed 42`
(length-1..6 in-FOV sequences — no adjacent forwards, consecutive turns alternate direction;
per (frame,length) keep 5 clean outcomes; drop red / flag yellow):

| metric | value |
|---|---|
| scenes | 36 (local HM3D val, semantic) |
| frames | **710** (720 − 8 few-objects − 2 sample-fail) |
| bodies (chassis radii) | 0.15 / 0.20 / 0.25 m (home-robot range) |
| outcomes | **63,900** (710 × 6 lengths × 5 keep × 3 radii — every bucket full) |
| in-FOV | **100 %** (all `path_exit_arc_m` null) |
| length balance | 10,650 outcomes each for L = 1..6 |
| collision by radius | **26.0 % / 29.1 % / 32.5 %** at r = 0.15 / 0.20 / 0.25 (chassis ↑ ⇒ collision ↑) |
| collided | 18,666; `contact_action_index` names which action collides (spread across L1..L6) |
| nav_check | keep_depth 66.5 %, keep_agree 32.1 %, review_depth_hole (yellow, human-check) 1.4 %, discard_noise (red, dropped) 0.2 % |
| validation | **0 / 710 violations** (incl. V9) |
| wall-clock | ~9.5 min |

Semantic caches for all 36 scenes are in `data/conseq/semantic_cache/*.sem.npz`
(kept — required to rebuild frames / re-collect; do not delete).

> An earlier `v1` (grid actions, single radius, pre-FOV-filter, old nav_check
> taxonomy) has been removed; `v2` supersedes it.

---

## How the QA layer consumes records

The pipeline emits **only raw continuous quantities**; discretisation (left/right/
ahead, closer/farther margins, thresholds) is the QA layer's job.

| Benchmark question | Read from |
|---|---|
| what targets are in the scene | `category_inventory` + `objects[]` |
| Q1 will you collide | `outcomes[].collided` / `first_contact_arc_m` |
| Q2 what did you hit | `outcomes[].contact.category` (+ `pixel` for grounding) |
| Q3 does body matter | multiple radii ⇒ multiple `outcomes` on the same frame |
| Q5 heading / bearing / in-view | `pose_end_full.heading_deg` + `object_relations[].full.bearing_deg / in_fov` |
| Q6 closer to which target | `object_relations[].delta_full.d_centroid_m` (negative = closer); compare across outcomes for the same instance |
| view-exit consequence | `view_exit.*` (compose any definition) |

---

## Config / tunable constants

All in `config.py` (import from there, never hard-code):

```
camera      HFOV 79°, 640×480, height 1.5 m, FOV half 39.5°
body        RADII (0.15, 0.20, 0.25) home-robot chassis, height 1.5 m  (edit RADII_M)
gen         turns ±{15,30}, forwards {0.5,1,1.5}, lengths 1..6, keep 5/length, pool ×3
            (no adjacent forwards; consecutive turns must alternate direction)
perception  obstacle band [0.05, 1.5] m, voxel 5 cm, dilation 1, min support 3 voxels
march       step 2 cm, d_max 5 m
attribution margin 0.10 m
nav         is_navigable y-tol 0.5 m, agree tol 0.3 m, fwd-cover min 0.90, strong support 6
view exit   end-visible depth slack 0.3 m
objects     min area 400 px, min valid-depth 20, subsample ≤ 2000 pts
semantic    1e6 surface samples, palette match tol 12 (0–255), assign tol 0.20 m
```

---

## Testing

```bash
$PY -m pytest tests/test_pl_*.py -q     # pipeline tests, pure numpy, no Habitat
```
Covers: action path/pose known answers + arc + wrap; depth roundtrip / floor /
voxel; object extraction filters + attribution majority-vote / tie-break / relax;
end-to-end judge on a synthetic wall + object (collision, after-state, view-exit)
+ record round-trip + validate-clean; each V-check triggered by a mutation; scene
discovery.

---

## Caveats & known issues

- **`nav_check` is QA metadata; the depth voxel field *defines* the collision label.**
  The refined taxonomy classifies depth-vs-navmesh agreement from the frame's own
  obstacle voxel field + forward-cone coverage (no pixel-projection artifact):
  `keep_agree` / `keep_depth` (navmesh conservative → trust depth) / `keep_visible`
  are green; `review_depth_hole` is yellow (forward depth hole → possible miss →
  human check, `human_check: true`); `discard_noise` is red. The old `discard_hidden`
  artifact (which used to flag ~83 %) is gone — on in-FOV `fov_len` data red is ~1 %.
  The `fov_len` collector already drops red and flags yellow, so do not otherwise
  filter on `nav_check.keep`.
- **Record size.** `records.jsonl` is large (~0.9 GB for v2 / 63,900 outcomes)
  because `object_relations` are stored for **every** object (structural included) ×
  every outcome. To shrink, restrict relations to non-structural objects, or gzip.
- **Geodesic is opt-in** (`--geodesic`); it is the costliest step and euclidean
  centroid/nearest deltas already answer "closer to which target". Without it,
  `dist_geodesic_m` fields are `null`.
- **Action model** is *hard-stop at first contact, no sliding* (pure geometry, the
  agent is never actually stepped in the simulator). `pose_end_exec` is the
  realised stop; `pose_end_full` the counterfactual.
- **Single-frame perception** — the obstacle field only sees the current frustum;
  consequences depending on unseen geometry are (correctly) not detected by the
  depth path. `nav_check` records where the mesh disagrees.
- **`contact.pixel` display height** (`floor_y + 0.5`) can fall below the image at
  close range (`pixel_in_frame: false`); the contact class/arc are unaffected.
```
