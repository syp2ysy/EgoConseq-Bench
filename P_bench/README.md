# EgoConseq-Bench

EgoConseq-Bench is a single-image embodied physical-consequence VQA benchmark.
It asks whether a vision-language model can look at one egocentric RGB image,
condition on a robot body footprint and a short action, and predict the local
physical consequence before acting.

This is not a navigation-policy benchmark. There is no goal, route, SPL, or
episode rollout. The core question is much smaller:

```text
Given the current first-person image, the robot body size, and a candidate
short action, will the swept robot body contact visible local geometry?
```

The project is currently in the demo / data-generation stage. The deterministic
oracle stack is implemented, and the most complete generated task is **O5 body
counterfactual consequence**.

---

## Current Status

Implemented:

- Habitat-Sim rendering wrapper with RGB + depth sensors.
- Depth-to-point-cloud oracle for visible local geometry.
- Floor removal, obstacle voxelization, swept-cylinder `d_safe` oracle.
- Per-radius Habitat navmesh cross-check.
- Visibility / no-contact evidence gates and disagreement taxonomy.
- O5 single-robot counterfactual case generation.
- O5 review webpage and baseline report.
- Unit tests for the deterministic core and O5 schema.

Not yet fully implemented:

- O1/O2/O3/O4/O6 generators and review pages.
- VLM evaluation loop. The intended order is:
  1. generate QA + GT,
  2. human-check the review page,
  3. then connect Qwen3-VL or another VLM.

---

## Core Idea

All tasks are designed around one geometry function:

```text
d_safe(body, action)
= distance the robot body can move along the action before first contact
```

The model only sees:

```text
RGB image + natural-language question
```

Depth, point clouds, navmesh, poses, and top-down plots are used only offline
for label generation and human review.

Two distances are stored for debugging:

- `d_safe depth`: the primary label oracle. It is computed from the current
  rendered depth frame by backprojecting visible geometry into a point cloud and
  sweeping the robot footprint through that visible local obstacle field.
- `d_safe nav`: an independent Habitat navmesh cross-check. It uses the global
  simulator navmesh and can see geometry that may not be visible in the current
  RGB frame. It is not the primary GT.

The project rule is:

```text
d_safe depth defines labels.
d_safe nav is used for sanity checking / hidden-geometry detection.
```

---

## Taxonomy

The benchmark taxonomy is organized by the operation performed over
`d_safe(body, action)`, not by scene object type or surface answer format.

| ID | Name | What It Tests | Current Status |
| --- | --- | --- | --- |
| O1 | Forward Clearance | How far the robot can safely move straight forward | designed, not fully generated |
| O2 | Aperture / Lateral Passability | Whether a visible opening can fit the robot footprint | designed, deferred |
| O3 | Turn-then-Forward Sweep | Whether a turn-then-forward action contacts geometry | designed, not fully generated |
| O4 | Cross-Action Consequence Comparison | Which complete action has the best physical consequence | planned next |
| O5 | Body Counterfactual Consequence | Whether changing only body width flips contact outcome | implemented demo |
| O6 | First-Contact Target Grounding | Which visible target/region would be contacted first | designed, deferred |

Readout format is a separate axis: magnitude, threshold, binary contact,
ranking, group flip, or target id.

---

## O5 Design

O5 is the current main implemented task.

Each question describes exactly **one** cylindrical-base robot. It does not ask
the model to compare two robots in the same prompt.

Example prompt:

```text
图中是一个圆柱形底盘的机器人，底盘直径约 0.8 米。
它从当前位置朝正前方移动约 1.5 米。
只看这张第一视角图，判断它的身体在当前可见的局部空间内会不会与障碍物发生接触。
请直接给出这个动作是否会接触的结论。
```

The GT rule for one case is:

```text
contact iff d_safe(radius, forward) < H
```

where `H` is a fixed physical forward distance in metres.

Counterfactual behavior is measured across cases sharing the same `group_id`:

```text
same RGB
same pose
same forward horizon H
different chassis diameter
```

The default O5 demo uses two radii:

```text
r_small = 0.10 m  -> diameter 0.20 m
r_large = 0.40 m  -> diameter 0.80 m
```

O5 groups include:

- `flip`: small body clears, large body contacts.
- `all_no_contact`: both bodies clear.
- `all_contact`: both bodies contact.

The control groups prevent shortcuts such as always answering contact for the
large body and no-contact for the small body.

Important prompt rule: height is fixed by the simulator and is not mentioned in
the question. The prompt varies only the chassis diameter.

---

## Simulator

The simulator is **Habitat-Sim**.

The wrapper lives in:

```text
egoconseq/sim/habitat_env.py
```

It configures one Habitat agent with:

- RGB camera sensor
- Depth camera sensor
- resolution: `640 x 480`
- horizontal FOV: `79 deg`
- camera height: `1.5 m`
- Habitat default agent height/radius for the base simulator agent

Rendering returns:

```python
rgb, depth, K, agent_state = sim.render(position, yaw)
```

where:

- `rgb`: `(H, W, 3)` uint8 RGB image
- `depth`: `(H, W)` float32 depth in metres
- `K`: camera intrinsics
- `agent_state`: Habitat pose state

For navmesh checks, the code recomputes Habitat navmesh per robot radius using:

```text
habitat_sim.NavMeshSettings.agent_radius
habitat_sim.NavMeshSettings.agent_height
```

That code lives in:

```text
egoconseq/sim/navmesh.py
```

---

## Dataset

The current demo uses **HM3D v0.2 val scenes** rendered through Habitat-Sim.

The local expected root is:

```text
/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d
```

The config file is expected at:

```text
/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json
```

The default O5 generation script uses these three val scenes:

```text
val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
val/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb
val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb
```

The absolute paths are defined in `scripts/generate_o5.py`.

To run on another machine, download/prepare HM3D v0.2 for Habitat-Sim according
to the official Habitat/HM3D data-access instructions, then either:

- place it at the path above, or
- update `HM3D_ROOT` in `egoconseq/config.py`, and update the scene paths in
  `scripts/generate_o5.py`.

The generated demo data under `data/demo/` is intentionally git-ignored.

References:

- Habitat paper: https://arxiv.org/abs/1904.01201
- HM3D paper: https://arxiv.org/abs/2109.08238

---

## Environment

The current working environment is:

```text
/home/zhangshan/miniconda3/envs/qwen3vl_habitat
```

Use its Python directly:

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python -m pytest -q
```

Core dependencies used by this repo:

- Python 3
- Habitat-Sim
- numpy
- scipy
- matplotlib
- Pillow
- pytest
- numpy-quaternion

The repo does not currently provide an `environment.yml` or `requirements.txt`.
On a fresh machine, install Habitat-Sim first following the Habitat-Sim version
compatible with your HM3D assets, then install the Python dependencies above.

Quick import check:

```bash
cd /home/zhangshan/syp/myvln/P_bench
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python - <<'PY'
import habitat_sim
import numpy
import scipy
import matplotlib
from PIL import Image
print("imports ok")
PY
```

---

## Repository Layout

```text
egoconseq/
  config.py                 # frozen constants and dataset paths
  geometry.py               # swept path and projection helpers
  manifest.py               # Case schema and JSONL I/O
  sim/
    habitat_env.py          # Habitat-Sim RGB/depth render wrapper
    navmesh.py              # per-radius navmesh cross-check
  oracle/
    pointcloud.py           # depth -> point cloud, floor estimation/removal
    voxel.py                # visible obstacle voxel field
    sweep.py                # swept-cylinder d_safe oracle
    disagreement.py         # depth-vs-navmesh taxonomy
    overlay.py              # RGB sweep overlay helpers
  gates/
    visibility.py           # visible sweep and no-contact evidence gates
    sanity.py               # monotonicity and step-size sanity checks
  tasks/
    prompts.py              # prompt builders
    instantiate.py          # O5 case construction
  eval/
    baselines.py            # O5 shortcut baselines
    metrics.py              # O5 metrics
  pipeline/
    sample_poses.py         # valid pose filtering helpers

scripts/
  generate_o5.py            # O5 dataset generation
  build_o5_review.py        # O5 HTML review page
  run_baselines_report.py   # O5 baseline report

tests/
  test_*.py                 # unit tests

data/demo/
  o5/                       # generated O5 demo artifacts, git-ignored
```

---

## Basic Checks

Run the Python unit tests:

```bash
cd /home/zhangshan/syp/myvln/P_bench
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python -m pytest -q
```

Run a quick import check for the simulator stack:

```bash
cd /home/zhangshan/syp/myvln/P_bench
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python - <<'PY'
import habitat_sim
from egoconseq.sim.habitat_env import EgoConseqSim
from egoconseq.oracle.sweep import d_safe_visible
print("simulator stack imports ok")
PY
```

---

## Generate O5 Demo Data

Generate O5 cases:

```bash
cd /home/zhangshan/syp/myvln/P_bench
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/generate_o5.py \
  --max-poses 500 \
  --target-flip 40 \
  --target-all-no-contact 20 \
  --target-all-contact 20
```

This writes:

```text
data/demo/o5/manifest.jsonl
data/demo/o5/img/*.png
data/demo/o5/topdown/*.png
```

`data/demo/` is generated output and can be deleted at any time. Re-run the
command above to recreate the O5 images and manifest.

The generation pipeline is:

```text
Habitat RGB/depth render
-> depth backprojection
-> agent-local ground-frame point cloud
-> floor estimation and removal
-> obstacle VoxelField
-> d_safe_visible for small and large bodies
-> per-radius d_safe_navmesh cross-check
-> disagreement taxonomy and sanity gates
-> O5 Case objects
-> manifest + RGB + top-down evidence plots
```

---

## Build Review Page

Build the HTML review page:

```bash
cd /home/zhangshan/syp/myvln/P_bench
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/build_o5_review.py
```

This writes:

```text
data/demo/o5/review.html
```

The review page is lightweight HTML. It references image files with relative
paths instead of embedding base64, so it must be opened from the generated O5
directory or through a static file server.

Recommended way to open it:

```bash
cd /home/zhangshan/syp/myvln/P_bench
python -m http.server 8765 --bind 0.0.0.0 --directory data/demo/o5
```

Then open:

```text
http://127.0.0.1:8765/review.html
```

If your browser is on a different machine, use your IDE's port forwarding /
port preview for port `8765`, or replace `127.0.0.1` with the server address
if the network allows it.

The page shows:

- first-person RGB image
- question
- GT answer
- body radius/diameter
- `d_safe depth`
- `d_safe nav`
- action horizon `H`
- GT derivation
- top-down oracle evidence
- case id / scene / group kind

---

## Build Baseline Report

Run O5 non-vision baselines:

```bash
cd /home/zhangshan/syp/myvln/P_bench
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/run_baselines_report.py
```

This writes:

```text
data/demo/o5/report.md
```

Baselines include random, majority, blind text-only, radius-only, and geometry
oracle diagnostics. These baselines are not the model-under-test; they are used
to detect shortcuts and verify that the generated O5 groups are meaningful.

---

## Manifest Format

Each generated case is a JSON object serialized in `manifest.jsonl`.

Important fields:

- `case_id`: unique case id
- `operation_id`: currently `O5` for generated demo data
- `readout_tag`: currently `binary_contact`
- `group_id`: ties counterfactual O5 cases together
- `scene_id`: HM3D scene id
- `pose`: `[x, y, z, yaw]`
- `body`: radius and diameter
- `action`: forward action and fixed metric horizon
- `image_path`: RGB image path
- `question`: model-visible prompt
- `answer`: GT label
- `d_safe_visible_m`: primary depth oracle distance
- `d_safe_navmesh_m`: navmesh cross-check distance
- `tags.gt_evidence`: raw values and label rule

The model payload is intentionally restricted:

```python
case.model_payload() == {
    "image_path": case.image_path,
    "question": case.question,
}
```

No depth, pose, navmesh, or GT evidence is exposed to the model.

---

## Common Issues

### `review.html` exists but the browser cannot open it

Do not rely on the IDE's "Open browser" button for a remote filesystem path.
Start a static server instead:

```bash
cd /home/zhangshan/syp/myvln/P_bench
python -m http.server 8765 --bind 0.0.0.0 --directory data/demo/o5
```

Then open:

```text
http://127.0.0.1:8765/review.html
```

If `curl http://127.0.0.1:8765/review.html` cannot connect, the server is not
running.

### Images do not show in the review page

Make sure the page is served from `data/demo/o5`, because `review.html` uses
relative paths like:

```text
img/O5-...png
topdown/O5-...png
```

### Habitat cannot load scenes

Check that these files exist:

```bash
ls /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json
ls /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
```

If your HM3D root differs, update `egoconseq/config.py` and the default scene
paths in `scripts/generate_o5.py`.

### Habitat logs are noisy

Use:

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet <command>
```

---

## Development Notes

- Keep generated artifacts under `data/demo/`; this directory is git-ignored.
- Use `rg` for code search.
- Do not treat `d_safe nav` as GT. It is a cross-check for hidden geometry and
  oracle sanity.
- Do not mention robot height in prompts. Height is fixed by the simulator; O5
  only varies footprint diameter.
- O5 horizon must stay metric-fixed across body sizes. Do not convert it to
  body-width units.
- Before claiming a generated dataset is ready for VLM evaluation, inspect
  `data/demo/o5/review.html` manually.
