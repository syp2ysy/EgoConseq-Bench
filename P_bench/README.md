# EgoConseq-Bench / P_bench

EgoConseq-Bench is a single-image embodied physical-consequence VQA benchmark.
This repository folder, `P_bench/`, contains the current demo implementation.

The benchmark asks a deliberately local question:

```text
Given one egocentric RGB image, one robot body footprint, and one short action,
what physical consequence would happen before the robot acts?
```

It is not a navigation-policy benchmark. There is no long-horizon goal, no SPL,
no episode rollout, and no planner evaluation. The core object is:

```text
d_safe(body, action)
= the distance the swept robot body can move along the action before first contact
```

The VLM sees only:

```text
RGB image + natural-language question
```

Depth, point clouds, navmesh, poses, and top-down plots are offline artifacts
used only for label generation, oracle checks, and human review.

---

## Current Status

Implemented:

- Habitat-Sim RGB/depth rendering wrapper.
- Depth-to-point-cloud oracle over the currently visible local scene geometry.
- Floor estimation/removal, obstacle voxelization, and swept-cylinder `d_safe`.
- Per-radius Habitat navmesh cross-check.
- Visibility gates, no-contact evidence gates, and disagreement taxonomy.
- O5 body-counterfactual demo generation.
- O5 HTML review page.
- O5 non-vision baseline report.
- Unit tests for the deterministic core and O5 schema.

Not yet fully implemented:

- O1/O2/O3/O4/O6 generators and review pages.
- Qwen3-VL or other VLM evaluation loop.
- Final multi-scene parameter freeze for the voxel oracle.

The current end-to-end runnable task is **O5 Body Counterfactual Consequence**.

---

## Reproduction Matrix

The current local environment used to build and test this demo is:

| Component | Current Value |
| --- | --- |
| Python | `3.9.23` |
| Habitat-Sim | `0.2.4` |
| HM3D | `v0.2` |
| Required HM3D split for the current demo | `val` |
| Local conda env | `/home/zhangshan/miniconda3/envs/qwen3vl_habitat` |
| Local HM3D root | `/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d` |
| RGB/depth resolution | `640 x 480` |
| Horizontal FOV | `79 deg` |
| Camera height | `1.5 m` |

Important: this project currently uses HM3D `val` only. The local machine has
100 HM3D v0.2 val scenes. It does not require HM3D train/test data for the O5
demo. If you later scale the benchmark beyond the demo, use train/minival for
development and reserve val or another held-out split for final reporting.

---

## Repository Location

On GitHub the project is expected to live under:

```text
EgoConseq-Bench/P_bench/
```

After cloning:

```bash
git clone https://github.com/syp2ysy/EgoConseq-Bench.git
cd EgoConseq-Bench/P_bench
```

On the current development machine, the working directory is:

```bash
cd /home/zhangshan/syp/myvln/P_bench
```

All commands below assume you are inside `P_bench/`.

---

## Environment Setup

### Existing Local Environment

On the current machine, use the existing environment directly:

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python -m pytest -q
```

Confirmed package versions in this environment:

```text
python             3.9.23
habitat-sim        0.2.4
numpy              1.26.4
scipy              1.13.1
matplotlib         3.8.4
pillow             11.0.0
pytest             8.4.2
numpy-quaternion   2023.0.3
```

Quick import check:

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python - <<'PY'
import habitat_sim
import numpy
import scipy
import matplotlib
from PIL import Image
import quaternion

print("habitat_sim", getattr(habitat_sim, "__version__", "unknown"))
print("imports ok")
PY
```

### Fresh Machine Setup

Create a Python 3.9 environment:

```bash
conda create -n egoconseq python=3.9 -y
conda activate egoconseq
```

Install Habitat-Sim first. The demo was validated with `habitat-sim==0.2.4`.
Habitat-Sim wheels are platform/CUDA dependent, so if the pip wheel is not
available for your machine, install the matching Habitat-Sim 0.2.4 build from
the official Habitat-Sim instructions.

```bash
pip install habitat-sim==0.2.4
```

Install the remaining Python dependencies:

```bash
pip install \
  numpy==1.26.4 \
  scipy==1.13.1 \
  matplotlib==3.8.4 \
  pillow==11.0.0 \
  pytest==8.4.2 \
  numpy-quaternion==2023.0.3
```

Then run:

```bash
python - <<'PY'
import habitat_sim
from egoconseq.sim.habitat_env import EgoConseqSim
from egoconseq.oracle.sweep import d_safe_visible
print("simulator stack imports ok")
PY
```

---

## Dataset: HM3D v0.2

### What Data Is Required?

The current O5 demo requires:

```text
HM3D v0.2 val split
```

The code expects Habitat-compatible HM3D assets with `.basis.glb` scene files
and the HM3D scene dataset config.

Required local files include:

```text
/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/
  hm3d_annotated_basis.scene_dataset_config.json
  val/
    00800-TEEsavR23oF/
      TEEsavR23oF.basis.glb
    00801-HaxA7YrQdEC/
      HaxA7YrQdEC.basis.glb
    00802-wcojb4TFT35/
      wcojb4TFT35.basis.glb
    ...
```

The current local installation has:

```text
hm3d/
  val/                 # 100 validation scenes
  hm3d_annotated_basis.scene_dataset_config.json
```

The O5 generator currently uses three val scenes by default:

```text
val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
val/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb
val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb
```

Those absolute paths are defined in:

```text
scripts/generate_o5.py
```

The shared root is defined in:

```text
egoconseq/config.py
```

### Download HM3D v0.2 Val

HM3D requires official data access. After your Habitat/HM3D credentials are
available, use Habitat-Sim's dataset downloader.

The installed downloader exposes the following relevant HM3D groups:

```text
hm3d_val_v0.2
hm3d_train_v0.2
hm3d_minival_v0.2
hm3d_full
```

For the current demo, download only val:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hm3d_val_v0.2 \
  --data-path /home/zhangshan/syp/datasets
```

If the downloader prompts for credentials:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hm3d_val_v0.2 \
  --data-path /home/zhangshan/syp/datasets \
  --username YOUR_USERNAME \
  --password YOUR_PASSWORD
```

For a fresh machine, replace `/home/zhangshan/syp/datasets` with your dataset
root. The expected final layout is:

```text
<DATA_ROOT>/versioned_data/hm3d-0.2/hm3d/
  hm3d_annotated_basis.scene_dataset_config.json
  val/
```

Then either keep this repo's default root:

```text
/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d
```

or edit:

```text
egoconseq/config.py
scripts/generate_o5.py
```

### Verify the Dataset

Run:

```bash
ls /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json
ls /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
find /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val -mindepth 1 -maxdepth 1 -type d | wc -l
```

Expected for the current local machine:

```text
100
```

---

## Simulator Details

The simulator is **Habitat-Sim**.

The wrapper lives in:

```text
egoconseq/sim/habitat_env.py
```

It creates one Habitat agent with:

- RGB camera sensor
- depth camera sensor
- resolution `640 x 480`
- horizontal FOV `79 deg`
- camera height `1.5 m`
- Habitat default simulator agent body for rendering

The renderer returns:

```python
rgb, depth, K, agent_state = sim.render(position, yaw)
```

where:

- `rgb`: `(H, W, 3)` uint8 RGB image
- `depth`: `(H, W)` float32 depth in metres
- `K`: camera intrinsics
- `agent_state`: Habitat pose state

For navmesh checks, the code recomputes Habitat navmesh per robot radius:

```text
habitat_sim.NavMeshSettings.agent_radius
habitat_sim.NavMeshSettings.agent_height
```

That code lives in:

```text
egoconseq/sim/navmesh.py
```

---

## Taxonomy

EgoConseq-Bench organizes questions by the operation performed over
`d_safe(body, action)`, not by scene object type or surface answer format.

| ID | Name | Main Capability | Current Status |
| --- | --- | --- | --- |
| O1 | Forward Clearance | Estimate ego-scaled free space straight ahead | designed |
| O2 | Aperture / Lateral Passability | Account for body width and lateral clearance | designed |
| O3 | Turn-then-Forward Sweep | Project a turn-plus-forward action into the scene | designed |
| O4 | Cross-Action Consequence Comparison | Compare multiple complete action commands | planned next |
| O5 | Body Counterfactual Consequence | Change only body size and check whether the consequence flips | implemented demo |
| O6 | First-Contact Target Grounding | Identify the visible first-contact region | designed |

Readout format is a separate axis:

```text
magnitude / threshold / binary_contact / ranking / group_flip / target-id
```

---

## O5 Task Design

O5 is the current runnable task.

Each question describes exactly **one** cylindrical-base robot. The prompt does
not compare two robots in the same question.

Example prompt:

```text
图中是一个圆柱形底盘的机器人，底盘直径约 0.8 米。
它从当前位置朝正前方移动约 1.5 米。
只看这张第一视角图，判断它的身体在当前可见的局部空间内会不会与障碍物发生接触。
请直接给出这个动作是否会接触的结论。
```

The single-case GT rule is:

```text
contact iff d_safe(radius, forward) < H
```

where `H` is a fixed physical forward distance in metres.

Counterfactual behavior is measured across cases sharing the same `group_id`:

```text
same RGB
same scene
same pose
same action direction
same physical horizon H
different chassis diameter
```

The demo uses:

```text
r_small = 0.10 m  -> diameter 0.20 m
r_large = 0.40 m  -> diameter 0.80 m
```

O5 groups:

- `flip`: small body clears, large body contacts.
- `all_no_contact`: both bodies clear.
- `all_contact`: both bodies contact.

The control groups prevent shortcuts such as always predicting contact for the
large body and no contact for the small body.

Prompt rule:

```text
Only chassis diameter varies.
Robot height is fixed by the simulator and is not mentioned in the prompt.
```

---

## Oracle Pipeline

The O5 generation pipeline is:

```text
1. Load one HM3D val scene in Habitat-Sim
2. Sample a navigable agent pose
3. Render RGB and depth from the current pose
4. Backproject the depth map into a 3D point cloud
5. Convert the point cloud into the agent-local ground frame
6. Estimate and remove the floor
7. Keep obstacle points in the robot-height band
8. Voxelize visible obstacle geometry
9. Sweep a circular robot footprint through the visible voxel field
10. Compute d_safe depth for each body radius
11. Recompute Habitat navmesh per radius
12. Compute d_safe nav as an independent sanity check
13. Apply disagreement, visibility, and sanity gates
14. Choose a fixed forward horizon H
15. Emit two single-robot O5 cases under one counterfactual group_id
16. Save RGB image, top-down evidence plot, and manifest JSONL
```

Two `d_safe` values are stored:

- `d_safe depth`: the primary GT oracle. It uses only the currently rendered
  depth frame, so labels are tied to visible single-image evidence.
- `d_safe nav`: an independent Habitat navmesh cross-check. It uses simulator
  scene geometry and may include geometry outside the current image.

Project rule:

```text
d_safe depth defines labels.
d_safe nav is for sanity checking and hidden-geometry diagnostics.
```

---

## End-to-End Quick Start

### 1. Run Unit Tests

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python -m pytest -q
```

Expected current result:

```text
117 passed
```

### 2. Generate O5 Demo Data

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/generate_o5.py \
  --max-poses 500 \
  --target-flip 40 \
  --target-all-no-contact 20 \
  --target-all-contact 20
```

Outputs:

```text
data/demo/o5/manifest.jsonl
data/demo/o5/img/*.png
data/demo/o5/topdown/*.png
```

`data/demo/` is generated output and is intentionally git-ignored. It can be
deleted and regenerated.

### 3. Build the HTML Review Page

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/build_o5_review.py
```

Output:

```text
data/demo/o5/review.html
```

The review page shows:

- first-person RGB image
- model-visible question
- GT answer
- body radius and diameter
- fixed action horizon `H`
- `d_safe depth`
- `d_safe nav`
- GT derivation
- top-down oracle evidence
- case id, scene id, group id, and group kind

### 4. Open the HTML Review Page

Do not rely on an IDE's "Open browser" button for a remote filesystem path.
Serve the generated directory:

```bash
python -m http.server 8765 --bind 0.0.0.0 --directory data/demo/o5
```

Then open:

```text
http://127.0.0.1:8765/review.html
```

If the browser is on your local laptop but the code runs on a remote server,
forward or preview port `8765` in your IDE/SSH setup. If direct networking is
allowed, replace `127.0.0.1` with the server address.

Check the server from the shell:

```bash
curl http://127.0.0.1:8765/review.html
```

If this cannot connect, the static server is not running.

### 5. Build the Baseline Report

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/run_baselines_report.py
```

Output:

```text
data/demo/o5/report.md
```

The report includes random, majority, blind text-only, radius-only, and oracle
diagnostic baselines. These are not the final model-under-test; they are used
to detect shortcuts and confirm that O5 groups are meaningful.

---

## Generated File Layout

After running O5 generation and review:

```text
data/demo/o5/
  manifest.jsonl
  review.html
  report.md
  img/
    O5-...-r010.png
    O5-...-r040.png
  topdown/
    O5-...-r010.png
    O5-...-r040.png
```

`review.html` uses relative paths:

```text
img/O5-...png
topdown/O5-...png
```

That is why the recommended command serves `data/demo/o5` as the web root.

---

## Manifest Format

Each generated case is one JSON object in:

```text
data/demo/o5/manifest.jsonl
```

Important fields:

- `case_id`: unique case id.
- `operation_id`: currently `O5`.
- `readout_tag`: currently `binary_contact`.
- `group_id`: ties counterfactual O5 cases together.
- `scene_id`: HM3D scene id.
- `pose`: `[x, y, z, yaw]`.
- `body`: radius and diameter.
- `action`: forward action and fixed metric horizon.
- `image_path`: RGB image path.
- `question`: model-visible prompt.
- `answer`: GT label.
- `d_safe_visible_m`: primary depth oracle distance.
- `d_safe_navmesh_m`: navmesh cross-check distance.
- `tags.gt_evidence`: raw values and label rule.

The model payload is intentionally restricted:

```python
case.model_payload() == {
    "image_path": case.image_path,
    "question": case.question,
}
```

No depth, pose, navmesh, top-down plot, or GT evidence is exposed to the model.

---

## Repository Layout

```text
P_bench/
  README.md
  goal.md
  pytest.ini

  egoconseq/
    config.py                 # constants and dataset paths
    geometry.py               # swept path and projection helpers
    manifest.py               # Case schema and JSONL I/O
    sim/
      habitat_env.py          # Habitat-Sim RGB/depth wrapper
      navmesh.py              # per-radius navmesh cross-check
    oracle/
      pointcloud.py           # depth -> point cloud
      voxel.py                # visible obstacle voxel field
      sweep.py                # swept-cylinder d_safe oracle
      disagreement.py         # depth-vs-navmesh taxonomy
      overlay.py              # RGB sweep overlay helpers
    gates/
      visibility.py           # visible sweep / no-contact evidence gates
      sanity.py               # monotonicity and step-size checks
    tasks/
      prompts.py              # prompt builders
      instantiate.py          # O5 case construction
    eval/
      baselines.py            # O5 shortcut baselines
      metrics.py              # O5 metrics
    pipeline/
      sample_poses.py         # valid pose filtering helpers

  scripts/
    generate_o5.py            # O5 generation
    build_o5_review.py        # O5 HTML review
    run_baselines_report.py   # O5 baseline report

  tests/
    test_*.py

  data/demo/
    o5/                       # generated artifacts, git-ignored
```

---

## Common Issues

### Habitat Cannot Load HM3D Scenes

Check the dataset files:

```bash
ls /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json
ls /home/zhangshan/syp/datasets/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
```

If your HM3D root is different, update:

```text
egoconseq/config.py
scripts/generate_o5.py
```

### `review.html` Exists but the Browser Cannot Open It

The generated HTML references images by relative paths. Start a static server:

```bash
python -m http.server 8765 --bind 0.0.0.0 --directory data/demo/o5
```

Open:

```text
http://127.0.0.1:8765/review.html
```

If `curl http://127.0.0.1:8765/review.html` fails, the server is not running or
the port is not forwarded.

### Images Do Not Show in the Review Page

Serve the page from `data/demo/o5`, not from the repository root. The HTML uses:

```text
img/...
topdown/...
```

### Habitat Logs Are Noisy

Use:

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet <command>
```

---

## Development Rules

- Keep generated artifacts under `data/demo/`; this directory is git-ignored.
- Do not treat `d_safe nav` as GT. It is a cross-check only.
- Do not expose depth, pose, navmesh, or oracle evidence to the VLM.
- Do not mention robot height in O5 prompts. Height is fixed by the simulator.
- O5 must keep the physical horizon `H` fixed across body sizes.
- O5 prompts should describe one robot per question, not compare two robots in
  the same prompt.
- Before VLM evaluation, inspect `data/demo/o5/review.html` manually.

---

## References

- Habitat-Sim: https://github.com/facebookresearch/habitat-sim
- Habitat-Lab: https://github.com/facebookresearch/habitat-lab
- Habitat paper: https://arxiv.org/abs/1904.01201
- HM3D paper: https://arxiv.org/abs/2109.08238
