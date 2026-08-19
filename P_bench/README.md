# EgoConseq-Bench

[中文说明](README.zh-CN.md)

EgoConseq-Bench generates first-person, action-conditioned consequence QA from
three simulators: R2R/Matterport3D, Habitat-GS/InteriorGS, and BEHAVIOR-1K. It
stores simulator-grounded records first, then deterministically compiles them
into one six-task candidate artifact.

This repository contains the collection, validation, compilation, evaluation,
and static-review pipeline. Raw simulator assets are not redistributed; obtain
them from their official providers under their respective licenses.

## Quick starts

Clone the outer repository, then enter this benchmark subtree:

```bash
git clone https://github.com/syp2ysy/EgoConseq-Bench.git
cd EgoConseq-Bench/P_bench
export P_BENCH_ROOT="$PWD"
```

Choose one workflow:

- **Use the frozen snapshot:** create the Habitat environment in section 1,
  download and extract both Hugging Face archives as described in
  [Current frozen training snapshot](#current-frozen-training-snapshot), then
  run the verification commands in section 5. You can immediately inspect the
  compiled QA, saved replay, and static browser; no simulator is launched.
- **Collect fresh records:** create both environments (sections 1-2), place the
  three official source datasets under `data/sources/` or set the documented
  overrides, build GS/B1K source authorities (section 3), run the three smokes
  (section 6), then launch the measured background workflow (section 7).

Restore downloaded files into real directories below `P_bench/data`; do not
replace `data/` with a symlink. Source authorities intentionally bind the
lexical source paths. The recovery snapshot is immutable evidence and cannot
be resumed with newer code; use a new output directory and manifest for fresh
collection.

## What is implemented

The active benchmark contract has exactly six tasks:

| ID | Question answered from the initial egocentric view and an action program |
| --- | --- |
| A1 | Will the robot collide? |
| A2 | During which 1-based action does the first collision occur? |
| A3 | Which initially visible object or surface is contacted first? |
| B1 | After a safe execution, how far is the selected initial-visible target? |
| B2 | After a safe execution, is that target in front, left, right, or rear? |
| C1 | Which of four real terminal renders is the true future view? |

Public input is one initial RGB image, body radius, optical-center height,
HFOV/VFOV, a canonical action sequence, and a target for B. Depth, semantic
labels, navmesh/full geometry, terminal pose, and GT certificates remain
private. Collision and safety require agreement between full-geometry and
initial-depth rollouts plus swept-corridor coverage.

`A4_checkpoint_direction` is a separate, non-headline diagnostic derived from
sealed records. It asks for a referent direction at the 25%, 50%, or 75% point
inside a Forward action. It does not change collection, the six-task registry,
or capacity stopping.

All candidate artifacts intentionally report `headline_eligible=false`.
`--gt-as-pred` checks scorer integrity; it is not a model result.

## Data flow and output layout

```text
read-only simulator assets
  -> records/<dataset>/<shard>/<scene>/records.jsonl
  -> artifacts/global/<checkpoint>/candidate_qa/
  -> artifacts/global/<checkpoint>/candidate_qa_report/index.html
```

The compiled artifact contains:

```text
candidate_qa/benchmark.json
candidate_qa/public/items.jsonl
candidate_qa/private/answers.jsonl
candidate_qa/private/atoms.jsonl
candidate_qa/private/source_map.json
candidate_qa/report.json
candidate_qa_report/index.html
```

The compiler reads sealed records and saved PNGs; it does not rerun a
simulator. `pipeline/` is the only benchmark implementation, `scripts/`
contains command-line entry points, and `tests/` contains deterministic tests.

## Current frozen training snapshot

The current recovery snapshot is train-seen, candidate-only data:

| Quantity | Value |
| --- | ---: |
| Source shards | 562 |
| Simulator-grounded records | 10,028 |
| Exact unique source frames | 9,798 |
| Six-task QA | 44,404 |
| A1 / A2 / A3 | 3,772 / 5,989 / 5,691 |
| B1 / B2 / C1 | 7,551 / 10,531 / 10,870 |

Its checkpoint is
`data/candidate_pool/abc1_train_seen_200k_20260816_451b61a/artifacts/global/recovery-562/checkpoint.json`
with SHA256
`968248000449cbaacde41871c0c30bec22d25464964cd8f41e12bfa5deb645b7`.

An access-controlled backup is hosted at the Hugging Face dataset repository
`syp115/pbench-abc1-train-seen-recovery-562`. Access to that repository is
required only to restore this snapshot; it is not required to collect fresh
data. After access is granted:

```bash
huggingface-cli download syp115/pbench-abc1-train-seen-recovery-562 \
  --repo-type dataset --local-dir /path/to/recovery-562-download

cd /path/to/recovery-562-download
sha256sum -c SHA256SUMS
cat pbench-abc1-recovery-562.tar.zst.part-* | zstd -d | \
  tar -xf - -C "$P_BENCH_ROOT"
zstd -dc pbench-supporting-fixtures.tar.zst | \
  tar -xf - -C "$P_BENCH_ROOT"
```

The extraction destination must be the actual cloned `P_bench` directory, not
a symlinked substitute. Verify the restored checkpoint before using it:

```bash
cd "$P_BENCH_ROOT"
PY="${EGOCONSEQ_HABITAT_PYTHON:-$(command -v python)}"
"$PY" scripts/check_abc_golden.py
echo "968248000449cbaacde41871c0c30bec22d25464964cd8f41e12bfa5deb645b7  data/candidate_pool/abc1_train_seen_200k_20260816_451b61a/artifacts/global/recovery-562/checkpoint.json" | \
  sha256sum -c -
"$PY" -m http.server 8789 --directory "$P_BENCH_ROOT"
# Open data/candidate_pool/abc1_train_seen_200k_20260816_451b61a/artifacts/global/recovery-562/candidate_qa_report/index.html
```

Do not report this recovery snapshot as a completed or headline benchmark. It
is a reproducible training/candidate snapshot registered in `docs/runs.json`.

## Tested system

The current code was tested on Linux with four RTX 4090 GPUs (24 GB each),
NVIDIA driver 575.57.08, and CUDA 12.x user-space packages. A single-scene
smoke needs one GPU; the background controller can schedule several GPUs.

Two Python environments are required because Isaac Sim 5.1 and the tested
Habitat stack use different Python versions:

| Runtime | Tested versions |
| --- | --- |
| Habitat/R2R/GS and controller | Python 3.9.23, Habitat-Sim 0.2.4 headless, Habitat-Lab 0.2.420230405, PyTorch 2.5.1, gsplat 1.5.3 |
| BEHAVIOR-1K | Python 3.11.15, BEHAVIOR/OmniGibson 3.9.1, Isaac Sim 5.1.0, BDDL 3.7.0, PyTorch 2.7.0+cu128 |

## 1. Create the Habitat environment

```bash
cd "$P_BENCH_ROOT"

conda env create -f environment/habitat.yml
conda activate egoconseq-habitat
export EGOCONSEQ_HABITAT_PYTHON="$(command -v python)"

python - <<'PY'
import habitat_sim, habitat, gsplat, numpy, torch
print("Habitat-Sim", habitat_sim.__version__)
print("NumPy", numpy.__version__)
print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())
PY
```

`environment/habitat.yml` mirrors the tested versions. PyTorch installs a CUDA
12 wheel; match it to a compatible NVIDIA driver. The GS collision-authority
preprocessor additionally needs `usd-core` when reading the official USD
files:

```bash
python -m pip install usd-core
```

## 2. Install BEHAVIOR-1K / OmniGibson

Use the official BEHAVIOR-1K v3.9.1 setup in a separate environment:

```bash
git clone -b v3.9.1 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

The tested checkout is tag `v3.9.1`, commit
`26f2c7ef7b9cf96bd0414f81e1e751e493762779`. Follow the official installation
guide if Isaac Sim requires a workstation-specific setup:

- BEHAVIOR installation: <https://behavior.stanford.edu/getting_started/installation.html>
- Isaac Sim 5.1 Python environment: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html>

Set the resulting interpreter through `EGOCONSEQ_B1K_PYTHON`; no repository
path is hard-coded.

## 3. Download and prepare simulator assets

### R2R / Matterport3D

1. Download the R2R-VLNCE v1-3 episode archive from
   <https://github.com/jacobkrantz/VLN-CE>.
2. Request Matterport3D academic access at
   <https://matterport.com/partners/meta> and download the MP3D scans.
3. Keep each scan's `.glb`, `.house`, `.navmesh`, and semantic PLY files under
   one MP3D scan root, together with
   `mp3d_annotated_basis.scene_dataset_config.json` at that root.

The R2R episode JSON is a scene whitelist; instructions and navigation goals
are not benchmark inputs.

### Habitat-GS / InteriorGS

Download the source assets at their pinned revisions:

- Habitat-GS scenes: <https://huggingface.co/datasets/RukawaY/gs_scenes>,
  revision `034f5938c40c55b873da81b1b6717484b40faae9`;
- InteriorGS labels: <https://huggingface.co/datasets/spatialverse/InteriorGS>,
  revision `5201ed9fd11fc2b8ac23e069796c386dbbf8f943`;
- SAGE-3D collision meshes:
  <https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh>.

Normalize each selected scene to this layout:

```text
GS_ROOT/
  train/<scene>/scene.gs.ply
  train/<scene>/scene.navmesh
  train/<scene>/labels.json
  splits/train.json
```

`splits/train.json` uses schema `egoconseq.scene_manifest.v1`; each scene row
contains `scene_id`, relative `path`, `source_scene`, and `split: "train"`.
Build the physical authority once from the official collision USDs:

```bash
"$EGOCONSEQ_HABITAT_PYTHON" scripts/build_gs_collision_authority.py \
  --data-root "$EGOCONSEQ_GS_ROOT" \
  --source-manifest "$EGOCONSEQ_GS_TRAIN_MANIFEST" \
  --collision-root "$SAGE3D_COLLISION_ROOT" --jobs 4 \
  --exclude-scene interior_0505_839970
```

The excluded scene is a known source-conversion failure. Do not silently omit
other scenes. Gaussian ellipsoids are used for rendering, never as the
collision authority.

### BEHAVIOR-1K source authority

After the official setup downloads the dataset, bind the installation and
derive per-scene authority fragments:

```bash
PY="$EGOCONSEQ_HABITAT_PYTHON"
mkdir -p "$P_BENCH_ROOT/data/b1k-authority/fragments"

"$PY" scripts/build_b1k_source_manifest.py verify-install \
  --data-root "$B1K_DATA_ROOT" \
  --source-root /path/to/BEHAVIOR-1K \
  --output "$P_BENCH_ROOT/data/b1k-authority/install.json"

"$PY" scripts/build_b1k_source_manifest.py derive-shard \
  --data-root "$B1K_DATA_ROOT" \
  --output-dir "$P_BENCH_ROOT/data/b1k-authority/fragments" \
  --scene-timeout-s 1200 --resume \
  --scenes <SCENE_ID_1> <SCENE_ID_2>

INSTALL_SHA=$(sha256sum "$P_BENCH_ROOT/data/b1k-authority/install.json" | cut -d' ' -f1)
"$PY" scripts/build_b1k_source_manifest.py assemble-catalog-audit \
  --data-root "$B1K_DATA_ROOT" \
  --install-manifest "$P_BENCH_ROOT/data/b1k-authority/install.json" \
  --expected-install-manifest-sha256 "$INSTALL_SHA" \
  --fragment-dir "$P_BENCH_ROOT/data/b1k-authority/fragments" \
  --output "$B1K_SOURCE_MANIFEST" \
  --audit-output "$B1K_CATALOG_AUDIT"
```

Run `build_b1k_source_manifest.py <subcommand> --help` for catalog sharding and
resume options. Authority derivation invokes `EGOCONSEQ_B1K_PYTHON`.

## 4. Configure local paths

```bash
cp .env.example .env.local
# Edit .env.local, then:
source .env.local
PY="$EGOCONSEQ_HABITAT_PYTHON"
cd "$P_BENCH_ROOT"
```

`.env.local` and `data/` are ignored by Git. Keep source datasets read-only.
`EGOCONSEQ_DATA_ROOT` defaults to `$P_BENCH_ROOT/data/sources`; the R2R,
Matterport3D, and GS-specific variables override only their corresponding
source when a different layout is necessary. `B1K_DATA_ROOT` similarly
defaults to `data/sources/behavior-1k-v3.9.1` in controller commands.
Before collection, the repository must be clean because each run records and
enforces the exact Git revision.

## 5. Verify the installation

```bash
"$PY" -m pytest -q
"$PY" scripts/check_abc_golden.py
```

The Golden command rebuilds the compact, authenticated three-dataset fixture,
checks all six tasks, and runs GT-as-pred replay. It does not launch a
simulator or validate collection throughput.

## 6. Run one-scene smokes

Use a scene that exists in your authenticated train-seen catalog. The examples
below keep the scientific gates unchanged and only reduce the collection size.

R2R:

```bash
REV=$(git rev-parse HEAD)
"$PY" scripts/collect.py --backend r2r --benchmark-partition train_seen \
  --scenes uNb9QFRL6hY --poses-per-scene 1 \
  --pose-candidates-per-scene 800 --ordinary-actions-per-pose 24 \
  --record-idle-stop-s 120 --scene-wallclock-stop-s 600 \
  --code-revision "$REV" --out data/smoke/r2r
```

GS:

```bash
"$PY" scripts/collect.py --backend gs --benchmark-partition train_seen \
  --scenes interior_0123_840023 --poses-per-scene 2 \
  --pose-candidates-per-scene 800 --ordinary-actions-per-pose 24 \
  --record-idle-stop-s 120 --scene-wallclock-stop-s 600 \
  --code-revision "$REV" --out data/smoke/gs
```

B1K should use the supervisor so the Isaac Sim worker is isolated:

```bash
"$PY" scripts/run_b1k_collection_shard.py run \
  --data-root "$B1K_DATA_ROOT" --source-manifest "$B1K_SOURCE_MANIFEST" \
  --output-dir data/smoke/b1k --gpu-id 0 --shard-id smoke-b1k \
  --code-revision "$REV" --scene-timeout-s 1200 \
  --scenes Pomaria_0_garden --collect-args \
  --poses-per-scene 1 --pose-candidates-per-scene 6000 \
  --ordinary-actions-per-pose 24 --record-idle-stop-s 120 \
  --scene-wallclock-stop-s 900 --benchmark-partition train_seen
```

A frame is retained when it supports at least one valid active task; it does
not need to support A1, A2, A3, B1, B2, and C1 simultaneously.

## 7. Run multi-GPU background collection

The controller uses a measured canary profile instead of guessing scene
capacity. The normal sequence is: build canary, run canary, derive profile,
build the immutable production manifest, then run it.

```bash
CANARY="$EGOCONSEQ_RUN_ROOT/my-canary"
RUN="$EGOCONSEQ_RUN_ROOT/my-train-seen-run"

"$PY" scripts/run_background_collection.py canary-build \
  --output-root "$CANARY" \
  --r2r-scenes uNb9QFRL6hY XcA2TqTSSAj \
  --gs-scenes interior_0123_840023 interior_0045_839925 \
  --b1k-scenes Pomaria_0_garden Pomaria_0_int \
  --python "$PY" --b1k-data-root "$B1K_DATA_ROOT" \
  --b1k-source-manifest "$B1K_SOURCE_MANIFEST" \
  --ordinary-actions-per-pose 24

"$PY" scripts/run_background_collection.py canary-run \
  --manifest "$CANARY/controller/canary-manifest.json"

"$PY" scripts/run_background_collection.py profile \
  --measurements "$CANARY/capacity-evidence.json" \
  --out "$CANARY/capacity-profile.json"

PROFILE_SHA=$(sha256sum "$CANARY/capacity-profile.json" | cut -d' ' -f1)
AUDIT_SHA=$(sha256sum "$B1K_CATALOG_AUDIT" | cut -d' ' -f1)
"$PY" scripts/run_background_collection.py build \
  --output-root "$RUN" --python "$PY" \
  --b1k-data-root "$B1K_DATA_ROOT" \
  --b1k-source-manifest "$B1K_SOURCE_MANIFEST" \
  --b1k-catalog-audit "$B1K_CATALOG_AUDIT" \
  --b1k-catalog-audit-sha256 "$AUDIT_SHA" \
  --capacity-profile "$CANARY/capacity-profile.json" \
  --capacity-profile-sha256 "$PROFILE_SHA" --rounds 20

nohup "$PY" scripts/run_background_collection.py run \
  --manifest "$RUN/controller/manifest.json" \
  > "$RUN/controller/nohup.log" 2>&1 &

"$PY" scripts/run_background_collection.py status \
  --manifest "$RUN/controller/manifest.json"
```

Stop through the controller rather than killing workers directly:

```bash
"$PY" scripts/run_background_collection.py stop \
  --manifest "$RUN/controller/manifest.json"
```

The controller schedules train-seen scenes only, carries accepted-pose
exclusions across catalog passes (1.5 m / 45 degrees), and publishes global QA
checkpoints. Reaching a per-dataset target is a floor, not permission to hide a
global capacity shortfall.

## 8. Compile, inspect, and evaluate

Background checkpoints already contain compiled QA and a browser. To append
the separate A4 diagnostic to a run browser:

```bash
CHECKPOINT="$RUN/artifacts/global/<checkpoint>"
"$PY" scripts/build_checkpoint_direction.py \
  --run-root "$RUN" --output "$CHECKPOINT/candidate_qa/diagnostics/checkpoint_direction.v1" \
  --update-report "$CHECKPOINT/candidate_qa_report/index.html"
```

Serve a static report from the repository root:

```bash
"$PY" -m http.server 8789 --directory "$P_BENCH_ROOT"
# Open http://127.0.0.1:8789/data/candidate_pool/<run>/artifacts/global/<checkpoint>/candidate_qa_report/index.html
```

Evaluation needs an external source-authority manifest; the artifact's private
source map cannot authorize itself:

```bash
BENCH="$RUN/artifacts/global/<checkpoint>/candidate_qa"
"$PY" scripts/eval_benchmark.py --benchmark "$BENCH" \
  --source-authority-manifest /path/to/source-authority.json \
  --gt-as-pred
```

For a real model, replace `--gt-as-pred` with `--predictions predictions.jsonl`.

## Data diversity and partitions

- Training collection is restricted to the frozen `train_seen` scene list.
- Future benchmark collection uses separate `test_unseen` scene families.
- Scene families, not individual frames, are the split boundary.
- Cross-pass pose exclusions prevent repeated nearby camera poses.
- The compiler applies deterministic per-frame/per-task diversity selection;
  exact source-frame SHA is the image identity.
- Action shortlists cover lengths L1-L6 and dynamic distance strata before
  physical certification. Failed individual tasks withhold only those tasks;
  they do not require an otherwise useful frame to satisfy all six heads.

See `pipeline/README.md` for module ownership and `docs/runs.json` for retained
artifact provenance.

## Troubleshooting

- **B1K launches the wrong Python:** set `EGOCONSEQ_B1K_PYTHON` to the Python
  inside the official BEHAVIOR environment.
- **A scene is missing:** check the frozen partition and the authenticated
  source manifest; do not add an unregistered scene by globbing.
- **Resume rejects a run:** resume is intentionally same-revision and
  same-contract only. Start a new output directory after code or contract
  changes.
- **No B target or too few C1 neighbours:** those are typed per-task
  withholds; the frame may still publish other valid tasks.
- **A candidate artifact says non-headline:** this is expected for the current
  candidate protocol.

## Licensing

This checkout does not declare a project-wide license. Simulator code and raw
assets remain subject to the Habitat, Matterport3D, Habitat-GS, InteriorGS,
SAGE-3D, Isaac Sim, OmniGibson, BDDL, and BEHAVIOR-1K licenses and access terms.
Do not redistribute restricted assets through this repository.
