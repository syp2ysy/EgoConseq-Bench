# Diverse 200K Train Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect at least 200,000 diverse training QA from R2R, GS, and B1K without touching reserved `test_unseen` scenes.

**Architecture:** Keep the existing collector, physical authorities, QA templates, and four-GPU controller. Use K=24 ordinary actions per accepted pose, schedule only `train_seen` scenes, stop on unique-frame and scene-family capacity, and apply the existing `diversity-selection.v1` once at compile checkpoints. A4 remains a non-headline diagnostic compiled once after collection.

**Tech Stack:** Python 3.9, NumPy, Habitat, gsplat, OmniGibson, pytest, existing background controller.

**Spec:** `AGENTS.md`, `pipeline/assets/scene_partitions.v1.json`, and the K18/K24 canary evidence under `data/candidate_pool/abc1_k*_canary_20260816_2ecc032/`.

## Global Constraints

- Do not change collision, safety, coverage, stability, B-target, or C1 GT gates.
- Do not add source replay, perceptual-hash gates, CLIP filtering, or extra validation passes.
- Use K=24; K18 undersupplies A1/A2 and K36 has negligible R2R/GS gain at substantially higher cost.
- Collect only `train_seen`; preserve all `test_unseen` scenes untouched.
- Use raw RGB SHA-256 as the frame identity and the existing 1.5 m / 45 degree pose exclusions across passes.
- Require at least 160K six-task headline QA, cap A4 at 60K, and require at least 200K combined training QA. A1>=15K is reported but is not a blocking quota.
- Run targeted tests during edits, then exactly one full pytest and one Golden gate before the pipeline commit.
- Keep the complete K24 canary tree until production manifest construction is complete because the capacity profile reopens that evidence.

---

### Task 1: Make production scheduling partition-aware

**Files:**
- Modify: `scripts/run_background_collection.py:770-829`
- Modify: `pipeline/background_collection.py:347-720`
- Test: `tests/test_pl_background_collection_closeout.py`
- Test: `tests/test_pl_scene_partitions.py`

**Interfaces:**
- Consumes: `scene_partitions.load().select_catalog(dataset, specs, benchmark_partition="train_seen")`
- Produces: a controller manifest whose `scene_catalog` contains scheduled train scenes and whose new `source_scene_catalog` contains the complete source catalog used by the capacity profile.

- [ ] **Step 1: Add a failing controller test**

  Construct full synthetic catalogs containing one `train_seen` and one `test_unseen` scene per dataset. Assert that every scheduled job and every `scene_catalog` row contains only the train scene, while `source_scene_catalog` retains both source scenes.

- [ ] **Step 2: Separate source catalog from scheduled catalog**

  In `_build`, retain the discovered catalogs and derive three train-only lists with the committed partition asset:

  ```python
  partitions = scene_partitions.load()
  r2r_train = partitions.select_catalog(
      "r2r", r2r_specs, benchmark_partition="train_seen")
  gs_train = partitions.select_catalog(
      "gs", gs_specs, benchmark_partition="train_seen")
  b1k_train = partitions.select_catalog(
      "b1k", b1k_specs, benchmark_partition="train_seen")
  ```

  Pass train-only scene IDs as the scheduled catalogs and full discovered scene IDs as `source_scene_catalog`.

- [ ] **Step 3: Update manifest semantics**

  Bump the controller schema once, with no compatibility adapter. Validate:

  - `source_scene_catalog` counts match the K24 profile: R2R=61, GS=54, B1K=50;
  - `scene_catalog` contains only R2R=61, GS=42, B1K=40 train scenes;
  - canary scenes are members of scheduled train scenes;
  - every scheduled scene is a subset of its source catalog;
  - the B1K audit/source manifest still binds the complete accepted source set of 50 scenes, not the scheduled subset of 40.

- [ ] **Step 4: Remove the unnecessary production-revision equality**

  Keep `canary_revision` in the profile as provenance, but remove the production check `profile["canary_revision"] == revision`. Capacity is a performance estimate, not GT authority. Continue requiring the same source manifest identities and K=24. This lets the completed K24 canary authorize a controller-only scheduling fix without rerunning simulation.

- [ ] **Step 5: Run only focused tests**

  ```bash
  PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
  $PY -m pytest -q \
    tests/test_pl_scene_partitions.py \
    tests/test_pl_background_collection_closeout.py
  ```

  Expected: all selected tests pass; no simulator starts.

---

### Task 2: Give the 200K target a measured frame buffer

**Files:**
- Modify: `pipeline/config.py:215-219`
- Test: `tests/test_pl_background_collection_closeout.py`

**Interfaces:**
- Produces: `BACKGROUND_MIN_UNIQUE_FRAMES_BY_DATASET = {"r2r": 15000, "gs": 15000, "b1k": 6000}`.

- [ ] **Step 1: Add a failing quota assertion**

  Assert the controller manifest publishes 15K/15K/6K unique-frame minima and still requires at least 12 scene families with at most 10% of frames from one family.

- [ ] **Step 2: Change only the frame targets**

  The existing diversity run retained 1,570 six-task QA plus 545 A4 items from 345 source frames, about 6.13 usable QA per frame. The K24 canary retained 461 headline QA from 96 unique frames, or 4.80 headline QA per frame. Thirty-two thousand frames project to about 196K combined QA, so use 36K frames for a measured buffer. Do not change per-frame caps or GT gates.

- [ ] **Step 3: Run the focused quota test**

  ```bash
  $PY -m pytest -q tests/test_pl_background_collection_closeout.py
  ```

---

### Task 3: Add one global composition stop with R2R overflow

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/background_collection.py:1189-1210`
- Modify: `pipeline/checkpoint_direction.py:398-565`
- Test: `tests/test_pl_background_collection_closeout.py`
- Test: `tests/test_pl_checkpoint_direction.py`

**Interfaces:**
- Produces: `BACKGROUND_MIN_HEADLINE_ITEMS_GLOBAL = 160000`, `BACKGROUND_MAX_A4_ITEMS = 60000`, and a dataset stop decision that uses R2R as the only overflow source after per-dataset floors are met.

- [ ] **Step 1: Add failing stop tests**

  Cover these exact cases using compiled checkpoint summaries:

  - GS and B1K may stop when their own frame/family/task floors are complete;
  - R2R may not stop at its own floor while global six-task headline QA is below 160K;
  - R2R stops once its own floor and the 160K global headline floor are both complete;
  - A1 below 15K is reported in final metrics but does not keep collection running.

- [ ] **Step 2: Implement the minimal overflow rule**

  Sum `total_supported_items` from the latest source-bound dataset quota checkpoints. Keep the existing per-dataset quota as a floor. Return `False` from `dataset_can_stop` for R2R while the global headline total is below 160K; GS and B1K retain their current per-dataset stop behavior. This directs surplus collection to the 61-scene R2R catalog instead of repeatedly probing a saturated GS catalog.

- [ ] **Step 3: Run the focused stop tests**

  ```bash
  $PY -m pytest -q tests/test_pl_background_collection_closeout.py
  ```

- [ ] **Step 4: Add the deterministic A4 run cap**

  After existing per-frame selection, if A4 exceeds 60K, retain the 60K lowest stable item-ID hashes and filter answers by the same ID set. Record input, retained, and dropped counts in `report.json`. Add a synthetic 60,001-item test proving the cap is exactly 60K and deterministic. This changes no GT and normally runs only once after collection.

---

### Task 4: Perform the single verification and commit

**Files:**
- Modify only files from Tasks 1-2 and their tests.

- [ ] **Step 1: Run one full suite**

  ```bash
  $PY -m pytest -q
  ```

  Expected: 100% pass and zero skip.

- [ ] **Step 2: Run one Golden gate**

  ```bash
  $PY scripts/check_abc_golden.py
  ```

  Expected: exact Golden rebuild, all six tasks present, and GT-as-pred=1.0.

- [ ] **Step 3: Commit the narrow change**

  ```bash
  git add pipeline/config.py pipeline/background_collection.py \
    scripts/run_background_collection.py \
    tests/test_pl_background_collection_closeout.py \
    tests/test_pl_scene_partitions.py
  git commit -m "fix: schedule diverse train collection"
  ```

---

### Task 5: Freeze the K24 capacity profile

**Files:**
- Read: `data/candidate_pool/abc1_k24_canary_20260816_2ecc032/capacity-evidence.json`
- Create: `data/candidate_pool/abc1_k24_canary_20260816_2ecc032/capacity-profile.json`

- [ ] **Step 1: Derive the profile once**

  ```bash
  K24=data/candidate_pool/abc1_k24_canary_20260816_2ecc032
  $PY scripts/run_background_collection.py profile \
    --measurements "$K24/capacity-evidence.json" \
    --out "$K24/capacity-profile.json"
  ```

- [ ] **Step 2: Read the operational values once**

  ```bash
  jq '{ordinary_actions_per_pose, datasets: (.datasets | map_values({
    records_per_scene, pose_attempt_cap, scene_wallclock_s,
    pooled_unique_frame_yield_per_record
  }))}' "$K24/capacity-profile.json"
  ```

  Expected: `ordinary_actions_per_pose` is exactly 24. This is a one-time configuration read, not another validation loop.

---

### Task 6: Build and launch the four-GPU train collection

**Files:**
- Create: `data/candidate_pool/abc1_train_seen_200k_20260816_<sha7>/controller/manifest.json`
- Create: the controller state/log files under the same run.

- [ ] **Step 1: Build a 20-pass maximum manifest**

  ```bash
  REV=$(git rev-parse HEAD)
  RUN="data/candidate_pool/abc1_train_seen_200k_20260816_${REV:0:7}"
  K24=data/candidate_pool/abc1_k24_canary_20260816_2ecc032
  PROFILE_SHA=$(jq -r .sha256 "$K24/capacity-profile.json")
  B1K_ROOT=/home/zhangshan/syp/datasets/behavior-1k-v3.9.1
  AUDIT="$B1K_ROOT/pbench-abc1-task4/catalog-authority-audit-v2-20260812/catalog-authority-audit.json"
  AUDIT_SHA=$(sha256sum "$AUDIT" | cut -d' ' -f1)

  $PY scripts/run_background_collection.py build \
    --output-root "$RUN" \
    --b1k-catalog-audit "$AUDIT" \
    --b1k-catalog-audit-sha256 "$AUDIT_SHA" \
    --capacity-profile "$K24/capacity-profile.json" \
    --capacity-profile-sha256 "$PROFILE_SHA" \
    --rounds 20
  ```

  Expected scheduled scenes per pass: R2R=61, GS=42, B1K=40. The 20 passes are a ceiling; each dataset stops after its first complete pass satisfying its quota.

- [ ] **Step 2: Launch one background controller**

  ```bash
  nohup "$PY" scripts/run_background_collection.py run \
    --manifest "$RUN/controller/manifest.json" \
    >"$RUN/controller/run.log" 2>&1 &
  echo $! >"$RUN/controller/controller.pid"
  ```

  The existing scheduler assigns the next scene to the least-loaded GPU. Do not start separate R2R, GS, or B1K controllers.

- [ ] **Step 3: Use the frozen per-scene budgets**

  - K=24 ordinary actions per accepted pose;
  - accepted-record target=20 per scene;
  - pose attempts: R2R=800, GS=800, B1K=6000;
  - wallclock: R2R=600 s, GS=600 s, B1K=900 s;
  - no-new-record stop=120 s;
  - pass 1 covers every train scene before any revisit;
  - pass 2+ reuse the existing 1.5 m / 45 degree exclusions.

---

### Task 7: Monitor by pass and finalize training artifacts

**Files:**
- Read: production controller state and pass-level compiled QA.
- Create: final A4 diagnostic under `candidate_qa/diagnostics/checkpoint_direction.v1/`.

- [ ] **Step 1: Check one status snapshot per catalog pass**

  ```bash
  $PY scripts/run_background_collection.py status \
    --manifest "$RUN/controller/manifest.json"
  ```

  Track only: unique frames, scene-family count/fraction, six task counts, failed scenes, and current pass. Do not rerun record replay or Golden during collection.

  At pass 3 and pass 4, record `headline QA / unique frame` and per-dataset frame growth. If GS growth is flattening, do not raise its pose budget; the R2R overflow rule supplies the remaining headline capacity.

- [ ] **Step 2: Let the existing compile schedule run**

  The controller performs one early compile after two usable scenes per dataset and one global compile at each pass boundary. Both use the same `diversity-selection.v1`; there is no per-scene full-dataset recompilation.

- [ ] **Step 3: Require the final train capacity**

  - unique frames: R2R>=15K, GS>=15K, B1K>=6K;
  - at least 12 train scene families per dataset;
  - no scene family exceeds 10% of a dataset's frames;
  - each six-task dataset has >=1K items per task and L1-L6 coverage;
  - global six-task headline QA>=160K;
  - A1>=15K is a reported capacity diagnostic, not a stop gate;
  - `capacity_shortfall` is an honest terminal result after pass 20; do not loosen GT or pose diversity to hide it.

- [ ] **Step 4: Compile A4 once after collection**

  ```bash
  $PY scripts/build_checkpoint_direction.py --run-root "$RUN"
  ```

  Keep A4 separate with `headline_eligible=false`, max three A4 items per frame, a deterministic global cap of 60K, and no new simulation.

- [ ] **Step 5: Freeze the training release**

  Count the final six-task QA plus the capped A4 artifact. Require headline>=160K, A4<=60K, and at least 200,000 total diverse QA. If the total is short, resume the R2R overflow queue for another already-declared catalog pass; do not change selectors or gates. Register the completed run in `docs/runs.json` only after the final artifact and browser are built.

## Expected Outcome

- K24 is used everywhere; K36 is discarded.
- Training collection draws from 143 train scenes per pass and never touches 33 reserved test scenes.
- Approximately 36K unique source frames provide a measured buffer above 200K final training QA.
- Image diversity is enforced by frame-first quotas and pose exclusions; action diversity is enforced by K24 stratified shortlist cells.
- The only recurring compile work is one pass-boundary global compile using the same selector as the final export.
