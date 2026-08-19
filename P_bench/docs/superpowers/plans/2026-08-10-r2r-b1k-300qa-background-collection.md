# R2R + B1K 300-QA Background Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a clean-revision, resumable, four-GPU R2R+B1K collection that authenticates at least 200 offered actions per counted pose, retains at most 30 valid outcomes, processes at least 200 poses per scene, and targets at least 300 candidate questions per dataset and subtask.

**Architecture:** Collection policy remains in the existing collector and is bound into each run contract.  A new simulator-free controller constructs scene-disjoint production jobs, runs a four-job production canary, validates durable outputs, and promotes automatically into R2R A/B, B1K A/B, R2R C1, and B1K C1 waves.  A separate pure auditor reads records/funnels/candidate artifacts and records pose, action-bank, source-validation, and task-quota status.

**Tech Stack:** Python 3.9, NumPy, Habitat, OmniGibson, existing transactional JSONL record spool, `subprocess`, `nohup`, four local RTX 4090 GPUs.

## Global Constraints

- Use `/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python` for repository code and the existing Behavior-1K Python selected by `b1k_source_builder` for OmniGibson children.
- R2R and B1K only; do not open GS or HM3D scenes.
- Main A/B requires at least 200 accepted pose records per registered scene.
- C1 processes at least 200 authenticated pose opportunities per scene and targets at least 300 dataset-wide candidate families.
- Every counted search-ready pose authenticates at least 200 unique offered action programs.
- The deterministic main selector retains 2--30 valid outcomes and never pads.
- Do not change collision, safety, publication, coverage, stability, semantic, or human-calibration thresholds.
- C1 remains candidate-only and `headline_eligible=false` until its independent gate is calibrated.
- Use exactly one job per GPU, force BLAS/OpenMP threads to one, and persist controller state after every transition.
- Preserve the unrelated untracked `docs/2026-08-10-benchmark-task-design.md`.

---

### Task 1: Bind the 200-action / 30-outcome collection policy

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/collection_cli.py`
- Modify: `pipeline/collection_runtime.py`
- Test: `tests/test_pl_collect_resume.py`
- Test: `tests/test_pl_action_sampling.py`

**Interfaces:**
- Produces: CLI field `min_materialized_actions_per_pose: int`.
- Produces: `collection_runtime.materialized_action_count(pools) -> int`.
- Consumes: existing `action_proposal.action_bank_manifest` and deterministic `action_sampling.select_natural_action_groups`.

- [ ] **Step 1: Write failing policy tests**

Add tests asserting:

```python
assert config.MAIN_ACTION_PROPOSAL_PER_LENGTH == 40
assert config.ACTION_CANDIDATE_MAX_PER_POSE == 30
args = collection_cli.build_parser().parse_args(base_args + [
    "--min-materialized-actions-per-pose", "200",
])
assert args.min_materialized_actions_per_pose == 200
```

Add a pose-bank test with 199 unique programs that returns the existing pose
state as rejected and increments `materialized_action_bank_shortfall`, plus a
200-program case that proceeds.  Assert the field changes
`collection_run_contract` and is therefore source-bound.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_collect_resume.py tests/test_pl_action_sampling.py -q
```

Expected: failures for the old values, missing CLI option, and missing bank-size gate.

- [ ] **Step 3: Implement the minimal policy**

Set:

```python
ACTION_CANDIDATE_MAX_PER_POSE = 30
MAIN_ACTION_PROPOSAL_PER_LENGTH = 40
```

Add a nonnegative CLI integer `--min-materialized-actions-per-pose`, defaulting
to zero for explicit small fixture/file runs.  In `_prepare_pose_candidates`,
after `_materialize_pose_pools` and before any full geometry precheck, compute
the number of unique tags.  Reject duplicates and reject a bank smaller than
the requested minimum.  The production manifest must pass `200`.

- [ ] **Step 4: Run focused tests and observe GREEN**

Run the command from Step 2. Expected: exit 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add pipeline/config.py pipeline/collection_cli.py \
  pipeline/collection_runtime.py tests/test_pl_collect_resume.py \
  tests/test_pl_action_sampling.py
git commit -m "feat: expand authenticated action collection"
```

### Task 2: Add pure production manifest and audit logic

**Files:**
- Create: `pipeline/background_collection.py`
- Create: `tests/test_pl_background_collection.py`

**Interfaces:**
- Produces: `build_manifest(*, revision: str, output_root: Path, r2r_scenes: Sequence[str], b1k_scenes: Sequence[str], paths: Mapping[str, str]) -> dict`.
- Produces: `validate_job_output(job: Mapping, *, minimum_actions: int = 200, minimum_main_poses: int = 200) -> dict`.
- Produces: `task_quota_status(benchmark_paths: Sequence[Path], *, minimum: int = 300) -> dict`.
- Produces: schema `egoconseq.r2r-b1k-background-controller.v1`.

- [ ] **Step 1: Write failing manifest tests**

Create synthetic scene catalogs and assert:

```python
manifest = build_manifest(
    revision="a" * 40,
    output_root=tmp_path / "run",
    r2r_scenes=[f"r{i}" for i in range(8)],
    b1k_scenes=[f"b{i}" for i in range(8)],
    paths=trusted_paths,
)
assert manifest["schema"] == "egoconseq.r2r-b1k-background-controller.v1"
assert len(manifest["waves"]["canary"]) == 4
assert {job["gpu_id"] for job in manifest["waves"]["canary"]} == {0, 1, 2, 3}
assert all("--poses-per-scene 200" in job["command"] for job in all_jobs)
assert all("--pose-candidates-per-scene 4000" in job["command"] for job in all_jobs)
assert all("--proposal-natural-per-length 40" in job["command"] for job in all_jobs)
assert all("--proposal-pairs-per-length 12" in job["command"] for job in all_jobs)
assert all("--min-materialized-actions-per-pose 200" in job["command"] for job in all_jobs)
```

Assert scene partitions are disjoint and exhaustive, output directories are
unique, no two jobs in a wave share a GPU, C1 commands contain
`--c1-suffix-candidates`, and B1K commands use the existing isolated supervisor.

- [ ] **Step 2: Write failing output-audit tests**

Construct tiny synthetic funnels/records and assert rejection for:

- 199 accepted main poses;
- one materialized bank of 199 entries;
- source validation violations;
- incomplete formal length coverage;
- missing run metadata or records;
- a candidate benchmark with any task count below 300; and
- a C1 artifact claiming `headline_eligible=true` without calibrated authority.

- [ ] **Step 3: Run tests and observe RED**

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_background_collection.py -q
```

Expected: import failure because the module is absent.

- [ ] **Step 4: Implement manifest and audit functions**

Keep this module simulator-free.  Commands must bind the clean revision, four
GPU IDs, one output per shard, production budgets, `--resume`, no arrays, and
thread environment variables.  Use canonical JSON and atomic writes from
`pipeline.io_utils`; do not duplicate record or source validators.

- [ ] **Step 5: Run tests and observe GREEN**

Run the command from Step 3. Expected: exit 0.

- [ ] **Step 6: Commit Task 2**

```bash
git add pipeline/background_collection.py tests/test_pl_background_collection.py
git commit -m "feat: plan quota-aware background collection"
```

### Task 3: Add the resumable local four-GPU controller

**Files:**
- Create: `scripts/run_background_collection.py`
- Modify: `pipeline/background_collection.py`
- Modify: `tests/test_pl_background_collection.py`

**Interfaces:**
- CLI: `build --output-root PATH` writes `controller/manifest.json`.
- CLI: `run --manifest PATH` executes waves and writes `controller/state.json`.
- CLI: `status --manifest PATH` prints JSON status without mutation.
- Consumes: Task 2 manifest and validation functions.

- [ ] **Step 1: Write failing state-machine tests**

Use injected fake `Popen`, clock, GPU-memory reader, and job validator.  Assert:

```python
state = scheduler.tick(state)
assert running_gpu_ids(state) == {0, 1, 2, 3}
assert no_gpu_overlap(state)
```

Cover successful canary auto-promotion, failed canary blocking later waves,
two bounded OOM retries, non-OOM `stuck`, 90-minute no-progress detection,
restart recovery, expected-output-based completion, and atomic state writes.

- [ ] **Step 2: Run tests and observe RED**

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_background_collection.py -q
```

Expected: missing scheduler and CLI entry points.

- [ ] **Step 3: Implement the controller**

The controller must never import Habitat or OmniGibson.  Launch each child in
its own process group, redirect stdout/stderr to its declared log, record PID
and start time, and poll every 30 seconds.  A job completes only when its output
auditor succeeds.  On restart, a live recorded PID stays running; a dead PID is
reclassified from durable output and log evidence.

For local launch, assign `CUDA_VISIBLE_DEVICES=<gpu>` to every child and
`OMNIGIBSON_GPU_ID=<gpu>` to B1K children.  Set `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.  Refuse dirty revisions,
pre-existing nonempty output roots, GPUs above 500 MiB before the first wave,
or projected storage that would leave under 120 GiB free.

- [ ] **Step 4: Run focused tests and observe GREEN**

Run the command from Step 2. Expected: exit 0.

- [ ] **Step 5: Commit Task 3**

```bash
git add pipeline/background_collection.py scripts/run_background_collection.py \
  tests/test_pl_background_collection.py
git commit -m "feat: orchestrate local collection waves"
```

### Task 4: Verify, pin, build the run manifest, and launch `nohup`

**Files:**
- Create at runtime: `data/candidate_pool/<run>/controller/manifest.json`
- Create at runtime: `data/candidate_pool/<run>/controller/state.json`
- Create at runtime: `data/candidate_pool/<run>/controller/controller.log`

**Interfaces:**
- Consumes the clean Git revision produced by Tasks 1--3.
- Produces a live controller PID and four canary child PIDs.

- [ ] **Step 1: Run final code gates**

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_background_collection.py \
  tests/test_pl_collect_resume.py tests/test_pl_action_sampling.py -q
$PY -m pytest -q
$PY scripts/check_abc_golden.py
git diff --check
```

Expected: every command exits 0; Golden reports the active 19-file manifest
byte-identical and unchanged coverage.

- [ ] **Step 2: Commit any final test-only corrections and assert clean code**

```bash
test -z "$(git status --porcelain --untracked-files=no)"
REVISION=$(git rev-parse HEAD)
```

The unrelated user document may remain untracked and must not be staged.

- [ ] **Step 3: Build the production manifest**

```bash
RUN=data/candidate_pool/r2r_b1k_300qa_$(date -u +%Y%m%dT%H%M%SZ)_${REVISION:0:7}
$PY scripts/run_background_collection.py build --output-root "$RUN"
```

Inspect the emitted summary: 61 registered R2R scenes, the authority-complete
B1K catalog, four GPUs, disjoint scene coverage, production budgets, and unique
output roots must all match.

- [ ] **Step 4: Launch the detached controller**

```bash
nohup "$PY" scripts/run_background_collection.py run \
  --manifest "$RUN/controller/manifest.json" \
  > "$RUN/controller/controller.log" 2>&1 &
echo $! > "$RUN/controller/controller.pid"
```

- [ ] **Step 5: Verify live launch**

After two polling intervals, verify:

```bash
kill -0 "$(cat "$RUN/controller/controller.pid")"
$PY scripts/run_background_collection.py status \
  --manifest "$RUN/controller/manifest.json"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Expected: controller alive; four canary jobs in `running`; GPU IDs exactly
0--3 with no overlap; logs and state exist; no immediate source/argument error.

- [ ] **Step 6: Report launch identifiers**

Report the clean revision, run root, controller PID, per-GPU job IDs/PIDs/logs,
current state, the exact completion gates, and the fact that C1 remains
candidate-only pending its human-backed authority.
