# R2R + B1K 300-QA Background Collection Design

## Objective

Run a resumable, source-bound, multi-GPU collection over the registered R2R
and Behavior-1K catalogs.  For each dataset, the resulting candidate artifact
must contain at least 300 QA items for each of A1, A2, A3, B1, B2, and C1.

This run collects candidate QA.  It does not change any physical, publication,
stability, coverage, or human-calibration threshold.  In particular, C1 remains
`headline_eligible=false` until its independent human-backed gate is calibrated.
GS and HM3D are outside this run.

## Frozen collection meanings

### Pose count

For the main A/B run, every registered scene must persist at least 200 accepted
pose records.  An accepted pose is a source-valid record with a nonempty formal
outcome set; a sampled or rendered pose that fails the oracle does not count.
The collector may inspect up to 4,000 deterministic pose candidates per scene
to fill this target.  Per-scene funnel counters are the authority for this gate.

C1 uses the same registered scene population and processes at least 200
authenticated pose opportunities per scene.  C1 family survival is inherently
lower, so it is governed by the dataset-wide requirement of at least 300
compiled C1 candidate questions rather than an impossible requirement of 200
surviving C1 families in every scene.

### Action count

Every search-ready pose must expose at least 200 unique action programs before
the physical oracle selects outcomes.  The run uses:

- a natural proposal cap of 40 programs per action length;
- lengths 1 through 6;
- 40 natural programs requested per length; and
- up to 12 depth-conditioned safe/collision pairs per length.

The frozen action catalog contains six usable length-1 programs and at least
40 natural programs at each of lengths 2--6.  Thus the natural bank alone has
at least `6 + 5 * 40 = 206` unique programs.  The persisted
`materialized_action_bank` is recomputed and counted; a pose with fewer than
200 unique entries fails closed and is not counted toward the 200-pose target.

### Retained outcomes

After unchanged full/depth consensus, publication margin, coverage, and
stability gates, the deterministic label-blind selector may retain between 2
and 30 outcomes per main pose.  Thirty is a maximum, not a quota: a pose is
never padded with failed or duplicate actions.  C1 retains its atomic
four-member family and does not use this 30-outcome cap.

## Alternatives considered

1. **Immediate monolithic full run.**  Simple, but a bad policy or throughput
   estimate can waste days of B1K compute before the first audit point.
2. **Auto-promoting production canary, then full run.**  Selected.  The first
   scene assigned to each worker uses the exact production policy.  Successful
   validation promotes the worker automatically; no human acknowledgement is
   required between canary and full collection.
3. **Several old 72-action runs with different seeds.**  Rejected because it
   repeats simulation and rendering, complicates cross-run deduplication, and
   does not authenticate one 200-action bank at a pose.

## Execution architecture

The run is pinned to one clean Git revision and one canonical JSON manifest.
A local `nohup` controller owns a durable state file and per-job logs.
Completion is based on authenticated output files and validation results, not
on process disappearance.

Four RTX 4090 GPUs are used in scene-disjoint shards:

1. production canary: two R2R scenes and two B1K scenes, one job per GPU;
2. R2R A/B: four persistent Habitat shards, one per GPU;
3. B1K A/B: four supervisor shards, one per GPU, retaining the existing
   per-scene OmniGibson process isolation;
4. R2R C1: four scene-disjoint shards;
5. B1K C1: four per-scene-isolated supervisor shards; and
6. quota audit/top-up: compile each dataset independently and schedule only a
   deficient task/scene cell if any count remains below 300.

R2R workers use eight semantic-query workers each and force BLAS/OpenMP thread
counts to one.  B1K workers use one semantic-query worker; GPU selection is
bound by both `CUDA_VISIBLE_DEVICES` and `OMNIGIBSON_GPU_ID` where applicable.
No two active jobs share a GPU.

## Canary promotion gates

A canary promotes automatically only when all of the following hold:

- the process is pinned to the manifest's clean Git revision;
- source-bound record validation reports zero violations;
- the scene persists at least 200 accepted A/B poses;
- every counted pose authenticates at least 200 unique offered actions;
- formal action-length coverage is complete;
- all required run metadata, funnel, records, and source identities exist;
- observed bytes per accepted pose project to fit within the 610 GiB free
  filesystem while preserving at least 120 GiB free; and
- no job has made zero durable progress for 90 minutes.

A failed canary marks its shard `stuck`; it never silently falls through to the
remaining scene list.  OOM receives at most two delayed retries.  Other errors
require diagnosis rather than blind relaunch.

## Data flow and completion

Each dataset is compiled separately from its authenticated shards.  The
controller records raw eligible counts before any final balancing.  Completion
requires all six task counts to be at least 300 and reports:

- counts by dataset, task, scene, action length, collision label, and C1 gate
  authority;
- per-scene accepted/search-ready pose counts and action-bank quantiles;
- validation and withhold-reason distributions;
- source/run-contract/records/candidate-artifact SHA-256 identities; and
- GPU-worker logs and elapsed throughput.

The controller does not relax a gate to satisfy a quota.  If a task remains
below 300, its state is `quota_shortfall`, and a top-up run uses a new seed plus
published pose exclusions while retaining the same scientific policy.

## Recovery and storage

Every shard uses transactional record groups and `--resume`.  The controller
atomically writes its state after each transition.  Existing completed scenes
are hash-validated before reuse.  Partial B1K scenes retain the existing
supervisor validation and isolation behavior.

Arrays and debug images are disabled.  Initial RGB and required C1 terminal
assets are retained.  Output roots, manifests, logs, and state are created
under one timestamped `data/candidate_pool/` run; no existing run is
overwritten.

## Required code changes and verification

Before launch:

1. raise the centralized natural proposal cap from 12 to 40;
2. raise the deterministic retained-outcome maximum from 12 to 30;
3. add a source-bound minimum materialized-action-bank gate of 200;
4. add the resumable local multi-GPU controller and quota auditor;
5. add deterministic RED-to-GREEN tests for all new policy and state-machine
   behavior; and
6. run focused tests, full `pytest -q`, `scripts/check_abc_golden.py`, and
   `git diff --check` before pinning the launch revision.

Golden must remain byte-identical because this changes future collection
policy, not the frozen input records or active QA compiler.  The controller is
launched only from the verified clean commit.
