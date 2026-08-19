# ABC-only Pipeline Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task.  Parallel
> subagents are intentionally not used in this repository session.

**Goal:** Finish the ABC-only slimming through a strict `conseq.v10` record,
fresh R2R records, and a new complete Golden gate.

**Architecture:** Normalize candidate source contexts first, then simplify
trust-boundary readers, then remove v9-only producers and consumers in one
schema break.  Each stage has an independent test and Golden boundary so an
artifact-format change cannot hide a task-semantic change.

**Tech Stack:** Python 3.9, pytest, NumPy, Habitat/R2R, canonical JSON and
SHA-256 manifests.

## Global Constraints

- Active taxonomy is exactly A1/A2/A3/B1/B2/C1; D is deferred.
- Record output is `conseq.v10`; no v9 adapter is allowed.
- Candidate artifacts always report `headline_eligible=false`.
- Keep full/depth consensus and swept-corridor certification.
- Use the Habitat Python at
  `/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python`.
- Preserve unrelated worktree changes.

---

### Task 1: Normalize private source contexts

**Files:**
- Modify: `pipeline/candidate_preview.py`
- Modify: `tests/test_pl_v16_candidate_preview.py`
- Modify: `scripts/check_abc_golden.py`
- Modify: `docs/golden/2026-08-05-r2r-abc-v10-24qa.json` (the 08-04 manifest is superseded and no longer rebuildable)

**Interfaces:**
- Consumes: `compile_main_records(records, ...)` projections and authenticated
  source records.
- Produces: `record_contexts.jsonl`, keyed by the `record_sha256` values used by
  emitted atoms.

- [ ] Add a test compiling one eligible and one fully rejected record and
  assert that only the eligible record digest appears in `record_contexts`.
- [ ] Run that test and verify it fails because contexts are currently
  collected before eligibility.
- [ ] Move context collection to the `built_for_atom` path and make validation
  require `set(contexts) == {atom["record_sha256"] for atom in atoms}`.
- [ ] Add negative tests for duplicate, noncanonical, missing, tampered, and
  merge-conflicting contexts; verify each exact validation error.
- [ ] Add a Golden-inventory test that creates an unregistered rebuilt file and
  expects `_check_digests` to report it.
- [ ] Change `_check_digests` to compare the complete relative file set before
  checking hashes.
- [ ] Rebuild the candidate artifact, add `record_contexts.jsonl` to
  `outputs_sha256`, update intentional v2 digests, and preserve coverage
  `18/7/8/2/4/6` plus `gt-as-pred=1.0`.
- [ ] Run candidate tests and `scripts/check_abc_golden.py`.

### Task 2: Finish formal/threat-model cleanup

**Files:**
- Modify: `pipeline/source_manifest.py`
- Modify: `pipeline/future_view_selection.py`
- Modify: `pipeline/scene_pool.py`
- Modify: `pipeline/semantic.py`
- Modify: `tests/test_pl_v16_b_geometry.py`
- Modify: `tests/test_pl_scene_pool.py`
- Modify: `tests/test_pl_mp3d_semantic.py`
- Modify: `tests/test_pl_v16_b_validation.py`
- Modify: `tests/test_pl_v16_c_assets.py`

**Interfaces:**
- Consumes: authenticated paths and expected SHA-256 digests.
- Produces: immutable byte snapshots for one single-user read, without
  concurrent-writer guarantees.

- [ ] Add/retain focused tests for wrong digest, truncated input, canonical
  escape, and final-component symlink rejection.
- [ ] Remove tests whose only event is mutation between two reads or replacement
  by a concurrent writer.
- [ ] Simplify readers to one open/read/digest transaction; keep `O_NOFOLLOW`
  and regular-file checks, remove before/after inode equality checks.
- [ ] Remove `FORMAL_SOURCES_SCHEMA` and stale formal-builder terminology with
  zero active imports.
- [ ] Run semantic, scene-pool, B-validation, C-asset, and source tests.
- [ ] Run the full suite and Golden gate; outputs must remain byte-identical to
  the newly frozen candidate v2 Golden.

### Task 3: Define the strict v10 synthetic contract

**Files:**
- Modify: `tests/_synthetic.py`
- Modify: `tests/test_pl_record.py`
- Modify: `tests/test_pl_validate_behavior.py`
- Modify: `pipeline/record.py`
- Modify: `pipeline/validate.py`

**Interfaces:**
- Produces: `record.SCHEMA_VERSION == "conseq.v10"` records containing only ABC
  branches.

- [ ] Change synthetic fixtures to the proposed v10 shape and add assertions
  that all retired top-level/outcome/evidence/future-state keys are absent.
- [ ] Run record/validation tests and verify failures still expect v9 fields.
- [ ] Set `SCHEMA_VERSION` to `conseq.v10`; delete retired serialization and
  validation branches rather than making them optional.
- [ ] Update exact-schema rejection tests to require v10 and reject v9 with a
  recollection message.
- [ ] Run record and validator tests until green.

### Task 4: Remove continuation and legacy consequence production

**Files:**
- Modify: `pipeline/consequence.py`
- Modify: `pipeline/rollout.py`
- Modify: `pipeline/collection_runtime.py`
- Modify: `pipeline/intervention_validation.py`
- Delete: `pipeline/q4_sampling.py`
- Modify: `tests/test_pl_rollout.py`
- Modify: `tests/test_pl_ground_disc.py`
- Modify: `tests/test_pl_goal_relations_v5.py`
- Modify: `tests/test_pl_collect_resume.py`

**Interfaces:**
- Consumes: the realized rollout and active ABC certificates.
- Produces: no continuation rollouts, goal relations, review cache, or probes.

- [ ] Add a collector test that spies on rollout evaluation and asserts no
  post-endpoint `terminal_options` call occurs.
- [ ] Run it and verify the current producer calls `terminal_options`.
- [ ] Delete terminal/start option construction and their future-state masks.
- [ ] Delete object/goal/geodesic construction and old probe validation.
- [ ] Move any still-active target-eligibility test out of
  `test_pl_goal_relations_v5.py`, then delete the obsolete test body.
- [ ] Remove now-unreferenced functions/imports and confirm `q4_sampling.py` has
  zero imports before deleting it.
- [ ] Run rollout, collection, ground-disc, objects, and validation tests.

### Task 5: Remove generic projections and review-only assets

**Files:**
- Modify: `pipeline/record.py`
- Modify: `pipeline/collection_runtime.py`
- Modify: `pipeline/collection_q4.py`
- Modify: `pipeline/candidate_preview.py`
- Modify: `tests/test_pl_record.py`
- Modify: `tests/test_pl_collect_resume.py`
- Modify: `tests/test_pl_v16_shared_oracle.py`
- Modify: `tests/test_pl_v16_c_source.py`

**Interfaces:**
- Consumes: source-pinned `b_target`, exact `b_endpoint_relation`, and C
  terminal PNG evidence.
- Produces: no target-reference/projection or review-WebP branches.

- [ ] Add an integration assertion that all six builders compile from a v10
  record lacking generic projections and review evidence.
- [ ] Run it and verify current serializers/validators require old branches.
- [ ] Delete target-reference/projection serialization, review evidence, and
  future-review WebP attachment.
- [ ] Refactor B/C endpoint reads to `execution.realized_pose` and remove the
  obsolete `future_state` wrapper if no active consumer remains.
- [ ] Run A/B/C candidate, shared-oracle, record, and collection tests.

### Task 6: Recollect and register v10 shards

**Files:**
- Modify: `docs/runs.json`
- Create: `data/candidate_pool/<v10-run>/records/r2r/main-r*/...`
- Done: replaced by `docs/golden/2026-08-05-r2r-abc-v10-24qa.json`; the
  08-04 manifest is marked `status: superseded` and the checker refuses
  to run it.

**Interfaces:**
- Consumes: registered R2R scenes/poses and the v10 collector.
- Produces: validated authoritative v10 shards and candidate QA.

- [ ] Reproduce each golden source setting from its authenticated run metadata,
  using fresh output directories and no overwrite of v9 data.
- [ ] Validate every shard with `scripts/check_records.py` and its exact
  `run_meta.json` SHA-256.
- [ ] Compile candidate QA with the registered C1 gate and verify all six task
  counts are nonzero; never duplicate cases to preserve old counts.
- [ ] Register the completed v10 run and freeze every input/output digest,
  complete inventory, coverage, and replay result.
- [ ] Run `scripts/check_abc_golden.py` twice to prove deterministic rebuilds.

### Task 7: Final closure audit

**Files:**
- Modify: `AGENTS.md`
- Modify: `pipeline/README.md`
- Modify: affected architecture tests.

**Interfaces:**
- Produces: documentation and gates matching the live v10 implementation.

- [ ] Search active code/tests for Q-family, D continuation, v9, formal builder,
  terminal options, generic projections, review WebP, and concurrent-mutation
  symbols; classify every remaining hit as active, archived, or remove it.
- [ ] Update schema/version/run instructions and line/module budget assertions.
- [ ] Run `python -m compileall -q pipeline scripts`.
- [ ] Run the full pytest suite.
- [ ] Run `scripts/check_abc_golden.py` and `eval_benchmark.py --gt-as-pred`.
- [ ] Inspect `git diff --check`, `git status --short`, and the final diff for
  unrelated generated artifacts.
