# Counterfactual-only C1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Remove every executable strict/suffix C1 path, make the new-observation-permitted counterfactual selector the only C1 implementation for R2R/B1K/GS, preserve measurable C1 diagnostics, and prove real six-task QA output on GS and B1K.

**Architecture:** Collection reserves counterfactual neighbours inside the natural action bank, certifies them through the ordinary consequence oracle, and attaches source-bound terminal RGB. Compilation always selects one query and three independently certified neighbours. Strict suffix collection, candidate review, matched selection, and future-view gate authority disappear; shared terminal-RGB primitives remain.

**Tech Stack:** Python 3.9, NumPy, Pillow, pytest, Habitat/GSplat/OmniGibson, JSONL source-bound artifacts.

## Global Constraints

- Work only in /home/zhangshan/syp/myvln/pbench-abc1-impl.
- Do not touch the primary worktree's browser changes.
- Do not change action grids, scientific thresholds, or A/B gates.
- C1 permits new observations and always has headline_eligible=false.
- No executable suffix compatibility adapter remains.
- Preserve immutable superseded Golden transition evidence.
- Use red-green TDD for behavior changes.
- Require a durable accepted record within 120 seconds after backend-ready.
- Name generated smoke roots with SMOKE_REV=$(git rev-parse --short HEAD).

---

### Task 1: Freeze the one-C1 CLI contract

**Files:**
- Modify: tests/test_pl_collect_resume.py
- Modify: tests/test_pl_v16_candidate_preview.py

**Interfaces:**
- Consumes: collection_cli.build_parser and
  build_v16_candidate_preview.build_arg_parser.
- Produces: failing tests proving users cannot start either retired C1 route.

- [ ] **Step 1: Write failing CLI rejection tests**

    def test_collection_cli_rejects_retired_c1_suffix_mode():
        with pytest.raises(SystemExit):
            collection_cli.build_parser().parse_args([
                "--scenes", "scene", "--out", "/tmp/out",
                "--c1-suffix-candidates"])

    @pytest.mark.parametrize("option", [
        "--future-view-gate",
        "--future-view-gate-sha256",
        "--future-view-gate-authority-id",
    ])
    def test_candidate_build_cli_rejects_retired_c1_gate(option):
        with pytest.raises(SystemExit):
            build_v16_candidate_preview.build_arg_parser().parse_args([
                option, "retired", "--records", "records.jsonl",
                "--run-meta-sha256", "0" * 64,
                "--output", "/tmp/out", "--report", "/tmp/report"])

- [ ] **Step 2: Verify RED**

Run:

    $PY -m pytest -q \
      tests/test_pl_collect_resume.py::test_collection_cli_rejects_retired_c1_suffix_mode \
      tests/test_pl_v16_candidate_preview.py -k retired_c1_gate

Expected: failures because both parsers still accept their retired options.

- [ ] **Step 3: Commit**

    git add tests/test_pl_collect_resume.py tests/test_pl_v16_candidate_preview.py
    git commit -m "test: require one counterfactual C1 route"

---

### Task 2: Remove strict collection and candidate review

**Files:**
- Delete: pipeline/c1_safe_suffix.py
- Delete: pipeline/candidate_review.py
- Delete: tests/test_pl_c1_safe_suffix.py
- Delete: tests/test_pl_b1k_candidate_review.py
- Modify: pipeline/action_proposal.py
- Modify: pipeline/collection_cli.py
- Modify: pipeline/collection_runtime.py
- Modify: pipeline/collection_support.py
- Modify: pipeline/formal_output_coverage.py
- Modify: pipeline/source_manifest.py
- Modify: pipeline/validate.py
- Modify: scripts/run_b1k_collection_shard.py
- Modify: tests/test_pl_collect_resume.py
- Modify: tests/test_pl_b1k_runtime.py
- Modify: tests/test_pl_b1k_pilot_tools.py
- Modify: tests/test_pl_family_plan.py

**Interfaces:**
- Consumes: natural c1_counterfactual.reserve_pose_slots.
- Produces: collection with no suffix flag, review tree, suffix policy, or review shard scope.

- [ ] **Step 1: Freeze the existing B1K orchestration regression**

Use the already-specific
test_collection_shard_partial_resume_fails_closed_on_tamper in
tests/test_pl_b1k_pilot_tools.py.  It creates a formally eligible partial
shard, mutates records.jsonl, resumes, and requires the supervisor to raise
"partial-valid artifact changed".  Do not add a duplicate test.

- [ ] **Step 2: Run that regression before deletion**

Run:

    $PY -m pytest \
      tests/test_pl_b1k_pilot_tools.py::test_collection_shard_partial_resume_fails_closed_on_tamper \
      -q

Expected: PASS; this is continuous regression protection.

- [ ] **Step 3: Delete the strict collection closure**

Reduce the action policy API to:

    def expected_policy_for_action_mode(
            action_mode, *, family_plan: bool = False) -> Optional[str]:
        base = _POLICY_BY_ACTION_MODE.get(str(action_mode or ""))
        if not family_plan:
            return base
        if base != DEPTH_CONDITIONED_POLICY:
            return None
        return EXACT_ACTION_FAMILY_AUGMENTED_POLICY

Remove c1_suffix_candidates branches and candidate-review counts/scopes. Do not replace removed behavior with no-op fields.

- [ ] **Step 4: Update shared callers and delete strict-only tests**

Every expected_policy_for_action_mode caller omits c1_suffix_candidates. Preserve B1K formal-shortfall and partial-shard checks.

- [ ] **Step 5: Verify**

Run:

    $PY -m pytest -q tests/test_pl_collect_resume.py \
      tests/test_pl_b1k_runtime.py tests/test_pl_b1k_pilot_tools.py \
      tests/test_pl_family_plan.py

Expected: collection-side one-C1 assertions and B1K safety regression pass.

- [ ] **Step 6: Commit**

    git add pipeline scripts tests
    git commit -m "refactor: remove strict C1 collection"

---

### Task 3: Remove matched selection and future-view gate authority

**Files:**
- Modify: pipeline/future_view_selection.py
- Modify: pipeline/benchmark_tasks.py
- Modify: pipeline/benchmark_builders.py
- Modify: pipeline/candidate_preview.py
- Modify: pipeline/gate_authority.py
- Modify: pipeline/config.py
- Modify: pipeline/qa_reasons.py
- Modify: scripts/build_v16_candidate_preview.py
- Delete: tests/test_pl_v16_c_metric.py
- Modify: tests/test_pl_v16_c_candidates.py
- Modify: tests/test_pl_task3_authority.py
- Modify: tests/test_pl_v16_candidate_preview.py
- Modify: tests/test_pl_a2_shortcut_audit.py
- Modify: tests/test_pl_b1k_contracts.py
- Modify: tests/test_pl_c1_counterfactual.py

**Interfaces:**
- Consumes: c1_counterfactual.candidate_selection_or_reason(record, outcome, *, asset_root).
- Produces: benchmark_tasks.c_candidate_selection(record, outcome, *, asset_root).

- [ ] **Step 1: Write a failing direct counterfactual behavior test**

Reuse the real synthetic counterfactual record fixture in
tests/test_pl_c1_counterfactual.py, including authenticated terminal PNGs with
new content.  Call benchmark_tasks.c_candidate_selection with only record,
primary outcome, and asset_root.  Assert the returned certificate has schema
c1-counterfactual-selection.v1 and headline_eligible is false.  The current
dual-route signature fails before evaluating the fixture.

- [ ] **Step 2: Verify RED**

Run the new test. Expected: signature error because image and expected_raw_image_sha256 are still required.

- [ ] **Step 3: Delete strict selection, preserving shared primitives**

Keep terminal RGB materialization/validation, renderer/source binding, _selection_descriptor, block-L1 functions, and counterfactual_c_outcome_eligibility/counterfactual_c_eligible_outcomes. Delete gate parsers, strict eligibility, matched metric selection, and matched certificate validation.

Canonical entry point:

    def c_candidate_selection(
            record: Dict, outcome: Dict, *, asset_root
            ) -> tuple[dict | None, str]:
        return c1_counterfactual.candidate_selection_or_reason(
            record, outcome, asset_root=asset_root)

- [ ] **Step 4: Remove gate authority end-to-end**

Remove gate fields from source authority, build CLI, source map, artifact writer/validator, builder signatures, config, and rejection registry. Record and run-meta SHA bindings remain mandatory.

- [ ] **Step 5: Verify focused compiler tests**

Run:

    $PY -m pytest -q tests/test_pl_collect_resume.py \
      tests/test_pl_c1_counterfactual.py tests/test_pl_v16_c_candidates.py \
      tests/test_pl_v16_candidate_preview.py tests/test_pl_task3_authority.py \
      tests/test_pl_a2_shortcut_audit.py tests/test_pl_b1k_contracts.py

- [ ] **Step 6: Commit**

    git add pipeline scripts tests
    git commit -m "refactor: make counterfactual C1 canonical"

---

### Task 4: Add six pairwise block-L1 diagnostics

**Files:**
- Modify: pipeline/c1_counterfactual.py
- Modify: tests/test_pl_c1_counterfactual.py

**Interfaces:**
- Consumes: future_view_selection.block_l1_certificate(left, right).
- Produces: selection["terminal_pair_block_l1"], six canonical rows included in certificate hashing and validator reconstruction.

- [ ] **Step 1: Write a failing certificate test**

Build a four-choice synthetic selection and assert terminal_pair_block_l1 has six sorted choice-id pairs, each with protocol block-l1.v1.

- [ ] **Step 2: Verify RED**

Expected: KeyError for terminal_pair_block_l1.

- [ ] **Step 3: Implement diagnostic-only pair certificates**

Retain authenticated decoded snapshots, enumerate itertools.combinations over the deterministic four-choice order, compute six block_l1_certificate values, sort by choice ids, and include them before the selection SHA. Never read this field in eligibility or choice ranking.

- [ ] **Step 4: Verify and commit**

    $PY -m pytest tests/test_pl_c1_counterfactual.py -q
    git add pipeline/c1_counterfactual.py tests/test_pl_c1_counterfactual.py
    git commit -m "feat: record C1 appearance diagnostics"

---

### Task 5: Port the three shortcut baselines

**Files:**
- Create: pipeline/c1_counterfactual_shortcut_audit.py
- Create: scripts/audit_c1_counterfactual_shortcuts.py
- Create: tests/test_pl_c1_counterfactual_shortcut_audit.py
- Delete: pipeline/c1_suffix_shortcut_audit.py
- Delete: scripts/audit_c1_suffix_shortcuts.py
- Delete: tests/test_pl_c1_suffix_shortcut_audit.py

**Interfaces:**
- Consumes: a validated candidate artifact and counterfactual certificates.
- Produces: egoconseq.c1-counterfactual-shortcut-audit.v1 with initial_to_option_similarity, options_only_visual_medoid, and motion_only baselines plus scene-clustered intervals.

- [ ] **Step 1: Write failing exact-baseline tests**

Use these three rows so every expected accuracy is mechanical:

    rows = [
        {"scene_id": "s1", "answer_position": 1,
         "initial_similarity_prediction": 1,
         "visual_medoid_prediction": 2, "motion_prediction": 1},
        {"scene_id": "s1", "answer_position": 3,
         "initial_similarity_prediction": 1,
         "visual_medoid_prediction": 3, "motion_prediction": 3},
        {"scene_id": "s2", "answer_position": 2,
         "initial_similarity_prediction": 2,
         "visual_medoid_prediction": 2, "motion_prediction": 4},
    ]

Assert initial_to_option_similarity=2/3,
options_only_visual_medoid=2/3, motion_only=2/3, and two calls with
resamples=40/seed=17 are identical.

- [ ] **Step 2: Verify RED**

Expected: ModuleNotFoundError for the new audit module.

- [ ] **Step 3: Implement an artifact-native audit**

Follow a2_shortcut_audit: validate the artifact, join answers/atoms/record contexts, authenticate and load images, recompute distances, derive motion-only features from public actions, and bootstrap by scene. Do not import candidate_review or c1_safe_suffix.

- [ ] **Step 4: Add the CLI and output guard**

Arguments: --benchmark, --out, --source-authority-manifest, --resamples, --seed. Refuse output inside the benchmark tree.

- [ ] **Step 5: Verify and commit**

    $PY -m pytest tests/test_pl_c1_counterfactual_shortcut_audit.py -q
    git add pipeline scripts tests
    git commit -m "feat: audit counterfactual C1 shortcuts"

---

### Task 6: Replace active Golden and retire strict data

**Files:**
- Create: docs/golden/2026-08-11-r2r-abc-counterfactual-v11.json
- Modify: scripts/check_abc_golden.py
- Modify: docs/runs.json
- Modify: README.md
- Modify: pipeline/README.md
- Preserve: superseded manifests and referenced immutable bytes.
- Delete generated data: data/candidate_pool/step3_smoke_baf56c0

**Interfaces:**
- Consumes: one source-bound R2R natural record with a query and at least three certified neighbours.
- Produces: active six-task Golden without a future-view gate.

- [ ] **Step 1: Re-scan strict data**

Run:

    rg -l '"c1_suffix_candidates"[[:space:]]*:[[:space:]]*true' \
      data/candidate_pool --glob run_meta.json | sort

Expected: only paths under data/candidate_pool/step3_smoke_baf56c0. Stop if another path appears.

- [ ] **Step 2: Resolve a counterfactual Golden record**

First rebuild a scratch artifact from the source-bound natural records at:

    data/candidate_pool/step10_canary_f18b801/records/r2r/canary-r00/records.jsonl

using its adjacent run_meta.json digest and a newly written gate-free authority
manifest.  Read public/items.jsonl and require at least one
C1_future_view_selection item whose private selection certificate schema is
c1-counterfactual-selection.v1.  If this exact corpus has no qualifying item,
run the ordinary R2R collector with seed 20260811, scene
17DRP5sb8fy, lengths 1--6, natural balanced actions, and stop at the first
qualifying counterfactual record.  Never reinterpret an old strict record.

- [ ] **Step 3: Write the new active manifest**

Remove required gate inputs and gate CLI argv, freeze six-task coverage and output digests, point supersedes to the previous active manifest, update the checker default and run registry.

- [ ] **Step 4: Delete only the resolved strict smoke root**

Validate the exact path and remove data/candidate_pool/step3_smoke_baf56c0. Report that generated data was deleted and is regenerable.

- [ ] **Step 5: Verify and commit tracked changes**

    $PY scripts/check_abc_golden.py
    git add docs README.md pipeline/README.md scripts/check_abc_golden.py
    git commit -m "test: freeze counterfactual C1 golden"

---

### Task 7: Static closure verification

**Files:**
- Modify only active files found by the scan.

**Interfaces:**
- Consumes: Tasks 2-6.
- Produces: no executable strict symbol or active instruction.

- [ ] **Step 1: Scan active code**

    rg -n "c1_suffix_candidates|c1-suffix-candidates|c1_safe_suffix|candidate_review|strict_c_outcome_eligibility|strict_c_eligible_outcomes|select_future_view_choices|future_view_gate" \
      pipeline scripts tests README.md pipeline/README.md

Expected: zero matches. Immutable superseded manifests are outside this scan.

- [ ] **Step 2: Run complete verification**

    $PY -m compileall -q pipeline scripts
    $PY -m pytest -q
    $PY scripts/check_abc_golden.py

Expected: zero failures, zero skips, Golden exit 0.

- [ ] **Step 3: Commit any residual tracked cleanup**

    git add pipeline scripts tests README.md
    git commit -m "refactor: remove residual strict C1 references"

---

### Task 8: Real GS six-task smoke

**Files:**
- Generated only: data/candidate_pool/gs_counterfactual_smoke_${SMOKE_REV}/

**Interfaces:**
- Consumes: exact verified Task-7 commit.
- Produces: validated GS records, six-task QA, gt-as-pred result, timing report, and C1 audit.

- [ ] **Step 1: Start one known-valid GS scene and timestamp renderer-ready**

Use scene interior_0007_840137, seed 20260811, lengths 1--6, camera heights
0.5/1.0/1.5, radii 0.15/0.20/0.25, balanced natural actions,
keep-per-length=1, min-objects=0, and the configured GS source manifest.  Write
with GS data root /home/zhangshan/syp/datasets/gs and source manifest
/home/zhangshan/syp/datasets/gs/splits/train.json to
data/candidate_pool/gs_counterfactual_smoke_${SMOKE_REV}/records/gs/main-r00.
Record process start, ready time, and every durable records.jsonl append.

- [ ] **Step 2: Enforce production timing**

First record must arrive within 120 seconds after ready. Continue until all six tasks compile or ten post-ready minutes elapse.

- [ ] **Step 3: Validate, build, and replay**

Run check_records with the exact adjacent run-meta SHA, build candidate QA without gate arguments, and run eval_benchmark.py --gt-as-pred.

Expected: A1/A2/A3/B1/B2/C1 are all nonzero and replay is 1.0.

- [ ] **Step 4: Run shortcut audit**

Report all three C1 baselines and explicitly compare GS options_only_visual_medoid with 25% chance.

---

### Task 9: Real B1K six-task smoke and timing diagnosis

**Files:**
- Generated only: data/candidate_pool/b1k_counterfactual_smoke_${SMOKE_REV}/

**Interfaces:**
- Consumes: same verified commit as GS.
- Produces: validated B1K records, six-task QA, gt-as-pred result, timing/profile report, and C1 audit.

- [ ] **Step 1: Start the ordinary B1K supervisor**

Use scene Wainscott_0_int, seed 20260811, lengths 1--6, camera heights
0.5/1.0/1.5, radii 0.15/0.20/0.25, balanced natural actions,
keep-per-length=1, min-objects=1, and the frozen full-catalog B1K source
manifest
/home/zhangshan/syp/datasets/behavior-1k-v3.9.1/pbench-abc1-task4/full-catalog-authority-complete-26-source-manifest.json,
with B1K data root /home/zhangshan/syp/datasets/behavior-1k-v3.9.1.  Write to
data/candidate_pool/b1k_counterfactual_smoke_${SMOKE_REV}/records/b1k/main-r00.
Require a structured initialization heartbeat within 30 seconds. Timestamp
backend-ready and every durable accepted-record append.

- [ ] **Step 2: Enforce the two-minute bound**

Fail performance acceptance when time-to-first-record or any inter-record gap exceeds 120 seconds after ready. Do not relax gates. If it fails, stop only after capturing a profiler sample and exact funnel counters.

- [ ] **Step 3: Bound total smoke**

Stop at six-task coverage or twenty post-ready minutes. Compile all accepted records at the cap; a heartbeat or partial task coverage is not success.

- [ ] **Step 4: Validate, build, replay, and audit**

Run source-bound record validation, gate-free candidate build, gt-as-pred, and the three-baseline C1 audit.

Expected: nonzero A1/A2/A3/B1/B2/C1 and replay 1.0.

- [ ] **Step 5: Report measured performance**

Report initialization time, accepted-record timestamps, records/minute, task counts, shortcut baselines, and the exact funnel/profile cause for any two-minute or coverage failure.
