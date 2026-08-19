> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# EgoConseq-Bench v1.3.1a Review Closure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the verified documentation, release-audit, artifact-loader, and
historical-funnel gaps left after the v1.3.1a implementation without deleting
active compatibility or diagnostic paths.

**Architecture:** Keep selection-bias feature extraction and comparison in
`pipeline/shortcut_audit.py`, with the CLI responsible only for loading paired
raw/published artifacts. Keep public artifact semantic checks centralized in
the question registry and mirror the checks required by the public-only viewer.
Extend the read-only historical funnel without treating historical proxy fields
as publication GT.

**Tech Stack:** Python 3.9, NumPy, scikit-learn, pytest, Markdown.

---

### Task 1: Replace the stale public protocol

**Files:**
- Modify: `BENCHMARK.md`
- Modify: `README.md`

- [ ] Rewrite the public taxonomy around four scored result heads, a separate
  Integration track, Direct Body Effect, four intervention protocols, and the
  non-scoring `rollout_stage` attribute.
- [ ] Remove public abstention/`answerable`, fixed 3+3 sampling, radius-dependent
  Q6 surface-gap, and old L-macro claims.
- [ ] Preserve L0--L4 only as a compatibility stage alias required by the
  current QA schema; state explicitly that it is not the leaderboard axis.
- [ ] Verify stale claims are absent:

```bash
rg -n "Every question contains|exactly three safe|answerable.: false|macro-averaged by level" \
  BENCHMARK.md README.md
```

### Task 2: Implement paired selection-bias audits

**Files:**
- Modify: `pipeline/shortcut_audit.py`
- Modify: `scripts/audit_shortcuts.py`
- Modify: `tests/test_pl_shortcut_audit.py`

- [ ] Add failing tests for a nuisance feature registry with blocking public
  metadata/artifact features and diagnostic-only public image-statistic probes.
- [ ] Add a failing test for a paired report that compares the same nuisance
  baseline on raw and published rows and returns
  `published_gain - raw_gain`.
- [ ] Add a failing CLI test requiring `--raw-benchmark` and
  `--published-benchmark` for the selection-bias mode.
- [ ] Run the targeted tests and confirm the missing API failures.
- [ ] Implement the minimum registry and paired comparison. Private depth,
  coverage, and failure counters must be rejected as fair features.
- [ ] Make only blocking nuisance deltas participate in the formal release
  verdict; encoder/RGB-depth/local-geometry probes remain diagnostic.
- [ ] Run:

```bash
$PY -m pytest tests/test_pl_shortcut_audit.py -v
```

### Task 3: Validate the new taxonomy at the HTTP trust boundary

**Files:**
- Modify: `scripts/serve_benchmark.py`
- Modify: `tests/test_pl_serve_benchmark.py`

- [ ] Add failing tests that mutate `result_head`, `rollout_stage`, and
  `score_role` independently while retaining valid version stamps.
- [ ] Verify each malformed public-only artifact currently loads.
- [ ] Extend `_assert_supported_schema` using `question_spec`,
  `result_head_for`, and `score_role_for`.
- [ ] Retain the current `level` check as a v10 compatibility-stage check.
- [ ] Run:

```bash
$PY -m pytest tests/test_pl_serve_benchmark.py -v
```

### Task 4: Extend the historical funnel

**Files:**
- Modify: `pipeline/funnel_audit.py`
- Modify: `tests/test_pl_funnel_audit.py`

- [ ] Add failing fixtures for Q8 `out_of_fov` versus `occluded` pair yield,
  Q9 endpoint-mask change, and matched-action mask-pair yield.
- [ ] Count by backend, unique scene, and unique physical group; sensor siblings
  must not increase the effective sample size.
- [ ] Label missing historical fields as unavailable rather than false.
- [ ] Preserve `historical_feasibility_only: true`.
- [ ] Run:

```bash
$PY -m pytest tests/test_pl_funnel_audit.py -v
```

### Task 5: Remove only verified dead compatibility logic

**Files:**
- Modify: `scripts/gen_benchmark.py`
- Modify: `pipeline/benchmark.py`
- Modify: `tests/test_pl_track.py`
- Modify: `tests/test_pl_question_levels.py`

- [ ] Delete `TRACK_WEIGHTS`, `SINGLE_ANSWER_TRACK`, `frame_track`,
  `family_in_track`, and their branches. Keep `--no-track` as a deprecated
  no-op so existing commands fail neither silently nor abruptly.
- [ ] Delete only `_RESPONSE_PROPERTIES["answerable"]`; retain validator and
  evaluator rejection of obsolete public/private `answerable` fields.
- [ ] Do not delete `run_qa_baselines.py`, record-level sensor-consistency
  validation, `BENCH_Q4_FLIP_RATIO`, or `FAMILY_LEVEL`.
- [ ] Run the affected tests.

### Task 6: Verification

- [ ] Run all focused tests:

```bash
$PY -m pytest -q \
  tests/test_pl_shortcut_audit.py \
  tests/test_pl_serve_benchmark.py \
  tests/test_pl_funnel_audit.py \
  tests/test_pl_question_levels.py
```

- [ ] Run the full suite:

```bash
CUDA_VISIBLE_DEVICES='' $PY -m pytest -q
```

- [ ] Revalidate the current preview artifact and confirm that the viewer still
  serves its complete questions and private GT in trusted local review mode.
