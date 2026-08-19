> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# Specific Semantic Categories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent placeholder semantic categories from entering target-dependent or Q3 QA, then rebuild the active combined artifact.

**Architecture:** Add one exact normalized predicate in `pipeline/config.py`. Reuse it at acquisition eligibility, benchmark eligibility, Q3 choice construction, and artifact validation so current records can be recompiled while future records are clean at source.

**Tech Stack:** Python 3.9, NumPy-oriented pipeline modules, pytest, JSONL artifact compiler.

---

### Task 1: Add Failing Semantic-Eligibility Tests

**Files:**
- Modify: `tests/test_pl_goal_relations_v5.py`
- Modify: `tests/test_pl_questions.py`
- Modify: `tests/test_pl_validate.py`

- [ ] **Step 1: Test the normalized category predicate and target selection**

Add parameterized cases proving placeholder labels are rejected and `chair`
remains eligible. Construct real synthetic frames and call
`objects.eligible_target_ids`.

- [ ] **Step 2: Test compiler-facing benchmark eligibility**

Create a synthetic record whose selected target category is `misc`. Assert
target-dependent `family_eligibility` rejects it. Create a collision outcome
whose contact category is `misc` and assert Q3 rejects it.

- [ ] **Step 3: Test artifact validation**

Build valid synthetic item/answer pairs, mutate `model_input.target_category` or
a Q3 semantic choice to `misc`, and assert `validate_artifact` reports a
non-specific semantic category.

- [ ] **Step 4: Run the focused tests and observe the intended failures**

Run:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest \
  tests/test_pl_goal_relations_v5.py \
  tests/test_pl_questions.py \
  tests/test_pl_validate.py -q
```

Expected: the new assertions fail because no shared non-specific-category
contract exists yet.

### Task 2: Implement the Single Category Contract

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/objects.py`
- Modify: `pipeline/benchmark.py`
- Modify: `pipeline/validate.py`

- [ ] **Step 1: Add the exact normalized predicate**

Define `NON_SPECIFIC_SEMANTIC_CATEGORIES` and
`is_specific_semantic_category(category)` in `pipeline/config.py`. Normalize
with `str(category or "").strip().casefold()` and perform exact membership.

- [ ] **Step 2: Apply it to acquisition and benchmark eligibility**

Reject non-specific categories in `eligible_target_ids`. In
`family_eligibility`, reject a selected target whose stored category fails the
predicate. Apply the same gate to Q3's physical contact category.

- [ ] **Step 3: Apply it to Q3 choices and grouped target builders**

Exclude non-specific Q3 distractors. Make the Q10 grouped builder reject a
non-specific selected target before constructing its public input.

- [ ] **Step 4: Apply it to artifact validation**

Reject non-specific `model_input.target_category` values and Q3 choice ids/text
that encode a denied semantic label.

- [ ] **Step 5: Run focused tests**

Run the Task 1 command. Expected: all focused tests pass.

### Task 3: Verify The Codebase

**Files:**
- No production changes

- [ ] **Step 1: Run the full suite**

```bash
$PY -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Inspect the diff and whitespace**

```bash
git diff --check
git diff --stat
```

Expected: no whitespace errors and only the planned files changed.

### Task 4: Recompile And Audit The Active Artifact

**Files:**
- Regenerate: `data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined`

- [ ] **Step 1: Preserve the prior artifact**

Rename the existing output to a descriptive `_pre_specific_semantic_filter`
directory. Do not modify source records.

- [ ] **Step 2: Compile from every non-empty source shard**

Reconstruct the `--datasets` arguments from
`data/candidate_pool/preview_v6_20260725_d9e37486a696/source_runs`, use seed
`20260725`, `--candidate-pool`, the existing private ID key, and the current
output path.

- [ ] **Step 3: Audit structured semantic fields**

Parse public items and assert:

```python
assert config.is_specific_semantic_category(
    item["model_input"]["target_category"])
```

whenever a target is present. For Q3, assert every semantic contact choice also
passes the predicate, excluding protocol choices such as
`insufficient_evidence`.

- [ ] **Step 4: Run end-to-end checks**

```bash
$PY scripts/check_benchmark.py \
  data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined \
  --allow-incomplete
$PY scripts/eval_benchmark.py \
  --benchmark data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined \
  --gt-as-pred --allow-incomplete
```

Expected: artifact validation passes and applicable GT-as-pred metrics are
`1.0`.

- [ ] **Step 5: Restart the local review server**

Point the existing trusted local review command at the rebuilt artifact and
verify its health endpoint and an item request.
