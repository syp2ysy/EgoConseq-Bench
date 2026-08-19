> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# Q6 Strict Distance-Trend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every nonzero Q6 physical distance change label strictly as closer or farther, with unchanged reserved for exact equality.

**Architecture:** Keep continuous rollout GT unchanged and centralize the QA label decision in `pipeline.benchmark`. Remove Q6's relative semantic margins from eligibility, while preserving evidence qualification and Q10's separately stated 30% constraint. Bump only the prompt contract and regenerate the QA artifact from existing `conseq.v6` records.

**Tech Stack:** Python 3.9, deterministic dictionary-based QA compiler, pytest, JSONL artifact compiler.

---

### Task 1: Pin The Strict Q6 Contract With Failing Tests

**Files:**
- Modify: `tests/test_pl_questions.py`

- [ ] **Step 1: Replace the relative-margin regression with strict-sign cases**

Add tests that build Q6 surface and geodesic questions from synthetic outcomes:

```python
@pytest.mark.parametrize(
    ("distance_change", "expected"),
    [(-0.5, "closer"), (0.5, "farther"), (0.0, "unchanged")],
)
def test_q6_surface_trend_uses_the_strict_distance_sign(
        distance_change, expected):
    outcome = _outcome(distance_change=distance_change)
    eligibility = B.family_eligibility(
        "Q6", outcome, 7, variant="surface_proximity")
    assert eligibility.eligible is True
    _items, answers = B.build_family_pair(
        family="Q6", variant="surface_proximity",
        record=_record(outcome), outcome=outcome,
        target_instance_id=7, image="images/f.png")
    assert answers[0]["structured_answer"]["choice_id"] == expected
```

Add the corresponding geodesic test and change the old `-0.6 m` grey-zone
expectation from rejected to eligible.

- [ ] **Step 2: Assert that Q6 no longer publishes a relative decision margin**

Remove Q6 from `test_private_answers_preserve_family_decision_margins` and add:

```python
def test_q6_private_answer_has_no_semantic_distance_dead_zone():
    outcome = _outcome(distance_change=-0.01)
    _items, answers = B.build_family_pair(
        family="Q6", variant="surface_proximity",
        record=_record(outcome), outcome=outcome,
        target_instance_id=7, image="images/f.png")
    assert answers[0]["structured_answer"]["choice_id"] == "closer"
    assert "distance_change_boundary_rel" not in answers[0]["margins"]
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_questions.py -q
```

Expected: failures show `-0.5` remains ineligible or `unchanged`, and the old
relative margin remains present.

### Task 2: Centralize Strict Trend Classification

**Files:**
- Modify: `pipeline/benchmark.py`
- Modify: `pipeline/config.py`
- Modify: `scripts/collect.py`

- [ ] **Step 1: Add one strict helper**

Add near the other benchmark label helpers:

```python
def q6_distance_trend(distance_change_m: float) -> str:
    """Classify deterministic Q6 distance change without a semantic tolerance."""
    change = float(distance_change_m)
    if change < 0.0:
        return "closer"
    if change > 0.0:
        return "farther"
    return "unchanged"
```

- [ ] **Step 2: Remove the relative Q6 eligibility boundary**

In `family_eligibility`, keep metric availability and anti-shortcut checks, but
remove the `0.10/0.30` acceptance test and its `distance_change_ambiguous`
rejection. For `geodesic_progress`, compare:

```python
if q6_distance_trend(delta) != q6_distance_trend(surface_delta):
    return Eligibility(False, "geodesic_surface_trend_disagreement")
```

Do not emit `distance_change_boundary_rel` or
`surface_distance_change_boundary_rel`.

- [ ] **Step 3: Make answer construction use the same helper**

Replace the relative calculation in `_family_answers` with:

```python
choice = q6_distance_trend(delta)
```

Use the same helper for Q10's geodesic/surface trend-agreement check, but retain
Q10's explicit 30% improvement constraint.

- [ ] **Step 4: Separate the Q10 threshold from the removed Q6 constants**

Replace:

```python
BENCH_Q6_UNCHANGED_REL = 0.10
BENCH_Q6_CHANGED_REL = 0.30
```

with:

```python
BENCH_Q10_REQUIRED_IMPROVEMENT_REL = 0.30
```

Use the Q10 name in `_joint_consequence` and `_joint_constraint_slack`. In the
selective Q6 probe search, rank any strictly positive distance increase as a
farther example rather than requiring 30%.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
$PY -m pytest tests/test_pl_questions.py -q
```

Expected: all question tests pass.

### Task 3: Bump And Test The Prompt Contract

**Files:**
- Modify: `pipeline/benchmark.py`
- Modify: `tests/test_pl_questions.py`
- Modify: `tests/test_pl_validate.py`
- Modify: `README.md`
- Modify: `BENCHMARK.md`

- [ ] **Step 1: Write the stale-contract regression**

Extend the artifact contract test so a manifest/item/answer stamped
`visible-ground-disc.v6` is rejected with `unsupported prompt contract`.

- [ ] **Step 2: Verify RED**

Run:

```bash
$PY -m pytest tests/test_pl_validate.py -q
```

Expected: the v6 stamp is still accepted because it is the current constant.

- [ ] **Step 3: Bump the contract and update literal assertions**

Set:

```python
PROMPT_CONTRACT_VERSION = "visible-ground-disc.v7"
```

Update Q6/version assertions and the documented current artifact contract in
`README.md` and `BENCHMARK.md`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
$PY -m pytest tests/test_pl_validate.py tests/test_pl_questions.py -q
```

Expected: both files pass and v6 is rejected.

### Task 4: Verify The Repository And Commit The Behavior

**Files:**
- Modify: files from Tasks 1--3

- [ ] **Step 1: Scan for obsolete Q6 semantics**

Run:

```bash
rg -n "BENCH_Q6_(UNCHANGED|CHANGED)|distance_change_ambiguous|distance_change_boundary_rel|surface_distance_change_boundary_rel" pipeline scripts tests
```

Expected: no obsolete production or test references.

- [ ] **Step 2: Run the full suite**

Run:

```bash
$PY -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add pipeline scripts tests README.md BENCHMARK.md
git commit -m "fix: classify Q6 distance trends strictly"
```

### Task 5: Regenerate And Audit The Active QA Artifact

**Files:**
- Regenerate: `data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined`

- [ ] **Step 1: Preserve the previous generated artifact outside Git**

Rename the existing directory with a `_pre_q6_strict` suffix so old and new
prompt contracts cannot be confused.

- [ ] **Step 2: Re-run the compiler using the artifact's recorded source roots**

Use the same dataset roots, backend labels, ID key, source quota, and compiler
options recorded by the existing manifest/private provenance, changing only the
output directory back to `candidate_qa_combined`.

- [ ] **Step 3: Audit every generated Q6 closed answer**

For every private Q6 closed answer with sufficient evidence, assert:

```python
expected = (
    "closer" if numeric_gt["distance_change_m"] < 0.0 else
    "farther" if numeric_gt["distance_change_m"] > 0.0 else
    "unchanged"
)
assert structured_answer["choice_id"] == expected
```

Also assert the manifest and every public item carry
`visible-ground-disc.v7`. Private answers carry the QA schema rather than a
prompt stamp, so join them by item ID and audit every Q6 numeric GT against the
strict label rule.

- [ ] **Step 4: Run end-to-end self-consistency**

Run:

```bash
$PY scripts/check_benchmark.py \
  data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined
$PY scripts/eval_benchmark.py \
  --benchmark data/candidate_pool/preview_v6_20260725_d9e37486a696/candidate_qa_combined \
  --gt-as-pred
```

Expected: validation passes and every applicable GT-as-pred score is `1.0`.
