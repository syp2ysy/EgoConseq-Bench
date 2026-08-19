# C1 Lazy Feasible-Triple Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eager evaluation of all ten C1 suffixes with a semantics-equivalent lazy feasible-triple traversal, while deleting the superseded eager selector code and preserving batched full-geometry precomputation.

**Architecture:** `pipeline/c1_safe_suffix.py` remains the simulator-free authority. A private helper enumerates and hash-orders all structurally feasible triples before labels are observed. The public selector caches labels by action SHA and evaluates only the suffixes required to prove or reject each triple. `pipeline/collection_runtime.py` remains unchanged so its one-navmesh-per-radius batch precheck is preserved.

**Tech Stack:** Python 3.9, deterministic SHA-256 canonicalization, pytest, existing R2R Habitat smoke runner.

## Global Constraints

- Work directly on `master`; the user explicitly authorized this workspace.
- Use `/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python`.
- Write failing focused tests before production edits.
- Keep `c1-safe-suffix-family.v3`, `c1_safe_suffix_family_per_pose.v3`, query eligibility, the exact ten suffixes, and all five structural-shortcut hard gates unchanged.
- Do not modify `pipeline/collection_runtime.py` or split its ten-member full-geometry batch.
- Delete only the eager `safe_variants` accumulation and post-hoc combination code made unreachable by the lazy traversal.
- Do not delete the suffix bank, validator, review replay, shortcut audit, or Golden explicit-action-file path.
- Run the full suite and `scripts/check_abc_golden.py` before completion.

---

### Task 1: Lock Lazy/Exhaustive Equivalence in RED Tests

**Files:**
- Modify: `tests/test_pl_c1_safe_suffix.py`

**Interfaces:**
- Consumes: `c1_safe_suffix.suffix_variants(query)`, `c1_safe_suffix.action_sha256(actions)`, and `c1_safe_suffix.select_safe_suffix_with_evaluator(...)`.
- Produces: an exhaustive test-only reference and call-count assertions that the production selector must satisfy.

- [ ] **Step 1: Add a test-only exhaustive reference**

Add a helper that reproduces the current v3 behavior from the frozen ten suffixes:

```python
def _exhaustive_selected_suffixes(query, labels, frame_id):
    variants = c1_safe_suffix.suffix_variants(query)
    feasible = []
    for chosen in itertools.combinations(variants, 3):
        if {value.suffix_kind for value in chosen} != \
                set(c1_safe_suffix.SUFFIX_CATEGORIES):
            continue
        turns = [float(value.suffix[0].deg) for value in chosen]
        if not (any(value < 0.0 for value in turns) and
                any(value > 0.0 for value in turns)):
            continue
        shas = sorted(c1_safe_suffix.action_sha256(value.actions)
                      for value in chosen)
        if all(labels[sha] == "safe" for sha in shas):
            feasible.append((c1_safe_suffix._domain_hash(
                c1_safe_suffix.TRIPLE_SELECTION_RULE, {
                    "frame_id": frame_id,
                    "query_action_sha256":
                        c1_safe_suffix.action_sha256(query),
                    "suffix_action_sha256": shas,
                }), tuple(chosen)))
    return (min(feasible, key=lambda value: value[0])[1]
            if feasible else None)
```

Keep this helper in tests only; do not add a second production selector.

- [ ] **Step 2: Add the 1024-assignment equivalence test**

For every bit mask from `0` through `2**10 - 1`, create the ten label values,
run the exhaustive helper, then run the public selector with a recording fake
evaluator. Assert that both return shortfall or the same three suffix action
SHAs in the same canonical order. For successful cases, rebuild both family
objects and assert exact dict equality.

- [ ] **Step 3: Add best-case, caching, and worst-case call-count tests**

Add separate tests asserting:

```python
assert best_case_suffix_calls == 3
assert len(calls) == len(set(calls))
assert worst_case_suffix_calls <= 10
```

The query-role call is counted separately. Include a failure assignment where
overlapping triples revisit the same suffix so the no-repeat assertion is a
real regression lock.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_c1_safe_suffix.py -q
```

Expected: the new best-case/caching tests fail because the current selector
always calls the suffix evaluator ten times. Existing tests remain green.

### Task 2: Implement Lazy Traversal and Delete Eager Selector Code

**Files:**
- Modify: `pipeline/c1_safe_suffix.py:450-506`
- Test: `tests/test_pl_c1_safe_suffix.py`

**Interfaces:**
- Produces: private `_ordered_feasible_suffix_triples(variants, *, frame_id, query_action_sha256)` returning immutable ordered triples of `SuffixVariant`.
- Preserves: `select_safe_suffix_with_evaluator(query_candidates, evaluate_label, *, frame_id)` return type and family bytes.

- [ ] **Step 1: Add the minimal private ordering helper**

Implement one simulator-free helper:

```python
def _ordered_feasible_suffix_triples(
        variants, *, frame_id: str, query_action_sha256: str):
    triples = []
    for chosen in itertools.combinations(variants, 3):
        if {value.suffix_kind for value in chosen} != set(SUFFIX_CATEGORIES):
            continue
        turns = [float(value.suffix[0].deg) for value in chosen]
        if not (any(value < 0.0 for value in turns) and
                any(value > 0.0 for value in turns)):
            continue
        selected_shas = sorted(action_sha256(value.actions)
                               for value in chosen)
        triples.append((_domain_hash(TRIPLE_SELECTION_RULE, {
            "frame_id": str(frame_id),
            "query_action_sha256": str(query_action_sha256),
            "suffix_action_sha256": selected_shas,
        }), tuple(chosen)))
    return tuple(chosen for _digest, chosen in sorted(
        triples, key=lambda value: value[0]))
```

The helper observes identities only, never labels.

- [ ] **Step 2: Replace exhaustive evaluation with a label cache**

Inside the existing strict-safe query loop:

```python
labels_by_sha = {}
for chosen in _ordered_feasible_suffix_triples(
        variants, frame_id=frame_id,
        query_action_sha256=action_sha256(query[1])):
    all_safe = True
    for value in sorted(chosen, key=lambda item:
                        action_sha256(item.actions)):
        digest = action_sha256(value.actions)
        if digest not in labels_by_sha:
            labels_by_sha[digest] = evaluate_label(
                value.tag, list(value.actions), role="suffix")
        if labels_by_sha[digest] != "safe":
            all_safe = False
            break
    if all_safe:
        selected = [query] + [
            (value.tag, list(value.actions)) for value in chosen]
        return selected, _family_value(
            selected, frame_id=frame_id), variants
```

Labels are scoped per query because each query has a distinct ten-member bank.

- [ ] **Step 3: Delete only superseded eager code**

Remove the old production-only blocks:

```python
safe_variants = []
for value in variants: ...
triples = []
for chosen in itertools.combinations(safe_variants, 3): ...
if triples: ...
```

Keep `itertools`, `suffix_variants`, category/sign invariants, family building,
and all validator/audit code.

- [ ] **Step 4: Run focused selector tests and verify GREEN**

```bash
$PY -m pytest tests/test_pl_c1_safe_suffix.py -q
```

Expected: all tests pass, including all 1024 assignments and exact family
equality.

- [ ] **Step 5: Run adjacent contract suites**

```bash
$PY -m pytest \
  tests/test_pl_c1_safe_suffix.py \
  tests/test_pl_c1_suffix_shortcut_audit.py \
  tests/test_pl_v16_c_candidates.py \
  tests/test_pl_b1k_candidate_review.py -q
```

Expected: all pass with no schema, review, validator, or audit changes.

- [ ] **Step 6: Commit the implementation**

```bash
git add pipeline/c1_safe_suffix.py tests/test_pl_c1_safe_suffix.py
git diff --cached --check
git commit -m "perf: lazily evaluate C1 suffix triples"
```

### Task 3: Verify Byte Stability, Pipeline Isolation, and Runtime Evidence

**Files:**
- No tracked production changes expected.
- Generated smoke output remains untracked and is not committed.

**Interfaces:**
- Consumes: the committed lazy selector.
- Produces: full-suite, Golden, and real-smoke evidence.

- [ ] **Step 1: Run the full deterministic suite**

```bash
$PY -m pytest -q
```

Expected: exit 0.

- [ ] **Step 2: Run the standard Golden gate**

```bash
$PY scripts/check_abc_golden.py
```

Expected: 19 byte-identical files, four clean shards, task coverage
`10/2/2/2/2/6`, and GT-as-pred `1.0`.

- [ ] **Step 3: Run the same-seed R2R suffix smoke**

Use a fresh explicit output and the same parameters as the pre-change smoke:

```bash
REV=$(git rev-parse HEAD)
OUT=data/candidate_pool/smoke_c1_lazy_r2r_20260810_${REV:0:7}
$PY scripts/collect.py \
  --backend r2r \
  --scenes E9uDoFAP3SH \
  --poses-per-scene 1 \
  --pose-candidates-per-scene 30 \
  --pose-pool-min-accepted-per-scene 20 \
  --pose-pool-max-batches 3 \
  --radii 0.2 \
  --camera-heights 1.5 \
  --fov 79 63.453048374758716 \
  --proposal-pairs-per-length 1 \
  --proposal-natural-per-length 1 \
  --c1-suffix-candidates \
  --out "$OUT" \
  --overwrite \
  --code-revision "$REV"
```

Expected: one source-valid C1 family and complete pending review.

- [ ] **Step 4: Compare semantic output with the pre-change smoke**

Read the one record from the old and new runs and assert exact equality for:

```python
old["selection"]["action_group_ids"]
old["selection"]["c1_safe_suffix_family"]
```

against the new record. Assert the new funnel keeps ten suffixes offered and
has `3 <= c1_safe_suffix_candidates_checked <= 10`. Record that suffix-side
rejection counts may be lower by design.

- [ ] **Step 5: Final repository checks**

```bash
git diff --check
git status --short --untracked-files=all
```

Expected: tracked worktree clean. The fresh generated smoke directory may be
untracked/ignored and must not be added to the commit.
