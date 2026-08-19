> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# Unified Funnel Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only audit over all current `conseq.v6` continuous records that measures Q4 direct-body and body-propagation yield without counting sensor siblings or copied shards as independent evidence.

**Architecture:** Put deterministic record grouping, physical-pair classification, and aggregation in `pipeline/funnel_audit.py`. Keep filesystem discovery, version-gated loading, provenance, and atomic report output in `scripts/audit_design_funnel.py`. The physical audit uses realized pose and persisted target centroids; it never uses radius-dependent surface distance as a propagation readout.

**Tech Stack:** Python 3.9, standard library, NumPy-free record analysis, pytest, existing `pipeline.record` and `pipeline.io_utils`.

---

### Task 1: Physical group extraction

**Files:**
- Create: `pipeline/funnel_audit.py`
- Create: `tests/test_pl_funnel_audit.py`

- [ ] **Step 1: Write failing tests**

Add fixtures with duplicate source copies and sensor siblings. Assert that one
`backend × scene × intervention group × probe group × action` is counted once,
that four-radius completeness is explicit, and that conflicting physical
signatures for the same radius fail closed.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_funnel_audit.py -v
```

Expected: collection fails because `pipeline.funnel_audit` does not exist.

- [ ] **Step 3: Implement deterministic extraction**

Group Q4 outcomes by their physical identity, validate exact action/radius
membership, compare collision/contact/realized-pose signatures across copies,
and retain per-profile evidence separately from the deduplicated physics.

- [ ] **Step 4: Verify extraction tests pass**

Run the targeted pytest command and require all extraction tests to pass.

### Task 2: Direct and propagation funnels

**Files:**
- Modify: `pipeline/funnel_audit.py`
- Modify: `tests/test_pl_funnel_audit.py`

- [ ] **Step 1: Write failing classification tests**

Cover monotone direct flips, all-safe/all-collision controls, flip-based pairs,
both-collide simple pairs, and both-collide branching pairs.

- [ ] **Step 2: Verify the new tests fail**

Run the targeted pytest command and confirm missing classification behavior.

- [ ] **Step 3: Implement pair classification**

Classify a both-collide pair as simple only when contact leg, physical contact
source/instance, and execution branch agree. Compute endpoint delta from realized
pose centers. Compute target-distance delta from initial-frame target centroid
and realized pose center; report unavailable when target geometry is absent.

- [ ] **Step 4: Verify the new tests pass**

Run the targeted pytest command.

### Task 3: Report and CLI

**Files:**
- Create: `scripts/audit_design_funnel.py`
- Modify: `pipeline/funnel_audit.py`
- Modify: `tests/test_pl_funnel_audit.py`

- [ ] **Step 1: Write failing report/CLI tests**

Assert backend/scene/group effective counts, stage-wise funnel counts, missing
metric counts, strict v6 loading, recursive shard discovery, and deterministic
JSON output.

- [ ] **Step 2: Verify the new tests fail**

Run the targeted pytest command.

- [ ] **Step 3: Implement report and entry point**

Load through `pipeline.record.read_records`, hash every input shard, attach code
revision/dirty state, write a bounded physical-group JSONL audit table and a
summary JSON atomically, and label all non-implemented image attacks as pending
rather than silently passing them.

- [ ] **Step 4: Verify the new tests pass**

Run the targeted pytest command.

### Task 4: Run the audit

**Files:**
- Generated, uncommitted: `data/audits/design_funnel_v1.json`
- Generated, uncommitted: `data/audits/design_funnel_v1.rows.jsonl`

- [ ] **Step 1: Run targeted and full tests**

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY -m pytest tests/test_pl_funnel_audit.py -v
$PY -m pytest -q
```

- [ ] **Step 2: Audit all current v6 records**

```bash
$PY scripts/audit_design_funnel.py \
  --records data/candidate_pool \
  --out data/audits/design_funnel_v1.json \
  --rows-out data/audits/design_funnel_v1.rows.jsonl
```

Obsolete `conseq.v5` shards are reported as rejected inputs and are never
adapted into the current contract.

- [ ] **Step 3: Interpret without overclaiming**

Report effective physical groups and scene support, both-collide simple versus
branching yield, direct-flip yield, target-metric availability, and which
candidate-freeze questions remain unanswerable. Do not freeze thresholds from
an audit with insufficient independent scene support.
