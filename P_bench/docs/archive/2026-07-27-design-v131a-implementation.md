> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# EgoConseq-Bench v1.3.1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Design Contract v1.3.1a end to end, recollect the missing continuous GT on HM3D/R2R/GS, compile a current QA artifact, and serve an All Cases viewer that explains every result head, intervention, question, and GT.

**Architecture:** Keep deterministic schema derivation, target geometry, QA projection, aggregation, and audits in `pipeline/`. Simulator-only rendering and contact attribution remain in `pipeline/sim.py` and `pipeline/gs_*.py`. Collection persists a two-layer future-state substrate; the artifact compiler applies visible-support gates and projects it into forced-choice tasks; the viewer reads only the validated artifact plus explicitly packaged private review fields.

**Tech Stack:** Python 3.9, NumPy, pytest, existing Habitat/GS adapters, static HTML/CSS/JavaScript, existing Python HTTP servers.

---

## Contract And File Map

| Unit | Responsibility | Primary files |
|---|---|---|
| Historical audit | Read-only yield evidence from old `conseq.v6`; never creates formal GT | `pipeline/funnel_audit.py`, `scripts/audit_design_funnel.py` |
| Future-state substrate | Stable base/target keys, fixed target reference set, physical/observation projections | `pipeline/record.py`, `pipeline/rollout.py`, `pipeline/consequence.py` |
| Record trust boundary | Recompute hashes, ranges, causes, masks and intervention invariants | `pipeline/validate.py` |
| Candidate sampling | Variable action counts, matched actions, directed body/sensor/Q8/Q9 reserves | `scripts/collect.py`, a NumPy-only sampling helper |
| QA projection | Forced-choice v1.3 tasks and semantic result-head metadata | `pipeline/benchmark.py`, `scripts/gen_benchmark.py` |
| Evaluation | Item-level null correction and six-level head aggregation | `scripts/eval_benchmark.py` |
| Release audits | Option/action/body/profile shortcuts and raw-vs-published nuisance deltas | `pipeline/shortcut_audit.py`, audit CLI |
| Review UI | Four-head navigation and complete per-case GT/provenance rendering | `scripts/serve_benchmark.py`, `scripts/benchmark_viewer.html` |

## Task 1: Close And Run The Historical Funnel Audit

**Files:**
- Modify: `pipeline/funnel_audit.py`
- Modify: `tests/test_pl_funnel_audit.py`
- Add: `scripts/audit_design_funnel.py`
- Add: `docs/superpowers/plans/2026-07-27-unified-funnel-audit.md`

- [ ] **Step 1: Add failing audit-contract tests**

Add tests that:

```python
def test_sensor_heading_comparison_is_wrap_aware():
    # +180 and -180 are the same physical heading.
    assert sensor_siblings_are_physically_equal(left, right)

def test_historical_target_metric_is_explicitly_a_proxy():
    assert row["target_metric_is_proxy"] is True
    assert row["usable_for_formal_propagation_gt"] is False
```

- [ ] **Step 2: Run the new tests and observe the wrap-around failure**

Run:

```bash
$PY -m pytest tests/test_pl_funnel_audit.py -v
```

- [ ] **Step 3: Make heading comparison wrap-aware and terminology current**

Use the shortest signed angular difference for realized headings. Rename audit-only `World Relation` output keys to `Endpoint & Target Relation`; keep a report note that old visible centroids and outcome-level evidence are feasibility proxies, not v1.3 compiled Q5a.

- [ ] **Step 4: Verify and commit the audit**

Run the targeted tests and full suite. Commit only the four reviewed audit files.

- [ ] **Step 5: Run on all current v6 records**

Run:

```bash
$PY scripts/audit_design_funnel.py \
  --records data/candidate_pool \
  --out data/audits/design_funnel_v131a_historical.json \
  --rows-out data/audits/design_funnel_v131a_historical.rows.jsonl
```

The report must say `historical_feasibility_only: true`.

## Task 2: Add The Two-Layer Substrate And Fixed Target Reference Set

**Files:**
- Modify: `pipeline/record.py`
- Modify: `pipeline/objects.py`
- Modify: `pipeline/outcome.py`
- Modify: `pipeline/validate.py`
- Test: `tests/test_pl_record.py`
- Test: `tests/test_pl_validate.py`

- [ ] **Step 1: Write failing schema tests**

Require every new record to contain:

```python
record["substrate"] == {
    "base_rollout_key_version": "base-rollout.v1",
    "target_projection_key_version": "target-projection.v1",
}
record["target_reference_sets"][str(instance_id)] == {
    "instance_id": instance_id,
    "frame": "world_xz",
    "points_xz_m": ...,
    "sha256": ...,
}
```

Assert two sensor/body/action siblings with the same target carry the identical point-set hash, while Q1/Q2/Q3/Q9 base keys do not contain a target id.

- [ ] **Step 2: Verify failures**

Run the two targeted files and confirm missing `substrate` / `target_reference_sets`.

- [ ] **Step 3: Implement deterministic keys and reference-set serialization**

Canonicalize point rows lexicographically, quantize only for stable serialization, hash canonical JSON, and store full private points once per record. Add pure functions:

```python
base_rollout_key(record, outcome) -> str
target_projection_key(base_key, target_instance_id) -> str
target_reference_set(points_xz, instance_id) -> dict
target_range_m(center_xz, reference_set) -> float
```

- [ ] **Step 4: Validate from source, not sibling fields**

The validator recomputes point-set hashes and keys, rejects non-finite/empty geometry, and checks target identity across intervention siblings.

- [ ] **Step 5: Run tests and commit**

## Task 3: Persist The Canonical Future-State Vector

**Files:**
- Modify: `pipeline/rollout.py`
- Modify: `pipeline/consequence.py`
- Modify: `pipeline/record.py`
- Modify: `pipeline/validate.py`
- Test: `tests/test_pl_rollout.py`
- Test: `tests/test_pl_goal_relations_v5.py`
- Test: `tests/test_pl_validate.py`

- [ ] **Step 1: Add failing future-state tests**

For each outcome require:

```python
outcome["future_state"] == {
    "collision": bool,
    "executed_action_prefix": list,
    "stop_stage": int | None,
    "realized_endpoint": {
        "x_m": float, "z_m": float, "heading_deg": float,
        "translation_m": float, "heading_change_deg": float,
    },
    "terminal_physical_safe_mask": [bool] * 5,
    "start_physical_safe_mask": [bool] * 5,
}
```

For each target projection require `target_range_before_m`,
`target_range_after_m`, terminal ego bearing, camera depth, future visibility,
and `invisible_cause in {None, "out_of_fov", "occluded"}`.

- [ ] **Step 2: Verify failures**

- [ ] **Step 3: Derive the vector mechanically**

Build the base vector from full-geometry rollouts and terminal probes. Run the same fixed five probes at the start pose. Build target projections from the fixed private target point set and actual realized endpoint; never use radius-subtracted surface gap for `target_range_m`.

- [ ] **Step 4: Distinguish Q8 causes**

Classify a target outside the terminal frustum as `out_of_fov`; classify a target whose reference geometry projects inside the frustum but whose rendered instance is absent/occluded as `occluded`. Persist the geometric test inputs for validator recomputation.

- [ ] **Step 5: Add source-derived validation**

Recompute endpoint diagnostics, both masks, range, bearing and visibility cause. Add mutation tests for each field.

- [ ] **Step 6: Verify and commit**

## Task 4: Remove Fixed 3+3 Sampling And Add Formal Matched-Action Units

**Files:**
- Create: `pipeline/action_sampling.py`
- Modify: `scripts/collect.py`
- Modify: `pipeline/validate.py`
- Test: `tests/test_pl_candidate_builder.py`
- Test: `tests/test_pl_validate.py`
- Test: `tests/test_pl_actions.py`

- [ ] **Step 1: Add failing sampling tests**

Test variable total/safe counts across poses, private-only pair ids, and exact matching on total distance bucket, primitive count, total/net turn, direction sequence and per-leg distance profile.

- [ ] **Step 2: Verify the old exact-3+3 assertions fail the tests**

- [ ] **Step 3: Implement a pure matching/selection module**

The collector persists a variable natural candidate set plus optional matched units. It does not balance labels per pose. The validator rejects a reintroduced fixed-count declaration and checks match-key equality without requiring every action to be paired.

- [ ] **Step 4: Verify and commit**

## Task 5: Add Directed Reserve Modes Required By v1.3.1a

**Files:**
- Modify: `scripts/collect.py`
- Modify: `pipeline/sim.py`
- Modify: `pipeline/gs_sim.py`
- Modify: `pipeline/config.py`
- Test: `tests/test_pl_collect_resume.py`
- Test: `tests/test_pl_candidate_builder.py`

- [ ] **Step 1: Add failing scheduler tests**

The explicit mode list must support:

```text
main
q4_direct
body_propagation
q4_sensor
target_pair
q8_fov
q8_occlusion
q9_mask_pair
q10
```

Assert `height-only` siblings hold FOV fixed and `FOV-only` siblings hold height fixed.

- [ ] **Step 2: Verify failures**

- [ ] **Step 3: Implement directed selection without evidence early filtering**

Search physical candidates first. Render all required siblings only after physical survival. Keep every profile's evidence status separately. Persist per-radius contact instance/source ids and target reference sets. Give `away + large-body early stop` its own budget.

- [ ] **Step 4: Implement Q8/Q9 necessity reserves**

Q8 FOV pairs share physical endpoint and differ only in FOV. Q8 occlusion pairs are matched actions at the same sensor profile. Q9 pairs must have different endpoint masks; all use the fixed five probes.

- [ ] **Step 5: Verify and commit**

## Task 6: Bump The Record/Prompt/QA Contracts

**Files:**
- Modify: `pipeline/record.py`
- Modify: `pipeline/benchmark.py`
- Modify: `pipeline/validate.py`
- Modify: `scripts/serve_benchmark.py`
- Modify: `AGENTS.md`
- Test: `tests/test_pl_v6_contract.py`

- [ ] **Step 1: Write failing trust-boundary tests**

Require reader, record validator, artifact validator, evaluator, baselines and HTTP loader to reject pre-v1.3 records/artifacts.

- [ ] **Step 2: Verify failures**

- [ ] **Step 3: Bump contracts exactly once**

Freeze the implementation stamps as:

```text
record schema:       conseq.v6                 -> conseq.v7
oracle contract:     ground-disc-visible-v4    -> ground-disc-visible-v5
benchmark taxonomy:  v7                        -> v8
prompt contract:     visible-ground-disc.v10   -> visible-ground-disc.v11
QA schema:           egoconseq.qa.v9            -> egoconseq.qa.v10
```

Do not add legacy adapters.

- [ ] **Step 4: Verify all entry points and commit**

## Task 7: Compile Sufficient-Only Forced-Choice Tasks By Result Head

**Files:**
- Modify: `pipeline/benchmark.py`
- Modify: `scripts/gen_benchmark.py`
- Modify: `pipeline/question_templates.py`
- Modify: `pipeline/validate.py`
- Test: `tests/test_pl_questions.py`
- Test: `tests/test_pl_question_levels.py`
- Test: `tests/test_pl_candidate_builder.py`

- [ ] **Step 1: Write failing task-projection tests**

Require public metadata:

```python
item["result_head"] in {
    "interaction", "endpoint_target_relation",
    "future_observation", "local_affordance",
    "constraint_integration", "diagnostic",
}
item["rollout_stage"] in {0, 1, 2, 3, 4}
item["intervention_protocol"] in {
    "absolute", "do_action", "do_body",
    "do_sensor_height", "do_sensor_fov",
    "information_ablation",
}
```

Assert no public choice or structured answer contains
`insufficient_evidence`/`answerable`.

- [ ] **Step 2: Verify failures**

- [ ] **Step 3: Implement Q1–Q10 projections from the substrate**

Q6/Q0a use strict sign of `target_range_after_m - target_range_before_m`; a margin withholds the item rather than rewriting it as unchanged. Q7 uses actual realized endpoint. Q8 is binary plus open center/area/cause diagnostics. Q9 uses terminal physical mask and includes start mask only in private review metadata. Q4 uses four radii, three adjacent transition choices, no singular radius.

- [ ] **Step 4: Implement necessity filters**

Persist private filter results:

```python
{
  "matched_action_label_change": bool,
  "q6_direction_heuristic_resistant": bool,
  "q7_rotation_only_sector_differs": bool,
  "q8_pair_type": "fov" | "occlusion" | None,
  "q9_endpoint_mask_differs_from_start": bool,
  "q9_matched_action_mask_pair": bool,
}
```

- [ ] **Step 5: Add four information-ablation views**

For the same item id stem compile `full`, `no_height`, `no_fov`, `no_both` inputs. The RGB and GT remain identical; only public calibration fields change.

- [ ] **Step 6: Verify and commit**

## Task 8: Implement Sensor Pairs, Body Propagation, And Integration Tracks

**Files:**
- Modify: `pipeline/benchmark.py`
- Modify: `scripts/gen_benchmark.py`
- Modify: `pipeline/validate.py`
- Test: `tests/test_pl_questions.py`
- Test: `tests/test_pl_candidate_builder.py`

- [ ] **Step 1: Add failing pairing tests**

Cover height-only and FOV-only physical invariance pairs, FOV-responsive Q8 pairs, flip-based propagation, BC-simple diagnostic and BC-branching primary candidates.

- [ ] **Step 2: Verify failures**

- [ ] **Step 3: Compile pairs with shared private identities**

Every pair stores its pair type, controlled variables, manipulated variable, both item ids, physical/reference-set invariants and expected effect transition. Missing required heads fail release rather than silently renormalizing.

- [ ] **Step 4: Implement radius-independent propagation GT**

Use center-to-fixed-reference-set distance. Enforce endpoint/distance grey zones and direction labels. BC-simple remains diagnostic.

- [ ] **Step 5: Verify and commit**

## Task 9: Replace Level Scoring With Head/Intervention Scoring

**Files:**
- Modify: `scripts/eval_benchmark.py`
- Modify: `pipeline/shortcut_audit.py`
- Test: `tests/test_pl_eval_qa.py`
- Test: `tests/test_pl_shortcut_audit.py`

- [ ] **Step 1: Add failing aggregation tests**

Implement item-level:

```python
s_adj_i = (score_i - null_i) / (1.0 - null_i)
```

then aggregate item → intervention group → scene → variant → task → result head. The VCS is the equal-weight macro over four required atomic heads; Integration and Direct Body are separately required outputs.

- [ ] **Step 2: Verify failures**

- [ ] **Step 3: Implement raw and null-corrected reports**

Report head scores, VCS, Direct Body, Propagation Tier, Action Fidelity, Sensor Physical Invariance split by height/FOV, View Responsiveness split by height/FOV, Calibration Utility across four ablations, and Integration.

- [ ] **Step 4: Extend release audits**

Run attacks within family × variant. Add profile/template/global-stat nuisance attacks and raw-vs-published deltas. Keep RGB/depth/local-geometry probes diagnostic only.

- [ ] **Step 5: Verify and commit**

## Task 10: Run Historical Audit And Three-Backend Directed Collection

**Files:**
- Generated: `data/audits/*.json`
- Generated: `data/conseq/v131a_pilot_*`
- Generated: logs under `data/conseq/logs/`

- [ ] **Step 1: Run the historical audit**

Use it only to size directed budgets.

- [ ] **Step 2: Run one-scene smoke per backend**

Use separate visible GPUs for HM3D, R2R and GS. Validate every nonempty shard and inspect one image/record manually before scaling.

- [ ] **Step 3: Launch the preregistered pilot**

Run at least 10 independent scenes per source and all required reserve modes. Each process writes a distinct root and log. Monitor GPU/process/shard progress until terminal.

- [ ] **Step 4: Re-run the unified funnel**

Select the fixed four-radius grid, grey zones, pair quotas and Propagation Tier only through the document's preregistered branches. Amend v2.0 Publication Contract with those numbers before formal collection.

- [ ] **Step 5: Run formal collection if the pilot supports it**

If a branch is inconclusive, add scenes. Never relax a release gate to make the first artifact compile.

## Task 11: Compile And Validate The v1.3.1a QA Artifact

**Files:**
- Generated: `data/benchmark/egoconseq_v131a_preview/`

- [ ] **Step 1: Compile with scene-atomic splits**

Include only evidence-sufficient items and package private review metadata separately.

- [ ] **Step 2: Run all trust and release checks**

Run record validation, artifact validation, shortcut audits, GT-as-pred evaluation and per-head yield tables. A preview may declare unresolved release gates, but the UI must display them.

- [ ] **Step 3: Inspect representative cases**

Manually inspect at least one case per head, each Q family, do(body), both sensor factors and each information-ablation condition.

## Task 12: Build The Four-Head All Cases Viewer

**Files:**
- Modify: `scripts/benchmark_viewer.html`
- Modify: `scripts/serve_benchmark.py`
- Modify: `tests/test_pl_benchmark_ui.py`
- Modify: `tests/test_pl_serve_benchmark.py`

- [ ] **Step 1: Write failing DOM/API tests**

Require navigation for the four atomic heads, Integration, Direct Body and Diagnostic. Require All Cases cards to expose complete prompt, all choices, GT, target, action, body/sensor calibration, future-state vector, realized endpoint, start/end masks, sibling links, necessity filters, publication state and provenance.

- [ ] **Step 2: Verify the current L0–L4 UI fails**

- [ ] **Step 3: Implement the review-focused layout**

Use a compact research-console layout:

- persistent head/task filter rail;
- coverage matrix showing case counts per head × intervention;
- dense card list with image and full QA/GT;
- expandable substrate/filters/provenance sections;
- direct links between matched action/body/sensor siblings;
- no human-score workflow on the primary screen.

Cards use ≤8 px radii, restrained neutral surfaces, semantic colors only for GT/filter state, Lucide-equivalent existing icons if available, and responsive stable dimensions.

- [ ] **Step 4: Add a design guide panel**

For every result head display: what it tests, controlled variables, manipulated variable, GT source, necessity filter, one canonical question and expected failure interpretation.

- [ ] **Step 5: Verify with browser screenshots**

Serve the compiled artifact, capture desktop and mobile screenshots with headless Chrome, verify no blank images, clipped question/GT, overlap or hidden All Cases content.

- [ ] **Step 6: Run full tests and start the final server**

Report the live local URL and exact artifact root.

## Final Verification

Run:

```bash
$PY -m pytest -q
$PY scripts/check_records.py <every-new-record-shard>
$PY scripts/check_benchmark.py --benchmark <artifact-root>
$PY scripts/eval_benchmark.py --benchmark <artifact-root> --gt-as-pred
```

Then verify the HTTP artifact loader, All Cases API, desktop screenshot, mobile screenshot, and one item from every published task family.
