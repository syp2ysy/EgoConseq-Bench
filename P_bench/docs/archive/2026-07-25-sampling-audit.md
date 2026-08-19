# EgoConseq-Bench — Sampling & Contract Audit (2026-07-25)

Evidence document for [`2026-07-25-sampling-freeze.md`](./2026-07-25-sampling-freeze.md).
This file holds findings and measurements only; the normative design lives in the
freeze document.

**Snapshot.** All numbers measured on `train_1to1to1_v3_20260724` (now
`diagnostic-only`) at 245–248 records. Collection was still running during the
audit, so counts shift by ±3 between measurements. Any figure entering a paper
must be re-bound to a frozen pool snapshot id.

**Method.** 16 subagents: 5 read one pipeline stage each, 5 assessed one quality
dimension each, 5 adversarially attempted to refute those assessments, 1
synthesised. 3.47 M tokens, 1071 tool calls. Refuted claims are recorded in §3 so
they are not reintroduced.

F-numbers are stable identifiers in discovery order; findings are grouped below
by subsystem.

---

## 1. Sampling shortcuts

### F1 — Constraint-propagation shortcut (blocking)

`goal.md` publishes "exactly three safe plus three collision programs" per pose,
and `intervention_group_id` is public. Rank the 6 items in each
`(intervention_group_id, body_radius_m)` cell by total commanded forward distance
and label the longest 3 as collision:

- **Q1 headline accuracy 0.9838** (balanced artifact), 0.9955 (raw pool)
- **741/741** complete cells contain exactly 3 collisions

Hiding `intervention_group_id` is not a fix — identical images, `observation_id`
and image hashes all recover the equivalence class.

### F2 — Distance shortcut (blocking)

A single public scalar (sum of forward distances), scene-disjoint 5-fold CV:

| family | random | majority | distance-only |
|---|---|---|---|
| Q1 collision | .333 | .500 | **.9089** |
| Q9 forward_safe | .333 | .667 | **.898** |
| Q6 surface_proximity | .250 | .661 | **.790** |
| Q2 stop_progress | .250 | .414 | **.767** |

Root cause: within a pose, collision actions average **+3.41 m** longer than safe
actions — **246/246 poses, zero exceptions**. Collision rate by nominal forward
distance is effectively deterministic: 0.000 below 1.5 m (n=1512), 0.981–1.000
above 5.5 m (n=930).

### F3 — Q4: radius band **and** a direct answer leak (blocking)

**(a) Scene-independent radius band.** A single global threshold `t = 0.2652 m`
classifies the 45 counterfactual (radius, label) pairs at **0.9556**.
Safe radii ∈ [0.100, 0.250], collision radii ∈ [0.250, 0.466].
Driven by `BENCH_Q4_FLIP_RATIO = 1.5` forcing a ≥50% gap exactly at the boundary,
over a scene-independent search range of `(0.10, 0.50)`.

**(b) The published body radius *is* the answer.**
`pipeline/benchmark.py:build_radius_counterfactual_pair` sorts the outcomes by
radius and uses `radii[0]` — the smallest — as the singular `body_radius_m`
written into `model_input`, while the same list becomes the candidate options.

| | balanced (n=11) | unbalanced (n=15) |
|---|---|---|
| `model_input.body_radius_m` == min(candidate radii) | **11/11** | **15/15** |
| `model_input.body_radius_m` == GT `max_safe_radius_m` | **8/11 = 72.7%** | **12/15 = 80%** |

The earlier "pick the smallest radius scores 72.7%" is therefore not a
distributional accident: the prompt hands the model a scalar equal to the answer
in ~3 of 4 items. Independent of (a), and unaffected by any option-order or label
balancing.

Together these make the family carrying R1 ~96% solvable without the image.

### F13 — Balancing silently deletes whole strata (blocking)

`scripts/gen_benchmark.py:534-570`: a `(family, variant, backend)` stratum with a
single answerable label and no insufficient sibling falls through to
`audit[stratum_key] = {..., "after": {}, "reason": "single_answer_stratum"}` and
is dropped, with **no gate failure and no warning**.

### F14 — The level track, not equalisation, is the dominant attrition (major)

Measured stage counts:
**58 982 raw eligible → 16 391 after track+cap (−72%) → 10 664 after balance →
2 427 after equalisation.**
`SINGLE_ANSWER_TRACK` with `TRACK_WEIGHTS L0 15 / L1 30 / L3 30 / L4 25`
(`scripts/gen_benchmark.py:46-52`) gives each image exactly one level, which
structurally caps every single-answer family's yield at its track weight. Yield
planning that models equalisation as the main loss is wrong by ~3×.

---

## 2. Contract and GT defects

### F4 — Published evidence value is binary (blocking for R3)

`pipeline/rollout.py` computes
`qualified_coverage = raw_coverage if meets_visibility_contract else 0.0`, where
`meets_visibility_contract` conjoins **five** conditions (`raw_coverage >= 0.90`,
`out_of_frame_samples == 0`, `occluded_samples == 0`,
`invalid_depth_samples <= max`, terminal + gap). Only the qualified value is
persisted: over all 248 insufficient outcomes the stored `evidence.physical`
carries exactly four keys — `coverage`, `coverage_protocol`, `status`,
`view_collision_estimate`. **`raw_coverage` and the per-cause counters are
computed but never written.**

Assertable: the published evidence value is two-valued `{0.0, 1.0}`, so
`EVIDENCE_COVERAGE_MIN = 0.90` never acts as a continuous threshold; all 4446
formal outcomes are `sufficient`, all 248 `insufficient` outcomes are hand-injected
selective probes.

**Not** assertable (an earlier draft did): that every insufficient probe is
"entirely out of frame". Any one of five conditions collapses the value, and v3
did not persist which fired.

### F5 — Effective camera height is wrong (blocking)

The agent root sits on the Recast navmesh, whose Y is quantised by
`cell_height = 0.2 m`. True optical-centre height above the *visible* floor is
`sensor_offset - floor_y`, but the prompt publishes the sensor offset
(`pipeline/benchmark.py:961` `_model_input` writes it directly). Measured
`floor_y` over 248 records: range **[-0.225, +0.275] m**, median **-0.125 m**,
p90 |floor_y| **0.175 m**.

Also affected: `pipeline/rollout.py:724` `certified_near_field_m` computes
`ground_entry = h / tan(vfov/2)` from the raw offset; `pipeline/rollout.py:807`
projects corridor ground samples to `y = 0` rather than `frame.floor_y`.

### F12 — `safe_mask` conflates three distinct facts (major)

`pipeline/rollout.py:643-657` computes `safe = [bool(value and eligible) ...]`,
folding together (i) physically non-colliding, (ii) certifiable from the image,
(iii) inside the publication margin. That single Boolean becomes the Q9 answer
key, while the question asks about physical safety.

Consumers — **five**, not one:
`pipeline/rollout.py:671` (produced), `pipeline/validate.py:668,685`,
`pipeline/benchmark.py:923` (Q9), `pipeline/benchmark.py:1433`
(Q10 `has_two_safe_options`), `pipeline/benchmark.py:1524` (Q10 constraint slack).
Note that `benchmark.py:1433/1524` already *name* the variable
`physical_safe_mask` — the name is wrong, which is itself evidence of the hazard.

### F7 — GS navmesh-snap contamination (major)

`pipeline/gs_geometry.py:163` returns non-navigable whenever the pre-baked navmesh
snap fails, even with zero Gaussian overlap: **82 of 427 published GS collisions
(19.2%)** have `distance_m > 0`. Such events also shift the realized stop pose,
so they contaminate every downstream consequence, not just Q1/Q3.

### F8 — Q4 margin contract conflict (major)

`pipeline/config.py:189` uses `BENCH_Q4_SAFE_CLEARANCE_M = 0.10` while `goal.md`
promises ≥ 0.30 m for safe examples. All 18 published Q4 safe rollouts lie in
[0.100, 0.147] m.

### F10 — Missing cross-sibling invariant (minor)

`_physical_signature` (`pipeline/validate.py:937`) covers only three scalars
(`collision`, `first_contact_arc_m`, `minimum_clearance_m`). The Q9 terminal
`full_geometry_safe_mask` and other physical consequences are unchecked across
sensor siblings.

---

## 3. Release-gate deadlocks

### F6 — Q3 has **two** independent deadlocks (major)

**(a) Impossible stratum.** `EXPECTED_CLOSED_STRATA` requires `no_collision`, but
`pipeline/benchmark.py:572` rejects Q3 whenever `collision == False`.

**(b) Singleton-label deadlock — survives fixing (a).**
`scripts/gen_benchmark.py:1566-1569` fails release when
`min(labels.values()) < RELEASE_MIN_LABEL_SUPPORT` (=3) over **all** of a family's
labels, while `balance_qa_pairs` deliberately preserves each label at
`min(len(values), 2*rarest)`. Q3's label space is unbounded free-text contact
categories, so one singleton long-tail category pins the minimum at 1 at any pool
size.

Corollary: **GS is not incapable of Q3.** GS produces 9 Q3 closed queries (33 raw
across 5 intervention groups), all carrying the single label `teatable`, so
`balance_qa_pairs` takes the `len(regular) < 2` path and drops the stratum. The
gate line "backend gs Q3 closed queries 0 < 3" comes from *balancing*, not from an
attribution-capability limit.

### F11 — Sensor-consistency grouping destroys exactly the Q5b evidence (blocking for R3)

`scripts/gen_benchmark.py:760-770` compares the **full `structured_answer`**
across sibling profiles and records a violation (`continue`, no group formed)
whenever signatures differ. For a closed item with insufficient evidence,
`structured_answer` **is** `insufficient_evidence`. So every sibling pair whose
evidence flips across sensor profiles — precisely the Q5b sample — is classified
as a physical-consistency violation and discarded.

---

## 4. Scale and power

### F9 — Clustering inflates apparent scale (major)

58 423 candidate QA items map to only **248 unique images / 114 scenes**. Measured
ICC 0.07–0.32, DEFF **1.9–4.1**: nominal n overstates precision by 2–4×.

Pool ceilings under the v3 policy (`min_accepted=3`, `max_attempts=24`,
`--poses-per-scene 1`): hm3d ~400 / r2r ~181 / gs ~138 groups. r2r **terminated**
at 177 accepted from 744 attempts (23.8%) over 61 scenes. Per-backend artifact
retention is backend-dependent (hm3d 1.00, gs 0.878, r2r 0.859), and equalisation
costs the *largest* backend, not the smallest.

The headline rests on **two** effectively single-family levels: L2 = Q4 alone, and
L4 = Q9 alone while Q10 is empty (`pipeline/benchmark.py:38-42`).
`primary_closed_macro` is additionally restricted to `evidence_status=='sufficient'`
∧ `evidence_partition=='primary_balanced'` (`scripts/eval_benchmark.py:534-540`),
so it never measures abstention at all.

---

## 5. Corrections recorded to prevent regression

Asserted during the audit and **wrong**. Do not reintroduce.

| Wrong claim | Correction |
|---|---|
| "Q10 is structurally unreachable at any pool size" | True only for **main** shards. `--require-q10-witness` runs `_run_q10_structured_reserve` (`scripts/collect.py:~2305`), which shortlists and evaluates same-length candidates. The defect is that `--coverage-first` never schedules reserve shards. |
| "Q9 turn_choice left/right is unreachable" | The 92-outcome figure conditions on publication eligibility, dropping 233 of 325 certified-asymmetric outcomes. Of the full 325, **160 (49.2%) do have both turn probes evidence-sufficient** — excluded by the near-boundary publication margin, not by the evidence contract. |
| "The monocular-depth baseline has a 100% ceiling by construction" | Conflates two baselines. **Stored rendered depth** participates in the agreement gate → *privileged upper bound*. **RGB → estimated monocular depth → geometry** never participates in filtering and remains the most important fair baseline. |
| "Q4 is depth-proof" | False. Disc-radius expansion is standard C-space Minkowski inflation. Q4's contribution is *body-conditioned causal evaluation*, not irreducibility to geometry. |
| "Hiding `intervention_group_id` fixes the group attack" | Insufficient — identical images recover membership. Removing the fixed 3+3 count is the fix. |
| "All closed questions have 3 options / 33% random" | False. All 13 closed variants build options dynamically. Measured: Q1 3, Q2 4, Q3 4, Q4 5, Q6 4, Q7 5, Q8 3 or 5, Q9 3 or 5, Q0a 5 → random floors 0.200–0.333. |
| "Q4's random baseline is 1/3" | Q4 closed items carry 5 options (3 radii + `none_safe` + `insufficient_evidence`) in 11/11 measured → uniform-random **0.20**. |
| "Constant-abstain scores well on the headline" | False — measured **0.0000** on `primary_balanced` closed items (n=5281). Its value is as an evidence-metric control. |
| "Every insufficient probe is entirely out of frame" | Over-read of a *qualified* coverage value. See F4. |
| "GS cannot produce Q3" | GS produces 9 Q3 closed queries; they are dropped by single-label balancing. See F6. |
| "`assign_scene_splits` is dead code" | **Wrong.** `split_mode="benchmark"` is the default of `write_qa_artifact` (`scripts/gen_benchmark.py:1333`); it is merely bypassed because the controller unconditionally passes `--candidate-pool`. The formal compiler must explicitly enter `split_mode="benchmark"`; do not delete it. |
| "Only L2 is a single-family level" | **L4 is too** — with Q10 empty, L4 = macro(Q9). |
| "The headline measures abstention" | It never does; it is gated to `sufficient` ∧ `primary_balanced`. |
| "Novelty is dead because each clause has an owner" | Methodologically invalid — every benchmark decomposes into owned clauses. The test is whether anyone measures the *joint* construct. |
| "~40 release gates at `gen_benchmark.py:215-235`" | That range defines **19** `RELEASE_MIN_*` constants; the gates themselves are constructed at `:1330-1500` and the per-backend/per-family loops. |
