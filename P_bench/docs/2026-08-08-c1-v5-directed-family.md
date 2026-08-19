# C1 v5: directed matched counterfactual families

Status: superseded historical design. The directed matched/atomic production
path was removed on 2026-08-10 and must not be used for new collection. Active
candidate C1 is `c1_safe_suffix_family_per_pose.v3`; this document is retained
only as design history.

## Task contract

C1 asks which of four simulator-rendered terminal RGB images corresponds to a
public action program from one initial egocentric RGB.  Every option must be a
completed-clear rollout from the same initial frame, scene, body, camera, and
renderer.  Newly exposed terminal surfaces are allowed.  Registration and
classical pose estimation are legitimate solution methods and must be reported
as full-input baselines, not rejected as shortcuts.

The collector first constructs an unordered four-member action family.  Family
identity, the queried member, and option positions are domain-separated hashes
of action atoms and the initial-frame identity.  They are fixed before terminal
RGB is inspected.  Pixel and semantic candidate gates operate symmetrically on
all six member pairs and may only accept or reject the whole family.

The dedicated action sampling policy is
`c1_matched_family_per_pose.v2`. Its terminal semantic metric is
`terminal-instance-histogram-l1.v2`; scene-wide structural instance IDs are
rederived from the hash-authenticated MP3D house authority. Records under this
policy contribute only C1;
they cannot enter A1--B2 populations.

Terminal asset failures under that externally authenticated policy are typed
at the operation that fails; collection never guesses a reason from exception
text. The frozen authority split is:

- `record_recomputable`: missing/incomplete/invalid checkpoints, endpoint-pose
  disagreement, and record-visible source/contract provenance;
- `collection_runtime_attested`: render-cache miss or mismatch, encoding, and
  publication failures.

The validator enforces the legal reason/authority matrix and mechanically
rebuilds checkpoint-derived reasons. Legacy depth-conditioned records may keep
their old untyped C1 withhold field: it does not participate in A1--B2, and a
directed C1 shard supersedes only their legacy C1 questions.

## Stage 0 probes

Structural capacity from an existing materialized action bank:

```bash
PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python
$PY scripts/audit_c1_family_capacity.py \
  data/candidate_pool/<run>/records/r2r/<shard>/records.jsonl \
  --collection-funnel \
  data/candidate_pool/<run>/records/r2r/<shard>/collection_funnel.json \
  --out data/candidate_pool/<run>/candidate_qa_report/c1_capacity.json
```

This report deliberately separates:

- `capped_structural_family_count`: a capped upper bound from all bank
  programs;
- `capped_formally_safe_family_count`: quartets whose four outcomes are both
  completed-clear and backed by a mechanically rebuilt shared stability
  certificate;
- `s0_with_formally_safe_family`: the primary, uncapped per-frame yield
  indicator;
- `collection_yield`: the funnel-authoritative attempted, search-ready,
  matched-family, persisted, and formally-safe counts. In particular,
  `search_ready_s0` is the denominator for Stage-0 family yield because a
  matched-family shortfall intentionally leaves no record;
- `materialized_bank_record_coverage`: whether every Stage-0 denominator
  actually carried the proposal bank needed for a structural audit;
- `realized_terminal_separation`: nearest-rank quantiles over all six endpoint
  pairs and over each family's nearest pair. These diagnostics stay in the
  report and never enter a record or GT digest.

It neither trusts `action_group_label == "safe"` nor promotes discarded,
unexecuted bank programs to formally safe outcomes. On
the 130 natural records audited during implementation, tight and medium had
zero structural families; loose had one structural family and zero formally
safe families. This confirms that the directed collector is required.

Oracle-only yield probes use one of the preregistered budgets 24, 48, or 96:

```bash
$PY scripts/collect.py <scene-and-pose-pool-arguments> \
  --c1-directed \
  --c1-oracle-only \
  --c1-proposals-per-length 24 \
  --c1-match-cell-preset tight \
  --out data/candidate_pool/<run>/records/r2r/c1-tight-b24-r00
```

For each initial frame, the collector hashes one target length from 2--6
before inspecting that pose's feasible programs,
materializes only that length, runs the existing full/depth/coverage precheck,
and atomically selects four safe members in one match cell.  The materialized
program budget is allocated as one third safe members from proposal pairs, one
third collision members from the same pairs, and one third natural controls.
`--c1-oracle-only` suppresses future rendering
and terminal PNG publication while retaining physical and stability oracles.
This blind length choice is intentionally not feasibility-adaptive: lower
yield at longer lengths is a measured property, not a cue the collector may
remove after seeing the scene.

The three calibration presets are:

| preset | planned forward | net yaw | realized displacement |
|---|---:|---:|---:|
| tight | 0.25 m | 15 deg | 0.25 m |
| medium | 0.50 m | 30 deg | 0.50 m |
| loose | 1.00 m | 45 deg | 1.00 m |

The widest, highest-yield preset may be frozen only after the yield, visual
distinction, and initial-input-necessity curves are measured.  The code does
not silently choose one.

## Candidate and selector audits

Directed records compile through `future-view-selection.v2`.  The compiler:

- rederives the exact four-member family;
- accepts authentic completed-clear terminal images even when they reveal new
  content;
- binds a separate `future-view-gate.v2` to the frozen match-cell preset;
- applies calibrated block-L1 and non-structural terminal-instance lower
  distinction rules to all six pairs, with no correct-specific bound;
- rejects missing semantic evidence rather than treating it as zero distance.

The pinned v2 authority is currently `pending_joint_calibration`, with both
thresholds and the selected match cell unset.  It is intentionally distinct
from the legacy v1 gate: v1 contains a correct-to-distractor upper bound that
is invalid for an exchangeable four-member family.

Blind selector diagnostics are written outside the benchmark artifact:

```bash
$PY scripts/audit_c1_selectors.py \
  --private-answers data/candidate_pool/<run>/candidate_qa/private/answers.jsonl \
  --asset-root data/candidate_pool/<run> \
  --out data/candidate_pool/<run>/candidate_qa_report/c1_selector_audit.json
```

The report includes answer-position, action-only and initial-only
leave-scene-out baselines, candidate-only visual extrema, and encoding-only
baselines with Wilson intervals.  Shuffled-initial registration remains an
explicit external diagnostic: a high score means the initial RGB may be
redundant, not that the selector leaked the answer.

## Remaining go/no-go gates

No formal C1 questions should be released until all of the following have
passed:

1. 320-s0, at least 20-scene oracle-only budget/preset probe;
2. capacity planning for 400 calibration plus 221 formal families, with at
   least twofold s0 redundancy and scene-disjoint splits;
3. 100 rendered-family candidate-quality pilot;
4. blind-selector and full-input classical baseline runs;
5. 100-question human stop-loss pilot, followed by the frozen calibration
   campaign.

The production gate asset remains pending until those measurements exist.

## Superseded v1 tombstones

The following local artifacts predate the v2 policy/terminal certificate and
are regression-only. They are not registered release inputs, must not be
backfilled into v2, and must not be redirected to a smoke run:

- `data/candidate_pool/v16_abc_preview_20260804_c1_probe`;
- `data/candidate_pool/c1_calibration100_20260808_5259765_dirty`;
- `data/candidate_pool/c1_v5_stage0_20260808_5259765`;
- `data/candidate_pool/c1_v5_stage0_20260808_5259765_p24`;
- `data/candidate_pool/c1_v5_stage0_smoke_20260808_5259765`.

Their only permitted use is reproducing a historical regression. New evidence
must be collected under v2.
