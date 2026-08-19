# ABC1 freeze remediation implementation plan

This plan implements the reviewed ABC1 freeze decision in the existing dirty
working tree.  Existing user changes are inputs, not disposable scratch work.
Each task must stage only the files it owns, use deterministic tests, and leave
scientific gates pending when their external evidence does not exist.

## Global invariants

- Keep `conseq.v11` and `egoconseq.qa.v16-candidate-preview` at the top level.
- Keep Batch1/Batch2/Bug1/Bug2.  Do not reset or overwrite unrelated changes.
- Do not backfill directed C1 v1 data into v2.
- Do not claim C1 publication readiness without held-out confirmation evidence.
- Do not invent human responses, calibration reports, or evidence hashes.
- Use the Habitat Python 3.9 interpreter from `AGENTS.md`.
- Add a failing focused test before every behavioral fix.
- The standard golden checker must pass at every clean freeze commit, except
  during the explicitly bounded one-time A2/B1 refreeze transition.

### Task 1: Complete the C1 structural-instance contract and Bug3 regression

Finish the already-started C1 A+ contract without changing the top-level
record schema.

Implementation requirements:

1. The directed-family policy is `c1_matched_family_per_pose.v2` and the
   terminal semantic metric is `terminal-instance-histogram-l1.v2`.
2. Every rendered terminal checkpoint used by a computed directed C1 family
   carries `structural_instance_ids`: a sorted, unique JSON list of strict
   positive integers.  Derive it from the terminal semantic instance
   histogram and the scene-wide MP3D house category authority, not from the
   initial visible-object list or `objects_entering_view_objects`.
3. Keep `terminal_semantic_pair_certificate(left, right)` as a strict two-
   argument API.  It must exclude the declared structural IDs from the
   non-structural histograms and fail closed on malformed/missing declarations.
4. The local validator checks only JSON shape/canonicality and internal
   certificate consistency.  Add/extend a source-bound validation helper that
   hash-verifies the house/category authority and re-derives the exact terminal
   structural-ID intersection before accepting it.
5. Non-directed v11 records remain exempt.  `c1_oracle_only` Stage-0 output
   must not fabricate a directed pair certificate.
6. Update synthetic fixtures and the three failing tests to declare terminal
   categories explicitly.  Add a rollout/writer regression proving a wall
   visible only at the terminal frame is classified structural and excluded.
7. Run the focused C1/rollout/validation tests and then the full test suite once.

Commit only Task 1 files, using a narrow Conventional Commit subject.

### Task 2: Freeze A2, B1, reason authorities, and action-length coverage

Implement the reviewed A/B contract changes.

1. A2 choices are only original one-based indexes of forward primitives.  An
   A2 item needs at least two forward actions, and its answer must identify a
   forward primitive.  Preserve the original sequence indexes (do not renumber
   the forward subsequence).  Add deterministic construction and validation
   tests.
2. Add a scene-clustered A2 shortcut audit covering constant forward ordinal,
   answer position, action count, and canonical sequence pattern.  Use a
   max-statistic cluster bootstrap/UCB and make the CLI fail closed on malformed
   task data.
3. B1 becomes `b1-metric-choices-rank-balanced.v3`.  Every displayed choice
   must satisfy `value_m + 0.006 >= 0.05`.  The certificate records
   `plausible_lower_bound_m` and `plausibility_tolerance_m`; validation must
   recheck them.  Do not assume a 1.0 m lower bound.
4. Replace free-form QA/report reason strings with a centralized authority and
   reject unknown reasons at the report/validator boundary.
5. Remove the inert `--min-complete-lengths` option and its no-op contract
   field.  Add an effective formal-output coverage check: general A/B results
   cover L1-L6 at the requested keep count; directed C1 v2 is explicitly scoped
   to L3/L4.
6. Add focused tests first, run relevant suites, and then the full suite once.

Commit only Task 2 files.

### Task 3: Close validation levels, gate authority, and selector replay

Make artifact trust boundaries explicit and reproducible.

1. Expose `validate_record_local` and `validate_record_source_bound`.  Local
   validation performs deterministic JSON/integrity checks without loading a
   dataset.  Source-bound validation additionally verifies trusted asset
   digests and re-derives source facts.  Pre-spool collection may use local;
   final compilation, formal checks, golden checks, and registered record
   validation use source-bound.
2. `scripts/check_records.py` requires an explicit
   `--validation-level local|source`; source mode retains explicit run-meta
   authority and must not guess.
3. A future-view gate is accepted only with both path and an independently
   supplied expected SHA.  Do not compute the expected SHA from the same file.
   Formal builds resolve the expected SHA from a committed freeze manifest or
   authoritative run registration.
4. Candidate `source_map` binds gate path, digest, and authority identifier.
   Artifact validation calls `validate_future_view_selection`, reconstructs
   the four directed family members, four action digests, four option PNGs, and
   all six pair certificates.
5. Golden checking verifies required-gate metadata and build-argument
   consistency.
6. Add negative tests for self-certified/mismatched gates and forged selector
   artifacts; run focused and full tests.

Commit only Task 3 files.

### Task 4: Defer the preregistered C1 scientific protocol tooling

**Superseded by the user's fast-cleanup scope on 2026-08-09.**  None of the
Task 4 deliverables (`check_c1_freeze.py`, `abc_shortcut_policy.v1.json`,
`c1_calibration_protocol.v1.json`, or a new 61-scene scheduler) are part of
this completed engineering freeze.  Keep C1 scientific readiness explicitly
pending until these tools and real v2 collection, pilot, and held-out
confirmation exist.

The following is the frozen specification for a later scientific-readiness
phase, not a claim about files delivered in this pass.  Do not generate fake
outcomes.

1. Freeze one configuration: tight preset, atomic family, L3/L4, six cells;
   no preset fallback and no nine-cell search.
2. Before collection, deterministically split the 61 eligible scenes into
   scene-disjoint 20/20/21 selection/confirmation/formal groups and commit a
   manifest SHA.  Validate capacity before launching.
3. Quotas are selection 200 (48 L3/152 L4), confirmation 200 (48/152), and
   formal 221 (52/169).  Minimum search-ready s0 counts are 400/400/442;
   planning attempt caps are 2163/3404/3719.
4. Selection uses only the predeclared common quantile ladder
   Q={0,5,10,15,20}; Q0=(0,0), other ranks use the preregistered nearest-rank
   rule.  Evaluate strictest first.  Require at least 120 survivors, at least
   five per scene, at least 20 L3 and 60 L4, maximum blind excess UCB <= .05,
   and full-input gain LCB >= .05.  Human labels cannot choose thresholds.
5. Pilot membership is fixed before labels: 100 items, five per selection
   scene, 24 L3/76 L4.  Three independent responses per item, at least nine
   qualified raters, 12-item qualification with >=9 correct, max 100 responses
   per rater, and a 90-second timeout counted incorrect.  Report scene-macro
   individual-response accuracy and Fleiss kappa with 100,000 scene-cluster
   bootstrap replicates, seed 20260809.  Both lower bounds must be >=.70 and
   >=.40 respectively.  Pilot only decides stop/continue.
6. Held-out confirmation has 200 fixed items/600 responses and uses identical
   gates without retuning.  Failure has no fallback.  Formal 221 generation is
   enabled only after a passing, hash-bound confirmation report.
7. A freeze manifest binds nine named evidence hashes only when real files
   exist; placeholders are invalid.  Pending/missing evidence must produce a
   non-ready status and a non-zero formal launch.

Commit only Task 4 files and documentation.

### Task 5: Complete A3 source binding and independent source replay

1. Pass the trusted expected semantic-PLY SHA through A3 exact-contact
   validation.  Source-bound validation re-derives contact instance/category
   from the hash-verified semantic PLY/house authority; a self-consistent proof
   alone is insufficient.
2. Add focused regressions proving an arbitrary self-consistent proof digest,
   a substituted semantic PLY, or a wrong category/instance is rejected.
3. Keep the existing exact-face gate unchanged.  Defer the broad independent
   replay runner and exact-face-vs-category production audit; record them as
   non-blocking scientific follow-up rather than expanding this cleanup pass.

Commit only Task 5 files.

### Task 6: Perform the bounded golden transition and repository cleanup

1. Establish audit authority without pretending it predates existing dirty
   changes.  Record the current baseline, then require two clean deterministic
   rebuilds before accepting a new manifest.
2. The user-approved Task 3 external-source-authority requirement adds
   `private/source_map.json` to the transition.  Direct comparison against the
   archived pre-transition manifest proves that report.json and report HTML
   had already been re-frozen earlier and did not change here.  The one-time
   refreeze therefore changes exactly six non-image artifacts:
   benchmark.json, public/items.jsonl, public/manifest.json,
   private/answers.jsonl, private/manifest.json, and private/source_map.json.
   The other 13 files stay byte-identical.
   Diff semantics are limited to A2 choices, B1 protocol/certificate fields,
   and Task 3 source-authority binding; no image, GT, or coverage changes, and
   GT replay remains 1.0 before/after.
3. Supersede directed C1 v1 with tombstones; never backfill it.  A missing
   `docs/runs.json` root is marked missing/archived, not redirected to a
   different smoke run.
4. Remove verified dead wrappers/constants/no-op parameters and the four dead
   test points.  This includes `c_candidate_eligibility`: it had no production
   caller, so restoring it only for A/B/C naming symmetry would recreate dead
   code.  Keep `action_bank_sha256` only in `_RETIRED_SELECTION_FIELDS` so v10
   records fail closed; it is distinct from active
   `materialized_action_bank_sha256`.  Add the intentional-pilot banner to
   the then-active diagnostic compiler.
5. Update public README/BENCHMARK/pipeline dataflow and active protocol docs;
   archive completed/obsolete plans and fix archive links.
6. Preserve useful Stage-0 evidence in a compact tracked report, then remove
   untracked profiler output, Python/pytest caches, and unregistered
   `data/allcases_static`.  Do not delete the 57 test modules wholesale.

Commit cleanup/docs separately from the golden manifest transition.

### Task 7: Final verification and truthful freeze report

Run focused suites, full pytest, golden rebuild/digest check, source-bound
validation for registered golden shards, registry checks, and deterministic
double-build checks.
Fix in-scope regressions with tests and narrow commits.  Report code readiness
separately from scientific readiness.  Scientific readiness remains pending
until new v2 collection, real human pilot, held-out confirmation, and formal
221 evidence all exist and are hash-bound.
