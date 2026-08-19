# Counterfactual-only C1 design

## Objective

EgoConseq has exactly one C1 definition across R2R, B1K, and GS.  Given a
certified clear query program `Q`, the correct choice is the terminal RGB
rendered after `Q`; three wrong choices are terminal RGB renders from distinct,
independently certified counterfactual programs near `Q`.  All terminal renders
use the pose's selected public camera height and FOV.  They may reveal surfaces
or objects absent from the initial observation.

The counterfactual selection remains candidate-only and always reports
`headline_eligible=false`.  No appearance threshold is invented in this
change.

## Supersedes

This design replaces the GS capability limits in the earlier ABC1 collection
plan.  The current GS geometry and semantic authorities expose all methods
needed by A1, A2, A3, B1, B2, and candidate C1.  Static availability is not
treated as certification: the six tasks must still pass the real smoke below.

## Canonical data flow

1. Sample a scene pose and deterministically draw one public height, FOV, and
   body radius.
2. Render the public RGB-D and use its depth evidence to materialize and
   shortlist L1--L6 action programs.
3. Reserve one eligible C1 query and counterfactual-neighbour slots before the
   expensive oracle stage.
4. Certify every query and neighbour with the same depth/full-geometry
   consequence pipeline as ordinary outcomes.
5. Render a minimal terminal RGB for each certified clear outcome with the same
   public camera setting.
6. At QA compilation time, select `Q` plus three action-distinct and
   image-distinct neighbours.  New observations are valid and are not compared
   against the initial visible inventory.
7. Compute the six pairwise `block-l1.v1` certificates over the four terminal
   images and store them in the private selection certificate.  These values
   are diagnostic only: they do not rank choices, reject items, or define an
   appearance threshold.

## Removal boundary

Delete the old strict C1 closure rather than disabling it:

- the `--c1-suffix-candidates` collection mode and all runtime/source-manifest
  branches that support it;
- strict suffix family construction and validation;
- candidate-review publication, reconciliation, completeness, and B1K shard
  aggregation;
- the matched future-view selector, strict initial-visible-space eligibility,
  appearance-gate parsing, configuration, CLI arguments, source authority, and
  publication binding;
- strict-only tests, the active strict Golden dependency, and documentation
  that instructs users to run the old route.

Preserve only shared primitives used by counterfactual C1: terminal RGB
materialization and validation, renderer/source binding, descriptor
construction, terminal image hashes, deterministic option permutation, and
counterfactual eligibility/selection validation.

The QA compiler always calls the counterfactual selector.  It accepts no C1
appearance gate and cannot prefer or fall back to another C1 implementation.

The old suffix sampling contract is not retained as a read-only adapter.  All
records produced with `c1_suffix_candidates=true` or action-sampling policy
`c1_safe_suffix_family_per_pose.v3` are invalid under the new pipeline.  The
known generated run to delete is:

```
data/candidate_pool/step3_smoke_baf56c0
```

Before deletion, scan all `run_meta.json` files again and fail if another
suffix-mode run exists without being listed.  No scientific or authoritative
run is silently reinterpreted.

## C1 quality diagnostics

The strict selector is deleted, but its three useful shortcut measurements are
ported to counterfactual certificates and retained as a renamed
counterfactual-C1 audit:

- `initial_to_option_similarity`: choose using only similarity to the initial
  image;
- `options_only_visual_medoid`: choose from option-image statistics without the
  initial image or action program;
- `motion_only`: choose using only the public action programs.

The audit reads compiled counterfactual items and does not import suffix-family
or candidate-review code.  It reports scene-clustered uncertainty as before.
The pairwise block-L1 values make terminal-view distinguishability measurable
without declaring an eligibility gate.  GS reports the visual-medoid baseline
separately; a value materially above its 25% four-choice chance level is an
explicit renderer-shortcut warning, not a reason to relabel the item.

## Golden replacement

The existing Golden C1 shard and required gate encode the retired matched
route.  Replace them with a source-bound R2R fixture containing one clear query
and at least three certified counterfactual neighbours.  Freeze a new active
Golden manifest with all six task types and no future-view gate input.

Immutable superseded manifests and the bytes they reference remain as audit
history because `check_abc_golden.py` verifies the superseded-transition
chain.  They are not accepted by any production parser, CLI, collector, or QA
compiler.  Only the active manifest is replaced.

## Tests

Deletion tests are written before production removal and must first fail on the
dual-route implementation.  They lock these properties:

- C1 compilation has one selector and no gate parameter;
- no strict suffix/review/gate CLI or source binding is accepted;
- no executable suffix policy or legacy selector remains;
- the active Golden has no future-view gate dependency.

Existing counterfactual behavior tests are regression protection and must stay
green throughout; they are not forced to fail artificially.  They lock these
properties:

- terminal views containing newly observed content remain eligible;
- R2R, B1K, and GS records use the same counterfactual certificate schema;
- query and distractor programs are distinct, all four images are distinct,
  option order is deterministic, and `headline_eligible` is false;
- a failed neighbour is skipped without invalidating the query or pose;
- all six block-L1 pair certificates are deterministic and validator-rebuilt;
- shared terminal RGB validation continues to reject damaged or source-unbound
  assets.

Add a B1K orchestration regression test that deleting candidate review does not
weaken partial-shard acceptance: damaged, incomplete, or source-unbound scene
outputs remain rejected.

Run targeted C1/compiler/source-authority tests, then `compileall`, the full
test suite, and the rebuilt six-task Golden.

## Real smoke acceptance

After static verification, run bounded real smokes from the implementation
commit, not an older canary:

- **GS:** collect and validate the minimum accepted records needed to compile at
  least one QA item for every task A1, A2, A3, B1, B2, and C1.  Static
  capability declarations do not satisfy this check.  After the renderer is
  ready, the first accepted record must appear within two minutes.  Stop after
  six-task coverage or a ten-minute post-ready wall-clock cap.
- **B1K:** print a structured heartbeat within 30 seconds of process start.  It
  must name the current initialization stage; it is allowed to say that
  OmniGibson is still initializing.  Once the backend reports ready, the first
  accepted record must appear within two minutes, and no later accepted-record
  gap may exceed two minutes.  Stop after six-task coverage or a twenty-minute
  post-ready wall-clock cap.

For each backend, "output" means a durably written accepted record, not a log
line, proposal, rendered frame, or rejected pose.  At the cap, compile every
accepted record once.  The smoke passes only when the resulting artifact has at
least one item in A1, A2, A3, B1, B2, and C1 and all source bindings validate.
A heartbeat or a record without six-task QA coverage is not a successful
smoke.

For both datasets, `--gt-as-pred` must replay at 1.0 and the emitted C1 choices
must use the counterfactual certificate with new observations permitted.
Run the three counterfactual C1 shortcut baselines on each smoke artifact and
report their values; for GS, call out the visual-medoid result explicitly.
If a bounded smoke cannot produce the required QA, report the exact funnel
shortfall and the measured time per accepted record.  Treat a first-record or
inter-record delay over two minutes as a performance failure to profile; do not
weaken any scientific gate to force success.

## Non-goals

- Do not change action grids, depth thresholds, collision/contact semantics, or
  A/B eligibility gates.
- Do not retain an executable diagnostic, archived, or compatibility version
  of strict C1.  Immutable Golden transition evidence is data, not an executable
  compatibility path.
- Do not turn block-L1 or shortcut measurements into publication gates in this
  change.
- Do not modify the user's unrelated browser work in the primary worktree.
