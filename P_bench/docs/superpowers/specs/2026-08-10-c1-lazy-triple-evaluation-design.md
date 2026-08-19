# C1 Lazy Feasible-Triple Evaluation Design

**Status:** Revised after runtime-cost review, pending written-spec approval

## Goal

Reduce C1 suffix-candidate depth/coverage/consensus work without changing the
query set, suffix bank, selected family, protocol identity, validator, or any
non-C1 behavior.

The selector keeps the current exact ten-member, three-category suffix bank.
It visits preregistered feasible triples in the same domain-hash order used by
the exhaustive selector and evaluates suffix labels only when a triple needs
them.  For any fixed query and label assignment, the selected triple and family
bytes remain identical to the current implementation.

This remains a candidate-generation protocol.  Option-distinctness thresholds
stay null, C1 remains `headline_eligible=false`, and human review remains
mandatory.

## Scope

The production behavior change is confined to
`pipeline/c1_safe_suffix.py:select_safe_suffix_with_evaluator` and a small
private helper in the same module.  Focused tests change under
`tests/test_pl_c1_safe_suffix.py`.

The following remain exactly unchanged:

- query eligibility: authenticated `natural` actions, L2--L6, ending Forward;
- query domain-hash order and the maximum of three strict-safe queries;
- `c1-safe-suffix-family.v3` and
  `c1_safe_suffix_family_per_pose.v3` protocol identities;
- the ten suffix actions, three category definitions, and turn-sign rule;
- family reconstruction, validation, candidate review, and shortcut audit;
- the pending option-distinctness asset and its digest;
- runtime action-bank materialization and batched full-geometry precheck;
- R2R/B1K simulator adapters and all A/B paths.

## Why the Mixed-Motion Bank Remains Mandatory

The active validator requires these family fields to be exactly `False`:

- `correct_is_unique_min_net_displacement`;
- `correct_is_only_query_heading`;
- `correct_is_only_query_position`;
- `same_position_yaw_ordering_exposed`;
- `motion_shortcut_risk`.

These are hard gates, not human-review-only diagnostics.  The current bank
provides three complementary motion categories:

```text
rotate_only:
  Turn(-45), Turn(+45), Turn(-30), Turn(+30)

translate_and_rotate:
  Turn(-45), Forward(0.5)
  Turn(+45), Forward(0.5)

translate_restore_heading:
  Turn(-45), Forward(0.5), Turn(+45)
  Turn(+45), Forward(0.5), Turn(-45)
  Turn(-30), Forward(0.5), Turn(+30)
  Turn(+30), Forward(0.5), Turn(-30)
```

Every accepted triple contains exactly one member from each category and both
negative and positive initial turns.  A single-primitive-only bank is outside
this design because it necessarily exposes one or more hard structural
shortcuts.

## Current Exhaustive Semantics

For one strict-safe query, the current selector:

1. evaluates all ten suffixes;
2. retains the strict-safe suffixes;
3. enumerates every three-member combination;
4. removes triples lacking one member from each category or both turn signs;
5. computes a domain hash from `frame_id`, query action SHA, and the sorted
   three suffix action SHAs;
6. selects the minimum-hash feasible all-safe triple.

The hash depends only on member identity, not labels.  Therefore the same
result can be found without evaluating all ten labels first.

## Lazy Equivalent Semantics

For the same query, the revised selector:

1. constructs all ten suffix variants without observing labels;
2. enumerates every structurally feasible triple using the unchanged category
   and sign rules;
3. sorts those triples by the unchanged domain hash;
4. visits triples in that fixed order;
5. visits members within a triple in action-SHA order, also label-independent;
6. calls `evaluate_label` only when that action SHA has no cached label;
7. skips the triple as soon as one member is known non-safe;
8. returns the first triple whose three labels are all `safe`;
9. returns shortfall only after every feasible triple is ruled out.

Each suffix label is cached by action SHA and evaluated at most once.  The best
case evaluates three suffixes.  The worst case evaluates all ten.  The query
loop, safe-query counter, backtracking limit, selected-member ordering, family
builder, and returned ten-member `offered` bank remain unchanged.

### Equivalence argument

Let `T` be the set of structurally feasible triples and `S` the subset whose
three members are all safe.  The exhaustive selector returns
`argmin(hash(t) for t in S)`.  The lazy selector visits all `T` in increasing
`hash(t)` order and returns the first member of `S`.  These expressions are
identical.  Cached partial evaluation changes only how `S` membership is
discovered, not `T`, `S`, the ordering, or the returned triple.

The equivalence is exhaustively tested over all `2^10 = 1024` assignments of
safe/non-safe labels to the ten suffixes.

## Runtime Cost Boundary

`precompute_full_geometry_candidates()` remains a complete ten-member batch.
For each body radius it rebuilds the navmesh once, then runs the inexpensive
physical collision precheck for all suffixes.  Splitting that function into
one call per suffix would multiply navmesh rebuilds by the number of evaluated
suffixes and is forbidden by this design.

The savings occur only in the already on-demand downstream call reached by
`evaluate_label`:

- `view_collision_rollout`;
- `corridor_coverage`;
- the remaining per-candidate depth/consensus evidence path;
- associated rejection accounting.

The full-geometry `physical_prechecks` count remains ten per explored query.
`c1_safe_suffix_candidates_offered` also remains ten per explored query.
`c1_safe_suffix_candidates_checked` becomes 3--10 instead of always ten, and
suffix-side rejection counters may decrease because unnecessary suffixes are
never sent through downstream evaluation.  Those report-count reductions are
expected consequences of lazy evaluation, not regressions.  They do not alter
the selected family.

Non-C1 code never calls this selector.  In C1 mode, collection returns before
the downstream A/B selection block, so no A/B result or counter is affected.

## Hard Gates Versus Diagnostics

The following remain hard gates:

- exactly one suffix from each frozen category;
- both left and right initial turns;
- four unique action programs, action groups, outcomes, and terminal images;
- four strict-safe outcomes;
- all five structural-shortcut fields exactly `False`;
- exact proposal, source-outcome, family, and pending-review replay.

Pixel/semantic pair distances, offered/checked counts, cache-hit counts, blind
shortcut audit scores, and wall time remain diagnostic evidence.  No numerical
threshold is added or populated.

## Separate Follow-up: Query-Source Expansion

Removing the current `variant == "natural"` query filter is not part of this
change.  That change enlarges the query set, changes query hash ordering and
family identity, requires a new protocol version, invalidates byte-equivalence
as an acceptance lock, and requires new R2R/B1K collection plus human review.

It will receive a separate design, implementation commit, and acceptance gate
after the lazy selector is verified.  Likewise, suffix tail-match reuse remains
deferred: the current B1K evidence contains 5 matches among 41 unique selected
safe actions (12.2%), all `rotate_only`, which does not justify new provenance
roles in this semantics-preserving change.

## Failure Behavior

Failure semantics remain unchanged.  A query produces shortfall only if no
structurally feasible all-safe triple exists.  Evaluation exceptions continue
to propagate through the existing fail-closed collection boundary.  There is
no fallback, threshold adjustment, or retired-protocol path.

## Verification

Focused deterministic tests must first fail, then pass, for:

1. The ten suffixes and three category definitions remain byte-for-byte
   unchanged.
2. Feasible triples require all three categories and both turn signs.
3. The first all-safe triple stops after exactly three suffix evaluations.
4. A failed member advances to the next feasible triple without reevaluating
   cached members.
5. Exhaustive shortfall evaluates each suffix at most once and no more than ten
   times.
6. Lazy and exhaustive reference selectors return the same selected action
   SHAs for all 1024 label assignments.
7. The rebuilt family from lazy selection is byte-identical to the exhaustive
   family for every successful assignment.
8. Every accepted family has all five structural-shortcut fields exactly
   `False`.
9. A triple missing one category or containing only one turn sign is rejected
   by reconstruction and record validation.
10. Query eligibility, query ordering, family schema, policy, pending gate, and
    returned ten-member `offered` bank remain unchanged.
11. Runtime still performs one batched ten-member full-geometry precompute per
    explored query; it does not precompute one navmesh per evaluated suffix.
12. Non-C1 collection never invokes the selector and the C1 early-return keeps
    A/B selection untouched.
13. Existing C1 candidate fixtures reproduce identical selected triples and
    family objects; only downstream checked/rejection counts may decrease.
14. The R2R explicit-action-file Golden remains byte-identical.

Before handoff, run focused C1 selector/validation/review/audit tests, full
`pytest -q`, and `scripts/check_abc_golden.py`.  Run a source-bound R2R C1
candidate smoke and compare its selected action SHAs/family object against the
pre-change smoke.  Report full-geometry prechecks, suffixes offered, suffixes
checked, terminal renders, and wall time.  B1K query-source/yield work belongs
to the separate follow-up and is not claimed by this optimization commit.
