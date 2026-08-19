> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# Q6 Strict Distance-Trend Design

## Goal

Q6 must report the direction of the measured ground-space distance change
without a semantic dead zone:

- `final_distance_m < initial_distance_m` means `closer`;
- `final_distance_m > initial_distance_m` means `farther`;
- exact equality means `unchanged`.

The comparison uses the stored deterministic physical distances. No percentage
or absolute-distance tolerance may change the answer label.

## Scope

Both Q6 variants use the same strict rule:

- `surface_proximity` compares the ground-plane gap between the body footprint
  and the target's near-floor support surface;
- `geodesic_progress` compares the navigable path distance to the target region.

The existing evidence tolerances remain evidence gates only. They may withhold
or abstain on a question whose public image cannot certify the distance, but
they may not turn a nonzero physical change into `unchanged`.

## Eligibility And Answers

Remove the `10%` unchanged threshold, the `30%` changed threshold, and the
intermediate grey-zone rejection from Q6 eligibility. Retain the existing
metric-availability and anti-shortcut eligibility checks. For the geodesic
variant, retain the requirement that the strict geodesic and surface trends
agree.

Centralize the strict comparison in one benchmark helper used by eligibility
and answer construction. This prevents the release gate and GT writer from
using different label definitions.

The old relative boundary margin is removed because it no longer represents a
publication boundary. Q6 open answers continue to expose the initial, final,
and signed distance change values.

## Versioning And Artifact Migration

Continuous `conseq.v6` records and `ground-disc-visible-v4` oracle values do not
change. The QA schema shape also remains `egoconseq.qa.v6`.

The prompt/answer contract is bumped from `visible-ground-disc.v6` to
`visible-ground-disc.v7`. Existing v6 QA artifacts are rejected by the normal
artifact contract check and must be regenerated from the current continuous
records.

## Verification

Regression tests must establish:

1. a `-0.5 m` change at a large initial distance is `closer`;
2. a positive nonzero change is `farther`;
3. exact zero is `unchanged`;
4. the former `10%--30%` grey-zone candidate is eligible;
5. eligibility and answer construction agree for both Q6 variants;
6. artifact contract checks reject `visible-ground-disc.v6`;
7. the regenerated artifact contains no Q6 item whose choice disagrees with
   the strict sign of its private numeric GT;
8. GT-as-pred evaluation remains self-consistent.
