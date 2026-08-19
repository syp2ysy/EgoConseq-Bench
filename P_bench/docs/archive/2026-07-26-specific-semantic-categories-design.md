> **Superseded — archived for provenance, not current.**
>
> Written for the Q1--Q10 taxonomy and the interactive review server,
> both of which have been removed. The active contract is the six-task
> A1/A2/A3/B1/B2/C1 ABC benchmark; see `AGENTS.md` and
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`.
> Kept because the reasoning behind still-live thresholds and helpers
> (for example `config.is_specific_semantic_category`) is recorded only here.

# Specific Semantic Categories Design

## Goal

Published QA must refer only to semantic categories that name a recognizable
object or surface class. Source-dataset placeholders such as `misc` and
`unknown` must never appear as a target description, Q3 contact label, or Q3
distractor.

## Category Contract

`pipeline.config` owns one exact, case-insensitive denylist:

```
misc, miscellaneous, unknown, unlabeled, unlabelled,
other, others, object, objects, background, void, none
```

Matching trims surrounding whitespace and compares the complete normalized
label. It does not use substring matching: a concrete source category that
merely contains one of these tokens remains eligible.

Broad but meaningful source classes such as `furniture`, `appliances`, and
`seating` remain eligible. They convey a recognizable category and are not
missing-label sentinels. Their quality can be audited separately without
conflating that judgment with placeholder removal.

## Enforcement Boundaries

1. `pipeline.objects.eligible_target_ids` rejects non-specific categories so
   future records do not select them as targets.
2. benchmark eligibility rejects non-specific target categories in existing
   records, allowing the current continuous records to be recompiled without
   re-collection.
3. Q3 eligibility and distractor construction reject non-specific contact
   categories.
4. artifact validation rejects any public target or Q3 choice that bypasses the
   preceding gates.

All boundaries call the same predicate. The compiler and validator do not
maintain private copies of the denylist.

## Versioning

This is a publication-eligibility correction. It changes neither continuous
physical GT nor the QA response schema, so `conseq.v6`, `egoconseq.qa.v6`,
`ground-disc-visible-v4`, and `visible-ground-disc.v7` remain unchanged.
Existing QA artifacts must be regenerated because their selected item set is
stale.

## Verification

Regression tests establish that:

- `misc`, `unknown`, casing variants, and whitespace variants are rejected;
- concrete labels such as `chair` remain eligible;
- old records carrying a `misc` target compile no target-dependent QA;
- a Q3 outcome attributed to `misc` is ineligible;
- artifact validation rejects leaked non-specific targets and Q3 choices.

After recompilation, audit every public item structurally and assert that no
target category or Q3 semantic choice belongs to the denylist. Then run
`check_benchmark.py` and GT-as-pred evaluation.
