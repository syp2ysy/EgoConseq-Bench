# Oracle v8 GT acceptance

> Historical acceptance note: this document froze the former `conseq.v9` /
> Q1–Q10 contract. The active ABC-only `conseq.v11` contract and Golden are
> registered in `docs/golden/2026-08-07-r2r-abc-v11-24qa.json`.

Status: candidate-code acceptance. Real-backend smoke and a newly collected
candidate remain required before any benchmark result is reportable.

## Breaking contracts

Oracle v8 intentionally rejects every older record and QA artifact:

- record schema: `conseq.v9`;
- physical/observation oracle: `ground-disc-visible-v8`;
- QA schema: `egoconseq.qa.v15`;
- shared pose-pool schema: `egoconseq.pose-pool.v3`;
- prompt contract remains `visible-ground-disc.v14`.

No compatibility reader or label migration exists. The corrected semantics can
change collision validity, contact attribution, Q4 private search evidence, and
Q8 invisible-cause labels. Recompile is insufficient: current records and pose
pools must be recollected under this contract.

## GT corrections

### GS invalid geometry is not collision

GS navmesh snap failures and unsupported-floor failures are proposal or
geometry-validity failures. They now produce an unknown collision state with an
excluded geometry source. They cannot enter Q1/Q2/Q3, radius probes, terminal
probe masks, or publication eligibility as Gaussian contact. Only overlap
certified by Gaussian geometry may use `collision_source=gaussian_contact`.

### Q3 uses a physical contact surface

HM3D/R2R navigation queries expose a configuration-space boundary for the disc
centre, not an obstacle surface. Oracle v8 stores that boundary privately and
extrapolates in the opposite horizontal direction by exactly the body radius.
The validator independently checks both distance and direction. If the
configuration boundary is missing, coincident, or farther than the configured
local reliability bound, the surface point is withheld and Q3 is ineligible.

Full-geometry and depth contact attribution both exclude floor-like support
categories. A rug or floor patch cannot be published as the contacted obstacle.

### Q4 search evidence is private and continuous

The adaptive radius search stores its private safe/collision bracket and
bracket width. The four public candidate radii are selected later. The
validator checks that the continuous critical radius lies inside the private
bracket, that the bracket meets search tolerance and endpoint requirements,
and that public labels and transition rank follow from those facts. Search
endpoints are not required to equal public grid values.

### Q8 invisible cause is operational

At the realized endpoint:

- outside the camera frustum is `out_of_fov`;
- inside the frustum with zero rendered target pixels is `occluded`;
- inside the frustum with nonzero but sub-threshold pixels is `too_small`;
- threshold-passing support is visible and has no invisible cause.

QA construction and artifact validation rederive this cause from private
projection support and the private sensor profile instead of trusting a stored
cause string.

## Independent artifact GT

Each private answer carries non-answer source facts. Artifact validation must
also bind every public action, body, sensor, target and constraint to the
corresponding private rollout. Q0a and Q1--Q10 answers are then recomputed from
that authenticated private bundle. Mutating a stored answer or changing the
public question while retaining a different private rollout must fail.

## Source identity

Semantic surface cache schema `semantic-surface.v3` includes the source file
content SHA-256 in its cache identity. File path, size, or timestamp equality
alone is not accepted. GS semantic attribution no longer fabricates bounding
box corners when no Gaussian surface points exist.

## Required verification

Before promotion from candidate-code acceptance:

1. run the full deterministic test suite with exit code 0;
2. verify immediate predecessor records (`conseq.v8` / oracle v6), QA v14
   artifacts, and pose-pool v2 manifests are rejected;
3. run one HM3D, one R2R, and one GS real-scene smoke;
4. inspect HM3D/R2R centre-to-contact distance and direction, dual-oracle Q3
   agreement, GS geometry-source taxonomy, Q4 private brackets, and all four
   Q8 visibility cases;
5. independently rederive every emitted candidate answer before starting the
   trusted-local All Cases viewer.

The viewer and candidate are not current until all five steps complete.
