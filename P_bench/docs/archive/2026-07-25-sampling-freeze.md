> **Superseded — archived for provenance, not current.**
>
> This is a Q1--Q10 era freeze contract. The active taxonomy is the six
> ABC tasks A1/A2/A3/B1/B2/C1; the normative upstream is
> `docs/superpowers/specs/2026-08-03-abc-first-order-consequence-freeze-design.md`
> and the implementation plan is
> `docs/superpowers/plans/2026-08-04-r2r-abc-candidate-pipeline.md`.
> Entry points named below (serve_benchmark.py, benchmark_viewer.html,
> serve_viz.html, the Q1--Q10 artifact stack) no longer exist. Do not
> implement anything from this file.

# EgoConseq-Bench — Publication-Contract Freeze (2026-07-25, rev 9)

**Status:** design frozen; P0-16, P0-13 and P0-4 implemented. Remaining P0 items
still block formal collection.
**Evidence:** [`docs/archive/2026-07-25-sampling-audit.md`](./archive/2026-07-25-sampling-audit.md) —
findings F1–F14, measurements, and the anti-regression correction table.
**Supersedes:** the sampling behaviour of `train_1to1to1_v1/v2/v3_20260724`.
**Blocks:** any new *formal* collection root until every P0 is in effect.

There are **sixteen P0 items**. Each changes what a published record *means*, so
all must be in effect before a formal root is opened. A **pilot** root may be
opened earlier under the conditions in §6.

| Category | Items |
|---|---|
| Sampling | P0-1, P0-2, P0-3, P0-5, P0-6 |
| GT / input contract | P0-4, P0-10, P0-13 |
| Generation & scoring | P0-7, P0-9, P0-11, P0-12, P0-14, P0-15 |
| Cross-cutting | P0-16 (schema version bump) |
| Deferred to experiment | P0-8 |

---

## 1. Core construct (frozen — do not renegotiate downstream)

> Within calibrated monocular **visually supported space**, can a VLM bind its own
> **body footprint**, a **metric action program**, and the **3D scene structure**
> to predict the physical and future-state consequences of executing that action,
> and abstain when the image does not support the prediction?

| Requirement | Measured primarily by | Must not be answerable from |
|---|---|---|
| R1 — understands **body size** | Q4; radius conditioning in Q1–Q3 | the printed radius list alone |
| R2 — understands **action consequence** | Q1/Q2/Q3, Q6/Q7/Q8, Q9/Q10 | the action program text alone |
| R3 — respects **visible space** | evidence protocol / abstention | family, backend, or action identity |

**Reporting axes.** Capability scores over
`physical / body-counterfactual / relation / view / maneuverability`.
L0–L4 is a *reasoning-stage* label only — not a validated difficulty ordering.

---

## 1b. Design requirements already satisfied (do not re-litigate)

Recorded so these are not later raised as defects.

**Distance-adaptive scoring tolerance.** Monocular metric perception degrades with
range, so tolerances must widen with distance. Already implemented in
`pipeline/tolerance.py`:

```
tolerance = max(EVAL_METRIC_FLOOR_M, EVAL_METRIC_RATIO * |reference|)
          = max(0.05 m, 0.25 * |reference|)
```

| true distance | allowed error |
|---|---|
| 0.1 m | ±0.05 m (floor protects near targets) |
| 1 m | ±0.25 m |
| 3 m | ±0.75 m |
| 5 m | ±1.25 m |

The ±25% ratio aligns with three existing standards: monocular-depth δ1 = 1.25,
SpatialRGPT-Bench ±25%, and VSI-Bench MRA (distance answers additionally average
over ten thresholds 0.50–0.95, itself a relative-error measure).

Correctly **not** distance-scaled, and to stay that way:
`EVAL_BEARING_TOL_DEG = 30.0` (angular error does not grow with range) and the
normalised image-centre tolerance 0.05. Area uses relative error with a 20-pixel
small-object floor.

---

## 2. P0 items

### P0-1 — Move the 50/50 label balance from *per pose* to *dataset level*

**This does not abandon label balance.** A ~50/50 safe:collision split is required
— without it the majority baseline scores high and the task is trivial. What is
removed is the *mechanism*: forcing exactly 3 safe + 3 collision **inside every
pose**.

Why the per-pose version backfires:

1. **It forces the sampler to cheat.** Whether a direction collides depends on how
   open that direction is. To hit exactly 3+3 at *every* pose, the cheapest
   strategy is "short actions = safe, long actions = collision". Measured: collision
   actions run **+3.41 m** longer than safe ones within the same pose, **246/246
   poses, zero exceptions** → distance alone predicts the label (F2).
2. **It makes the group rank-orderable.** Because "exactly 3+3" is published and
   the group id is public, sorting a group's 6 items by distance and calling the
   longest 3 collisions scores **0.9838** (F1).

Frozen replacement — balance is enforced at two levels, both stronger than 3+3:

**(a) Per-pose composition follows the scene, unforced.**

```
pose A (open):     5 safe + 1 collision
pose B (cluttered): 2 safe + 4 collision
pose C:             3 safe + 3 collision
```

Global split stays ~50/50, but the per-group count is no longer predictable, which
kills the rank attack outright.

**(b) Matched action pairs (P0-2) are the real fix.** Pairs that agree on distance,
segment count and turn profile but differ in outcome remove the *correlation*
between action features and the label — strictly stronger than balancing counts.

The global 50/50 split is then produced by the artifact compiler as an **outcome**
of sampling, never as a constraint imposed on the collector.

Files: `scripts/collect.py:495` `_select_formal_action_groups`; assertion at
`pipeline/validate.py:218`.

### P0-2 — Matched action pairs **without a new fixed ratio**
Directed search for safe/collision pairs agreeing on: total forward distance
(same value or bucket), primitive count, total **and** net turn angle,
**turn-direction sequence**, and **per-segment distance profile**, with identical
radius/FOV/height. Matching on aggregate distance alone leaves the segment pattern
recoverable.

Anti-requirement — matched pairs must not become the new invariant:
variable pairs per pose, variable unmatched actions per pose, varying total and
safe counts, `pair_id` in **private** GT only, global label balance at compile time.

### P0-3 — Q4: critical-radius stratification and input-contract repair

**(a) Arithmetic.** With search range `(0.10, 0.50)` and
`BENCH_Q4_FLIP_RATIO = 1.5` enforcing `first_collision >= 1.5 * largest_safe`, a
safe endpoint cannot exceed `0.333 m`. "Cover the full range", "balance every bucket",
and "keep 1.5×" are jointly unsatisfiable. Resolution:
- a **scored overlap band** — pilot value **[0.15, 0.32] m** — inside which radius
  buckets must be balanced and on which `radius_options_only` is gated;
- radii above the band appear only as collision endpoints, excluded from
  radius-label balancing;
- direct-search so `r*` covers the band;
- balance the position of the largest safe radius among printed options **to the
  extent the publication grid permits** — see (c): under a fixed 4-value grid plus
  the 1.5× rule only 2 of 4 positions are reachable, so position balance is a
  constraint on grid selection, not something achievable on an arbitrary grid;
- `BENCH_Q4_FLIP_RATIO = 1.5` is a **pilot parameter**, revisited with OQ-2.

**(b) Remove the singular body radius from Q4 — do not resample it.**
`build_radius_counterfactual_pair` currently uses `radii[0]` as both the singular
`body_radius_m` and the smallest option. Injecting an *independently sampled*
scalar instead would be a nuisance variable: Q4 asks for the largest safe radius
among several hypothetical bodies, so there is no question-relevant "current
robot radius". Frozen public input:

```
Q4 public input:
  candidate_body_radii_m: [...]   # values set by the publication grid, see (c)
  actions: [...]
  # no singular body_radius_m
```

A singular radius is only meaningful if Q4 is reformulated as "you are r0; what
happens at r1". Consequently the P0-11 battery splits: `body_radius_only` applies
to Q1–Q3; Q4 is attacked by `radius_options_only`.

**(c) Two grids, and the publication grid is NOT frozen yet.**
A single fixed grid `{0.15, 0.20, 0.25, 0.30}` combined with the 1.5× rule is
infeasible — enumerated:

| largest_safe | required collision ≥ | available in grid |
|---|---|---|
| 0.15 | 0.225 | 0.25, 0.30 |
| 0.20 | 0.300 | 0.30 (unique) |
| 0.25 | 0.375 | **none** |
| 0.30 | 0.450 | **none** |

Only 2 of 4 positions are reachable, so "balance the position of the largest safe
radius" cannot hold. Frozen resolution — separate the grids:

- **Collection grid:** dense/continuous GT over 0.10–0.50 m, stored with the
  clearance stratification of (d). Collection is not constrained by the printed
  grid.
- **Publication grid:** **deliberately left open until OQ-2** fixes the margin and
  `BENCH_Q4_FLIP_RATIO`. Freezing it now would re-create the conflict above.
- **Invariant that does hold regardless of grid:** every formal Q4 item must
  contain **at least one safe and at least one collision radius among the printed
  options**.
- `radius_options_only` (P0-11) is gated on the *option set composition*, not only
  on a single threshold, so a grid that leaks through its composition also fails.

**(d) Stratified clearance storage (required for OQ-2).** Do not stop at the first
endpoint satisfying the 0.10 m floor; store candidates stratified by clearance
band, or the 0.30 m subset of the human pilot is empty and OQ-2 is unrunnable.

### P0-4 — Effective camera height, with an explicit coordinate convention

**`frame.floor_y` is an estimate**, fitted from the visible depth cloud. Name and
document it as *quality-gated visible floor estimate*; never call it the true floor.

**Convention A is frozen** (minimum change; an earlier draft listed two options
without choosing, which is not an implementable contract):

```
camera_y      = nominal sensor offset          # unchanged, internal
ground points = on the canonical fitted floor plane
public height = signed normal distance(camera centre, floor plane)
```

Mixing `point_y = floor_y` with `camera_y = effective_height` is the one
combination that is wrong. **The nominal sensor offset is retained internally** for
true camera extrinsics (e.g. `pipeline/rollout.py:reproject_initial_view`) and in
private provenance. The rule is therefore:

> All **published** scale, near-field and ground-entry computations use the
> effective height; internal camera extrinsics and same-frame projection continue
> to use the nominal sensor offset under the chosen convention.

If the floor may tilt, the rigorous quantity is the **normal distance from the
camera centre to the fitted plane**.

**One canonical floor plane per intervention group, with a stated provenance.**
The plane is fitted deterministically **once per physical pose** and shared by all
sensor siblings. Per-FOV depth fits are used **only** for quality and consistency
checking and never alter GT — otherwise the sensor profile would change geometric
GT through `floor_y`, violating the core invariant.

Floor gate metrics: minimum support point count, plane-fit residual, maximum tilt,
and cross-sibling consistency of the effective height. **If full tilted-plane
support is not implemented, gate on tilt and keep only near-horizontal floors** —
do not silently apply an axis-aligned formula to a tilted plane.

---

#### P0-4 decisions, frozen after the pre-implementation audit

The section above states the contract but not enough of it to implement. Three
decisions are now frozen. The audit that produced them is recorded first, since
the magnitudes are the reason the decisions went the way they did.

**Measured on the frozen v3 pool (248 records, 248 distinct poses).** The public
`camera_height_m` is the raw nominal offset today:

```
published camera_height_m       [0.5, 1.0, 1.5]        nominal offset only
floor_y                         [-0.225, +0.275]       11 distinct values
effective (nominal - floor_y)   [0.525, 1.725]
error from ignoring floor_y     up to 0.275 m = 55% of the smallest published height
floor_y quantum                 0.05 m (histogram bin width)
scenes whose floor_y varies across poses   48/117, max within-scene spread 0.500 m
```

This is not a rounding correction. `certified_near_field_m = h / tan(vfov/2)`
propagates the height error linearly into the near-field blind strip and from
there into evidence qualification.

**What `floor_y` actually feeds** (all 25 consumers traced; stated precisely so
the fix is not over-scoped):

| Consumer | Affected | Nature |
|---|---|---|
| full-geometry oracle (navmesh) | **no** | physical GT is independent of `floor_y` |
| depth oracle via `frame.vf` (`rollout.py:770`) | yes | evidence — permitted to vary with the sensor |
| `eligible_target_ids` → `record["targets"]` | yes | **decides which questions exist** |
| `quality.visible_floor_ratio` → frame acceptance | yes | **decides whether the frame is collected** |

The last two are the reason this is a P0. A sensor profile change alters
`floor_y`, which alters the question set and the acceptance decision — a sampling
coupling, not the permitted evidence variation.

##### D1 — A deterministic robust plane fit, with no silent scalar fallback

`perception.estimate_floor_height` is a 0.05 m histogram mode over a
`[-0.25, 0.30]` window that **returns 0.0 when fewer than 200 points fall in the
window**. It cannot support a normal, a tilt, a residual or a signed distance,
and its fallback is not an estimate at all — it is an unflagged default that
looks like a measurement.

Frozen: fit a near-horizontal plane by deterministic robust regression.

```
y = a*x + b*z + c
n = normalize([-a, 1, -b])          # n_y > 0 by construction
plane: n · p + d = 0
```

Seed the candidate floor band with the existing histogram, then fit by trimmed
least squares / IRLS or fixed-seed RANSAC. The estimate persists at least:

```
normal_local, offset_m, support_count, inlier_count, inlier_ratio,
residual_rmse_m, residual_p95_m, tilt_deg, support_extent_xz_m,
status, rejection_reasons
```

**Insufficient support, excessive residual, too small a support extent, or too
much tilt rejects the pose/group. Falling back to 0.0 is forbidden.** The physics
model is still a horizontal SE(2) disc, so this version does **not** claim slope
support: the normal exists for the quality gate and for exact calibration, and
formal data accepts near-horizontal floors only.

##### D2 — Canonical plane from a fixed reference profile; navmesh is diagnostic only

Navmesh height was considered as a sensor-independent floor and is **rejected**:

1. Poses are drawn from `pf.get_random_navigable_point()` (`sim.py:163`), so
   `snap_point(position)` returns a near-zero offset. It would degenerate into
   publishing the nominal height again and would not fix the measured visible-floor
   deviation.
2. The navmesh is a *traversability proxy*, not the visible support surface in the
   rendered image. Using it silently redefines the public quantity from "camera
   height above the visible floor" to "camera height above the navigation surface".
3. The backends disagree about what a navmesh *is*. GS loads a pre-baked
   `scene.navmesh` (`gs_sim.py:56-58`) and cannot rebake; HM3D rebakes per body
   radius (`sim.py:289`). Calibrating against it would bake that asymmetry into
   published scale. (An earlier audit note claiming GS has no pathfinder was
   wrong — it has one, which makes the asymmetry worse, not better.)

Frozen canonical calibration profile:

```
nominal height = config.CAMERA_HEIGHT_M   = 1.5 m
HFOV           = config.HFOV_DEG          = 79°
VFOV           = config.vfov_for_hfov(79) ≈ 63.45°
```

This is already the default reference profile (`config.reference_camera_height`
selects the height nearest it, order-invariantly) and the one pose sampling uses.
Each physical pose is rendered **once** at this profile and the canonical plane is
fitted from it — **including when that profile is not itself published**, as a
private calibration render.

All sensor siblings share that one plane. A sibling's own depth may still be
fitted, for consistency checking only:

```
sibling estimate agrees      -> accept
sibling estimate disagrees   -> reject the sibling, or reject the group atomically
sibling estimate never replaces the canonical plane
```

A union of sibling clouds is **not** used: adding or removing one FOV/height
profile would change the calibration of every sample in the group.

Navmesh height may be persisted as a private diagnostic
(`navmesh_plane_delta_m`); it must not determine the public height.

##### Convention A, stated precisely enough to implement

The coordinate frame is unchanged:

```
camera centre   c = (0, nominal_sensor_offset, 0)
canonical plane n · p + d = 0,  n_y > 0
public height     = n · c + d
```

which degenerates, for a horizontal floor, to `nominal_sensor_offset - floor_y`.

Corridor and ground samples may no longer be hard-coded at `(x, 0, z)`. They lie
on the canonical plane:

```
y(x, z) = -(n_x*x + n_z*z + d) / n_y
```

and are then projected with the **nominal** offset, which is the convention
`perception.project_ground` already implements. The two quantities are named
apart so a call site cannot silently use the wrong one:

```
nominal_camera_offset_m               # internal extrinsics, private provenance
camera_height_above_visible_floor_m   # the public calibrated quantity
```

##### D3 — No final 5 cm quantization

The histogram survives **only** as an initializer for the candidate floor band.
Final plane coefficients and the height come from the continuous fit; persisting a
residual does not repair a quantized centre. Frozen:

- the private record stores the full float plane and every fit statistic;
- the **public** height is quantized to `0.01 m`, not `0.05 m`;
- public derived quantities (`certified_near_field_m`, ground entry) are
  recomputed **from that same published height**, so a user can reproduce them;
- viewer display precision is independent of stored precision.

##### Scope — every floor-relative consumer, not the six call sites

The list in the section above is the *published-scale* subset. The canonical plane
must reach all of:

- obstacle band and voxel field (`frame.py:122`, `perception.obstacle_mask`)
- visible-floor quality gate (`frame.py:129`, `collect.py:151`)
- object support and target eligibility (`objects.py:52,60`, `record.py:22`)
- corridor ground projection (`rollout.py:816-828`)
- near-field / ground entry (`rollout.py:784,795`)
- future / checkpoint frames
- record, manifest, prompt, viewer, validator

`frame.py:118` estimates `floor_y` independently per rendered image and **must**
instead receive the canonical plane; otherwise FOV keeps changing the target set
and the acceptance decision through the floor estimate. Future frames inherit or
transform the same canonical world plane rather than re-deciding the physical
reference surface.

##### Implementation order

```
FloorPlaneEstimate + pure-NumPy tests
  → canonical calibration at the fixed reference profile
  → Frame / record schema
  → every floor-relative consumer
  → public input and near-field
  → sibling / group validator
  → small Habitat + GS pilot
```

##### Implementation decisions, frozen before stage 0

**Rename, do not alias.** `sensor.camera_height_m` changes meaning, so the name
goes with it: `nominal_camera_offset_m` internally,
`camera_height_above_visible_floor_m` publicly. No compatibility alias — v6 is
already a breaking contract, and a hard break makes every stale read fail loudly
instead of silently returning the wrong quantity. **The nominal offset may appear
only in the private record and provenance; never in a public item or manifest.**

**The canonical plane is stored pose-local, not world.** `perception.to_agent_ground`
already maps each sibling's cloud into one agent-root frame using that sibling's
own nominal offset, so the same pose yields the same pose-local plane at every
height and FOV — there is no second height conversion to perform. A world plane
would import scene-coordinate magnitudes, a yaw transform and their float error
for no benefit. Future frames transform the plane by pose rather than persisting a
world copy.

```
coordinate_frame = "agent_pose_local"
camera_centre    = (0, nominal_camera_offset_m, 0)
effective_height = n · camera_centre + d
```

**The reference profile is a fixed logical calibration with a cached render.**
Every pose requests the 1.5 m / 79° calibration observation regardless of which
profiles are published; when that profile is also a public sibling the same
`RenderObservation` is reused rather than re-rendered. `sim.sample_pose` already
renders it and computes `estimate_floor_height` for the floor gate
(`sim.py:171-181`) and then discards the observation — it should return a
`PoseCalibration` carrying the observation and the plane. Note that `rng.choice`
is consumed *before* the render and the render consumes no RNG, so reusing the
observation does not change sampling. If a render ever did consume pose RNG, that
is a state coupling to fix, not something to paper over with a second render.

##### Stage 0 shape

`FloorPlaneEstimate` represents a **successful fit only**. A rejected fit has no
legal `normal_local`, `offset_m` or `y_at()`, and modelling one invites NaN or a
pseudo-zero being persisted as if it were geometry:

```
FloorPlaneEstimate   normal_local, offset_m, height_above(), y_at()
FloorPlaneFitResult  estimate | None, rejection_reasons, and every diagnostic
                     that is computable even when the fit fails
```

All metrics live on the result, so there is exactly one place to read them and no
pair of fields to drift apart.

- `support_extent_xz_m` alone is insufficient — a long, one-cell-wide band scores
  a large extent. Persist `support_extent_x_m` and `support_extent_z_m`
  separately, plus an occupancy-grid cell count and area.
- Bit-identical results under row permutation require a deterministic
  lexicographic sort of the candidate points first; otherwise the float reduction
  order changes with the input order.
- The 0.05 m histogram produces the candidate band only. Final coefficients come
  from the continuous fit.
- Every rejection returns an explicit reason. No 0.0 fallback, and no NaN or
  infinity may reach JSON.

Gate thresholds committed with stage 0 are **provisional**: they are set for real
on the 10–20 scene/source pilot, as §7 requires. A 3–5 scene run is a backend
smoke test, not a threshold study.

##### Per-frame fields, and what stays private

```
canonical_floor_plane                # shared by every sibling      private
profile_floor_plane_check            # this profile's own fit, QA only  private
nominal_camera_offset_m              #                              private
camera_height_above_visible_floor_m  #                              PUBLIC
```

The canonical plane's normal, offset and residuals **do not enter public QA** —
publishing them would leak scene geometry beyond the calibrated height.

##### Sibling rules for the validator

Effective height is *not* constant across all siblings, and asserting that it is
would be wrong:

| Sibling pair | Effective height |
|---|---|
| same nominal height, different FOV | **must be equal** |
| same FOV, different nominal height | **must differ** |

and the difference is exactly

```
h_eff_2 - h_eff_1 = n_y * (nominal_offset_2 - nominal_offset_1)
```

Every sibling's canonical plane must be bit-identical. A profile's own fit may
only *verify* the canonical plane; it may never replace it.

##### Stages 2–4 as implemented — the plane replaces the estimate end to end

`perception.estimate_floor_height` is **deleted**, not deprecated. While it
existed any consumer could reach for a floor of its own; a test now asserts the
attribute is gone.

`Frame.floor_y: float` is replaced by `Frame.floor_plane: FloorPlaneEstimate`,
supplied by the caller and mandatory. `build_frame` raises rather than defaulting.
`SensorProfile.camera_height_m` is renamed `nominal_camera_offset_m` with no
alias, so every stale read fails loudly.

**One height convention, everywhere.** Height above the floor is the signed
normal distance `n · p + d` (`FloorPlaneEstimate.height_above_points`), the same
quantity that defines the published height. The vertical drop `y − y_at(x, z)`
is a *different* number on a tilted floor, and mixing the two would let the
obstacle band and the published height disagree. `y_at` survives for a distinct
question — *where the surface is*, for reprojecting a ground sample — and the two
methods are documented against each other.

Two heights are now distinguished at every call site:

| Quantity | Used for | Example |
|---|---|---|
| `nominal_camera_offset_m` | camera-relative geometry | `to_agent_ground`, `project_ground`, `reproject_initial_view` |
| `camera_height_above_visible_floor_m` | floor-relative geometry | ground-entry ray, `certified_near_field_*`, visible-floor gate |

`corridor_coverage_details` needed both, and previously used the nominal offset
for both. Its ground samples no longer reproject at a hard-coded `y = 0`; they
lie on the canonical plane at `y_at(px, pz)`.

**Future frames transform, they do not re-fit.** A checkpoint inherits the base
pose's plane through `FloorPlaneEstimate.transformed`, composing the two
`world_from_local` maps. `R` is symmetric and involutive, so `n_y` — and
therefore the tilt — is preserved exactly, and the round trip is the identity.
Measured residual for a point on the base plane, re-expressed in the checkpoint
frame: `2.2e-16`. The calibrated height then changes by exactly
`n_world · (p_checkpoint − p_base)`, which is the floor's slope along the
displacement: planar motion holds the agent root at one world `y`, so a
checkpoint down a ramp genuinely is further from the floor. Re-fitting would
have let the same action be judged against two different surfaces.

**Record schema.** `floor_y` is replaced by `floor_calibration` — the whole
`FloorPlaneFitResult`, plane plus every diagnostic the pilot must report — and
`camera_height_above_visible_floor_m` at full precision. `build_record` requires
the calibration and rejects one whose `estimate` is not the plane its frame was
built with, so a record cannot be re-derived against a surface it never used.

**Public contract.** `model_input.camera_height_m` becomes
`camera_height_above_visible_floor_m`, rounded to `PUBLIC_HEIGHT_DECIMALS = 2`.
`certified_near_field_m` is recomputed **from that rounded value**, not from the
private one, so a reader holding only the public item reproduces it exactly. The
nominal offset and the plane never appear in a public item.

**Consumers found by sweeping, not by listing.** Two `.html` viewers read the
height (`benchmark_viewer.html` the public one, `serve_viz.html` the private
sensor dict) and `near_field_report.py` keys its report by it. Enumerating call
sites has now undercounted four separate times on this benchmark; the working
rule is to grep every file type for the field name.

**And a fifth: a green suite is not coverage.** Making `floor_plane` a mandatory
keyword-only argument left three *production* `build_frame` call sites broken —
the collector's sibling loop and both viewer paths — through a fully passing
suite, because nothing executes them without Habitat. Formal collection would
have crashed on its first pose.

The standing guard is structural, not behavioural:
`test_every_production_call_site_supplies_the_canonical_floor` parses every file
under `scripts/` and `pipeline/` and asserts each `build_frame` call passes
`floor_plane` and each `build_record` call passes `floor_calibration`. It does
not need the branch to be runnable, which is exactly the point. **Any future
mandatory argument on a Habitat-only path needs the same treatment.**

Rebuilding a frame from a stored record goes through `record.require_floor_plane`
— the one place a persisted record becomes a floor, shared with the validator.
The viewer may not re-fit (that is the per-image estimate returning through the
viewer) and may not fall back to a level floor.

Mutation-checked: removing the checkpoint transform fails 4 tests, publishing the
mount offset instead of the calibrated height fails 4.

##### Stage 5 as implemented — the validator

Per record: the stored `floor_calibration` must rebuild through
`FloorPlaneEstimate.from_json` with no rejection reasons, and
`camera_height_above_visible_floor_m` must equal `n · c + d` recomputed from
**that record's own plane** and its own nominal offset. Nothing falls back to a
level floor at the origin; a record whose calibration cannot be rebuilt has no
floor, and every quantity derived from one is unverifiable.

Per group: the plane must be **bit-identical** across siblings (exact float
equality, since a per-profile fit may only verify the canonical plane), and the
published heights must satisfy the relation above. The baseline is the first
sibling with a *usable* calibration, not `siblings[0]` — anchoring on the first
record would let one corrupt calibration silently switch off the whole group's
floor checks.

Tolerance is `_FLOOR = 1e-9`, not `_POS = 1e-3`: these are exact identities
between numbers already in the record, so the tolerance is float rounding, not
geometry. At `_POS` a 1 mm publication error would pass unnoticed.

The public leak guard gains `nominal_camera_offset_m`, `floor_calibration`,
`floor_plane`, `floor_y` and `camera_height_m`. The last is the pre-v6 name whose
*meaning* changed, so a stale writer fails instead of publishing the mount offset
under the old key.

##### Stage 6 — the threshold pilot (`scripts/floor_plane_pilot.py`)

**Completed 2026-07-25** on clean revision
`57ece20ee5df086cc17eb135d1f6fa2acfa53aa5`. Each backend covered 15 scenes
with 40 pose indices per scene; all three reports were eligible for threshold
analysis, ended with `threshold_freeze_status="supported"`, and exited
successfully.

| Backend | Poses | Floor fitted | Fit rate | Visible-floor pass among fitted | Coverage-gate rejection |
|---|---:|---:|---:|---:|---:|
| HM3D | 600 | 304 | 50.7% | 59.2% | 11.8% |
| R2R | 600 | 360 | 60.0% | 74.7% | 13.8% |
| GS | 600 | 170 | 28.3% | 100.0% | 50.0% |

The freeze retains the conservative values that already passed both synthetic
contamination sweeps with zero wrong fits:

| Gate | Frozen value |
|---|---:|
| `FLOOR_MIN_SUPPORT` | 200 |
| `FLOOR_MIN_SUPPORT_EXTENT_M` | 0.60 m |
| `FLOOR_MIN_SUPPORT_CELLS` | 40 |
| `FLOOR_MIN_INLIER_RATIO` | 0.60 |
| `FLOOR_MIN_INLIER_COVERAGE_RATIO` | 0.85 |
| `FLOOR_MAX_INLIER_RMSE_M` | 0.010 m |
| `FLOOR_MAX_TILT_DEG` | 5.0 degrees |

`supported` means the scene-clustered pilot estimated some yield changes
precisely; it does not automatically authorize a threshold change. The real
pilot has no wrong-fit labels, so yield evidence alone cannot justify relaxing
a correctness gate. In particular, lowering the coverage ratio would require
rerunning both synthetic contamination sweeps and retaining zero wrong fits.
GS's lower yield is accepted for the preview and remains a capacity input for
later formal sampling. Raw reports are untracked runtime data under
`data/floor_pilot/threshold_15x40_57ece20/`.

```
python scripts/floor_plane_pilot.py --backend {hm3d,r2r,gs} \
    --max-scenes 15 --poses-per-scene 40 \
    --out data/floor_pilot/<backend>.json --rows-out data/floor_pilot/<backend>.jsonl
```

Two design points, both about not conditioning on the answer:

* It does **not** sample through `sim.sample_pose`, which already rejects a pose
  whose floor will not fit — and it does **not** draw its own poses either. Both
  would answer a different question. `sample_pose` is split at the floor-fit
  boundary into `sim.pose_observations` (every gate before it) and `sample_pose`
  (the floor gate, consuming that generator); the pilot consumes the same
  generator, so the population it measures is the collector's. Each backend's
  gates live in one `POSE_SAMPLING` dict per session class — HM3D and GS
  genuinely differ on the depth-hole ratio, clearance margin, retry budget and
  visible-floor floor, which is why neither may be spelled out at a call site.
  `prepare_pose_sampling` (navmesh rebake at the largest primary radius +
  deterministic pathfinder seed) and `pose_has_publication_clearance` moved to
  `pipeline/pose_calibration.py` so the collector and the pilot share one copy.
* The sweep replays hypothesis **selection** over every recorded band, not just
  re-judging whichever band won under today's constants, so a relaxed threshold
  that would have let a different band win is scored correctly. Only the final
  tie-break on plane coefficients is not replayed; it decides exact ties in all
  four preceding keys.

`fit_floor_plane_with_attempts` exposes the per-band attempts, and
`FloorPlaneFitResult.plane_y_at_origin_m` records where a withheld plane sat —
a scalar, deliberately, so the pilot can replay the ambiguity check without
being able to reach a plane the gates refused.

**The three pre-fit gates cannot be swept offline.** `FLOOR_MIN_SUPPORT`,
`FLOOR_MIN_SUPPORT_EXTENT_M` and `FLOOR_MIN_SUPPORT_CELLS` run before a plane
exists, so the poses they rejected have no fit statistics to re-judge. Their
sweep reports `reached_fit`, not `accepted`, and is flagged
`"replayable": false` — a flat accept curve there would read as "loosening this
buys nothing", which is not what the data says. Moving one needs the pilot re-run.

The report answers each item the pilot was asked for: `support_cell_count`,
`inlier_cell_count`, their ratio, quantiles stratified by backend and by scene,
and the `floor_plane_explains_too_little_ground` rejection rate with the coverage
ratio distributions on both sides of the gate.

**What the pilot measures, stated so it cannot be misread.** It takes the *first*
candidate per pose index that reaches the floor fit and stops. Taking the first is
what keeps the sample unconditioned: continuing past a rejected floor — which is
what the collector does — would make every recorded pose one whose floor already
passed, and every gate rejection rate would read zero.

The price is that a fitted floor is **not** an accepted pose. The collector also
gates on `low_visible_floor_ratio` and then moves to the next candidate. So the
report says `floor_fit_accept_rate`, never `accept_rate`, and reports the
visible-floor ratio beside it under `visible_floor_gate` with
`measured_not_applied: true`. `pose_calibration.visible_floor_ratio` is shared, so
the gate and the report cannot measure different quantities.

**Population.** One pose per pose index, first candidate reaching the fit,
pre-floor gates the collector's own, and by default **no cross-pose diversity
exclusions** — so this is the *first-round* candidate distribution. The collector
additionally rejects poses too close to ones it already accepted
(`pose_is_diverse`, now shared), so a later coverage round sees a different
distribution; `--pose-exclusions` takes the collector's own payload to measure
that instead. The report carries `population` and derives
`pose_exclusions_applied` from the rows rather than from a flag, so it cannot
claim a population its own measurements contradict.

**Diversity thresholds have one source.** `config.POSE_DIVERSITY_POSITION_M = 1.5`
and `config.POSE_DIVERSITY_YAW_DEG = 45.0`; `scripts/collect.py` and the pilot both
default to them and both expose `--min-pose-position-m` / `--min-pose-yaw-deg`,
because the collector permits overriding them and a pilot pinned to a constant
cannot reproduce a run that did. While the pilot held its own 1.0 m / 30°, feeding
both the same exclusion payload still produced different populations — a candidate
1.2 m from an excluded pose was kept by one and dropped by the other.

**Admissibility is a conjunction; freezing a gate is a separate question.** Two
fields, because one was doing both jobs badly.

`eligible_for_threshold_analysis` requires all three of: the tree is clean, the
tree did **not** move during the run (sampled again at the end — a run spanning a
commit describes two code states and can certify neither), and the run covered
`MIN_PILOT_SCENES = 10` scenes **each at least `MIN_POSES_PER_SCENE = 20` deep**.
Per scene, not pooled: a pooled row total let one scene carry the whole run — 191
poses in one scene and a single pose in nine others satisfied "10 scenes, 200
rows" and said nothing about cross-scene behaviour, which is the only thing a
threshold has to hold across. These are floors for detecting a short run, **not
targets**; the recommended pilot stays 15 scenes × 40 pose indices and nobody
should shorten a run to the line.

`threshold_freeze_status` is `pending | inconclusive | supported`. Every candidate
threshold in the sweep carries `delta_vs_current`: a **scene-clustered** bootstrap
interval on the yield change (`BOOTSTRAP_RESAMPLES = 2000`, fixed seed), the count
of scenes whose verdict actually flips, and a verdict. Poses inside one scene
share a floor, a mesh and a lighting, so resampling poses would give an interval
far too narrow to be honest. A candidate is `supported` only when at least
`MIN_SCENES_SUPPORTING_A_CHANGE = 5` scenes change verdict and the interval is no
wider than `MAX_YIELD_CI_WIDTH = 0.10`. **Precision, not significance** — a delta
confidently near zero is a perfectly good reason to move a gate; a delta that
could be anywhere in a ten-point band is not, whichever side of zero it favours.

**An inconclusive gate keeps its current value.** The answer to inconclusive is
more scenes, never a looser gate. `main` exits 2 when ineligible and 3 when no
gate change is supported.

Standing obligation, unchanged: **relaxing `FLOOR_MIN_INLIER_COVERAGE_RATIO`
requires re-running both synthetic contamination sweeps and confirming zero wrong
fits.** Yield alone may not move that constant — it is the gate that fixed the
seed-capture regression.

### P0-5 — Evidence coverage bucketing **on raw coverage**
Replace max-`departure_score` probe selection (`scripts/collect.py:1499`) with
target-coverage bucketed search on `raw_coverage`. Persist the full record — a
single `primary_failure_cause` loses information, since one rollout can be
simultaneously out-of-FOV, occluded and depth-invalid:

```
raw_coverage, qualified_coverage,
failure_causes[],            # SET, not one label
out_of_frame_samples, occluded_samples, invalid_depth_samples,
terminal_coverage, max_unsupported_run_m
```

`primary_failure_cause` may be derived for display under a deterministic priority.

Grey zone, numeric (pilot values, revisited with OQ-3): sufficient candidate
`raw_coverage >= 0.95` with no hard failure; insufficient `<= 0.80` or an explicit
hard failure; **0.80–0.95 not published**.

Scope: the action legitimately influences whether the corridor leaves the FOV, so
the goal is not to make `action_only` theoretically powerless on evidence — it is
to remove the dataset-level anomalous correlation via matched sampling.

### P0-6 — GS collision taxonomy, excluded from the **whole** consequence suite
Split into `gaussian_contact` / `navmesh_boundary` / `unsupported_floor`.
A `navmesh_boundary` or `unsupported_floor` event changes the realized stop pose
and therefore contaminates Q6 distance change, Q7 terminal bearing, Q8 future
visibility and Q9 terminal probes — not only Q1/Q3. Default: excluded from the
entire primary consequence suite. Drop-offs and support surfaces, if studied
later, are a **separate task**.

### P0-7 — Q3 gate on **superclasses**
Concrete contact categories are dynamic and cannot be enumerated in
`EXPECTED_CLOSED_STRATA`. Gate on `structural_contact` and `object_contact` for
scene count and support; scoring keeps the exact category.

### P0-8 — Q4 margin contract *(deferred to OQ-2; see §7 for the non-circular order)*
`BENCH_Q4_SAFE_CLEARANCE_M = 0.10` vs `goal.md`'s 0.30 m. Not decided here.

### P0-9 — Hierarchical aggregation, **two** headline scores

**Two parallel primary metrics, not one.** R3 (visible-space abstention) is part of
the core construct, but the consequence headline is restricted to
`evidence_status=='sufficient'`, so it structurally cannot measure R3. Folding
abstention into the same scalar would be worse. Frozen:

| Metric | Covers | Restriction |
|---|---|---|
| **Visible Consequence Score** | R1, R2 | closed ∧ sufficient ∧ `primary_balanced` |
| **Evidence Responsiveness / Abstention Score** | R3 | the evidence-calibration track, incl. Q5b |

A paper claiming R1–R3 must report both; neither alone is "the" headline.

Frozen aggregation order for the Visible Consequence Score — note the explicit
**variant level**, without which a family carrying more variants (Q6 has 2, Q9 has
3) or more reachable groups is silently over-weighted:

```
item
  → group × family × variant
  → scene × family × variant
  → family × variant
  → family              (variants equally weighted)
  → capability          (families equally weighted)
  → macro over the five capabilities
```

| Capability | Families |
|---|---|
| physical | Q1, Q2, Q3 |
| body-counterfactual | Q4 |
| relation | Q6, Q7 |
| view | Q8 |
| maneuverability | Q9, Q10 |

`Q0a` is diagnostic only. Open leaderboard and evidence calibration are separate.

**Missing-family policy (required — silent renormalisation is forbidden):**
- if OQ-1 retains Q10, a formal score requires Q10 to meet a minimum scene/group
  support;
- if OQ-1 removes Q10, the capability map is **formally amended first**;
- a family may never disappear and be silently renormalised away;
- since GS cannot supply Q10, the cross-backend headline additionally reports a
  **common-family intersection score**.

Report alongside: item micro-average, per-family item/group/scene counts, and
scene-cluster bootstrap CI with effective sample size.

### P0-10 — `sensor_invariant_consequence_signature`
`_same_signature()` coerces trailing fields to `float` and raises on a list, so
this is a new structure. A single extra mask is too narrow — under a camera change
every physical fact should hold:

```
sensor_invariant_consequence_signature = {
    physical numeric signature,                 # collision / contact arc / min clearance — tolerance
    executed action prefix,                     # exact
    realized pose,                              # tolerance, wrap-aware heading
    contact source + category,                  # exact
    true distance & bearing to a common target, # tolerance
    terminal full-geometry action definitions,  # exact
    tuple(physical_safe_mask),                  # exact  (post P0-13)
}
```

Explicitly **permitted to vary**: `future_view`, projected pixel area,
`depth_safe_mask`, `evidence_sufficient_mask`, and every `evidence_status`.
This split is the operational statement of the core design — physics invariant,
evidence variable.

### P0-11 — Executable shortcut gate
- single-choice families: **balanced accuracy**; multi-select: exact match against
  a **pre-registered random-strategy null**
- report **gain over majority**; scene-cluster bootstrap 95% CI; gate the UCB
- threshold `UCB(shortcut − majority) <= 0.05`, frozen before pilot

**Three-valued outcome** (a 10–20 scene pilot cannot resolve 0.05):

| verdict | condition | action |
|---|---|---|
| pass | `UCB <= 0.05` | proceed |
| fail | `LCB > 0.05` | fix the sampler |
| **inconclusive** | interval spans 0.05 | **add pilot scenes** — never relax the threshold |

**Attack strength frozen before the pilot** (otherwise a deliberately weak attack
"passes"): scalar threshold, logistic regression, small tree/boosting, action-token
n-gram, radius/choice-position rules; the attack's score is the **maximum** over
these.

Three statistical rules, frozen before the pilot:
1. **Majority and attack use the same metric**, then a **paired** scene-cluster
   bootstrap of the difference — not two independently bootstrapped numbers.
2. **Features, hyper-parameters and seeds are pre-registered.** Because the score
   is a maximum over several models, use **nested CV or a max-statistic bootstrap**
   so the maximum is not optimistically biased.
3. **`inconclusive` is not a pass.** It means "add pilot scenes". The formal stage
   requires **every** attack to reach `pass`.

Attacks: `action_only`, `total-distance threshold`, `body_radius_only` (Q1–Q3),
`radius_options_only` (Q4), `option_only`, `same-image group rank attack`,
`backend/family-only`.

Run on **both** raw pool (tests the sampler) and artifact (tests the release) —
**except `option_only`, which exists only on the artifact**. Report scene-disjoint
and group-disjoint folds **separately**.

### P0-12 — Sensor-consistency grouping and paired scoring
`scripts/gen_benchmark.py:760-770` must compare the **physical signature**
(P0-10), not `structured_answer`. As written, any sibling pair whose evidence
flips is recorded as a violation and discarded — exactly the Q5b sample.

**Q5 must not stay Q1-only.** `annotate_sensor_consistency_groups` currently hard-
codes `if item.get("family") != "Q1": continue` (`scripts/gen_benchmark.py:727`),
and Q1 reads the `physical` head, which is the head that almost never flips across
sensors (measured 0/18). Keeping Q1-only guarantees Q5b returns zero samples again.

Frozen **family/variant → evidence head** map (this is what decides where Q5b can
exist at all):

| family / variant | evidence head | Q5b viable? |
|---|---|---|
| Q1, Q2, Q3, Q4 | `physical` | rarely — keep as the **pre-registered near-zero control** |
| Q0a | initial `goal.surface_distance` | no — sufficient-only diagnostic |
| Q6, Q7 | `goal` | yes |
| Q8 | `future_view` | yes |
| **Q9 forward_safe** | `terminal_options[0]` | measured 0/18 — control |
| **Q9 turn_choice** | `terminal_options[1,2]` | **yes — 10/18 flip, primary Q5b base** |
| **Q9 certified_mask** | `terminal_options[0..4]` | **yes — 8/18** |

**Q5b must cover Q9 at minimum**, or it re-derives an empty set.

Pair identity conditions — a Q5 pair must share **scene, pose, action, radius,
private target instance, and physical signature**; only the sensor profile differs.

Frozen paired scoring:
- **Q5a (invariance):** both sides evidence-sufficient; both physical answers
  correct **and** mutually consistent.
- **Q5b (responsiveness):** one side insufficient and correctly abstained, the
  other sufficient and correctly answered the physical label; **exact match over
  the pair**.
- **Closed** items use pair exact match. **Open** items use the existing per-field
  tolerances (`pipeline/tolerance.py`) — never literal exact match on a float JSON
  blob.
- FOV and height interventions are **reported separately**; directions never pooled.

### P0-13 — Replace `safe_mask` with three masks, across all six consumers
Abolish the ambiguous field:

```
physical_safe_mask          # physics only — the answer key
evidence_sufficient_mask    # certifiable from the image
publication_eligible_mask   # margin — governs emission only
```

Rules: Q9/Q10 answers derive **only** from `physical_safe_mask`; insufficient
evidence makes the whole item abstain; the publication margin decides whether an
item is emitted and **never** changes the physical answer. Derived fields are
renamed accordingly (`num_safe_continuations` → `physical_num_safe_continuations`).

Consumers to migrate: `pipeline/rollout.py:643-671` (produce),
`pipeline/validate.py:668,685`, `pipeline/benchmark.py:923` (Q9),
`pipeline/benchmark.py:1433` (Q10 `has_two_safe_options`),
`pipeline/benchmark.py:1524` (Q10 constraint slack).

**Amended after implementation.** "Five consumers" undercounted: the review viewer
`scripts/serve_viz.html` also read `safe_mask` / `num_safe_continuations`, in a
column literally headed *Conservative* — so a reviewer inspecting a merely
unsupported or near-boundary option saw the word "blocked". It now renders the
three masks as three columns (*Physics (answer)* / *Evidence* / *Publishable*),
with a test pinning those labels and field names so the consumer cannot rot
again. Enumerated consumer lists keep undercounting here; sweep every file type,
not just `*.py`.

Two further points settled during implementation:

- **The emission gates already existed** and were not the defect.
  `benchmark.py:800-810` already refuses a Q9 variant whose probed options are
  near the margin (`terminal_option_near_boundary`), and Q10 already required
  `_all_terminal_option_evidence_sufficient` plus that same Q9 check. The bug
  was that margin and evidence *also* rewrote the answer, so they were counted
  twice: once as a legitimate withholding, once as a false "unsafe".
- **`physical_safe_mask` is by definition equal to `full_geometry_safe_mask`.**
  Both are persisted: the latter is the raw oracle output that pairs with
  `depth_safe_mask`, the former is the declared answer key. Keeping both is only
  defensible because each is **independently recomputed from
  `full_geometry_rollouts`** — an equality check between the two derived lists
  is not a pin, since a re-merge flips both together and passes it. P0-10's
  physical signature can then name the answer key directly.

**Every mask is recomputed, never cross-compared.** `pipeline/rollout.py` exports
the four derivations as pure functions and `validate_record` re-derives each
persisted mask through the same function:

| mask | source of truth |
|---|---|
| `full_geometry_safe_mask`, `physical_safe_mask` | `option_physical_safe_mask(full_geometry_rollouts)` |
| `depth_safe_mask` | `option_depth_safe_mask(depth_rollouts)` |
| `depth_supported_safe_mask` | `option_depth_supported_safe_mask(depth_safe, depth_corridor_coverage)` |
| `evidence_sufficient_mask` | `option_evidence_sufficient_mask(option_consensus)` |
| `publication_eligible_mask` | `option_publication_eligible_mask(full_geometry_rollouts)` |

The margin mask needs this most: it alone decides emission at
`benchmark.py:800-810`, so an unpinned one silently publishes near-boundary items
or withholds sound ones with nothing else in the record disagreeing.

**The derivations fail closed, and the mask type is part of the contract.** An
absent verdict is not a safe verdict: `not collision` read a missing depth
rollout as safe, and `first_contact_arc_m or 0.0` credited a collision with a
whole probe's worth of margin it was never measured to have. Both now require the
measurement to be present. The validator likewise requires `isinstance(value,
bool)` before comparing — `bool(value)` accepted a `0/1` encoding that no reader
is promised. `TERMINAL_OPTION_FORWARD_M` is the single source for the probe
length that both `terminal_option_actions()` and the margin rule use, so a
collision can never be scored against a probe that was never run.

Derived fields now follow physics: `maneuverability_score` and
`no_safe_probe_continuation` are computed from `physical_safe_mask`, not from the
old conjunction.

### P0-14 — Balancing must not delete strata silently
`scripts/gen_benchmark.py:534-570` drops a single-answerable-label stratum with
`reason: "single_answer_stratum"` and no gate failure. "Warning **or** failure" is
too loose — frozen rule:

| dropped stratum | outcome |
|---|---|
| required by `BACKEND_CAPABILITY_MATRIX` for that backend | **release gate FAIL** |
| unsupported for that backend, or diagnostic-only | logged **warning** with the dropped count |

Related: the per-family label-support gate (`:1566-1569`) must not be evaluated
over singleton long-tail labels that balancing preserves — evaluate over the P0-7
superclasses.

### P0-15 — One family per image, hard cap on questions per image

**Requirement (design decision, 2026-07-25): do not build many different questions
on one image.** More images, fewer questions each.

Measured on the current balanced artifact:

| | value |
|---|---|
| images | 216 |
| questions | 10 522 |
| **questions per image** | median **53**, max **190**, mean 48.7 |
| families per image | median 2, max 4 |

Driver: `--max-pairs-per-image-family` defaults to **50** — that is 50 pairs per
family *per image*, and the median image serves 2 families. Consequence is F9:
nominal 10 522 items carry the precision of ~1/2 to 1/4 that many (DEFF 1.9–4.1),
and manual review keeps revisiting the same image.

Frozen rules:

1. **One family per image** (currently median 2, max 4). Near-zero cost — it only
   stops one image from serving both e.g. Q1 and Q8.
2. **Cap = at most 6 *primary closed items* per image.** The unit matters: the
   existing `--max-pairs-per-image-family` counts *pairs*, and 6 closed/open pairs
   is 12 QA items, not 6 questions. Frozen unit is **primary closed items**; the
   open leaderboard is a separate track and does not count toward the image-scale
   claim.
3. **Allocation priority.** Scarce protocols are allocated first — Q4, Q10, and
   Q5 paired items — and only then is one-family-per-image applied to the
   remainder. Otherwise the rarest families lose their images to the common ones.

This replaces the blind `SINGLE_ANSWER_TRACK` hash (which discards 72% of eligible
items *before* balancing, F14). Allocation happens **after** computing which
families each image can actually support, so the image-isolation intent is kept
without blind attrition.

**Target scale, restated in images rather than questions.** Cost is dominated by
rendering new images; extra actions on an existing image are nearly free (pure
geometry). Measured throughput: r2r ~21 s/attempt, **gs ~252 s/attempt**, accept
rate ~20–24%, i.e. ~88 s per usable r2r image but **~20 min per usable gs image**.

| questions/image | images for ~10k questions | gs single-GPU time |
|---|---|---|
| 50 (current) | ~216 | ~1.5 days |
| **6 (frozen)** | ~1700 | ~8 days |
| 1 | ~10 000 | ~70 days |

Frozen target: **800–1200 images × ~6 questions ≈ 5000–7000 questions.** That has
*higher* effective sample size than today's 10 522 (4–5× more images), keeps manual
review to 6 questions per image, and costs ~270 gs images ≈ **3.7 gs GPU-days**.

**Source ratio stays 1:1:1.** gs is ~12× slower and therefore caps the total, but
1:1:1 is an explicit fairness claim of the design and remains frozen here. If it is
ever relaxed, that is a **separate source-quota decision** with its own
justification — it must not be left suspended inside this freeze.

`assign_scene_splits` is **not** dead code and must not be deleted: the formal
compiler explicitly enters `split_mode="benchmark"` to keep scene-atomic splits;
only the candidate-pool path bypasses it.

### P0-16 — Schema and contract version bump

P0-4, P0-5 and P0-13 change the meaning of persisted fields (published camera
height, evidence record, safety masks). Continuing to call both old and new data
`conseq.v5` would make the two silently mixable — the exact failure the whole
freeze exists to prevent. Frozen:

```
pipeline/record.py    SCHEMA_VERSION          conseq.v5               → conseq.v6
pipeline/record.py    ORACLE_CONTRACT_VERSION ground-disc-visible-v3  → ground-disc-visible-v4
pipeline/benchmark.py PROMPT_CONTRACT_VERSION visible-ground-disc.v4  → visible-ground-disc.v5
pipeline/benchmark.py QA_SCHEMA_VERSION       egoconseq.qa.v5         → egoconseq.qa.v6
```

The later Q0a sufficient-only publication rule changes its public choice
contract without changing record or oracle semantics, so it advances only
`PROMPT_CONTRACT_VERSION` from `visible-ground-disc.v5` to
`visible-ground-disc.v6`.

The later strict Q6 distance-trend rule removes the relative semantic dead zone:
any negative distance change is closer, any positive change is farther, and only
exact equality is unchanged. It advances only `PROMPT_CONTRACT_VERSION` from
`visible-ground-disc.v6` to `visible-ground-disc.v7`.

The new reader, validator and HTTP loader **reject v5 outright** — no compatibility
shim. Existing v5 pools stay `diagnostic-only`.

**Amended after implementation.** "Reader, validator and HTTP loader" understated
the boundary and left holes. Measured against the real v5 artifact
`data/benchmark/smoke_sensor_reserve_20260724`: `eval_benchmark.py` scored it to
completion (exit 0, `Q1: 1.0`); `run_qa_baselines.py` read it with no version
check; `aggregate_human.py` produced `pooled_accuracy=1.0` from it; and
`build_paper_snapshot.py` reported `available=True` and embedded three of its
retired-protocol QA examples. An artifact is also stamped in three independently
written places — manifest, every public item, every private answer — so a
manifest-only gate passes a partially regenerated artifact.

The frozen requirement is therefore: **all six** artifact entry points check
**all three** stamps through one shared gate,
`pipeline.validate.assert_artifact_contract`.

| Entry point | What a stale artifact would have produced |
|---|---|
| `serve_benchmark.py` (also backs `serve_viz.py`) | retired prompts shown to a human reviewer |
| `eval_benchmark.py` | a headline number under retired semantics |
| `check_benchmark.py` | a passing release check |
| `run_qa_baselines.py` | a shortcut floor incomparable to the headline |
| `aggregate_human.py` | a published human ceiling under retired semantics |
| `build_paper_snapshot.py` | retired question text and GT embedded in the paper |

New entry points extend that gate rather than re-implementing the comparison. The
enumeration is the failure mode: both misses above came from trusting a list of
entry points instead of sweeping for every reader of
`manifest.json` / `items.jsonl` / `answers.jsonl`.

Checking the same *values* is not enough if two entry points read different
*files*. `manifest["files"]["items"]` is the authoritative pointer, and
`run_qa_baselines.py` and `build_paper_snapshot.py` were hardcoding
`public/items.jsonl`: given a manifest declaring a stale file next to a
current-stamped decoy at the default path, the evaluator refused while those two
scored and published the decoy. All six now resolve the declared path through
`pipeline.validate.artifact_items_path`, which also treats the declared name as
untrusted input — relative, `.jsonl`, contained in `public/` — so the traversal
guard that previously existed only in the viewer now applies everywhere.

`aggregate_human.py` additionally only accepted a **flat** `items.jsonl` /
`answers.jsonl` layout that no current generator writes, so it could only ever
have been run on a hand-staged directory. It now reads the same
`public/` + `private/` split every other entry point uses.

`AGENTS.md` pinned `conseq.v5` while this section was still a plan, and flipped to
`conseq.v6` in the same commit that implemented P0-16, together with the code and
the tests. **Done** — the repository now states `conseq.v6` everywhere it states a
current contract.

Note the two version families are independent and only one of them moved.
`BENCHMARK_VERSION = "v5"` is the **question taxonomy** (which families and
variants exist, Q0a--Q10); `conseq.v6` / `egoconseq.qa.v6` /
`visible-ground-disc.v7` / `ground-disc-visible-v4` are the **persistence and
prompt contracts**. P0-16 bumped the latter only. Text that says "QA v5" meaning
the taxonomy is correct; text that says "QA v5" meaning the schema is stale.

---

## 3. Open questions requiring experiments (not frozen)

| ID | Question | Experiment |
|---|---|---|
| OQ-1 | Does the Q10 reserve yield 4 same-length candidates at usable rates? | Q10 reserve pilot, 10–20 scenes × {hm3d, r2r}. If yield ≈ 0, formally amend the capability map. |
| OQ-2 | Is 0.10 m safe clearance humanly answerable? | Human pilot over 0.10 / 0.20 / 0.30 m subsets. Decides P0-8, revisits `FLIP_RATIO`. |
| OQ-3 | Achievable sensor-profile flip rate per evidence head, and direction? | Sensor reserve pilot; stratify (fixed height, vary FOV) and (fixed FOV, vary height) **separately**. |
| OQ-4 | What quotas do the flagship protocols need? | Pilot yield + power analysis. Q5b is paired → McNemar power depends on the **discordant-pair rate**. Re-estimate DEFF from the pilot. |
| OQ-5 | Does the matched-pair sampler have acceptable yield? | Pilot; relax matching tolerance only in a **pre-registered** order. |

---

## 4. Implementation matrix

| P0 | Primary files | Tests to add | Acceptance metric |
|---|---|---|---|
| P0-1 | `collect.py:495`, `validate.py:218` | safe-count varies across poses | group-rank attack UCB ≤ .05 |
| P0-2 | new sampling module, `collect.py` | matched-pair key equality; no fixed pair count | `action_only` UCB ≤ .05 |
| P0-3 | `collect.py`, `benchmark.py`, `validate.py`, `gen_benchmark.py` | no singular `body_radius_m`; exactly four printed radii; Closed has three adjacent transition intervals; Open has a four-value physical safety vector plus all-safe/all-collision controls; rank support is counted by scene | rank-macro accuracy; `radius_options_only` UCB ≤ .05; every backend has all three ranks and both control kinds |
| P0-4 | `perception.py:114`, `frame.py:118/122/129`, `objects.py:52/60`, `rollout.py:784/795/816-828`, `benchmark.py:971/975`, `collect.py:151`, record/manifest/viewer/validator | `published_height == signed_normal_distance(camera_centre, canonical_floor_plane)`, fitted once per physical pose at the fixed reference profile (1.5 m / 79°) and shared by every sibling; no scalar fallback, no final 5 cm quantization | cross-sibling effective-height agreement; 0 poses published with a fallback floor |
| P0-5 | `collect.py:1499`, `rollout.py:865+` | `failure_causes` set persisted; grey zone enforced | raw-coverage distribution spans buckets |
| P0-6 | `gs_geometry.py:163` | taxonomy assigned; non-`gaussian_contact` excluded suite-wide | 0 published consequences from snap events |
| P0-7 | `gen_benchmark.py:269,1566` | superclass gate passes with long-tail labels | Q3 release gate reachable |
| P0-9 | `eval_benchmark.py:534-561` | six-level aggregation; missing-family policy | headline invariant to family item counts |
| P0-10 | `validate.py:937,1047` | list fields compared exactly; numeric with tolerance | 0 sensor-invariance violations |
| P0-11 | new audit script | three-valued verdict; attack battery maximum; paired bootstrap; pre-registered features/seeds | **formal stage: every applicable attack must `pass`**; `inconclusive` → enlarge the pilot, never a pass |
| P0-12 | `gen_benchmark.py:760-770` | evidence-flip pairs form Q5b groups, not violations | Q5b pair count > 0 |
| P0-13 | `rollout.py:643-671`, `validate.py:668`, `benchmark.py:923/1433/1524` | Q9 answer independent of margin | margin change does not alter any answer key |
| P0-14 | `gen_benchmark.py:534-570` | stratum drop raises gate failure | 0 silent drops |
| P0-15 | `gen_benchmark.py:46-52,105-122,421-473` | one family per image; **≤6 primary closed items per image**; scarce protocols (Q4/Q10/Q5-paired) allocated first | ≥800 images; DEFF lower than v3's 1.9–4.1 |
| P0-16 | `record.py:13-14`, `benchmark.py:23-24`, `validate.py`, `serve_benchmark.py`, `eval_benchmark.py`, `check_benchmark.py`, `run_qa_baselines.py`, `aggregate_human.py`, `build_paper_snapshot.py` | record reader rejects `conseq.v5`; all six artifact entry points reject `egoconseq.qa.v5` in manifest, item **and** answer | 0 v5 artifacts accepted by any entry point |

---

## 5. Target pipeline shape and code boundaries

```
train scene resolution
  → pose / sensor rendering + floor calibration
  → action candidates + matched / directed reserve search
  → full-geometry GT + rendered-depth evidence qualification
  → continuous records compiled into a margin-filtered QA artifact
  → shortcut / human / GT / release gates
```

**The key structural change:** collection stores *continuous, auditable candidate
GT*; publication margins, label balancing and templating happen in the compiler.
Every publication condition pushed into `collect.py` distorts the sampling
distribution — the mechanism behind F1, F2 and F3.

Division of responsibility — note that acquisition keeps more than orchestration:

| Acquisition (`collect.py`) | Compilation (`gen_benchmark.py`) |
|---|---|
| simulator invocation | publication margins |
| **oracle validity** | task eligibility |
| **dual-readout agreement** | label / source balancing |
| **floor calibration** | templating |
| continuous GT | split assignment |
| raw evidence statistics | release gates |
| atomic persistence | |

Dual-readout agreement and floor calibration are **acquisition validity** and
cannot be deferred to the compiler.

Code boundaries: extract pure action matching, Q4 bucketing and evidence-bucket
selection into a **NumPy-only sampling module** (testable without a simulator);
shortcut attacks live in a **standalone audit script**, never in the scorer; the
controller schedules an **explicit mode list** (`main`, `q4`, `q10`, `sensor`) so a
coverage-first path can never silently skip reserves again.

---

## Q2 contract amendment (2026-07-26)

The former `stop_progress` task divided cumulative commanded forward distance at
an arbitrary 50% boundary. It is replaced by `execution_stage`:

- Q2 is emitted only for programs with at least two Forward legs;
- the closed answer is `contact_during_forward_N` or `complete`, with choices
  generated from the program's actual number of Forward legs;
- the open answer reports the contact Forward-leg number, metres executed within
  that leg, and total executed forward metres;
- full-geometry and rendered-depth contact arcs must resolve to the same Forward
  leg when evidence is sufficient;
- there is no half-progress margin and collection no longer filters candidates
  on Q2 bins; Q2 eligibility belongs to artifact compilation;
- the two metric fields use MRA and are also reported with MAE.

This was a hard QA break at the time: taxonomy `v6`, schema
`egoconseq.qa.v8`, prompt contract `visible-ground-disc.v9`, and eval report
`egoconseq.qa.eval.v6`. These QA versions are superseded by the Q4 amendment
below. Record schema `conseq.v6` and oracle contract
`ground-disc-visible-v4` remain unchanged because the stored continuous rollout
already contains the required GT.

---

## Q4 contract amendment (2026-07-27)

The former Q4 asked for a maximum safe radius from three candidate radii plus
dead `none_safe` and `insufficient_evidence` options. Under the required true
flip, only two candidates could actually be correct, so its nominal five-way
chance rate was not its effective chance rate. It also exposed a singular
`body_radius_m` equal to the smallest candidate in every audited item.

The replacement freezes the task structure, but deliberately does not freeze
the numerical grid or grey-zone constants before their pilots:

- public Q4 input contains exactly four increasing
  `candidate_body_radii_m` values and no singular `body_radius_m`;
- Closed `radius_transition_interval` has exactly three choices:
  `between_1_2`, `between_2_3`, and `between_3_4`; true-flip items only;
- the Closed primary metric is macro accuracy over transition rank, after
  averaging groups within scenes, so a constant-rank predictor scores 1/3 even
  if the item counts are imbalanced;
- Open `radius_outcome_vector` returns exactly four `safe`/`collision` values;
  it includes true flips plus all-safe and all-collision controls, and is the
  sole Q4 measurement of monotonicity and no-flip behavior;
- the public near-field certificate is independently recomputed from the
  **largest** candidate radius;
- `critical_radius_m`, transition rank, pairing metadata, clearances, and
  publication margins remain private;
- formal publication uses a geometric grey zone around the continuously
  searched critical radius. The four-value grid, absolute margin, relative
  margin, and single-grid versus grid-bank choice have no defaults and require
  the Q4 yield and human-answerability pilots;
- formal release requires all three transition ranks to have scene support and
  both Open control kinds to exist for every supported backend.

This is a hard QA break: taxonomy `v7`, schema `egoconseq.qa.v9`, prompt contract
`visible-ground-disc.v10`, and eval report `egoconseq.qa.eval.v7`. Record schema
`conseq.v6` and oracle contract `ground-disc-visible-v4` are unchanged.

---

## 6. Acceptance criteria (pilot → formal)

1. `action_only`, `total-distance threshold`, `group rank attack`:
   `UCB(gain over majority) <= 0.05` on both raw pool and artifact.
2. `radius_options_only` on Q4 meets the same bound; every transition rank has
   the frozen minimum scene support and every supported backend supplies both
   Open control kinds.
3. Abstention not predictable from action text (Q9 evidence shortcut closed).
4. `raw_coverage` and failure-cause distributions show a genuine gradient.
5. `check_records.py` passes on every shard; `sensor_invariant_consequence_signature` passes.
6. No public field carries the raw sensor offset; effective height consistent with
   the fitted floor plane.
7. **Human answerability gate** — numeric thresholds frozen before the pilot:
   minimum human accuracy per family, minimum inter-annotator agreement
   (Krippendorff α or Fleiss κ), and the CI width at which the comparison is
   declared. This gate is what detects "anti-shortcut made the questions
   unanswerable".
8. **Pilot GT audit** — manual review of corrected camera height, coverage cause
   set, GS collision source, Q4 flips, contact category.

**DEFF note.** The 1.9–4.1 measured on v3 is a planning prior only; the new sampler
changes the clustering structure, so the pilot must re-estimate it.

---

## 7. Execution order (resolves the P0-8 circularity)

1. ~~Stop and freeze `train_1to1to1_v3_20260724` as diagnostic~~ — **done
   2026-07-25 06:47 EDT**; 248 records across 29 `records.jsonl` (27 non-empty,
   2 empty); all non-empty shards pass validate with 0 violations.
2. Sign off on this document.
3. Implement in dependency order. **P0-8 stays open.**

```
P0-16 (done)               versioning first — otherwise renamed fields still say v5
  → P0-13 (done)           split safe_mask (changes the answer key)
  → P0-4 (done 2026-07-25) effective camera height (changes every published scale)
  → P0-10                  sensor-invariant signature
  → P0-12                  sensor grouping — needs P0-10's signature
  → P0-1 / P0-2            sampler
  → P0-3 / P0-5 / P0-6     Q4, evidence buckets, GS taxonomy
  → P0-7 / P0-9 / P0-14 / P0-15    compiler: gates, aggregation, allocation
  → P0-11                  shortcut gate last — it scores everything above
```
4. **Pilot root** (not formal): `0.10 m` as a *storage* floor only, persisting
   continuous clearance GT, stratified by clearance band. Schedule all modes
   (main + q4 + q10 + sensor), 10–20 scenes per source. Nothing is claimed
   publishable.
5. Generate 0.10 / 0.20 / 0.30 m filtered subsets from the one pilot pool.
6. Run OQ-2 → decide P0-8 and revisit `BENCH_Q4_FLIP_RATIO`.
7. Measure §6 on the pilot.
8. Only if §6 passes **and** P0-8 is decided: open a new formal `--out-root`.
9. Baselines (RGB-monocular-depth, direct VLM, privileged depth) and human ceiling.
