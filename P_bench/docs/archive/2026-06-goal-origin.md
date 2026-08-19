> **Superseded origin document — archived, not current.**
>
> This is the first framing of the benchmark and is kept for provenance only.
> The authoritative design is `docs/archive/2026-07-27-benchmark-design-freeze.md`;
> the operational protocol is `BENCHMARK.md`. Several statements below were
> overturned and must not be implemented from here:
>
> - the level-first taxonomy: QA v15 removes the public `level` field and
>   scores by result head, with `rollout_stage` kept as a non-scoring attribute;
> - "exactly three safe plus three collision programs" per pose: a fixed
>   per-pose label contract is now rejected by `pipeline/validate.py`, which
>   requires naturally variable action candidates;
> - the safe-clearance figure: `pipeline/config.py` is authoritative.
>
> What remains current is the ground-disc definition it introduced: the body is
> a 2D circular footprint whose radius is the only body variable.

# EgoConseq-Bench: Visible-Space Action Consequence Reasoning

> Can a vision-language model bind its own continuous footprint, the 3D scene,
> a metric action program, and partial observability to predict structured
> embodied consequences and abstain when the image is insufficient?

## 1. Research Scope

EgoConseq-Bench evaluates short-horizon future-state reasoning from one
calibrated monocular egocentric image. It is not a navigation-policy benchmark
and does not ask a model to generate future RGB pixels. The model receives:

\[
(O_0, h_{opt}, HFOV, VFOV, r_B, A, g)
\rightarrow
Y_{1:K}, E_Y,
\]

where `O_0` is the current RGB observation, `r_B` is the robot footprint
radius, `A` is a metric action program, `g` is an optional target instance,
`Y` contains structured consequences, and `E_Y` states whether the observation
supports each consequence.

The public input is exactly RGB, camera optical-center height above the floor,
HFOV/VFOV, body radius, action program, and the target when needed. Resolution
is declared once in the public artifact manifest. Focal length, depth, semantic
labels, pose, maps, and simulator state are private.

The V5 robot abstraction is a **2D ground disc**. Radius is the only
embodiment variable.
The benchmark makes no claims about vertical clearance, articulated geometry,
dynamics, pushing, or deformable contact. A fixed near-ground support band is
an oracle construction rule used to remove floors and ignore purely suspended
surfaces; it is not a robot parameter.

## 2. Observation Contract

Camera optical-center height, HFOV, and VFOV are calibrated model inputs. The
height provides the metric floor reference and FOV defines angular calibration.
Resolution is manifest metadata rather than per-question prompt text. Focal
length is not public. Camera height and FOV are **observation-only** variables:
changing either may change RGB evidence, but never the physical rollout for a
fixed scene, pose, disc radius, and action.

The task is restricted to visible-space consequences:

- every contact-capable obstacle is represented where its ground support is
  visible;
- the explicitly certified near-floor strip contains no hidden obstacle;
- a fully visible, clear swept corridor may be treated as clear;
- if relevant path space leaves the image or passes behind an occluder, the
  correct response is `insufficient_evidence`.

Full-scene geometry determines privileged physical GT. Initial RGB-D determines
whether that GT is supported by the supplied view. These labels must remain
separate: an outcome may be physically known to the benchmark but unknowable
from the public image.

## 3. Action Programs

Main programs contain one to six primitives and at least one forward motion.
Adjacent primitives strictly alternate between Turn and Forward; consecutive
turns and consecutive forwards are invalid. Forward distances are selected
from `{0.5, 1.0, 1.5, 2.0, 2.5, 3.0}` metres. Main turns use
`{+/-15, +/-30, +/-45}` degrees. Every benchmark program contains translation;
rotation-only questions are outside the task.

The robot turns in place, then translates along its current heading. On first
contact it stops immediately and skips all remaining primitives. Every final
state is therefore the realized state, not the nominal endpoint behind an
obstacle.

Main action groups reuse exactly the same programs across radii and calibrated
sensor profiles. Each complete pose contains exactly one program at each length
1--6, with exactly three safe and three collision programs globally.
Selective-evidence samples use the same action
vocabulary so abstention cannot be predicted from action text alone.

## 4. Structured Consequences

Each realized checkpoint stores four related heads.

### Physical interaction

- completion and executed forward-distance fraction;
- clearance trajectory and minimum clearance;
- collision and first-contact arc length;
- contact object attributed by the full-geometry and depth readouts, kept only
  when the two same-scene readouts agree.

### Goal-relative state

- disc-to-target ground-surface distance;
- fixed-target-region geodesic distance when the navmesh supports it;
- surface/geodesic distance change and closest approach;
- target bearing and bearing sector;
- whether execution stops before the nominal target-relative state.

### Future observation

- endpoint target visibility, normalized image center, and projected area ratio;
- a closed-only diagnostic counting initially visible non-structural instances
  absent at the realized endpoint;
- checkpoint observations at realized poses.

### Future options

- certified mask over exactly five local probes: Forward 1 m and
  `+/-15`/`+/-30` degree turn-then-Forward 1 m;
- `no_safe_probe_continuation` when none of those five probes is certified;
- no claim that five failed probes prove a global dead end.

## 5. GT and Evidence Requirements

HM3D and R2R-train/MP3D use scene geometry plus rendered depth. GS uses
Gaussian full geometry plus rendered depth; this must not be described as an
independent mesh oracle. An answerable physical sample requires:

- full-geometry and depth collision labels to agree;
- first-contact arc estimates to differ by at most `0.30 m`;
- at least 90% initial-view coverage of the realized swept-floor corridor;
- collision examples to stop with at least `0.50 m` nominal forward path left;
- safe examples to retain at least `0.30 m` minimum clearance.

The full-geometry and depth readouts are two geometric passes over the same
scene, not independent oracles: each locates its own contact point against one
shared semantic index, and Q3 is emitted only when their instance/category
decisions agree. Q4
tests several genuinely rolled-out radii, includes both safe and colliding
choices, and balances the rank of the largest safe candidate to prevent a
"pick the smallest radius" shortcut.

Q6 has `surface_proximity` and closed-only `geodesic_progress`; a sample is
emitted only outside relative grey zones and when surface and geodesic trends
agree. Q7 requires both executed turn and translation plus measurable parallax,
so it cannot collapse to a rotation-only bearing update. Q8 primarily asks
target visibility at the realized endpoint and retains a closed-only
leaving-count diagnostic over initially visible non-structural instances; both
use visibility grey zones. Q9 evidence for all five probes is certified from
the initial observation; a private terminal render may provide physical GT but
may not turn an unobservable continuation into a sufficient sample. Q10 writes
its active constraints in the question and requires every participating head
to be sufficient and outside its margin. Its four unique candidates always
share one action length, so program length cannot identify a candidate, and all
five terminal probes must be supported even when maneuverability is not an
active constraint. Its explicit 30% geodesic-progress condition uses a
25--35% publication grey zone.

## 6. Level-First Question Taxonomy

Questions are organized by imagination depth, not by surface wording. This
depth is an *a priori* design axis -- how many steps of physical consequence a
family is designed to ask the model to imagine -- fixed when the family is
authored. It is orthogonal to a question's *operator* (what it asks for, e.g.
Q10's joint selection): a family's level is a deliberate design assignment, not
a value mechanically derived from its operator or template wording.

| Level | Families | Capability |
|---|---|---|
| L0 | Q0a | static metric anchoring; diagnostic only |
| L1 | Q1--Q3 | collision, multi-Forward execution stage/distances, and first-contact identity |
| L2 | Q4 | true safe-to-collision disc-radius flips |
| L3 | Q6 | surface proximity and closed-only geodesic progress with trend agreement |
| L3 | Q7 | bearing after executed turn, translation, and parallax |
| L3 | Q8 | endpoint target visibility plus a closed-only leaving-count diagnostic |
| L4 | Q9 | fixed five-probe continuation feasibility |
| L4 | Q10 | selection under explicitly stated joint constraints |

Q5 is an evaluation protocol, not a generated question family. It measures
physical-answer consistency across matched camera-height/FOV observations.

L0 is reported only for error attribution. L1--L4 form the core evaluation.
Every enabled family and answer format has at least five controlled templates
that preserve semantics and do not name the intervention being measured.
Q0a is compiled only when the visible target points support the initial surface
distance within the evidence tolerance. Its four ordinal distance choices do
not include an abstention option; unsupported Q0a candidates are withheld.

Closed questions and registered structured-open variants are separate
leaderboards. Q6 `geodesic_progress` is closed-only. Closed questions include
`insufficient_evidence`; open questions use a strict JSON schema and return
`{"answerable": false}` when evidence is insufficient.

## 7. Causal Groups and Splits

Each world pose defines an intervention group. Siblings share the world pose,
action bank, and split assignment. Three axes are controlled independently:

- **disc radius:** image and action fixed, physical clearance may change;
- **sensor profile:** disc and action fixed, physical GT must not change;
- **action:** image and disc fixed, future consequences may change.

Larger nested discs cannot gain clearance or turn a collision into a safe
rollout. Sensor changes may change evidence sufficiency but cannot change full
physical GT. Groups are written atomically. Source-train collection first
creates an unbalanced `split=unassigned` candidate pool; no EgoConseq test set
is invented during sampling. After the pool and selection policy are frozen,
any benchmark partition must be scene-atomic and keep each intervention group
together.

The dataset must deliberately include conflicting outcomes: closer but less
visible, safe but retaining fewer certified probes, visibility-improving but
farther, and nominally closer actions stopped by contact. Q10 is the flagship
operator: the prompt explicitly lists which subset of safety, geodesic goal
progress, endpoint visibility, and follow-up-probe constraints must hold.

## 8. Evaluation

Report closed and structured-open results separately, with macro averages over
L1--L4 and per-family diagnostics. Required metrics include:

- collision accuracy, false-safe rate, and false-completion rate;
- stop-progress, contact, distance, bearing, visibility, and option metrics;
- constrained-selection exact set match;
- radius monotonicity and Q5 sensor-consistency violations;
- evidence accuracy, false/missed abstention, calibration, and risk-coverage;
- positive-distance MRA and 25% relative accuracy.

Risk-coverage ties are evaluated as confidence blocks and never ordered by GT.
Coverage uses the full expected item set. Required controls include majority,
prompt-only, image-only, action-only, disc-only, option-only, direct VLM,
monocular-depth geometry, and privileged-oracle upper bounds.

## 9. Release Gates

Before release, HM3D, R2R-train/MP3D, and GS-train must independently satisfy:

1. zero record/artifact schema and physical-invariance violations;
2. complete action-length 1--6 coverage and the common visible-space tasks:
   Q1--Q4, Q6 surface proximity, Q7, Q8, and Q9;
3. at least 50 genuine Q4 radius transitions overall, with at least 25 from
   each source and balanced transition ranks;
4. sufficient/insufficient action-text distributions that cannot be separated
   by a shortcut baseline;
5. GT-as-prediction scores of 1.0 for every populated family and format;
6. human review of stored GT, top-down paths, checkpoints, and sibling groups;
7. strong VLM performance below the human ceiling on joint conflict groups.

The source capability matrix is explicit: HM3D and R2R/MP3D additionally must
provide Q6 geodesic progress and Q10, both of which depend on their navigation
meshes. R2R contributes only a train-scene whitelist; paths, starts, goals, and
instructions are never used. GS deliberately omits the map-aware variants; GS
absence is permitted, but GS emission is rejected. This boundary does not
relax any common L1--L4 or Q4 requirement.

## 10. Hard Exclusions

V5 does not evaluate vertical clearance, pushing, opening doors, movable-object
dynamics, social response, stair traversal, arbitrary robot meshes, future
video realism, long-horizon task success, global planning, or real-world safety.
