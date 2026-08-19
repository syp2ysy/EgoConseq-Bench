# B1K ABC1 Adapter Implementation Plan

## Global constraints

- R2R is an immutable baseline. Keep `r2r_v16_collection_contract`, existing
  R2R validation messages, schemas, and current Golden outputs byte-identical.
- The new backend is named `b1k`: modules use the `b1k_` prefix and records use
  `source_dataset="b1k"`, `official_split="train"`, and
  `split_authority="project_defined"` in the trusted source manifest.
- Scene count comes from the installed catalog. Fewer than three scenes is an
  installation failure; no code may assume exactly 50 scenes.
- B1K uses base `InteractiveTraversableScene` scenes, `DummyTask`, no robot,
  and the existing abstract disc radii 0.15/0.20/0.25 m.
- C1 remains candidate-review only while its B1K gate is pending. Never borrow
  an R2R threshold or emit a formally eligible benchmark.
- Production changes follow test-first development. Habitat-only dependencies
  must remain lazily isolated from the B1K Python 3.11 environment.

## Task 1: Contract dispatch and R2R-only gates

Add failing synthetic tests, then implement a B1K v11 collection contract and
one dataset dispatcher. Route all current hard calls through it while leaving
the existing R2R builder unchanged. Add `b1k` to source-format and physical
authority validation, preserving the train split check.

Remove the silent B target failure: R2R follows its current exact PLY path;
B1K calls a source-bound B1K semantic authority; unsupported strict contracts
raise a clear error. Generalize C1 source and renderer binding so R2R produces
the exact existing atoms and B1K binds the real B1K scene authority and Isaac
renderer. Generalize the benchmark stability-authority check without changing
R2R serialized certificates or errors. Tests must prove B1, B2, and C1 no
longer disappear merely because the source is B1K.

## Task 2: Pure B1K geometry and semantic authorities

Add `pipeline/b1k_geometry.py` and `pipeline/b1k_semantic.py` with no top-level
OmniGibson import. The world mapping is
`PBench[x,y,z] = OmniGibson[x,z,-y]`; yaw 0 faces OG +Y and yaw +90 degrees
faces OG -X.

Build ground support from loaded floor collision surfaces. Clip each real
collision component independently to the 0.05--0.30 m obstacle band, union its
2-D footprint, and create radius-conditioned configuration spaces for the
three frozen radii. Sample from this free space rather than `floor_trav`.
Expose the geometry-query/nav interface consumed by rollout, including
unsupported-floor attribution and closest-obstacle contact evidence.

Build stable instance IDs from sorted runtime prim identities. Store raw
category/model/prim privately and expose the frozen B1K synset label publicly.
Provide exact contact confirmation and B-target geometry from canonical
runtime triangles, bound to one derived scene-authority digest. Add synthetic
tests for handness, all radii, concavities, unsupported floor, substitution,
contact winner/runner-up, and target atoms.

## Task 3: B1K simulator, discovery, CLI, and candidate review

Add `pipeline/b1k_sim.py` with lazy OmniGibson imports. Load base scenes with a
DummyTask and no robot, create an external VisionSensor for RGB,
`depth_linear`, instance and semantic segmentation, and return the existing
session/render interfaces. Verify physical reset by object/joint pose with
absolute 1e-6 m/rad tolerances; image variation is diagnostic only.

Add B1K source-manifest discovery as the fourth dataset path. It records the
actual scene list and hashes scene JSON, layouts, referenced encrypted assets,
derived geometry/semantic authority, initial state, simulator, and asset
versions. Encrypted USD geometry is accessed only after OmniGibson loads it.

Extend collection CLI/runtime to `--backend b1k` with explicit data-root and
source-manifest arguments. A/B use the normal candidate builder. With a
pending C1 gate, archive the four terminal members and exact pair metrics in
`candidate_review/`, never formal `candidate_qa`. Add source-bound local tests
and a fake-runtime end-to-end test proving nonzero B1/B2 and one four-member
C1 review family.

## Task 4: Installation, probes, and real pilot

Conservatively remove only the previously audited literal stale targets and
Python/pytest caches; use no glob and preserve `data/conseq/semantic_cache`.
Re-run full pytest and R2R Golden immediately after cleanup.

Install the pinned BEHAVIOR-1K v3.9.1 source revision in a Python 3.11 conda
environment named `behavior`, with data rooted at
`/home/zhangshan/syp/datasets/behavior-1k-v3.9.1`. Pre-create the data and
appdata directories, export all OmniGibson paths before setup, log with
pipefail/tee, verify versions and quarantine partial installs as
`.failed-<UTC>` rather than resuming them.

Before the real collection, run three deterministic scenes through probes that
always record and continue: collision-minus-visual slice area and one-sided
Hausdorff; timed AABB-prescreened 2.5 cm disc queries; reset pose error plus RGB
PSNR/change ratio; semantic resolution rate and unresolved examples. Only an
inability to extract runtime geometry or build/query C-space blocks adapter
execution.

Run a real source-bound pilot in deterministic scene order until A1/A2/A3/B1/
B2 each have at least one candidate and C1 has one four-member review family,
or the installed catalog is exhausted. Report A2 full/depth contact-arc
differences and consensus by radius without changing the 0.30 m tolerance.
After the three-scene smoke passes, partition every installed scene into four
deterministic, scene-disjoint GPU shards with independent output directories
and resumable commands. Do not claim readiness if B1, B2, or C1 yield is zero.

## Final verification

- Habitat Python: full pytest, `scripts/check_abc_golden.py`, record validation,
  GT-as-pred, and `git diff --check`.
- Worktree Golden may differ only in the two already measured absolute-path
  authority digests; after integration at the canonical repository path the
  checker must report all 19 current files byte-identical.
- B1K Python: synthetic/fake-runtime suites, three real scene loads with
  nonconstant RGB, valid depth, multiple instance IDs, source replay, and the
  real candidate pilot.
- Final status stays `headline_eligible=false`; formal B1K C1 calibration and
  GS collection support are out of scope.
