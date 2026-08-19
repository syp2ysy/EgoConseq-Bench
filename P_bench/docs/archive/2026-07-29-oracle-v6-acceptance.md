> **Superseded — archived.** Oracle v6 was replaced by v7 and then v8;
> the current acceptance record is `docs/2026-07-30-oracle-v8-acceptance.md`.

# Oracle v6 implementation acceptance

Status: code-complete before the directed-collection smoke.
Amended by the closeout review (see "Closeout amendments" below); statements
in the original sections hold as amended there.

## Semantic change

`ground-disc-visible-v6` changes contact attribution, not the record shape.
Full-geometry contact probes now:

1. trust the recorded world contact point;
2. invert the pose transform to obtain contact-local X/Z;
3. evaluate the canonical fitted floor plane at that X/Z;
4. sample the configured signed-normal obstacle band; and
5. transform those probes back to world coordinates for semantic assignment.

Depth attribution uses the same band and projects its representative point at
the band midpoint. `conseq.v8`, `egoconseq.qa.v13`, and
`visible-ground-disc.v14` are unchanged. Readers reject oracle-v5 records and
no compatibility adapter exists.

## Semantic cache v2

`semantic-surface.v2` cache identity includes backend/format, resolved source
paths with byte size and nanosecond mtime, decode parameters, and the cache
schema. Canonical JSON is SHA-256 hashed into the filename and embedded in the
NPZ. Cache writes use fsync followed by atomic replace. A missing, corrupt, or
mismatched metadata record is a cache miss.

## Collection durability

Every shard writes `collection_funnel.json` after each scene. It independently
records record/oracle versions, code revision and dirty state, config and run
contract digests, stage counters, skip reasons, coarse position/heading
coverage, and per-scene deltas. Resume preserves prior counters. SIGINT marks
the shard interrupted and returns 130. The controller signals every active
child process group and cannot run candidate generation after interruption.

## Sensitivity checks observed

The new tests were run against the old behavior before implementation:

- missing world-to-local inverse: inverse-transform test failed;
- root-Y contact probes on a tilted floor: canonical-floor probe test failed;
- depth projection at `GROUND_ORACLE_HEIGHT_M / 2`: midpoint test failed;
- oracle v5 current: stale-v5 reader test failed to raise;
- fixed semantic-cache filename: source/parameter invalidation tests failed;
- unsaved argparse parser: invalid CLI test raised `NameError`;
- Q4 clearance without epsilon: boundary test raised;
- interrupted controller without a stage guard: generation-blocking contract
  was absent.

The targeted suites and the full deterministic suite passed after the changes.
Real-backend acceptance still requires the preregistered 20-scene smoke and a
clean-revision funnel report; this document does not claim those results early.

## Closeout amendments (2026-07-30 review round)

A three-way audit of the v6 diff produced regression-tested corrections; all
are sampler/statistics/report-side and none changes public physical GT, so
the oracle contract stays `ground-disc-visible-v6`.

- **Funnel resume counters.** `restore_counters` previously used
  `Counter.update` (addition); every resume inflated `pool_L*` /
  `directed_pool_L*` by one full action bank. Restore is now assignment and
  idempotent (nonempty double-restore test).
- **Q8 FOV proposal keeps both flip directions.** The reference-set frustum
  test is monotone in FOV at fixed resolution, so it can never predict
  narrow-only visibility; the previous proposal therefore silently dropped
  that entire witness direction and also pre-rejected on the initial-cloud
  reprojection. The proposal is now a loose necessity gate over every FOV
  pair at the shard's single anchor height, pixel-support straddle ranks the
  narrow-only direction, reprojection is a ranking feature only, and the
  rendered sibling audit remains the sole visibility authority. q8_fov /
  q8_occlusion shards collected before this fix are sampling-biased and are
  marked superseded in `docs/runs.json`.
- **Contact anchor semantics (private field).** The v6 rewrite had silently
  changed full-geometry `contact.xy` to the obstacle surface point; both
  oracles now report the disc centre at contact again, with the surface
  point in `point_3d` and attribution geometry only.
- **Depth representative point.** "Band midpoint" is now measured along the
  plane normal (`height / normal_y`), matching the full-geometry probes; the
  original sentence above claimed this before it was true on tilted floors.
- **Interruption contract covers --coverage-first.** The original
  "cannot run candidate generation after interruption" claim did not hold on
  the coverage-first path, which bypassed the staged guard, exited 0 and
  left `controller_state.json` at `running`. It now shares the same
  contract (SIGINT → `status=interrupted`, exit 130, no compile).
- **Cache-miss defences are now tested and wider.** Corrupt or
  foreign-provenance NPZ payloads at the correct digest path are covered by
  mutation tests; the miss handler also catches `BadZipFile`/`zlib.error`/
  `EOFError`, the temp file uses `mkstemp`, and the decode seed is one
  shared constant.
- **Self-labelling reports.** Evaluator output now carries
  `role=scorer_integrity_check|model_evaluation` and `headline_eligible`;
  numbers produced with `--gt-as-pred` or `--allow-incomplete` can no longer
  masquerade as benchmark results.
- **Durability.** `collection_funnel.json` and `run_meta.json` are written
  with fsync-backed atomic replace so they never lag the fsync-backed
  `records.jsonl` after power loss; an argparse exit no longer stamps the
  previous run's funnel as failed.
- **Sampling bias is on the record.** Directed proposal caps and ranking
  keys are written into each shard's run contract
  (`directed_proposal_bias`) and README; `*_proposal_capped` funnel counters
  quantify the cut.
- **Q8 direction and funnel protocol are preregistered.** FOV proposals
  round-robin the predicted narrow-only and wide-only directions, and the
  rendered audit keeps searching until both are realized or the proposal
  budget is exhausted. Occlusion reporting uses separate cumulative segments
  for matched action pairs and pair-target trials, with the expansion point
  and units recorded under `stage_funnels`.
- **The three-backend 20-scene direction pilot resolves the FOV score role.**
  On revision `be2c0c0`, realized wide-only witnesses covered 20 HM3D, 17 R2R,
  and 8 GS scenes; realized narrow-only witnesses were 0 on every backend
  (only 2 HM3D narrow-only proposals were predicted, and neither survived the
  rendered sibling audit). Under the preregistered `5 scenes x 2 backends`
  rule, wide-only is eligible for the primary Q8 FOV slice. Narrow-only remains
  diagnostic-only and its absence may not be hidden by pooling both directions.
- **The same pilot establishes Q8 occlusion feasibility, not three-backend
  support.** Realized occlusion witnesses covered 3 HM3D and 6 R2R scenes, while
  GS produced 0. These records may seed the post-fix supplement, but occlusion
  remains below the primary support requirement until the supplement is
  audited. A zero-yield GS occlusion stratum must be reported explicitly rather
  than hidden by pooling backends.
- **Unrecorded behaviour changes from the v6 round, noted here:**
  `floor_plane_pilot` quantiles switched from nearest-rank to linear
  interpolation when the shared helper was adopted — historical pilot
  numbers are not comparable; the semantic cache now requires the source
  files to exist (`resolve(strict=True)`), so caches are no longer portable
  without their sources; Q0a distance-bin edges/ids/labels are one frozen
  `config.BENCH_DISTANCE_BINS` contract.
- **Formal-source promotion is now independently authenticated.**
  `egoconseq.formal-sources.v3` accepts only completed shared-pool-v2 runs and
  requires the trusted runs registry to pin every shard's records digest,
  run-metadata digest, record count, and canonical model-input RGB set, in
  addition to the pool manifest and indexed RGB/geometry assets. The earlier
  inline oracle-v6 runs remain valid
  audit provenance but are not formal compiler inputs. Consumers open the
  fixed repository registry and compile each records/run-metadata file from
  one hashed byte snapshot, so manifests cannot redirect registry authority or
  exploit an ABA replacement between hashing and decoding.
