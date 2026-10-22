# QA build

## Current benchmark — 2026-10-16

The 2026-10-17 directory cleanup restores one current root: `data/benchmark`.
QA is in `benchmark/<set>/`, selection metadata in `metadata/{seen,unseen}/`,
raw model responses in `inference/responses/`, and final scores in `evaluation/results/`.
`release.json` indexes both partitions and their separate image roots.
Only paths changed: questions, image references, responses and scores are preserved.
Original manifests and the path mapping are archived in `data/archive/benchmark_layout_20261017/`.
Historical protocol identifiers below describe provenance, not additional active benchmarks.

The GT3 rebuild is complete. The seen main set has 5,000 questions over all 143 scenes
(R2R 61, GS 42, B1K 40); the unseen main set has 2,000 over all 60 scenes (29, 21, 10).
The separate radius diagnostics contain 300 seen pairs and 150 unseen pairs. SFT contains
299,999 questions from 147,051 observations over all 143 seen scenes: R2R 212,390,
GS 62,433 and B1K 25,176. No hard-quota shortfalls or blocking validation problems remain.
The benchmark is in `data/benchmark` (all four sets); SFT is in `data/sft/seen`.

Independent verification checked the output and source hashes, selected/public IDs, explicit image
review keeps, dataset/task quotas, pair sizes, observation isolation, seen/unseen scene separation
and the SFT maximum of three items per observation. Evidence and distribution summaries are in
`data/analysis/v11_release_check/`. The refreshed static review page is
`data/analysis/benchmark_review/`, served at `http://localhost:8765/analysis/benchmark_review/`.
It separates the 7,000 main questions from 450 diagnostic pairs and uses the established task names.

Soft targets remain approximate: seen representation is 38.24/40.18/21.58 and unseen
38.10/39.25/22.65 (atomic/merged/coarse). SFT is 55.55/28.82/15.62; B1K's coarse share is 43.95%.
The ten most frequent SFT programs account for 40.43%, versus 40.87% in v10. The share of SFT
observations with a close neighbour is 57.7%, versus the corrected v10 figure of 60.1%.
These deviations are reported, not described as proven supply limits or perfect diversity.
Image screening does not certify artifact-free inputs: normal close-ups and local scan defects
remain permitted, and the page supports further human review.

Direction-capacity retries now continue only while the integer capacity bounds tighten; they stop
when no new constraint is learned, rather than after an arbitrary three attempts. Unseen uses seed
20261016, with the same quotas and thresholds. Its frozen review table includes 6,430 additional
candidate-option scores and one explicit assistant visual-audit rejection of a severe GS fog/splat
image (`gs-b9736c95903e197e296c`), after which unseen was rebuilt. This is recorded separately from
human labels. Seen retains its own already-frozen table. No simulator or GT rerun was performed.

### Intended use and action coverage

The benchmark diagnoses action consequences given an initial view and a supplied action sequence:
collision events, spatial relations and visual outcomes. Its deliberate balancing supports comparison
within this task scope; its scores do not estimate an arbitrary navigation policy's on-policy performance.
SFT trains the same action-consequence QA abilities, rather than instruction-following action imitation.
The action prior comes from R2R/RxR training trajectories, followed by proposal restrictions and selection
balancing. It is not the distribution of a current policy's complete generated action chunks.
In particular, the fine generator requires a forward action, excludes five consecutive forward atoms,
limits consecutive turn atoms to three, and uses 1–6 atoms per program (0.25 m / 15° per atom).
The published SFT has no pure five- or seven-step forward programs (1.25 m / 1.75 m), nor continuous
same-direction turns beyond 45°. This does not mean all odd-length programs are absent. These coverage
limits bound transfer claims; they do not invalidate the benchmark's stated diagnostic scope.
The SFT view redundancy and B1K representation imbalance are reported above. Image screening and passing
structural checks do not establish human answerability or the absence of all rendering defects.
An optional GS image spot-check is at `http://localhost:8765/analysis/gs_v11_spotcheck_20261016/`:
100 unique original images per split, seed 20261016; its sampling manifest is stored beside the page.

## Build inputs

The release uses [seen.json](post_QA/specs/seen.json) and
[unseen.json](post_QA/specs/unseen.json), the `*_meta_gt3` inputs and new output directories.
Existing image scores are reused; the benchmark and SFT are jointly selected.
No simulator runs and no GT is recomputed by [post_QA/build](post_QA/build).

## Tasks and wording

Released v8 wording (80 templates, 10 per task), with the v2 motion model in the system prompt: the
start ground height is kept for the whole motion and the body is a cylinder from 5 cm above it to
10 cm above the camera. A2/A3 are asked given that a collision occurs and A4/B1/B2/B3/C1 given a
collision-free sequence; only A1 asks whether a collision occurs. GS has A1, A2, A4 and B2.
Each item uses one length unit for radius, camera height, forwards and the distance answer;
metres and centimetres are half and half (by items) in every set, dataset and task.
The v11 wording removes the reference to a "stated" radius from the shared system prompt, uses the
starting ground consistently for camera height, fixes degrees, and states the valid start, rotation,
first-contact and command-end conventions. C1 states that one candidate is correct. Task definitions,
direction sectors and distance precision (0.01 m = 1 cm) are unchanged.

## Steps

1. **Candidates.** Rows of status-ok `train_seen` observations. Image checks on unmarked images: a
   `dark_pixel_fraction` (every channel below 12) above 5% (benchmark) or 20% (SFT), or a confirmed
   rejection in the hash-keyed review CSV (see below). All three datasets require explicitly kept
   images. Missing required reviews are coverage gaps. Edge density is recorded only;
   close-ups and low texture are valid navigation views. C1 is checked on all five images; one failing
   image drops the whole family. Benchmark-only filters: A4/B2 at least 2° from a direction boundary; B1/B3 distance
   change at least max(0.30 m, 10% of the initial distance) and distance at least 0.05 m. A2 needs two
   public forward commands. Every row carries its action features (command count, total forward
   distance, total turn, turn side, first command) and, for A2, the forward-count group (2 or 3+),
   the position of the colliding forward and, when the program has a unique longest forward (whatever
   the answer), whether the colliding forward is it. The tables are trimmed: within each observation,
   benchmark flag and answer group, one row is kept for every value of every other balanced field
   (first command, actions, radius, checkpoint fraction, distance change). A1 rows of one program at
   adjacent radii with opposite answers form radius pairs, trimmed the same way.
2. **Benchmark (5,000) and radius pairs (100 per dataset).** Integer quotas per dataset and task
   (A2 800, others 600). Hard cells: A1 collision 50/50; A2 per forward-count group (sizes by supply),
   the colliding forward first/last (2 forwards) or first/middle/last (3+) in equal numbers; A4/B2 25
   per joint direction label over all datasets; at least 30% of the programs of every dataset and task
   (and of the pairs) start with a turn: these are picked first in every cell, and turn picks a cell
   cannot make go to the other cells of its dataset and task. Soft targets: representation 40/40/20
   (atomic/merged/coarse) with the command count, forward distance, turn amount and turn side each
   spread evenly within a representation (the coarse programs are the long ones); radius, camera
   height × FOV; A2 colliding forward the unique longest at chance rate, 1/F for F forwards (programs
   whose longest forward is tied do not count); A3 categories at their natural share of observations, none above 15% (rare categories
   stay); B distance bins and change; A4/B3 checkpoint fractions. One question per observation, unused
   scenes first, a scene cap (exceeded only by a cell that cannot be filled otherwise), no two
   observations within 0.5 m and 45° on one floor, DINOv2 cosine below 0.90 to the chosen observations
   of the same dataset and scene in a pose-near view (same floor, within 2 m and 45°).
3. **SFT (~300k).** The evaluation observations (benchmark and pair set) are fixed first. Excluded,
   the first reason logged in `metadata/excluded.parquet` (observation or C1 family, reason, value,
   evaluation image): image checks; within 0.5 m of an evaluation observation (same scene and floor, any
   yaw); the same initial image; a DINOv2 cosine of at least 0.90 to a pose-near evaluation image (the
   benchmark rule; the evaluation features of the build are reused, cosines on the GPU, compared at
   full precision). C1 options are not compared: they are meant to look alike. About 37.5k per task, split over datasets by
   the observations left after the exclusions; 30% of A1 as radius pairs; representation 60/30/10; the same action and parameter targets,
   answers balanced softly (A1 collision, A2 position within each forward-count group, A4/B2 labels,
   A3 capped categories). First one
   question per observation on views at least 0.5 m / 45° apart, then up to three per observation
   (another task or program) under a neighbourhood cap. Every round balances the whole set so far (a
   pair counts as its two items, each with its radius and answer).
4. **Render and check.** Units, templates, C1 answer positions (A–D balanced) and the shown radius of
   B/A4/C1 (least used legal one) are assigned per item, independently of answers; a pair shares all
   of them except the radius. Five versions (full, no_radius, no_height, no_fov, no_parameters) come
   from one selection. Marked images are named by the source image hash, the pixel and the marker
   style and rendered on every build. The checks run on the final items and block a release: the
   items are the selected questions once each (two per pair) and every version has the same items;
   every answer, image reference and prompt line (actions, radius, height, FOV, checkpoint, premise,
   question, answer format and unit) against the meta GT and the observation; hard cells, first
   command and the A2 longest flag re-derived from the GT; benchmark gates and dark limits; quotas,
   turn-first shares, labels, units and C1 positions; image contents (initial image = meta hash,
   marked image = its rendering, C1 option = its content hash); ablations differ only by the removed
   lines; pairs differ only by the radius and answer; no observation shared across sets and none of
   the SFT near the benchmark; every SFT initial image below 0.90 to the pose-near evaluation images,
   its nearest one and the cosine recorded in the selection.

## Image filtering for v11

Filtering belongs to the existing meta-to-QA build. It does not modify meta, GT, collection or
export. An initial image rejected by the shared gate excludes all questions at that observation.
A rejected C1 option excludes the whole candidate family; options are not replaced or deduplicated.
The ordinary sampler fills quotas from the remaining candidates and reports any shortfall.

The image-review rule rejects severe, extensive reconstruction/rendering damage: broken surfaces,
missing geometry, stretched objects or nonphysical fragments/clouds. Normal close-ups, plain walls,
dark objects, ordinary blur and small local scan defects are not reasons for a defect rejection.

`Qualities.rejection()` is the single gate for initial images, C1 options and final QA validation:

- All three datasets require `keep` in the frozen table named by `image_reviews` in the spec.
- `reject` excludes the image. A missing hash is an audit-coverage gap, not a defect label.
- The existing RGB<12 dark-area limits stay at benchmark 5% and SFT 20%. These independent legacy
  limits still apply to reviewed images; they are not proof of a rendering defect.
- The user chose to exclude GS images with mean luminance >=180 and standard deviation <=15.
  Luminance is computed on the full original RGB image in float32 as 0.299R+0.587G+0.114B.
  This is an additional GS quality policy, including blank close-ups; it is not a confirmed defect
  label. It rejects 2,291 seen and 114 unseen initial images. The separate B1K near-black rule stays
  removed. Edge density is calculated only by the review/statistics page, not for admission.

The build only looks up decisions in memory. It never invokes the review model. Initial-image
dark fractions are loaded from meta into the shared cache, with no full-image decode pass.
C1 option dark fractions are computed once when needed, or prefetched in parallel; planning also
prefetches unreviewed options. Known rejected or unreviewed options require no pixel decode in
normal selection. Final QA validation uses the same cache and never permits missing reviews.
The duplicate initial-image quality/edge summaries were removed from the build report; the existing
statistics page computes both metrics for published images. Target marking checks the
source hash while reading it; final validation checks the displayed originals, marks and options.
There is no second source-hash pass for marked images.

`selection_shortfall` and candidate review gaps are reported separately, without claiming that
completing reviews must fill every quota. The release copies and hashes the frozen review CSV and
its manifest. Normal publication does not write an unused candidate-review queue.

### Existing offline scorer

`scripts/review_images.py` remains a separate utility: prepare an explicit image inventory, score
that inventory, and merge scores with human labels into a frozen table. No command starts another
command. It keeps the existing Qwen3-VL-8B prompt, preprocessing, single-token Yes/No score, vLLM
batching and protocol/hash cache. Changing a threshold reuses the scores. Explicit mandatory quality
rules override human labels; human labels override model decisions. Human uncertainty otherwise
omits the hash from the table. `prepare --rules-output` can generate the cheap exclusions alongside
an initial-image inventory. Dark fractions come from meta; only GS images need decoding for its
additional rule. `merge --rules` applies those exclusions without requiring model scores for them.

`merge --thresholds` reads a JSON mapping dataset names to numeric thresholds; every dataset in the
supplied inventories needs a value. There is no default fallback to 0.6 or dataset exemption.
The chosen thresholds are R2R 0.999, GS 0.9 and B1K 0.9, frozen in
`data/analysis/image_review/v11_full/thresholds.json`. These are operating choices, not measured
population error guarantees. Scores are rankings, not calibrated defect probabilities, and fresh
GPU inference is not promised to be bitwise identical.

The existing optional `sample` command now draws random examples from predicted keep/reject groups
and reuses the blind HTML page. `calibrate` only counts human decisions in those groups; it performs
no threshold search. Normal images in the reject group and bad images in the keep group are shown
separately, with missing and uncertain labels. They must not be pooled into a population defect rate.

```bash
# Optional small check of an already scored inventory; no model invocation.
python -m scripts.review_images sample --images <images.csv> \
  --results <cache>/results.jsonl --thresholds <thresholds.json> \
  --per-decision 50 --output <new-check-directory>
python -m scripts.review_images calibrate --sample <new-check-directory>/sample_manifest.csv \
  --labels <human-labels.csv> --report <check-report.json>

# Freeze only the supplied inventory after its rule/threshold review.
python -m scripts.review_images merge --results <cache>/results.jsonl \
  --images <initial-images.csv> <needed-option-images.csv> \
  --thresholds <thresholds.json> --human <human-labels.csv> --rules <image-rules.csv> \
  --output data/analysis/image_review/v11_calibrated/image_reviews.csv
```

For missing C1 reviews, the existing `post_QA.build --prepare-reviews` mode writes only a provisional
selection's pending-image list and report. Its missing-review exception lives in `Selector` alone;
normal selection and final validation always require keep. It cannot publish QA. No all-options
review is required in advance.

The authorized scoring run uses one vLLM process per GPU (0, 1, 2, 3), tensor parallel size 1,
input batches of 256 and separate worker caches. Prompt, image preprocessing, precision and engine
settings retain the existing protocol. After skipping cached/human-reviewed images and hard rule
exclusions, 301,330 initial images need scores. Worker scores are verified and merged before freezing
the table. C1 reviews cover only candidate option images needed for selection, including reserves;
unreviewed replacements remain blocked in the final build. The partial table made at 0.6 is not used.

## Outputs

ms-swift format (`messages` + `images`), image paths relative to the GT3 meta root declared in each
release (only the marked target images are new, under that root's `marked/`). Seen writes
`data/benchmark/benchmark/{seen,seen_radius_pairs}/QA*.json`, its `metadata/seen/`,
`data/sft/seen/` and, last, `data/benchmark/release.json` (sets, image root, input and file
hashes, the image checks and the DINOv2 weights hash); it then removes `data/qa_draft/` and the marked
images no kept version references (paths resolved against each version's image root; nothing is
removed if a version cannot be read). Drafts (`--draft`, a
subset of datasets or `--sft-total`) go to `data/qa_draft/`. `metadata/selection.parquet` holds every
item with its fields and image hashes; `metadata/report.json` gives per set: observations, scenes and
the share of close views, and per set, dataset and task: scenes, answers (all A3 categories and the
broad-label share), units, radii, cameras, representations, command counts, forward and turn bins,
turn sides, action patterns, A2 positions and longest shares per F against 1/F, B bins and changes, shortfalls,
problems and simple baselines.

## Unseen (test/unseen scenes)

Same GT and the same build as Seen, from fresh poses in all 60 test/unseen scenes (R2R 29, GS 21, B1K 10):
`scripts/collect_meta.sh <new-meta-root> propose <dataset> <poses per scene> <gpu>` writes
`<new-meta-root>/inputs` with [pipeline/meta/propose.py](pipeline/meta/propose.py), then the usual
collection and export. The proposer draws coarse programs from
`data/seen_meta_gt3/inputs/coarse_pool.json`, the coarse command lists of the frozen seen inputs. The current benchmark uses
`post_QA/specs/unseen.json` and `data/unseen_meta_gt3`; a new collection needs its own root and spec.

- **Poses.** From navigable anchor points with a uniform heading: the anchor itself, a point on an R2R
  reference path facing the next waypoint ±30° (the 11 val_unseen scenes have paths), and the starts from
  which a training-prior program reaches the anchor's 6 m forward contact with a chosen forward atom at
  20–80% of it. Kept when the tallest body starts free at 0.10 m on a support surface; no two within 0.5 m
  and 45° on one floor. GS poses are put at the height of the source floor level that covers them (the GS
  sampler can pick a small coplanar shell, such as a rug, that does not).
- **Cameras.** Per scene, poses in hashed order take the configurations in turn: 12 (4 heights × 3 FOVs),
  B1K 9 (no 1.25 m, as its seen scenes).
- **Programs.** Every pose has both granularities. Fine: a bank from the frozen
  `pipeline/assets/fine_action_prior.json` (R2R 10,819 and RxR 19,990 training trajectories, equal weight;
  sha256 `c989aba3…`), with prior draws, coverage programs and local edits, 24 programs (B1K 12); each as its
  atomic list and up to two merged lists. Coarse: 6 programs drawn from the frozen seen inputs of the dataset,
  one count per input program; a coarse list equal to one of the pose's fine lists is dropped, so a command
  list has one representation in an observation. Generated frequencies are not expected to equal the prior:
  deduplication, coverage programs, edits, validity and selection change them.
- **Benchmark** (no SFT): 2,000 items, 0.4 × the seen totals per dataset and task (R2R 796, GS 404, B1K 800;
  A2 320; A4/B2 10 per direction label), 50 radius pairs per dataset, every other rule as Seen. Hard check:
  every test/unseen scene has at least a third of its even share of the benchmark (R2R 9, GS 6, B1K 26 items;
  supply varies, e.g. two dark MP3D scenes have 46 and 71 usable images of 200), except a scene the spec lists
  as unusable with its reason.
- **Rare direction labels.** A4/B2 label slots go, rarest label first, to a dataset that keeps a spare
  observation of the label when one does; if a dataset still delivers fewer (another task or a near view took
  them), its quota for that label is capped at what it delivered and the benchmark is selected again (at most
  three times). Seen never needed either.
- **Collection notes.** A C1 family whose end view cannot be rendered (no finite depth, e.g. facing open sky)
  is dropped and counted (B1K: 11 families); an observation whose initial view has no depth is
  `render_failed` (B1K: 1). B1K renders are not bit-reproducible between runs (Isaac Sim): geometry and GT
  repeat exactly, images differ by sensor-level noise; image hashes are recorded in the meta.

## GT v3: closed-part start rule (2026-10-14)

The start rule rejected a start when the body centre lay inside a convex collision part, without checking that
the part is closed. GS splits its building shell (`SM_floorplan`) into open pieces that pass the convexity test,
so starts inside whole rooms were rejected: seen GS lost 8,720 of 110,968 observations in 19 of 36 scenes
(interior_0393 all, interior_0405 64%, interior_0103 50%), unseen interior_0191 kept 1 of 8,000 proposals.
The audit (`data/analysis/start_audit`: per scene, the part that rejects each start, its source prim or object
category, closedness, size and share of navigable points) put the seen GS rejections at 7,732 by open
building-shell pieces that enclose navigable floor, 988 unclear and none clearly correct (a closed object part
at most 3 m across with no navigable point inside); B1K parts are closed hulls and its rejections stand (41 B1K
starts under a wall hull that covers floor area are an asset mismatch between navigation and collision
geometry, not a start-rule error). A part now encloses a solid only
when it is watertight (every edge shared by exactly two faces) and convex; open parts still collide in every
sweep. GT contract `seen-meta-gt.v3`. The rule only decides whether a start is valid: observations valid under
both rules keep identical rows (verified on 2,724 unseen GS observations, images included). Of the 8,720 seen
starts, 8,682 now start free, 33 are blocked at 0.10 m and 5 stay rejected: they lie inside one closed object
part (interior_0126_840028, a 12-face box 0.75 × 0.9 × 6.57 m with no navigable point inside, "unclear" in the
audit only for being longer than 3 m).

Current meta roots (v10 QA has been retired; v11 references the GT3 roots below):
- `data/unseen_meta_gt3`: GS proposed and collected again (4,108 observations; interior_0191 200); R2R and B1K
  carried over unchanged (`python -m pipeline.meta.export --reuse`: tables copied, images hard-linked).
- `data/seen_meta_gt3`: R2R and B1K carried over; GS = the old ok observations (reused) + the 8,720 rejected
  starts and the 6 seen GS scenes the old pools never had (1,000 poses each, fine and coarse programs),
  collected alone (`propose.py inputs --missing-from data/seen_meta`) and merged with `--reuse data/seen_meta
  <re-collected root>` (an observation's rows come from the last root that has it). Result: 116,968 observations
  in all 42 seen GS scenes, 116,963 collected (`ok`) and 5 rejected as inside_solid. `ok` means collected, not
  usable: 266 have every execution in start contact and give no A1 candidate (233 as in v2, and the 33 released
  starts blocked at 0.10 m), 116,697 give A1 candidates and 85,872 A4/B2 candidates; A1, A2, A4 and B2 each have
  candidates in all 42 scenes. The supplementary collection was merged from `data/seen_meta_gt3/_add`;
  its inputs and card are preserved in `data/archive/meta_retired_20261016`.
- `reuse` writes only datasets the output does not have yet; replacing exported data goes through the export with
  its checks and commit.
Separate: GS interior_0407 has starts on low platforms (≈0.2 m) that the GS floor-level support puts the body
under, so most proposals are blocked; it keeps 108 poses.


## Provenance

`data/seen_meta`, `data/unseen_meta` and `data/seen_meta_gt3/_add` are retired; the path mapping, checksums
and completion checks are in
[data/archive/meta_retired_20261016/retirement.json](data/archive/meta_retired_20261016/retirement.json).
The frozen seen inputs that the coarse pool was computed from are checked by
`data/archive/seen_meta_inputs_20261012/SHA256SUMS` (R2R and B1K are the files in `data/seen_meta_gt3/inputs`;
GS is the archived pre-supplement file). Frozen GT3 data cards and v11 releases retain their original hashes and
historical source paths, resolved through this mapping.

Known provenance gap, predating retirement: 24 seen R2R collection records expect `scenes.jsonl`
SHA256 `b04624fc3c8e360813960bbba41cfe66a423cabaeb3bf0520792e997707211a5`.
That historical file was not recovered. The available file (hash recorded in the retirement manifest)
is preserved, without claiming it matches the historical collection input.

Inference, evaluation and table commands now default to `data/benchmark`; pass
`--benchmark-root data/benchmark --sets unseen` for unseen inference/evaluation.
Retired v10 specs remain in `data/archive/v10_retired_20261015/post_QA/specs/`.
The exact release-time code is preserved in `data/archive/v11_release_code_20261016.tar.gz`;
this maintenance cleanup does not regenerate questions or change GT.
