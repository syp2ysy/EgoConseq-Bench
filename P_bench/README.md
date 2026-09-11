# EgoConseq

## Code and data distribution

The current code is in [EgoConseq-Bench/P_bench](https://github.com/syp2ysy/EgoConseq-Bench/tree/main/P_bench).
The full `data/` snapshot is distributed separately through
[syp115/EgoConseq-Bench](https://huggingface.co/datasets/syp115/EgoConseq-Bench).
**Upload status:** the data upload is pending; use the commands below after all volumes are available.

It includes Benchmark, SFT, source records, saved responses and evaluation artifacts;
model weights and external simulator/source-dataset installations are not bundled.
The snapshot preserves internal symbolic links and hard links.

From this `P_bench/` directory, download and restore the archive volumes:

```bash
hf auth login  # required when accessing the private dataset repository
hf download syp115/EgoConseq-Bench --repo-type dataset \
  --include 'data.tar.zst.part-*' SHA256SUMS archive_manifest.json \
  --local-dir ../egoconseq-data-download
(cd ../egoconseq-data-download && sha256sum -c SHA256SUMS)
cat ../egoconseq-data-download/data.tar.zst.part-* | zstd -d | tar -xf -
```

Restore into a fresh checkout: extraction recreates `data/` and overwrites matching
paths. All volumes are required. See the dataset card for the exact archive size
and inventory. Copy `.env.example` to `.env.local`, set paths to your installed
runtimes and source assets, then `source .env.local`. The model-specific environment
requirements are documented in the inference README below. Collection metadata may
retain original absolute source paths; recollection requires configuring the
corresponding external source assets.

The collection environment in `environment/habitat.yml` uses Python 3.9;
inference and LLM evaluation require Python 3.10 or later and their own model
runtimes. Run collection/build tests in the Habitat environment, and judge tests
in a Python 3.10+ environment with NumPy and pytest:

```bash
"${EGOCONSEQ_HABITAT_PYTHON:-python}" -m pytest --ignore=tests/test_pl_judge_evaluation.py
"${EVAL_PYTHON:-python}" -m pytest tests/test_pl_judge_evaluation.py
```

Editable statistics are generated with
`python visualization/plot_benchmark_figures.py` after restoring the data;
the saved SVG/PDF/PNG figures and counts are in `output/figures/paper_20260911/`.

Predict robot action consequences from an initial egocentric RGB image.
The current seen catalog contains **58,557 compact records**: B1K 11,750,
GS 24,569, and R2R 22,238. B1K expansion was stopped at the user's request;
accepted records are retained and catalog metadata is sealed.

## Data and outputs

```text
data/metadata/train/seen_updates/current/   existing records and source images
data/benchmark/
  benchmark/seen/{QA*.json,images/}          4,999 QA per input variant, shared images
  benchmark/unseen/{QA*.json,images/}        2,000 QA per input variant, shared images
  metadata/frozen.json                      frozen hashes and all 7,000 SFT exclusions
  metadata/{seen,unseen}/                   record_index.json and report.json
  inference/                               Qwen/Cosmos code and launch scripts
    responses/                             model responses for the current saved QA
    logs/                                  runtime logs
  index.html                               one combined browser with All/Seen/Unseen filters
```

Benchmark compilation never rewrites records, resamples poses, or changes
radii, camera heights, FOV, actions or GT. One record supplies one benchmark QA.
**Keep all original 5,000 Seen benchmark records excluded from SFT**, including the withdrawn A3 question's record.
Unseen records are benchmark-only and never enter SFT. The combined browser uses
All / Seen / Unseen filters for cases and statistics; it does not merge or rewrite
the two QA files, indices or images. The B3 update also checks DINO similarity
across the combined 7,000 initial images, not just within each split.

After the authorized A3 withdrawal, the frozen release has 4,999 Seen + 2,000
Unseen QA. The original 7,000 record exclusions remain unchanged. The 40 audited
A3 case outputs and their 1 benchmark / 16 SFT questions were removed; all other
record fields and retained QA are unchanged. Directory names describe splits, not
sample counts. `metadata/frozen.json` binds both QA files, private
indices and the sorted image-path/digest lists, and contains all excluded record
UIDs. It does not duplicate assets. Do not reselect this release or change GT;
wording revisions require explicit authorization and updated frozen hashes.
HTML presentation may still be regenerated. `freeze --root <benchmark-root>` seals
a future release once and refreshes its template-use statistics from the actual QA.

Benchmark and SFT share `post_QA/templates.py::SYSTEM_PROMPT`
(`abc1-prompts-v8-clear-questions`). The system defines the mobile-robot role,
sequential forward/turn motions, circular collision footprint and camera alignment.
It says "Follow the listed motions exactly, without changing the path" and
"Assume continuous ground." Body radius, camera optical-center height and the
horizontal/vertical fields of view appear in the user message's Configuration block.
All eight tasks have ten concise question templates, shared with SFT.
Use `refresh-prompts --keep-templates` to synchronize wording while preserving
template assignments, images, GT and selections; hash transitions are recorded
in `frozen.json` under `prompt_updates`.

Camera height is the configured vertical offset relative to the robot's local
ground reference: exactly **0.5 / 1.0 / 1.5 m**, read directly from
`sensor.nominal_camera_offset_m`. It is not a post-hoc distance to the scene mesh
or a fitted plane. `repair-parameters` synchronizes this setting and its wording
across frozen indices, Benchmark, SFT (including existing ablation exports) and
HTML. Source records, images, poses, actions, radii, FOV and GT are not changed.
The latest summary replaces `data/benchmark/metadata/parameter_audit.json` in place.

Regenerate the combined browser from the saved QA and metadata with:

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/build_seen_benchmark.py browser
```

Serve the browser with compressed HTML and in-memory list thumbnails:

```bash
/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python scripts/serve_benchmark.py --port 8000
```

Open `http://localhost:8000/`, or add
`#view=cases&task=B3` to show B3 directly. Opening a case displays the saved
original images; thumbnails do not change benchmark or inference inputs and
create no disk cache. The page also opens directly as a file, using original images.

## Evaluate saved responses

```bash
bash data/benchmark/evaluation/run_eval.sh \
  --responses data/benchmark/inference/responses/cosmos3-edge_thinking_off_response.json
```

Except for A3, the evaluator extracts explicit answers with regex and sends longer
answers to a text LLM without GT, then applies fixed scoring rules, including distance
accuracy at **±0.25 m and ±0.5 m**. A3 uses the LLM to judge each answer's semantic
agreement with GT and reports overall A3 accuracy.

`--rules-only` runs without an LLM; rerun the command without this flag to finish.
The default extractor is cached Qwen3-8B. To use an existing API, pass
`--extractor-model MODEL --base-url URL` with `INFER_API_KEY`.
Per-item answers and independent score summaries are saved under
`data/benchmark/evaluation/results/<student>/`.
See [evaluation rules and output format](data/benchmark/evaluation/README.md).

## Compile

```bash
HABITAT_PY=/home/zhangshan/miniconda3/envs/qwen3vl_habitat/bin/python

$HABITAT_PY scripts/build_seen_benchmark.py compile \\
  --output outputs/benchmark-build/new_run --seed 20260906

$HABITAT_PY scripts/build_seen_benchmark.py compile-sft \\
  --output data/sft/seen_v2 \\
  --exclude-index data/benchmark/metadata/frozen.json

```

Compiler working bundles belong in `outputs/benchmark-build/`, separate from the
frozen public package; do not compile over the current release.

The compiler scans structured records once, preserves alternatives by action
length, and expands stratified candidate batches as needed. It screens black
holes/blank or blurred images and loads pretrained DINOv2-S/14 for full-view
features. DINO cosine exclusion is global across selected initial images;
features stay in memory. C1 endpoint images receive quality checks without
requiring its four answer options to be semantically dissimilar.

A4/B1/B2/B3 use one unnumbered red dot with white/black outlines, centered on the
original fixed surface point. The following command only regenerates HTML from
saved QA and metadata; it does not redraw markers, change prompts, reselect QA
or rerun DINO:

```bash
$HABITAT_PY scripts/build_seen_benchmark.py browser --root data/benchmark
```

Task totals are in [seen_5000_v1.json](post_QA/specs/seen_5000_v1.json).
Lengths are allocated as evenly as the image-qualified supply allows: L1–L6
for ordinary tasks and C1, L3–L6 for A2. A2 has no L3 Turn-first; ordinal/rank
balance is conditional on length and start. Every supported dataset/task
has at least 30% Turn-first. Scene and action/answer diversity guide tie-breaking.
Exact achieved distributions and nearest image pairs are in the report/browser.

### B3: intermediate camera-to-point distance

The B3 release was derived from saved geometry, without recollecting or rewriting
records. It kept all 5,000 Seen record IDs
(and therefore the SFT exclusion set), reassigns 600 Seen and 250 Unseen QA to
B3, and replaces only cross-split visually similar Unseen observations.

| Task | Seen | Unseen | Total |
|---|---:|---:|---:|
| A1 | 600 | 250 | 850 |
| A2 | 800 | 250 | 1,050 |
| A3 | 599 | 250 | 849 |
| A4 | 600 | 250 | 850 |
| B1 | 600 | 250 | 850 |
| B2 | 600 | 250 | 850 |
| B3 | 600 | 250 | 850 |
| C1 | 600 | 250 | 850 |

B3 uses only B1K/R2R, split equally. All earlier actions are completed;
the query is at 25%, 50% or 75% of the **specified Forward action's distance**.
GT is 3D Euclidean distance from that moment's camera optical center to the
fixed surface point, not travel distance or endpoint distance. The full action
sequence is certified safe. Pose, height, FOV and the original question's
body radius are retained when a record is reassigned.

The incremental allocation balances B3 lengths, progress fractions and Forward
ordinals, and keeps at least 30% Turn-first per dataset. A2 retains conditional
ordinal/rank balance with adaptive length counts. Distribution details and
changed IDs are recorded in each split's `report.json`; features are never saved.
Ordinary `compile` supports B3 with length/start balancing. The completed one-off
B3 migration and its dedicated tests have been retired. SFT is rebuilt separately
as `seen_v2`. The table above includes the subsequent one-question A3 withdrawal.
New evaluations must match responses to the current saved QA hashes.

## Model responses (no scoring)

Each split keeps `QA.json` as full input plus `QA_no_radius.json`, `QA_no_height.json`,
`QA_no_fov.json` and `QA_no_parameters.json`. These delete only the corresponding
saved Configuration lines (both HFOV/VFOV for `no_fov`), exactly like SFT ablations.
System prompts, questions/actions, GT, IDs and image references are otherwise unchanged.
Use `run.sh <model> --variant no_height`, for example; responses are named by variant.
Regenerate these JSON files after an authorized full-QA update with
`scripts/build_seen_benchmark.py export-benchmark --root data/benchmark`.

Each model family has its own `infer_*.py` in `data/benchmark/inference/`.
Only QA input, HTTP transport, response storage and resume logic live in `common.py`.
A single launcher supports 15 configurations, including Qwen baselines and the
selected embodied/spatial understanding models:

```bash
bash data/benchmark/inference/run.sh list
bash data/benchmark/inference/run.sh all --dry-run
bash data/benchmark/inference/run.sh qwen3vl-4b --limit-per-task 1
bash data/benchmark/inference/run.sh all
```

`all` runs models sequentially and resumes existing responses. No model is loaded
for `--dry-run`. Full runs read all 6,999 QA; `--limit-per-task 1` selects 16 QA
across both splits, including C1. Checkpoint downloads share the Hugging Face cache.
Qwen-family models use Transformers, SenseNova-SI uses vLLM, and SpatioLM uses
its official custom spatial model. Cosmos3-Edge uses the **reasoner**, not Policy-DROID.
See [inference usage and environment requirements](data/benchmark/inference/README.md).
The five cached small models can run sequentially, followed by a shared Qwen3-8B judge:

```bash
bash data/benchmark/inference/run.sh small --cached --serve --evaluate
```

`small` selects Qwen3-VL-4B, RoboBrain2.5-4B, RoboInter-3B, RynnBrain1.1-2B and
Cosmos3-Edge. The queue uses v8 QA, pins cached checkpoint revisions and performs
no downloads. Progress is recorded in `data/benchmark/inference/pipeline_status.json`;
responses and evaluation results are checked against the source QA hashes before resume.

Input is deliberately unchanged:

- **System:** exactly the stored `system` message, defining the robot role,
  sequential motion, in-place turns, circular footprint and camera alignment,
  plus the shared continuous-ground assumption
  for all tasks. No inference-only prompt is appended.
- **User:** the complete stored text, including radius, camera height,
  HFOV/VFOV, numbered actions, task question and any direction convention.
- **Images:** resolve paths relative to that split's `QA.json` and
  replace each `<image>` in place with a native image content block.
  A1–B3 receive one image, including the existing marker for A4/B1/B2/B3.
  C1 receives **initial image → question and A label → A image → B label →
  B image → C label → C image → D label → D image** in a single user message.
  Images are not tiled, relabeled, or reordered. Resizing uses the model's
  own processor. Native-adapter settings are recorded in the output;
  API server launch settings and runtime messages are retained in its log.
- **GT:** the stored `assistant` message is excluded from model input.

Each task has ten concise, equivalent question templates and its own answer format.
A2/A3 explicitly state that a collision occurs; A2 uses the full action list's
numbering. A4/B1/B2/B3/C1 state that the sequence is collision-free. Only C1 asks
for A/B/C/D. The prompt revision changes no images, geometry, task IDs or GT.
A4/B3 show the queried action and distance percentage on a separate `Query moment`
line. Distance questions specify 3D straight-line distance from the camera optical
center; direction questions retain the original 24-class boundaries.

For an explicitly authorized wording update, synchronize the frozen Benchmark,
SFT (including existing ablation exports), template metadata and HTML in place:

```bash
$HABITAT_PY scripts/build_seen_benchmark.py refresh-prompts --keep-templates
```

This reads indexed inputs, not source records or images. It preserves existing
template assignments, QA IDs, GT and image order, and refreshes
the frozen hashes. Temporary staging files are removed after publication.
Responses produced from earlier prompts remain earlier-version results; rerun
inference before reporting performance on the new prompts.

Outputs are `data/benchmark/inference/responses/<alias>_response.json`, with `config`,
model/processor metadata, and `predictions`. Each prediction contains `split`,
`id`, `dataset`, `task_id`, the unedited model `answer`, and `finish_reason`
(`stop` or `length`; SpatioLM's text-only chat interface reports `unknown`).
The script builds
the JSON; the VLM is not asked to generate IDs or a JSON wrapper. Generation
is greedy with a default limit of 4,096 new tokens (`--max-new-tokens`).
Thinking mode and smoke runs have separate filenames; requested template options
are saved without rewriting the benchmark prompt.

Each completed batch updates the same JSON. Running the same command resumes
by `(split, id)`; use a different `--output` for a different model, input JSON
or generation setting. Do not run two writers against the same output file.
No images, GT or records are copied, and no scoring or LLM judging is performed.
For offline use, pass the local snapshot directory to `--model`; this also avoids
the tokenizer metadata request made by Transformers 4.57.3 for cached model IDs.

## SFT (training only)

`seen_v2` is rebuilt from the current Seen records and covers all eight tasks,
including B3: **270,880 QA from 53,553 records, with 36,709 B3 examples**.
The exclusion input is the frozen 7,000-record benchmark list;
Unseen records never enter training. Exact coverage, task counts, unreadable
source paths and residual imbalance are in `data/sft/seen_v2/metadata/summary.json`.
Rank-balanced A2 has less supply; scarce questions are not duplicated to force
equal task counts. One previously known unreadable GS initial image is skipped.
The superseded `seen_v1` has been removed after validating the new export.

The export excludes extremely dark views: at least 90% of native pixels have
all RGB channels at most 8. Cleanup removed 39 QA and 96 unreferenced export
images, synchronized all five JSONL views and canonical A3 names, and left
source records/assets intact. No refill was needed (0.0144% of QA removed).
The compact audit is `metadata/quality_cleanup.json`. For an existing export,
`python scripts/build_seen_benchmark.py clean-sft` applies this same cleanup;
future `compile-sft` exports already skip these dark initial/C1 views.

`compile-sft` excludes the benchmark's entire records, covers every remaining
record with an eligible QA, then supplements less frequent tasks. It selects
at most two distinct QA per record/task; it does not export every surface point
or duplicate questions with different wording. Partial-task records are valid.
Task targets follow the smallest capped task supply, not a fixed dataset size.
A1 balances collision/safe within starts; A2 balances distance ranks conditional
on dataset/length/start. Scarce supply is reported instead of failing the build.
Lengths, both starts, categories and the 24 directions are sampled across the
available supply. This is not the benchmark's exact quota system.
B3 shares the benchmark's saved-geometry GT calculation and safe-sequence rule;
GS supplies neither B1 nor B3. Sampling rotates distance bins, action lengths,
starts, Forward ordinals and 25%/50%/75% checkpoints. Different checkpoints are
distinct questions but still share the two-QA record/task cap.

```text
data/sft/seen_v2/
  images/{b1k,gs,r2r}/       shared original, marked and terminal images
  json/full.jsonl           independent messages + images training examples
  metadata/selection.jsonl  frozen record/case/point, template, inputs and GT
  metadata/summary.json     coverage, source bindings and actual distributions
```

Images are hardlinked where possible; a marked image is rendered once per
record/point. No DINO, new oracle checks, recollection, or validation/test split.
Source records and the published benchmark are never modified.
Unreadable initial images exclude only their records; a broken terminal image
excludes the affected C1 questions, not other usable tasks. Exact paths and
achieved coverage are reported in `metadata/summary.json`.

Derive parameter ablations **directly from the saved `json/full.jsonl`**, sharing
the same images, QA IDs, order, system prompt, wording, C1 options and GT:

```bash
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params height
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params radius
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params fov
$HABITAT_PY scripts/build_seen_benchmark.py export-sft --root data/sft/seen_v2 --hide-params height radius fov
```

These produce `json/no_height.jsonl`, `no_radius.jsonl`, `no_fov.jsonl` and
`no_parameters.jsonl`. Only the corresponding configuration lines are removed;
`no_parameters` removes the entire `Configuration` block. FOV means both HFOV and
VFOV. Action distances, turns, the system's circular-body convention and A4/B3
checkpoint fractions remain unchanged. No templates are reselected or rerendered.
Combined subsets also work. Every SFT `images` entry is relative to the SFT
dataset root (`images/...`), including ablations. Move `images/`, `json/` and
`metadata/` together; no path rewriting or re-export is needed after a move.
Benchmark image paths are relative to the directory containing its `qa.json`.

The output follows the [ms-swift multimodal dataset format](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html):
one image for A1–B3, five ordered images for C1. Both SFT launchers set the
native `ROOT_IMAGE_DIR` to the SFT root, resolving `images/...` without rewriting
JSON or registering a custom dataset. Saved system/user messages and assistant
GT are used directly; only assistant responses contribute to the SFT loss.
Compilation does not start training.

### Full-parameter Qwen3-VL-4B SFT (four A100s)

`scripts/train_sft_full.sh` trains the LLM, vision encoder and alignment modules;
it does not use LoRA or freeze any of these components. It reuses the environment
below, with DeepSpeed (locally installed: 0.18.3). On another server, set
`SFT_PYTHON` to that server's compatible Python, and use `SFT_MODEL` for a local
base-model directory if needed. Default model loading is offline; set
`HF_HUB_OFFLINE=0` when the weights need downloading.

| Setting | Default and rationale |
|---|---|
| GPUs / sharding | Four A100s, BF16, native ZeRO-3; no CPU offload |
| Learning rates | LLM `1e-5`, vision `1e-6`, aligner `5e-6`; smaller visual updates to limit disruption of pretrained features |
| Effective batch | `4 GPUs × 1 example × 8 accumulation steps = 32` |
| Schedule | One epoch, cosine decay, 3% warmup; a starting configuration, not a tuned optimum |
| Optimizer | AdamW, weight decay `0.01`, gradient clipping `1.0` |
| Images / context | At most 1,024 visual tokens per image, 8,192 total tokens; C1 retains all five images |
| Memory / data | LLM and vision gradient checkpointing, SDPA, lazy image processing, no packing or validation split |
| Saving | Every 1,000 optimizer steps; retain one resumable checkpoint |

Uses the [native ms-swift 3.12 parameters](https://swift.readthedocs.io/en/v3.12/Instruction/Command-line-parameters.html).
Different module learning rates automatically select ms-swift's multimodal
optimizer grouping. Its logged `learning_rate` can be the first (vision) group,
not the LLM rate. No custom trainer, dataset adapter or DeepSpeed JSON is added.

```bash
# Run on the four-A100 training host, not a single-GPU machine.
bash scripts/train_sft_full.sh

# Same images; choose one of full/no_height/no_radius/no_fov/no_parameters.
# Each variant gets its own output directory and starts from the base model.
SFT_VARIANT=no_height bash scripts/train_sft_full.sh

# Exact continuation, including optimizer state.
bash scripts/train_sft_full.sh \
  --resume_from_checkpoint outputs/sft/qwen3vl-4b-full-finetune/full/checkpoint-1000
```

Outputs go to `outputs/sft/qwen3vl-4b-full-finetune/<variant>/`, separate from
the historical LoRA output directory. Checkpoints include optimizer state and
are much larger than LoRA checkpoints; allow room for the next save before the
previous checkpoint is pruned. Changing GPU count or per-device batch also
requires adjusting accumulation to keep the effective batch at 32. This launcher
has argument-level checks, not a completed four-A100 training or memory test.

### Train Qwen3-VL with ms-swift (LoRA launcher)

Reuse the existing `qwen3vl` Conda environment, not `qwen3vl_habitat`:

```bash
conda activate qwen3vl
python -m pip install --no-cache-dir ms-swift==3.12.6 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 transformers==4.57.3 \
  peft==0.18.0 accelerate==1.12.0
bash scripts/train_sft_lora.sh
```

Pin ms-swift 3.12.6 for this PyTorch 2.5.1 environment: the 4.5.2 training
callbacks unconditionally import the newer `torch.distributed.fsdp.FSDPModule`.
The launcher therefore uses the 3.x native `--train_type lora` argument.
`full.jsonl` means all robot input parameters are present, not full-parameter
fine-tuning. The existing output directory name is kept for checkpoint continuity.

The launcher uses the cached Qwen3-VL-4B-Instruct model, four GPUs with DDP,
BF16 LoRA (rank 16, alpha 32), frozen vision/aligner, and one epoch over all
270,880 QA. Effective batch size is 32. It does not create a validation split,
evaluate the benchmark during training, precompute visual features, or pack the
whole dataset before startup. Images are loaded lazily. Checkpoints (at most two,
including optimizer state for resuming), TensorBoard events, and native training
metadata stay under `outputs/sft/qwen3vl-4b-full/`.

Run detached from the project root:

```bash
mkdir -p outputs/sft/qwen3vl-4b-full
nohup bash scripts/train_sft_lora.sh > outputs/sft/qwen3vl-4b-full/train.log 2>&1 < /dev/null &
```

Append native ms-swift arguments to override defaults, e.g.
`--resume_from_checkpoint outputs/sft/qwen3vl-4b-full/checkpoint-1000`.
For parameter ablations, first export the matching JSONL, then change both
`--dataset` and `--output_dir`; each ablation starts from the same base weights.
`SFT_PYTHON` and `SFT_MODEL` override the interpreter and model. Model loading is
offline by default; set `HF_HUB_OFFLINE=0` only when downloading a different model.

## Tasks and collection

| Task | Question |
|---|---|
| A1 | Does the sequence collide? |
| A2 | Which action first collides? |
| A3 | Which initially visible object/category is contacted first? |
| A4 | Where is the marked surface point during a Forward action? |
| B1 | How far is the endpoint camera from the marked point? |
| B2 | Where is the marked point at the endpoint? |
| B3 | How far is the camera from the marked point during a specified Forward action? |
| C1 | Which image is the true future view? |

GS excludes A3/B1/B3. A4/B1/B2/B3 and every C1 family member use safe, complete
sequences. A3 currently retains the existing first-contact definition; it is
not a newly defined safe-action question. A2 evaluation reports start-conditioned
accuracy because alternating sequences have a structural collision-index parity prior.

Unseen scenes use `scripts/collect_unseen_records.py`, which invokes the existing
collector and writes `abc1.record.v3`, without compiling a benchmark. R2R uses
18 official test scenes; GS uses 9 official InteriorGS val scenes. B1K uses
10 project-held-out scenes, **not an official test split** (its native catalog
still declares train).

```bash
$HABITAT_PY scripts/collect_unseen_records.py \
  --data-root /home/zhangshan/syp/datasets \
  --b1k-source-manifest /home/zhangshan/syp/datasets/behavior-1k-v3.9.1/pbench-abc1-task4/catalog-authority-audit-v2-20260812/audit-derived-source-manifest.json \
  --b1k-python /home/zhangshan/miniconda3/envs/behavior/bin/python
```

GPUs 0/1 own disjoint B1K scenes; GPU 2 runs R2R and GPU 3 runs GS. Each worker
finishes one scene's pose budget before loading the next. Default targets are
3000/3000/2000 records (up to 8013 after per-scene rounding); actual accepted
counts may be lower. Outputs live under
`data/metadata/test/unseen/<dataset>/<worker>/<scene>/`. The final root
`manifest.json` references these original shards, without copying records or
images. `collection.json` tracks progress; rerunning the same command resumes.
Unseen records are evaluation-only and never enter seen SFT.

Existing records use `plan/collect/dry-run`,
`update-records` or `refine-surfaces`; workers finish all assigned records of
one scene before loading the next. These collection paths remain separate from
benchmark compilation. See [implementation notes](BENCHMARK_IMPROVEMENTS.md).
