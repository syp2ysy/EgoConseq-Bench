# EgoConseq

[中文](README.zh-CN.md)

Predict robot action consequences from an initial egocentric RGB image. Eight tasks in three
groups: collision events (A1 whether, A2 at which action, A3 with what), spatial relations of a
marked point (A4/B2 direction, B3/B1 distance at a checkpoint/the end) and the visual outcome (C1).
The Seen and Unseen benchmarks have 5,000 and 2,000 questions, plus separate radius-pair diagnostic sets; the SFT set (~300k)
comes from the same seen scenes. See [task definitions and scoring](data/benchmark/evaluation/README.md).

## Data and setup

Code: [EgoConseq-Bench](https://github.com/syp2ysy/EgoConseq-Bench/tree/main/P_bench).
Data: [Hugging Face](https://huggingface.co/datasets/syp115/EgoConseq-Bench).
The data archive includes QA, images, SFT, source records and saved experiment artifacts;
model weights and external simulators are separate.

Restore all archive volumes into a fresh checkout, from this directory:

```bash
hf download syp115/EgoConseq-Bench --repo-type dataset --local-dir ../egoconseq-data-download
(cd ../egoconseq-data-download && sha256sum -c SHA256SUMS)
cat ../egoconseq-data-download/data.tar.zst.part-* | zstd -d | tar -xf -
```

Extraction recreates `data/` and overwrites matching paths. Configure local runtime and
source-asset paths in [.env.example](.env.example), copied to `.env.local`, then
`source .env.local`. Collection uses [environment/habitat.yml](environment/habitat.yml)
(Python 3.9); inference/evaluation use separate Python 3.10+ model environments.

| Path | Contents |
|---|---|
| `data/{seen,unseen}_meta_gt3/` | Meta data: images, GT tables of every task, C1 future views, marked target images |
| `data/benchmark/benchmark/{seen,seen_radius_pairs}/` | Seen QA JSON: main set in full and four parameter ablations; radius pairs in full only |
| `data/benchmark/benchmark/{unseen,unseen_radius_pairs}/` | Unseen QA JSON: main set in full and four parameter ablations; radius pairs in full only |
| `data/benchmark/metadata/{seen,unseen}/` | Selection, distribution report and previews |
| `data/benchmark/release.json` | Release manifest: QA sets, image root, input and file hashes |
| `data/sft/seen/` | SFT JSONL (five versions) and selection metadata |
| `data/analysis/benchmark_review/` | Interactive case gallery and release-backed statistics |
| `weights/` | Symlink to local model checkpoints in `/data/zhangshan/weights/` |

QA image paths are relative to the GT3 meta root declared in each release; images are never copied. The SFT excludes
the benchmark and diagnostic observations and every observation within 0.5 m of them.

## Infer and evaluate

- [Inference](data/benchmark/inference/README.md): model selection, environments and parameter ablations.
  Use `--benchmark-root data/benchmark --sets seen` (or `seen_radius_pairs`); the Unseen
  release is `--benchmark-root data/benchmark --sets unseen` (or `unseen_radius_pairs`).
- [Evaluation](data/benchmark/evaluation/README.md): score saved responses and export comparison tables;
  the radius-pair set reports item and pair accuracy separately from the main set.

## Build and train

Meta data (observations and GT of every task, Seen and Unseen alike) is collected by
[scripts/collect_meta.sh](scripts/collect_meta.sh) with [pipeline/meta](pipeline/meta).
The benchmark, the radius-pair set and (Seen) the SFT are then built from it by [post_QA/build](post_QA/build)
following a spec (rules and offline image-review commands:
[BENCHMARK_IMPROVEMENTS.md](BENCHMARK_IMPROVEMENTS.md)). The current release uses GT3 and a frozen image-review table. Seen and Unseen share one benchmark root and retain separate selection metadata.

```bash
# Check a draft before replacing the current published partition.
"$EGOCONSEQ_HABITAT_PYTHON" -m post_QA.build --spec post_QA/specs/seen.json --draft --sft-total 20000
"$EGOCONSEQ_HABITAT_PYTHON" -m post_QA.build --spec post_QA/specs/seen.json    # data/benchmark, data/sft/seen
"$EGOCONSEQ_HABITAT_PYTHON" -m post_QA.build --spec post_QA/specs/unseen.json  # data/benchmark
```

Training launchers use ms-swift 3.12.x, default to four GPUs, read `data/sft/seen/` (`SFT_ROOT`)
and images from `data/seen_meta_gt3/` (`ROOT_IMAGE_DIR`).
Set `SFT_PYTHON` and optionally `SFT_MODEL` for the training host; weights are loaded offline by default.

```bash
bash scripts/train_sft_full.sh
# Alternative: LoRA
bash scripts/train_sft_lora.sh
```

Full SFT accepts `SFT_VARIANT=full/no_height/no_radius/no_fov/no_parameters`.
Training defaults and output paths are defined in the [full SFT](scripts/train_sft_full.sh)
and [LoRA](scripts/train_sft_lora.sh) launchers.

Browse the current cases and statistics at [the local review page](http://localhost:8765/analysis/benchmark_review/).
See [the page guide](scripts/review_ui/README.md) to regenerate it from another release.
