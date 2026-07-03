# Qwen3-VL StreamVLN VLN Project

本项目是在 StreamVLN 轨迹数据上训练和评估 Qwen3-VL 的 Vision-Language Navigation
代码库。当前主线是：在 NAVIDA-style 单轮模仿学习 baseline 上加入训练期 depth auxiliary
loss，让模型在推理时不使用 depth head 的情况下提升 R2R 导航表现。

当前最重要的结论：

```text
Decision-aligned depth supervision improves VLN with zero inference overhead.
Dense/full-frame depth supervision can hurt stop/action calibration if it is
misaligned with the navigation decision.
```

## 1. Project Status

**Current best R2R-only model**

```text
outputs/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_zero3
```

**Current best R2R val_unseen evaluation**

```text
results/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_val_unseen_best_policy/final_metrics.json
```

**Current R2R+RxR mixed model**

```text
outputs/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_zero3
```

**Current best overall R2R val_unseen evaluation**

```text
results/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_val_unseen_tok512_fixed2_sample/final_metrics.json
```

The R2R+RxR mixed depth model is currently the strongest checked result:
SR `0.5095`, SPL `0.4487`, NE `5.6161` on full R2R `val_unseen`.

## 2. Repository Layout

```text
config/
  train_r2r_zero3_navida_random_stop2.yaml       # R2R baseline train config
  train_r2r_zero3_depth_stage1.yaml              # R2R depth-head warmup
  train_r2r_zero3_depth_stage2_w05to01.yaml      # R2R best dynamic-depth run
  train_r2r_rxr_zero3_depth_stage1.yaml          # R2R+RxR mixed warmup
  train_r2r_rxr_zero3_depth_stage2_w05to01.yaml  # R2R+RxR mixed joint training
  vln_r2r.yaml                                   # Habitat R2R eval config

scripts/
  train_r2r_baseline.sh
  train_r2r_depth_stage1.sh
  train_r2r_depth_stage2.sh
  train_r2r_rxr_depth_stage1.sh
  train_r2r_rxr_depth_stage2_w05to01.sh
  eval_r2r_4gpu.sh
  eval_r2r_best_policy.sh
  extract_streamvln_depth_pose.py
  extract_r2r_depth_pose.py
  extract_scalevln_trajectory_data.py
  zero3.json

vln_baseline/
  dataset.py              # StreamVLN -> Qwen-VL single-turn samples
  collator.py             # Qwen chat template, labels, depth loading/alignment
  depth_head.py           # lightweight MLP + PixelShuffle depth readout
  losses.py               # SILog depth loss
  model.py                # Qwen3-VL loading, freezing, depth-head save/load
  actions.py              # action text parsing and primitive expansion
  eval_metrics.py         # distributed eval aggregation checks
  habitat_extensions.py   # Habitat R2R registration

train.py                  # HF Trainer + DeepSpeed training entrypoint
eval.py                   # Habitat evaluation entrypoint
outputs/                  # local checkpoints, ignored by git
results/                  # local eval outputs, ignored by git
logs/                     # local train/eval logs, ignored by git
```

## 3. Environment

Two local conda environments are used:

| Env | Use |
|---|---|
| `qwen3vl` | Training with Qwen3-VL, DeepSpeed ZeRO-3, bf16, FlashAttention 2 |
| `qwen3vl_habitat` | Habitat-based R2R evaluation |

Basic training environment:

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl
pip install -r requirements.txt
```

`requirements.txt` covers the Python side:

```text
torch
transformers
accelerate
deepspeed
pillow
numpy
scipy
PyYAML
fastdtw
flash-attn
safetensors
pytest
```

Habitat is installed separately in `qwen3vl_habitat`. The evaluator expects Habitat-Lab,
Habitat-Sim, Matterport3D/R2R-CE assets, and the local R2R config in `config/vln_r2r.yaml`.
The local evaluation/replay environment currently uses:

```text
habitat-lab 0.2.4
habitat-sim 0.2.4
hydra-core 1.3.2
omegaconf 2.3.0
attrs 25.3.0
numpy 1.26.4
torch 2.5.1+cu124
```

Important runtime assumptions:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
```

## 4. Data Dependencies

The checked-in configs currently use absolute local paths. If another machine uses different
locations, update the YAML files under `config/`.

### Official Download Sources

There are three different data layers. Do not mix them up:

| Layer | What it contains | Source | Local target used here |
|---|---|---|---|
| StreamVLN trajectory data | StreamVLN annotations and pre-collected RGB trajectories for R2R/RxR/EnvDrop plus ScaleVLN metadata | Hugging Face dataset `cywan/StreamVLN-Trajectory-Data`; linked by the official StreamVLN repo | `/home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data` |
| VLN-CE episode JSON | Habitat episode definitions for R2R/RxR evaluation and replay | Official StreamVLN data preparation / R2R-CE style data | `/home/zhangshan/syp/StreamVLN-R2RCE-Test/data/datasets` |
| 3D scene meshes | Matterport3D scenes for R2R/RxR; HM3D scenes for ScaleVLN | Matterport3D / HM3D official Habitat downloads | `/home/zhangshan/syp/StreamVLN-R2RCE-Test/data/scene_datasets`, `/home/zhangshan/syp/datasets/scene_datasets`, `/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2` |

Canonical URLs used by this project:

| Resource | URL |
|---|---|
| StreamVLN project page | <https://streamvln.github.io/> |
| StreamVLN official code | <https://github.com/InternRobotics/StreamVLN> |
| StreamVLN trajectory data | <https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data> |
| Matterport3D official dataset page | <https://niessner.github.io/Matterport/> |
| HM3D official dataset page | <https://aihabitat.org/datasets/hm3d/> |
| Habitat-Sim dataset download docs | <https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md> |

The official StreamVLN repository states that the observation-action trajectory data should
be downloaded from Hugging Face and extracted to its trajectory-data folder. In this project
we use the following local equivalent:

```bash
pip install -U huggingface_hub
huggingface-cli login
mkdir -p /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data

huggingface-cli download cywan/StreamVLN-Trajectory-Data \
  --repo-type dataset \
  --local-dir /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data \
  --local-dir-use-symlinks False
```

If disk is limited, download only the subsets needed for the current experiments:

```bash
huggingface-cli download cywan/StreamVLN-Trajectory-Data \
  --repo-type dataset \
  --include "R2R/*" "RxR/*" "ScaleVLN/*" "README.md" \
  --local-dir /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data \
  --local-dir-use-symlinks False
```

The R2R/RxR Habitat episode JSON files used by evaluation/replay are stored locally under:

```text
/home/zhangshan/syp/StreamVLN-R2RCE-Test/data/datasets/
  r2r/train/train.json.gz
  r2r/val_seen/val_seen.json.gz
  r2r/val_unseen/val_unseen.json.gz
  rxr/train/train_guide.json.gz
  rxr/train/train_guide_gt.json.gz
  rxr/train/train_follower.json.gz
  rxr/train/train_follower_gt.json.gz
  rxr/val_seen/*.json.gz
  rxr/val_unseen/*.json.gz
```

Those files come from the StreamVLN/R2R-CE style Habitat dataset package, not from the
Qwen training code. Follow the official StreamVLN data preparation page if rebuilding them
from scratch.

Matterport3D scene meshes are required for R2R/RxR Habitat replay and evaluation. Download
them only after accepting the Matterport3D academic license. Do not commit real credentials:

```bash
export MATTERPORT_TOKEN_ID=<your_token_id>
export MATTERPORT_TOKEN_SECRET=<your_token_secret>

# Official Matterport3D Habitat download script; the script itself is provided
# by the Matterport3D release instructions.
python download_mp.py --task habitat \
  -o /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/scene_datasets
```

HM3D is used for ScaleVLN replay. It is downloaded with Habitat-Sim's dataset downloader
after obtaining HM3D access credentials:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hm3d_train_habitat_v0.2 hm3d_train_configs_v0.2 \
  --data-path /home/zhangshan/syp/datasets/versioned_data \
  --username "$MATTERPORT_TOKEN_ID" \
  --password "$MATTERPORT_TOKEN_SECRET"
```

Locally this produced:

```text
/home/zhangshan/syp/datasets/versioned_data/hm3d-0.2
```

The depth maps used by the auxiliary loss are not downloaded from Hugging Face as final
training targets. They are generated by replaying the downloaded trajectories in Habitat
with the scripts in this repository, described below.

### StreamVLN Training Data

Expected layout:

```text
/home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/
  R2R/
    annotations_v1-3.json
    images/{scan}_{traj}/
      rgb/NNN.jpg
      depth/NNN.png
      pose/NNN.json
  RxR/
    annotations.json
    images/{scan}_{traj}/
      rgb/NNN.jpg
      depth/NNN.png
      camera_params/NNN.json or pose/NNN.json
  ScaleVLN/
    annotations.json
    annotations_10k.json
    scalevln_subset_150k.json.gz
    images/{episode}/
      rgb/NNN.jpg
      depth/NNN.png
      pose/NNN.json
```

Depth PNG convention used by the current training code:

```text
depth_m = raw_png / 255 * 10.0    # 8-bit normalized depth
```

The collator also supports 16-bit depth by detecting the PNG mode/dtype and using `65535`.
Current experiments use the 8-bit depth generated by Habitat replay.

### R2R Evaluation Data

`config/vln_r2r.yaml` expects:

```text
scenes_dir: /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/scene_datasets/
data_path:  /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/datasets/r2r/{split}/{split}.json.gz
```

R2R `val_unseen` evaluation is checked against:

```text
expected episodes = 1839
expected split counts = 460 / 460 / 460 / 459
distributed_valid = true
```

### Extracting Depth/Pose From StreamVLN Trajectories

Depth maps are not estimated by a monocular depth model. They are produced by replaying
the recorded StreamVLN action trajectory in Habitat with an RGB-D sensor:

1. Load a StreamVLN annotation file containing `id`, `video`, `instructions`, and `actions`.
2. Load the matching Habitat dataset split from `--data_path` and scenes from `--scenes_dir`.
3. For each episode, reset Habitat to the matching episode id and step through the recorded actions.
4. At every frame, save Habitat depth to `depth/NNN.png` and camera intrinsics/extrinsics to `pose/NNN.json`.
5. The saved depth is `uint8`: `depth_png = clip(normalized_depth, 0, 1) * 255`.
6. Training decodes it as `depth_m = depth_png / 255 * 10.0`, so the current depth resolution is about 3.9 cm.
7. For R2R, the script also compares replayed RGB with the existing `rgb/NNN.jpg` and reports RGB mean absolute difference. Use this as the alignment sanity check.

The generic entrypoint is:

```text
scripts/extract_streamvln_depth_pose.py
```

It is a thin wrapper around:

```text
scripts/extract_r2r_depth_pose.py
```

Both are kept in this project. The generic wrapper exists so users do not have to know
that the original implementation was first written for R2R.

R2R depth/pose extraction:

```bash
conda activate qwen3vl_habitat
python scripts/extract_streamvln_depth_pose.py \
  --annotation_path /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/R2R/annotations_v1-3.json \
  --image_root /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/R2R/images \
  --config_path /home/zhangshan/syp/myvln/qwen3VL_vln/config/vln_r2r.yaml \
  --data_path /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/datasets/r2r/{split}/{split}.json.gz \
  --scenes_dir /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/scene_datasets/ \
  --split train \
  --skip_complete
```

For RxR, use the same script and output layout, but pass the RxR annotation/image paths
and the matching RxR Habitat dataset. Pick the guide/follower JSON that corresponds to the
StreamVLN RxR annotations you are replaying, then verify RGB alignment from the printed
`rgb_mean_abs_diff_*` values before training:

```bash
conda activate qwen3vl_habitat
python scripts/extract_streamvln_depth_pose.py \
  --annotation_path /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/RxR/annotations.json \
  --image_root /home/zhangshan/syp/datasets/StreamVLN-Trajectory-Data/RxR/images \
  --config_path /home/zhangshan/syp/StreamVLN-R2RCE-Test/StreamVLN/config/vln_r2r.yaml \
  --data_path /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/datasets/rxr/train/train_guide_gt.json.gz \
  --scenes_dir /home/zhangshan/syp/StreamVLN-R2RCE-Test/data/scene_datasets/ \
  --split train \
  --skip_complete
```

The current local RxR folders contain `depth/` and `camera_params/`; the trainer only consumes
`depth/`. If you replay with the script above, it writes `pose/` files next to `depth/`.

ScaleVLN trajectory extraction follows the same `rgb/depth/pose` folder layout:

```bash
conda activate qwen3vl_habitat
python scripts/extract_scalevln_trajectory_data.py \
  --limit 10000 \
  --skip_complete
```

Before using a new dataset split, verify RGB/depth frame alignment manually. The current
depth loss is sensitive to coordinate mismatch.

## 5. Baseline Training

The baseline is NAVIDA-style single-turn imitation:

```text
system: You are a helpful assistant.
user: historical RGB observations + current RGB observation + instruction
assistant: forward 25 cm, turn left 15 degree
```

There is no multi-round dialogue state in the model context; previous assistant output is not fed
back into later decisions.

Run baseline training:

```bash
conda activate qwen3vl
bash scripts/train_r2r_baseline.sh
```

Main baseline settings:

| Item | Value |
|---|---|
| Backbone | `Qwen/Qwen3-VL-4B-Instruct` |
| Data | StreamVLN R2R |
| History images | 8 |
| Image size | `308x252` |
| Chunking | NAVIDA-style random action chunks |
| Max train action segments | 3 |
| Stop upsample ratio | 2 |
| LR | `2e-5` |
| Per-device batch | 4 |
| Grad accumulation | 4 |
| Effective batch | 16 |
| Precision | bf16 |
| Attention | FlashAttention 2 |
| Distributed | DeepSpeed ZeRO-3 |
| Frozen | vision tower |
| Trainable | visual merger + LLM |

Baseline config:

```text
config/train_r2r_zero3_navida_random_stop2.yaml
```

Baseline output:

```text
outputs/qwen3vl_streamvln_r2r_navida_random_stop2_zero3
```

## 6. Depth-Auxiliary Training

Depth is used only during training. Evaluation uses the normal `generate()` path and does not
run the depth head.

Current cleaned depth design:

| Item | Value |
|---|---|
| Supervised frame | current frame only |
| Depth target coordinate | RGB input coordinate system |
| Depth target size | `308x252` |
| Depth resize | nearest |
| Tap layer | `language_model.layers[24]` |
| Depth head | lightweight MLP readout |
| PixelShuffle patch | 2 |
| Predicted grid | `16x20` |
| Loss | SILog |
| SILog alpha | 1.0 |
| Metric L1 / gradient loss | disabled |

Two-stage schedule:

### Stage 1: depth-head warmup

Only `depth_head.*` is trainable.

```bash
conda activate qwen3vl
bash scripts/train_r2r_depth_stage1.sh
```

Config/output:

```text
config/train_r2r_zero3_depth_stage1.yaml
outputs/qwen3vl_streamvln_r2r_depth_stage1_current_patch2_zero3
```

Key settings:

| Item | Value |
|---|---|
| Loss | depth only |
| LR | `1e-4` |
| Gradient checkpointing | false |
| Plateau min steps | 500 |
| Saved artifact | `depth_head.safetensors` |

### Stage 2: joint action + depth training

LLM + visual merger + depth head are trainable; vision tower remains frozen.

Best R2R-only dynamic-depth run:

```bash
conda activate qwen3vl
CONFIG_PATH=config/train_r2r_zero3_depth_stage2_w05to01.yaml \
  bash scripts/train_r2r_depth.sh
```

Config/output:

```text
config/train_r2r_zero3_depth_stage2_w05to01.yaml
outputs/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_zero3
```

Dynamic depth weight:

```text
depth_weight starts at 0.5
hold first 30% of training
cosine decay to 0.1
```

### R2R+RxR mixed depth training

Stage 1:

```bash
bash scripts/train_r2r_rxr_depth_stage1.sh
```

Stage 2:

```bash
bash scripts/train_r2r_rxr_depth_stage2_w05to01.sh
```

Configs/outputs:

```text
config/train_r2r_rxr_zero3_depth_stage1.yaml
outputs/qwen3vl_streamvln_r2r_rxr_depth_stage1_current_patch2_zero3

config/train_r2r_rxr_zero3_depth_stage2_w05to01.yaml
outputs/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_zero3
```

The current mixed Stage 2 run has finished training and has completed full R2R
`val_unseen` evaluation.

## 7. Evaluation

Use the best-policy evaluator for current comparisons:

```bash
conda activate qwen3vl_habitat
bash scripts/eval_r2r_best_policy.sh \
  outputs/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_zero3 \
  results/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_val_unseen_best_policy \
  val_unseen
```

Default best-policy settings:

```text
generation_max_new_tokens = 64
execute_action_segments = 2
action_execution_policy = fixed
decode_strategy = sample
history_images = 8
split_num = 4
```

The current strongest R2R+RxR mixed result uses the same policy except
`generation_max_new_tokens = 512`. The longer decode budget improves the mixed model
without changing the trained weights or adding inference-time depth computation.

`scripts/eval_r2r_best_policy.sh` wraps `scripts/eval_r2r_4gpu.sh`, which launches 4
deterministic non-overlapping splits and aggregates results with:

```bash
python scripts/aggregate_eval_results.py "$OUTPUT_DIR" \
  --split val_unseen \
  --expected-splits 4 \
  --expected-episodes 1839 \
  --require-valid
```

Do not compare against older invalid evals produced by independent Habitat `get_splits()`
calls; valid distributed evals must have 1839 unique episode/instruction pairs.

## 8. Main Results So Far

All R2R rows below are full `val_unseen` results unless noted otherwise.

| Run | Eval policy | SR | Delta SR | SPL | Delta SPL | NE | Delta NE | Oracle SR | Avg steps | >=400 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline, same best policy | tok64 fixed2 sample | 0.4252 | +0.0000 | 0.3818 | +0.0000 | 6.5368 | +0.0000 | 0.4878 | 76.00 | 41 |
| Old polluted full-frame depth | older/default; historical note | 0.3948 | -0.0304 | 0.3380 | -0.0438 | 7.2472 | +0.7104 | 0.4948 | 85.46 | n/a |
| Clean current-depth w=0.3 | tok512; historical note | 0.4318 | +0.0066 | 0.3868 | +0.0050 | 6.5763 | +0.0395 | 0.4981 | 74.07 | n/a |
| Clean current-depth w=0.3 | tok64 fixed2 sample | 0.4410 | +0.0158 | 0.3938 | +0.0119 | 6.4861 | -0.0507 | 0.5084 | 74.16 | 31 |
| Clean current-depth fixed w=0.5 | tok64 fixed2 sample | 0.4492 | +0.0239 | 0.3954 | +0.0136 | 6.2518 | -0.2851 | 0.5237 | 79.42 | 43 |
| R2R-only dynamic depth `0.5 -> 0.1` | tok64 fixed2 sample | 0.4584 | +0.0332 | 0.4104 | +0.0286 | 6.4373 | -0.0995 | 0.5193 | 78.22 | 36 |
| R2R+RxR mixed dynamic depth `0.5 -> 0.1` | tok64 fixed2 sample | 0.4965 | +0.0712 | 0.4348 | +0.0529 | 5.7680 | -0.7689 | 0.5595 | 76.38 | 38 |
| R2R+RxR mixed dynamic depth `0.5 -> 0.1` | tok512 fixed2 sample | 0.5095 | +0.0843 | 0.4487 | +0.0668 | 5.6161 | -0.9207 | 0.5688 | 76.09 | 40 |

Policy sweep on the cleaned current-depth w=0.3 checkpoint:

| Eval variant | SR | SPL | NE | Oracle SR | Avg steps | >=400 |
|---|---:|---:|---:|---:|---:|---:|
| v1 tok64 fixed2 sample | 0.4410 | 0.3938 | 6.4861 | 0.5084 | 74.16 | 31 |
| v2 tok64 fixed3 sample | 0.4209 | 0.3715 | 6.6128 | 0.4970 | 75.71 | 36 |
| v3 tok64 fixed2 greedy | 0.4361 | 0.3864 | 6.7519 | 0.5111 | 74.11 | 38 |
| v4 tok64 fixed3 greedy | 0.4339 | 0.3837 | 6.7080 | 0.4948 | 72.47 | 36 |
| v5 tok64 adaptive-turn-safe sample | 0.4290 | 0.3862 | 6.5353 | 0.4834 | 75.91 | 44 |
| v6 tok64 low-temp fixed2 sample | 0.4318 | 0.3882 | 6.5071 | 0.4943 | 73.26 | 29 |

Best overall result vs same-policy baseline:

| Metric | Baseline | R2R-only best depth | R2R+RxR tok64 | R2R+RxR tok512 | Tok512 delta vs baseline | Tok512 delta vs R2R+RxR tok64 |
|---|---:|---:|---:|---:|---:|---:|
| SR | 0.4252 | 0.4584 | 0.4965 | 0.5095 | +0.0843 | +0.0131 |
| SPL | 0.3818 | 0.4104 | 0.4348 | 0.4487 | +0.0668 | +0.0139 |
| NE | 6.5368 | 6.4373 | 5.7680 | 5.6161 | -0.9207 | -0.1518 |
| Oracle SR | 0.4878 | 0.5193 | 0.5595 | 0.5688 | +0.0810 | +0.0092 |
| Avg steps | 76.00 | 78.22 | 76.38 | 76.09 | +0.09 | -0.29 |
| >=400 steps | 41 | 36 | 38 | 40 | -1 | +2 |

Interpretation:

1. Depth supervision is useful when it is aligned with the current navigation decision.
2. Full-frame/history depth supervision was harmful: it raised Oracle SR slightly but hurt
   SR/SPL and increased path length/late-stop behavior.
3. Resizing depth into the same coordinate system as RGB is required. The old run supervised
   non-equivalent coordinates and should not be used as evidence against depth.
4. The best current recipe is current-frame depth + patch2 readout + SILog alpha 1.0 +
   dynamic weight `0.5 -> 0.1`.
5. Adding RxR data on top of the cleaned depth recipe gives the largest jump: tok64 mixed
   training improves SR by `+7.12` points and SPL by `+5.29` points over the same-policy
   baseline.
6. Increasing only the decode budget from tok64 to tok512 on the same R2R+RxR weight further
   improves SR from `0.4965` to `0.5095`, SPL from `0.4348` to `0.4487`, and NE from
   `5.7680` to `5.6161`. Avg steps stay essentially flat (`76.38 -> 76.09`), so this is not
   just a longer-trajectory artifact.
7. The strongest checked result is therefore R2R+RxR mixed dynamic depth with tok512 fixed2
   sample evaluation: SR improves by `+8.43` points and SPL by `+6.68` points over the
   same-policy baseline, with NE reduced by `0.92`.

## 9. Action Space

| Habitat id | Text output |
|---:|---|
| 0 | `stop` |
| 1 | `forward 25 cm` |
| 2 | `turn left 15 degree` |
| 3 | `turn right 15 degree` |

Example assistant output:

```text
forward 25 cm, turn right 15 degree
```

## 10. Quick Start

### Train R2R baseline

```bash
conda activate qwen3vl
bash scripts/train_r2r_baseline.sh
```

### Train best R2R depth model

```bash
conda activate qwen3vl
bash scripts/train_r2r_depth_stage1.sh

CONFIG_PATH=config/train_r2r_zero3_depth_stage2_w05to01.yaml \
  bash scripts/train_r2r_depth.sh
```

### Train R2R+RxR mixed depth model

```bash
conda activate qwen3vl
bash scripts/train_r2r_rxr_depth_stage1.sh
bash scripts/train_r2r_rxr_depth_stage2_w05to01.sh
```

### Evaluate on R2R val_unseen

```bash
conda activate qwen3vl_habitat
bash scripts/eval_r2r_best_policy.sh \
  outputs/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_zero3 \
  results/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_val_unseen_best_policy \
  val_unseen
```

Evaluate the current best R2R+RxR mixed model:

```bash
conda activate qwen3vl_habitat
bash scripts/eval_r2r_best_policy.sh \
  outputs/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_zero3 \
  results/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_val_unseen_best_policy \
  val_unseen
```

Evaluate the current strongest tok512 variant on the same R2R+RxR mixed weight:

```bash
conda activate qwen3vl_habitat
bash scripts/eval_r2r_4gpu.sh \
  outputs/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_zero3 \
  results/qwen3vl_streamvln_r2r_rxr_depth_stage2_current_patch2_w05to01_hold30_val_unseen_tok512_fixed2_sample \
  val_unseen \
  --generation_max_new_tokens 512 \
  --execute_action_segments 2 \
  --action_execution_policy fixed \
  --decode_strategy sample
```

### Check a running evaluation

```bash
OUTPUT_DIR=results/qwen3vl_streamvln_r2r_depth_stage2_current_patch2_w05to01_hold30_val_unseen_best_policy
for d in "$OUTPUT_DIR"/split_*; do
  [ -d "$d/log" ] || continue
  printf '%s\t' "${d##*/}"
  find "$d/log" -name 'stats_*.json' | wc -l
done
```

When finished, `final_metrics.json` is written under the eval output directory.

## 11. Practical Notes

- Keep R2R/RxR/ScaleVLN configs in sync with local dataset paths before launching training.
- Check RGB/depth alignment before trusting a new depth dataset.
- R2R baseline and depth results should be compared under the same best-policy eval.
- Depth head is an auxiliary training module. It is saved in checkpoints but not used by eval.
