#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

cd "$PROJECT_DIR"

deepspeed --num_gpus 4 train.py \
  --config config/train_r2r_zero3_navida_random_stop2.yaml \
  --output_dir outputs/qwen3vl_streamvln_r2r_navida_random_stop2_zero3
