#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
CONFIG_PATH="${CONFIG_PATH:-config/train_r2r_zero3_depth.yaml}"

cd "$PROJECT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
DEFAULT_RUN_NAME="$(basename "${CONFIG_PATH%.yaml}")_$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"
echo "Logging training output to $LOG_FILE"

deepspeed --num_gpus 4 train.py \
  --config "$CONFIG_PATH" \
  "$@" 2>&1 | tee "$LOG_FILE"
