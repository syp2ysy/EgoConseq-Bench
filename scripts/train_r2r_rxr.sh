#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate qwen3vl
cd "$PROJECT_DIR"

NUM_GPUS="${NUM_GPUS:-4}"
VISIBLE_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt "$VISIBLE_GPUS" ]; then
  echo "Requested $NUM_GPUS GPUs, but only $VISIBLE_GPUS are visible. Check CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
exec torchrun --nproc_per_node "$NUM_GPUS" --master_port "${MASTER_PORT:-25420}" train.py \
  --config config/train_r2r_rxr.yaml "$@"
