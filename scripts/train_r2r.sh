#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate qwen3vl
cd "$PROJECT_DIR"

NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "No visible CUDA GPUs. Set CUDA_VISIBLE_DEVICES to the GPUs to use." >&2
  exit 1
fi
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
exec torchrun --nproc_per_node "$NUM_GPUS" train.py \
  --config config/train_r2r.yaml "$@"
