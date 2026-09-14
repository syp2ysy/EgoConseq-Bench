#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 MODEL_PATH OUTPUT_DIR [SPLIT] [eval.py options]}"
OUTPUT_DIR="${2:?Usage: $0 MODEL_PATH OUTPUT_DIR [SPLIT] [eval.py options]}"
SPLIT="${3:-val_unseen}"
if [ "$#" -gt 3 ]; then
  shift 3
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --max_episodes|--max_episodes=*)
      echo "Use eval.py for partial evaluation; this launcher checks full split coverage." >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate qwen3vl_habitat
cd "$PROJECT_DIR"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
  GPU_IDS=()
  for ((gpu=0; gpu<GPU_COUNT; gpu++)); do GPU_IDS+=("$gpu"); done
fi
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt "${#GPU_IDS[@]}" ]; then
  echo "NUM_GPUS must fit the visible CUDA GPUs." >&2
  exit 1
fi
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR")" ]; then
  echo "Use an empty OUTPUT_DIR to avoid mixing evaluation results: $OUTPUT_DIR" >&2
  exit 1
fi

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"
pids=()
for ((split_id=0; split_id<NUM_GPUS; split_id++)); do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$split_id]}" python -u eval.py \
    --model_path "$MODEL_PATH" \
    --config config/vln_r2r.yaml \
    --split "$SPLIT" \
    --output "$OUTPUT_DIR/split_$split_id" \
    --split_num "$NUM_GPUS" \
    --split_id "$split_id" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}" \
    > "$OUTPUT_DIR/split_$split_id.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$OUTPUT_DIR/eval_pids.txt"
status=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then code=0; else code="$?"; status=1; fi
  printf '%s %s\n' "$pid" "$code" >> "$OUTPUT_DIR/eval_exit_codes.txt"
done
if [ "$status" -ne 0 ]; then exit "$status"; fi
AGG_ARGS=(--split "$SPLIT" --expected-splits "$NUM_GPUS" --require-valid)
if [ "$SPLIT" = "val_unseen" ]; then AGG_ARGS+=(--expected-episodes 1839); fi
python scripts/aggregate_eval_results.py "$OUTPUT_DIR" "${AGG_ARGS[@]}"
