#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 MODEL_PATH OUTPUT_DIR [SPLIT]}"
OUTPUT_DIR="${2:?Usage: $0 MODEL_PATH OUTPUT_DIR [SPLIT]}"
SPLIT="${3:-val_unseen}"
if [ "$#" -gt 3 ]; then
  shift 3
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

cd /home/zhangshan/syp/myvln/qwen3VL_vln
source /home/zhangshan/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_habitat

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1

LOG_PREFIX="logs/$(basename "$(dirname "$OUTPUT_DIR")")_$(basename "$OUTPUT_DIR")"
LAUNCH_STAGGER_SECONDS="${EVAL_LAUNCH_STAGGER_SECONDS:-0}"
mkdir -p "$OUTPUT_DIR" logs
: > "$OUTPUT_DIR/eval_pids.txt"
: > "$OUTPUT_DIR/eval_exit_codes.txt"
pids=()

for split_id in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$split_id" python -u eval.py \
    --model_path "$MODEL_PATH" \
    --config config/vln_r2r.yaml \
    --split "$SPLIT" \
    --output "$OUTPUT_DIR/split_$split_id" \
    --history_images 8 \
    --execute_action_segments 2 \
    --max_action_history 200 \
    --generation_max_new_tokens 512 \
    --split_num 4 \
    --split_id "$split_id" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_PREFIX}_split${split_id}.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  printf "%s\n" "$pid" >> "$OUTPUT_DIR/eval_pids.txt"
  if [ "$LAUNCH_STAGGER_SECONDS" != "0" ] && [ "$split_id" -lt 3 ]; then
    sleep "$LAUNCH_STAGGER_SECONDS"
  fi
done

status=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    code=0
  else
    code="$?"
    status=1
  fi
  printf "%s %s\n" "$pid" "$code" >> "$OUTPUT_DIR/eval_exit_codes.txt"
done
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
AGG_ARGS=(--split "$SPLIT" --expected-splits 4 --require-valid)
if [ "$SPLIT" = "val_unseen" ]; then
  AGG_ARGS+=(--expected-episodes 1839)
fi
python scripts/aggregate_eval_results.py "$OUTPUT_DIR" "${AGG_ARGS[@]}" > "$OUTPUT_DIR/final_metrics.json.tmp"
mv "$OUTPUT_DIR/final_metrics.json.tmp" "$OUTPUT_DIR/final_metrics.json"
printf "all eval splits finished at %s\n" "$(date)" > "$OUTPUT_DIR/done.txt"
