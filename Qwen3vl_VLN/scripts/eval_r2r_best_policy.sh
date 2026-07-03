#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 MODEL_PATH OUTPUT_DIR [SPLIT]}"
OUTPUT_DIR="${2:?Usage: $0 MODEL_PATH OUTPUT_DIR [SPLIT]}"
SPLIT="${3:-val_unseen}"

cd /home/zhangshan/syp/myvln/qwen3VL_vln

# Current best R2R val_unseen evaluation policy, selected by the June 2026
# policy sweep on the cleaned current-frame depth auxiliary model.
EVAL_LAUNCH_STAGGER_SECONDS="${EVAL_LAUNCH_STAGGER_SECONDS:-12}" \
  bash scripts/eval_r2r_4gpu.sh "$MODEL_PATH" "$OUTPUT_DIR" "$SPLIT" \
    --generation_max_new_tokens 64 \
    --execute_action_segments 2 \
    --action_execution_policy fixed \
    --decode_strategy sample
