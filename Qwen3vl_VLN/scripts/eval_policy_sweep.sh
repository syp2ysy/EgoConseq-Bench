#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 MODEL_PATH OUTPUT_ROOT MODEL_TAG [SPLIT]}"
OUTPUT_ROOT="${2:?Usage: $0 MODEL_PATH OUTPUT_ROOT MODEL_TAG [SPLIT]}"
MODEL_TAG="${3:?Usage: $0 MODEL_PATH OUTPUT_ROOT MODEL_TAG [SPLIT]}"
SPLIT="${4:-val_unseen}"

cd /home/zhangshan/syp/myvln/qwen3VL_vln
mkdir -p "$OUTPUT_ROOT" logs
START_INDEX="${EVAL_SWEEP_START_INDEX:-1}"
END_INDEX="${EVAL_SWEEP_END_INDEX:-6}"
read -r -a COMMON_ARGS <<< "${EVAL_SWEEP_COMMON_ARGS:-}"

run_variant() {
  local index="$1"
  shift
  local name="$1"
  shift
  if [ "$index" -lt "$START_INDEX" ] || [ "$index" -gt "$END_INDEX" ]; then
    printf "skip variant=%s index=%s start=%s end=%s\n" "$name" "$index" "$START_INDEX" "$END_INDEX"
    return 0
  fi
  local output_dir="$OUTPUT_ROOT/${MODEL_TAG}_${name}_${SPLIT}"
  local log_file="logs/${MODEL_TAG}_${name}_${SPLIT}_sweep.log"
  {
    printf "variant=%s\n" "$name"
    printf "model=%s\n" "$MODEL_PATH"
    printf "output=%s\n" "$output_dir"
    printf "started=%s\n" "$(date)"
    printf "args=%q " "$@"
    printf "\n"
  } | tee "$log_file"

  EVAL_LAUNCH_STAGGER_SECONDS="${EVAL_LAUNCH_STAGGER_SECONDS:-12}" \
    bash scripts/eval_r2r_4gpu.sh "$MODEL_PATH" "$output_dir" "$SPLIT" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee -a "$log_file"
  printf "finished=%s\n" "$(date)" | tee -a "$log_file"
}

run_variant 1 v1_navida_fixed2_sample \
  --execute_action_segments 2 \
  --action_execution_policy fixed \
  --decode_strategy sample

run_variant 2 v2_fixed3_sample \
  --execute_action_segments 3 \
  --action_execution_policy fixed \
  --decode_strategy sample

run_variant 3 v3_navida_fixed2_greedy \
  --execute_action_segments 2 \
  --action_execution_policy fixed \
  --decode_strategy greedy

run_variant 4 v4_fixed3_greedy \
  --execute_action_segments 3 \
  --action_execution_policy fixed \
  --decode_strategy greedy

run_variant 5 v5_adaptive_turn_safe_sample \
  --execute_action_segments 2 \
  --action_execution_policy adaptive_turn_safe \
  --decode_strategy sample

run_variant 6 v6_lowtemp_fixed2_sample \
  --execute_action_segments 2 \
  --action_execution_policy fixed \
  --decode_strategy sample \
  --temperature 0.1

python scripts/analyze_eval_policy_sweep.py "$OUTPUT_ROOT" --model-tag "$MODEL_TAG" --split "$SPLIT" \
  | tee "$OUTPUT_ROOT/${MODEL_TAG}_${SPLIT}_summary.md"
