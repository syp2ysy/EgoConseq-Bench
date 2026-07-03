#!/bin/bash
set -euo pipefail

DEPTH_SWEEP_ROOT="${1:?Usage: $0 DEPTH_SWEEP_ROOT BASELINE_MODEL_PATH OUTPUT_ROOT [SPLIT]}"
BASELINE_MODEL_PATH="${2:?Usage: $0 DEPTH_SWEEP_ROOT BASELINE_MODEL_PATH OUTPUT_ROOT [SPLIT]}"
OUTPUT_ROOT="${3:?Usage: $0 DEPTH_SWEEP_ROOT BASELINE_MODEL_PATH OUTPUT_ROOT [SPLIT]}"
SPLIT="${4:-val_unseen}"

cd /home/zhangshan/syp/myvln/qwen3VL_vln
mkdir -p "$OUTPUT_ROOT" logs

BEST_VARIANT=$(python - <<'PY' "$DEPTH_SWEEP_ROOT" "$SPLIT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
split = sys.argv[2]
rows = []
for metrics_path in root.glob(f"*_{split}/final_metrics.json"):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    rows.append((float(metrics.get("success_rate", 0.0)), metrics_path.parent.name))
if not rows:
    raise SystemExit(f"No completed final_metrics.json found under {root}")
rows.sort(reverse=True)
print(rows[0][1])
PY
)

case "$BEST_VARIANT" in
  *v1_navida_fixed2_sample*) EXTRA=(--execute_action_segments 2 --action_execution_policy fixed --decode_strategy sample) ;;
  *v2_fixed3_sample*) EXTRA=(--execute_action_segments 3 --action_execution_policy fixed --decode_strategy sample) ;;
  *v3_navida_fixed2_greedy*) EXTRA=(--execute_action_segments 2 --action_execution_policy fixed --decode_strategy greedy) ;;
  *v4_fixed3_greedy*) EXTRA=(--execute_action_segments 3 --action_execution_policy fixed --decode_strategy greedy) ;;
  *v5_adaptive_turn_safe_sample*) EXTRA=(--execute_action_segments 2 --action_execution_policy adaptive_turn_safe --decode_strategy sample) ;;
  *v6_lowtemp_fixed2_sample*) EXTRA=(--execute_action_segments 2 --action_execution_policy fixed --decode_strategy sample --temperature 0.1) ;;
  *) echo "Unsupported best variant: $BEST_VARIANT" >&2; exit 1 ;;
esac

BASELINE_TAG="baseline_${BEST_VARIANT}"
BASELINE_OUT="$OUTPUT_ROOT/${BASELINE_TAG}"
LOG="logs/${BASELINE_TAG}_eval.log"

{
  printf "depth_sweep_root=%s\n" "$DEPTH_SWEEP_ROOT"
  printf "best_variant=%s\n" "$BEST_VARIANT"
  printf "baseline_model=%s\n" "$BASELINE_MODEL_PATH"
  printf "baseline_output=%s\n" "$BASELINE_OUT"
  printf "started=%s\n" "$(date)"
  printf "args=%q " "${EXTRA[@]}"
  printf "\n"
} | tee "$LOG"

EVAL_LAUNCH_STAGGER_SECONDS="${EVAL_LAUNCH_STAGGER_SECONDS:-12}" \
  bash scripts/eval_r2r_4gpu.sh "$BASELINE_MODEL_PATH" "$BASELINE_OUT" "$SPLIT" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"

printf "finished=%s\n" "$(date)" | tee -a "$LOG"
