#!/bin/bash
set -euo pipefail

DEPTH_OUTPUT_ROOT="${1:?Usage: $0 DEPTH_OUTPUT_ROOT MODEL_TAG BASELINE_MODEL BASELINE_OUTPUT_ROOT [SPLIT]}"
MODEL_TAG="${2:?Usage: $0 DEPTH_OUTPUT_ROOT MODEL_TAG BASELINE_MODEL BASELINE_OUTPUT_ROOT [SPLIT]}"
BASELINE_MODEL="${3:?Usage: $0 DEPTH_OUTPUT_ROOT MODEL_TAG BASELINE_MODEL BASELINE_OUTPUT_ROOT [SPLIT]}"
BASELINE_OUTPUT_ROOT="${4:?Usage: $0 DEPTH_OUTPUT_ROOT MODEL_TAG BASELINE_MODEL BASELINE_OUTPUT_ROOT [SPLIT]}"
SPLIT="${5:-val_unseen}"

cd /home/zhangshan/syp/myvln/qwen3VL_vln
mkdir -p "$BASELINE_OUTPUT_ROOT" logs

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-1800}"
EXPECTED_VARIANTS="${EXPECTED_VARIANTS:-6}"
SUMMARY_PATH="$DEPTH_OUTPUT_ROOT/${MODEL_TAG}_${SPLIT}_summary.md"
BEST_PATH="$DEPTH_OUTPUT_ROOT/${MODEL_TAG}_${SPLIT}_best_policy.json"
LAUNCH_MARKER="$BASELINE_OUTPUT_ROOT/baseline_launch.started"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

count_finished() {
  find "$DEPTH_OUTPUT_ROOT" -maxdepth 2 -name final_metrics.json -print 2>/dev/null | wc -l
}

write_summary_and_best() {
  python scripts/analyze_eval_policy_sweep.py "$DEPTH_OUTPUT_ROOT" --model-tag "$MODEL_TAG" --split "$SPLIT" \
    | tee "$SUMMARY_PATH"
  python - "$DEPTH_OUTPUT_ROOT" "$MODEL_TAG" "$SPLIT" "$BEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
model_tag = sys.argv[2]
split = sys.argv[3]
out = Path(sys.argv[4])

rows = []
for path in sorted(root.glob(f"{model_tag}_*_{split}/final_metrics.json")):
    metrics = json.loads(path.read_text())
    name = path.parent.name
    prefix = f"{model_tag}_"
    suffix = f"_{split}"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    rows.append({"variant": name, "metrics": metrics, "path": str(path.parent)})

rows.sort(key=lambda row: float(row["metrics"].get("success_rate") or 0.0), reverse=True)
if len(rows) < 1:
    raise SystemExit("no completed variants found")
out.write_text(json.dumps(rows[0], indent=2) + "\n")
print(rows[0]["variant"])
PY
}

policy_args() {
  case "$1" in
    v1_navida_fixed2_sample)
      printf '%s\n' --generation_max_new_tokens 64 --execute_action_segments 2 --action_execution_policy fixed --decode_strategy sample
      ;;
    v2_fixed3_sample)
      printf '%s\n' --generation_max_new_tokens 64 --execute_action_segments 3 --action_execution_policy fixed --decode_strategy sample
      ;;
    v3_navida_fixed2_greedy)
      printf '%s\n' --generation_max_new_tokens 64 --execute_action_segments 2 --action_execution_policy fixed --decode_strategy greedy
      ;;
    v4_fixed3_greedy)
      printf '%s\n' --generation_max_new_tokens 64 --execute_action_segments 3 --action_execution_policy fixed --decode_strategy greedy
      ;;
    v5_adaptive_turn_safe_sample)
      printf '%s\n' --generation_max_new_tokens 64 --execute_action_segments 2 --action_execution_policy adaptive_turn_safe --decode_strategy sample
      ;;
    v6_lowtemp_fixed2_sample)
      printf '%s\n' --generation_max_new_tokens 64 --execute_action_segments 2 --action_execution_policy fixed --decode_strategy sample --temperature 0.1
      ;;
    *)
      printf 'unknown variant: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

printf "[%s] watching depth sweep: %s\n" "$(timestamp)" "$DEPTH_OUTPUT_ROOT"
while true; do
  finished="$(count_finished)"
  printf "[%s] completed variants: %s/%s\n" "$(timestamp)" "$finished" "$EXPECTED_VARIANTS"
  if [ "$finished" -ge "$EXPECTED_VARIANTS" ]; then
    break
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done

best_variant="$(write_summary_and_best | tail -1)"
printf "[%s] best depth policy: %s\n" "$(timestamp)" "$best_variant"

baseline_output="$BASELINE_OUTPUT_ROOT/${MODEL_TAG}_baseline_${best_variant}_${SPLIT}"
if [ -f "$baseline_output/final_metrics.json" ]; then
  printf "[%s] baseline already complete: %s\n" "$(timestamp)" "$baseline_output"
  exit 0
fi
if [ -f "$LAUNCH_MARKER" ]; then
  printf "[%s] baseline launch marker exists, not launching twice: %s\n" "$(timestamp)" "$LAUNCH_MARKER"
  exit 0
fi

mapfile -t args < <(policy_args "$best_variant")
{
  printf "started=%s\n" "$(timestamp)"
  printf "best_variant=%s\n" "$best_variant"
  printf "baseline_model=%s\n" "$BASELINE_MODEL"
  printf "baseline_output=%s\n" "$baseline_output"
  printf "args=%q " "${args[@]}"
  printf "\n"
} | tee "$LAUNCH_MARKER"

EVAL_LAUNCH_STAGGER_SECONDS="${EVAL_LAUNCH_STAGGER_SECONDS:-12}" \
  bash scripts/eval_r2r_4gpu.sh "$BASELINE_MODEL" "$baseline_output" "$SPLIT" "${args[@]}"

printf "[%s] baseline finished: %s\n" "$(timestamp)" "$baseline_output"
