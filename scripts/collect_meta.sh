#!/bin/bash
# Meta data of a new output root: unseen inputs, collection in shards, export.
#   scripts/collect_meta.sh data/unseen_meta_next propose r2r 150 1   unseen inputs: poses in every test/unseen scene
#                                                                (150 each, GPU 1), then <root>/inputs/r2r.jsonl.gz
#   scripts/collect_meta.sh data/seen_meta_next r2r 12 0 2      GPU0 runs shards 0-5, GPU2 runs shards 6-11
#   scripts/collect_meta.sh data/seen_meta_next b1k 8 2 3       B1K runs in EGOCONSEQ_B1K_PYTHON (OmniGibson)
#   scripts/collect_meta.sh data/seen_meta_next export          write the final <root>/ format
# Outputs: <root>/_work/<dataset>-<shard>; rerunning the same command resumes.
# Inputs: <root>/inputs. propose.py draws coarse programs from data/seen_meta_gt3/inputs/coarse_pool.json
# (built from the frozen seen inputs; their original generator is archived in data/archive/seen_meta_inputs_20261012).
# New unseen inputs are made by `propose` (pipeline/meta/propose.py).
# Needs EGOCONSEQ_HABITAT_PYTHON and the dataset roots (EGOCONSEQ_DATA_ROOT, see .env.example).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$1
shift
DATASET=$1
[ "$1" = propose ] && DATASET=$2
if [ "$DATASET" != b1k ]; then
  PY=${EGOCONSEQ_HABITAT_PYTHON:?set EGOCONSEQ_HABITAT_PYTHON}
else
  PY=${EGOCONSEQ_B1K_PYTHON:?set EGOCONSEQ_B1K_PYTHON}
  B1K=${B1K_DATA_ROOT:-${EGOCONSEQ_DATA_ROOT:?set EGOCONSEQ_DATA_ROOT}/behavior-1k-v3.9.1}
  export OMNIGIBSON_DATA_PATH=$B1K OMNIGIBSON_APPDATA_PATH=$B1K/appdata OMNIGIBSON_HEADLESS=True \
    OMNI_KIT_ACCEPT_EULA=YES
fi
if [ "$1" = export ]; then
  exec "$PY" -m pipeline.meta.export --work $ROOT/_work --output $ROOT
fi
: "${EGOCONSEQ_DATA_ROOT:?set EGOCONSEQ_DATA_ROOT}"
export PBENCH_HABITAT_GPU_DEVICE_ID=0 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet OMP_NUM_THREADS=3
mkdir -p $ROOT/_work/logs
device() {  # Isaac Sim (B1K) selects its GPU itself and rejects CUDA_VISIBLE_DEVICES
  if [ "$DATASET" = b1k ]; then DEVICE=(env -u CUDA_VISIBLE_DEVICES OMNIGIBSON_GPU_ID=$1); else DEVICE=(env CUDA_VISIBLE_DEVICES=$1); fi
}
if [ "$1" = propose ]; then
  COUNT=$3
  device "$4"
  PROPOSE=("$PY" -u -m pipeline.meta.propose poses --dataset "$DATASET" --count "$COUNT" --output $ROOT/inputs)
  if [ "$DATASET" = b1k ]; then  # one scene per process (Isaac Sim keeps the GPU memory of a closed scene)
    for SCENE in $("$PY" -c "from pipeline.meta.propose import scene_rows; print(*[s for d, s in scene_rows() if d == 'b1k'])"); do
      "${DEVICE[@]}" "${PROPOSE[@]}" --scenes "$SCENE"
    done
  else
    "${DEVICE[@]}" "${PROPOSE[@]}"
  fi >> $ROOT/_work/logs/propose-$DATASET.log 2>&1
  exec "$PY" -m pipeline.meta.propose inputs --dataset "$DATASET" --output $ROOT/inputs
fi
SHARDS=$2
shift 2
GPUS=("$@")
PIDS=()
for ((i = 0; i < SHARDS; i++)); do
  device "${GPUS[i * ${#GPUS[@]} / SHARDS]}"
  OUT=$ROOT/_work/$DATASET-$i
  until grep -qs '"status": "complete"' $OUT/report.json; do  # B1K runs one scene per process and resumes
    BEFORE=$(grep -so '"observations": [0-9]*' $OUT/report.json || true)  # none before the first run
    "${DEVICE[@]}" "$PY" -u -m pipeline.meta.collect --inputs $ROOT/inputs --dataset "$DATASET" --output $OUT \
      --shards "$SHARDS" --shard-index $i || exit 1
    # Isaac Sim exits 0 after a Python error: an unfinished run that adds no observation would only repeat it
    grep -qs '"status": "complete"' $OUT/report.json \
      || [ "$(grep -so '"observations": [0-9]*' $OUT/report.json || true)" != "$BEFORE" ] \
      || { echo "no progress" >&2; exit 1; }
  done >> $ROOT/_work/logs/$DATASET-$i.log 2>&1 &
  PIDS+=($!)
done
STATUS=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[i]}" || { echo "$DATASET shard $i failed, see $ROOT/_work/logs/$DATASET-$i.log" >&2; STATUS=1; }
done
exit $STATUS
