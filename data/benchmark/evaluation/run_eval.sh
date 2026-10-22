#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec "${EVAL_PYTHON:-/home/zhangshan/miniconda3/envs/qwen3vl/bin/python}" -B "$(dirname "${BASH_SOURCE[0]}")/eval.py" "$@"
