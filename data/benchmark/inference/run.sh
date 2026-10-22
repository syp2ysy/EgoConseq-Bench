#!/usr/bin/env bash
set -e
export PYTHONDONTWRITEBYTECODE=1
exec "${INFER_PYTHON:-/home/zhangshan/miniconda3/envs/qwen3vl/bin/python}" -B "$(dirname "${BASH_SOURCE[0]}")/run.py" "$@"
