#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
exec "${INFER_PYTHON:-/home/zhangshan/miniconda3/envs/qwen3vl/bin/python}" -B run.py "$@"
