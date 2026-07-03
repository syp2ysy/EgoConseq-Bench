#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=config/train_r2r_zero3_depth_stage2.yaml "$SCRIPT_DIR/train_r2r_depth.sh" "$@"
