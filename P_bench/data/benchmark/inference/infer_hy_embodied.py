#!/usr/bin/env python3
"""HY-Embodied VLM text answers via the official HYV3VL plugin, not VLA actions."""

from common import api_predictor, run


if __name__ == "__main__":
    run(api_predictor, model="tencent/Hy-Embodied-VLM-1.0", backend="api")
