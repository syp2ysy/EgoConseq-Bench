#!/usr/bin/env python3
"""SenseNova-SI-1.5 text answers via an InternVL-compatible vLLM server."""

from common import api_predictor, run


if __name__ == "__main__":
    run(api_predictor, model="sensenova/SenseNova-SI-1.5-InternVL3-8B", backend="api")
