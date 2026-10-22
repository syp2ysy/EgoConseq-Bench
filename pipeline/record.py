"""FrameRecord assembly + jsonl I/O (append-style; numpy -> python)."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np



B1K_RENDERER_INSTANCE_PROTOCOL = "b1k-observation.v5"
# Official OmniGibson rendering (collection contract b1k-visible-space-abc1.v7).
B1K_OBSERVATION_PROFILE_SHA256 = \
    "ae50ab14d51c0ccbc0eba398e06ecf79b9b9e5f612f684026e319bf483faee3a"
B1K_C1_RENDER_MODE = "persistent-counterfactual-bank-batch.v2"
B1K_INSTANCE_TRIANGLES_SCHEMA = "b1k-runtime-instance-triangles.v1"


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        json_value(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_atom_sha256(value: dict) -> str:
    """Public name for canonical hashes used by persisted v16 atoms."""
    return _canonical_sha256(value)


def json_value(o):
    """Convert arrays/scalars to strict JSON; unavailable numbers become null."""
    if isinstance(o, dict):
        return {k: json_value(v) for k, v in o.items()
                if not isinstance(k, str) or not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [json_value(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        value = float(o)
        return value if math.isfinite(value) else None
    if isinstance(o, np.ndarray):
        return json_value(o.tolist())
    return o


