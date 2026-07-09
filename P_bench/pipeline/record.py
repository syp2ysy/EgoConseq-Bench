"""FrameRecord assembly + jsonl I/O (append-style; numpy -> python)."""

from __future__ import annotations

import json
from typing import Iterator, List

import numpy as np

from pipeline import config

SCHEMA_VERSION = "conseq.v1"


def _jsonable(o):
    """Recursively convert numpy scalars/arrays to plain python for json."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _object_public(obj: dict) -> dict:
    """Drop transient fields; ensure dist_geodesic_m key exists."""
    pub = {k: v for k, v in obj.items() if not k.startswith("_")}
    pub.setdefault("dist_geodesic_m", None)
    return pub


def build_record(frame, outcomes: List[dict], *,
                 image_path: str, depth_path=None, semantic_path=None) -> dict:
    objs = [_object_public(o) for o in frame.objects]
    n_nonstruct = sum(0 if o["is_structural"] else 1 for o in objs)
    rec = {
        "schema_version": SCHEMA_VERSION,
        "frame_id": frame.frame_id,
        "scene_id": frame.scene_id,
        "scene_glb": frame.scene_glb,
        "pose": {"position": list(frame.position), "yaw_rad": frame.yaw_rad},
        "sensor": {"camera_height_m": config.CAMERA_HEIGHT_M,
                   "hfov_deg": config.HFOV_DEG,
                   "resolution": list(config.RESOLUTION)},
        "floor_y": frame.floor_y,
        "image_path": image_path,
        "depth_path": depth_path,
        "semantic_path": semantic_path,
        "quality": frame.quality,
        "n_objects": len(objs),
        "n_nonstructural": n_nonstruct,
        "category_inventory": frame.category_inventory,
        "objects": objs,
        "outcomes": outcomes,
    }
    return _jsonable(rec)


def append_record(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(record)) + "\n")


def read_records(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
