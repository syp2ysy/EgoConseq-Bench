"""Frozen action controls reused across poses for the A1 exact slice."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Tuple

from pipeline import actions as A


CONTRACT = "exact-action-catalog.v1"
ASSET_PATH = Path(__file__).with_name("assets") / \
    "exact_action_catalog.v1.json"
ANCHORS_PER_POSE = 5
CATALOG_SHA256 = \
    "bf3820f92060c3e3de856ff853637954aee6529463ca474408564449ac09d9de"


@dataclass(frozen=True)
class ControlAnchor:
    """One immutable metric program in the cross-pose control catalog."""

    tag: str
    actions: Tuple[A.Action, ...]

    @property
    def length(self) -> int:
        return len(self.actions)


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def catalog_sha256() -> str:
    """Digest the parsed asset rather than its incidental JSON whitespace."""
    return _canonical_sha256(json.loads(ASSET_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def load_catalog() -> Tuple[ControlAnchor, ...]:
    """Load and validate the committed thirteen-action catalog."""
    value = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != CONTRACT:
        raise ValueError("A1 control catalog schema is invalid")
    if catalog_sha256() != CATALOG_SHA256:
        raise ValueError("A1 control catalog digest changed")
    raw = value.get("anchors")
    if not isinstance(raw, list):
        raise ValueError("A1 control catalog anchors are invalid")
    anchors = []
    for row in raw:
        tag = str(row.get("tag") or "")
        actions = tuple(A.parse_actions(row.get("actions") or []))
        A.validate_physics_actions(actions)
        if not tag or len(actions) < 1 or \
                A.total_forward_m(actions) > 6.0 + 1e-9:
            raise ValueError("A1 control catalog entry is invalid")
        anchors.append(ControlAnchor(tag=tag, actions=actions))
    if len(anchors) != 13 or len({anchor.tag for anchor in anchors}) != 13:
        raise ValueError("A1 control catalog must contain thirteen unique tags")
    if {anchor.length for anchor in anchors} != {1, 2, 3, 4, 5, 6}:
        raise ValueError("A1 control catalog must cover L1-L6")
    return tuple(anchors)


def anchors_for_pose(*, dataset: str, scene_id: str,
                     pose_index: int) -> Tuple[ControlAnchor, ...]:
    """Select five anchors; a 13-pose cycle attempts each anchor five times."""
    anchors = load_catalog()
    digest = hashlib.sha256(
        f"{dataset}:{scene_id}:a1-control-rotation-v1".encode("utf-8")
    ).digest()
    base = int.from_bytes(digest[:8], "big") % len(anchors)
    offset = (base + ANCHORS_PER_POSE * int(pose_index)) % len(anchors)
    return tuple(
        anchors[(offset + index) % len(anchors)]
        for index in range(ANCHORS_PER_POSE)
    )
