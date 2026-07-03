"""
egoconseq.manifest
==================
Case schema and JSONL I/O for EgoConseq-Bench.

Design contract (§8.2):
- All fields have sensible defaults so partial construction works.
- model_payload() returns ONLY {image_path, question}: the single enforcement
  point ensuring the model never sees depth / pose / navmesh data.
- Serialization: to_dict / from_dict + write_jsonl / read_jsonl (plain json, no
  third-party schema library).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Case:
    """One benchmark case (single image + question + full offline metadata)."""

    # ── identity ─────────────────────────────────────────────────────────────
    case_id: str = ""
    operation_id: Optional[str] = None       # e.g. "O5"
    readout_tag: Optional[str] = None        # e.g. "binary_contact"
    episode_id: Optional[str] = None
    group_id: Optional[str] = None

    # ── scene / sensor ───────────────────────────────────────────────────────
    scene_id: Optional[str] = None
    pose: List[float] = field(default_factory=list)           # [x, y, z, yaw]
    sensor_profile: Dict[str, Any] = field(default_factory=dict)
    # {camera_height, hfov, resolution}

    # ── body geometry ─────────────────────────────────────────────────────────
    body: Dict[str, Any] = field(default_factory=dict)
    # {radius_m, diameter_m}

    # ── action ────────────────────────────────────────────────────────────────
    action: Dict[str, Any] = field(default_factory=dict)
    # {type, angle_deg, horizon_m, horizon_body_widths, horizon_reference}

    # ── safety distances ─────────────────────────────────────────────────────
    d_safe_visible_m: Optional[float] = None
    d_safe_visible_body_widths: Optional[float] = None
    d_safe_navmesh_m: Optional[float] = None
    margin_body_widths: Optional[float] = None

    # ── oracle ────────────────────────────────────────────────────────────────
    oracle_agreement: Optional[str] = None    # "agree" | "disagree"

    # ── geometry evidence ─────────────────────────────────────────────────────
    contact_point_3d: Optional[List[float]] = None   # [x, y, z]
    contact_pixel: Optional[List[int]] = None        # [u, v]
    requires_hidden_geometry: Optional[bool] = None
    fov_supported: Optional[bool] = None

    # ── quality gates ─────────────────────────────────────────────────────────
    gates: Dict[str, Any] = field(default_factory=dict)
    # {visible_sweep_ratio, valid_depth_ratio, depth_hole_ratio, ...}

    # ── model-visible payload (only these two!) ───────────────────────────────
    image_path: Optional[str] = None
    question: Optional[str] = None

    # ── answer schema ─────────────────────────────────────────────────────────
    answer: Dict[str, Any] = field(default_factory=dict)
    # {answer_type, label, ...}

    # ── free-form tags ────────────────────────────────────────────────────────
    tags: Dict[str, Any] = field(default_factory=dict)
    # {geometry_tag, difficulty, body_radius, action_type}

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize every field to a plain dict (JSON-compatible)."""
        return {
            "case_id": self.case_id,
            "operation_id": self.operation_id,
            "readout_tag": self.readout_tag,
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "scene_id": self.scene_id,
            "pose": self.pose,
            "sensor_profile": self.sensor_profile,
            "body": self.body,
            "action": self.action,
            "d_safe_visible_m": self.d_safe_visible_m,
            "d_safe_visible_body_widths": self.d_safe_visible_body_widths,
            "d_safe_navmesh_m": self.d_safe_navmesh_m,
            "margin_body_widths": self.margin_body_widths,
            "oracle_agreement": self.oracle_agreement,
            "contact_point_3d": self.contact_point_3d,
            "contact_pixel": self.contact_pixel,
            "requires_hidden_geometry": self.requires_hidden_geometry,
            "fov_supported": self.fov_supported,
            "gates": self.gates,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Case":
        """Reconstruct a Case from a plain dict produced by to_dict()."""
        return cls(
            case_id=d.get("case_id", ""),
            operation_id=d.get("operation_id"),
            readout_tag=d.get("readout_tag"),
            episode_id=d.get("episode_id"),
            group_id=d.get("group_id"),
            scene_id=d.get("scene_id"),
            pose=d.get("pose") or [],
            sensor_profile=d.get("sensor_profile") or {},
            body=d.get("body") or {},
            action=d.get("action") or {},
            d_safe_visible_m=d.get("d_safe_visible_m"),
            d_safe_visible_body_widths=d.get("d_safe_visible_body_widths"),
            d_safe_navmesh_m=d.get("d_safe_navmesh_m"),
            margin_body_widths=d.get("margin_body_widths"),
            oracle_agreement=d.get("oracle_agreement"),
            contact_point_3d=d.get("contact_point_3d"),
            contact_pixel=d.get("contact_pixel"),
            requires_hidden_geometry=d.get("requires_hidden_geometry"),
            fov_supported=d.get("fov_supported"),
            gates=d.get("gates") or {},
            image_path=d.get("image_path"),
            question=d.get("question"),
            answer=d.get("answer") or {},
            tags=d.get("tags") or {},
        )

    def model_payload(self) -> Dict[str, Any]:
        """Return ONLY the two fields the model is allowed to see.

        This is the single enforcement point: depth, pose, navmesh, oracle
        data are never included here, even if they exist on the Case.
        """
        return {
            "image_path": self.image_path,
            "question": self.question,
        }


# ── JSONL I/O ────────────────────────────────────────────────────────────────

def write_jsonl(cases: List[Case], path: "pathlib.Path | str") -> None:
    """Write a list of Case objects to a JSONL file (one JSON object per line)."""
    path = pathlib.Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: "pathlib.Path | str") -> List[Case]:
    """Read a JSONL file and return a list of Case objects."""
    path = pathlib.Path(path)
    cases: List[Case] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(Case.from_dict(json.loads(line)))
    return cases
