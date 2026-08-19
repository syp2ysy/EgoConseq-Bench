"""Public/private QA pair builders over eligible rollout outcomes."""

from __future__ import annotations

import hashlib
import json
from typing import Dict

from pipeline import future_view_selection
from pipeline.benchmark import C_TASK_QUESTIONS, PUBLIC_HEIGHT_DECIMALS
from pipeline.benchmark_tasks import (
    build_a_candidate_projection, build_b_candidate_projection,
)


def build_a_candidate(*, task_id: str, record: Dict, outcome: Dict,
                      image: str, expected_raw_image_sha256: str,
                      asset_dir=None, authenticated_rgb=None,
                      a_eligibility_evidence=None) -> tuple[dict, dict]:
    """Build one v16 Closed Exact A item alongside its private answer."""
    return build_a_candidate_projection(
        task_id=task_id, record=record, outcome=outcome, image=image,
        stable_id=_stable_id,
        expected_raw_image_sha256=expected_raw_image_sha256,
        asset_dir=asset_dir, authenticated_rgb=authenticated_rgb,
        evidence=a_eligibility_evidence)


def build_b_candidate(*, task_id: str, record: Dict, outcome: Dict,
                      image: str, expected_raw_image_sha256: str,
                      asset_dir=None, authenticated_rgb=None,
                      numbered_dot_cache=None,
                      b_outcome_evidence=None) -> tuple[dict, dict]:
    """Build one v16 Closed Exact B item alongside its private answer."""
    return build_b_candidate_projection(
        task_id=task_id, record=record, outcome=outcome, image=image,
        stable_id=_stable_id,
        expected_raw_image_sha256=expected_raw_image_sha256,
        asset_dir=asset_dir, authenticated_rgb=authenticated_rgb,
        numbered_dot_cache=numbered_dot_cache,
        evidence=b_outcome_evidence)


def build_c_candidate(*, record: Dict, outcome: Dict, image: str,
                      expected_raw_image_sha256: str,
                      asset_root, selection: dict,
                      authenticated_rgb=None) -> tuple[dict, dict]:
    """Build the frozen v16 Closed Exact C1 item and private answer."""
    return future_view_selection.project_c_candidate(
        rec=record, outcome=outcome, selection=selection, image=image,
        expected_raw_image_sha256=expected_raw_image_sha256,
        asset_root=asset_root,
        question=C_TASK_QUESTIONS["C1_future_view_selection"],
        stable_id=_stable_id,
        public_height_decimals=PUBLIC_HEIGHT_DECIMALS,
        authenticated_rgb=authenticated_rgb)

def _stable_id(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
