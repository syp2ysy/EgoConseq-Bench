"""Committed A1 control actions must not drift with the live sampler."""

from __future__ import annotations

from collections import Counter
import importlib

from pipeline import actions as A


EXPECTED_TAGS = {
    "K1-00", "K1-01", "K1-02",
    "K2-01", "K2-02",
    "K3-01", "K3-02",
    "K4-02",
    "K5-01", "K5-02", "K5-03",
    "K6-00", "K6-02",
}


def test_committed_catalog_has_the_measured_thirteen_actions():
    catalog = importlib.import_module("pipeline.action_control_catalog")

    anchors = catalog.load_catalog()

    assert {anchor.tag for anchor in anchors} == EXPECTED_TAGS
    assert {anchor.length for anchor in anchors} == {1, 2, 3, 4, 5, 6}
    assert all(len(anchor.actions) == anchor.length for anchor in anchors)
    assert all(A.total_forward_m(anchor.actions) <= 6.0 for anchor in anchors)
    assert catalog.catalog_sha256() == catalog.CATALOG_SHA256


def test_committed_catalog_is_read_once_per_process():
    catalog = importlib.import_module("pipeline.action_control_catalog")
    catalog.load_catalog.cache_clear()

    first = catalog.load_catalog()
    second = catalog.load_catalog()

    assert first is second
    assert catalog.load_catalog.cache_info().misses == 1
    assert catalog.load_catalog.cache_info().hits == 1


def test_five_anchor_rotation_is_balanced_over_thirteen_poses():
    catalog = importlib.import_module("pipeline.action_control_catalog")

    scheduled = [
        anchor.tag
        for pose_index in range(13)
        for anchor in catalog.anchors_for_pose(
            dataset="r2r", scene_id="scene-a", pose_index=pose_index)
    ]

    assert len(scheduled) == 13 * 5
    assert Counter(scheduled) == Counter({tag: 5 for tag in EXPECTED_TAGS})
    assert catalog.anchors_for_pose(
        dataset="r2r", scene_id="scene-a", pose_index=3
    ) == catalog.anchors_for_pose(
        dataset="r2r", scene_id="scene-a", pose_index=3
    )
