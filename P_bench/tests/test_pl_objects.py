"""Object extraction + contact attribution tests (pure numpy)."""

import numpy as np
import pytest

from pipeline import config, perception, objects
from tests._synthetic import LEVEL_FLOOR, make_frame

K = config.intrinsics()


def _synth_frame_arrays(semantic):
    """Depth=2.0 everywhere -> valid-pixel ground arrays matching `semantic`."""
    depth = np.full(semantic.shape, 2.0, dtype=np.float32)
    pts_cam, uv = perception.unproject(depth, K)
    pts = perception.to_agent_ground(pts_cam)
    sem = semantic[uv[:, 1], uv[:, 0]].astype(np.int64)
    return pts, uv, sem


def test_extract_objects_filters_and_fields():
    H, W = 100, 100
    semantic = np.zeros((H, W), dtype=np.int64)
    semantic[10:40, 10:40] = 5      # 900 px -> kept
    semantic[0:5, 0:5] = 3          # 25 px  -> dropped (area)
    pts, uv, sem = _synth_frame_arrays(semantic)
    id_to_cat = {5: "sofa", 3: "cup"}

    objs = objects.extract_objects(pts, uv, sem, id_to_cat)
    ids = [o["instance_id"] for o in objs]
    assert ids == [5]                # 0 and 3 dropped
    o = objs[0]
    assert o["category"] == "sofa" and o["is_structural"] is False
    assert o["mask_area_px"] == 900  # assigned valid-depth pixels
    assert o["n_points_raw"] == 900
    assert 1.5 < o["dist_centroid_m"] < 3.0
    assert "_points_xz" in o


def test_extract_objects_structural_flag():
    semantic = np.zeros((60, 60), dtype=np.int64)
    semantic[5:55, 5:55] = 2
    pts, uv, sem = _synth_frame_arrays(semantic)
    objs = objects.extract_objects(pts, uv, sem, {2: "wall"})
    assert objs[0]["is_structural"] is True


def test_attribute_majority_vote():
    a = np.tile([0.0, 0.0, 1.0], (10, 1))       # id 5 at (0,1)
    b = np.tile([0.05, 0.0, 1.0], (3, 1))       # id 12 at (0.05,1)
    pts = np.vstack([a, b])
    sem = np.array([5] * 10 + [12] * 3)
    r = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.1, id_to_cat={5: "sofa", 12: "table"})
    assert r["instance_id"] == 5 and r["category"] == "sofa"
    assert abs(r["vote_fraction"] - 10 / 13) < 1e-9
    assert r["votes"] == {"5": 10, "12": 3}


def test_attribute_tie_breaks_lowest_id():
    pts = np.tile([0.0, 0.0, 1.0], (10, 1))
    sem = np.array([5] * 5 + [8] * 5)
    r = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.1, id_to_cat={5: "a", 8: "b"})
    assert r["instance_id"] == 5


def test_attribute_relax_once():
    # points at distance 0.15; radius 0 + margin 0.10 -> first pass (0.10) misses,
    # relax (0.20) catches.
    pts = np.tile([0.15, 0.0, 1.0], (5, 1))
    sem = np.array([7] * 5)
    r = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.0,
                                  id_to_cat={7: "chair"}, margin=0.10)
    assert r["unattributed"] is False and r["instance_id"] == 7


def test_attribute_unattributed_when_far():
    pts = np.tile([1.0, 0.0, 1.0], (5, 1))
    sem = np.array([7] * 5)
    r = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.0,
                                  id_to_cat={7: "chair"}, margin=0.10)
    assert r["unattributed"] is True and r["instance_id"] is None


def test_attribute_ignores_unlabelled_zero():
    pts = np.vstack([np.tile([0.0, 0.0, 1.0], (20, 1)),
                     np.tile([0.02, 0.0, 1.0], (5, 1))])
    sem = np.array([0] * 20 + [9] * 5)
    r = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.1, id_to_cat={9: "lamp"})
    assert r["instance_id"] == 9


def test_attribute_height_band_excludes_floor_and_overhang():
    # Contact disk at (0,1): 20 floor points (y=0), 3 knee-height obstacle points
    # (y=0.3), 2 table-top overhang points (y=1.0). All inside the xz disk.
    floor = np.tile([0.0, 0.0, 1.0], (20, 1))     # id 3
    obstacle = np.tile([0.0, 0.3, 1.0], (3, 1))   # id 7
    overhang = np.tile([0.0, 1.0, 1.0], (2, 1))   # id 11
    pts = np.vstack([floor, obstacle, overhang])
    sem = np.array([3] * 20 + [7] * 3 + [11] * 2)
    id_to_cat = {3: "floor", 7: "chair", 11: "table"}

    # No band (legacy default): floor points win by count -> wrong attribution.
    no_band = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.1, id_to_cat=id_to_cat)
    assert no_band["instance_id"] == 3

    # Tall body band (0.05, 1.5): floor excluded, real obstacle wins.
    tall = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.1,
                                     id_to_cat=id_to_cat, height_band=(0.05, 1.5),
                                     floor_plane=LEVEL_FLOOR)
    assert tall["instance_id"] == 7

    # Short body band top 0.5: the 1.0 m table overhang is out of reach and excluded.
    short = objects.attribute_contact(pts, sem, (0.0, 1.0), radius=0.1,
                                      id_to_cat=id_to_cat, height_band=(0.05, 0.5),
                                      floor_plane=LEVEL_FLOOR)
    assert short["instance_id"] == 7
    assert "11" not in short["votes"]


def test_attribute_height_band_ignores_non_blocking_ground_coverings():
    pts = np.array([
        [0.0, 0.08, 1.0],
        [0.01, 0.08, 1.0],
        [0.0, 0.18, 1.02],
    ])
    sem = np.array([4, 4, 7])

    result = objects.attribute_contact(
        pts, sem, (0.0, 1.0), radius=0.1,
        id_to_cat={4: "rug", 7: "chair"},
        height_band=config.GROUND_OBSTACLE_BAND_M,
        floor_plane=LEVEL_FLOOR)

    assert result["instance_id"] == 7
    assert result["category"] == "chair"


def test_b1k_raw_categories_drive_predicates_but_machine_labels_are_stored():
    """Catches WordNet synsets bypassing raw structural/contact exclusions."""
    pts = np.array([
        [0.00, 0.10, 1.00], [0.02, 0.10, 1.00],
        [0.00, 0.10, 1.20], [0.02, 0.10, 1.20],
    ])
    uv = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
    sem = np.array([3, 3, 7, 7])
    machine = {3: "floor.n.01", 7: "chair.n.01"}
    raw = {3: "floor", 7: "straight_chair"}

    extracted = objects.extract_objects(
        pts, uv, sem, machine, predicate_categories=raw,
        min_area_px=1, min_valid=1)

    assert [value["category"] for value in extracted] == [
        "floor.n.01", "chair.n.01"]
    assert [value["is_structural"] for value in extracted] == [True, False]
    assert [value["_predicate_category"] for value in extracted] == [
        "floor", "straight_chair"]
    attributed = objects.attribute_contact(
        pts, sem, (0.01, 1.0), 0.1, machine,
        predicate_categories=raw, height_band=(0.05, 0.30),
        floor_plane=LEVEL_FLOOR)
    assert attributed["instance_id"] == 7
    assert attributed["category"] == "chair.n.01"
    assert "3" not in attributed["votes"]


def test_partial_predicate_category_map_falls_back_per_instance():
    """Catches one raw-label entry erasing unrelated machine categories."""
    points = np.array([[0.0, 0.15, 1.0], [0.01, 0.15, 1.0]])
    result = objects.attribute_contact(
        points, np.array([7, 7]), (0.0, 1.0), 0.1,
        {7: "chair.n.01"}, predicate_categories={3: "floor"},
        height_band=config.GROUND_OBSTACLE_BAND_M,
        floor_plane=LEVEL_FLOOR)

    assert result["instance_id"] == 7
    assert result["category"] == "chair.n.01"


def test_initial_visible_entity_inventory_keeps_only_category_identity():
    record = {
        "objects": [
            {"instance_id": 9, "category": "chair", "centroid_px": [90, 20]},
            {"instance_id": 3, "category": "table", "centroid_px": [30, 40]},
            {"instance_id": 7, "category": "chair", "centroid_px": [70, 25]},
        ],
    }

    assert objects.initial_visible_entity_inventory(record) == [
        {"instance_id": 7, "category": "chair"},
        {"instance_id": 9, "category": "chair"},
        {"instance_id": 3, "category": "table"},
    ]


def test_initial_visible_entity_inventory_is_deterministic_and_gt_blind():
    """Catches candidate order being inherited from an answer-aware caller."""
    first = {
        "objects": [
            {"instance_id": 11, "category": "chair", "centroid_px": [1, 2]},
            {"instance_id": 5, "category": "chair", "centroid_px": [3, 4]},
        ],
    }
    second = {"objects": list(reversed(first["objects"]))}

    assert objects.initial_visible_entity_inventory(first) == \
        objects.initial_visible_entity_inventory(second)
    assert [value["instance_id"] for value in
            objects.initial_visible_entity_inventory(first)] == [5, 11]
