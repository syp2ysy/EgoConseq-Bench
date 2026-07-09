"""Object extraction + contact attribution tests (pure numpy)."""

import numpy as np

from pipeline import config, perception, objects

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
