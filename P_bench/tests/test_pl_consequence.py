"""End-to-end judge tests on a synthetic Frame (no Habitat) + record/validate."""

import math

import numpy as np

from pipeline import consequence, record, validate
from pipeline.actions import Turn, Forward
from pipeline.body import Cylinder
from tests._synthetic import make_frame


# --- collision -----------------------------------------------------------

def test_wall_collision_radius():
    fr = make_frame()
    out = consequence.judge(fr, Cylinder(radius_m=0.25), [Forward(5.0)])
    assert out["collided"] is True
    arc = out["first_contact_arc_m"]
    assert 0.55 < arc < 1.0                      # ~ 1 - r - dilation margin
    assert abs(out["contact"]["xy"][1] - arc) < 1e-9
    assert out["contact"]["category"] == "wall"


def test_smaller_radius_contacts_later():
    fr = make_frame()
    a = consequence.judge(fr, Cylinder(radius_m=0.40), [Forward(5.0)])["first_contact_arc_m"]
    b = consequence.judge(fr, Cylinder(radius_m=0.10), [Forward(5.0)])["first_contact_arc_m"]
    assert b > a                                 # thinner body gets closer


def test_pure_turn_no_collision():
    fr = make_frame()
    out = consequence.judge(fr, Cylinder(0.25), [Turn(90)])
    assert out["collided"] is False
    assert out["pose_end_exec"] == out["pose_end_full"]
    assert out["view_exit"]["end_in_fov"] is True     # in-place convention
    assert out["view_exit"]["end_dist_m"] < 1e-9


# --- after-state (object relations) -------------------------------------

def test_after_state_known_answer():
    fr = make_frame()
    out = consequence.judge(fr, Cylinder(0.25), [Turn(90), Forward(1.0)])
    rel = out["object_relations"][0]["full"]
    assert abs(rel["dist_centroid_m"] - math.sqrt(5)) < 1e-6
    assert abs(consequence.A.wrap_deg(rel["bearing_deg"] - (-116.565)) ) < 0.05
    assert rel["in_fov"] is False


# --- view exit -----------------------------------------------------------

def test_view_exit_forward_stays_in_cone():
    fr = make_frame()
    ve = consequence.judge(fr, Cylinder(0.25), [Forward(1.0)])["view_exit"]
    assert ve["path_exit_arc_m"] is None
    assert ve["end_in_fov"] is True


def test_view_exit_right_turn_leaves_cone():
    fr = make_frame()
    ve = consequence.judge(fr, Cylinder(0.25), [Turn(90), Forward(1.0)])["view_exit"]
    assert ve["path_exit_arc_m"] is not None and ve["path_exit_arc_m"] < 0.05
    assert ve["end_in_fov"] is False
    assert abs(ve["end_heading_offset_deg"] - 90.0) < 1e-9


def test_nav_check_null_without_nav():
    fr = make_frame()
    out = consequence.judge(fr, Cylinder(0.25), [Forward(1.0)], nav=None)
    assert out["nav_check"] is None
    for r in out["object_relations"]:
        assert r["full"]["dist_geodesic_m"] is None


# --- refined nav_check verdict (5 branches, artifact-free) ---------------

def _verdict(fr, acts, d_nav, d_depth, radius=0.25):
    return consequence._nav_verdict(fr, acts, d_nav, d_depth, radius)["verdict"]


def test_verdict_keep_agree():
    fr = make_frame()                                    # depth all valid -> fwd_cover=1
    assert _verdict(fr, [Forward(5.0)], d_nav=2.0, d_depth=2.1) == "keep_agree"


def test_verdict_keep_depth_when_navmesh_conservative():
    fr = make_frame()                                    # forward view fully covered
    # navmesh stops early (0.5) but depth marched clear far (2.0) -> trust depth
    assert _verdict(fr, [Forward(5.0)], d_nav=0.5, d_depth=2.0) == "keep_depth"


def test_verdict_review_on_forward_depth_hole():
    fr = make_frame()
    fr.depth = fr.depth.copy()
    H, W = fr.depth.shape
    fr.depth[int(0.25 * H):int(0.90 * H), int(0.30 * W):int(0.70 * W)] = 0.0   # central hole
    assert _verdict(fr, [Forward(5.0)], d_nav=0.5, d_depth=2.0) == "review_depth_hole"


def test_verdict_keep_visible_on_solid_obstacle():
    fr = make_frame()                                    # wall voxels at z=1
    # depth stops earlier (at the wall, 1.0) than navmesh (2.0); strong support -> keep
    assert _verdict(fr, [Forward(5.0)], d_nav=2.0, d_depth=1.0) == "keep_visible"


def test_verdict_discard_noise_on_isolated_support():
    fr = make_frame()
    # depth "stops" at 0.3 where there is no obstacle mass -> noise -> discard (red)
    assert _verdict(fr, [Forward(5.0)], d_nav=2.0, d_depth=0.3) == "discard_noise"


# --- record round-trip + validate ---------------------------------------

def test_record_roundtrip_and_validate_clean(tmp_path):
    fr = make_frame()
    seqs = [[Forward(1.0)], [Turn(90), Forward(1.0)], [Turn(-45)],
            [Turn(30), Forward(0.5), Turn(-30), Forward(0.5)]]
    outcomes = []
    for i, acts in enumerate(seqs):
        oc = consequence.judge(fr, Cylinder(0.25), acts, nav=None)
        oc["outcome_id"] = f"F-test-b025-a{i:02d}"
        outcomes.append(oc)

    rec = record.build_record(fr, outcomes, image_path="img/F-test.png")
    path = tmp_path / "records.jsonl"
    record.append_record(str(path), rec)

    loaded = list(record.read_records(str(path)))
    assert len(loaded) == 1
    assert loaded[0]["schema_version"] == "conseq.v1"
    # transient field must not survive serialization
    assert "_points_xz" not in loaded[0]["objects"][0]

    violations = validate.validate_record(loaded[0])
    assert violations == [], "\n".join(violations)
