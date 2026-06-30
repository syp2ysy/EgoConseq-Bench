from egoconseq import config as C

def test_frozen_constants_match_design():
    assert C.HFOV_DEG == 79
    assert C.RESOLUTION == (640, 480)
    assert C.CAMERA_HEIGHT_M == 1.5
    assert C.CYLINDER_HEIGHT_M == 1.5
    assert C.RADII_M == (0.10, 0.25, 0.40)
    assert C.MARCH_STEP_M == 0.02
    assert C.OBSTACLE_BAND_M == (0.05, 1.5)
    assert C.GATE_VISIBLE_SWEEP_RATIO == 0.7
    assert C.GATE_VALID_DEPTH_RATIO == 0.9
    assert C.GATE_DEPTH_HOLE_RATIO == 0.05
    assert C.MARGIN_BODYWIDTHS == 0.5
    assert C.HM3D_VAL_DIR.endswith("hm3d-0.2/hm3d/val")

def test_hw_helper():
    assert C.hw() == [480, 640]
