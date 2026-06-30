import numpy as np
from egoconseq.geometry import swept_path

def test_forward_path_length_and_heading():
    pts = swept_path(turn_deg=0, forward_m=1.0, step=0.02)
    assert abs(pts[0][0]) < 1e-9 and abs(pts[0][1]) < 1e-9      # starts at origin
    assert abs(pts[-1][1] - 1.0) < 0.02                         # ends ~1m forward (+z)
    assert all(abs(p[2] - 0.0) < 1e-9 for p in pts)            # heading unchanged

def test_turn_then_forward_heading():
    pts = swept_path(turn_deg=15, forward_m=1.0, step=0.02)
    h = np.deg2rad(15)
    assert all(abs(p[2] - h) < 1e-9 for p in pts)              # all samples carry new heading
    # final point displaced along rotated heading
    assert pts[-1][0] > 0 and pts[-1][1] > 0
