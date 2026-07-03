import numpy as np
from egoconseq.geometry import swept_path, project_contact
from egoconseq.oracle.pointcloud import backproject, to_agent_ground
from egoconseq import config


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


# ---------------------------------------------------------------------------
# Tests for project_contact (O6 substrate)
# ---------------------------------------------------------------------------

def make_K(W=640, H=480, hfov_deg=79):
    """Build a pinhole K matrix from image size and horizontal FOV."""
    fx = W / (2 * np.tan(np.deg2rad(hfov_deg / 2)))
    fy = fx  # square pixels
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def test_contact_directly_ahead_projects_to_horizontal_center():
    """A contact point on the optical axis should project near image centre u ≈ cx.

    Ground-frame point: (x=0, y=camera_height, z=d) is directly ahead ON THE
    OPTICAL AXIS.  Projecting it should yield u ≈ cx (horizontal centre).
    """
    W, H = 640, 480
    K = make_K(W, H)
    cx = K[0, 2]
    camera_height = config.CAMERA_HEIGHT_M

    # contact at 2 m ahead, directly on optical axis (x=0, y=camera_height, z=2.0)
    point_3d = (0.0, camera_height, 2.0)
    u, v = project_contact(point_3d, K, camera_height=camera_height, pitch=0.0)

    assert abs(u - cx) < 1.0, f"u={u} should be near cx={cx}"


def test_contact_right_of_center_has_larger_u():
    """A contact to the right (+x) should project to larger u than centre."""
    W, H = 640, 480
    K = make_K(W, H)
    cx = K[0, 2]
    camera_height = config.CAMERA_HEIGHT_M

    point_center = (0.0, camera_height, 2.0)
    point_right = (0.5, camera_height, 2.0)

    u_center, _ = project_contact(point_center, K, camera_height=camera_height)
    u_right, _ = project_contact(point_right, K, camera_height=camera_height)

    assert u_right > u_center


def test_project_contact_roundtrip():
    """Round-trip: backproject a pixel → to_agent_ground → project_contact → same pixel.

    This guards the v-sign convention consistency between backproject and project_contact.
    The round-trip error must be < 1 pixel in both u and v.
    """
    W, H = 640, 480
    K = make_K(W, H)
    camera_height = config.CAMERA_HEIGHT_M

    # Original pixel: off-centre, both horizontally and vertically
    u0, v0 = 200.0, 150.0
    depth = 3.0  # metres

    # Step 1: synthesise single-pixel depth image and backproject
    depth_img = np.zeros((H, W), dtype=np.float32)
    depth_img[int(v0), int(u0)] = depth
    pts_cam = backproject(depth_img, K)
    assert len(pts_cam) == 1, "Expected exactly one valid pixel"

    # Step 2: camera frame → agent ground frame
    pts_ground = to_agent_ground(pts_cam, camera_height=camera_height, pitch=0.0)
    point_3d = tuple(pts_ground[0])  # (x, y_ground, z)

    # Step 3: project_contact back to pixel
    u1, v1 = project_contact(point_3d, K, camera_height=camera_height, pitch=0.0)

    # Round-trip error must be < 1 pixel
    assert abs(u1 - u0) < 1.0, f"u round-trip error: {abs(u1-u0):.4f} px"
    assert abs(v1 - v0) < 1.0, f"v round-trip error: {abs(v1-v0):.4f} px"
