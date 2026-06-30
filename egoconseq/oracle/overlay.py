"""Overlay renderer for the sweep oracle (Task 10).

Draws the swept-corridor footprint and contact marker on an RGB image.

Coordinate conventions (matching project-wide standard)
---------------------------------------------------------
Agent-local ground frame: +z = forward, +x = right, +y = up.
Camera intrinsics K follow the pinhole model in geometry.project_contact.

Public API
----------
draw_sweep_overlay(rgb, path_pixels, contact_pixel=None) -> PIL.Image
    Draw the swept-corridor centerline + left/right edges, and an optional
    contact marker, on a copy of *rgb*.

compute_corridor_pixels(path_samples, floor_y, K, radius, d_safe, camera_height)
    Project the swept-path centerline and left/right edges to pixel coords.
    Returns (center_pixels, left_pixels, right_pixels), each a list of (u,v).
    Only samples up to arc-length d_safe are included.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from egoconseq import config
from egoconseq.geometry import project_contact, swept_path

# Colour scheme (BGR-style names are fine; PIL uses RGB tuples)
_COLOR_CENTER = (255, 230, 0)       # yellow: corridor centerline
_COLOR_EDGE = (255, 140, 0)         # orange: corridor width edges
_COLOR_CONTACT = (255, 50, 50)      # red: contact marker
_COLOR_CONTACT_FILL = (255, 50, 50, 160)  # semi-transparent fill (unused in non-RGBA)

_LINE_WIDTH_CENTER = 2
_LINE_WIDTH_EDGE = 1
_CONTACT_RADIUS_PX = 12  # pixel radius of the contact circle


def compute_corridor_pixels(
    path_samples: List[Tuple[float, float, float]],
    floor_y: float,
    K: np.ndarray,
    radius: float,
    d_safe: float,
    step: float = config.MARCH_STEP_M,
    camera_height: float = config.CAMERA_HEIGHT_M,
    display_height_above_floor: float = 0.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Project swept-path samples to image pixels, up to d_safe arc length.

    For each sample (x, z, heading) in *path_samples*, three points are
    projected:
        - center: (x, floor_y + display_height, z)
        - left:   (x - radius*cos(h), floor_y + display_height, z + radius*sin(h))
        - right:  (x + radius*cos(h), floor_y + display_height, z - radius*sin(h))

    where the left/right offsets are perpendicular to the heading direction.

    Parameters
    ----------
    path_samples : list of (x, z, heading_rad)
        Output of geometry.swept_path.
    floor_y : float
        Floor y-coordinate in the agent ground frame.
    K : (3,3) array
        Camera intrinsic matrix.
    radius : float
        Footprint disk radius in metres (used for left/right edge offset).
    d_safe : float
        Only include samples with arc-length <= d_safe.
    step : float
        Sample spacing in metres (same as passed to swept_path).
    camera_height : float
        Camera height above the floor.
    display_height_above_floor : float
        Y offset above floor_y at which to place the projected corridor
        samples.  0 = on the floor, 0.1 = slightly above floor (avoids
        the very-bottom-of-image crowding for near-floor samples).

    Returns
    -------
    center_pixels, left_pixels, right_pixels : each a list of (u, v) floats
    """
    y_disp = floor_y + display_height_above_floor

    W, H = config.RESOLUTION

    center_pixels: List[Tuple[float, float]] = []
    left_pixels: List[Tuple[float, float]] = []
    right_pixels: List[Tuple[float, float]] = []

    for i, (x, z, heading) in enumerate(path_samples):
        arc_len = i * step
        if arc_len > d_safe + 1e-9:
            break
        if z <= 0:
            # Behind the camera — skip (can happen for extreme turn_deg)
            continue

        # Perpendicular direction (pointing left of heading)
        # heading is yaw from +z axis: heading=0 → +z forward.
        # +x = right, so left perp = (-cos(h), sin(h)) in (x, z) plane
        perp_x = -np.cos(heading)
        perp_z = np.sin(heading)

        c_pt = (x, y_disp, z)
        l_pt = (x + radius * perp_x, y_disp, z + radius * perp_z)
        r_pt = (x - radius * perp_x, y_disp, z - radius * perp_z)

        for pt, store in [(c_pt, center_pixels), (l_pt, left_pixels), (r_pt, right_pixels)]:
            try:
                u, v = project_contact(pt, K, camera_height=camera_height)
                # Only include if approximately within image bounds (generous margin)
                if -W < u < 2 * W and -H < v < 2 * H:
                    store.append((float(u), float(v)))
            except ValueError:
                # z_cam <= 0 (point behind camera)
                pass

    return center_pixels, left_pixels, right_pixels


def draw_sweep_overlay(
    rgb: np.ndarray,
    path_pixels: Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
    ],
    contact_pixel: Optional[Tuple[float, float]] = None,
) -> Image.Image:
    """Draw the swept corridor footprint + optional contact marker on a copy of rgb.

    Parameters
    ----------
    rgb : (H, W, 3) uint8 ndarray
        Source RGB image (will NOT be modified in-place; a copy is made).
    path_pixels : (center_pixels, left_pixels, right_pixels)
        Each is a list of (u, v) float pixel coordinates.  Produced by
        ``compute_corridor_pixels``.  May be empty lists.
    contact_pixel : (u, v) or None
        Pixel location of the first contact point.  Draws a circle if not None.

    Returns
    -------
    PIL.Image.Image (mode "RGB")
        Annotated copy of the input image.
    """
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    center_pixels, left_pixels, right_pixels = path_pixels

    def _to_int_pairs(pts: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        return [(int(round(u)), int(round(v))) for (u, v) in pts]

    # Draw left and right corridor edges
    lpts = _to_int_pairs(left_pixels)
    rpts = _to_int_pairs(right_pixels)

    if len(lpts) >= 2:
        draw.line(lpts, fill=_COLOR_EDGE, width=_LINE_WIDTH_EDGE)
    if len(rpts) >= 2:
        draw.line(rpts, fill=_COLOR_EDGE, width=_LINE_WIDTH_EDGE)

    # Connect corresponding left-right pairs with thin cross-strokes every N samples
    # to make it look like a corridor (every 5th sample to avoid visual clutter)
    n = min(len(lpts), len(rpts))
    stride = max(1, n // 10)
    for k in range(0, n, stride):
        draw.line([lpts[k], rpts[k]], fill=_COLOR_EDGE, width=1)

    # Draw centerline on top of the edges
    cpts = _to_int_pairs(center_pixels)
    if len(cpts) >= 2:
        draw.line(cpts, fill=_COLOR_CENTER, width=_LINE_WIDTH_CENTER)
    elif len(cpts) == 1:
        u, v = cpts[0]
        draw.ellipse(
            [(u - 2, v - 2), (u + 2, v + 2)],
            outline=_COLOR_CENTER,
            fill=_COLOR_CENTER,
        )

    # Draw contact marker
    if contact_pixel is not None:
        u, v = int(round(contact_pixel[0])), int(round(contact_pixel[1]))
        r = _CONTACT_RADIUS_PX
        # Outer ring
        draw.ellipse(
            [(u - r, v - r), (u + r, v + r)],
            outline=_COLOR_CONTACT,
            width=3,
        )
        # Inner crosshair
        draw.line([(u - r, v), (u + r, v)], fill=_COLOR_CONTACT, width=2)
        draw.line([(u, v - r), (u, v + r)], fill=_COLOR_CONTACT, width=2)
        # Small filled dot at centre
        d = 3
        draw.ellipse(
            [(u - d, v - d), (u + d, v + d)],
            fill=_COLOR_CONTACT,
        )

    return img
