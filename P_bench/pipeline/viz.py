"""Debug visualisations: object overlay (RGB) + top-down evidence map.

Pure matplotlib/numpy over a Frame + one judged outcome. No Habitat.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re

import numpy as np

from pipeline import config, perception
from pipeline import actions as A
from pipeline.io_utils import atomic_write_binary


@dataclass(frozen=True)
class AuthenticatedRGB:
    """One immutable raw image snapshot; bytes never enter artifacts."""

    source_path: str
    resolved_source_path: str
    raw_bytes: bytes
    sha256: str
    width_px: int
    height_px: int


def authenticate_raw_rgb_image(
        source_image, *, expected_sha256: str | None = None,
        expected_resolution,
        authenticated: AuthenticatedRGB | None = None) -> AuthenticatedRGB:
    """Authenticate native RGB bytes and the manifest-only sensor raster."""
    from PIL import Image

    source_path = Path(source_image)
    if authenticated is not None:
        if not isinstance(authenticated, AuthenticatedRGB):
            raise TypeError("cached RGB must be an AuthenticatedRGB")
        if (authenticated.resolved_source_path !=
                str(source_path.resolve(strict=True)) or
                (expected_sha256 is not None and
                 authenticated.sha256 != expected_sha256) or
                (authenticated.width_px, authenticated.height_px) !=
                tuple(expected_resolution)):
            raise ValueError("cached RGB disagrees with requested source")
        return authenticated
    raw_bytes = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("raw RGB digest does not match expected digest")
    if (not isinstance(expected_resolution, (list, tuple)) or
            len(expected_resolution) != 2 or
            any(isinstance(value, bool) or not isinstance(value, int) or
                value <= 0 for value in expected_resolution)):
        raise ValueError("record sensor resolution is invalid")
    width, height = expected_resolution
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        if opened.mode != "RGB":
            raise ValueError("raw image must use native RGB mode")
        if opened.size != (width, height):
            raise ValueError("raw image resolution disagrees with record sensor")
        opened.load()
    return AuthenticatedRGB(
        source_path=str(source_path),
        resolved_source_path=str(source_path.resolve(strict=True)),
        raw_bytes=raw_bytes, sha256=actual_sha256,
        width_px=width, height_px=height)


def materialize_target_point_image(
        authenticated: AuthenticatedRGB, surface_anchor: dict,
        destination: Path) -> dict:
    """Mark the exact visible 3D anchor pixel, without drawing a bbox."""
    from PIL import Image, ImageDraw

    pixel = surface_anchor["pixel_xy_px"]
    x, y = (int(value) for value in pixel)
    width, height = authenticated.width_px, authenticated.height_px
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("surface anchor pixel is outside the image")

    with Image.open(io.BytesIO(authenticated.raw_bytes)) as opened:
        image = opened.copy()
        image.load()
    draw = ImageDraw.Draw(image)
    radius = int(config.TARGET_POINT_MARKER_RADIUS_PX)
    for r, color in ((radius, (0, 0, 0)),
                     (radius - 2, (255, 255, 255)),
                     (radius - 4, (220, 24, 24))):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    payload = io.BytesIO()
    image.save(
        payload, format="PNG", optimize=False,
        compress_level=config.QA_MARKED_PNG_COMPRESSION_LEVEL)
    encoded = payload.getvalue()
    digest = hashlib.sha256(encoded).hexdigest()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_binary(destination, lambda stream: stream.write(encoded))
    return {
        "marked": True,
        "path": str(destination),
        "sha256": digest,
        "raw_path": authenticated.source_path,
        "raw_sha256": authenticated.sha256,
        "width_px": width,
        "height_px": height,
        "marker": {
            "kind": "surface_point",
            "style": "red-dot-white-black-outline",
            "radius_px": radius,
            "center_xy": [x, y],
        },
    }


def _color(iid):
    rng = np.random.default_rng(int(iid) * 9973 + 1)
    return rng.integers(60, 255, size=3)


def object_overlay(frame) -> np.ndarray:
    """RGB with each visible instance tinted by its id; returns (H,W,3) uint8."""
    H, W = frame.depth.shape
    inst = np.zeros((H, W), np.int64)
    inst[frame.pts_uv[:, 1], frame.pts_uv[:, 0]] = frame.pts_sem
    out = frame.rgb.copy()
    for o in frame.objects:
        m = inst == o["instance_id"]
        out[m] = (0.45 * out[m] + 0.55 * _color(o["instance_id"])).astype(np.uint8)
    return out


def save_evidence(frame, outcome, path, target_instance=None):
    """RGB overlay + top-down evidence for one outcome -> PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    acts = A.parse_actions(outcome["actions"])
    physical = outcome["physical"]
    execution = outcome["execution"]
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    # -- left: object overlay + contact pixel --
    ax[0].imshow(object_overlay(frame)); ax[0].axis("off")
    ax[0].set_title(f"objects (cov={np.mean(frame.pts_sem>0):.2f})")
    c = physical.get("contact")
    if c and c["pixel"] and c["pixel_in_frame"]:
        ax[0].plot(c["pixel"][0], c["pixel"][1], "o", ms=16, mfc="none", mec="red", mew=2)
        ax[0].text(c["pixel"][0], c["pixel"][1] - 12, f"hit: {c['category']}",
                   color="red", fontsize=9, ha="center")
    # -- right: top-down (x=right, z=forward) --
    a = ax[1]
    om = perception.obstacle_mask(
        frame.pts, frame.floor_plane,
        band=config.GROUND_OBSTACLE_BAND_M)
    ob = frame.pts[om][:, [0, 2]]
    if ob.shape[0] > 6000:
        ob = ob[np.linspace(0, ob.shape[0] - 1, 6000).astype(int)]
    a.scatter(ob[:, 0], ob[:, 1], s=1, c="0.7", label="obstacles")
    # Current sensor view cone.
    R = 4.0
    for s in (+1, -1):
        ang = math.radians(s * frame.sensor.hfov_deg / 2.0)
        a.plot([0, R * math.sin(ang)], [0, R * math.cos(ang)], "b--", lw=1)
    # path + endpoint
    samples = A.sample_path(acts, config.MARCH_STEP_M)
    px = [p[0] for p in samples]; pz = [p[1] for p in samples]
    a.plot(px, pz, "g-", lw=2, label="path")
    pf = execution["nominal_pose"]; h = math.radians(pf["heading_deg"])
    a.arrow(pf["x"], pf["z"], 0.4 * math.sin(h), 0.4 * math.cos(h),
            head_width=0.12, color="green")
    # body swept corridor: chassis footprint (radius) along the path
    r = outcome["body"]["radius_m"]
    step = max(1, int(round(r / config.MARCH_STEP_M)))   # ~1 circle per radius of travel
    for i in range(0, len(samples), step):
        a.add_patch(Circle((samples[i][0], samples[i][1]), r,
                           fill=False, ec="green", lw=0.6, alpha=0.35))
    a.add_patch(Circle((0, 0), r, fill=False, ec="blue", lw=1.5,
                       label=f"body r={r:.2f}m"))         # start footprint
    a.add_patch(Circle((pf["x"], pf["z"]), r, fill=False, ec="green", lw=1.5))  # end footprint
    # contact
    if c:
        a.plot(c["xy"][0], c["xy"][1], "rx", ms=12, mew=3, label="contact")
        a.add_patch(Circle((c["xy"][0], c["xy"][1]), r, fill=False, ec="red", lw=1.5))
    # objects + before/after distance line to target
    for o in frame.objects:
        if o["is_structural"]:
            continue
        gx, gz = o["ground_xy_centroid"]
        a.plot(gx, gz, "k.", ms=4)
        a.text(gx, gz, o["category"], fontsize=6, color="black")
        if target_instance is not None and o["instance_id"] == target_instance:
            a.plot([0, gx], [0, gz], "c--", lw=1)                 # before
            a.plot([pf["x"], gx], [pf["z"], gz], "c-", lw=1.5)    # after
    a.plot(0, 0, "b^", ms=10)
    xs = [0, pf["x"]] + px; zs = [0, pf["z"]] + pz   # ensure body circles stay in view
    a.set_xlim(min(a.get_xlim()[0], min(xs) - r), max(a.get_xlim()[1], max(xs) + r))
    a.set_ylim(min(a.get_ylim()[0], min(zs) - r), max(a.get_ylim()[1], max(zs) + r))
    a.set_aspect("equal"); a.grid(alpha=0.3); a.set_xlabel("x (right)"); a.set_ylabel("z (fwd)")
    a.set_title(
        f"topdown collision={physical['collision']} "
        f"arc={physical.get('first_contact_arc_m')}")
    a.legend(loc="upper right", fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()
