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
from pipeline.io_utils import atomic_write_binary, link_or_copy_file, sha256_file


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
        source_image, *, expected_sha256: str,
        expected_resolution,
        authenticated: AuthenticatedRGB | None = None) -> AuthenticatedRGB:
    """Authenticate native RGB bytes and the manifest-only sensor raster."""
    from PIL import Image

    if (not isinstance(expected_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("raw RGB expected digest is invalid")
    source_path = Path(source_image)
    if authenticated is not None:
        if not isinstance(authenticated, AuthenticatedRGB):
            raise TypeError("cached RGB must be an AuthenticatedRGB")
        if (authenticated.resolved_source_path !=
                str(source_path.resolve(strict=True)) or
                authenticated.sha256 != expected_sha256 or
                (authenticated.width_px, authenticated.height_px) !=
                tuple(expected_resolution)):
            raise ValueError("cached RGB disagrees with requested source")
        return authenticated
    raw_bytes = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
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


def numbered_dot_placement(markers, *, width: int,
                           height: int) -> tuple[list, str | None]:
    """Resolve marker pixels, or say why this set cannot be drawn.

    A record may legitimately carry a marker the renderer cannot draw: the
    inventory only requires a finite centroid inside the raster, while the dot
    needs a whole radius of margin. Answering with a reason instead of raising
    lets the question be refused at eligibility time, where every other
    unpublishable case is already handled, rather than aborting a whole build.
    """
    radius = int(config.A3_MARKER_RADIUS_PX)
    rendered = []
    for marker in markers:
        if (not isinstance(marker, dict) or
                marker.get("kind") != "numbered_dot"):
            return [], "invalid numbered-dot marker"
        number = marker.get("number")
        center = marker.get("center_xy")
        if (not isinstance(number, int) or isinstance(number, bool) or
                number <= 0 or not isinstance(center, (list, tuple)) or
                len(center) != 2):
            return [], "invalid numbered-dot marker"
        try:
            x_float, y_float = (float(value) for value in center)
        except (TypeError, ValueError):
            return [], "invalid numbered-dot marker"
        if not (math.isfinite(x_float) and math.isfinite(y_float)):
            return [], "invalid numbered-dot marker"
        x, y = int(round(x_float)), int(round(y_float))
        if (x - radius < 0 or y - radius < 0 or
                x + radius >= int(width) or y + radius >= int(height)):
            return [], "numbered-dot marker is outside image bounds"
        if any(math.hypot(x - prior_x, y - prior_y) <
               config.A3_MARKER_MIN_SEPARATION_PX
               for prior_x, prior_y, _prior_number in rendered):
            return [], "numbered-dot markers overlap"
        rendered.append((x, y, number))
    if not rendered:
        return [], "numbered-dot image requires at least one marker"
    return rendered, None


def materialize_numbered_dot_image(
        authenticated: AuthenticatedRGB, markers,
        asset_dir, asset_id: str, *, encoded_cache: dict | None = None) -> dict:
    """Create a deterministic lossless PNG with bounded numbered dots."""
    from PIL import Image, ImageDraw, ImageFont

    if (not isinstance(asset_id, str) or
            not re.fullmatch(r"[abd]-[0-9a-f]{16}", asset_id)):
        raise ValueError("invalid numbered-dot asset id")
    if not isinstance(authenticated, AuthenticatedRGB):
        raise ValueError("numbered-dot source must be authenticated RGB")
    source_path = Path(authenticated.resolved_source_path)
    root = Path(asset_dir).resolve()
    destination = (root / f"{asset_id}.png").resolve()
    if destination.parent != root:
        raise ValueError("numbered-dot destination escapes asset directory")
    if source_path == destination:
        raise ValueError("numbered-dot source and destination must differ")
    width, height = authenticated.width_px, authenticated.height_px
    radius = int(config.A3_MARKER_RADIUS_PX)
    rendered, rejection = numbered_dot_placement(
        markers, width=width, height=height)
    if rejection is not None:
        raise ValueError(rejection)
    if not rendered:
        raise ValueError("numbered-dot image requires at least one marker")
    cache_key = (
        authenticated.sha256, width, height, radius, tuple(rendered))
    cached = encoded_cache.get(cache_key) if encoded_cache is not None else None
    if cached is None:
        with Image.open(io.BytesIO(authenticated.raw_bytes)) as opened:
            image = opened.copy()
            image.load()
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for x, y, number in rendered:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(220, 0, 0), outline=(255, 255, 255), width=2)
            label = str(number)
            left, top, right, bottom = draw.textbbox(
                (0, 0), label, font=font)
            draw.text(
                (x - (right - left) / 2 - left,
                 y - (bottom - top) / 2 - top),
                label, fill=(255, 255, 255), font=font)
        payload = io.BytesIO()
        image.save(
            payload, format="PNG", optimize=False,
            compress_level=config.QA_MARKED_PNG_COMPRESSION_LEVEL)
        expected_bytes = payload.getvalue()
        expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
        if destination.exists():
            if sha256_file(destination) != expected_sha256:
                raise ValueError(
                    "conflicting numbered-dot derivative already exists")
        else:
            atomic_write_binary(
                destination, lambda stream: stream.write(expected_bytes))
        if encoded_cache is not None:
            encoded_cache[cache_key] = {
                "path": str(destination), "sha256": expected_sha256}
    else:
        if (not isinstance(cached, dict) or
                cached.get("sha256") is None or cached.get("path") is None):
            raise TypeError("numbered-dot encoded cache entry is invalid")
        expected_sha256 = str(cached["sha256"])
        cached_path = Path(str(cached["path"]))
        if destination.exists():
            if sha256_file(destination) != expected_sha256:
                raise ValueError(
                    "conflicting numbered-dot derivative already exists")
        else:
            linked = link_or_copy_file(cached_path, destination)
            if not linked and sha256_file(destination) != expected_sha256:
                destination.unlink(missing_ok=True)
                raise ValueError("copied numbered-dot derivative changed")
    return {
        "marked": True,
        "path": str(destination),
        "sha256": expected_sha256,
        "raw_path": authenticated.source_path,
        "raw_sha256": authenticated.sha256,
        "width_px": int(width),
        "height_px": int(height),
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
