"""C1 terminal-image authority and counterfactual selection primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import stat
from typing import Mapping, Optional

import numpy as np
from PIL import Image

from pipeline import actions as action_geometry
from pipeline import config, dataset_contracts, io_utils
from pipeline import outcome as outcome_fields
from pipeline import perception, record
from pipeline import scene_pool, viz
from pipeline.benchmark import (
    SharedVisibleSpaceEvidence, shared_visible_space_certificate,
)
from pipeline.frame import Frame, TerminalRGBObservation


TERMINAL_RGB_ASSET_SCHEMA = "terminal-rgb-asset.v1"
RENDERER_PROTOCOL = "habitat-sim-rgb.v1"
B1K_RENDERER_PROTOCOL = "b1k-observation.v5"
GS_RENDERER_PROTOCOL = "gsplat-rgb-ed.v1"
BLOCK_L1_PROTOCOL = "block-l1.v1"
TERMINAL_ASSET_RECORD_AUTHORITY = "record_recomputable"
TERMINAL_ASSET_RUNTIME_AUTHORITY = "collection_runtime_attested"
TERMINAL_ASSET_WITHHOLD_REASON_AUTHORITIES = {
    "terminal_checkpoint_missing": frozenset({
        TERMINAL_ASSET_RECORD_AUTHORITY}),
    "terminal_checkpoint_incomplete": frozenset({
        TERMINAL_ASSET_RECORD_AUTHORITY}),
    "terminal_checkpoint_pose_invalid": frozenset({
        TERMINAL_ASSET_RECORD_AUTHORITY}),
    "terminal_pose_disagreement": frozenset({
        TERMINAL_ASSET_RECORD_AUTHORITY}),
    "terminal_cache_miss": frozenset({
        TERMINAL_ASSET_RUNTIME_AUTHORITY}),
    "terminal_render_transaction_unavailable": frozenset({
        TERMINAL_ASSET_RUNTIME_AUTHORITY}),
    "terminal_provenance_invalid": frozenset({
        TERMINAL_ASSET_RECORD_AUTHORITY,
        TERMINAL_ASSET_RUNTIME_AUTHORITY,
    }),
    "terminal_encoding_invalid": frozenset({
        TERMINAL_ASSET_RUNTIME_AUTHORITY}),
    "terminal_publication_invalid": frozenset({
        TERMINAL_ASSET_RUNTIME_AUTHORITY}),
}
TERMINAL_ASSET_WITHHOLD_REASONS = frozenset(
    TERMINAL_ASSET_WITHHOLD_REASON_AUTHORITIES)


class TerminalRGBAssetError(ValueError):
    """Expected terminal-asset failure with a frozen attribution authority."""

    def __init__(self, message: str, *, reason: str, authority: str):
        legal = TERMINAL_ASSET_WITHHOLD_REASON_AUTHORITIES.get(reason)
        if legal is None or authority not in legal:
            raise ValueError(
                "terminal RGB reason/authority pair is not frozen")
        super().__init__(message)
        self.reason = reason
        self.authority = authority


def _terminal_asset_error(
        message: str, *, reason: str,
        authority: str = TERMINAL_ASSET_RECORD_AUTHORITY,
        cause: Optional[Exception] = None) -> TerminalRGBAssetError:
    error = TerminalRGBAssetError(
        message, reason=reason, authority=authority)
    if cause is not None:
        error.__cause__ = cause
    return error


@dataclass(frozen=True)
class EncodedNativeRGB:
    """One deterministic lossless encoding of a native RGB raster."""

    png_bytes: bytes
    png_sha256: str
    pixel_sha256: str
    mode: str
    width_px: int
    height_px: int


@dataclass(frozen=True)
class _DecodedNativeRGB:
    """Validated encoding plus the native pixels decoded from it once."""

    encoded: EncodedNativeRGB
    pixels: np.ndarray


def _resolution(value) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(item, bool) or not isinstance(item, int) or
                item <= 0 for item in value)):
        raise ValueError("native RGB resolution is invalid")
    return int(value[0]), int(value[1])


def _pixel_sha256(rgb: np.ndarray) -> str:
    height, width, _channels = rgb.shape
    header = f"rgb8-pixels.v1\0{width}\0{height}\0".encode("ascii")
    return hashlib.sha256(header + np.ascontiguousarray(rgb).tobytes()).hexdigest()


def encode_native_rgb_png(
        rgb: np.ndarray, *, expected_resolution) -> EncodedNativeRGB:
    """Encode native uint8 HxWx3 RGB without conversion or resizing."""
    width, height = _resolution(expected_resolution)
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise ValueError("terminal RGB must be uint8")
    if array.shape != (height, width, 3):
        raise ValueError("terminal RGB shape disagrees with native resolution")
    canonical = np.ascontiguousarray(array)
    output = io.BytesIO()
    Image.fromarray(canonical, mode="RGB").save(
        output, format="PNG", optimize=False, compress_level=9)
    payload = output.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        decoded.load()
        if decoded.mode != "RGB" or decoded.size != (width, height):
            raise ValueError("terminal PNG mode or size changed during encoding")
        if not np.array_equal(np.asarray(decoded), canonical):
            raise ValueError("terminal PNG is not a lossless pixel encoding")
    return EncodedNativeRGB(
        png_bytes=payload,
        png_sha256=hashlib.sha256(payload).hexdigest(),
        pixel_sha256=_pixel_sha256(canonical),
        mode="RGB", width_px=width, height_px=height,
    )


def _terminal_checkpoint(outcome: dict) -> dict:
    checkpoints = outcome.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise _terminal_asset_error(
            "terminal checkpoint is missing",
            reason="terminal_checkpoint_missing")
    terminal = checkpoints[-1]
    if not isinstance(terminal, dict):
        raise _terminal_asset_error(
            "terminal checkpoint is not a completed endpoint",
            reason="terminal_checkpoint_incomplete")
    try:
        requested_progress = float(terminal["requested_progress"])
        realized_progress = float(terminal["realized_progress"])
    except (KeyError, TypeError, ValueError) as error:
        raise _terminal_asset_error(
            "terminal checkpoint is not a completed endpoint",
            reason="terminal_checkpoint_incomplete", cause=error)
    if (not math.isfinite(requested_progress) or
            not math.isfinite(realized_progress) or
            abs(requested_progress - 1.0) > 1e-9 or
            abs(realized_progress - 1.0) > 1e-9):
        raise _terminal_asset_error(
            "terminal checkpoint is not a completed endpoint",
            reason="terminal_checkpoint_incomplete")
    pose = terminal.get("pose")
    if not isinstance(pose, dict):
        raise _terminal_asset_error(
            "terminal checkpoint pose is invalid",
            reason="terminal_checkpoint_pose_invalid")
    try:
        expected = outcome_fields.realized_pose(outcome)
        values = {
            key: float(pose[key]) for key in ("x", "z", "heading_deg")}
        endpoint_values = {
            key: float(expected[key])
            for key in ("x", "z", "heading_deg")}
    except (KeyError, TypeError, ValueError) as error:
        raise _terminal_asset_error(
            "terminal checkpoint pose is invalid",
            reason="terminal_checkpoint_pose_invalid", cause=error)
    position_error = max(
        abs(values[axis] - endpoint_values[axis]) for axis in ("x", "z"))
    heading_error = abs(action_geometry.wrap_deg(
        values["heading_deg"] - endpoint_values["heading_deg"]))
    if (not all(math.isfinite(value) for value in (
            *values.values(), *endpoint_values.values())) or
            position_error > config.C1_TERMINAL_POSITION_TOL_M or
            heading_error > config.C1_TERMINAL_HEADING_TOL_DEG):
        raise _terminal_asset_error(
            "terminal checkpoint disagrees with realized endpoint",
            reason="terminal_pose_disagreement")
    return values


def terminal_render_cache_key(outcome: dict) -> tuple[float, float, float]:
    """Return the existing collection cache key for the true terminal pose."""
    pose = _terminal_checkpoint(outcome)
    return tuple(round(pose[key], 6) for key in ("x", "z", "heading_deg"))


def _expected_world_pose(base_frame: Frame, local_pose: dict) -> tuple[np.ndarray, float]:
    position = perception.world_from_local(
        np.asarray([[local_pose["x"], 0.0, local_pose["z"]]],
                   dtype=np.float64),
        base_frame.position, base_frame.yaw_rad,
    )[0]
    yaw = base_frame.yaw_rad - math.radians(local_pose["heading_deg"])
    return position, yaw


def _source_binding(source: dict, *, scene_id: str) -> dict:
    if not isinstance(source, dict):
        raise ValueError("terminal RGB source provenance is missing")
    try:
        scene_pool.validate_source_asset_provenance(source)
    except scene_pool.SceneCatalogError as error:
        raise ValueError("terminal RGB source provenance is invalid") from error
    dataset = source.get("source_dataset")
    if dataset not in {"r2r", "b1k", "gs"}:
        raise ValueError("terminal RGB source dataset is unsupported")
    if (source.get("scene_id") != scene_id or
            source.get("official_split") != "train"):
        label = str(dataset).upper()
        raise ValueError(
            f"terminal RGB source does not match the {label} frame")
    scene_asset = next(
        (asset for asset in source["source_assets"]
         if asset.get("role") == "scene"), None)
    if not isinstance(scene_asset, dict):
        raise ValueError("terminal RGB source lacks its scene asset")
    value = {
        "scene_id": str(scene_id),
        "source_dataset": str(dataset),
        "official_split": "train",
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_assets_sha256": source["source_assets_sha256"],
        "scene_asset_sha256": scene_asset["sha256"],
    }
    if dataset == "r2r":
        return value
    if dataset == "gs":
        binding = dataset_contracts.resolve_gs_collision_binding(source)
        return {
            **value,
            "collision_authority_sha256":
                binding.collision_authority_sha256,
            "source_bundle_authority_sha256": binding.authority_sha256,
        }
    if dataset != "b1k":
        raise ValueError("terminal RGB source dataset is unsupported")
    if source.get("split_authority") != "project_defined":
        raise ValueError("terminal RGB B1K split authority is invalid")
    authority_asset = next(
        (asset for asset in source["source_assets"]
         if asset.get("role") == "scene_authority"), None)
    if not isinstance(authority_asset, dict):
        raise ValueError("terminal RGB source lacks its scene authority")
    return {
        **value,
        "scene_authority_sha256": authority_asset["sha256"],
    }


def _renderer_binding(
        source_binding: dict, *, render_transaction: Optional[str] = None
        ) -> dict:
    dataset = source_binding.get("source_dataset")
    if dataset == "b1k":
        if render_transaction != record.B1K_C1_RENDER_MODE:
            raise ValueError(
                "B1K terminal RGB requires its simultaneous batch "
                "transaction")
        return {
            "protocol": B1K_RENDERER_PROTOCOL,
            "backend": "omnigibson",
            "color_format": "uint8_rgb",
            "source_scene_sha256":
                source_binding["scene_authority_sha256"],
            "observation_profile_sha256":
                record.B1K_OBSERVATION_PROFILE_SHA256,
            "c1_render_mode": record.B1K_C1_RENDER_MODE,
        }
    if dataset == "gs":
        return {
            "protocol": GS_RENDERER_PROTOCOL,
            "backend": "gsplat",
            "color_format": "uint8_rgb",
            "source_scene_sha256": source_binding["scene_asset_sha256"],
            "collision_authority_sha256":
                source_binding["collision_authority_sha256"],
            "source_bundle_authority_sha256":
                source_binding["source_bundle_authority_sha256"],
            "near_m": float(config.GS_RENDER_NEAR_M),
            "far_m": float(config.GS_RENDER_FAR_M),
        }
    return {
        "protocol": RENDERER_PROTOCOL,
        "backend": "habitat_sim",
        "color_format": "uint8_rgb",
        "source_scene_sha256": source_binding["scene_asset_sha256"],
    }


def _validate_cached_terminal(
        base_frame: Frame, terminal, *, local_pose: dict) -> None:
    if not isinstance(terminal, (Frame, TerminalRGBObservation)):
        raise ValueError(
            "cached terminal RGB must be a Frame or RGB observation")
    expected_position, expected_yaw = _expected_world_pose(base_frame, local_pose)
    if not np.array_equal(terminal.position, expected_position):
        raise ValueError("cached terminal pose position disagrees")
    if float(terminal.yaw_rad) != float(expected_yaw):
        raise ValueError("cached terminal pose yaw disagrees")
    if (terminal.scene_id != base_frame.scene_id or
            terminal.scene_glb != base_frame.scene_glb):
        raise ValueError("cached terminal scene disagrees")
    if terminal.sensor.to_dict() != base_frame.sensor.to_dict():
        raise ValueError("cached terminal sensor disagrees")
    if not np.allclose(
            np.asarray(terminal.K), np.asarray(base_frame.K),
            rtol=0.0, atol=1e-6):
        raise ValueError("cached terminal sensor intrinsics disagree")


def materialize_terminal_rgb_asset(
        root, *, base_frame: Frame, outcome: dict,
        render_cache: Mapping, source: dict,
        collection_contract: Optional[dict],
        render_transaction: Optional[str] = None) -> dict:
    """Persist the already-rendered true endpoint as one bound native PNG."""
    dataset = (source or {}).get("source_dataset")
    try:
        source_binding = _source_binding(
            source, scene_id=base_frame.scene_id)
        if dataset == "gs":
            if collection_contract is not None:
                raise ValueError(
                    "GS terminal RGB must not use a legacy collection "
                    "contract")
        else:
            expected_contract = record.collection_contract(source, "main")
            if collection_contract != expected_contract:
                label = "B1K" if dataset == "b1k" else "R2R"
                raise ValueError(
                    f"terminal RGB requires the strict {label} main contract")
            if (dataset == "b1k" and render_transaction !=
                    expected_contract["c1_render_mode"]):
                raise ValueError(
                    "B1K terminal RGB requires its simultaneous batch "
                    "transaction")
    except ValueError as error:
        raise _terminal_asset_error(
            str(error), reason="terminal_provenance_invalid", cause=error)
    local_pose = _terminal_checkpoint(outcome)
    key = terminal_render_cache_key(outcome)
    terminal = render_cache.get(key)
    if terminal is None:
        raise _terminal_asset_error(
            "cached terminal RGB is missing",
            reason="terminal_cache_miss",
            authority=TERMINAL_ASSET_RUNTIME_AUTHORITY)
    try:
        _validate_cached_terminal(base_frame, terminal, local_pose=local_pose)
    except ValueError as error:
        raise _terminal_asset_error(
            str(error), reason="terminal_provenance_invalid",
            authority=TERMINAL_ASSET_RUNTIME_AUTHORITY,
            cause=error)
    sensor = base_frame.sensor.to_dict()
    try:
        encoded = encode_native_rgb_png(
            terminal.rgb, expected_resolution=sensor["resolution"])
    except ValueError as error:
        raise _terminal_asset_error(
            str(error), reason="terminal_encoding_invalid",
            authority=TERMINAL_ASSET_RUNTIME_AUTHORITY,
            cause=error)
    root_path = Path(root).resolve()
    try:
        root_path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _terminal_asset_error(
            "terminal RGB publication directory is unavailable",
            reason="terminal_publication_invalid",
            authority=TERMINAL_ASSET_RUNTIME_AUTHORITY,
            cause=error)
    relative = Path("terminal_rgb") / f"{encoded.png_sha256}.png"
    try:
        _publish_terminal_png(
            root_path, relative.name, encoded.png_bytes)
        if _read_terminal_asset(
                root_path, relative.as_posix()) != encoded.png_bytes:
            raise ValueError("terminal RGB publication bytes changed")
    except (OSError, ValueError) as error:
        raise _terminal_asset_error(
            str(error), reason="terminal_publication_invalid",
            authority=TERMINAL_ASSET_RUNTIME_AUTHORITY,
            cause=error)
    expected_position, expected_yaw = _expected_world_pose(
        base_frame, local_pose)
    binding = {
        "frame_id": str(base_frame.frame_id),
        "outcome_id": str(outcome.get("outcome_id") or ""),
        "base_rollout_key": record.base_rollout_key(base_frame, outcome),
        "action_sha256": record.action_program_sha256(
            outcome.get("actions") or []),
        "terminal_checkpoint_sha256": record.canonical_atom_sha256({
            "requested_progress": 1.0,
            "realized_progress": 1.0,
            "pose": local_pose,
        }),
    }
    if not binding["outcome_id"]:
        raise _terminal_asset_error(
            "terminal RGB outcome id is missing",
            reason="terminal_provenance_invalid")
    value = {
        "schema": TERMINAL_RGB_ASSET_SCHEMA,
        "path": relative.as_posix(),
        "byte_size": len(encoded.png_bytes),
        "png_sha256": encoded.png_sha256,
        "pixel_sha256": encoded.pixel_sha256,
        "mode": encoded.mode,
        "resolution": [encoded.width_px, encoded.height_px],
        "binding": binding,
        "terminal_pose": {
            "local": local_pose,
            "world_position": expected_position.tolist(),
            "world_yaw_rad": float(expected_yaw),
        },
        "sensor": sensor,
        "renderer": _renderer_binding(
            source_binding, render_transaction=render_transaction),
        "source": source_binding,
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def _read_terminal_asset(root, relative_path: str) -> bytes:
    """Read one shard-relative asset, refusing symlinks and escapes.

    O_NOFOLLOW plus the canonical-path check is the whole defence here: the
    threat model has no concurrent or untrusted writer, so the bytes are
    authenticated by the SHA-256 the caller compares afterwards, not by
    re-resolving the descriptor.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("terminal RGB asset path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("terminal RGB asset path must be relative")
    root_path = Path(root).resolve()
    candidate = root_path / relative
    try:
        resolved = candidate.resolve(strict=True)
        normalized = resolved.relative_to(root_path).as_posix()
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            "terminal RGB asset path escapes or is missing") from error
    if normalized != relative.as_posix():
        raise ValueError(
            "terminal RGB asset path is not canonical or is a symlink")
    descriptor = None
    try:
        descriptor = os.open(
            candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("terminal RGB asset is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            return stream.read()
    except OSError as error:
        raise ValueError("terminal RGB asset cannot be opened") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_terminal_png(root: Path, filename: str, payload: bytes) -> None:
    """Publish one terminal PNG, refusing to overwrite different bytes.

    Re-publishing the same asset is expected -- a rebuild produces identical
    bytes -- so an existing file with a matching payload is success. Different
    bytes mean two distinct assets claimed one name, which is a real collision
    and must fail rather than be resolved by last-writer-wins.
    """
    directory = Path(root) / "terminal_rgb"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists():
        try:
            existing = _read_terminal_asset(
                root, f"terminal_rgb/{filename}")
        except ValueError as error:
            raise ValueError(
                "conflicting terminal RGB asset already exists") from error
        if existing != payload:
            raise ValueError("conflicting terminal RGB asset already exists")
        return
    io_utils.atomic_write_binary(
        target, lambda stream: stream.write(payload), durable=True)


def _decode_native_rgb_png(
        payload: bytes, *, expected_resolution,
        expected_png_sha256: str, expected_pixel_sha256: str,
        expected_byte_size: int) -> _DecodedNativeRGB:
    width, height = _resolution(expected_resolution)
    if (len(payload) != expected_byte_size or
            hashlib.sha256(payload).hexdigest() != expected_png_sha256):
        raise ValueError("terminal RGB payload identity changed")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            if (opened.format != "PNG" or opened.mode != "RGB" or
                    opened.size != (width, height)):
                raise ValueError(
                    "terminal RGB payload is not a native RGB PNG")
            pixels = np.ascontiguousarray(np.asarray(opened))
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("terminal RGB payload is not a valid PNG") from error
    pixel_sha256 = _pixel_sha256(pixels)
    if pixel_sha256 != expected_pixel_sha256:
        raise ValueError("terminal RGB pixel identity changed")
    return _DecodedNativeRGB(
        encoded=EncodedNativeRGB(
            png_bytes=payload, png_sha256=expected_png_sha256,
            pixel_sha256=pixel_sha256, mode="RGB",
            width_px=width, height_px=height),
        pixels=pixels)


def validate_terminal_rgb_asset_metadata(rec: dict, outcome: dict) -> dict:
    """Rederive a terminal RGB atom without opening its PNG payload."""
    atom = outcome.get("terminal_rgb_asset")
    if not isinstance(atom, dict) or atom.get("schema") != TERMINAL_RGB_ASSET_SCHEMA:
        raise ValueError("terminal RGB atom is missing or invalid")
    atom_payload = {key: value for key, value in atom.items() if key != "sha256"}
    if atom.get("sha256") != record.canonical_atom_sha256(atom_payload):
        raise ValueError("terminal RGB atom hash is invalid")
    local_pose = _terminal_checkpoint(outcome)
    pose = rec.get("pose") or {}
    try:
        base_position = np.asarray(pose["position"], dtype=np.float64)
        base_yaw = float(pose["yaw_rad"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("terminal RGB record pose is invalid") from error
    if (base_position.shape != (3,) or not np.all(np.isfinite(base_position)) or
            not math.isfinite(base_yaw)):
        raise ValueError("terminal RGB record pose is invalid")
    expected_position = perception.world_from_local(
        np.asarray([[local_pose["x"], 0.0, local_pose["z"]]],
                   dtype=np.float64), base_position, base_yaw,
    )[0]
    expected_yaw = base_yaw - math.radians(local_pose["heading_deg"])
    expected_pose = {
        "local": local_pose,
        "world_position": expected_position.tolist(),
        "world_yaw_rad": float(expected_yaw),
    }
    if atom.get("terminal_pose") != expected_pose:
        raise ValueError("terminal pose binding disagrees with the record")
    sensor = rec.get("sensor")
    if not isinstance(sensor, dict) or atom.get("sensor") != sensor:
        raise ValueError("terminal RGB sensor binding disagrees with the record")
    expected_source = _source_binding(
        rec.get("source"), scene_id=str(rec.get("scene_id") or ""))
    if atom.get("source") != expected_source:
        raise ValueError("terminal RGB source binding disagrees with the record")
    collection_contract = rec.get("collection_contract") or {}
    expected_renderer = _renderer_binding(
        expected_source,
        render_transaction=(
            collection_contract.get("c1_render_mode")
            if expected_source.get("source_dataset") == "b1k" else None),
    )
    if atom.get("renderer") != expected_renderer:
        raise ValueError("terminal RGB renderer binding disagrees with the record")
    expected_binding = {
        "frame_id": str(rec.get("frame_id") or ""),
        "outcome_id": str(outcome.get("outcome_id") or ""),
        "base_rollout_key": record.stored_base_rollout_key(rec, outcome),
        "action_sha256": record.action_program_sha256(
            outcome.get("actions") or []),
        "terminal_checkpoint_sha256": record.canonical_atom_sha256({
            "requested_progress": 1.0,
            "realized_progress": 1.0,
            "pose": local_pose,
        }),
    }
    if (atom.get("binding") != expected_binding or
            outcome.get("base_rollout_key") != expected_binding["base_rollout_key"]):
        raise ValueError("terminal RGB binding disagrees with the record")
    resolution = atom.get("resolution")
    if (resolution != sensor.get("resolution") or atom.get("mode") != "RGB"):
        raise ValueError("terminal RGB native resolution is invalid")
    path = atom.get("path")
    if path != f"terminal_rgb/{atom.get('png_sha256')}.png":
        raise ValueError("terminal RGB asset path is not content addressed")
    return atom


def _validate_terminal_rgb_asset(
        rec: dict, outcome: dict, *, asset_root) -> _DecodedNativeRGB:
    """Rederive one terminal atom and decode its pinned PNG exactly once."""
    atom = validate_terminal_rgb_asset_metadata(rec, outcome)
    path = atom["path"]
    payload = _read_terminal_asset(asset_root, path)
    return _decode_native_rgb_png(
        payload,
        expected_resolution=atom["resolution"],
        expected_png_sha256=atom.get("png_sha256"),
        expected_pixel_sha256=atom.get("pixel_sha256"),
        expected_byte_size=atom.get("byte_size"),
    )


def validate_terminal_rgb_asset(
        rec: dict, outcome: dict, *, asset_root) -> EncodedNativeRGB:
    """Rederive one terminal atom from record fields and pinned PNG bytes."""
    return _validate_terminal_rgb_asset(
        rec, outcome, asset_root=asset_root).encoded


def validate_terminal_rgb_asset_with_pixels(
        rec: dict, outcome: dict, *, asset_root
        ) -> tuple[EncodedNativeRGB, np.ndarray]:
    """Return one authenticated terminal encoding and its decoded RGB array."""
    decoded = _validate_terminal_rgb_asset(
        rec, outcome, asset_root=asset_root)
    return decoded.encoded, decoded.pixels


def block_l1_features(rgb: np.ndarray) -> np.ndarray:
    """Return uint64 channel sums over native non-overlapping 8x8 blocks."""
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("block-l1.v1 requires native uint8 HxWx3 RGB")
    height, width, _channels = array.shape
    if height <= 0 or width <= 0 or height % 8 or width % 8:
        raise ValueError("block-l1.v1 requires H and W divisible by 8")
    blocks = np.ascontiguousarray(array).reshape(
        height // 8, 8, width // 8, 8, 3)
    return blocks.sum(axis=(1, 3), dtype=np.uint64)


def block_l1_certificate(left: np.ndarray, right: np.ndarray) -> dict:
    """Compute an order-independent exact rational native appearance distance."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError("block-l1.v1 images must have identical native shape")
    left_features = block_l1_features(left_array)
    right_features = block_l1_features(right_array)
    return _block_l1_certificate_from_features(
        left_array, right_array, left_features, right_features)


def _block_l1_certificate_from_features(
        left: np.ndarray, right: np.ndarray,
        left_features: np.ndarray, right_features: np.ndarray) -> dict:
    """Build the exact pair atom from already cached block sums."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (left_array.shape != right_array.shape or
            left_features.shape != right_features.shape):
        raise ValueError("block-l1.v1 cached feature shapes disagree")
    delta = left_features.astype(np.int64) - right_features.astype(np.int64)
    numerator = int(np.abs(delta).sum(dtype=np.uint64))
    height, width, _channels = left_array.shape
    denominator = int(height * width * 3 * 255)
    identities = sorted((_pixel_sha256(left_array), _pixel_sha256(right_array)))
    value = {
        "protocol": BLOCK_L1_PROTOCOL,
        "block_size_px": 8,
        "resolution": [int(width), int(height)],
        "pair_pixel_sha256": identities,
        "numerator": numerator,
        "denominator": denominator,
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def _action_summary(outcome: dict) -> dict:
    try:
        parsed = action_geometry.parse_actions(outcome.get("actions") or [])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("future-view action program is invalid") from error
    if not parsed:
        raise ValueError("future-view action program is empty")
    try:
        endpoint = outcome_fields.realized_pose(outcome)
        x_m = float(endpoint["x"])
        z_m = float(endpoint["z"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("future-view realized endpoint is invalid") from error
    if not (math.isfinite(x_m) and math.isfinite(z_m)):
        raise ValueError("future-view realized endpoint is invalid")
    return {
        "primitive_count": len(parsed),
        "turn_count": sum(
            isinstance(value, action_geometry.Turn) for value in parsed),
        "planned_forward_distance_m": action_geometry.total_forward_m(parsed),
        "net_yaw_deg": action_geometry.wrap_deg(
            action_geometry.net_turn_deg(parsed)),
        "realized_displacement_m": action_geometry.planar_distance_m(
            x_m, z_m),
    }


def selection_descriptor_from_terminal_atom(
        rec: dict, outcome: dict, atom: dict) -> dict:
    """Build a C1 descriptor from an already validated terminal atom."""
    summary = _action_summary(outcome)
    value = {
        "outcome_id": str(outcome.get("outcome_id") or ""),
        "base_rollout_key": str(outcome.get("base_rollout_key") or ""),
        "action_sha256": atom["binding"]["action_sha256"],
        "body_sha256": record.canonical_atom_sha256({
            "body": outcome.get("body") or {},
        }),
        "terminal_rgb_atom_sha256": atom["sha256"],
        "terminal_rgb_path": atom["path"],
        "terminal_rgb_sha256": atom["png_sha256"],
        "terminal_pixel_sha256": atom["pixel_sha256"],
        "terminal_pose": atom["terminal_pose"],
        "sensor_sha256": record.canonical_atom_sha256(atom["sensor"]),
        "renderer_sha256": record.canonical_atom_sha256(atom["renderer"]),
        "source_sha256": record.canonical_atom_sha256(atom["source"]),
        "action_summary": summary,
    }
    if not value["outcome_id"]:
        raise ValueError("future-view outcome id is missing")
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def _selection_descriptor_from_snapshot(
        rec: dict, outcome: dict, snapshot: EncodedNativeRGB) -> dict:
    atom = outcome["terminal_rgb_asset"]
    if (snapshot.png_sha256 != atom.get("png_sha256") or
            snapshot.pixel_sha256 != atom.get("pixel_sha256")):
        raise ValueError("terminal RGB snapshot identity changed")
    return selection_descriptor_from_terminal_atom(rec, outcome, atom)


def _completed_clear_c_eligibility(
        rec: dict, outcome: dict, *,
        shared_evidence: SharedVisibleSpaceEvidence | None = None
        ) -> tuple[bool, str]:
    certificate, reason = (
        shared_visible_space_certificate(rec, outcome)
        if shared_evidence is None else
        shared_evidence.require(rec, outcome))
    if certificate is None:
        return False, reason
    if (certificate["summary"].get("collision") is not False or
            not outcome_fields.is_completed_clear(outcome)):
        return False, "completed_clear_required"
    return True, "eligible"


def _terminal_rgb_eligibility(
        rec: dict, outcome: dict, *, asset_root) -> tuple[bool, str]:
    # A typed withhold is a decision the collector recorded, not a damaged
    # PNG, so it keeps its own name: "invalid" would send a reader looking for
    # a corrupt asset that was never written on purpose.
    withhold = outcome.get("terminal_rgb_asset_withhold")
    if withhold in TERMINAL_ASSET_WITHHOLD_REASONS:
        return False, str(withhold)
    try:
        validate_terminal_rgb_asset(rec, outcome, asset_root=asset_root)
    except (KeyError, TypeError, ValueError):
        return False, "terminal_rgb_asset_invalid"
    return True, "eligible"


def counterfactual_c_outcome_eligibility(
        rec: dict, outcome: dict, *, asset_root) -> tuple[bool, str]:
    """Authenticate one candidate-only C1 terminal observation.

    Counterfactual candidates may reveal surfaces that were not visible in the
    initial frame.  They still require a mechanically clear, source-bound
    rollout and an authenticated terminal RGB captured with the record sensor;
    only the strict initial-visible-space continuity test is omitted here.
    """
    eligible, reason = _completed_clear_c_eligibility(rec, outcome)
    if not eligible:
        return False, reason
    return _terminal_rgb_eligibility(
        rec, outcome, asset_root=asset_root)


def project_c_candidate(
        *, rec: dict, outcome: dict, selection: dict, image,
        expected_raw_image_sha256: str, asset_root, question: str,
        stable_id, public_height_decimals: int,
        authenticated_rgb=None) -> tuple[dict, dict]:
    """Thin public/private projection over an already certified selection."""
    raw_asset = viz.authenticate_raw_rgb_image(
        image, expected_sha256=expected_raw_image_sha256,
        expected_resolution=rec["sensor"]["resolution"],
        authenticated=authenticated_rgb)
    item_id = "c-" + stable_id(
        rec.get("observation_id") or rec["frame_id"],
        outcome.get("outcome_id"), "C1_future_view_selection")
    oracle_ref = {
        "frame_id": rec["frame_id"],
        "outcome_id": outcome.get("outcome_id"),
        "base_rollout_key": outcome.get("base_rollout_key"),
        "shared_certificate_sha256":
            outcome["shared_oracle_stability"]["sha256"],
        "terminal_rgb_atom_sha256":
            outcome["terminal_rgb_asset"]["sha256"],
        "future_view_selection_sha256": selection["sha256"],
    }
    choices = [{
        "id": value["id"],
        "image": str(
            (Path(asset_root) / value["terminal_rgb_path"]).resolve()),
        "image_sha256": value["terminal_rgb_sha256"],
    } for value in selection["choices"]]
    sensor = rec["sensor"]
    item = {
        "id": item_id, "result_head": "C",
        "task_id": "C1_future_view_selection", "question": question,
        "answer_format": "closed_exact",
        "model_input": {
            "initial_rgb": str(image),
            "initial_rgb_sha256": raw_asset.sha256,
            "camera_height_above_visible_floor_m": round(
                float(rec["camera_height_above_visible_floor_m"]),
                public_height_decimals),
            "hfov_deg": float(sensor["hfov_deg"]),
            "vfov_deg": float(sensor["vfov_deg"]),
            "body_radius_m": float(outcome["body"]["radius_m"]),
            "actions": list(outcome.get("actions") or []),
        },
        "choices": choices, "oracle_ref": oracle_ref,
    }
    private = {
        "id": item_id, "task_id": "C1_future_view_selection",
        "canonical_answer": selection["canonical_answer"],
        "oracle_ref": dict(oracle_ref),
        "selection_certificate": selection,
        "human_answerability": {
            "task_id": "C1_future_view_selection", "status": "pending",
        },
        "input_asset": {
            "path": str(image), "sha256": raw_asset.sha256,
            "record_image_path": rec.get("image_path"),
        },
    }
    return item, private
