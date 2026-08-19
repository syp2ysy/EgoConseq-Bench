"""Frozen BEHAVIOR-1K renderer and observation transaction contracts.

The module is intentionally free of OmniGibson imports.  It defines the
deterministic data contract used at the lazy runtime trust boundary; the real
renderer is loaded only by :mod:`pipeline.b1k_sim` and the diagnostic CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import record


OBSERVATION_PROTOCOL = "b1k-observation.v5"
PROFILE_PROBE_SCHEMA = "b1k-render-profile-probe.v1"
PROFILE_ASSET = Path(__file__).with_name("assets") / \
    "b1k_observation.v5.json"
MAX_SYNC_RENDER_COUNT = 16
MAX_OBSERVATION_TRANSACTION_S = 1.0
GEOMETRY_MODALITIES = ("depth_linear", "seg_instance_id")
OBSERVATION_MODALITIES = ("rgb",) + GEOMETRY_MODALITIES


def load_frozen_profile(path: Path = PROFILE_ASSET) -> dict:
    """Load and authenticate the measured observation profile asset."""
    value = json.loads(Path(path).read_text())
    if value.get("schema") != PROFILE_PROBE_SCHEMA:
        raise ValueError("B1K observation profile schema is invalid")
    if value.get("c1_render_mode") != record.B1K_C1_RENDER_MODE:
        raise ValueError("B1K published C1 render mode changed")
    atom = value.get("profile_atom")
    if not isinstance(atom, dict) or atom.get("protocol") != \
            OBSERVATION_PROTOCOL:
        raise ValueError("B1K observation profile atom is invalid")
    if atom.get("sha256") != profile_atom_sha256(atom):
        raise ValueError("B1K observation profile digest changed")
    if atom.get("sha256") != record.B1K_OBSERVATION_PROFILE_SHA256:
        raise ValueError("B1K observation published digest changed")
    if tuple(atom.get("modalities") or ()) != OBSERVATION_MODALITIES:
        raise ValueError("B1K observation profile modalities changed")
    if int(atom.get("sync_render_count", 0)) != \
            int(value.get("sync_render_count", -1)):
        raise ValueError("B1K observation profile sync count disagrees")
    return value


def array_sha256(value) -> str:
    """Hash an array without conflating equal bytes of different arrays."""
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }, sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def first_stable_render(frames: Sequence[Any]) -> int | None:
    """Return the one-based start of the final exact, repeated suffix."""
    hashes = [array_sha256(value) for value in frames]
    if len(hashes) < 2:
        return None
    for index in range(len(hashes) - 1):
        suffix = hashes[index:]
        if len(set(suffix)) == 1:
            return index + 1
    return None


def remap_native_instance_mask(
        raw_mask, *, id_to_prim_path: Mapping[Any, Any],
        visual_prim_to_instance_id: Mapping[str, int]) -> tuple[np.ndarray,
                                                                  dict]:
    """Map one frame's ephemeral OG IDs to frozen benchmark instance IDs."""
    values = np.asarray(raw_mask)
    if (values.ndim != 2 or not np.issubdtype(values.dtype, np.integer) or
            np.any(values < 0)):
        raise ValueError("B1K native instance mask must be nonnegative HxW")
    metadata = {int(key): str(value)
                for key, value in id_to_prim_path.items()}
    observed = {int(value) for value in np.unique(values)}
    missing = observed - set(metadata)
    if missing:
        raise ValueError(
            "B1K native instance metadata omits observed IDs: "
            f"{sorted(missing)}")
    source_map = {str(key): int(value)
                  for key, value in visual_prim_to_instance_id.items()}
    if any(value <= 0 for value in source_map.values()):
        raise ValueError("B1K frozen instance IDs must be positive")
    raw_to_stable = {
        raw_id: source_map.get(metadata[raw_id], 0)
        for raw_id in observed
    }
    result = np.zeros(values.shape, dtype=np.int64)
    for raw_id, stable_id in raw_to_stable.items():
        if stable_id:
            result[values == raw_id] = stable_id
    return np.ascontiguousarray(result), {
        "observed_raw_instance_count": len(observed),
        "mapped_source_instance_count": sum(
            stable_id > 0 for stable_id in raw_to_stable.values()),
        "unmapped_pixel_ratio": float(np.mean(result == 0)),
    }


def _defaults_key(key: str) -> str | None:
    if key.startswith("/rtx-transient/"):
        return None
    if not key.startswith("/rtx/"):
        raise ValueError("B1K renderer profile keys must live under /rtx")
    return "/rtx-defaults/" + key.removeprefix("/rtx/")


def profile_setting_writes(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return active settings plus persistent ``/rtx-defaults`` mirrors."""
    writes: dict[str, Any] = {}
    for raw_key, value in settings.items():
        key = str(raw_key)
        if key in writes:
            raise ValueError("B1K renderer profile contains duplicate keys")
        writes[key] = value
        defaults = _defaults_key(key)
        if defaults is not None:
            writes[defaults] = value
    return writes


def derive_sync_render_count(
        *, first_stable_renders: Sequence[int],
        worst_render_seconds: float) -> int:
    """Freeze the slowest measured sync point plus one safety render."""
    values = [int(value) for value in first_stable_renders]
    if not values or any(value < 1 for value in values):
        raise ValueError("B1K sync calibration requires positive frame counts")
    render_seconds = float(worst_render_seconds)
    if not 0.0 < render_seconds:
        raise ValueError("B1K sync calibration requires positive render time")
    result = max(values) + 1
    if result > MAX_SYNC_RENDER_COUNT:
        raise ValueError("B1K sync calibration exceeds the render-count limit")
    if result * render_seconds >= MAX_OBSERVATION_TRANSACTION_S:
        raise ValueError("B1K observation transaction exceeds one second")
    return result


def profile_atom_sha256(atom: Mapping[str, Any]) -> str:
    value = {key: item for key, item in atom.items() if key != "sha256"}
    return record.canonical_atom_sha256(value)


def build_profile_atom(
        *, settings: Mapping[str, Any], settings_readback: Mapping[str, Any],
        antialiasing_name: str,
        renderer_input_resolution: Sequence[int],
        renderer_output_resolution: Sequence[int], sync_render_count: int,
        modalities: Sequence[str], runtime_identity: Mapping[str, Any],
        reset_accumulation: bool) -> dict:
    """Build a source-bound v5 renderer profile after a successful probe."""
    input_resolution = [int(value) for value in renderer_input_resolution]
    output_resolution = [int(value) for value in renderer_output_resolution]
    if (len(input_resolution) != 2 or len(output_resolution) != 2 or
            any(value <= 0 for value in input_resolution + output_resolution)):
        raise ValueError("B1K renderer resolution must be positive width/height")
    if input_resolution != output_resolution:
        raise ValueError(
            "B1K renderer input and output resolution must match")
    count = int(sync_render_count)
    if not 1 <= count <= MAX_SYNC_RENDER_COUNT:
        raise ValueError("B1K sync render count is outside the frozen range")
    expected_readback = profile_setting_writes(settings)
    observed_readback = dict(settings_readback)
    if observed_readback != expected_readback:
        raise ValueError("B1K renderer settings readback differs from profile")
    modality_list = [str(value) for value in modalities]
    if not modality_list or len(set(modality_list)) != len(modality_list):
        raise ValueError("B1K observation modalities must be unique")
    value = {
        "protocol": OBSERVATION_PROTOCOL,
        "antialiasing": str(antialiasing_name),
        "settings": dict(settings),
        "settings_readback": observed_readback,
        "renderer_input_resolution": input_resolution,
        "renderer_output_resolution": output_resolution,
        "sync_render_count": count,
        "modalities": modality_list,
        "runtime_identity": dict(runtime_identity),
        "reset_accumulation": bool(reset_accumulation),
        "rgb_authority": "frozen-asset-no-rerender-hash.v1",
    }
    return {**value, "sha256": record.canonical_atom_sha256(value)}


def select_profile_from_probe(
        profile_rows: Sequence[Mapping[str, Any]], *,
        runtime_identity: Mapping[str, Any]) -> dict:
    """Select one measured renderer profile without guessing renderer state.

    Geometry modalities must be pose-current, stable, and independent of the
    pose rendered immediately before them.  RGB history-independence is not a
    physical-GT gate: it only chooses between the cheap sequential C1 renderer
    and the isolated temporary four-sensor renderer.
    """
    candidates = []
    for raw_row in profile_rows:
        row = dict(raw_row)
        input_resolution = row.get("renderer_input_resolution")
        output_resolution = row.get("renderer_output_resolution")
        if input_resolution is None or output_resolution is None:
            continue
        if list(input_resolution) != list(output_resolution):
            continue
        stable = row.get("first_stable_render") or {}
        current = row.get("current_pose_differs") or {}
        history = row.get("history_independent") or {}
        if any(
                name not in stable or not bool(current.get(name)) or
                not bool(history.get(name))
                for name in GEOMETRY_MODALITIES):
            continue
        try:
            sync_render_count = derive_sync_render_count(
                first_stable_renders=[stable[name]
                                      for name in GEOMETRY_MODALITIES],
                worst_render_seconds=float(row["worst_render_seconds"]),
            )
            atom = build_profile_atom(
                settings=row["settings"],
                settings_readback=row["settings_readback"],
                antialiasing_name=str(row["antialiasing"]),
                renderer_input_resolution=input_resolution,
                renderer_output_resolution=output_resolution,
                sync_render_count=sync_render_count,
                modalities=OBSERVATION_MODALITIES,
                runtime_identity=runtime_identity,
                reset_accumulation=bool(row.get("reset_accumulation")),
            )
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((
            int(row.get("quality_rank", 0)),
            float(row["worst_render_seconds"]) * sync_render_count,
            str(row.get("profile_id", "")), row, atom,
            sync_render_count,
        ))
    if not candidates:
        raise ValueError("no B1K renderer profile passed the measured gates")
    _rank, _cost, _profile_id, row, atom, sync_render_count = min(candidates)
    rgb_pose_pure = bool(
        (row.get("history_independent") or {}).get("rgb"))
    return {
        "schema": PROFILE_PROBE_SCHEMA,
        "selected_profile_id": str(row.get("profile_id", "")),
        "profile_atom": atom,
        "sync_render_count": sync_render_count,
        "rgb_pose_pure": rgb_pose_pure,
        "c1_render_mode": (
            "sequential-single-sensor.v1" if rgb_pose_pure else
            "temporary-counterfactual-bank-batch.v1"),
    }
