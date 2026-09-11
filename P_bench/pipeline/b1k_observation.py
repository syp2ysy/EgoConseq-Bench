"""Pinned B1K observation contract with OmniGibson-owned rendering."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import record


OBSERVATION_PROTOCOL = record.B1K_RENDERER_INSTANCE_PROTOCOL
PROFILE_SCHEMA = "b1k-observation-profile.v1"
LEGACY_PROFILE_SCHEMA = "b1k-render-profile-probe.v1"
PROFILE_ASSET = Path(__file__).with_name("assets") / \
    "b1k_observation.v5.json"
LEGACY_V5_PROFILE_ASSET = Path(__file__).with_name("assets") / \
    "b1k_observation.legacy-v5.json"
LEGACY_V6_PROFILE_ASSET = Path(__file__).with_name("assets") / \
    "b1k_observation.legacy-v6.json"
GEOMETRY_MODALITIES = ("depth_linear", "seg_instance_id")
OBSERVATION_MODALITIES = ("rgb",) + GEOMETRY_MODALITIES


def profile_atom_sha256(atom: Mapping[str, Any]) -> str:
    value = {key: item for key, item in atom.items() if key != "sha256"}
    return record.canonical_atom_sha256(value)


def _defaults_key(key: str) -> str | None:
    if key.startswith("/rtx-transient/"):
        return None
    if not key.startswith("/rtx/"):
        raise ValueError("B1K renderer profile keys must live under /rtx")
    return "/rtx-defaults/" + key.removeprefix("/rtx/")


def profile_setting_writes(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Expand legacy active renderer settings to their persistent mirrors."""
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


def load_frozen_profile(
        path: Path | None = None, *, contract_version: str | None = None
        ) -> dict:
    """Load the immutable observation profile for one B1K contract."""
    version = (
        record.B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION
        if contract_version is None else str(contract_version))
    profiles = {
        record.B1K_V5_COLLECTION_CONTRACT_VERSION: (
            LEGACY_V5_PROFILE_ASSET, LEGACY_PROFILE_SCHEMA),
        record.B1K_V16_COLLECTION_CONTRACT_VERSION: (
            LEGACY_V6_PROFILE_ASSET, LEGACY_PROFILE_SCHEMA),
        record.B1K_OFFICIAL_RENDER_COLLECTION_CONTRACT_VERSION: (
            PROFILE_ASSET, PROFILE_SCHEMA),
    }
    try:
        default_path, expected_schema = profiles[version]
        expected_mode = record.B1K_C1_RENDER_MODE_BY_CONTRACT[version]
        expected_digest = \
            record.B1K_OBSERVATION_PROFILE_SHA256_BY_CONTRACT[version]
    except KeyError as error:
        raise ValueError(
            "B1K observation profile contract version is unsupported") \
            from error
    value = json.loads(Path(default_path if path is None else path).read_text())
    if value.get("schema") != expected_schema:
        raise ValueError("B1K observation profile schema is invalid")
    if value.get("c1_render_mode") != expected_mode:
        raise ValueError("B1K published C1 render mode changed")
    atom = value.get("profile_atom")
    if not isinstance(atom, dict) or atom.get("protocol") != \
            OBSERVATION_PROTOCOL:
        raise ValueError("B1K observation profile atom is invalid")
    if atom.get("sha256") != profile_atom_sha256(atom):
        raise ValueError("B1K observation profile digest changed")
    if atom.get("sha256") != expected_digest:
        raise ValueError("B1K observation published digest changed")
    if tuple(atom.get("modalities") or ()) != OBSERVATION_MODALITIES:
        raise ValueError("B1K observation profile modalities changed")
    if int(atom.get("sync_render_count", 0)) != \
            int(value.get("sync_render_count", -1)):
        raise ValueError("B1K observation profile sync count disagrees")
    return value


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
    raw_ids = np.asarray(sorted(observed), dtype=values.dtype)
    stable_ids = np.asarray(
        [raw_to_stable[int(raw_id)] for raw_id in raw_ids],
        dtype=np.int64)
    result = stable_ids[np.searchsorted(raw_ids, values)]
    return np.ascontiguousarray(result), {
        "observed_raw_instance_count": len(observed),
        "mapped_source_instance_count": sum(
            stable_id > 0 for stable_id in raw_to_stable.values()),
        "unmapped_pixel_ratio": float(np.mean(result == 0)),
    }
