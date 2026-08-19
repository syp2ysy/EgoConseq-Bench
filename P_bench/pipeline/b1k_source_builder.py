"""Pure B1K installation inputs, source manifests, and shard plans.

Encrypted USD files are authenticated as opaque bytes.  Geometry is never
read from them here; the runtime bootstrap derives geometry only after
OmniGibson has loaded the scene.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Iterable, Mapping, Sequence

from pipeline import b1k_semantic, record
from pipeline.io_utils import sha256_file
from pipeline.scene_pool import B1K_SOURCE_MANIFEST_VERSION


SOURCE_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_PYTHON = Path(
    os.environ.get("EGOCONSEQ_B1K_PYTHON", sys.executable))
PINNED_INSTALL = {
    "source_commit": "26f2c7ef7b9cf96bd0414f81e1e751e493762779",
    "source_tag": "v3.9.1",
    "python": "3.11.15",
    "omnigibson": "3.9.1",
    "bddl": "3.7.0",
    "isaac_sim": "5.1.0.0",
    "torch": "2.7.0+cu128",
    "behavior-1k-assets": "3.9.0",
    "omnigibson-robot-assets": "3.8.2",
}
DEFAULT_SCENE_TIMEOUT_S = 1200
DEFAULT_COLLECTION_SCENE_TIMEOUT_S = 21600
CATALOG_AUTHORITY_AUDIT_SCHEMA = "b1k-catalog-authority-audit.v1"
CATALOG_AUTHORITY_AUDIT_SELECTION_SCOPE = CATALOG_AUTHORITY_AUDIT_SCHEMA
CATALOG_AUTHORITY_AUDIT_CATALOG_COUNT = 51


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path, root: Path) -> dict:
    """Return a root-relative opaque file identity."""
    root = Path(root).resolve(strict=True)
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"B1K input escapes data root: {resolved}") from error
    if not resolved.is_file():
        raise ValueError(f"B1K input is not a file: {resolved}")
    return {
        "path": relative.as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def installed_scene_ids(data_root: Path) -> list[str]:
    """Discover the actual installed scene metadata catalog."""
    scenes_root = (Path(data_root).resolve(strict=True) /
                   "behavior-1k-assets" / "scenes")
    if not scenes_root.is_dir():
        raise ValueError(f"B1K scenes directory is absent: {scenes_root}")
    scene_ids = []
    for directory in sorted(
            value for value in scenes_root.iterdir() if value.is_dir()):
        matches = sorted((directory / "json").glob("*_best.json"))
        if len(matches) != 1:
            raise ValueError(
                f"B1K scene {directory.name} has {len(matches)} canonical JSONs")
        scene_ids.append(directory.name)
    if len(scene_ids) < 3:
        raise ValueError("B1K installation must contain at least 3 scenes")
    return scene_ids


def canonical_scene_json(data_root: Path, scene_id: str) -> Path:
    root = Path(data_root).resolve(strict=True)
    directory = root / "behavior-1k-assets" / "scenes" / str(scene_id)
    matches = sorted((directory / "json").glob("*_best.json"))
    if len(matches) != 1:
        raise ValueError(
            f"B1K scene {scene_id} has {len(matches)} canonical JSONs")
    return matches[0]


def _referenced_encrypted_assets(
        data_root: Path, scene_json: Path) -> list[Path]:
    try:
        payload = json.loads(scene_json.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"cannot decode canonical B1K scene JSON: {scene_json}") from error
    init_info = (payload.get("objects_info") or {}).get("init_info")
    if not isinstance(init_info, dict) or not init_info:
        raise ValueError(f"B1K scene has no canonical object inputs: {scene_json}")
    assets = set()
    root = Path(data_root).resolve(strict=True)
    for identity, raw in sorted(init_info.items()):
        if not isinstance(raw, dict):
            raise ValueError(f"B1K object input is invalid: {identity}")
        args = raw.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"B1K object args are invalid: {identity}")
        category = str(args.get("category") or "").strip()
        model = str(args.get("model") or "").strip()
        if not category or not model:
            raise ValueError(f"B1K object asset identity is absent: {identity}")
        assets.add(root / "behavior-1k-assets" / "objects" / category /
                   model / "usd" / f"{model}.encrypted.usd")
    missing = [str(path) for path in sorted(assets) if not path.is_file()]
    if missing:
        raise ValueError("B1K referenced encrypted asset is absent: " +
                         ", ".join(missing[:10]))
    return sorted(assets)


def _validate_authority(authority: Mapping) -> dict:
    value = dict(authority)
    if value.get("schema") != "b1k-derived-scene-authority.v1":
        raise ValueError("B1K runtime scene authority schema is invalid")
    unhashed = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != record.canonical_atom_sha256(unhashed):
        raise ValueError("B1K runtime scene authority digest is invalid")
    b1k_semantic.canonical_replay_binding(value)
    return value


def build_scene_entry(
        data_root: Path, scene_id: str, authority: Mapping) -> dict:
    """Bind static source bytes to one runtime-derived scene authority."""
    root = Path(data_root).resolve(strict=True)
    scene_json = canonical_scene_json(root, scene_id)
    scene_identity = file_identity(scene_json, root)
    layout_root = scene_json.parents[1] / "layout"
    layouts = sorted(path for path in layout_root.rglob("*") if path.is_file())
    if not layouts:
        raise ValueError(f"B1K scene {scene_id} has no layout inputs")
    encrypted = _referenced_encrypted_assets(root, scene_json)
    return {
        "scene_id": str(scene_id),
        "scene_json": scene_identity,
        "layouts": [file_identity(path, root) for path in layouts],
        "encrypted_assets": [file_identity(path, root) for path in encrypted],
        # The canonical scene JSON contains OG's serialized ``state`` and is
        # the input OG actually loads.  The runtime then dump/load-replays the
        # resulting live state before every render.
        "initial_state": dict(scene_identity),
        "initial_state_replay": {
            "schema": "b1k-canonical-scene-json-replay.v1",
            "canonical_input": "scene_json",
            "runtime_replay": "omnigibson_dump_state_load_state",
        },
        "scene_authority": _validate_authority(authority),
    }


def build_source_manifest(
        entries: Iterable[Mapping], *, installed_scene_ids: Sequence[str],
        selection_scope: str, simulator: Mapping,
        asset_versions: Mapping) -> dict:
    """Build an honestly scoped authenticated B1K source manifest."""
    catalog = sorted(str(value) for value in installed_scene_ids)
    if len(catalog) < 3 or len(catalog) != len(set(catalog)):
        raise ValueError("installed B1K catalog is invalid")
    selected = sorted((dict(value) for value in entries),
                      key=lambda value: str(value.get("scene_id")))
    selected_ids = [str(value.get("scene_id") or "") for value in selected]
    if (not selected_ids or len(selected_ids) != len(set(selected_ids)) or
            not set(selected_ids).issubset(catalog)):
        raise ValueError("selected B1K source scope is invalid")
    scope = str(selection_scope).strip()
    if not scope:
        raise ValueError("B1K selection scope must be nonempty")
    complete = selected_ids == catalog
    return {
        "schema_version": B1K_SOURCE_MANIFEST_VERSION,
        "dataset": "b1k",
        "official_split": "train",
        "split_authority": "project_defined",
        "installed_catalog_count": len(catalog),
        "installed_catalog_sha256": _canonical_sha256({
            "scene_ids": catalog,
        }),
        "selected_scene_count": len(selected_ids),
        "selection_scope": scope,
        "catalog_complete": complete,
        "simulator": dict(simulator),
        "asset_versions": dict(asset_versions),
        "scenes": selected,
    }


def build_install_manifest(
        *, data_root: Path, source_root: Path, observed: Mapping,
        scene_ids: Sequence[str]) -> dict:
    """Verify every pinned install field and bind the actual catalog."""
    values = {str(key): str(value) for key, value in observed.items()}
    for field, expected in PINNED_INSTALL.items():
        actual = values.get(field)
        if actual != expected:
            raise ValueError(
                f"B1K {field} version differs: {actual!r} != {expected!r}")
    expected_editable_root = str(
        (Path(source_root).resolve() / "OmniGibson").resolve())
    observed_editable_root = str(Path(
        values.get("omnigibson_editable_root", "")).resolve())
    if observed_editable_root != expected_editable_root:
        raise ValueError(
            "B1K OmniGibson editable root differs: "
            f"{observed_editable_root!r} != {expected_editable_root!r}")
    catalog = sorted(str(value) for value in scene_ids)
    if len(catalog) < 3 or len(catalog) != len(set(catalog)):
        raise ValueError("installed B1K catalog is invalid")
    return {
        "schema": "b1k-verified-install.v1",
        "verified": True,
        "data_root": str(Path(data_root).resolve()),
        "source_root": str(Path(source_root).resolve()),
        "omnigibson_editable_root": expected_editable_root,
        "versions": {key: values[key] for key in PINNED_INSTALL},
        "installed_catalog_count": len(catalog),
        "installed_scene_ids": catalog,
        "installed_catalog_sha256": _canonical_sha256({
            "scene_ids": catalog,
        }),
    }


def _partition(values: Sequence[str], count: int) -> list[list[str]]:
    quotient, remainder = divmod(len(values), count)
    shards = []
    offset = 0
    for index in range(count):
        size = quotient + int(index < remainder)
        shards.append(list(values[offset:offset + size]))
        offset += size
    return shards


def build_shard_plan(
        scene_ids: Sequence[str], *, data_root: Path, output_root: Path,
        gpu_ids: Sequence[int]) -> dict:
    """Return four scene-disjoint commands for resumable authority derivation."""
    catalog = sorted(str(value) for value in scene_ids)
    gpu_values = [int(value) for value in gpu_ids]
    if len(gpu_values) != 4 or len(set(gpu_values)) != 4:
        raise ValueError("B1K full derivation requires four distinct GPU IDs")
    root = Path(data_root).resolve()
    output = Path(output_root).resolve()
    shards = []
    for index, (gpu_id, assigned) in enumerate(zip(
            gpu_values, _partition(catalog, 4))):
        shard_output = output / f"shard-{index:02d}"
        arguments = [
            f"OMNIGIBSON_DATA_PATH={root}",
            f"OMNIGIBSON_APPDATA_PATH={root / 'appdata'}",
            "OMNIGIBSON_HEADLESS=True",
            "OMNI_KIT_ACCEPT_EULA=YES",
            f"OMNIGIBSON_GPU_ID={gpu_id}",
            str(BEHAVIOR_PYTHON),
            str(SOURCE_ROOT / "scripts" / "build_b1k_source_manifest.py"),
            "derive-shard", "--data-root", str(root),
            "--output-dir", str(shard_output), "--resume",
            "--scene-timeout-s", str(DEFAULT_SCENE_TIMEOUT_S), "--scenes",
            *assigned,
        ]
        shards.append({
            "shard_index": index,
            "gpu_id": gpu_id,
            "scene_ids": assigned,
            "output_dir": str(shard_output),
            "command": " ".join(shlex.quote(value) for value in arguments),
        })
    return {
        "schema": "b1k-source-manifest-shard-plan.v1",
        "installed_catalog_count": len(catalog),
        "shard_count": 4,
        "shards": shards,
    }


def build_collection_shard_plan(
        scene_ids: Sequence[str], *, data_root: Path,
        source_manifest: Path, output_root: Path,
        gpu_ids: Sequence[int], code_revision: str,
        collect_args: Sequence[str] = (),
        scene_timeout_s: int = DEFAULT_COLLECTION_SCENE_TIMEOUT_S) -> dict:
    """Return four OG-free, per-scene-isolated collection commands.

    The referenced source manifest is consumed by each ``collect.py`` child;
    this planner intentionally does not open it so a full-catalog collection
    plan can be materialized while authority derivation is still resumable.
    """
    catalog = sorted(str(value) for value in scene_ids)
    if not catalog or len(catalog) != len(set(catalog)):
        raise ValueError("B1K collection catalog is invalid")
    gpu_values = [int(value) for value in gpu_ids]
    if len(gpu_values) != 4 or len(set(gpu_values)) != 4:
        raise ValueError("B1K collection requires four distinct GPU IDs")
    timeout_s = int(scene_timeout_s)
    if timeout_s <= 0:
        raise ValueError("B1K collection scene timeout must be positive")
    revision = str(code_revision).strip()
    if not revision:
        raise ValueError("B1K collection code revision must be nonempty")
    root = Path(data_root).resolve()
    manifest = Path(source_manifest).resolve()
    output = Path(output_root).resolve()
    forwarded = [str(value) for value in collect_args]
    shards = []
    for index, (gpu_id, assigned) in enumerate(zip(
            gpu_values, _partition(catalog, 4))):
        shard_output = output / f"shard-{index:02d}"
        arguments = [
            f"OMNIGIBSON_DATA_PATH={root}",
            f"OMNIGIBSON_APPDATA_PATH={root / 'appdata'}",
            "OMNIGIBSON_HEADLESS=True",
            "OMNI_KIT_ACCEPT_EULA=YES",
            f"OMNIGIBSON_GPU_ID={gpu_id}",
            str(BEHAVIOR_PYTHON),
            str(SOURCE_ROOT / "scripts" / "run_b1k_collection_shard.py"),
            "run", "--data-root", str(root),
            "--source-manifest", str(manifest),
            "--output-dir", str(shard_output),
            "--gpu-id", str(gpu_id),
            "--shard-id", f"b1k-{index:02d}",
            "--code-revision", revision,
            "--resume", "--scene-timeout-s", str(timeout_s),
            "--scenes", *assigned,
        ]
        if forwarded:
            arguments.extend(["--collect-args", *forwarded])
        shards.append({
            "shard_index": index,
            "gpu_id": gpu_id,
            "scene_ids": assigned,
            "output_dir": str(shard_output),
            "command": " ".join(shlex.quote(value) for value in arguments),
        })
    return {
        "schema": "b1k-collection-shard-plan.v1",
        "installed_catalog_count": len(catalog),
        "source_manifest": str(manifest),
        "scene_process_isolation": True,
        "shard_count": 4,
        "shards": shards,
    }
