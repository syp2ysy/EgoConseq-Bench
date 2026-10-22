"""Habitat native navigation built from the complete official GS collision USD.

The cached mesh contains every source triangle, transformed to the same Y-up
frame as the Gaussian renderer. Recast derives a separate native NavMesh for
each body. This is navigation collision, not Bullet rigid-body contact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from pipeline import io_utils


SCENE_SCHEMA = "gs-full-source-native-scene.v1"
_ROOT = Path(__file__).resolve().parents[1]


def _verify_source(path, source_asset):
    path = Path(os.path.abspath(os.fspath(path)))
    if (source_asset.get("role") != "collision_mesh" or
            path.stat().st_size != source_asset.get("bytes") or
            io_utils.sha256_file(path) != source_asset.get("sha256")):
        raise ValueError("GS native collision source digest differs from authority")
    return path


def resolve_collision_usd(scene_dir, scene_id, source_asset):
    """Locate an exact source digest, retaining the USD symlink's extension."""
    source_id = str(scene_id).rsplit("_", 1)[-1]
    if not source_id.isdigit():
        raise ValueError("GS native scene has an invalid official source ID")
    relative = Path(source_id) / f"{source_id}_collision.usd"
    root = Path(scene_dir).parent.parent
    candidates = []
    override = os.environ.get("EGOCONSEQ_GS_COLLISION_ROOT")
    if override:
        candidates.append(Path(override) / relative)
    candidates.append(root / "_official_sage3d_collision" / "Collision_Mesh" / relative)
    hf_root = Path(os.environ.get(
        "HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    snapshots = hf_root / "datasets--spatialverse--SAGE-3D_Collision_Mesh" / "snapshots"
    candidates.extend(sorted(snapshots.glob(f"*/Collision_Mesh/{relative}")))
    for path in candidates:
        if path.is_file():
            try:
                return _verify_source(path, source_asset)
            except ValueError:
                continue
    raise FileNotFoundError(
        f"authenticated official collision USD unavailable for {scene_id}; "
        "set EGOCONSEQ_GS_COLLISION_ROOT to the SAGE-3D Collision_Mesh directory")


def ensure_native_scene(collision_usd, source_asset, cache_dir):
    """Authenticate and cache the full source mesh, without height clipping."""
    source = _verify_source(collision_usd, source_asset)
    cache = Path(cache_dir)
    manifest_path = cache / "scene.json"
    mesh_path = cache / "scene.glb"
    stage_path = cache / "scene.stage_config.json"
    stage_config = {
        "render_asset": "scene.glb", "collision_asset": "scene.glb",
        "up": [0, 1, 0], "front": [0, 0, -1], "units_to_meters": 1.0,
    }
    converter = hashlib.sha256(
        Path(__file__).read_bytes() +
        (_ROOT / "scripts/build_gs_collision_authority.py").read_bytes()
    ).hexdigest()
    if manifest_path.is_file() and mesh_path.is_file() and stage_path.is_file():
        try:
            result = json.loads(manifest_path.read_text())
            if (result["schema"] == SCENE_SCHEMA and
                    result["source_asset"] == source_asset and
                    result["converter_sha256"] == converter and
                    json.loads(stage_path.read_text()) == stage_config and
                    result["mesh_sha256"] == io_utils.sha256_file(mesh_path)):
                return result
        except (KeyError, TypeError, ValueError):
            pass

    # Reuse the source reader that authored scene.collision.npz. It applies
    # official USD transforms and the fixed InteriorGS Z-up -> Y-up rotation.
    from scripts.build_gs_collision_authority import _triangulated_collision_components
    import trimesh

    components = _triangulated_collision_components(source)
    triangles = np.concatenate([component.triangles for component in components])
    vertices = triangles.reshape(-1, 3).astype(np.float32)
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3),
        process=False)
    payload = mesh.export(file_type="glb")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / f"scene.{os.getpid()}.glb.tmp"
    temporary.write_bytes(payload)
    temporary.replace(mesh_path)
    io_utils.atomic_write_json(stage_path, stage_config)
    result = {
        "schema": SCENE_SCHEMA, "source_asset": dict(source_asset),
        "source_path": str(source), "converter_sha256": converter,
        "coordinate_transform": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        "geometry_scope": "complete_source_triangles_no_height_clipping",
        "component_count": len(components), "triangle_count": len(triangles),
        "bounds_min": vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": vertices.max(axis=0).astype(float).tolist(),
        "mesh_path": str(mesh_path.resolve()),
        "mesh_sha256": io_utils.sha256_file(mesh_path),
        "stage_path": str(stage_path.resolve()),
        "stage_sha256": io_utils.sha256_file(stage_path),
    }
    io_utils.atomic_write_json(manifest_path, result)
    return result


def _source(scene_dir, collision_authority):
    source_asset = next(row for row in collision_authority.binding_atom()["source_assets"]
                        if row["role"] == "collision_mesh")
    source = resolve_collision_usd(scene_dir, collision_authority.scene_id, source_asset)
    cache_root = Path(os.environ.get(
        "EGOCONSEQ_GS_NATIVE_CACHE", _ROOT / "data/cache/gs_native"))
    return source, source_asset, cache_root / collision_authority.scene_id / source_asset["sha256"][:16]


def scene_mesh(scene_dir, collision_authority):
    """The authenticated complete collision mesh in Habitat world coordinates, without a simulator."""
    return ensure_native_scene(*_source(scene_dir, collision_authority))


