#!/usr/bin/env python3
"""Preprocess official SAGE-3D collision USD into a runtime authority.

The expensive USD traversal happens once.  Collection only loads the small,
source-bound planar artifact and therefore does not depend on Isaac Sim, USD,
or the Gaussian ellipsoid approximation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

import habitat_sim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import gs_collision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_asset(role: str, path: Path) -> tuple[str, int, str]:
    resolved = path.resolve(strict=True)
    return role, int(resolved.stat().st_size), _sha256_file(resolved)


def _lexical_collision_path(path) -> Path:
    """Validate collision bytes without erasing the USD filename suffix.

    Hugging Face snapshots expose ``*.usd`` symlinks whose resolved cache blob
    has no extension.  USD selects its file-format plugin from the lexical
    suffix, so resolving that symlink before ``Usd.Stage.Open`` makes valid
    official assets unreadable.  The artifact still hashes the resolved bytes
    at the source-binding boundary via :func:`_source_asset`.
    """
    value = Path(os.path.abspath(os.fspath(path)))
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _usd_modules():
    try:
        from pxr import Usd, UsdGeom
    except ImportError as error:
        raise RuntimeError(
            "USD preprocessing requires the optional 'usd-core' package") \
            from error
    return Usd, UsdGeom


def _triangulated_collision_components(path: Path) -> tuple[
        gs_collision.SourceCollisionComponent, ...]:
    Usd, UsdGeom = _usd_modules()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open GS collision USD: {path}")
    if str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z":
        raise ValueError("official GS collision USD must be Z-up")
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1e-12:
        raise ValueError("official GS collision USD must use metres")
    transform_cache = UsdGeom.XformCache()
    components = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(
            mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(
            mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if (points.ndim != 2 or points.shape[1] != 3 or
                counts.ndim != 1 or indices.ndim != 1 or
                int(counts.sum()) != len(indices) or
                np.any(counts < 3)):
            raise ValueError(
                f"GS collision mesh topology is invalid: {prim.GetPath()}")
        matrix = np.asarray(
            transform_cache.GetLocalToWorldTransform(prim),
            dtype=np.float64)
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
        world = (homogeneous @ matrix)[:, :3]
        world = gs_collision.interiorgs_zup_to_pbench_xyz(world)
        if np.all(counts == 3):
            faces = indices.reshape(-1, 3)
        else:
            faces = []
            offset = 0
            for count in counts:
                face = indices[offset:offset + int(count)]
                offset += int(count)
                faces.extend(
                    (face[0], face[index], face[index + 1])
                    for index in range(1, len(face) - 1))
            faces = np.asarray(faces, dtype=np.int64)
        components.append(gs_collision.SourceCollisionComponent(
            str(prim.GetPath()), world[faces]))
    if not components:
        raise ValueError("GS collision USD contains no mesh components")
    return tuple(components)


def catalog_conversion_inputs(data_root, manifest_path, collision_root, *,
                              excluded_scenes=()
                              ) -> list[tuple[str, Path, Path, Path]]:
    """Resolve every GS train scene to its official collision USD.

    The InteriorGS scene name contains a display prefix plus the official
    numeric scene ID.  SAGE-3D collision assets are keyed by that numeric ID.
    Every scheduled path is resolved and checked before catalog conversion
    starts.  A scene may be omitted only through the explicit, validated
    ``excluded_scenes`` list, so a failed authority cannot silently reduce
    coverage.
    """
    root = Path(data_root).resolve(strict=True)
    manifest_file = Path(manifest_path).resolve(strict=True)
    collision = Path(collision_root).resolve(strict=True)
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    if value.get("dataset") != "gs" or not isinstance(
            value.get("scenes"), list):
        raise ValueError("GS collision catalog manifest is invalid")
    excluded = tuple(str(value).strip() for value in excluded_scenes)
    if any(not value for value in excluded) or \
            len(excluded) != len(set(excluded)):
        raise ValueError("GS collision catalog exclusions are invalid")
    excluded_set = set(excluded)
    train_scene_ids = set()
    rows = []
    seen = set()
    for entry in value["scenes"]:
        if not isinstance(entry, dict) or entry.get("split") != "train":
            continue
        scene_id = str(entry.get("scene_id") or "").strip()
        source_scene = str(entry.get("source_scene") or "").strip()
        if not scene_id or not source_scene or scene_id in seen:
            raise ValueError("GS collision catalog scene identity is invalid")
        seen.add(scene_id)
        train_scene_ids.add(scene_id)
        source_id = source_scene.rsplit("_", 1)[-1]
        if not source_id.isdigit() or not scene_id.endswith(f"_{source_id}"):
            raise ValueError(
                f"GS scene {scene_id!r} has an invalid official source ID")
        if scene_id in excluded_set:
            continue
        raw_directory = Path(str(entry.get("path") or scene_id))
        directory = (
            raw_directory.resolve() if raw_directory.is_absolute() else
            (root / raw_directory).resolve())
        try:
            directory.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"GS collision scene path escapes data root: {scene_id}") \
                from error
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        usd = collision / source_id / f"{source_id}_collision.usd"
        if not usd.is_file():
            raise FileNotFoundError(usd)
        rows.append((
            scene_id, directory, usd,
            directory / "scene.collision.npz"))
    unknown = sorted(excluded_set - train_scene_ids)
    if unknown:
        raise ValueError(
            "excluded scene is not a GS train scene: " + ", ".join(unknown))
    if not rows:
        raise ValueError("GS collision catalog contains no train scenes")
    return rows


def _build_catalog_row(row, overwrite: bool) -> dict:
    scene_id, directory, usd, output = row
    if output.is_file() and not overwrite:
        expected_assets = tuple(_source_asset(role, path) for role, path in (
            ("scene", directory / "scene.gs.ply"),
            ("navmesh", directory / "scene.navmesh"),
            ("semantic", directory / "labels.json"),
            ("collision_mesh", usd),
        ))
        try:
            existing = gs_collision.load_collision_artifact(
                output, expected_source_assets=expected_assets)
            if existing.scene_id != scene_id:
                raise ValueError(
                    "GS collision artifact scene identity differs")
        except (EOFError, OSError, TypeError, ValueError):
            pass
        else:
            return {"scene_id": scene_id, "status": "existing",
                    "artifact_path": str(output)}
    result = build_one(directory, usd, output, scene_id=scene_id)
    return {
        "scene_id": scene_id,
        "status": "built",
        "artifact_path": result["artifact_path"],
        "artifact_bytes": result["artifact_bytes"],
        "artifact_file_sha256": result["artifact_file_sha256"],
    }


def build_catalog(data_root, manifest_path, collision_root, *, jobs: int = 1,
                  overwrite: bool = False, excluded_scenes=()) -> dict:
    """Build a resumable full-train collision catalog with bounded workers."""
    excluded = tuple(str(value) for value in excluded_scenes)
    rows = catalog_conversion_inputs(
        data_root, manifest_path, collision_root,
        excluded_scenes=excluded)
    workers = int(jobs)
    if workers < 1:
        raise ValueError("GS collision catalog jobs must be positive")
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_build_catalog_row, row, bool(overwrite)): row[0]
            for row in rows
        }
        for future in as_completed(pending):
            scene_id = pending[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append({
                    "scene_id": scene_id,
                    "error": f"{type(error).__name__}: {error}",
                })
            else:
                results.append(result)
                print(json.dumps({
                    "event": "gs_collision_catalog_progress",
                    "completed": len(results),
                    "total": len(rows),
                    **result,
                }, sort_keys=True), flush=True)
    summary = {
        "schema": "gs-collision-catalog-build.v1",
        "total_scenes": len(rows),
        "built_scenes": sum(row["status"] == "built" for row in results),
        "existing_scenes": sum(
            row["status"] == "existing" for row in results),
        "excluded_scenes": list(excluded),
        "failed_scenes": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(summary, sort_keys=True))
    return summary


def build_one(scene_dir, collision_usd, output_path, *, scene_id=None) -> dict:
    """Build one source-bound artifact and return its persisted metadata."""
    directory = Path(scene_dir).resolve(strict=True)
    resolved_scene_id = (
        directory.name if scene_id is None else str(scene_id).strip())
    if not resolved_scene_id:
        raise ValueError("GS collision scene identity is empty")
    collision = _lexical_collision_path(collision_usd)
    scene = directory / "scene.gs.ply"
    navmesh = directory / "scene.navmesh"
    semantic = directory / "labels.json"
    for path in (scene, navmesh, semantic, collision):
        if not path.is_file():
            raise FileNotFoundError(path)
    pathfinder = habitat_sim.PathFinder()
    pathfinder.load_nav_mesh(str(navmesh))
    if not pathfinder.is_loaded:
        raise ValueError(f"GS navmesh failed to load: {navmesh}")
    vertices = np.asarray(
        pathfinder.build_navmesh_vertices(), dtype=np.float64)
    if len(vertices) % 3:
        raise ValueError("GS navmesh triangle stream is malformed")
    components = _triangulated_collision_components(collision)
    levels = gs_collision.build_planar_levels(
        collision_components=components,
        navmesh_triangles=vertices.reshape(-1, 3, 3),
    )
    assets = tuple(_source_asset(role, path) for role, path in (
        ("scene", scene),
        ("navmesh", navmesh),
        ("semantic", semantic),
        ("collision_mesh", collision),
    ))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = gs_collision.write_collision_artifact(
        output,
        scene_id=resolved_scene_id,
        source_assets=assets,
        levels=levels,
    )
    return {
        **metadata,
        "artifact_path": str(output.resolve()),
        "artifact_bytes": int(output.stat().st_size),
        "artifact_file_sha256": _sha256_file(output),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir")
    parser.add_argument("--collision-usd")
    parser.add_argument(
        "--out", help="default: <scene-dir>/scene.collision.npz")
    parser.add_argument("--data-root")
    parser.add_argument("--source-manifest")
    parser.add_argument("--collision-root")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.scene_dir is None:
        if not all((args.data_root, args.source_manifest,
                    args.collision_root)):
            raise ValueError(
                "catalog conversion requires --data-root, "
                "--source-manifest, and --collision-root")
        result = build_catalog(
            args.data_root, args.source_manifest, args.collision_root,
            jobs=args.jobs, overwrite=args.overwrite,
            excluded_scenes=args.exclude_scene)
        json.dump(result, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.collision_usd is None:
        raise ValueError("single-scene conversion requires --collision-usd")
    output = args.out or str(Path(args.scene_dir) / "scene.collision.npz")
    result = build_one(args.scene_dir, args.collision_usd, output)
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
