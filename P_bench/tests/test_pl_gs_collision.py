import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, Polygon
import torch

from pipeline import (
    actions, b1k_geometry, config, consensus, dataset_contracts, floor_plane,
    gs_collision, gs_semantic, gs_sim, rollout, scene_pool,
)
from pipeline.b1k_geometry import TriangleComponent
from scripts import build_gs_collision_authority


def _source_assets():
    return (
        ("scene", 11, "1" * 64),
        ("navmesh", 12, "2" * 64),
        ("semantic", 13, "3" * 64),
        ("collision_mesh", 14, "4" * 64),
    )


def _level():
    return gs_collision.FrozenPlanarLevel(
        ground_y_m=0.0,
        support=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        obstacle_identities=("/Root/chair",),
        obstacle_footprints=(
            Polygon([(1.0, -1.0), (1.5, -1.0),
                     (1.5, 1.0), (1.0, 1.0)]),
        ),
    )


def test_collision_preprocessor_preserves_usd_name_for_cache_symlink(tmp_path):
    blob = tmp_path / "content-addressed-blob"
    blob.write_bytes(b"usd")
    snapshot = tmp_path / "scene_collision.usd"
    snapshot.symlink_to(blob)

    observed = build_gs_collision_authority._lexical_collision_path(snapshot)

    assert observed == snapshot.absolute()
    assert observed.suffix == ".usd"


def test_collision_builder_cli_runs_outside_repository(tmp_path):
    script = Path(build_gs_collision_authority.__file__).resolve()

    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path,
        capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert "--exclude-scene" in completed.stdout


def test_collision_catalog_maps_all_manifest_scenes_to_official_ids(tmp_path):
    data_root = tmp_path / "gs"
    collision_root = tmp_path / "collision"
    scenes = []
    for scene_id in ("interior_0007_840137", "interior_0045_839925"):
        directory = data_root / "train" / scene_id
        directory.mkdir(parents=True)
        source_id = scene_id.rsplit("_", 1)[-1]
        source = collision_root / source_id / f"{source_id}_collision.usd"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"usd")
        scenes.append({
            "scene_id": scene_id,
            "source_scene": scene_id.removeprefix("interior_"),
            "path": f"train/{scene_id}",
            "split": "train",
        })
    manifest = data_root / "splits" / "train.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "dataset": "gs", "scenes": scenes, "schema_version": 1,
    }))

    rows = build_gs_collision_authority.catalog_conversion_inputs(
        data_root, manifest, collision_root)

    assert [row[0] for row in rows] == [
        "interior_0007_840137", "interior_0045_839925"]
    assert [row[2].name for row in rows] == [
        "840137_collision.usd", "839925_collision.usd"]
    assert all(row[3].name == "scene.collision.npz" for row in rows)


def test_collision_catalog_excludes_only_explicit_train_scene(tmp_path):
    data_root = tmp_path / "gs"
    collision_root = tmp_path / "collision"
    scenes = []
    for scene_id in ("interior_0007_840137", "interior_0505_839970"):
        directory = data_root / "train" / scene_id
        directory.mkdir(parents=True)
        source_id = scene_id.rsplit("_", 1)[-1]
        source = collision_root / source_id / f"{source_id}_collision.usd"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"usd")
        scenes.append({
            "scene_id": scene_id,
            "source_scene": scene_id.removeprefix("interior_"),
            "path": f"train/{scene_id}",
            "split": "train",
        })
    manifest = data_root / "splits" / "train.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "dataset": "gs", "scenes": scenes, "schema_version": 1,
    }))

    rows = build_gs_collision_authority.catalog_conversion_inputs(
        data_root, manifest, collision_root,
        excluded_scenes=("interior_0505_839970",))

    assert [row[0] for row in rows] == ["interior_0007_840137"]
    with pytest.raises(ValueError, match="not a GS train scene"):
        build_gs_collision_authority.catalog_conversion_inputs(
            data_root, manifest, collision_root,
            excluded_scenes=("not-in-catalog",))


def test_collision_catalog_rebuilds_obsolete_existing_artifact(
        tmp_path, monkeypatch):
    directory = tmp_path / "scene"
    directory.mkdir()
    for name in ("scene.gs.ply", "scene.navmesh", "labels.json"):
        (directory / name).write_bytes(name.encode("ascii"))
    usd = tmp_path / "scene.usd"
    usd.write_bytes(b"usd")
    output = directory / "scene.collision.npz"
    output.write_bytes(b"obsolete")
    calls = []

    def build(scene_dir, collision_usd, output_path, *, scene_id):
        calls.append((scene_dir, collision_usd, output_path, scene_id))
        output.write_bytes(b"current")
        return {
            "artifact_path": str(output), "artifact_bytes": 7,
            "artifact_file_sha256": "a" * 64,
        }

    monkeypatch.setattr(build_gs_collision_authority, "build_one", build)
    result = build_gs_collision_authority._build_catalog_row(
        ("scene", directory, usd, output), overwrite=False)

    assert result["status"] == "built"
    assert calls == [(directory, usd, output, "scene")]


def test_collision_catalog_build_uses_manifest_scene_identity(
        tmp_path, monkeypatch):
    """Catches binding a path alias instead of the catalog scene id."""
    directory = tmp_path / "manifest-path-alias"
    directory.mkdir()
    for name in ("scene.gs.ply", "scene.navmesh", "labels.json"):
        (directory / name).write_bytes(name.encode("ascii"))
    usd = tmp_path / "scene.usd"
    usd.write_bytes(b"usd")
    output = directory / "scene.collision.npz"
    calls = []

    def build(scene_dir, collision_usd, output_path, *, scene_id):
        calls.append((scene_dir, collision_usd, output_path, scene_id))
        return {
            "artifact_path": str(output), "artifact_bytes": 7,
            "artifact_file_sha256": "a" * 64,
        }

    monkeypatch.setattr(build_gs_collision_authority, "build_one", build)
    catalog_scene_id = "interior_0007_840137"

    result = build_gs_collision_authority._build_catalog_row(
        (catalog_scene_id, directory, usd, output), overwrite=False)

    assert result["status"] == "built"
    assert calls == [(directory, usd, output, catalog_scene_id)]


def test_collision_catalog_rebuilds_source_stale_existing_artifact(
        tmp_path, monkeypatch):
    """Catches reusing valid NPZ geometry derived from another USD."""
    directory = tmp_path / "scene"
    directory.mkdir()
    sources = {
        "scene": directory / "scene.gs.ply",
        "navmesh": directory / "scene.navmesh",
        "semantic": directory / "labels.json",
        "collision_mesh": tmp_path / "scene.usd",
    }
    for role, path in sources.items():
        path.write_bytes(f"current-{role}".encode("ascii"))
    output = directory / "scene.collision.npz"
    current_assets = tuple(
        build_gs_collision_authority._source_asset(role, path)
        for role, path in sources.items())
    stale_assets = list(current_assets)
    stale_assets[-1] = (
        "collision_mesh", stale_assets[-1][1], "f" * 64)
    gs_collision.write_collision_artifact(
        output, scene_id=directory.name,
        source_assets=stale_assets, levels=(_level(),))
    calls = []

    def build(scene_dir, collision_usd, output_path, *, scene_id):
        calls.append((scene_dir, collision_usd, output_path, scene_id))
        return {
            "artifact_path": str(output), "artifact_bytes": 7,
            "artifact_file_sha256": "a" * 64,
        }

    monkeypatch.setattr(build_gs_collision_authority, "build_one", build)

    result = build_gs_collision_authority._build_catalog_row(
        (directory.name, directory, sources["collision_mesh"], output),
        overwrite=False)

    assert result["status"] == "built"
    assert calls == [(
        directory, sources["collision_mesh"], output, directory.name)]


def test_gs_collision_artifact_is_deterministic_and_source_bound(tmp_path):
    """Changing source identity or serialized geometry must invalidate it."""
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    for path in (first, second):
        gs_collision.write_collision_artifact(
            path, scene_id="interior_0007_840137",
            source_assets=_source_assets(), levels=(_level(),))

    assert first.read_bytes() == second.read_bytes()
    file_sha256 = hashlib.sha256(first.read_bytes()).hexdigest()
    authority = gs_collision.load_collision_artifact(
        first, expected_file_sha256=file_sha256,
        expected_source_assets=_source_assets())
    assert authority.scene_id == "interior_0007_840137"
    assert authority.authority == "gs_collision_mesh"
    assert authority.content_authority_sha256 == \
        authority.binding_atom()["content_sha256"]
    assert authority.geometry_authority_sha256 == file_sha256
    assert authority.ground_y([0.0, 0.2, 0.0]) == pytest.approx(0.0)

    forged_assets = list(_source_assets())
    forged_assets[-1] = ("collision_mesh", 14, "5" * 64)
    with pytest.raises(ValueError, match="source assets"):
        gs_collision.load_collision_artifact(
            first, expected_file_sha256=file_sha256,
            expected_source_assets=tuple(forged_assets))


def test_gs_collision_artifact_drives_radius_conditioned_contact(tmp_path):
    """Removing the loaded footprint or body expansion must fail this test."""
    path = tmp_path / "authority.npz"
    gs_collision.write_collision_artifact(
        path, scene_id="interior_0007_840137",
        source_assets=_source_assets(), levels=(_level(),))
    authority = gs_collision.load_collision_artifact(path)
    nav = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.20)
    assert nav.geometry_authority_sha256 == hashlib.sha256(
        path.read_bytes()).hexdigest()

    assert nav.query_pose((0.0, 0.0, 0.0)).navigable is True
    contact = nav.query_pose((0.85, 0.0, 0.0))
    assert contact.navigable is False
    assert contact.geometry_source == "gs_collision_mesh"
    closest = nav.closest_obstacle((0.85, 0.0, 0.0))
    assert closest["obstacle_identity"] == "/Root/chair"
    assert closest["surface_protocol"] == \
        "sage3d-source-collision-footprint.v1"
    assert np.allclose(closest["world_point"], [1.0, 0.175, 0.0])


def test_gs_ground_alignment_prefers_dominant_near_coplanar_support(
        tmp_path):
    """A tiny raised shell must not replace the actual navmesh floor.

    Official InteriorGS navmeshes report points about 0.20 m above the source
    floor.  Several collision USDs also contain millimetre-raised horizontal
    furniture shells.  Nearest-height selection alone therefore picks the
    shell even when the source floor covers the same XZ position.
    """
    obstacle_identities = ("/Root/wall",)
    obstacle_footprints = (GeometryCollection(),)

    def level(height, support):
        return gs_collision.FrozenPlanarLevel(
            ground_y_m=height,
            support=support,
            obstacle_identities=obstacle_identities,
            obstacle_footprints=obstacle_footprints,
        )

    floor = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    shell = Polygon([(-0.5, -0.5), (0.5, -0.5),
                     (0.5, 0.5), (-0.5, 0.5)])
    path = tmp_path / "near-coplanar.npz"
    gs_collision.write_collision_artifact(
        path, scene_id="interior_0405_840145",
        source_assets=_source_assets(),
        levels=(
            level(-0.12, floor),
            level(0.0, floor),
            level(0.01, shell),
        ),
    )

    authority = gs_collision.load_collision_artifact(path)

    assert authority.ground_y([0.0, 0.20, 0.0]) == pytest.approx(0.0)


def test_gs_collision_runtime_uses_distance_conditioning_without_buffers(
        tmp_path, monkeypatch):
    """GS startup must not build the prohibitively large obstacle buffers."""
    path = tmp_path / "prepared-authority.npz"
    gs_collision.write_collision_artifact(
        path, scene_id="interior_0007_840137",
        source_assets=_source_assets(), levels=(_level(),))

    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata_json"].item()))
    assert metadata["collision_query_mode"] == "nearest_obstacle_distance"
    assert all(
        "configuration_obstacle_geometry_index" not in radius
        for level in metadata["levels"]
        for radius in level["radius_support"])

    def forbidden(_values):
        raise AssertionError("GS runtime must not rebuild obstacle unions")

    monkeypatch.setattr(b1k_geometry, "unary_union", forbidden)
    authority = gs_collision.load_collision_artifact(path)
    nav = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.20)

    assert authority.binding_atom()["schema"] == \
        "gs-collision-footprint-authority.v3"
    assert nav.query_pose((0.85, 0.0, 0.0)).navigable is False


def test_distance_conditioned_planar_queries_match_buffered_geometry():
    """Nearest-distance C-space is the exact disc-buffer predicate."""
    level = _level()
    levels = ((
        level.ground_y_m, level.support, level.obstacle_footprints),)
    common = {
        "levels": levels,
        "obstacle_identities": level.obstacle_identities,
        "authority_name": "synthetic",
        "contact_surface_protocol": "synthetic.v1",
    }
    buffered = b1k_geometry.B1KGeometryAuthority.from_planar_levels(**common)
    prepared_support = ({
        float(radius): level.support.buffer(
            -float(radius), quad_segs=b1k_geometry._BUFFER_QUAD_SEGS)
        for radius in config.RADII_M
    },)
    distance = b1k_geometry.B1KGeometryAuthority.from_planar_levels(
        **common, collision_query_mode="distance",
        prepared_supported_centers=prepared_support)
    poses = [
        (-4.9, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.79, 0.0, 0.0),
        (0.80, 0.0, 0.0),
        (0.81, 0.0, 0.0),
        (1.25, 0.0, 0.0),
        (1.71, 0.0, 0.0),
    ]
    for radius in config.RADII_M:
        buffered_nav = buffered.bind(
            [0.0, 0.0, 0.0], 0.0, radius_m=radius)
        distance_nav = distance.bind(
            [0.0, 0.0, 0.0], 0.0, radius_m=radius)
        expected = buffered_nav.query_many(poses)
        observed = distance_nav.query_many(poses)
        assert [value.navigable for value in observed] == [
            value.navigable for value in expected]
        assert [value.obstacle_index for value in observed] == [
            value.obstacle_index for value in expected]
        assert [value.clearance_m for value in observed] == pytest.approx([
            value.clearance_m for value in expected])


def test_gs_collision_loader_rejects_geometry_tampering(tmp_path):
    """Changing an obstacle ring without rebuilding its digest must fail."""
    path = tmp_path / "authority.npz"
    gs_collision.write_collision_artifact(
        path, scene_id="interior_0007_840137",
        source_assets=_source_assets(), levels=(_level(),))
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays["geometry_wkb_bytes"][0] ^= np.uint8(1)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="authority digest"):
        gs_collision.load_collision_artifact(path)


def test_gs_collision_artifact_preserves_vertical_wall_footprints(tmp_path):
    """A zero-area wall projection must remain a collision obstacle."""
    level = gs_collision.FrozenPlanarLevel(
        ground_y_m=0.0,
        support=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        obstacle_identities=("/Root/wall",),
        obstacle_footprints=(GeometryCollection((
            LineString(((1.0, -1.0), (1.0, 1.0))),
        )),),
    )
    path = tmp_path / "wall.npz"
    gs_collision.write_collision_artifact(
        path, scene_id="interior_0007_840137",
        source_assets=_source_assets(), levels=(level,))

    authority = gs_collision.load_collision_artifact(path)
    nav = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.20)

    assert nav.query_pose((0.79, 0.0, 0.0)).navigable is True
    assert nav.query_pose((0.81, 0.0, 0.0)).navigable is False
    assert nav.closest_obstacle((0.81, 0.0, 0.0))["obstacle_identity"] == \
        "/Root/wall"


def test_build_gs_planar_levels_uses_source_floor_and_vertical_wall():
    floor = np.asarray([
        [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0]],
        [[-2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]],
    ])
    wall = np.asarray([
        [[1.0, 0.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]],
        [[1.0, 0.0, -1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
    ])
    navmesh = floor.copy()
    navmesh[:, :, 1] = 0.20

    levels = gs_collision.build_planar_levels(
        collision_components=(
            TriangleComponent("/Root/floor", floor),
            TriangleComponent("/Root/wall", wall),
        ),
        navmesh_triangles=navmesh,
    )

    assert len(levels) == 1
    assert levels[0].ground_y_m == pytest.approx(0.0)
    assert levels[0].support.area == pytest.approx(16.0)
    assert levels[0].obstacle_identities == ("/Root/floor", "/Root/wall")
    assert levels[0].obstacle_footprints[0].is_empty
    assert levels[0].obstacle_footprints[1].length == pytest.approx(2.0)


def test_gs_collision_union_repairs_invalid_source_projection():
    """Official USD slivers must not make offline conversion nondeterministic."""
    bow_tie = Polygon([
        (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0),
    ])
    wall = LineString(((2.0, 0.0), (2.0, 1.0)))

    first = gs_collision.canonical_planar_union((bow_tie, wall))
    second = gs_collision.canonical_planar_union((bow_tie, wall))

    assert first.is_valid
    assert first.equals(second)
    assert first.area == pytest.approx(0.5)
    assert first.length > wall.length


def test_gs_collision_overlap_uses_same_precision_repair():
    first = Polygon([
        (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0),
    ])
    second = Polygon([
        (1.0 + 3e-10, -1.0), (3.0, -1.0),
        (3.0, 1.0), (1.0 + 3e-10, 1.0),
    ])

    assert gs_collision.planar_overlap_area(first, second) == \
        pytest.approx(1.0)


def test_gs_collision_vectorized_slab_projection_matches_scalar_contract():
    """The fast USD converter must preserve the frozen triangle clipping."""
    low, high = 0.05, 0.30
    triangles = np.asarray([
        # Fully inside the band.
        [[0.0, 0.10, 0.0], [1.0, 0.10, 0.0], [0.0, 0.10, 1.0]],
        # Crosses the lower plane.
        [[0.0, 0.00, 0.0], [1.0, 0.20, 0.0], [0.0, 0.20, 1.0]],
        # Crosses both planes without an original vertex inside.
        [[0.0, -0.10, 0.0], [1.0, 0.40, 0.0], [0.0, 0.40, 1.0]],
        # A vertical wall projects to a line.
        [[2.0, 0.00, 0.0], [2.0, 0.40, 0.0], [2.0, 0.20, 1.0]],
        # Entirely outside the band and therefore absent.
        [[3.0, 0.40, 0.0], [4.0, 0.40, 0.0], [3.0, 0.40, 1.0]],
    ], dtype=np.float64)

    observed = gs_collision.clipped_triangle_projections(
        triangles, low, high)
    expected = tuple(
        b1k_geometry._clipped_triangle_projection(triangle, low, high)
        for triangle in triangles
    )
    expected = tuple(value for value in expected if not value.is_empty)

    assert len(observed) == len(expected)
    assert all(left.equals(right)
               for left, right in zip(observed, expected))


def test_interiorgs_zup_coordinates_map_to_pbench_yup():
    source = np.asarray([[2.0, -3.0, 4.0], [-5.0, 6.0, 7.0]])
    assert np.array_equal(
        gs_collision.interiorgs_zup_to_pbench_xyz(source),
        [[2.0, 4.0, 3.0], [-5.0, 7.0, -6.0]],
    )


def test_gs_scene_provenance_binds_preprocessed_collision_authority(tmp_path):
    files = {}
    for name in ("scene.gs.ply", "scene.navmesh", "labels.json",
                 "scene.collision.npz"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        files[name] = path
    spec = scene_pool.SceneSpec(
        scene_id="interior_0007_840137",
        source_dataset="gs",
        official_split="train",
        scene_path=str(files["scene.gs.ply"]),
        navmesh_path=str(files["scene.navmesh"]),
        semantic_path=str(files["labels.json"]),
        semantic_metadata_path=None,
        semantic_format="gs_bbox",
        scene_dataset_config=None,
        provenance_path=str(tmp_path / "split.json"),
        provenance_sha256="a" * 64,
        collision_authority_path=str(files["scene.collision.npz"]),
    )

    assert [row["role"] for row in spec.provenance()["source_assets"]] == [
        "scene", "navmesh", "semantic", "collision_authority",
    ]
    binding = dataset_contracts.resolve_gs_collision_binding(
        spec.provenance())
    atom = dataset_contracts.gs_collision_binding_atom(binding)
    assert atom["collision_authority_sha256"] == \
        spec.provenance()["source_assets"][3]["sha256"]
    assert atom["protocol_atom"]["geometry"] == {
        "protocol": "sage3d-source-collision-footprint.v1",
        "body_model": "radius-conditioned-planar-disc",
        "source": "official_sage3d_collision_usd",
    }
    assert "sigma" not in str(atom).lower()


def test_gs_session_rejects_collision_artifact_for_another_scene(tmp_path):
    paths = {}
    for name in ("scene.gs.ply", "scene.navmesh", "labels.json"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    embedded_assets = tuple(
        (role, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for role, path in (
            ("scene", paths["scene.gs.ply"]),
            ("navmesh", paths["scene.navmesh"]),
            ("semantic", paths["labels.json"]),
        )) + (("collision_mesh", 1, "f" * 64),)
    collision_path = tmp_path / "scene.collision.npz"
    gs_collision.write_collision_artifact(
        collision_path, scene_id="wrong-scene",
        source_assets=embedded_assets, levels=(_level(),))
    spec = scene_pool.SceneSpec(
        scene_id="interior_0007_840137", source_dataset="gs",
        official_split="train", scene_path=str(paths["scene.gs.ply"]),
        navmesh_path=str(paths["scene.navmesh"]),
        semantic_path=str(paths["labels.json"]),
        semantic_metadata_path=None, semantic_format="gs_bbox",
        scene_dataset_config=None, provenance_path=str(tmp_path / "split.json"),
        provenance_sha256="a" * 64,
        collision_authority_path=str(collision_path),
    )

    with pytest.raises(ValueError, match="scene identity"):
        gs_sim.GsSimSession(spec, device="cpu")


def test_gs_session_uses_collision_artifact_and_navmesh_only_for_proposals(
        tmp_path, monkeypatch):
    paths = {}
    for name in ("scene.gs.ply", "scene.navmesh", "labels.json"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    embedded_assets = tuple(
        (role, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for role, path in (
            ("scene", paths["scene.gs.ply"]),
            ("navmesh", paths["scene.navmesh"]),
            ("semantic", paths["labels.json"]),
        )) + (("collision_mesh", 1, "f" * 64),)
    collision_path = tmp_path / "scene.collision.npz"
    gs_collision.write_collision_artifact(
        collision_path, scene_id="interior_0007_840137",
        source_assets=embedded_assets, levels=(_level(),))
    spec = scene_pool.SceneSpec(
        scene_id="interior_0007_840137", source_dataset="gs",
        official_split="train", scene_path=str(paths["scene.gs.ply"]),
        navmesh_path=str(paths["scene.navmesh"]),
        semantic_path=str(paths["labels.json"]),
        semantic_metadata_path=None, semantic_format="gs_bbox",
        scene_dataset_config=None, provenance_path=str(tmp_path / "split.json"),
        provenance_sha256="a" * 64,
        collision_authority_path=str(collision_path),
    )

    class FakePathFinder:
        is_loaded = True

        def load_nav_mesh(self, _path):
            return True

        def get_random_navigable_point(self):
            return np.asarray([0.0, 0.2, 0.0])

        def distance_to_closest_obstacle(self, _position):
            return 4.0

    class FakeSemantic:
        id_to_cat = {1: "chair"}
        alignment_certificate = None

        @staticmethod
        def assign(points):
            return np.zeros(len(points), dtype=np.int64)

        @staticmethod
        def visible_depth_view(points, ids, **binding):
            return points, ids, binding

    monkeypatch.setattr(gs_sim.habitat_sim, "PathFinder", FakePathFinder)
    monkeypatch.setattr(gs_sim, "load_gs", lambda *_args, **_kwargs: {
        "means": torch.zeros((1, 3)),
        "opacities": torch.ones(1),
        "device": "cpu",
    })
    monkeypatch.setattr(
        gs_sim.gs_semantic, "load_bbox_index",
        lambda *_args, **_kwargs: FakeSemantic())

    session = gs_sim.GsSimSession(spec, device="cpu")

    assert session.nav([0.0, 0.0, 0.0], 0.0).authority == \
        "gs_collision_mesh"
    assert session.proposal_nav(
        [0.0, 0.0, 0.0], 0.0, radius_m=0.2).authority == \
        "gs_navmesh_proposal"
    assert session._sampling_pf.get_random_navigable_point()[1] == \
        pytest.approx(0.0)
    visible = session.frame_semantic_index(
        np.zeros((1, 3)), np.zeros(1, dtype=np.int64))
    assert visible[2]["geometry_authority_sha256"] == \
        session.collision_authority_binding.authority_sha256


def test_gs_physical_rollout_carries_collision_artifact_digest(tmp_path):
    path = tmp_path / "authority.npz"
    gs_collision.write_collision_artifact(
        path, scene_id="interior_0007_840137",
        source_assets=_source_assets(), levels=(_level(),))
    authority = gs_collision.load_collision_artifact(path)
    nav = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.20)

    outcome = rollout.physical_rollout(nav, [actions.Forward(1.0)])

    assert outcome["physical"]["authority"] == "gs_collision_mesh"
    assert outcome["physical"]["geometry_authority_sha256"] == \
        hashlib.sha256(path.read_bytes()).hexdigest()


def test_gs_semantics_use_per_frame_visible_depth_for_a3():
    base = gs_semantic.BboxSemanticIndex(
        mins=np.asarray([[0.5, 0.0, -0.5], [1.5, 0.0, -0.5]]),
        maxs=np.asarray([[1.5, 1.0, 0.5], [2.5, 1.0, 0.5]]),
        ids=np.asarray([1, 2]), id_to_cat={1: "chair", 2: "table"},
        alignment_certificate={"accepted": True, "sha256": "c" * 64},
    )
    chair = np.asarray([
        [1.0 + 0.002 * index, 0.10, -0.05 + 0.004 * index]
        for index in range(24)
    ])
    table = np.asarray([
        [2.0 + 0.002 * index, 0.10, -0.05 + 0.004 * index]
        for index in range(24)
    ])
    world = np.concatenate([chair, table])
    view = base.visible_depth_view(
        world, np.asarray([1] * len(chair) + [2] * len(table)),
        geometry_authority_sha256="a" * 64,
        semantic_source_sha256="b" * 64,
    )

    assert np.array_equal(view.instance_points(1), chair)
    confirmed = view.confirm_contact_instances([(1, [1.0, 0.175, 0.0])])[0]
    assert confirmed["confirmed"] is True
    assert confirmed["schema"] == \
        "gs-visible-contact-instance-identity.v1"
    assert confirmed["instance_id"] == 1


def test_gs_visible_a3_proof_is_bound_to_the_exact_source_bundle():
    from tests._synthetic import source_provenance

    source = source_provenance("synthetic", dataset="gs")
    binding = dataset_contracts.resolve_authority_binding(source)
    base = gs_semantic.BboxSemanticIndex(
        mins=np.asarray([[-0.5, 0.0, 0.5], [1.5, 0.0, 0.5]]),
        maxs=np.asarray([[0.5, 1.0, 1.5], [2.5, 1.0, 1.5]]),
        ids=np.asarray([1, 2]), id_to_cat={1: "chair", 2: "table"},
        alignment_certificate={"accepted": True, "sha256": "c" * 64},
    )
    chair = np.asarray([[0.01 * i, 0.1, 1.0] for i in range(24)])
    table = np.asarray([[2.0 + 0.01 * i, 0.1, 1.0] for i in range(24)])
    semantic_sha256 = next(
        row["sha256"] for row in source["source_assets"]
        if row["role"] == "semantic")
    view = base.visible_depth_view(
        np.concatenate([chair, table]),
        np.asarray([1] * len(chair) + [2] * len(table)),
        geometry_authority_sha256=binding.source_sha256,
        semantic_source_sha256=semantic_sha256)

    proof = view.confirm_contact_instances([(1, chair[0])])[0]

    assert proof["confirmed"] is True
    assert consensus.a3_exact_contact_identity_valid(
        proof, full_instance_id=1, depth_instance_id=1,
        authority_binding=binding)
    forged = dict(proof, semantic_source_sha256="f" * 64)
    forged["global_universe_sha256"] = \
        gs_semantic.complete_visible_instance_universe_sha256(
            geometry_authority_sha256=forged["geometry_authority_sha256"],
            semantic_source_sha256=forged["semantic_source_sha256"],
            alignment_certificate_sha256=
                forged["alignment_certificate_sha256"])
    assert not consensus.a3_exact_contact_identity_valid(
        forged, full_instance_id=1, depth_instance_id=1,
        authority_binding=binding)
