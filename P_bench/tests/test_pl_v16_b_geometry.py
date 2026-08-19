"""Exact complete-mesh geometry tests for v16 B1/B2."""

import hashlib
import tracemalloc

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pipeline import config, geometry, semantic
from tests._synthetic import LEVEL_FLOOR


def _write_target_scene(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = tmp_path / "TARGET.glb"
    scene.write_bytes(b"")
    vertices = np.array([
        (-0.2, 2.0, 0.10, 0, 0, 0),
        (0.2, 2.0, 0.20, 0, 0, 0),
        (0.0, 2.2, 0.20, 0, 0, 0),
    ], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    faces = np.empty(1, dtype=[
        ("vertex_indices", "O"), ("object_id", "i4"),
    ])
    faces[0] = ([0, 1, 2], 0)
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ], text=False).write(tmp_path / "TARGET_semantic.ply")
    (tmp_path / "TARGET.house").write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
    )
    return scene, tmp_path / "TARGET_semantic.ply"


def test_ground_support_clips_slab_crossings_and_ignores_tabletop():
    """Catches vertex-only band filtering and tabletop projection shortcuts."""
    crossing = np.array([[
        [0.0, 0.00, 0.0],
        [2.0, 0.40, 0.0],
        [0.0, 0.40, 2.0],
    ]])
    tabletop = np.array([[
        [0.0, 0.80, 0.0],
        [2.0, 0.80, 0.0],
        [0.0, 0.80, 2.0],
    ]])
    leg = np.array([[
        [3.0, 0.10, 0.0],
        [3.2, 0.20, 0.0],
        [3.0, 0.20, 0.2],
    ]])

    support = semantic.clip_target_ground_support(
        np.concatenate([crossing, tabletop, leg]), LEVEL_FLOOR)

    triangles = np.asarray(support["triangles_xz_m"])
    assert triangles.shape[1:] == (3, 2)
    assert len(triangles) >= 2
    assert np.any(triangles[:, :, 0] >= 3.0)
    assert not np.any((triangles[:, :, 0] > 0.9) &
                      (triangles[:, :, 0] < 2.1) &
                      (triangles[:, :, 1] > 0.9))


def test_point_to_exact_union_handles_inside_edges_vertices_and_concavity():
    """Catches convex-hull or sampled-point distance replacing polygon union."""
    support = {
        "triangles_xz_m": [
            [[0, 0], [2, 0], [2, 1]],
            [[0, 0], [2, 1], [0, 1]],
            [[0, 0], [1, 1], [1, 2]],
            [[0, 0], [1, 2], [0, 2]],
            [[9, 9], [9, 9], [9, 9]],
        ],
        "segments_xz_m": [], "points_xz_m": [],
    }

    assert semantic.point_to_ground_support_distance_m([0.25, 0.25], support) == 0
    assert semantic.point_to_ground_support_distance_m([2.0, 0.5], support) == 0
    assert semantic.point_to_ground_support_distance_m([0.0, 0.0], support) == 0
    assert semantic.point_to_ground_support_distance_m([1.5, 1.5], support) == \
        pytest.approx(0.5)
    with pytest.raises(ValueError, match="empty"):
        semantic.point_to_ground_support_distance_m(
            [0, 0], {"triangles_xz_m": np.zeros((2, 3, 2)).tolist(),
                     "segments_xz_m": [], "points_xz_m": []})


def test_point_to_exact_union_returns_a_deterministic_nearest_witness():
    """The B1 Task-GT certificate must be rebuildable, not merely attested."""
    support = {
        "triangles_xz_m": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [3.0, 0.0], [2.0, 1.0]],
        ],
        "segments_xz_m": [[[4.0, 0.0], [4.0, 1.0]]],
        "points_xz_m": [[5.0, 0.0]],
    }

    distance, witness = semantic.point_to_ground_support_nearest(
        [1.5, 0.25], support)

    assert distance == pytest.approx(0.5)
    # The two triangles are tied.  Canonical component/index ordering makes
    # the first witness authoritative and stable across rebuilds.
    assert witness == pytest.approx([1.0, 0.25])
    assert semantic.point_to_ground_support_distance_m(
        [1.5, 0.25], support) == distance


def test_batched_support_distances_are_bit_exact_with_scalar_authority():
    support = {
        "triangles_xz_m": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [3.0, 0.0], [2.0, 1.0]],
            [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
        ],
        "segments_xz_m": [
            [[4.0, 0.0], [4.0, 1.0]],
            [[6.0, 2.0], [6.0, 2.0]],
        ],
        "points_xz_m": [[5.0, 0.0]],
    }
    points = np.random.default_rng(20260813).uniform(
        low=[-1.0, -1.0], high=[7.0, 3.0], size=(100, 2))
    expected = np.asarray([
        semantic.point_to_ground_support_distance_m(point, support)
        for point in points
    ])

    actual = geometry.points_to_ground_support_distances_m(points, support)

    assert np.array_equal(actual, expected)


def test_point_inside_support_uses_the_query_as_its_witness():
    support = {
        "triangles_xz_m": [
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        ],
        "segments_xz_m": [],
        "points_xz_m": [],
    }

    assert semantic.point_to_ground_support_nearest(
        [0.25, 0.25], support) == (0.0, [0.25, 0.25])


def test_full_surface_centroid_ignores_order_winding_and_subdivision():
    """Catches sampled/capped centroids and face-layout dependent results."""
    one = np.array([[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ]])
    midpoint = np.array([1.0, 1.0, 0.0])
    split = np.array([
        [one[0, 0], one[0, 1], midpoint],
        [one[0, 1], one[0, 2], midpoint],
        [one[0, 2], one[0, 0], midpoint],
    ])

    expected = np.array([2.0 / 3.0, 2.0 / 3.0, 0.0])
    np.testing.assert_allclose(
        semantic.area_weighted_surface_centroid(one), expected, atol=1e-12)
    np.testing.assert_allclose(
        semantic.area_weighted_surface_centroid(split[::-1, ::-1]),
        expected, atol=1e-12)
    with pytest.raises(ValueError, match="nondegenerate"):
        semantic.area_weighted_surface_centroid(np.zeros((1, 3, 3)))


def test_full_surface_centroid_preserves_tiny_surface_under_subdivision():
    """Catches an absolute per-face cutoff changing GT after subdivision."""
    tiny = np.array([[
        [0.0, 0.0, 0.0], [1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0],
    ]])
    center = tiny[0].mean(axis=0)
    split = np.array([
        [tiny[0, 0], tiny[0, 1], center],
        [tiny[0, 1], tiny[0, 2], center],
        [tiny[0, 2], tiny[0, 0], center],
    ])

    expected = np.array([1e-8 / 3, 1e-8 / 3, 0.0])
    np.testing.assert_allclose(
        semantic.area_weighted_surface_centroid(tiny), expected, atol=1e-24)
    np.testing.assert_allclose(
        semantic.area_weighted_surface_centroid(split), expected, atol=1e-24)


def test_ground_support_keeps_projected_segments_beside_distant_area():
    """Catches a vertical nearby component disappearing from exact union GT."""
    vertical = np.array([[
        [1.0, 0.05, 0.0], [1.0, 0.30, 0.0], [1.0, 0.20, 2.0],
    ]])
    distant_area = np.array([[
        [9.0, 0.10, 0.0], [10.0, 0.10, 0.0], [9.0, 0.10, 1.0],
    ]])

    support = semantic.clip_target_ground_support(
        np.concatenate([vertical, distant_area]), LEVEL_FLOOR)

    assert support["segments_xz_m"]
    assert support["triangles_xz_m"]
    assert semantic.point_to_ground_support_distance_m([0.0, 1.0], support) == \
        pytest.approx(1.0)


def test_ground_support_rejects_segment_only_projection():
    """Catches a zero-area XZ target becoming formal B support."""
    vertical = np.array([[
        [1.0, 0.05, 0.0], [1.0, 0.30, 0.0], [1.0, 0.20, 2.0],
    ]])

    with pytest.raises(ValueError, match="nondegenerate 2-D triangle"):
        semantic.clip_target_ground_support(vertical, LEVEL_FLOOR)


def _write_two_target_scene(tmp_path):
    scene, ply_path = _write_target_scene(tmp_path)
    ply = PlyData.read(ply_path, mmap=False)
    vertices = np.asarray(ply["vertex"].data)
    second = vertices.copy()
    second["x"] += 4.0
    combined_vertices = np.concatenate([vertices, second])
    faces = np.empty(2, dtype=[
        ("vertex_indices", "O"), ("object_id", "i4"),
    ])
    faces[0] = ([0, 1, 2], 0)
    faces[1] = ([3, 4, 5], 1)
    PlyData([
        PlyElement.describe(combined_vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ], text=False).write(ply_path)
    (tmp_path / "TARGET.house").write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "C  1  2 table  3 table  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
        "O  1 0 1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n")
    return scene, ply_path


def test_authenticated_face_index_scans_source_once_for_two_instances(
        tmp_path, monkeypatch):
    """Catches one full PLY hash/scan being repeated for every B target."""
    scene, ply_path = _write_two_target_scene(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    source_sha = hashlib.sha256(ply_path.read_bytes()).hexdigest()
    original_read = semantic.os.read
    sequential_bytes = 0

    def counted_read(descriptor, size):
        nonlocal sequential_bytes
        payload = original_read(descriptor, size)
        sequential_bytes += len(payload)
        return payload

    monkeypatch.setattr(semantic.os, "read", counted_read)

    first = index.target_geometry_atom(
        1, LEVEL_FLOOR, expected_semantic_ply_sha256=source_sha)
    after_first = sequential_bytes
    second = index.target_geometry_atom(
        2, LEVEL_FLOOR, expected_semantic_ply_sha256=source_sha)

    assert after_first == ply_path.stat().st_size
    assert sequential_bytes == after_first
    assert first["full_triangle_count"] == 1
    assert second["full_triangle_count"] == 1
    assert second["category"] == "table"
    assert second["reference_centroid"]["world_xyz_m"][0] == \
        pytest.approx(4.0)


def test_authenticated_face_index_cache_evicts_at_hard_scene_limit(
        tmp_path, monkeypatch):
    first_scene, first_ply = _write_two_target_scene(tmp_path / "first")
    second_scene, second_ply = _write_two_target_scene(tmp_path / "second")
    first_index = semantic.load_mp3d_semantic_index(
        first_scene, cache_dir=tmp_path / "cache-first", sample_count=1)
    second_index = semantic.load_mp3d_semantic_index(
        second_scene, cache_dir=tmp_path / "cache-second", sample_count=1)
    monkeypatch.setattr(
        config, "MP3D_TARGET_FACE_INDEX_CACHE_MAX_SCENES", 1)
    semantic._MP3D_TARGET_FACE_INDEX_CACHE.clear()
    original_read = semantic.os.read
    sequential_bytes = 0

    def counted_read(descriptor, size):
        nonlocal sequential_bytes
        payload = original_read(descriptor, size)
        sequential_bytes += len(payload)
        return payload

    monkeypatch.setattr(semantic.os, "read", counted_read)
    first_sha = hashlib.sha256(first_ply.read_bytes()).hexdigest()
    second_sha = hashlib.sha256(second_ply.read_bytes()).hexdigest()

    first_index.target_geometry_atom(
        1, LEVEL_FLOOR, expected_semantic_ply_sha256=first_sha)
    second_index.target_geometry_atom(
        1, LEVEL_FLOOR, expected_semantic_ply_sha256=second_sha)
    first_index.target_geometry_atom(
        1, LEVEL_FLOOR, expected_semantic_ply_sha256=first_sha)

    assert sequential_bytes == (
        2 * first_ply.stat().st_size + second_ply.stat().st_size)


def test_authenticated_face_index_rejects_entry_over_hard_byte_budget(
        tmp_path, monkeypatch):
    scene, ply_path = _write_two_target_scene(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    monkeypatch.setattr(
        config, "MP3D_TARGET_FACE_INDEX_CACHE_MAX_BYTES", 1)
    semantic._MP3D_TARGET_FACE_INDEX_CACHE.clear()

    with pytest.raises(MemoryError, match="face-index cache byte budget"):
        index.target_geometry_atom(
            1, LEVEL_FLOOR,
            expected_semantic_ply_sha256=hashlib.sha256(
                ply_path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    ("exception_type", "exception_args"), [
        (KeyboardInterrupt, ("cancelled",)), (SystemExit, (19,)),
    ])
def test_authenticated_face_index_scan_closes_before_base_exception(
        tmp_path, monkeypatch, exception_type, exception_args):
    scene, ply_path = _write_two_target_scene(tmp_path)
    index = semantic.MP3DSemanticIndex(
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.int32), {1: "chair", 2: "table"},
        ply_path, hashlib.sha256(ply_path.read_bytes()).hexdigest())
    original_open = semantic.os.open
    original_close = semantic.os.close
    opened = []
    closed = []

    def tracked_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    def interrupt(*_args, **_kwargs):
        raise exception_type(*exception_args)

    monkeypatch.setattr(semantic.os, "open", tracked_open)
    monkeypatch.setattr(semantic.os, "close", tracked_close)
    monkeypatch.setattr(semantic.os, "read", interrupt)

    with pytest.raises(exception_type) as caught:
        index.target_geometry_atom(
            1, LEVEL_FLOOR,
            expected_semantic_ply_sha256=index._semantic_ply_sha256)

    assert caught.value.args == exception_args
    assert opened and opened[-1] in closed


def test_authenticated_face_index_does_not_copy_full_vertex_prefix(tmp_path):
    """Catches restoring the ~scene-sized immutable vertex-prefix copy."""
    scene, ply_path = _write_target_scene(tmp_path)
    ply = PlyData.read(ply_path, mmap=False)
    base = np.asarray(ply["vertex"].data)
    vertex_count = 400_000
    vertices = np.zeros(vertex_count, dtype=base.dtype)
    vertices[:3] = base
    faces = np.empty(1, dtype=[
        ("vertex_indices", "O"), ("object_id", "i4"),
    ])
    faces[0] = ([0, 1, 2], 0)
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ], text=False).write(ply_path)
    source_sha = hashlib.sha256(ply_path.read_bytes()).hexdigest()
    index = semantic.MP3DSemanticIndex(
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.int32), {1: "chair"}, ply_path, source_sha)
    vertex_prefix_bytes = vertex_count * vertices.dtype.itemsize
    del vertices, ply, base

    tracemalloc.start()
    try:
        atom = index.target_geometry_atom(
            1, LEVEL_FLOOR, expected_semantic_ply_sha256=source_sha)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert atom["full_triangle_count"] == 1
    assert peak < vertex_prefix_bytes // 2


def test_target_geometry_atom_is_source_pinned_and_not_sampled(tmp_path):
    """Catches a mutated PLY or sampled instance cache becoming B authority."""
    scene, ply = _write_target_scene(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    source_sha = hashlib.sha256(ply.read_bytes()).hexdigest()

    atom = index.target_geometry_atom(
        1, LEVEL_FLOOR, expected_semantic_ply_sha256=source_sha)

    assert atom["schema"] == "b-target-geometry.v1"
    assert atom["instance_id"] == 1 and atom["category"] == "chair"
    assert atom["semantic_ply_sha256"] == source_sha
    assert atom["full_triangle_count"] == 1
    assert atom["ground_support"]["triangles_xz_m"]
    assert atom["reference_centroid"]["world_xz_m"] == \
        pytest.approx([0.0, -2.0666666667])
    assert index.instance_points(1).shape[0] == 1

    changed_scene, changed_ply = _write_target_scene(tmp_path / "changed")
    changed_index = semantic.load_mp3d_semantic_index(
        changed_scene, cache_dir=tmp_path / "cache-changed", sample_count=1)
    original_sha = hashlib.sha256(changed_ply.read_bytes()).hexdigest()
    payload = bytearray(changed_ply.read_bytes())
    payload[-1] ^= 1
    changed_ply.write_bytes(payload)
    with pytest.raises(ValueError, match="semantic PLY digest"):
        changed_index.target_geometry_atom(
            1, LEVEL_FLOOR, expected_semantic_ply_sha256=original_sha)

    with pytest.raises(ValueError, match="semantic PLY digest"):
        index.target_geometry_atom(
            1, LEVEL_FLOOR, expected_semantic_ply_sha256="0" * 64)
