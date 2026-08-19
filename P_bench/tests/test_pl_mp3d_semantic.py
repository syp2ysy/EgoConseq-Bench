"""MP3D semantic-mesh adapter tests."""

import hashlib
import os

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from pipeline import config, semantic, sim


class _SemanticQuerySpy:
    def __init__(self):
        self.workers = []

    def query(self, points, *, k, workers):
        self.workers.append(int(workers))
        count = len(points)
        return np.zeros(count, dtype=np.float64), np.zeros(
            count, dtype=np.int64)


def test_semantic_assign_parallelizes_only_full_frame_queries():
    index = semantic.SemanticIndex(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([7], dtype=np.int32), {7: "chair"},
        query_workers=8)
    spy = _SemanticQuerySpy()
    index._tree = spy

    small = np.zeros(
        (config.SEMANTIC_ASSIGN_PARALLEL_MIN_POINTS - 1, 3),
        dtype=np.float32)
    large = np.zeros(
        (config.SEMANTIC_ASSIGN_PARALLEL_MIN_POINTS, 3),
        dtype=np.float32)

    assert np.all(index.assign(small) == 7)
    assert np.all(index.assign(large) == 7)
    assert spy.workers == [1, 8]


def test_semantic_assign_parallel_workers_preserve_exact_ids():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    instances = np.array([1, 2, 3], dtype=np.int32)
    queries = np.repeat(points, 10_000, axis=0)
    serial = semantic.SemanticIndex(
        points, instances, {1: "a", 2: "b", 3: "c"},
        query_workers=1)
    parallel = semantic.SemanticIndex(
        points, instances, {1: "a", 2: "b", 3: "c"},
        query_workers=4)

    np.testing.assert_array_equal(
        parallel.assign(queries, tol=1e-9),
        serial.assign(queries, tol=1e-9))


def test_semantic_index_rejects_nonpositive_query_workers():
    with pytest.raises(ValueError, match="query workers must be positive"):
        semantic.SemanticIndex(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([1], dtype=np.int32), {1: "chair"},
            query_workers=0)


def test_mp3d_loader_propagates_semantic_query_workers(tmp_path):
    scene = _write_mp3d_semantics(tmp_path)

    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1,
        query_workers=6)

    assert index._query_workers == 6


def test_sim_semantic_loader_propagates_query_workers(monkeypatch):
    observed = {}
    sentinel = object()

    def load(scene_glb, *, query_workers):
        observed.update(scene_glb=scene_glb, query_workers=query_workers)
        return sentinel

    monkeypatch.setattr(semantic, "load_mp3d_semantic_index", load)

    assert sim.load_scene_semantic_index(
        "scene.glb", "mp3d_ply", query_workers=8) is sentinel
    assert observed == {"scene_glb": "scene.glb", "query_workers": 8}


def _write_mp3d_semantics(tmp_path, face_object_ids=(0, 1)):
    scene = tmp_path / "AAA.glb"
    scene.write_text("")
    vertices = np.array([
        (0.0, 0.0, 0.0, 255, 0, 0),
        (1.0, 0.0, 0.0, 255, 0, 0),
        (0.0, 1.0, 0.0, 255, 0, 0),
        (0.0, 0.0, 1.0, 0, 255, 0),
        (1.0, 0.0, 1.0, 0, 255, 0),
        (0.0, 1.0, 1.0, 0, 255, 0),
    ], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    faces = np.empty(2, dtype=[
        ("vertex_indices", "O"), ("object_id", "i4"),
    ])
    faces[0] = ([0, 1, 2], int(face_object_ids[0]))
    faces[1] = ([3, 4, 5], int(face_object_ids[1]))
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ], text=False).write(tmp_path / "AAA_semantic.ply")
    (tmp_path / "AAA.house").write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "C  1  2 dining_table  5 table  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
        "O  1 0 1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
    )
    return scene


def _write_many_face_mp3d_semantics(
        tmp_path, face_count=20, *, bad_face_index=None,
        bad_vertex_coordinate=None):
    scene = tmp_path / "MANY.glb"
    scene.write_text("")
    vertices = np.zeros(3 * face_count, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    faces = np.empty(face_count, dtype=[
        ("vertex_indices", "O"), ("object_id", "i4"),
    ])
    for index in range(face_count):
        start = 3 * index
        vertices["x"][start:start + 3] = [index, index + 0.5, index]
        vertices["y"][start:start + 3] = [0.0, 0.0, 0.5]
        faces[index] = ([start, start + 1, start + 2], 0)
    if bad_face_index is not None:
        faces[-1]["vertex_indices"][0] = int(bad_face_index)
    if bad_vertex_coordinate is not None:
        vertices[-1]["x"] = float(bad_vertex_coordinate)
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ], text=False).write(tmp_path / "MANY_semantic.ply")
    (tmp_path / "MANY.house").write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
    )
    return scene


def _run_complete_face_buffer_query(
        path, function, points, witness_ids, categories, *, query_workers=1):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(descriptor).st_size
        pinned = semantic.mmap.mmap(
            descriptor, 0, access=semantic.mmap.ACCESS_READ)
    finally:
        os.close(descriptor)
    try:
        return function(
            pinned, size, path, np.asarray(points, dtype=np.float64),
            list(witness_ids), dict(categories),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            query_workers=query_workers)
    finally:
        pinned.close()


def test_a3_bounded_query_is_exactly_equal_to_exhaustive(tmp_path):
    _write_mp3d_semantics(tmp_path)
    path = tmp_path / "AAA_semantic.ply"
    points = np.array([
        [0.25, 0.0, -0.25],
        [0.25, 0.5, -0.25],
        [0.25, 1.0, -0.25],
    ])
    kwargs = (path, points, [1, 1, 2], {1: "chair", 2: "table"})

    exhaustive = _run_complete_face_buffer_query(
        kwargs[0], semantic._query_mp3d_complete_face_universe_buffer_exhaustive,
        *kwargs[1:])
    bounded = _run_complete_face_buffer_query(
        kwargs[0], semantic._query_mp3d_complete_face_universe_buffer_bounded,
        *kwargs[1:])

    assert bounded == exhaustive


def test_a3_bounded_query_deduplicates_points_not_witnesses(tmp_path):
    semantic._reset_mp3d_query_diagnostics()
    _write_mp3d_semantics(tmp_path)
    path = tmp_path / "AAA_semantic.ply"
    point = [0.25, 0.0, -0.25]
    points = [point, point, point]
    witnesses = [1, 2, 1]
    categories = {1: "chair", 2: "table"}

    exhaustive = _run_complete_face_buffer_query(
        path, semantic._query_mp3d_complete_face_universe_buffer_exhaustive,
        points, witnesses, categories)
    bounded = _run_complete_face_buffer_query(
        path, semantic._query_mp3d_complete_face_universe_buffer_bounded,
        points, witnesses, categories)

    assert bounded == exhaustive
    diagnostics = semantic._mp3d_query_diagnostics()
    assert diagnostics["bounded_request_points"] == 3
    assert diagnostics["bounded_unique_query_points"] == 1
    assert diagnostics["bounded_query_points"] == 1
    profile_rows = semantic._mp3d_query_profile_rows()
    assert len(profile_rows) == 1
    assert profile_rows[0]["request_count"] == 3
    assert profile_rows[0]["unique_witness_instance_count"] == 2
    assert profile_rows[0]["scene_face_count"] == 2
    assert profile_rows[0]["scene_instance_count"] == 2
    assert 1 <= profile_rows[0]["shortlisted_instance_count"] <= 2
    assert 0.0 < profile_rows[0]["shortlisted_face_ratio"] <= 1.0
    assert profile_rows[0]["witness_categories"] == ["chair", "table"]


def test_a3_bounded_query_parallelizes_unique_points_exactly(tmp_path):
    semantic._reset_mp3d_query_diagnostics()
    _write_mp3d_semantics(tmp_path)
    path = tmp_path / "AAA_semantic.ply"
    points = [
        [0.25, 0.0, -0.25],
        [0.25, 0.5, -0.25],
        [0.25, 1.0, -0.25],
    ]
    witnesses = [1, 1, 2]
    categories = {1: "chair", 2: "table"}

    exhaustive = _run_complete_face_buffer_query(
        path, semantic._query_mp3d_complete_face_universe_buffer_exhaustive,
        points, witnesses, categories)
    bounded = _run_complete_face_buffer_query(
        path, semantic._query_mp3d_complete_face_universe_buffer_bounded,
        points, witnesses, categories, query_workers=4)

    assert bounded == exhaustive
    diagnostics = semantic._mp3d_query_diagnostics()
    assert diagnostics["bounded_parallel_batches"] == 1
    assert diagnostics["bounded_parallel_query_points"] == 3


def test_a3_branch_bound_continues_at_equal_runner_lower_bound():
    calls = []
    distances = {1: 0.0, 2: 1.0, 3: 1.0}

    ranked, evaluated = semantic._branch_and_bound_instance_top_two(
        np.zeros(3),
        np.array([1, 2, 3]),
        np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0]]),
        lambda instance_id: calls.append(instance_id) or distances[instance_id],
        slack_m=1e-9)

    assert ranked == [(0.0, 1), (1.0, 2)]
    assert evaluated == [1, 2, 3]
    assert calls == [1, 2, 3]


def test_a3_branch_bound_slack_changes_only_shortlist():
    distances = {1: 0.0, 2: 1.0, 3: 2.0}
    minima = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0 + 5e-10, 0.0, 0.0],
    ])

    without_slack = semantic._branch_and_bound_instance_top_two(
        np.zeros(3), np.array([1, 2, 3]), minima, minima,
        distances.__getitem__, slack_m=0.0)
    with_slack = semantic._branch_and_bound_instance_top_two(
        np.zeros(3), np.array([1, 2, 3]), minima, minima,
        distances.__getitem__, slack_m=1e-9)

    assert without_slack[0] == with_slack[0] == [(0.0, 1), (1.0, 2)]
    assert without_slack[1] == [1, 2]
    assert with_slack[1] == [1, 2, 3]


def test_a3_aabb_cache_has_no_face_permutation_or_lazy_rows(tmp_path):
    semantic._MP3D_TARGET_FACE_INDEX_CACHE.clear()
    _write_mp3d_semantics(tmp_path)
    path = tmp_path / "AAA_semantic.ply"

    _run_complete_face_buffer_query(
        path, semantic._query_mp3d_complete_face_universe_buffer_bounded,
        [[0.25, 0.0, -0.25]], [1], {1: "chair", 2: "table"})

    assert len(semantic._MP3D_TARGET_FACE_INDEX_CACHE) == 1
    entry = next(iter(semantic._MP3D_TARGET_FACE_INDEX_CACHE.values()))
    assert set(entry) >= {
        "instance_aabb_instance_ids", "instance_aabb_min_world_xyz",
        "instance_aabb_max_world_xyz", "cache_bytes",
    }
    assert not any("permutation" in key or "face_rows" in key
                   or "triangles" in key for key in entry)
    np.testing.assert_array_equal(
        entry["instance_aabb_instance_ids"], [1, 2])
    np.testing.assert_array_equal(
        entry["instance_aabb_min_world_xyz"],
        [[0.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    np.testing.assert_array_equal(
        entry["instance_aabb_max_world_xyz"],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])


def test_a3_optimized_error_uses_exhaustive_result_and_counts_fallback(
        monkeypatch):
    sentinel = {"queries": [], "streaming_instance_faces_sha256": {}}
    semantic._reset_mp3d_query_diagnostics()
    monkeypatch.setattr(
        semantic, "_query_mp3d_complete_face_universe_buffer_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("optimized failed")))
    monkeypatch.setattr(
        semantic, "_query_mp3d_complete_face_universe_buffer_exhaustive",
        lambda *_args, **_kwargs: sentinel)

    assert semantic._query_mp3d_complete_face_universe_buffer(
        None, 0, "unused", np.zeros((1, 3)), [1], {1: "chair"},
        "a" * 64) is sentinel
    assert semantic._mp3d_query_diagnostics()["fallback_invocations"] == 1


def test_a3_optimized_error_preserves_exhaustive_error_order(monkeypatch):
    semantic._reset_mp3d_query_diagnostics()
    monkeypatch.setattr(
        semantic, "_query_mp3d_complete_face_universe_buffer_bounded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("source digest changed")))
    monkeypatch.setattr(
        semantic, "_query_mp3d_complete_face_universe_buffer_exhaustive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("MP3D semantic PLY contains non-triangle faces")))

    with pytest.raises(ValueError, match="non-triangle faces"):
        semantic._query_mp3d_complete_face_universe_buffer(
            None, 0, "unused", np.zeros((1, 3)), [1], {1: "chair"},
            "a" * 64)


def test_load_mp3d_semantic_index_maps_face_instances_and_categories(tmp_path):
    scene = _write_mp3d_semantics(tmp_path)

    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=100)

    assert index.id_to_cat == {1: "chair", 2: "table"}
    chair = index.instance_points(1)
    table = index.instance_points(2)
    assert chair.shape == (1, 3)
    assert table.shape == (1, 3)
    # MP3D Z-up -> Habitat Y-up: (x, y, z) -> (x, z, -y).
    assert np.allclose(chair[0], [1.0 / 3.0, 0.0, -1.0 / 3.0])
    assert np.allclose(table[0], [1.0 / 3.0, 1.0, -1.0 / 3.0])
    assigned = index.assign(np.vstack([chair, table]), tol=1e-4)
    assert assigned.tolist() == [1, 2]


def test_mp3d_full_instance_triangles_are_canonical_and_hash_bound(tmp_path):
    """Catches sampled centroids being reused as the full-geometry authority."""
    scene = _write_mp3d_semantics(tmp_path)

    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    chair = index.instance_triangles(1)
    table = index.instance_triangles(2)

    assert chair.shape == (1, 3, 3)
    assert table.shape == (1, 3, 3)
    # Both face order and vertex order are canonical. MP3D Z-up coordinates
    # have already been transformed to Habitat world XYZ.
    np.testing.assert_array_equal(chair, np.array([[  # x, y, z
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]], dtype=np.float32))
    np.testing.assert_array_equal(table, np.array([[
        [0.0, 1.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]], dtype=np.float32))
    assert len(index.instance_triangles_sha256(1)) == 64
    assert index.instance_triangles_sha256(1) != \
        index.instance_triangles_sha256(2)


def test_mp3d_full_instance_triangles_are_not_capped_by_surface_sampling(
        tmp_path):
    """Catches rebuilding the authority from the capped centroid index."""
    scene = _write_many_face_mp3d_semantics(tmp_path, face_count=20)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    assert index.instance_points(1).shape == (16, 3)
    assert index.instance_triangles(1).shape == (20, 3, 3)


@pytest.mark.parametrize("bad_index", [-1, 60])
def test_mp3d_full_triangle_query_rejects_invalid_vertex_indices(
        tmp_path, bad_index):
    """Catches NumPy negative wrapping and out-of-range face references."""
    scene = _write_many_face_mp3d_semantics(
        tmp_path, bad_face_index=bad_index)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    with pytest.raises(ValueError, match="vertex index"):
        index.instance_triangles(1)


@pytest.mark.parametrize("bad_coordinate", [float("nan"), float("inf")])
def test_mp3d_full_triangle_query_rejects_nonfinite_referenced_vertices(
        tmp_path, bad_coordinate):
    """Catches NaN/Inf target geometry entering identity and geometry hashes."""
    scene = _write_many_face_mp3d_semantics(
        tmp_path, bad_vertex_coordinate=bad_coordinate)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    with pytest.raises(ValueError, match="finite vertex coordinates"):
        index.instance_triangles(1)


def test_mp3d_full_instance_triangle_hash_ignores_query_and_face_order(
        tmp_path):
    """Catches nondeterministic hashes tied to memmap/query traversal order."""
    scene = _write_mp3d_semantics(tmp_path, face_object_ids=(0, 0))
    first = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache-a", sample_count=1)
    first_hash = first.instance_triangles_sha256(1)
    # Reverse the two source face records without changing either instance's
    # geometry. A per-instance authority must not inherit global PLY order.
    reordered_dir = tmp_path / "reordered"
    reordered_dir.mkdir()
    reordered_scene = _write_mp3d_semantics(
        reordered_dir, face_object_ids=(0, 0))
    ply = PlyData.read(
        reordered_dir / "AAA_semantic.ply", mmap=False)
    reversed_faces = np.asarray(ply["face"].data)[::-1].copy()
    for face in reversed_faces:
        face["vertex_indices"] = face["vertex_indices"][::-1]
    PlyData([
        PlyElement.describe(np.asarray(ply["vertex"].data), "vertex"),
        PlyElement.describe(reversed_faces, "face"),
    ], text=False).write(reordered_dir / "AAA_semantic.ply")
    second = semantic.load_mp3d_semantic_index(
        reordered_scene, cache_dir=tmp_path / "cache-b", sample_count=100)

    # Repeated and reordered queries must not change either geometry or hash.
    second.instance_triangles(1)
    assert second.instance_triangles_sha256(1) == first_hash
    np.testing.assert_array_equal(
        second.instance_triangles(1), first.instance_triangles(1))


def test_mp3d_full_instance_triangle_query_fails_closed_for_unknown_id(
        tmp_path):
    """Catches category fallback when face-level exact identity is absent."""
    scene = _write_mp3d_semantics(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=100)

    with pytest.raises(KeyError, match="instance 99"):
        index.instance_triangles(99)
    with pytest.raises(KeyError, match="instance 99"):
        index.instance_triangles_sha256(99)


def test_mp3d_contact_identity_is_confirmed_against_full_faces(tmp_path):
    """Catches a sampled witness being accepted without exact face support."""
    scene = _write_mp3d_semantics(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    result = index.confirm_contact_instance(
        1, [0.25, 0.0, -0.25], candidate_instance_ids=[1, 2])

    assert result == {
        "authority": "mp3d_full_face_universe",
        "schema": "mp3d-contact-face-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 1,
        "category": "chair",
        "streaming_instance_faces_sha256":
            result["streaming_instance_faces_sha256"],
        "contact_face_distance_m": 0.0,
        "runner_up_face_distance_m": 1.0,
        "face_distance_margin_m": 1.0,
        "global_query_protocol":
            "mp3d-complete-face-instance-universe.v1",
        "global_winner_instance_id": 1,
        "global_runner_up_instance_id": 2,
        "semantic_ply_sha256": hashlib.sha256(
            (tmp_path / "AAA_semantic.ply").read_bytes()).hexdigest(),
        "global_universe_sha256": semantic.complete_face_universe_sha256(
            global_query_protocol=
                "mp3d-complete-face-instance-universe.v1",
            semantic_ply_sha256=hashlib.sha256(
                (tmp_path / "AAA_semantic.ply").read_bytes()).hexdigest()),
    }


def test_mp3d_contact_identity_withholds_near_tied_full_faces(tmp_path):
    """Catches deterministic instance-id tie breaking replacing ambiguity."""
    scene = _write_mp3d_semantics(tmp_path)
    ply_path = tmp_path / "AAA_semantic.ply"
    ply = PlyData.read(ply_path, mmap=False)
    vertices = np.asarray(ply["vertex"].data).copy()
    vertices[3:6]["z"] = 0.01
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(np.asarray(ply["face"].data), "face"),
    ], text=False).write(ply_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    result = index.confirm_contact_instance(
        1, [0.25, 0.005, -0.25], candidate_instance_ids=[1, 2])

    assert result["confirmed"] is False
    assert result["reason"] == "contact_face_identity_near_tie"
    assert result["instance_id"] is None
    assert result["contact_face_distance_m"] == pytest.approx(0.005)
    assert result["runner_up_face_distance_m"] == pytest.approx(0.005)


def test_mp3d_contact_identity_rejects_closer_hidden_global_instance(tmp_path):
    """Catches resolving winner/runner-up only inside the visible inventory."""
    scene = _write_mp3d_semantics(tmp_path)
    ply_path = tmp_path / "AAA_semantic.ply"
    ply = PlyData.read(ply_path, mmap=False)
    vertices = np.asarray(ply["vertex"].data).copy()
    vertices[0:3]["z"] = 0.05
    vertices[3:6]["z"] = 0.0
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(np.asarray(ply["face"].data), "face"),
    ], text=False).write(ply_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    result = index.confirm_contact_instance(
        1, [0.25, 0.0, -0.25], candidate_instance_ids=[1])

    assert result["confirmed"] is False
    assert result["reason"] == "global_contact_winner_mismatch"
    assert result["global_winner_instance_id"] == 2
    assert result["global_runner_up_instance_id"] == 1
    assert result["global_query_protocol"] == \
        "mp3d-complete-face-instance-universe.v1"
    assert "global_face_count" not in result
    assert "global_instance_count" not in result


def test_mp3d_contact_identity_keeps_unrelated_unlabelled_face_in_universe(
        tmp_path):
    """An unrelated unlabelled face must not invalidate a labelled witness."""
    scene = _write_mp3d_semantics(tmp_path)
    (tmp_path / "AAA.house").write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
        "O  1 0 -1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
    )
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    result = index.confirm_contact_instance(
        1, [0.25, 0.0, -0.25], candidate_instance_ids=[1])

    assert result["confirmed"] is True
    assert result["instance_id"] == 1
    assert result["global_runner_up_instance_id"] == 2


def test_mp3d_contact_identity_rejects_requested_unlabelled_face(tmp_path):
    """A face without a public category can never become an A3 answer."""
    scene = _write_mp3d_semantics(tmp_path)
    (tmp_path / "AAA.house").write_text(
        "ASCII 1.1\n"
        "C  0  1 chair  3 chair  0 0 0 0 0\n"
        "O  0 0 0  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
        "O  1 0 -1  0 0 0  1 0 0  0 1 0  1 1 1  0 0 0 0\n"
    )
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)

    with pytest.raises(
            ValueError, match="requested instance has no house category"):
        index.confirm_contact_instance(
            2, [0.25, 1.0, -0.25], candidate_instance_ids=[2])


def test_mp3d_contact_identity_batches_seven_points_in_one_universe_pass(
        tmp_path, monkeypatch):
    scene = _write_mp3d_semantics(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    expected = index.confirm_contact_instance(
        1, [0.25, 0.0, -0.25], candidate_instance_ids=[1])
    calls = []
    original = semantic._binary_ply_layout_from_buffer

    def counted(buffer, path):
        calls.append(path)
        return original(buffer, path)

    monkeypatch.setattr(
        semantic, "_binary_ply_layout_from_buffer", counted)
    requests = [(1, [0.25, 0.0, -0.25]) for _index in range(7)]

    results = index.confirm_contact_instances(requests)

    assert len(calls) == 1
    assert len(results) == 7
    assert all(value["confirmed"] is True for value in results)
    assert all(value["global_winner_instance_id"] == 1 for value in results)
    assert all(value["contact_face_distance_m"] == pytest.approx(0.0)
               for value in results)
    assert all(value == expected for value in results)
    assert len({value["global_universe_sha256"] for value in results}) == 1
    assert all(value["semantic_ply_sha256"] ==
               hashlib.sha256((tmp_path / "AAA_semantic.ply").read_bytes()).hexdigest()
               for value in results)


def test_mp3d_contact_batch_rejects_ply_changed_after_index_load(tmp_path):
    scene = _write_mp3d_semantics(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    ply_path = tmp_path / "AAA_semantic.ply"
    header_size, _vertices, _faces = semantic._binary_ply_layout(ply_path)
    payload = bytearray(ply_path.read_bytes())
    payload[header_size] ^= 1
    ply_path.write_bytes(payload)

    with pytest.raises(ValueError, match="source digest"):
        index.confirm_contact_instances([
            (1, [0.25, 0.0, -0.25]) for _index in range(7)])


@pytest.mark.parametrize(
    ("exception_type", "exception_args"), [
        (KeyboardInterrupt, ("cancelled",)),
        (SystemExit, (17,)),
    ])
def test_mp3d_pinned_query_closes_before_reraising_base_exception(
        tmp_path, monkeypatch, exception_type, exception_args):
    scene = _write_mp3d_semantics(tmp_path)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    original_mmap = semantic.mmap.mmap
    mappings = []

    def tracked_mmap(*args, **kwargs):
        value = original_mmap(*args, **kwargs)
        mappings.append(value)
        return value

    def interrupt(*_args, **_kwargs):
        raise exception_type(*exception_args)

    monkeypatch.setattr(semantic.mmap, "mmap", tracked_mmap)
    monkeypatch.setattr(semantic, "_mp3d_source_triangles", interrupt)

    with pytest.raises(exception_type) as caught:
        index.confirm_contact_instance(
            1, [0.25, 0.0, -0.25], candidate_instance_ids=[1])

    assert caught.value.args == exception_args
    assert len(mappings) == 1
    assert mappings[0].closed is True


def test_mp3d_contact_batch_does_not_cache_or_accumulate_full_triangles(
        tmp_path, monkeypatch):
    scene = _write_many_face_mp3d_semantics(tmp_path, face_count=200)
    index = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=1)
    monkeypatch.setattr(
        semantic, "_canonical_triangles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("batch proof must not retain full triangles")))

    first = index.confirm_contact_instances([
        (1, [0.1, 0.0, -0.1]) for _index in range(7)])
    first_hash = index._streaming_instance_face_hashes[1]
    second = index.confirm_contact_instances([
        (1, [0.1, 0.0, -0.1]) for _index in range(7)])

    assert index._instance_triangles == {}
    assert first[0]["reason"] == "global_runner_evidence_missing"
    assert second[0]["reason"] == "global_runner_evidence_missing"
    assert index._streaming_instance_face_hashes[1] == first_hash


def test_mp3d_semantic_cache_roundtrip(tmp_path, monkeypatch):
    scene = _write_mp3d_semantics(tmp_path)
    first = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=100)
    monkeypatch.setattr(
        semantic,
        "_decode_mp3d_labelled_points",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an unchanged input must use the cache")),
    )

    cached = semantic.load_mp3d_semantic_index(
        scene, cache_dir=tmp_path / "cache", sample_count=100)

    assert cached.id_to_cat == first.id_to_cat
    assert np.array_equal(cached.instance_points(1), first.instance_points(1))
    caches = list((tmp_path / "cache").glob("mp3d-AAA.*.sem.npz"))
    assert len(caches) == 1
    with np.load(caches[0], allow_pickle=False) as values:
        assert str(values["cache_metadata_json"].item()).startswith(
            '{"backend":"mp3d_ply","cache_schema":"semantic-surface.v3"')


def test_mp3d_semantic_cache_key_tracks_source_and_sampling(
        tmp_path, monkeypatch):
    scene = _write_mp3d_semantics(tmp_path)
    cache_dir = tmp_path / "cache"
    first = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)
    assert first.instance_points(1).shape == (1, 3)

    calls = []

    def replacement(_path, sample_count):
        calls.append(sample_count)
        return (
            np.array([[9.0, 8.0, 7.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
        )

    monkeypatch.setattr(
        semantic, "_decode_mp3d_labelled_points", replacement)
    semantic_ply = tmp_path / "AAA_semantic.ply"
    stat = semantic_ply.stat()
    os.utime(
        semantic_ply,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )

    source_changed = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)
    sampling_changed = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=101)

    assert calls == [100, 101]
    np.testing.assert_array_equal(
        source_changed.instance_points(1), [[9.0, 8.0, 7.0]])
    np.testing.assert_array_equal(
        sampling_changed.instance_points(1), [[9.0, 8.0, 7.0]])
    assert len(list(cache_dir.glob("mp3d-AAA.*.sem.npz"))) == 3


def test_mp3d_semantic_cache_key_tracks_content_with_same_size_and_mtime(
        tmp_path, monkeypatch):
    scene = _write_mp3d_semantics(tmp_path)
    cache_dir = tmp_path / "cache"
    semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)
    semantic_ply = tmp_path / "AAA_semantic.ply"
    original_stat = semantic_ply.stat()
    payload = bytearray(semantic_ply.read_bytes())
    payload[-1] ^= 1
    semantic_ply.write_bytes(payload)
    os.utime(
        semantic_ply,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    calls = []

    def replacement(_path, sample_count):
        calls.append(sample_count)
        return (
            np.array([[9.0, 8.0, 7.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
        )

    monkeypatch.setattr(
        semantic, "_decode_mp3d_labelled_points", replacement)

    changed = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)

    assert calls == [100]
    np.testing.assert_array_equal(
        changed.instance_points(1), [[9.0, 8.0, 7.0]])


def test_mp3d_semantic_cache_rebuilds_on_corrupt_or_mismatched_payload(
        tmp_path, monkeypatch):
    """The embedded-metadata check is an independent defence: a file at the
    correct digest path whose payload is corrupt or carries foreign
    provenance must be a cache miss, never a crash or a silent hit."""
    scene = _write_mp3d_semantics(tmp_path)
    cache_dir = tmp_path / "cache"
    first = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)
    cache_path = next(cache_dir.glob("mp3d-AAA.*.sem.npz"))

    calls = []

    def replacement(_path, sample_count):
        calls.append(sample_count)
        return (
            np.array([[9.0, 8.0, 7.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
        )

    monkeypatch.setattr(
        semantic, "_decode_mp3d_labelled_points", replacement)

    # Truncated / garbage bytes at the correct digest path.
    cache_path.write_bytes(b"PK\x03\x04 not a real npz")
    corrupt = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)
    np.testing.assert_array_equal(
        corrupt.instance_points(1), [[9.0, 8.0, 7.0]])
    assert calls == [100]

    # A well-formed NPZ whose embedded provenance does not match.
    with np.load(cache_path, allow_pickle=False) as values:
        points = np.asarray(values["points"]).copy()
        instances = np.asarray(values["inst"]).copy()
    semantic._write_cache_atomic(
        cache_path, points=points, instances=instances,
        metadata='{"backend":"someone_else"}')
    mismatched = semantic.load_mp3d_semantic_index(
        scene, cache_dir=cache_dir, sample_count=100)
    np.testing.assert_array_equal(
        mismatched.instance_points(1), [[9.0, 8.0, 7.0]])
    assert calls == [100, 100]
    assert first.instance_points(1).shape == (1, 3)


def test_sim_semantic_dispatch_uses_mp3d_adapter(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        semantic, "load_mp3d_semantic_index",
        lambda path, **kwargs: calls.append((path, kwargs)) or sentinel)

    result = sim.load_scene_semantic_index(
        "/data/mp3d/AAA/AAA.glb", "mp3d_ply")

    assert result is sentinel
    assert calls == [(
        "/data/mp3d/AAA/AAA.glb", {"query_workers": 1})]
