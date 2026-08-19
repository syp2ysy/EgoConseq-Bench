import copy
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

from pipeline import (
    b1k_geometry, b1k_semantic, consensus, dataset_contracts, record,
    semantic,
)
from pipeline import config
from tests._synthetic import LEVEL_FLOOR_FIT, _outcome


def test_b1k_python_has_an_environment_override(tmp_path):
    expected = tmp_path / "behavior" / "bin" / "python"
    environment = dict(os.environ)
    environment["EGOCONSEQ_B1K_PYTHON"] = str(expected)
    root = Path(__file__).resolve().parents[1]

    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "from pipeline import b1k_source_builder; "
                "print(b1k_source_builder.BEHAVIOR_PYTHON)"
            ),
        ],
        cwd=root,
        env=environment,
        text=True,
    ).strip()

    assert output == str(expected)


def test_b1k_python_defaults_to_current_interpreter():
    environment = dict(os.environ)
    environment.pop("EGOCONSEQ_B1K_PYTHON", None)
    root = Path(__file__).resolve().parents[1]

    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "from pipeline import b1k_source_builder; "
                "print(b1k_source_builder.BEHAVIOR_PYTHON)"
            ),
        ],
        cwd=root,
        env=environment,
        text=True,
    ).strip()

    assert output == sys.executable


def _b1k_binding(source_sha256: str):
    return dataset_contracts.AuthorityBinding(
        source_dataset="b1k",
        identity_schema="b1k-contact-triangle-identity.v1",
        authoritative_source_role="scene_authority",
        source_sha256=source_sha256,
    )


def test_degenerate_triangle_distance_is_finite_without_runtime_warning():
    triangle = np.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]], dtype=np.float64)
    point = np.array([[0.5, 1.0, 0.0]], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        batched = b1k_semantic._point_triangle_distances_m(point, triangle)

    assert batched.tolist() == [1.0]


def test_b1k_world_mapping_preserves_frozen_heading_handedness():
    """Catches an axis permutation or yaw sign that mirrors B1K motion."""
    og_points = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])

    pbench = b1k_geometry.og_to_pbench_xyz(og_points)

    assert np.array_equal(pbench, np.array([
        [1.0, 3.0, -2.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ]))
    assert np.array_equal(
        b1k_geometry.pbench_to_og_xyz(pbench), og_points)
    assert b1k_geometry.og_yaw_to_pbench_yaw_rad(0.0) == 0.0
    assert b1k_geometry.og_yaw_to_pbench_yaw_rad(np.pi / 2) == np.pi / 2


def _rectangle_triangles(x0, x1, z0, z1, *, y):
    return np.array([
        [[x0, y, z0], [x1, y, z0], [x1, y, z1]],
        [[x0, y, z0], [x1, y, z1], [x0, y, z1]],
    ], dtype=np.float64)


def _geometry(*obstacles):
    floor = b1k_geometry.TriangleComponent(
        "/floor", _rectangle_triangles(-2.0, 2.0, -2.0, 2.0, y=0.0))
    return b1k_geometry.B1KGeometryAuthority(
        floor_components=[floor],
        collision_components=list(obstacles),
    )


def test_b1k_geometry_builds_each_frozen_radius_configuration_space():
    """Catches one shared C-space silently erasing the public body axis."""
    obstacle = b1k_geometry.TriangleComponent(
        "/box", _rectangle_triangles(-0.1, 0.1, -0.1, 0.1, y=0.2))
    authority = _geometry(obstacle)

    observed = {
        radius: authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=radius)
        .query_pose((0.31, 0.0, 0.0))
        for radius in config.RADII_M
    }

    assert tuple(authority.radii_m) == config.RADII_M
    assert [observed[radius].navigable for radius in config.RADII_M] == [
        True, True, False]
    assert [observed[radius].clearance_m for radius in config.RADII_M] == \
        pytest.approx([0.06, 0.01, -0.04])
    assert observed[0.25].geometry_source == "b1k_geometry"
    assert observed[0.25].obstacle_index == 0


def test_b1k_geometry_indexes_components_and_keeps_source_order_on_ties():
    """The hot nearest query must be indexed without changing tie breaks."""
    left = b1k_geometry.TriangleComponent(
        "/left", _rectangle_triangles(-0.6, -0.4, -0.1, 0.1, y=0.2))
    right = b1k_geometry.TriangleComponent(
        "/right", _rectangle_triangles(0.4, 0.6, -0.1, 0.1, y=0.2))
    authority = _geometry(left, right)

    level = authority._levels[0]
    query = authority.bind(
        [0.0, 0.0, 0.0], 0.0, radius_m=0.15
    ).query_pose((0.0, 0.0, 0.0))

    assert level.obstacle_tree is not None
    assert level.obstacle_tree_component_indices == (0, 1)
    assert query.navigable is True
    assert query.clearance_m == pytest.approx(0.25)
    assert query.obstacle_index == 0


def test_b1k_obstacle_footprint_skips_triangles_outside_body_band(
        monkeypatch):
    """Do not enter the Python/Shapely hot loop for impossible heights."""
    component = b1k_geometry.TriangleComponent("/mixed", np.array([
        [[0.0, 0.20, 0.0], [1.0, 0.20, 0.0], [0.0, 0.20, 1.0]],
        [[0.0, 1.00, 0.0], [1.0, 1.00, 0.0], [0.0, 1.00, 1.0]],
        [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, -1.0, 1.0]],
    ], dtype=np.float64))
    original = b1k_geometry._clipped_triangle_projection
    calls = []

    def counted(triangle, low, high):
        calls.append(np.asarray(triangle))
        return original(triangle, low, high)

    monkeypatch.setattr(
        b1k_geometry, "_clipped_triangle_projection", counted)
    footprint = b1k_geometry._obstacle_footprint(component, 0.0)

    assert len(calls) == 1
    assert footprint.area == pytest.approx(0.5)


def test_b1k_query_many_batches_geos_without_changing_scalar_results(
        monkeypatch):
    """Path marching must not perform one Python/GEOS scan per 2 cm point."""
    left = b1k_geometry.TriangleComponent(
        "/left", _rectangle_triangles(-0.6, -0.4, -0.1, 0.1, y=0.2))
    right = b1k_geometry.TriangleComponent(
        "/right", _rectangle_triangles(0.4, 0.6, -0.1, 0.1, y=0.2))
    nav = _geometry(left, right).bind(
        [0.0, 0.0, 0.0], 0.0, radius_m=0.15)
    poses = [
        (-1.9, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.9, 0.0, 0.0),
    ]
    expected = [nav.query_pose(pose) for pose in poses]

    monkeypatch.setattr(
        b1k_geometry.B1KGeometryAuthority, "_nearest_component",
        lambda *_args, **_kwargs: pytest.fail(
            "query_many fell back to scalar nearest-component queries"))
    observed = nav.query_many(poses)

    assert observed == expected


def test_b1k_geometry_preserves_a_real_component_concavity():
    """Catches replacing a component union with its convex hull."""
    concave = b1k_geometry.TriangleComponent(
        "/l-shaped-counter",
        np.concatenate([
            _rectangle_triangles(-1.0, -0.2, -1.0, 1.0, y=0.2),
            _rectangle_triangles(-0.2, 1.0, -1.0, -0.2, y=0.2),
        ]),
    )
    nav = _geometry(concave).bind(
        [0.0, 0.0, 0.0], 0.0, radius_m=0.15)

    assert nav.is_navigable((0.5, -0.5, 0.0)) is True
    assert nav.is_navigable((-0.5, 0.5, 0.0)) is False


def test_b1k_geometry_attributes_a_disc_beyond_loaded_floor_support():
    """Catches treating absent floor support as a lateral collision."""
    floor = b1k_geometry.TriangleComponent(
        "/small-floor", _rectangle_triangles(-1.0, 1.0, -1.0, 1.0, y=0.0))
    authority = b1k_geometry.B1KGeometryAuthority(
        floor_components=[floor], collision_components=[])
    nav = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.2)

    query = nav.query_pose((0.9, 0.0, 0.0))

    assert query.navigable is False
    assert query.clearance_m == 0.0
    assert query.obstacle_index is None
    assert query.geometry_source == "unsupported_floor"


def test_b1k_geometry_clips_obstacles_to_band_above_loaded_floor():
    """Catches interpreting the 0.05--0.30 m band as absolute world Y."""
    floor = b1k_geometry.TriangleComponent(
        "/raised-floor", _rectangle_triangles(
            -2.0, 2.0, -2.0, 2.0, y=1.0))
    in_band = b1k_geometry.TriangleComponent(
        "/raised-box", _rectangle_triangles(
            -0.1, 0.1, -0.1, 0.1, y=1.2))
    overhead = b1k_geometry.TriangleComponent(
        "/overhead", _rectangle_triangles(
            -0.1, 0.1, -0.1, 0.1, y=1.31))

    blocked = b1k_geometry.B1KGeometryAuthority(
        floor_components=[floor], collision_components=[in_band])
    clear = b1k_geometry.B1KGeometryAuthority(
        floor_components=[floor], collision_components=[overhead])

    assert blocked.bind(
        [0.0, 1.0, 0.0], 0.0, radius_m=0.2
    ).is_navigable((0.25, 0.0, 0.0)) is False
    assert clear.bind(
        [0.0, 1.0, 0.0], 0.0, radius_m=0.2
    ).is_navigable((0.0, 0.0, 0.0)) is True


def test_b1k_geometry_keeps_two_floor_levels_physically_independent():
    """Catches median-height bands mixing obstacles between floor levels."""
    lower_floor = b1k_geometry.TriangleComponent(
        "/floor-lower", _rectangle_triangles(
            -2.0, 2.0, -2.0, 2.0, y=0.0))
    upper_floor = b1k_geometry.TriangleComponent(
        "/floor-upper", _rectangle_triangles(
            -2.0, 2.0, -2.0, 2.0, y=2.0))
    lower_obstacle = b1k_geometry.TriangleComponent(
        "/box-lower", _rectangle_triangles(
            -0.6, -0.4, -0.1, 0.1, y=0.2))
    upper_obstacle = b1k_geometry.TriangleComponent(
        "/box-upper", _rectangle_triangles(
            0.4, 0.6, -0.1, 0.1, y=2.2))
    authority = b1k_geometry.B1KGeometryAuthority(
        floor_components=[upper_floor, lower_floor],
        collision_components=[upper_obstacle, lower_obstacle])
    lower = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.2)
    upper = authority.bind([0.0, 2.0, 0.0], 0.0, radius_m=0.2)

    assert lower.is_navigable((-0.5, 0.0, 0.0)) is False
    assert lower.is_navigable((0.5, 0.0, 0.0)) is True
    assert upper.is_navigable((-0.5, 0.0, 0.0)) is True
    assert upper.is_navigable((0.5, 0.0, 0.0)) is False
    between = authority.bind([0.0, 1.0, 0.0], 0.0, radius_m=0.2)
    assert between.query_pose((0.0, 0.0, 0.0)).geometry_source == \
        "unsupported_floor"

    rng = np.random.default_rng(23)
    samples = [
        authority.sample_position(rng, radius_m=0.2) for _ in range(40)
    ]
    assert {float(value[1]) for value in samples} == {0.0, 2.0}
    for position in samples:
        assert authority.bind(
            position, 0.0, radius_m=0.2
        ).is_navigable((0.0, 0.0, 0.0)) is True


def test_b1k_geometry_splits_stacked_surfaces_inside_one_floor_component():
    """Catches collapsing a multi-surface collision component to its median Y."""
    stacked_floor = b1k_geometry.TriangleComponent(
        "/stacked-floor-component",
        np.concatenate([
            _rectangle_triangles(-2.0, 2.0, -2.0, 2.0, y=0.0),
            _rectangle_triangles(-2.0, 2.0, -2.0, 2.0, y=2.0),
        ]))
    lower_obstacle = b1k_geometry.TriangleComponent(
        "/box-lower", _rectangle_triangles(
            -0.6, -0.4, -0.1, 0.1, y=0.2))
    upper_obstacle = b1k_geometry.TriangleComponent(
        "/box-upper", _rectangle_triangles(
            0.4, 0.6, -0.1, 0.1, y=2.2))

    authority = b1k_geometry.B1KGeometryAuthority(
        floor_components=[stacked_floor],
        collision_components=[lower_obstacle, upper_obstacle])
    lower = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=0.2)
    upper = authority.bind([0.0, 2.0, 0.0], 0.0, radius_m=0.2)

    assert lower.is_navigable((-0.5, 0.0, 0.0)) is False
    assert lower.is_navigable((0.5, 0.0, 0.0)) is True
    assert upper.is_navigable((-0.5, 0.0, 0.0)) is True
    assert upper.is_navigable((0.5, 0.0, 0.0)) is False


def test_b1k_geometry_samples_each_radius_from_derived_free_space():
    """Catches falling back to an unrelated runtime floor-traversability map."""
    obstacle = b1k_geometry.TriangleComponent(
        "/box", _rectangle_triangles(-0.3, 0.3, -0.3, 0.3, y=0.2))
    authority = _geometry(obstacle)
    rng = np.random.default_rng(11)

    for radius in config.RADII_M:
        nav = authority.bind([0.0, 0.0, 0.0], 0.0, radius_m=radius)
        for _ in range(5):
            position = authority.sample_position(rng, radius_m=radius)
            rebound = authority.bind(position, 0.0, radius_m=radius)
            assert nav.query_pose((position[0], -position[2], 0.0)).navigable
            assert rebound.query_pose((0.0, 0.0, 0.0)).navigable


def test_b1k_geometry_samples_directly_from_minimum_clearance_space():
    """Catches spending the pose budget on points the hard gate rejects."""
    obstacle = b1k_geometry.TriangleComponent(
        "/box", _rectangle_triangles(-0.3, 0.3, -0.3, 0.3, y=0.2))
    authority = _geometry(obstacle)
    rng = np.random.default_rng(19)

    for _ in range(40):
        position = authority.sample_position(
            rng, radius_m=0.25, minimum_clearance_m=0.30)
        clearance = authority.bind(
            position, 0.0, radius_m=0.25
        ).clearance((0.0, 0.0, 0.0))
        assert clearance >= 0.30 - 1e-9


def test_b1k_pose_sampling_does_not_rebuffer_static_scene_geometry(
        monkeypatch):
    """The clearance predicate is indexed; a pose draw must not rebuild it."""
    from shapely.geometry.base import BaseGeometry

    obstacle = b1k_geometry.TriangleComponent(
        "/box", _rectangle_triangles(-0.3, 0.3, -0.3, 0.3, y=0.2))
    authority = _geometry(obstacle)

    def reject_rebuffer(_geometry, *_args, **_kwargs):
        raise AssertionError("pose sampling rebuilt static obstacle geometry")

    monkeypatch.setattr(BaseGeometry, "buffer", reject_rebuffer)
    position = authority.sample_position(
        np.random.default_rng(19), radius_m=0.25,
        minimum_clearance_m=0.30)

    assert authority.bind(
        position, 0.0, radius_m=0.25
    ).clearance((0.0, 0.0, 0.0)) >= 0.30 - 1e-9


def test_b1k_geometry_reports_closest_component_contact_evidence():
    """Catches contact output coming from a C-space boundary, not the object."""
    obstacle = b1k_geometry.TriangleComponent(
        "/box", _rectangle_triangles(-0.1, 0.1, -0.1, 0.1, y=0.2))
    nav = _geometry(obstacle).bind(
        [0.0, 0.0, 0.0], 0.0, radius_m=0.2)

    hit = nav.closest_obstacle((0.25, 0.0, 0.0))

    assert hit["world_point"] == pytest.approx([0.1, 0.175, 0.0])
    assert hit["world_normal"] == pytest.approx([1.0, 0.0, 0.0])
    assert hit["distance_m"] == pytest.approx(0.15)
    assert hit["obstacle_index"] == 0
    assert hit["obstacle_identity"] == "/box"
    assert hit["surface_protocol"] == \
        "b1k-closest-collision-footprint.v1"


def _runtime_instances():
    return [
        b1k_semantic.RuntimeInstanceSpec(
            prim_identity="/World/z-chair",
            raw_category="chair",
            model="chair_model",
            synset_label="chair.n.01",
            triangles=_rectangle_triangles(0.4, 0.6, -0.1, 0.1, y=0.2),
        ),
        b1k_semantic.RuntimeInstanceSpec(
            prim_identity="/World/a-table",
            raw_category="breakfast_table",
            model="table_model",
            synset_label="table.n.02",
            triangles=_rectangle_triangles(-0.1, 0.1, -0.1, 0.1, y=0.2),
        ),
    ]


def _authority_atom(instances=None):
    floor = [b1k_geometry.TriangleComponent(
        "/floor", _rectangle_triangles(-2.0, 2.0, -2.0, 2.0, y=0.0))]
    collision = [b1k_geometry.TriangleComponent(
        "/collision", _rectangle_triangles(-0.1, 0.1, -0.1, 0.1, y=0.2))]
    value = b1k_semantic.derive_scene_authority_atom(
        floor_components=floor,
        collision_components=collision,
        instances=_runtime_instances() if instances is None else instances,
    )
    return floor, collision, value


def _semantic_authority(instances=None):
    values = _runtime_instances() if instances is None else instances
    floor, collision, _atom = _authority_atom(values)
    _geometry, index, atom = b1k_semantic.build_b1k_authorities(
        floor_components=floor,
        collision_components=collision,
        instances=values)
    return index, atom


def test_b1k_semantic_ids_are_stable_and_public_labels_are_synsets():
    """Catches runtime enumeration order or raw labels leaking publicly."""
    instances = _runtime_instances()
    index, _atom = _semantic_authority(instances[::-1])

    assert index.id_to_cat == {1: "table.n.02", 2: "chair.n.01"}
    assert index.instance_id_for_prim("/World/a-table") == 1
    assert index.instance_id_for_prim("/World/z-chair") == 2
    assert index.public_instances() == [
        {"instance_id": 1, "category": "table.n.02"},
        {"instance_id": 2, "category": "chair.n.01"},
    ]
    assert "breakfast_table" not in repr(index.public_instances())
    assert "table_model" not in repr(index.public_instances())
    assert "/World/a-table" not in repr(index.public_instances())


def test_b1k_instance_points_are_canonical_surface_samples_and_defensive():
    """Catches bbox proxies, unstable ordering, or mutable authority state."""
    index, _atom = _semantic_authority()
    triangles = index.instance_triangles(1)
    expected = np.unique(np.concatenate([
        triangles.reshape(-1, 3),
        triangles.mean(axis=1),
    ], axis=0), axis=0)

    first = index.instance_points(1)
    second = index.instance_points(1)

    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    first[:] = 999.0
    assert np.array_equal(index.instance_points(1), expected)
    with pytest.raises(KeyError, match="unknown"):
        index.instance_points(999)


def test_b1k_category_layers_keep_machine_identity_private_and_stable():
    """Catches raw/display labels entering identity atoms or public instances."""
    index, atom = _semantic_authority()
    public_before = copy.deepcopy(index.public_instances())
    digest_before = copy.deepcopy(atom)

    assert index.category_layers(1) == {
        "machine": "table.n.02",
        "raw": "breakfast_table",
        "display": "breakfast table",
    }
    assert set(index.category_layers(1)) == {"machine", "raw", "display"}
    assert index.public_instances() == public_before
    assert atom == digest_before
    with pytest.raises(KeyError, match="unknown"):
        index.category_layers(999)


def test_b1k_scene_authority_is_order_invariant_and_detects_substitution():
    """Catches a digest that ignores triangle content or runtime metadata."""
    instances = _runtime_instances()
    floor, collision, first = _authority_atom(instances)
    second = b1k_semantic.derive_scene_authority_atom(
        floor_components=floor[::-1],
        collision_components=collision[::-1],
        instances=instances[::-1],
    )
    changed = list(instances)
    changed[0] = b1k_semantic.RuntimeInstanceSpec(
        prim_identity=changed[0].prim_identity,
        raw_category=changed[0].raw_category,
        model=changed[0].model,
        synset_label=changed[0].synset_label,
        triangles=changed[0].triangles + np.array([1.0, 0.0, 0.0]),
    )
    _floor, _collision, substituted = _authority_atom(changed)

    assert first == second
    assert first["schema"] == "b1k-derived-scene-authority.v1"
    assert first["frame"] == "pbench_world_xyz"
    assert first["sha256"] == record.canonical_atom_sha256({
        key: value for key, value in first.items() if key != "sha256"
    })
    assert substituted["sha256"] != first["sha256"]


def test_b1k_authority_factory_binds_one_digest_to_both_authorities():
    """Catches geometry and semantics authenticating different scene states."""
    floor, collision, expected = _authority_atom()

    geometry, semantic, atom = b1k_semantic.build_b1k_authorities(
        floor_components=floor,
        collision_components=collision,
        instances=_runtime_instances(),
    )

    assert atom == expected
    assert geometry.scene_authority_sha256 == expected["sha256"]
    assert semantic.scene_authority_sha256 == expected["sha256"]


def test_b1k_direct_constructors_reject_a_stale_scene_digest():
    """Catches substituted triangles being vouched for by an old scene hash."""
    instances = _runtime_instances()
    floor, collision, original = _authority_atom(instances)
    changed = list(instances)
    changed[0] = b1k_semantic.RuntimeInstanceSpec(
        prim_identity=changed[0].prim_identity,
        raw_category=changed[0].raw_category,
        model=changed[0].model,
        synset_label=changed[0].synset_label,
        triangles=changed[0].triangles + np.array([1.0, 0.0, 0.0]),
    )
    changed_floor = [b1k_geometry.TriangleComponent(
        floor[0].identity,
        floor[0].triangles + np.array([1.0, 0.0, 0.0]))]

    with pytest.raises(ValueError, match="build_b1k_authorities"):
        b1k_semantic.B1KSemanticAuthority(
            changed, scene_authority_sha256=original["sha256"])
    with pytest.raises(ValueError, match="build_b1k_authorities"):
        b1k_geometry.B1KGeometryAuthority(
            floor_components=changed_floor,
            collision_components=collision,
            scene_authority_sha256=original["sha256"])


def test_b1k_semantic_assigns_points_against_runtime_triangles():
    """Catches semantic assignment using runtime list order or bbox proxies."""
    index, _atom = _semantic_authority()

    assigned = index.assign(np.array([
        [0.0, 0.2, 0.0],
        [0.5, 0.2, 0.0],
        [2.0, 2.0, 2.0],
    ]), tol=0.05)

    assert assigned.tolist() == [1, 2, 0]


def test_b1k_batched_triangle_distances_match_scalar_exact_distance():
    """Catches a vectorized semantic query changing the exact metric."""
    triangles = np.concatenate([
        _rectangle_triangles(-0.4, 0.2, -0.5, 0.1, y=0.2),
        _rectangle_triangles(0.1, 0.7, -0.2, 0.8, y=0.6),
    ])
    points = np.array([
        [0.0, 0.2, 0.0],
        [0.4, 0.9, 0.3],
        [-0.7, 0.4, -0.6],
        [1.1, -0.2, 0.5],
    ], dtype=np.float64)

    batched = b1k_semantic._point_triangle_distances_m(
        points, triangles, point_chunk_size=2, triangle_chunk_size=1)
    scalar = np.array([
        b1k_semantic._point_triangle_distance_m(point, triangles)
        for point in points
    ])

    assert batched == pytest.approx(scalar, abs=1e-12)


def test_b1k_thresholded_triangle_distances_prune_only_impossible_pairs(
        monkeypatch):
    """The assignment fast path must preserve every tolerance hit exactly."""
    near = _rectangle_triangles(-0.2, 0.2, -0.2, 0.2, y=0.2)
    far = _rectangle_triangles(9.8, 10.2, 9.8, 10.2, y=0.2)
    triangles = np.concatenate([near, far])
    points = np.array([
        [0.0, 0.2, 0.0],
        [10.0, 0.23, 10.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float64)
    exact = b1k_semantic._point_triangle_distances_m(points, triangles)
    original = b1k_semantic._paired_point_triangle_distances_m
    evaluated_pair_counts = []

    def measured(point_values, triangle_values):
        evaluated_pair_counts.append(len(point_values))
        return original(point_values, triangle_values)

    monkeypatch.setattr(
        b1k_semantic, "_paired_point_triangle_distances_m", measured)
    thresholded = b1k_semantic._point_triangle_distances_m(
        points, triangles, max_distance_m=0.05)

    assert thresholded[:2] == pytest.approx(exact[:2], abs=1e-12)
    assert np.isinf(thresholded[2])
    assert sum(evaluated_pair_counts) == 4
    assert sum(evaluated_pair_counts) < len(points) * len(triangles)


def test_b1k_batched_assignment_preserves_distance_then_id_tie_break():
    """Catches vectorization making overlapping instance ties order-dependent."""
    triangles = _rectangle_triangles(-0.2, 0.2, -0.2, 0.2, y=0.2)
    instances = [
        b1k_semantic.RuntimeInstanceSpec(
            "/World/z-copy", "copy", "z", "copy.n.01", triangles),
        b1k_semantic.RuntimeInstanceSpec(
            "/World/a-copy", "copy", "a", "copy.n.01", triangles),
    ]
    index, _atom = _semantic_authority(instances)

    assigned = index.assign(np.array([[0.0, 0.2, 0.0]]), tol=0.05)

    assert assigned.tolist() == [1]


def test_b1k_contact_confirmation_publishes_global_winner_and_runner_up():
    """Catches resolving an A3 witness only within caller candidates."""
    index, atom = _semantic_authority()

    result = index.confirm_contact_instance(
        1, [0.0, 0.2, 0.0], candidate_instance_ids=[1])

    assert result == {
        "authority": "b1k_runtime_triangle_universe",
        "schema": "b1k-contact-triangle-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 1,
        "category": "table.n.02",
        "runtime_instance_triangles_sha256":
            index.instance_triangles_sha256(1),
        "contact_triangle_distance_m": 0.0,
        "runner_up_triangle_distance_m": pytest.approx(0.4),
        "triangle_distance_margin_m": pytest.approx(0.4),
        "global_query_protocol":
            "b1k-runtime-instance-triangle-universe.v1",
        "scene_authority_sha256": atom["sha256"],
        "global_universe_sha256":
            b1k_semantic.complete_triangle_universe_sha256(
                global_query_protocol=
                    "b1k-runtime-instance-triangle-universe.v1",
                scene_authority_sha256=atom["sha256"]),
        "global_winner_instance_id": 1,
        "global_runner_up_instance_id": 2,
    }
    mismatch = index.confirm_contact_instance(
        2, [0.0, 0.2, 0.0], candidate_instance_ids=[2])
    assert mismatch["confirmed"] is False
    assert mismatch["reason"] == "global_contact_winner_mismatch"
    assert mismatch["global_winner_instance_id"] == 1
    assert mismatch["global_runner_up_instance_id"] == 2


def test_b1k_contact_confirmation_is_a_source_bound_a3_proof():
    """Catches the shared A3 gate remaining hard-coded to MP3D fields."""
    index, atom = _semantic_authority()
    identity = index.confirm_contact_instance(1, [0.0, 0.2, 0.0])

    assert consensus.a3_exact_contact_identity_valid(
        identity,
        full_instance_id=1,
        depth_instance_id=1,
        authority_binding=_b1k_binding(atom["sha256"]),
    ) is True
    assert consensus.a3_exact_contact_identity_valid(
        identity,
        full_instance_id=1,
        depth_instance_id=1,
        authority_binding=_b1k_binding("0" * 64),
    ) is False

    semantic_sha256 = "a" * 64
    substituted_protocol = {
        "authority": "mp3d_full_face_universe",
        "schema": "mp3d-contact-face-identity.v1",
        "confirmed": True,
        "reason": "confirmed",
        "instance_id": 1,
        "category": "table.n.02",
        "streaming_instance_faces_sha256": "b" * 64,
        "contact_face_distance_m": 0.0,
        "runner_up_face_distance_m": 0.4,
        "face_distance_margin_m": 0.4,
        "global_query_protocol":
            "mp3d-complete-face-instance-universe.v1",
        "semantic_ply_sha256": semantic_sha256,
        "global_universe_sha256": semantic.complete_face_universe_sha256(
            global_query_protocol=
                "mp3d-complete-face-instance-universe.v1",
            semantic_ply_sha256=semantic_sha256),
        "global_winner_instance_id": 1,
        "global_runner_up_instance_id": 2,
    }
    assert consensus.a3_exact_contact_identity_valid(
        substituted_protocol,
        full_instance_id=1,
        depth_instance_id=1,
        authority_binding=_b1k_binding(atom["sha256"]),
    ) is False


def test_b1k_contact_proof_survives_shared_stability_certificate():
    """Catches certificate summaries discarding the B1K proof vocabulary."""
    index, atom = _semantic_authority()
    identity = index.confirm_contact_instance(1, [0.0, 0.2, 0.0])
    outcome = _outcome(collision=True, progress=0.4 / 1.5)
    outcome["physical"]["authority"] = "b1k_geometry"
    outcome["physical"]["contact"]["full_geometry_attribution"] = {
        "instance_id": 1, "category": "table.n.02", "unattributed": False,
    }
    outcome["physical"]["contact"]["depth_mask_attribution"] = {
        "instance_id": 1, "category": "table.n.02", "unattributed": False,
    }
    outcome["depth_physical"]["contact"] = {
        "depth_mask_attribution": {
            "instance_id": 1, "category": "table.n.02",
            "unattributed": False,
        },
    }
    rows = [{
        "perturbation_id": perturbation["id"],
        "transform": {
            key: perturbation[key] for key in ("x_m", "z_m", "yaw_deg")
        },
        "physical": copy.deepcopy(outcome["physical"]),
        "depth_physical": copy.deepcopy(outcome["depth_physical"]),
        "corridor_coverage": 1.0,
        "exact_contact_identity": copy.deepcopy(identity),
    } for perturbation in consensus.R2R_A_STABILITY_PERTURBATIONS]

    certificate = consensus.build_a_stability_certificate(
        outcome["actions"], rows,
        authority_binding=_b1k_binding(atom["sha256"]))

    assert certificate["summary"]["contact_instance_stable"] is True
    assert certificate["summary"]["contact_instance_id"] == 1


def test_b1k_target_geometry_uses_synset_triangles_and_scene_authority():
    """Catches proxy target geometry or a source-unbound B atom."""
    index, atom = _semantic_authority()

    target = index.target_geometry_atom(
        1, LEVEL_FLOOR_FIT.estimate,
        expected_scene_authority_sha256=atom["sha256"],
        pose={"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
    )

    assert target["schema"] == "b1k-b-target-geometry.v1"
    assert target["instance_id"] == 1
    assert target["category"] == "table.n.02"
    assert target["scene_authority_sha256"] == atom["sha256"]
    assert target["full_triangle_protocol"] == \
        "b1k-runtime-instance-triangles.v1"
    assert target["full_triangle_count"] == 2
    assert target["full_triangles_sha256"] == \
        index.instance_triangles_sha256(1)
    assert target["ground_support"]["frame"] == "pbench_world_xz"
    assert target["reference_centroid"]["frame"] == "pbench_world_xyz"
    assert target["reference_centroid"]["world_xyz_m"] == \
        pytest.approx([0.0, 0.2, 0.0])
    assert "breakfast_table" not in repr(target)
    assert "table_model" not in repr(target)
    assert "/World/a-table" not in repr(target)
    assert target["sha256"] == record.canonical_atom_sha256({
        key: value for key, value in target.items() if key != "sha256"
    })
    with pytest.raises(ValueError, match="scene authority digest"):
        index.target_geometry_atom(
            1, LEVEL_FLOOR_FIT.estimate,
            expected_scene_authority_sha256="0" * 64,
            pose={"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
        )


@pytest.mark.parametrize("invalid_instance_id", [True, 1.9])
def test_b1k_semantic_rejects_coercive_instance_ids(invalid_instance_id):
    """Catches bool/float aliases selecting a different stable instance."""
    index, atom = _semantic_authority()

    with pytest.raises(TypeError, match="positive integral"):
        index.instance_triangles(invalid_instance_id)
    with pytest.raises(TypeError, match="positive integral"):
        index.confirm_contact_instance(
            invalid_instance_id, [0.0, 0.2, 0.0])
    with pytest.raises(TypeError, match="positive integral"):
        index.target_geometry_atom(
            invalid_instance_id, LEVEL_FLOOR_FIT.estimate,
            expected_scene_authority_sha256=atom["sha256"],
            pose={"position": [0.0, 0.0, 0.0], "yaw_rad": 0.0})
