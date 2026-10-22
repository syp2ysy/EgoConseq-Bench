"""Collect every observation of the inputs and all of its GT, loading each scene once (GT v2).

    python -m pipeline.meta.collect --inputs "<root>/inputs" --dataset r2r \
        --output "<root>/_work/r2r-0" [--shards 12 --shard-index 0]

Shards are equal contiguous ranges of the scene-grouped inputs (a boundary scene opens in two shards);
one run per output directory at a time.

Motion model (frozen scene, fixed floor): every observation has one start floor y0, the support surface
under its input pose. The camera is at y0 + camera height at the start, every checkpoint and every end;
the body is a vertical cylinder over (y0 + 0.05, y0 + camera height + 0.10) that moves horizontally at y0
and turns in place. Per observation:
  1. find y0, render RGB / depth / semantics and save the RGB;
  2. sweep every program at every radius against the frozen scene (pipeline.meta.geometry): start
     validity, first contact and the command it falls in (A1/A2), the contacted instance (A3);
  3. for programs free at some radius: analytic end and checkpoint cameras, relations to the target
     points (B1/B2/B3/A4) and C1 families of terminal views, each with the radii it applies to.
Correctness failures produce no GT and are counted; quality measures are stored as numbers.

Output: records.jsonl.gz (one record per observation), images/, future_views/, scene_assets.json,
collection.json (GT contract, which a resumed run must match; the code of resumed runs is listed), report.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import difflib
import fcntl
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from pipeline import actions as A, config, contact_identity, io_utils, perception, surface_points
from pipeline.meta import geometry as G, manifest

GEOMETRY_SOURCES = {"r2r": "mp3d-scene-glb", "gs": "sage3d-collision-mesh",
                    "b1k": "b1k-collision-meshes-of-every-object-in-its-initial-state"}
TASKS_BY_DATASET = {"r2r": ["A1", "A2", "A3", "B1", "B2", "B3", "A4", "C1"],
                    "gs": ["A1", "A2", "B2", "A4"],
                    "b1k": ["A1", "A2", "A3", "B1", "B2", "B3", "A4", "C1"]}
CAMERA_HEIGHTS_M, HFOVS_DEG = (.5, 1., 1.25, 1.5), (79., 90., 110.)  # every camera configuration of the inputs
GT_CONFIG = json.loads(json.dumps({
    "schema": "seen-meta-gt.v3",
    "motion": "frozen scene; one start floor y0 per observation for every radius, checkpoint and end; "
              "the body moves horizontally at y0 and turns in place",
    "radii_m": [.10, .15, .20, .25],
    "body": {"top_above_camera_m": config.FINE_BODY_TOP_ABOVE_CAMERA_M,
             "ground_clearance_m": config.FINE_BODY_GROUND_CLEARANCE_M},
    "atom": {"forward_m": manifest.ATOM_FORWARD_M, "turn_deg": manifest.ATOM_TURN_DEG},
    "geometry": {"sources": GEOMETRY_SOURCES, "tolerance_m": G.TOLERANCE_M,
                 "contact_window_m": G.CONTACT_WINDOW_M, "boundary_m": G.BOUNDARY_M},
    "start_support": {"r2r": "mesh", "b1k": "mesh", "gs": "authenticated SAGE-3D source floor",
                      "mesh": {"method": "highest horizontal mesh face under the input pose",
                               "search_above_m": .02, "search_below_m": .20, "min_normal_y": .9}},
    "start_solid": "a start is invalid when the body centre lies strictly inside a closed (watertight) convex "
                   "collision part (GS, B1K); open parts enclose nothing but still collide",
    "a3": {"contact_region_m": contact_identity.CONTACT_REGION_M, "min_points": contact_identity.MIN_POINTS,
           "face_tie_margin_m": config.A3_CONTACT_FACE_TIE_MARGIN_M,
           "face_max_distance_m": contact_identity.CONTACT_FACE_MAX_M},
    "target": {"preferred_min_distance_m": .5, "gs_ray_tolerance_m": .03, "gs_ray_tolerance_fraction": .01},
    "checkpoint_fractions": [.25, .5, .75],
    "direction": {"axis_half_width_deg": 7.5, "min_planar_range_m": config.TARGET_DIRECTION_MIN_RANGE_M},
    "c1": {"options": 4, "min_position_gap_m": .26, "min_yaw_gap_deg": 14.9},
    "budget": {"consequence_programs": 12, "c1_families": 2},
    "tasks_by_dataset": TASKS_BY_DATASET,
}))  # JSON-exact, so a stored contract compares equal
RECORDS = "records.jsonl.gz"
CONTRACT_KEYS = ("gt", "dataset", "shards", "shard_index", "inputs", "scenes", "engines")  # a resumed run must match
RAY_PAD_M = 1e-5  # face search margin, beyond trimesh's ray slack (1e-13 barycentric, 1e-6 behind the origin)


# -- scene ----------------------------------------------------------------------------------------

@dataclasses.dataclass
class Scene:
    """One opened scene: its frozen collision triangles (Habitat world) and asset identities."""
    scene_id: str
    scene_glb: str
    triangles: np.ndarray
    assets: dict
    solid_parts: bool = False  # GS / B1K collision bodies are closed convex parts; MP3D scans are surfaces
    part_of_face: np.ndarray = None  # B1K: collision component of every face (GS: found by connectivity)
    face_instance: np.ndarray = None  # B1K: object instance owning every face
    box_min: np.ndarray = dataclasses.field(init=False, repr=False)
    box_max: np.ndarray = dataclasses.field(init=False, repr=False)
    _mesh: object = dataclasses.field(default=None, init=False, repr=False)
    _parts: tuple = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self):
        xz = self.triangles[..., [0, 2]]
        self.box_min, self.box_max = xz.min(1), xz.max(1)

    def mesh(self):
        if self._mesh is None:
            import trimesh
            vertices = self.triangles.reshape(-1, 3)
            self._mesh = trimesh.Trimesh(vertices, np.arange(len(vertices)).reshape(-1, 3), process=False)
        return self._mesh

    def inside_solid(self, point):
        """Whether a point lies strictly inside one of the scene's closed convex collision parts (open or non-convex parts, such as room shells, never enclose a solid)."""
        if not self.solid_parts:
            return False
        if self._parts is None:  # faces grouped by part once; a part is examined when a query falls in its box
            labels, triangles = self.part_of_face, self.triangles
            if labels is None:
                import trimesh
                mesh = trimesh.Trimesh(self.triangles.reshape(-1, 3), np.arange(self.triangles.size // 3).reshape(-1, 3))
                labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
                triangles = mesh.triangles
            order = np.argsort(labels, kind="stable")
            starts = np.flatnonzero(np.r_[True, np.diff(labels[order]) != 0])
            triangles = triangles[order]
            self._parts = (triangles, np.r_[starts, len(order)], np.minimum.reduceat(triangles.min(1), starts),
                           np.maximum.reduceat(triangles.max(1), starts), {})
        triangles, bounds, lo, hi, convex = self._parts
        p = np.asarray(point, dtype=float)
        for i in np.flatnonzero(np.all((lo <= p) & (p <= hi), axis=1)):
            if i not in convex:
                convex[i] = convex_part(triangles[bounds[i]:bounds[i + 1]])
            part = convex[i]
            if part is not None and np.all(part[0] < p) and np.all(p < part[1]) and np.all(part[2] @ p < part[3]):
                return True
        return False

    def obstacles(self, lo_xz, hi_xz, low_y, high_y):
        crop = np.flatnonzero(np.all((self.box_max >= lo_xz) & (self.box_min <= hi_xz), axis=1))
        return G.Obstacles(self.triangles[crop], low_y, high_y, crop)

    def support_below(self, x, z, input_y):
        """Highest horizontal face under (x, z) in the search window around the input height."""
        rule = GT_CONFIG["start_support"]["mesh"]
        sel = np.flatnonzero((self.box_min[:, 0] <= x) & (x <= self.box_max[:, 0]) &
                             (self.box_min[:, 1] <= z) & (z <= self.box_max[:, 1]))
        a, b, c = (self.triangles[sel, i] for i in range(3))
        e1, e2, w = (b - a)[:, [0, 2]], (c - a)[:, [0, 2]], np.array([x, z]) - a[:, [0, 2]]
        den = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (w[:, 0] * e2[:, 1] - e2[:, 0] * w[:, 1]) / den
            v = (e1[:, 0] * w[:, 1] - w[:, 0] * e1[:, 1]) / den
        normal = np.cross(b - a, c - a)
        normal_y = np.abs(normal[:, 1]) / np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
        y = a[:, 1] + u * (b[:, 1] - a[:, 1]) + v * (c[:, 1] - a[:, 1])
        ok = ((np.abs(den) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) &
              (normal_y >= rule["min_normal_y"]) & (y >= input_y - rule["search_below_m"]) &
              (y <= input_y + rule["search_above_m"]))
        if not ok.any():
            return None
        best = int(np.argmax(np.where(ok, y, -np.inf)))
        return {"ground_y": float(y[best]), "face": int(sel[best]), "normal_y": float(normal_y[best])}


def convex_part(tri):
    """(min, max, outward normals, offsets) of a closed convex part, or None when it is not closed (every edge
    shared by exactly two faces; an open surface, such as a piece of a room shell, encloses nothing) or not
    convex. Open parts still collide: only this start test ignores them."""
    import trimesh
    if not trimesh.Trimesh(tri.reshape(-1, 3), np.arange(len(tri) * 3).reshape(-1, 3)).is_watertight:  # merges vertices
        return None
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    size = np.linalg.norm(normal, axis=1)
    tri, normal = tri[size > 1e-12], normal[size > 1e-12] / size[size > 1e-12, None]
    if not len(tri):
        return None
    vertices = tri.reshape(-1, 3)
    normal *= np.sign(((tri[:, 0] - vertices.mean(0)) * normal).sum(1))[:, None]  # outward
    offset = (normal * tri[:, 0]).sum(1)
    step = max(1, (1 << 22) // len(normal))  # vertices per block: bounded memory, stops at the first outlier
    if any(np.max(vertices[i:i + step] @ normal.T - offset) > 1e-3 for i in range(0, len(vertices), step)):
        return None  # e.g. walls joined around a room
    return vertices.min(0), vertices.max(0), normal, offset


def scene_geometry(dataset, session, scene_id):
    if dataset == "r2r":
        import trimesh
        path = Path(config.MP3D_ROOT).resolve(strict=True) / scene_id / f"{scene_id}.glb"
        mesh = trimesh.load(path).to_geometry()
        v = mesh.vertices  # MP3D assets are Z-up; Habitat world is Y-up (x, z, -y)
        triangles = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)[mesh.faces]
        assets = {"geometry": str(path), "geometry_sha256": io_utils.sha256_file(path),
                  "semantic_ply_sha256": getattr(session._sem, "_semantic_ply_sha256", None)}
        return str(path), triangles, assets
    if dataset == "gs":
        import trimesh
        from pipeline import gs_native
        evidence = gs_native.scene_mesh(session.scene_dir, session._collision)
        mesh = trimesh.load(evidence["mesh_path"]).to_geometry()
        assets = {"geometry": evidence["mesh_path"], "geometry_sha256": evidence["mesh_sha256"],
                  "source_sha256": evidence["source_asset"]["sha256"],
                  "scene_binding_sha256": session._collision.binding_atom()["sha256"]}
        return session.scene_dir, np.asarray(mesh.vertices)[mesh.faces], assets
    components = session.collision_components
    triangles = np.concatenate([c.triangles for c in components])
    part = np.repeat(np.arange(len(components)), [len(c.triangles) for c in components])
    owner = np.array([session._semantic.instance_id_for_component(c.identity) for c in components])[part]
    assets = {"geometry": session.scene_glb, "scene_authority_sha256": session.scene_authority_sha256,
              "geometry_sha256": manifest.digest(np.round(triangles, 6).tolist())}
    return session.scene_glb, triangles, assets, part, owner


def open_scene(dataset, scene):
    """Open one scene (inputs/scenes.jsonl row) with every camera configuration of the GT contract.

    Assets come from the configured dataset roots; the split only selects the source manifest,
    so Seen and Unseen scenes open the same way.
    """
    heights, fovs = list(CAMERA_HEIGHTS_M), [(f, config.vfov_for_hfov(f)) for f in HFOVS_DEG]
    scene_id, split = scene["scene_id"], scene["source_split"]
    if dataset == "r2r":
        from pipeline import dataset_contracts
        from pipeline.sim import SimSession
        root = Path(config.MP3D_ROOT).resolve(strict=True)
        session = SimSession(str(root / scene_id / f"{scene_id}.glb"),
                             scene_dataset_cfg=str(root / "mp3d_annotated_basis.scene_dataset_config.json"),
                             semantic_format=dataset_contracts.dataset_source_contract("r2r").semantic_format,
                             source_dataset="r2r", official_split=split, heights=heights, fovs=fovs,
                             semantic_query_workers=1, gpu_device_id=int(os.environ.get("PBENCH_HABITAT_GPU_DEVICE_ID", "0")))
    elif dataset == "gs":
        from pipeline.gs_sim import GsSimSession
        from pipeline.scene_pool import discover_gs_scenes
        manifest_path = Path(config.GS_ROOT) / "splits" / f"{split}.json"
        session = GsSimSession(discover_gs_scenes(config.GS_ROOT, manifest_path, requested=(scene_id,),
                                                  source_split=split)[0], heights=heights)
    else:  # one official observation profile for every B1K scene
        from pipeline.b1k_sim import B1KSimSession
        from pipeline.scene_pool import discover_b1k_train_scenes
        session = B1KSimSession(discover_b1k_train_scenes(config.B1K_ROOT, config.B1K_SOURCE_MANIFEST,
                                                          requested=(scene_id,))[0], heights=heights, fovs=fovs)
    glb, triangles, assets, *owners = scene_geometry(dataset, session, scene_id)
    return session, Scene(scene_id, glb, np.ascontiguousarray(triangles, dtype=np.float64), assets,
                          solid_parts=dataset != "r2r", part_of_face=owners[0] if owners else None,
                          face_instance=owners[1] if owners else None)


# -- observation frame ----------------------------------------------------------------------------

def start_support(session, scene, dataset, requested):
    """The start floor y0 under the input pose, or None when there is no support surface there."""
    if dataset == "gs":
        try:
            support = session.support_ground(requested.copy())
        except ValueError:
            return None
        return {"ground_y": float(support["ground_y"]), "method": support["method"],
                "evidence": {"input_y": float(requested[1])}}
    hit = scene.support_below(float(requested[0]), float(requested[2]), float(requested[1]))
    if hit is None:
        return None
    point = np.array([[requested[0], hit["ground_y"], requested[2]]])
    instance = (int(scene.face_instance[hit["face"]]) if scene.face_instance is not None
                else int(session.assign_instances(point)[0]))
    return {"ground_y": hit["ground_y"], "method": "highest_horizontal_mesh_face_under_input_pose",
            "evidence": {"input_y": float(requested[1]), "face": hit["face"], "normal_y": hit["normal_y"],
                         "instance_id": instance, "category": str(session.id_to_cat.get(instance, ""))}}


def render_frame(session, scene, obs_id, position, yaw, camera_height, hfov):
    """The observation frame at the fixed floor; a camera that differs from the request raises ValueError."""
    from pipeline import floor_plane
    from pipeline.frame import build_frame

    vfov = config.vfov_for_hfov(hfov)
    rendered = session.render(position, yaw, camera_height, hfov, vfov)
    camera = session.sensor_pose_evidence(position, yaw, camera_height, hfov, vfov)
    if not np.allclose(camera["position_world"], np.asarray(position) + [0., camera_height, 0.], atol=1e-5, rtol=0):
        raise ValueError("rendered camera differs from the fixed-floor camera")
    plane = floor_plane.FloorPlaneEstimate((0., 1., 0.), 0.)  # the fixed start floor of the agent frame
    frame = build_frame(session, position, yaw, frame_id=obs_id, scene_id=scene.scene_id, scene_glb=scene.scene_glb,
                        floor_plane=plane, cam_h=camera_height, hfov=hfov, vfov=vfov, rendered=rendered)
    quality = {"visible_floor_ratio": float(frame.quality["visible_floor_ratio"]),
               "valid_depth_ratio": float(frame.quality["valid_depth_ratio"]),
               "dark_pixel_fraction": float(np.mean(np.max(frame.rgb, axis=2) < 12))}
    return frame, {"camera": {**camera, "yaw_rad": float(yaw)}, "image_quality": quality}


def gs_ray_validator(scene, frame):
    """GS target pixels must hit the collision mesh where gsplat's expected depth says; the hit is the point.

    Only a first hit within the tolerance of the expected depth passes, so only faces near that segment can
    decide: the mesh's triangle index supplies them (boxes padded beyond trimesh's numerical slack) and the
    same trimesh ray test runs on them, in face order.
    """
    import trimesh

    tree = scene.mesh().triangles_tree
    camera = np.asarray(frame.position, dtype=float) + [0., frame.sensor.nominal_camera_offset_m, 0.]
    tol = GT_CONFIG["target"]

    def validate(_pixel, world):
        direction = np.asarray(world, dtype=float) - camera
        expected = float(np.linalg.norm(direction))
        unit, slack = direction / expected, max(tol["gs_ray_tolerance_m"], tol["gs_ray_tolerance_fraction"] * expected)
        ends = camera + np.linspace(-RAY_PAD_M, expected + slack + RAY_PAD_M,
                                    int(math.ceil((expected + slack) / .25)) + 2)[:, None] * unit
        faces = sorted({int(f) for a, b in zip(ends[:-1], ends[1:])
                        for f in tree.intersection(np.r_[np.minimum(a, b) - RAY_PAD_M, np.maximum(a, b) + RAY_PAD_M])})
        if not faces:
            return None
        near = trimesh.Trimesh(scene.triangles[faces].reshape(-1, 3), np.arange(3 * len(faces)).reshape(-1, 3),
                               process=False)
        hits, _, _ = near.ray.intersects_location([camera], [unit], multiple_hits=False)
        if not len(hits):
            return None
        actual = float(np.linalg.norm(hits[0] - camera))
        return hits[0] if abs(actual - expected) <= slack else None
    return validate


def select_target(scene, dataset, frame, obs_id):
    """One visible object's surface points, preferring points at least the preferred distance away."""
    from post_QA.categories import display_category_name
    validator = gs_ray_validator(scene, frame) if dataset == "gs" else None
    preferred = GT_CONFIG["target"]["preferred_min_distance_m"]
    target = (surface_points.select_target(frame, record_uid=obs_id, min_distance_m=preferred, point_validator=validator)
              or surface_points.select_target(frame, record_uid=obs_id, point_validator=validator))
    if target is None:
        return None
    return {"instance_id": target["instance_id"], "category": display_category_name(target["category"]),
            "source_category": target["category"],
            "points": [{"point_id": p["point_id"], "pixel_xy": p["pixel_xy_px"], "world_xyz": p["world_xyz_m"]}
                       for p in target["points"]]}


# -- collision ------------------------------------------------------------------------------------

def command_index(spans, atom):
    return None if atom is None else next(i for i, (b, e) in enumerate(spans) if b <= atom < e)


def forward_m(atoms):
    return sum(a["m"] for a in atoms if a["type"] == "forward")


def image_evidence(body_frame, commands, radius, executed_m):
    """Whether the initial depth view alone predicts the collision, and how well it covers the corridor."""
    from pipeline import rollout
    actions = A.parse_actions(commands)
    view = rollout.view_collision_rollout(body_frame, actions, radius)
    cov = rollout.corridor_coverage_details(body_frame, actions, radius, max_arc_m=executed_m)
    keys = ("raw_coverage", "supported_samples", "total_samples", "out_of_frame_samples", "invalid_depth_samples",
            "occluded_samples", "ground_visibility_start_m", "near_field_exempt_m", "max_unsupported_run_m")
    index = view.get("contact_action_index")
    return {"depth_view_collision": bool(view["collision"]),
            "depth_view_contact_command": None if index is None else index + 1,
            "corridor": {k: cov.get(k) for k in keys}}


def sweep(session, dataset, scene, frame, record, programs, floor_y, xz, yaw, counters):
    """Every program at every radius against the frozen scene; A3 from one batched instance query."""
    body, radii = record["body"], GT_CONFIG["radii_m"]
    paths = [G.poses(xz, yaw, p["atomic_actions"]) for p in programs]
    reach = max(radii) + .05
    points = np.array([xz] + [pose[:2] for path in paths for pose in path])
    obstacles = scene.obstacles(points.min(0) - reach, points.max(0) + reach,
                                floor_y + body["ground_clearance_m"], floor_y + body["top_m"])
    body_frame = dataclasses.replace(frame, vf=perception.VoxelField(frame.pts[perception.obstacle_mask(
        frame.pts, frame.floor_plane, band=(body["ground_clearance_m"], body["top_m"]))]))
    executions, contacts = [], []
    for program, path in zip(programs, paths):
        atomic = program["atomic_actions"]
        for radius in radii:
            result = G.execute(obstacles, xz, yaw, atomic, radius)
            execution = {"execution_id": manifest.digest([record["obs_id"], program["program_id"], radius])[:16],
                         "program_id": program["program_id"], "radius_m": radius,
                         "status": "ok" if result["valid_start"] else "start_in_contact"}
            counters[f"execution:{execution['status']}"] += 1
            executions.append(execution)
            if not result["valid_start"]:
                continue
            index = result["first_collision_atom"]
            execution.update(collision=result["collision"], first_collision_atom=index)
            travel = None
            if result["collision"]:
                contact = result["contact"]
                travel = forward_m(atomic[:index]) + contact["travel_in_atom_m"]
                heading = path[index - 1][2] if index else yaw
                pairs = sorted({(f, *np.round(p, 6)) for f, p in zip(contact["faces"], contact["points"])})
                execution["contact"] = {"travel_m": travel, "travel_in_atom_m": contact["travel_in_atom_m"],
                                        "stop_pose": [*contact["centre_xz"], heading],
                                        "faces": [int(p[0]) for p in pairs], "points": [list(p[1:]) for p in pairs]}
                contacts.append(execution)
            facts = []
            for commands in program["commands"]:
                spans = commands["command_atomic_spans"]
                fact = {"evidence": image_evidence(body_frame, commands["actions"], radius, travel),
                        "first_collision_command": None, "contact_progress_m": None, "command_length_m": None}
                if result["collision"]:
                    k = command_index(spans, index)
                    begin, end = spans[k]
                    fact.update(first_collision_command=k + 1,
                                contact_progress_m=forward_m(atomic[begin:index]) + result["contact"]["travel_in_atom_m"],
                                command_length_m=forward_m(atomic[begin:end]))
                facts.append(fact)
            execution["commands"] = facts
    if contacts and "A3" in TASKS_BY_DATASET[dataset]:
        flat = [p for e in contacts for p in e["contact"]["points"]]
        owners = None if scene.face_instance is None else \
            [int(scene.face_instance[f]) for e in contacts for f in e["contact"]["faces"]]
        resolved = iter(contact_identity.resolve(session, dataset, flat, owners))
        for execution in contacts:
            points = execution["contact"]["points"]
            mine = [next(resolved) for _ in points]
            execution["contact_identity"] = contact_identity.identify(dataset, frame, points, mine, body, floor_y,
                                                                      session.id_to_cat)
            counters[f"a3:{execution['contact_identity'].get('reason', 'certified')}"] += 1
    return executions


# -- consequences ---------------------------------------------------------------------------------

def relations(target, initial_camera, camera):
    """Target points from a camera: 3D distance to the optical centre and camera-frame direction."""
    values = []
    for point in target["points"]:
        world = np.asarray(point["world_xyz"], dtype=float)
        local = perception.local_from_world(world[None, :], camera["position_world"], camera["yaw_rad"])[0]
        planar = math.hypot(local[0], local[2])
        value = {"point_id": point["point_id"], "distance_m": float(np.linalg.norm(local)),
                 "initial_distance_m": float(np.linalg.norm(world - initial_camera["position_world"])),
                 "planar_range_m": planar}
        if planar >= GT_CONFIG["direction"]["min_planar_range_m"]:
            azimuth = float(A.wrap_deg(math.degrees(math.atan2(local[0], local[2]))))
            elevation = float(math.degrees(math.atan2(local[1], planar)))
            value.update(azimuth_deg=azimuth, elevation_deg=elevation,
                         horizontal=A.horizontal_direction(azimuth), vertical=A.vertical_direction(elevation))
        values.append(value)
    return values


def checkpoints(commands, spans):
    """25/50/75% of every public forward command, mapped to its native atom and the partial distance in it."""
    stage = 0
    for index, (command, (begin, _end)) in enumerate(zip(commands, spans)):
        if command["type"] != "forward":
            continue
        stage += 1
        for fraction in GT_CONFIG["checkpoint_fractions"]:
            distance = command["m"] * fraction
            atom = math.ceil(distance / manifest.ATOM_FORWARD_M - 1e-9) - 1
            yield {"action_index": index + 1, "forward_stage": stage, "fraction": fraction,
                   "atom_index": begin + atom, "partial_forward_m": distance - manifest.ATOM_FORWARD_M * atom}


def segments(atomic):
    """Commands after merging consecutive forwards and consecutive same-direction turns."""
    keys = ["F" if a["type"] == "forward" else "R" if a["deg"] > 0 else "L" for a in atomic]
    return sum(1 for i, key in enumerate(keys) if i == 0 or key != keys[i - 1])


def consequence_programs(obs_id, programs, safe):
    """Up to the budget of programs free at some radius, spread over merged command counts."""
    buckets = {n: [] for n in range(1, 7)}
    for value in sorted(programs, key=lambda p: manifest.digest([obs_id, p["program_id"]])):
        if safe.get(value["program_id"]):
            buckets[min(6, max(1, segments(value["atomic_actions"])))].append(value["program_id"])
    chosen, limit = [], GT_CONFIG["budget"]["consequence_programs"]
    while len(chosen) < limit and any(buckets.values()):
        for length in (1, 2, 3, 4, 1, 2, 3, 4, 5, 6):
            if buckets[length] and len(chosen) < limit:
                chosen.append(buckets[length].pop())
    return set(chosen)


def distinct_endpoint(a, b):
    c = GT_CONFIG["c1"]
    return (np.linalg.norm(np.asarray(a["position_world"]) - b["position_world"]) > c["min_position_gap_m"]
            or abs(A.wrap_deg(math.degrees(a["yaw_rad"] - b["yaw_rad"]))) >= c["min_yaw_gap_deg"])


def consequences(session, dataset, scene, frame, record, programs, executions, floor_y, xz, yaw, output, counters):
    cam_h, obs_id = record["camera_height_m"], record["obs_id"]
    target = select_target(scene, dataset, frame, obs_id)
    initial = record["camera"]
    safe = {}
    for e in executions:
        if e["status"] == "ok" and not e["collision"]:
            safe.setdefault(e["program_id"], []).append(e["radius_m"])
    chosen = consequence_programs(obs_id, programs, safe)

    def camera(pose):
        return {"position_world": [pose[0], floor_y + cam_h, pose[1]], "yaw_rad": pose[2]}

    outcomes = []
    for program in programs:
        if program["program_id"] not in chosen:
            continue
        atomic = program["atomic_actions"]
        path = G.poses(xz, yaw, atomic)
        end = camera(path[-1])
        outcome = {"program_id": program["program_id"], "radii_m": sorted(safe[program["program_id"]]),
                   "end": {"camera": end, "relations": relations(target, initial, end) if target else []},
                   "checkpoints": []}
        for k, commands in enumerate(program["commands"] if target else ()):
            for request in checkpoints(commands["actions"], commands["command_atomic_spans"]):
                x, z, heading = path[request["atom_index"] - 1] if request["atom_index"] else (*xz, yaw)
                position = np.array([x, z]) + request["partial_forward_m"] * G.heading(heading)
                view = camera((float(position[0]), float(position[1]), heading))
                outcome["checkpoints"].append({**request, "command_list": k, "camera": view,
                                               "relations": relations(target, initial, view)})
        outcomes.append(outcome)
    return {"target": target, "target_status": "ok" if target else "no_visible_target", "outcomes": outcomes,
            "c1": c1_families(session, record, programs, outcomes, floor_y, output, counters)
            if "C1" in TASKS_BY_DATASET[dataset] else []}


def c1_families(session, record, programs, outcomes, floor_y, output, counters):
    """Query + 3 programs from the same start, all free at the family's radii, with distinct ends and pixels."""
    from pipeline.future_view_selection import block_l1_certificate, encode_native_rgb_png
    by_id = {p["program_id"]: p for p in programs}
    ends = {o["program_id"]: o for o in outcomes}
    cam_h, hfov = record["camera_height_m"], record["hfov_deg"]
    vfov = config.vfov_for_hfov(hfov)
    views, families = {}, []

    def view(camera):
        key = manifest.digest([camera["position_world"], camera["yaw_rad"]])
        if key not in views:
            position = np.array([camera["position_world"][0], floor_y, camera["position_world"][2]])
            rgb = session.render(position, camera["yaw_rad"], cam_h, hfov, vfov).rgb
            readback = session.sensor_pose_evidence(position, camera["yaw_rad"], cam_h, hfov, vfov)
            if not np.allclose(readback["position_world"], camera["position_world"], atol=1e-5, rtol=0):
                raise ValueError("terminal render camera differs from the labelled camera")
            views[key] = (encode_native_rgb_png(rgb, expected_resolution=record["resolution"]), rgb)
        return views[key]

    def tokens(program_id):
        return [tuple(sorted(a.items())) for a in by_id[program_id]["atomic_actions"]]

    for query in sorted(ends, key=lambda p: manifest.digest([record["obs_id"], p])):
        if len(families) >= GT_CONFIG["budget"]["c1_families"]:
            break
        if by_id[query]["atomic_actions"][-1]["type"] != "forward":
            continue
        chosen, radii = [query], set(ends[query]["radii_m"])
        for other in sorted(ends, key=lambda p: (-difflib.SequenceMatcher(None, tokens(query), tokens(p),
                                                                          autojunk=False).ratio(), p)):
            if other in chosen or not radii & set(ends[other]["radii_m"]):
                continue
            if all(distinct_endpoint(ends[other]["end"]["camera"], ends[c]["end"]["camera"]) for c in chosen):
                chosen.append(other)
                radii &= set(ends[other]["radii_m"])
            if len(chosen) == GT_CONFIG["c1"]["options"]:
                break
        if len(chosen) < GT_CONFIG["c1"]["options"]:
            continue
        try:
            rendered = [view(ends[p]["end"]["camera"]) for p in chosen]
        except ValueError as error:  # an end view without depth (e.g. facing open sky) or a moved camera
            counters[f"c1:render_failed:{type(error).__name__}"] += 1
            continue
        if len({encoded.pixel_sha256 for encoded, _ in rendered}) != len(chosen):
            counters["c1:identical_terminal_pixels"] += 1
            continue
        options = []
        for program_id, (encoded, _rgb) in zip(chosen, rendered):
            path = output / "future_views" / f"{encoded.png_sha256}.png"
            if not path.exists() or io_utils.sha256_file(path) != encoded.png_sha256:  # a cut-off write is redone
                io_utils.atomic_write_binary(path, lambda stream: stream.write(encoded.png_bytes))
            options.append({"program_id": program_id, "image": f"future_views/{encoded.png_sha256}.png",
                            "pixel_sha256": encoded.pixel_sha256})
        family_id = manifest.digest([record["obs_id"], query])[:20]
        answer = int(family_id[:8], 16) % len(options)
        options.insert(answer, options.pop(0))
        l1 = [block_l1_certificate(rendered[i][1], rendered[j][1]) for i, j in itertools.combinations(range(4), 2)]
        families.append({"family_id": family_id, "query_program_id": query, "radii_m": sorted(radii),
                         "options": options, "answer_index": answer,
                         "min_block_l1": min(p["numerator"] / p["denominator"] for p in l1)})
    return families


# -- one observation ------------------------------------------------------------------------------

def collect_observation(session, scene, row, output, counters):
    dataset, obs_id = row["dataset"], row["obs_id"]
    cam_h, hfov = row["camera_height_m"], row["hfov_deg"]
    record = {"obs_id": obs_id, "dataset": dataset, "scene_id": row["scene_id"], "sources": row["sources"],
              "requested_pose": row["pose"], "camera_height_m": cam_h, "hfov_deg": hfov,
              "vfov_deg": config.vfov_for_hfov(hfov), "body": config.fine_body(cam_h)}
    requested = np.asarray(row["pose"]["position"], dtype=float)
    yaw = float(row["pose"]["yaw_rad"])
    support = start_support(session, scene, dataset, requested)
    if support is None:
        return {**record, "status": "no_start_support"}
    floor_y = support["ground_y"]
    position = np.array([requested[0], floor_y, requested[2]])
    record.update(start_support=support, pose={"position": position.tolist(), "yaw_rad": yaw})
    body = record["body"]
    if scene.inside_solid([requested[0], floor_y + (body["ground_clearance_m"] + body["top_m"]) / 2, requested[2]]):
        return {**record, "status": "start_inside_solid"}
    try:
        frame, seen = render_frame(session, scene, obs_id, position, yaw, cam_h, hfov)
    except ValueError as error:  # the rendered camera differs from the request, or the frame has no depth
        counters[f"render:{error}"] += 1
        return {**record, "status": "render_failed"}
    image = f"images/{obs_id}.png"
    Image.fromarray(frame.rgb).save(output / image)
    record.update(status="ok", camera=seen["camera"], image=image, image_sha256=io_utils.sha256_file(output / image),
                  image_quality=seen["image_quality"],
                  resolution=list(frame.rgb.shape[1::-1]),
                  visible_entities=[{k: x[k] for k in ("instance_id", "category", "source_category") if k in x}
                                    for x in frame.objects])
    programs = row["programs"]
    record["programs"] = programs
    xz = position[[0, 2]]
    record["executions"] = sweep(session, dataset, scene, frame, record, programs, floor_y, xz, yaw, counters)
    record.update(consequences(session, dataset, scene, frame, record, programs, record["executions"],
                               floor_y, xz, yaw, output, counters))
    return record


# -- run ------------------------------------------------------------------------------------------

def engines():
    from importlib import metadata
    found = {}
    for name in ("habitat-sim", "gsplat", "omnigibson", "trimesh", "shapely", "numpy"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def contract(args, inputs, scenes):
    repository = Path(__file__).resolve().parents[2]
    code = sorted(set((repository / "pipeline").glob("*.py")) | set((repository / "pipeline/meta").glob("*.py")))
    return {"gt": GT_CONFIG, "dataset": args.dataset, "shards": args.shards, "shard_index": args.shard_index,
            "inputs": {"path": str(inputs), "sha256": io_utils.sha256_file(inputs)},
            "scenes": {"path": str(scenes), "sha256": io_utils.sha256_file(scenes)},
            "engines": engines(),
            "code_sha256": {p.relative_to(repository).as_posix(): io_utils.sha256_file(p) for p in code}}


def resume(record_path):
    """Keep the complete records of an earlier run (streamed); return their ids and status counts."""
    done, statuses = set(), Counter()
    if not record_path.exists():
        return done, statuses
    temporary = record_path.with_name("tmp." + record_path.name)
    with gzip.open(record_path, "rt") as source, gzip.open(temporary, "wt", encoding="utf-8") as sink:
        try:
            for line in source:
                if not line.endswith("\n"):
                    break
                record = json.loads(line)
                sink.write(line)
                done.add(record["obs_id"])
                statuses[f"observation:{record['status']}"] += 1
        except EOFError:  # an interrupted last append
            pass
    temporary.replace(record_path)
    return done, statuses


def run(args):
    inputs = (args.inputs / f"{args.dataset}.jsonl.gz").resolve()
    scenes_path = (args.inputs / "scenes.jsonl").resolve()
    output = args.output.resolve()
    for name in ("images", "future_views"):
        (output / name).mkdir(parents=True, exist_ok=True)
    lock = open(output / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{output} is being written by another run") from None
    expected = json.loads(json.dumps(contract(args, inputs, scenes_path)))
    contract_path, record_path, report_path = output / "collection.json", output / RECORDS, output / "report.json"
    assets_path = output / "scene_assets.json"
    stored = json.loads(contract_path.read_text()) if contract_path.exists() else expected
    if any(stored[k] != expected[k] for k in CONTRACT_KEYS):
        raise ValueError("GT contract, inputs, scenes or engines changed since this output was started; use a new output")
    code, history = manifest.digest(expected["code_sha256"]), stored.get("resumed_with_code", [])
    if code != (history[-1] if history else manifest.digest(stored["code_sha256"])):  # code of every resumed run
        stored["resumed_with_code"] = history + [code]
    io_utils.atomic_write_json(contract_path, stored)
    done, statuses = resume(record_path)
    counters = Counter({k: v for k, v in (json.loads(report_path.read_text())["counters"]
                                          if report_path.exists() else {}).items()
                        if not k.startswith("observation:")})
    counters.update(statuses)
    assets = json.loads(assets_path.read_text()) if assets_path.exists() else {}
    scenes = {(r["dataset"], r["scene_id"]): r for r in map(json.loads, scenes_path.read_text().splitlines())}
    with gzip.open(inputs, "rt") as stream:
        total = sum(1 for _ in stream)
    size = sum(1 for i in range(total) if i * args.shards // total == args.shard_index)
    started, opened = time.time(), set()
    rows = (r for i, r in enumerate(manifest.read_jsonl_gz(inputs))
            if i * args.shards // total == args.shard_index and r["obs_id"] not in done)
    if args.limit:
        rows = itertools.islice(rows, args.limit)
    for scene_id, group in itertools.groupby(rows, key=lambda r: r["scene_id"]):
        if opened and args.dataset == "b1k":  # Isaac Sim keeps GPU memory of a closed scene: one B1K scene per run
            break
        if scene_id in opened:
            raise ValueError(f"{inputs}: observations of scene {scene_id} are not contiguous")
        opened.add(scene_id)
        session, scene = open_scene(args.dataset, scenes[(args.dataset, scene_id)])
        recorded = assets.setdefault(scene_id, {})
        for key, value in json.loads(json.dumps(scene.assets)).items():  # fields added later are filled in
            if recorded.setdefault(key, value) != value:
                raise ValueError(f"{scene_id}: scene assets changed since this output was started")
        io_utils.atomic_write_json(assets_path, assets)
        print(json.dumps({"event": "open_scene", "scene": scene_id}), flush=True)
        try:
            for row in group:
                record = collect_observation(session, scene, row, output, counters)
                counters[f"observation:{record['status']}"] += 1
                with gzip.open(record_path, "at", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                done.add(row["obs_id"])
                io_utils.atomic_write_json(report_path, {"status": "collecting", "observations": len(done),
                                                         "counters": dict(counters),
                                                         "elapsed_s": time.time() - started})
        finally:
            session.close()
    io_utils.atomic_write_json(report_path, {"status": "complete" if len(done) == size else "partial",
                                             "observations": len(done), "shard_observations": size,
                                             "counters": dict(counters), "elapsed_s": time.time() - started})
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(GEOMETRY_SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new observations (smoke tests)")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        parser.error("invalid shard")
    run(args)


if __name__ == "__main__":
    main()
