"""Unseen meta inputs: fresh start poses in every test/unseen scene, fine and coarse programs for each.

    python -m pipeline.meta.propose poses --dataset r2r --scenes 2azQ1b91cZZ ... --count 150 --output "<root>/inputs"
    python -m pipeline.meta.propose inputs --dataset r2r --output "<root>/inputs"

`poses` opens a scene as the collector does and keeps proposed start poses that have a support surface and,
for the tallest body at the smallest radius, a free start outside any solid; none is within 0.5 m and 45 deg
of a kept one on the same floor. Proposals from every navigable anchor point (uniform heading):
  scene_random    the anchor itself
  expert_path     a point on an R2R reference path facing the next waypoint +-30 deg (scenes with paths)
  contact_target  when the anchor's 6 m forward trace hits the scene: the starts from which a prior program
                  (one atom; four atoms; two, three, five or six atoms) reaches that contact with a chosen
                  forward atom at 20-80% of it. A proposal only: the collector decides every answer.
Writes poses/<dataset>/<scene>.json; a scene with that file is skipped.

`inputs` (no simulator) takes the poses of every scene in hashed order and gives each the next camera
configuration in turn (B1K without the 1.25 m height, as its seen scenes), a fine bank from the frozen
pipeline/assets/fine_action_prior.json (prior draws, coverage programs, local edits; each as its atomic list
and up to two merged lists) and COARSE coarse programs drawn from the frozen seen inputs of the dataset
(one count per input program); a coarse list that is already a fine list of the pose is dropped and a program
left with none is drawn again, so a command list has one representation. Writes <dataset>.jsonl.gz in the
format of pipeline.meta.manifest, scenes.jsonl (every test/unseen scene) and manifest.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
import functools
import gzip
import json
import math
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq

from pipeline import config, io_utils
from pipeline.meta import geometry as G, manifest

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "pipeline/assets/fine_action_prior.json"
PARTITIONS = ROOT / "pipeline/assets/scene_partitions.v1.json"
# Coarse command lists of the frozen 2026-10-12 seen inputs per dataset: sorted keys, counts, source sha256.
COARSE_POOL = ROOT / "data/seen_meta_gt3/inputs/coarse_pool.json"
SEED = "unseen-meta-v1"
ORIGIN = {"test_unseen": SEED, "train_seen": "seen-meta-propose-v1"}  # source origin of proposed rows
HFOVS = (79., 90., 110.)
HEIGHTS = {"r2r": (.5, 1., 1.25, 1.5), "gs": (.5, 1., 1.25, 1.5), "b1k": (.5, 1., 1.5)}
FINE_LIMIT = {"r2r": 24, "gs": 24, "b1k": 12}  # programs per fine bank (B1K collection is the costly one)
COARSE = 6
RADIUS = .10
HORIZONS = {1: .1, 2: .1, 3: .1, 4: .5, 5: .1, 6: .1}  # prior draw lengths
ATOM = {"F": ("forward", .25), "L": ("turn", -15.), "R": ("turn", 15.)}


# -- scenes ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=2)
def scene_rows(partition="test_unseen") -> dict:
    """Every scene of a partition with the source split its assets are listed under."""
    partitions = json.loads(PARTITIONS.read_text())["datasets"]
    with gzip.open(Path(config.R2R_TRAIN_EPISODES).parents[1] / "val_unseen/val_unseen.json.gz", "rt") as stream:
        val_unseen = {Path(e["scene_id"]).stem for e in json.load(stream)["episodes"]}
    gs = {e["scene_id"]: split for split in ("train", "val")
          for e in json.loads((Path(config.GS_ROOT) / "splits" / f"{split}.json").read_text())["scenes"]}
    split = {"r2r": lambda s: "train" if partition == "train_seen" else "val_unseen" if s in val_unseen else "test",
             "gs": gs.get, "b1k": lambda s: "train"}
    return {(d, s): {"dataset": d, "scene_id": s, "partition": partition.replace("_", "/"), "source_split": split[d](s)}
            for d in ("r2r", "gs", "b1k") for s, v in sorted(partitions[d].items()) if v["partition"] == partition}


@functools.lru_cache(maxsize=1)
def expert_paths() -> dict:
    """R2R VLN-CE reference paths by scene (the splits that publish them)."""
    paths = {}
    for split in ("train", "val_seen", "val_unseen"):
        with gzip.open(Path(config.R2R_TRAIN_EPISODES).parents[1] / split / f"{split}.json.gz", "rt") as stream:
            for episode in json.load(stream)["episodes"]:
                paths.setdefault(Path(episode["scene_id"]).stem, []).append(episode["reference_path"])
    return paths


# -- poses ----------------------------------------------------------------------------------------

def start_floor(session, scene, dataset, position, body):
    """The start floor under a pose when the tallest body starts free there at RADIUS, else None."""
    from pipeline.meta.collect import start_support
    support = start_support(session, scene, dataset, np.asarray(position, dtype=float))
    if support is None:
        return None
    y0, xz = support["ground_y"], np.asarray(position, dtype=float)[[0, 2]]
    if scene.inside_solid([xz[0], y0 + (body["ground_clearance_m"] + body["top_m"]) / 2, xz[1]]):
        return None
    obstacles = scene.obstacles(xz - .5, xz + .5, y0 + body["ground_clearance_m"], y0 + body["top_m"])
    return None if obstacles.blocked(xz, RADIUS) else y0


def contact_arc(scene, xz, yaw, y0, body):
    """Travel of the body centre to the first contact of a 6 m forward, or None."""
    end = xz + 6. * G.heading(yaw)
    obstacles = scene.obstacles(np.minimum(xz, end) - .5, np.maximum(xz, end) + .5,
                                y0 + body["ground_clearance_m"], y0 + body["top_m"])
    result = G.execute(obstacles, xz, yaw, [{"type": "forward", "m": 6.}], RADIUS)
    return result["contact"]["travel_in_atom_m"] if result.get("collision") else None


def back_project(xz, yaw, arc, pattern, index, fraction):
    """The start from which atom `index` (a forward) of the pattern reaches the contact at `fraction` of it."""
    start_yaw = yaw + sum(math.radians(ATOM[t][1]) for t in pattern[:index] if t != "F")
    before = np.array(G.poses([0., 0.], start_yaw, atoms(pattern[:index]))[-1][:2]) if index else np.zeros(2)
    return xz + arc * G.heading(yaw) - (before + fraction * ATOM["F"][1] * G.heading(yaw)), start_yaw


def expert_pose(session, rng, paths):
    path = rng.choice(paths)
    index = rng.randrange(len(path) - 1)
    a, b = np.asarray(path[index], dtype=float), np.asarray(path[index + 1], dtype=float)
    point = a + (b - a) * rng.random()
    snapped = np.asarray(session.pathfinder.snap_point(point), dtype=float)
    if not np.isfinite(snapped).all() or np.linalg.norm((snapped - point)[[0, 2]]) > .2:
        return None
    return snapped, math.atan2(-(b - a)[0], -(b - a)[2]) + math.radians(rng.uniform(-30., 30.))


def on_floor(session, dataset, position):
    """GS: the pose at the height of the source floor level covering it (nearest in height). The GS sampler
    puts a navmesh point at the level nearest in height, which can be a small coplanar shell (a rug) that
    does not cover it, and the collector's support rule then finds no floor there."""
    if dataset != "gs":
        return position
    import shapely
    point = shapely.Point(float(position[0]), float(position[2]))
    levels = [float(level.ground_y_m) for level in session._collision._ground_levels
              if abs(float(level.ground_y_m) - position[1]) <= config.NAV_Y_DELTA_M and shapely.covers(level.support, point)]
    return np.array([position[0], min(levels, key=lambda y: abs(y - position[1])), position[2]]) if levels else position


def near(kept, position, yaw):
    return any(math.dist(k["position"][::2], position[::2]) < .5 and abs(k["position"][1] - position[1]) < .5
               and abs((math.degrees(k["yaw_rad"] - yaw) + 180) % 360 - 180) < 45. for k in kept)


def propose_scene(dataset, scene_id, count, output, partition="test_unseen"):
    from pipeline.meta.collect import open_scene
    path = output / "poses" / dataset / f"{scene_id}.json"
    if path.exists():
        return
    prior = json.loads(PRIOR.read_text())["patterns"]
    seed = int(manifest.digest([SEED, dataset, scene_id])[:12], 16)
    rng = random.Random(seed)
    session, scene = open_scene(dataset, scene_rows(partition)[(dataset, scene_id)])
    try:
        session.pathfinder.seed(seed % 2 ** 31)
        sample = getattr(session, "sample_navigable_point", session.pathfinder.get_random_navigable_point)
        paths = expert_paths().get(scene_id) if dataset == "r2r" else None
        body = config.fine_body(max(HEIGHTS[dataset]))
        kept, counters, anchors = [], Counter(), 0
        while len(kept) < count and anchors < 40 * count:
            anchors += 1
            position = np.asarray(sample(), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                continue
            position = on_floor(session, dataset, position)
            yaw = rng.uniform(-math.pi, math.pi)
            proposals = [(position, yaw, "scene_random", None, None)]
            expert = expert_pose(session, rng, paths) if paths else None
            if expert:
                proposals.append((*expert, "expert_path", None, None))
            y0 = start_floor(session, scene, dataset, position, body)
            arc = None if y0 is None else contact_arc(scene, position[[0, 2]], yaw, y0, body)
            if arc is not None and arc > .15:
                length = rng.choices([2, 3, 5, 6], [.35, .45, .1, .1])[0]
                for n, index in ((1, 0), (4, anchors % 4), (length, rng.randrange(length))):
                    pattern = draw(prior, rng, n, index)
                    xz, heading = back_project(position[[0, 2]], yaw, arc, pattern, index, rng.uniform(.2, .8))
                    proposals.append((np.array([xz[0], position[1], xz[1]]), heading, "contact_target", pattern, index))
            for p, y, kind, target, index in proposals:
                counters[f"{kind}:proposed"] += 1
                p = on_floor(session, dataset, p)
                if len(kept) >= count or start_floor(session, scene, dataset, p, body) is None or near(kept, p, y):
                    continue
                counters[f"{kind}:kept"] += 1
                kept.append({"position": [float(v) for v in p], "yaw_rad": float(y), "pose_proposal": kind,
                             "target": target, "contact_index": index})
    finally:
        session.close()
    io_utils.atomic_write_json(path, {"dataset": dataset, "scene_id": scene_id, "count": count, "anchors": anchors,
                                      "counters": dict(counters), "poses": kept})
    print(json.dumps({"scene": scene_id, "poses": len(kept), "anchors": anchors, **counters}), flush=True)


# -- programs -------------------------------------------------------------------------------------

def atoms(pattern):
    return [{"type": "forward", "m": .25} if t == "F" else {"type": "turn", "deg": ATOM[t][1]} for t in pattern]


def allowed(pattern):
    return "F" in pattern and "FFFFF" not in pattern and all(len(run) <= 3 for run in pattern.split("F"))


def draw(prior, rng, length=None, contact=None):
    """A pattern observed in the training trajectories, weighted by its frequency (optionally a forward at
    `contact`)."""
    length = length or rng.choices(list(HORIZONS), list(HORIZONS.values()))[0]
    choices = [(p, w) for p, w in prior[str(length)].items()
               if w > 0 and allowed(p) and (contact is None or p[contact] == "F")]
    return rng.choices([p for p, _ in choices], [w for _, w in choices])[0]


def local_edits(pattern):
    edits = set()
    for i, token in enumerate(pattern):
        if token in "LR":
            edits.add(pattern[:i] + ("R" if token == "L" else "L") + pattern[i + 1:])
        if i + 1 < len(pattern) and (token == "F") != (pattern[i + 1] == "F"):
            edits.add(pattern[:i] + pattern[i + 1] + token + pattern[i + 2:])
        if token == "F" and pattern.count("F") > 1:
            edits.add(pattern[:i] + pattern[i + 1:])
    if len(pattern) < 6:
        edits.add(pattern + "F")
    return edits


def bank(prior, seed, target, limit):
    """Pattern -> proposal source: the target, fixed prior draws (repeats kept once), the coverage programs
    and local edits of drawn ones (up to 3, or 1 for small banks)."""
    rng, patterns = random.Random(seed), {}
    if target:
        patterns[target] = "contact_target"
    for _ in range(limit - (6 if limit == 24 else 3)):
        patterns.setdefault(draw(prior, rng), "prior")
    for pattern in ("F", "FF") if limit == 24 else (rng.choice(("F", "FF")),):
        patterns.setdefault(pattern, "coverage")
    parents, added = list(patterns), 0
    rng.shuffle(parents)
    for parent in parents:
        edits = sorted(p for p in local_edits(parent) if p not in patterns and allowed(p)
                       and prior[str(len(p))].get(p, 0) > 0)
        if edits:
            patterns[rng.choice(edits)] = "local_edit"
            added += 1
            if len(patterns) >= limit or added == (3 if limit == 24 else 1):
                break
    return patterns


def command_lists(pattern):
    """Every split of the pattern's runs into commands of one to three equal atoms; the first is atomic."""
    out = []

    def visit(i, commands):
        if i == len(pattern):
            out.append(commands)
        for size in range(1, 4):
            if i + size > len(pattern) or pattern[i:i + size] != pattern[i] * size:
                break
            kind, unit = ATOM[pattern[i]]
            visit(i + size, commands + [{"type": kind, "m" if kind == "forward" else "deg": unit * size}])
    visit(0, [])
    return out


def canonical(actions) -> str:
    return json.dumps([{k: float(v) if k in ("m", "deg") else v for k, v in a.items()} for a in actions],
                      sort_keys=True)


def coarse_pool(dataset):
    """Coarse command lists of the frozen seen inputs, one count per input program."""
    pool = json.loads(COARSE_POOL.read_text())[dataset]
    return pool["keys"], pool["counts"]


def programs_of(pose, obs_id, dataset, prior, pool):
    seed = int(obs_id.split("-")[1], 16)
    fine, lists = [], set()
    for pattern, source in bank(prior, seed, pose["target"], FINE_LIMIT[dataset]).items():
        atomic, *merged = command_lists(pattern)
        merged = sorted(merged, key=lambda c: manifest.digest([obs_id, c]))[:2]
        fine += [manifest.program(atomic, "atomic", source)] + [manifest.program(c, "merged", source) for c in merged]
        lists |= {canonical(c) for c in [atomic] + merged}
    rng, drawn, coarse = random.Random(seed), set(), []
    while len(drawn) < COARSE:
        key = rng.choices(*pool)[0]
        kept = [json.loads(c) for c in json.loads(key) if c not in lists]
        if key in drawn or not kept:
            continue
        drawn.add(key)
        coarse += [manifest.program(c, "coarse", "seen_coarse_inputs") for c in kept]
    return manifest.merge_programs(fine + coarse)


def write_inputs(dataset, output, partition="test_unseen", base=None, missing_from=None):
    """The inputs of the partition's scenes with poses; with `base` (a frozen inputs folder), its rows and scene
    rows come first, unchanged, and only scenes it lacks are added; with `missing_from` (a meta root), only the
    rows that root has no ok observation for (to collect just those and reuse the rest)."""
    prior = json.loads(PRIOR.read_text())["patterns"]
    pool = coarse_pool(dataset)
    cameras = [(h, f) for h in HEIGHTS[dataset] for f in HFOVS]
    kept = list(manifest.read_jsonl_gz(base / f"{dataset}.jsonl.gz")) if base else []
    known = {r["scene_id"] for r in kept}
    rows, scenes = [], {}
    for (d, scene_id), _ in scene_rows(partition).items():
        path = output / "poses" / d / f"{scene_id}.json"
        if d != dataset or not path.exists() or scene_id in known:
            continue
        poses = sorted(json.loads(path.read_text())["poses"],
                       key=lambda p: manifest.digest([dataset, scene_id, p["position"], p["yaw_rad"]]))
        scenes[scene_id] = len(poses)
        for i, pose in enumerate(poses):
            height, hfov = cameras[i % len(cameras)]
            obs_id = manifest.observation_id(dataset, scene_id, pose["position"], pose["yaw_rad"], height, hfov)
            rows.append({"obs_id": obs_id, "dataset": dataset, "scene_id": scene_id,
                         "pose": {"position": pose["position"], "yaw_rad": pose["yaw_rad"]},
                         "camera_height_m": height, "hfov_deg": hfov,
                         "programs": programs_of(pose, obs_id, dataset, prior, pool),
                         "sources": [{"origin": ORIGIN[partition], "record_uid": obs_id,
                                      "pose_proposal": pose["pose_proposal"]}]})
    if missing_from:
        observations = pq.read_table(missing_from / "observations" / f"{dataset}.parquet", columns=["obs_id", "status"])
        ok = {o for o, s in zip(*(observations[c].to_pylist() for c in ("obs_id", "status"))) if s == "ok"}
        kept, rows = [r for r in kept if r["obs_id"] not in ok], [r for r in rows if r["obs_id"] not in ok]
    manifest.write_jsonl_gz(output / f"{dataset}.jsonl.gz", kept + rows)  # scenes stay contiguous
    scene_list = [json.loads(line) for line in (base / "scenes.jsonl").read_text().splitlines()] if base else []
    listed = {(r["dataset"], r["scene_id"]) for r in scene_list}
    scene_list += [r for (d, s), r in scene_rows(partition).items()
                   if (d, s) not in listed and (not base or (d == dataset and s in scenes))]
    scenes_path = output / "scenes.jsonl"  # the same for every dataset; replaced whole (a collector may read it)
    scenes_path.with_name("tmp.scenes.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in scene_list))
    scenes_path.with_name("tmp.scenes.jsonl").replace(scenes_path)
    summary_path = output / "manifest.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"schema": manifest.SCHEMA}
    summary.setdefault("datasets", {})[dataset] = {
        "observations": len(kept) + len(rows), "scenes": scenes,
        **({"base": {"path": str(base), "observations": len(kept), "scenes": len(known)}} if base else {}),
        **({"missing_from": str(missing_from)} if missing_from else {}),
        "pose_proposals": dict(Counter(r["sources"][0]["pose_proposal"] for r in rows)),
        "cameras": dict(Counter(f"{r['camera_height_m']:g}m/{r['hfov_deg']:g}deg" for r in rows)),
        "programs": dict(Counter(p["proposal_source"] for r in rows for p in r["programs"])),
        "command_lists": dict(Counter(c["representation"] for r in rows for p in r["programs"] for c in p["commands"])),
        "fine_action_prior_sha256": io_utils.sha256_file(PRIOR),
        "coarse_pool": {"source_sha256": json.loads(COARSE_POOL.read_text())[dataset]["source_sha256"],
                        "pool_sha256": io_utils.sha256_file(COARSE_POOL),
                        "programs": sum(pool[1]), "distinct": len(pool[0])},
        "inputs_sha256": io_utils.sha256_file(output / f"{dataset}.jsonl.gz")}
    io_utils.atomic_write_json(summary_path, summary)
    print(json.dumps({dataset: summary["datasets"][dataset]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("poses", "inputs"))
    parser.add_argument("--dataset", choices=tuple(HEIGHTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", help="poses: the scenes (default: every unseen scene of the dataset)")
    parser.add_argument("--count", type=int, default=150, help="poses: poses per scene")
    parser.add_argument("--partition", choices=tuple(ORIGIN), default="test_unseen")
    parser.add_argument("--base", type=Path, help="inputs: a frozen inputs folder whose rows are kept")
    parser.add_argument("--missing-from", type=Path, help="inputs: only rows this meta root has no ok observation for")
    args = parser.parse_args()
    if args.step == "inputs":
        write_inputs(args.dataset, args.output, args.partition, args.base, args.missing_from)
        return
    for scene_id in args.scenes or [s for d, s in scene_rows(args.partition) if d == args.dataset]:
        propose_scene(args.dataset, scene_id, args.count, args.output, args.partition)


if __name__ == "__main__":
    main()
