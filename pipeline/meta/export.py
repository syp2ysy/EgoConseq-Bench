"""Export collected meta records into the single final format, validated before it replaces anything.

    python -m pipeline.meta.export --work "<root>/_work" --output "<root>" [--partial]
    python -m pipeline.meta.export --output "<root>" --recover   # after an interrupted export
    python -m pipeline.meta.export --reuse "<source-root>" [re-collected root ...] --datasets gs --output "<new-root>"

Exports the datasets of the shards in --work; every dataset must be complete (all shards finished, same
inputs and starting code) unless --partial. Image hashes are checked while the records are written to
<output>/_export.staging, where the images are hard links; the tables are then checked (unique ids,
complete references, row counts). The commit is a journal written before anything is replaced: the files
of these datasets are swapped in (earlier ones kept in _export.previous until the commit is recorded),
then the exported shards are deleted (a --partial export keeps them). Exports of one output run one at a
time; an interrupted one blocks the next until --recover completes or discards it:

<root>/
  images/<dataset>/<obs_id>.png    initial RGB
  future_views/<sha256>.png        C1 option views (content-addressed, shared by datasets)
  observations/<dataset>.parquet   every observation: sources, input and adopted pose, camera, target, status
  executions/<dataset>.parquet     every geometric execution (program x radius) and its contact
  tasks/<TASK>/<dataset>.parquet   one row per GT item (every task table exists, possibly empty)
  dataset_card.json                GT contract, definitions, per-dataset counts and collection runs
A table over all datasets: pyarrow.parquet.read_table("<root>/tasks/A1").

Only correct GT is written (valid start, unique visible contact for A3, validated target);
QA quality measures (evidence, boundary margins, distance change, view difference) are columns.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pipeline import io_utils
from pipeline.meta import manifest
from pipeline.meta.collect import GT_CONFIG, RECORDS, TASKS_BY_DATASET

PARTITIONS = json.loads((Path(__file__).parents[1] / "assets/scene_partitions.v1.json").read_text())["datasets"]
HISTORY = ("resumed_with_code", "code_changes", "backfilled_assets")  # version history of a shard's run
STAGING, PREVIOUS, JOURNAL, CARD = "_export.staging", "_export.previous", "_export.journal.json", "dataset_card.json"
AZIMUTH_BOUNDS_DEG =(-172.5, -97.5, -82.5, -7.5, 7.5, 82.5, 97.5, 172.5)
ELEVATION_BOUNDS_DEG = (-7.5, 7.5)
TASKS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1")

FLOATS = pa.list_(pa.float64())
ACTION = pa.struct([("type", pa.string()), ("m", pa.float64()), ("deg", pa.float64())])
POINT = pa.struct([("point_id", pa.string()), ("pixel_xy", pa.list_(pa.int32())), ("world_xyz", FLOATS)])
ENTITY = pa.struct([("instance_id", pa.int64()), ("category", pa.string()), ("source_category", pa.string())])
SOURCE = pa.struct([("origin", pa.string()), ("record_uid", pa.string()), ("pose_proposal", pa.string())])
COMMANDS = [("command_list", pa.int32()), ("representation", pa.string()), ("actions", pa.list_(ACTION)),
            ("n_actions", pa.int32())]
EXECUTION_BASE = [("question_id", pa.string()), ("obs_id", pa.string()), ("execution_id", pa.string()),
                  ("program_id", pa.string()), ("radius_m", pa.float64())] + COMMANDS
OUTCOME_BASE = [("question_id", pa.string()), ("obs_id", pa.string()), ("program_id", pa.string()),
                ("radii_m", FLOATS)] + COMMANDS
MOMENT = [("point_id", pa.string()), ("pixel_xy", pa.list_(pa.int32())), ("camera_position", FLOATS),
          ("camera_yaw_rad", pa.float64())]
CHECKPOINT = [("checkpoint_action", pa.int32()), ("checkpoint_fraction", pa.float64())]
DIRECTION = [("horizontal", pa.string()), ("vertical", pa.string()), ("azimuth_deg", pa.float64()),
             ("elevation_deg", pa.float64()), ("boundary_margin_deg", pa.float64()), ("planar_range_m", pa.float64())]
DISTANCE = [("distance_m", pa.float64()), ("initial_distance_m", pa.float64()), ("distance_change_m", pa.float64())]
SCHEMAS = {
    "observations": pa.schema([
        ("obs_id", pa.string()), ("dataset", pa.string()), ("scene", pa.string()), ("split", pa.string()),
        ("pose_group_id", pa.string()), ("status", pa.string()), ("sources", pa.list_(SOURCE)),
        ("requested_position", FLOATS), ("requested_yaw_rad", pa.float64()), ("floor_y", pa.float64()),
        ("support_method", pa.string()), ("support_category", pa.string()),
        ("image", pa.string()), ("image_sha256", pa.string()), ("width", pa.int32()), ("height", pa.int32()),
        ("camera_height_m", pa.float64()), ("hfov_deg", pa.float64()), ("vfov_deg", pa.float64()),
        ("camera_position", FLOATS), ("camera_yaw_rad", pa.float64()),
        ("body_top_m", pa.float64()), ("ground_clearance_m", pa.float64()),
        ("visible_floor_ratio", pa.float64()), ("valid_depth_ratio", pa.float64()),
        ("dark_pixel_fraction", pa.float64()),
        ("visible_entities", pa.list_(ENTITY)), ("target_status", pa.string()), ("target_instance_id", pa.int64()),
        ("target_category", pa.string()), ("target_source_category", pa.string()),
        ("target_points", pa.list_(POINT))]),
    "executions": pa.schema([
        ("execution_id", pa.string()), ("obs_id", pa.string()), ("program_id", pa.string()),
        ("radius_m", pa.float64()), ("status", pa.string()), ("collision", pa.bool_()),
        ("first_collision_atom", pa.int32()), ("atomic_actions", pa.list_(ACTION)),
        ("contact_travel_m", pa.float64()), ("stop_pose", FLOATS), ("contact_points", pa.list_(FLOATS)),
        ("contact_identity", pa.string()), ("contact_instance_id", pa.int64()), ("contact_category", pa.string())]),
    "A1": pa.schema(EXECUTION_BASE + [("collision", pa.bool_()), ("depth_view_collision", pa.bool_()),
                                      ("corridor_coverage", pa.float64()), ("corridor_out_of_frame", pa.int32()),
                                      ("corridor_occluded", pa.int32()), ("corridor_samples", pa.int32())]),
    "A2": pa.schema(EXECUTION_BASE + [("first_collision_action", pa.int32()), ("forward_commands", pa.int32()),
                                      ("contact_progress_m", pa.float64()), ("command_length_m", pa.float64()),
                                      ("depth_view_contact_command", pa.int32())]),
    "A3": pa.schema(EXECUTION_BASE + [("category", pa.string()), ("source_category", pa.string()),
                                      ("instance_id", pa.int64()), ("visible_points", pa.int32())]),
    "B1": pa.schema(OUTCOME_BASE + MOMENT + DISTANCE),
    "B2": pa.schema(OUTCOME_BASE + MOMENT + DIRECTION),
    "B3": pa.schema(OUTCOME_BASE + MOMENT + CHECKPOINT + DISTANCE),
    "A4": pa.schema(OUTCOME_BASE + MOMENT + CHECKPOINT + DIRECTION),
    "C1": pa.schema(OUTCOME_BASE + [("family_id", pa.string()), ("options", pa.list_(pa.string())),
                                    ("option_program_ids", pa.list_(pa.string())), ("answer_index", pa.int32()),
                                    ("min_block_l1", pa.float64())]),
}
DEFINITIONS = {
    "motion": "frozen scene; each observation has one start floor y0 (floor_y), shared by every radius, checkpoint "
              "and end; the body moves horizontally at y0 (a drop does not lower it, a slope or step entering the "
              "body band is an obstacle) and turns in place",
    "pose": "input pose x, z, yaw kept; floor_y is the support surface under the input pose; camera at floor_y + "
            "camera_height_m for the start, every checkpoint and every end",
    "body": "vertical cylinder of radius_m over (floor_y + ground_clearance_m, floor_y + body_top_m), "
            "body_top_m = camera_height_m + 0.10 m",
    "execution": "atomic program (0.25 m forwards, 15 deg turns) swept against the scene mesh clipped to the body band "
                 "(R2R: MP3D scene mesh; GS: SAGE-3D collision mesh); a start within the radius of an obstacle is "
                 "invalid; a collision is an overlap deeper than 0.1 mm (touching is not), and the first one ends "
                 "the execution; turns never collide",
    "actions": "public command list [{type: forward|turn, m, deg}]; positive deg = right turn; command_list indexes "
               "the program's command lists (coarse / atomic / merged descriptions of the same path)",
    "A2.first_collision_action": "1-based index of the command during whose forward motion the body first overlaps "
                                 "an obstacle (an overlap that begins exactly at a command's end belongs to that "
                                 "command)",
    "A3": "category of the single labelled instance touched at the first contact, with its contact region visible "
          "in the initial image",
    "radii_m": "B/A4/C1 answers do not depend on the radius; radii_m lists the radii at which the program "
               "(for C1: all four programs) is collision-free",
    "distance_m": "3D distance from the camera optical centre at the moment to the target point",
    "direction": "target in the camera frame at the moment: horizontal front/right/rear/left within +-7.5 deg of the "
                 "axis, diagonals between; vertical above/below beyond +-7.5 deg elevation, else level",
    "checkpoint": "after command checkpoint_action has executed checkpoint_fraction of its forward distance",
    "C1": "terminal views of 4 programs collision-free at radii_m with distinct ends and pixels; answer_index is "
          "0-based in options",
    "split": "scene partition; pose_group_id groups observations at the same 1 cm / 0.5 m level / 1 deg pose",
    "start_inside_solid": "the body centre lies inside a closed (watertight) convex collision part (GS, B1K); such "
                          "observations have no image and no GT. Open parts (pieces of a room shell) enclose nothing "
                          "but still collide. Known limitation: a box-shaped room modelled as one closed part still "
                          "counts",
}


def _actions(commands):
    return [{"type": a["type"], "m": a.get("m"), "deg": a.get("deg")} for a in commands]


def _margin(azimuth, elevation):
    return min(min(abs(azimuth - b) for b in AZIMUTH_BOUNDS_DEG), min(abs(elevation - b) for b in ELEVATION_BOUNDS_DEG))


def table_file(name, dataset):
    return Path("tasks" if name in TASKS else "", name, f"{dataset}.parquet")


def dataset_files(dataset):
    return [Path("images", dataset)] + [table_file(name, dataset) for name in SCHEMAS]


class Tables:
    """Every table of every exported dataset is created up front, so a table without rows is written empty."""

    def __init__(self, root: Path, datasets):
        self.root, self.counts = root, Counter()
        self.rows = {(name, dataset): [] for name in SCHEMAS for dataset in datasets}
        self.writers = {}
        for name, dataset in self.rows:
            self.path(name, dataset).parent.mkdir(parents=True, exist_ok=True)
            self.writers[name, dataset] = pq.ParquetWriter(self.path(name, dataset), SCHEMAS[name])

    def path(self, name, dataset):
        return self.root / table_file(name, dataset)

    def add(self, name, row, dataset):
        self.rows[name, dataset].append(row)
        self.counts[f"{name}/{dataset}"] += 1
        if len(self.rows[name, dataset]) >= 50000:
            self.flush((name, dataset))

    def flush(self, key):
        if self.rows[key]:
            self.writers[key].write_table(pa.Table.from_pylist(self.rows[key], schema=SCHEMAS[key[0]]))
            self.rows[key] = []

    def close(self):
        for key in self.writers:
            self.flush(key)
            self.writers[key].close()


def _questions(record):
    from post_QA.categories import a3_answer

    obs, dataset = record["obs_id"], record["dataset"]
    supported = TASKS_BY_DATASET[dataset]
    programs = {p["program_id"]: p for p in record["programs"]}
    points = {p["point_id"]: p for p in (record.get("target") or {}).get("points", [])}

    def commands(program_id, k):
        value = programs[program_id]["commands"][k]
        return {"command_list": k, "representation": value["representation"],
                "actions": _actions(value["actions"]), "n_actions": len(value["actions"])}

    def executed(task, execution, k):
        return {"question_id": f"{task}-" + manifest.digest([task, obs, execution["execution_id"], k])[:20],
                "obs_id": obs, "execution_id": execution["execution_id"], "program_id": execution["program_id"],
                "radius_m": execution["radius_m"], **commands(execution["program_id"], k)}

    for execution in record["executions"]:
        if execution["status"] != "ok":
            continue
        identity = execution.get("contact_identity") or {}
        for k, facts in enumerate(execution["commands"]):
            corridor = facts["evidence"]["corridor"]
            yield "A1", {**executed("A1", execution, k), "collision": execution["collision"],
                         "depth_view_collision": facts["evidence"]["depth_view_collision"],
                         "corridor_coverage": corridor["raw_coverage"],
                         "corridor_out_of_frame": corridor["out_of_frame_samples"],
                         "corridor_occluded": corridor["occluded_samples"], "corridor_samples": corridor["total_samples"]}
            if not execution["collision"]:
                continue
            listed = programs[execution["program_id"]]["commands"][k]["actions"]
            yield "A2", {**executed("A2", execution, k), "first_collision_action": facts["first_collision_command"],
                         "forward_commands": sum(a["type"] == "forward" for a in listed),
                         "contact_progress_m": facts["contact_progress_m"],
                         "command_length_m": facts["command_length_m"],
                         "depth_view_contact_command": facts["evidence"]["depth_view_contact_command"]}
            if "A3" in supported and identity.get("status") == "certified":
                yield "A3", {**executed("A3", execution, k),
                             "category": a3_answer(record, {"instance_id": identity["instance_id"],
                                                            "category": identity["category"]}),
                             "source_category": identity["category"], "instance_id": identity["instance_id"],
                             "visible_points": identity["evidence"]["visible_points"]}
    for outcome in record.get("outcomes", []):
        pid = outcome["program_id"]
        for k in range(len(programs[pid]["commands"])):
            moments = [("B1", "B2", (), "end", outcome["end"])] + [
                ("B3", "A4", (("checkpoint_action", c["action_index"]), ("checkpoint_fraction", c["fraction"])),
                 f"a{c['action_index']}f{c['fraction']}", c) for c in outcome["checkpoints"] if c["command_list"] == k]
            for distance_task, direction_task, checkpoint, where, moment in moments:
                camera = moment["camera"]
                for rel in moment["relations"]:
                    row = {"obs_id": obs, "program_id": pid, "radii_m": outcome["radii_m"], **commands(pid, k),
                           **dict(checkpoint), "point_id": rel["point_id"],
                           "pixel_xy": points[rel["point_id"]]["pixel_xy"],
                           "camera_position": camera["position_world"], "camera_yaw_rad": camera["yaw_rad"]}
                    key = [obs, pid, k, rel["point_id"], where]
                    if distance_task in supported:
                        yield distance_task, {**row, "question_id": f"{distance_task}-" + manifest.digest(
                                                  [distance_task, *key])[:20],
                                              "distance_m": rel["distance_m"],
                                              "initial_distance_m": rel["initial_distance_m"],
                                              "distance_change_m": abs(rel["distance_m"] - rel["initial_distance_m"])}
                    if direction_task in supported and "horizontal" in rel:
                        yield direction_task, {**row, "question_id": f"{direction_task}-" + manifest.digest(
                                                   [direction_task, *key])[:20],
                                               "horizontal": rel["horizontal"], "vertical": rel["vertical"],
                                               "azimuth_deg": rel["azimuth_deg"], "elevation_deg": rel["elevation_deg"],
                                               "boundary_margin_deg": _margin(rel["azimuth_deg"], rel["elevation_deg"]),
                                               "planar_range_m": rel["planar_range_m"]}
    for family in record.get("c1", []):
        pid = family["query_program_id"]
        for k in range(len(programs[pid]["commands"])):
            yield "C1", {"question_id": "C1-" + manifest.digest(["C1", obs, family["family_id"], k])[:20],
                         "obs_id": obs, "program_id": pid, "radii_m": family["radii_m"], **commands(pid, k),
                         "family_id": family["family_id"], "options": [o["image"] for o in family["options"]],
                         "option_program_ids": [o["program_id"] for o in family["options"]],
                         "answer_index": family["answer_index"], "min_block_l1": family["min_block_l1"]}


def _observation(record, image):
    dataset, obs = record["dataset"], record["obs_id"]
    requested = record["requested_pose"]
    pose = record.get("pose") or requested
    support = record.get("start_support") or {}
    target = record.get("target") or {}
    quality = record.get("image_quality") or {}
    return {
        "obs_id": obs, "dataset": dataset, "scene": record["scene_id"],
        "split": PARTITIONS[dataset][record["scene_id"]]["partition"],
        "pose_group_id": manifest.digest([dataset, record["scene_id"], round(pose["position"][0], 2),
                                          round(pose["position"][2], 2), round(pose["position"][1] * 2),
                                          round(math.degrees(pose["yaw_rad"]) % 360)])[:20],
        "status": record["status"], "sources": record["sources"],
        "requested_position": requested["position"], "requested_yaw_rad": requested["yaw_rad"],
        "floor_y": support.get("ground_y"), "support_method": support.get("method"),
        "support_category": (support.get("evidence") or {}).get("category"),
        "image": image, "image_sha256": record.get("image_sha256"),
        "width": (record.get("resolution") or [None, None])[0], "height": (record.get("resolution") or [None, None])[1],
        "camera_height_m": record["camera_height_m"], "hfov_deg": record["hfov_deg"], "vfov_deg": record["vfov_deg"],
        "camera_position": (record.get("camera") or {}).get("position_world"), "camera_yaw_rad": pose["yaw_rad"],
        "body_top_m": record["body"]["top_m"], "ground_clearance_m": record["body"]["ground_clearance_m"],
        "visible_floor_ratio": quality.get("visible_floor_ratio"), "valid_depth_ratio": quality.get("valid_depth_ratio"),
        "dark_pixel_fraction": quality.get("dark_pixel_fraction"),
        "visible_entities": record.get("visible_entities") or [],
        "target_status": record.get("target_status"), "target_instance_id": target.get("instance_id"),
        "target_category": target.get("category"), "target_source_category": target.get("source_category"),
        "target_points": target.get("points") or []}


def _execution(obs, execution, programs):
    contact = execution.get("contact") or {}
    identity = execution.get("contact_identity") or {}
    return {"execution_id": execution["execution_id"], "obs_id": obs, "program_id": execution["program_id"],
            "radius_m": execution["radius_m"], "status": execution["status"],
            "collision": execution.get("collision"), "first_collision_atom": execution.get("first_collision_atom"),
            "atomic_actions": _actions(programs[execution["program_id"]]["atomic_actions"]),
            "contact_travel_m": contact.get("travel_m"), "stop_pose": contact.get("stop_pose"),
            "contact_points": contact.get("points"),
            "contact_identity": identity.get("status") and (identity["status"] if identity["status"] == "certified"
                                                            else f"withheld:{identity.get('reason')}"),
            "contact_instance_id": identity.get("instance_id"), "contact_category": identity.get("category")}


def dataset_gt(gt, dataset):
    """The part of a GT contract that defines one dataset's GT (other datasets' entries do not change it)."""
    view = json.loads(json.dumps(gt))
    view["geometry"]["sources"] = gt["geometry"]["sources"][dataset]
    rule = gt["start_support"][dataset]
    view["start_support"] = gt["start_support"][rule] if isinstance(rule, str) and rule in gt["start_support"] else rule
    view["tasks_by_dataset"] = gt["tasks_by_dataset"][dataset]
    return view


def check_runs(shards, partial=False):
    """Shard contracts by directory; a dataset is exported only from one complete, consistent run."""
    contracts, runs, assets = {}, {}, {}
    if not shards:
        raise ValueError("no collected shards to export")
    for shard in shards:
        contract = json.loads((shard / "collection.json").read_text())
        status = json.loads((shard / "report.json").read_text())["status"]
        if dataset_gt(contract["gt"], contract["dataset"]) != dataset_gt(GT_CONFIG, contract["dataset"]):
            raise ValueError(f"{shard}: collected under a different GT contract")
        if status != "complete" and not partial:
            raise ValueError(f"{shard}: collection is {status}")
        scene_assets = json.loads((shard / "scene_assets.json").read_text())
        for scene, value in scene_assets.items():
            if assets.setdefault((contract["dataset"], scene), value) != value:
                raise ValueError(f"{contract['dataset']}/{scene}: shards were collected from different scene assets")
        contracts[shard.name] = {"inputs": contract["inputs"], "scenes": contract["scenes"],
                                 "engines": contract["engines"], "code_sha256": manifest.digest(contract["code_sha256"]),
                                 "scene_assets": scene_assets,
                                 **{k: contract[k] for k in HISTORY if k in contract}}
        runs.setdefault(contract["dataset"], []).append(contract)
    for dataset, values in runs.items():
        if len({(c["shards"], c["inputs"]["sha256"], c["scenes"]["sha256"], manifest.digest(c["engines"]),
                 manifest.digest(c["code_sha256"])) for c in values}) != 1:
            raise ValueError(f"{dataset}: shards come from different inputs, shard counts, engines or code")
        if sorted(c["shard_index"] for c in values) != list(range(values[0]["shards"])) and not partial:
            raise ValueError(f"{dataset}: shards are missing or repeated")
    return contracts


def validate(tables: Tables):
    """Unique ids, complete references and row counts of the written tables."""
    def column(name, field):
        return [v for (n, d) in tables.writers if n == name
                for v in pq.read_table(tables.path(n, d), columns=[field]).column(field).to_pylist()]

    for name, dataset in tables.writers:
        if pq.ParquetFile(tables.path(name, dataset)).metadata.num_rows != tables.counts[f"{name}/{dataset}"]:
            raise ValueError(f"{name}/{dataset}: written rows differ from exported rows")
    observations = column("observations", "obs_id")
    executions = column("executions", "execution_id")
    if len(set(observations)) != len(observations) or len(set(executions)) != len(executions):
        raise ValueError("observation or execution ids are not unique")
    known_obs, known_exec = set(observations), set(executions)
    for task in TASKS:
        ids = column(task, "question_id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{task}: question ids are not unique")
        if not set(column(task, "obs_id")) <= known_obs:
            raise ValueError(f"{task}: rows reference unknown observations")
        if task in ("A1", "A2", "A3") and not set(column(task, "execution_id")) <= known_exec:
            raise ValueError(f"{task}: rows reference unknown executions")


def _checked(path, digest):
    if io_utils.sha256_file(path) != digest:
        raise ValueError(f"{path}: image differs from its recorded hash")


@contextlib.contextmanager
def _locked(out: Path):
    """One export (or recovery) of an output at a time, over its checks, commit and cleanup."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".export.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _commit(out: Path, journal: dict):
    """Apply a written journal. Every step checks its own state, so the commit can be rerun after an
    interruption at any point; once the commit is recorded only the cleanup is left."""
    staging, previous = out / STAGING, out / PREVIOUS
    if journal["state"] == "prepared":
        for relative in map(Path, journal["files"]):
            if not (staging / relative).exists():  # already published
                continue
            if relative.as_posix() in journal["replaced"] and not (previous / relative).exists():
                (previous / relative).parent.mkdir(parents=True, exist_ok=True)
                (out / relative).rename(previous / relative)
            (out / relative).parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).rename(out / relative)
        (out / "future_views").mkdir(exist_ok=True)
        for view in sorted((staging / "future_views").glob("*")):  # shared by datasets: keep one verified copy
            target = out / "future_views" / view.name
            if target.exists() and io_utils.sha256_file(target) == view.stem:
                view.unlink()
            else:
                view.replace(target)
        if (out / CARD).exists() and not (previous / CARD).exists():
            previous.mkdir(exist_ok=True)
            shutil.copy2(out / CARD, previous / CARD)
        io_utils.atomic_write_json(out / CARD, journal["card"], durable=True)
        journal["state"] = "committed"
        io_utils.atomic_write_json(out / JOURNAL, journal, durable=True)
    for link in map(Path, journal["links"]):
        link.unlink(missing_ok=True)
    for shard in map(Path, journal["shards"]):  # an exported dataset's images live only in the export now
        shutil.rmtree(shard, ignore_errors=True)
    shutil.rmtree(previous, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    (out / JOURNAL).unlink()


def recover(out: Path):
    """Finish an interrupted export from its journal, or discard one interrupted before its commit."""
    with _locked(out):
        for temporary in [*out.glob(f".{JOURNAL}.*.tmp"), *out.glob(f".{CARD}.*.tmp")]:  # killed while writing
            temporary.unlink()
        if (out / JOURNAL).exists():
            _commit(out, json.loads((out / JOURNAL).read_text()))
            return "committed"
        if (out / STAGING).exists():  # nothing was replaced and the staged images are links: sources are intact
            shutil.rmtree(out / STAGING)
            return "discarded"
        return "nothing to recover"


def reuse(sources, out: Path, datasets):
    """Carry datasets over from other meta roots into a new one without collecting them again (for a GT contract
    change that does not touch their GT). An observation's rows come from the last source that has it, so a root
    of re-collected observations listed after the original replaces their rows there. Tables are written, initial
    images and C1 option views hard-linked; the card entry lists every source root, its GT schema and the
    observations it gave, with all their collection runs. The sources are not changed. Only datasets the output
    does not have yet are written: replacing exported data goes through export(), with its checks and commit."""
    with _locked(out):
        present = [d for d in datasets if any((out / f).exists() for f in dataset_files(d))]
        if present:
            raise SystemExit(f"{out}: already has {', '.join(present)}; reuse only writes new datasets")
        card = json.loads((out / CARD).read_text()) if (out / CARD).exists() else {}
        card.update(schema="seen-meta.v2", gt_contract=GT_CONFIG, definitions=DEFINITIONS)
        for dataset in datasets:
            owner = {}  # obs_id -> the last source that has it
            for i, source in enumerate(sources):
                ids = pq.read_table(source / table_file("observations", dataset), columns=["obs_id"])["obs_id"]
                owner.update(dict.fromkeys(ids.to_pylist(), i))
            mine = [pa.array([o for o, i in owner.items() if i == k], pa.string()) for k in range(len(sources))]
            rows, links = {}, set()
            for name in SCHEMAS:
                parts = []
                for k, source in enumerate(sources):
                    table = pq.read_table(source / table_file(name, dataset))
                    table = table.filter(pc.is_in(table["obs_id"], value_set=mine[k]))
                    if name == "observations":
                        links |= {(source, p) for p in table["image"].to_pylist() if p}
                    if name == "C1":
                        links |= {(source, p) for options in table["options"].to_pylist() for p in options}
                    parts.append(table)
                merged = pa.concat_tables(parts)
                (out / table_file(name, dataset)).parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(merged, out / table_file(name, dataset))
                rows[name] = merged.num_rows
                if name == "observations":
                    status = Counter(merged["status"].to_pylist())
            for source, relative in sorted(links):
                (out / relative).parent.mkdir(parents=True, exist_ok=True)
                if not (out / relative).exists():  # C1 option views: named by their hash, shared by datasets
                    os.link(source / relative, out / relative)
            entries = [json.loads((s / CARD).read_text()) for s in sources]
            card.setdefault("datasets", {})[dataset] = {
                "exported_at_utc": datetime.now(timezone.utc).isoformat(), "observation_status": dict(status),
                "rows": rows, "collection_runs": {f"{Path(s).name}/{run}": value for s, e in zip(sources, entries)
                                                  for run, value in e["datasets"][dataset]["collection_runs"].items()},
                "reused_from": [{"root": str(Path(s).resolve()), "gt_schema": e["gt_contract"]["schema"],
                                 "observations": len(mine[k])} for k, (s, e) in enumerate(zip(sources, entries))]}
        io_utils.atomic_write_json(out / CARD, card)
    return card


def export(work: Path, out: Path, partial=False):
    """Export the shards in work; once everything is checked and committed, the shards go."""
    with _locked(out):
        if (out / JOURNAL).exists() or (out / STAGING).exists():
            raise SystemExit(f"{out}: an earlier export was interrupted; run with --recover")
        staging = out / STAGING
        try:
            journal = _prepare(work, out, staging, partial)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        io_utils.atomic_write_json(out / JOURNAL, journal, durable=True)
        _commit(out, journal)
    return journal["card"]


def _prepare(work: Path, out: Path, staging: Path, partial: bool) -> dict:
    """Write and check every table and image link in staging; return the commit journal."""
    shards = sorted(p for p in work.iterdir() if (p / RECORDS).exists())
    contracts = check_runs(shards, partial)
    datasets = sorted({json.loads((shard / "collection.json").read_text())["dataset"] for shard in shards})
    (staging / "future_views").mkdir(parents=True)
    tables, statuses, sources, checks = Tables(staging, datasets), Counter(), {}, []
    with ThreadPoolExecutor(16) as pool:  # image hashes are checked while the records are exported
        for shard in shards:
            with gzip.open(shard / RECORDS, "rt") as stream:
                for line in stream:
                    record = json.loads(line)
                    dataset, obs = record["dataset"], record["obs_id"]
                    statuses[f"{dataset}/{record['status']}"] += 1
                    image = f"images/{dataset}/{obs}.png" if record.get("image") else None
                    if image:
                        sources[image] = shard / record["image"]
                        checks.append(pool.submit(_checked, sources[image], record["image_sha256"]))
                    tables.add("observations", _observation(record, image), dataset)
                    if record["status"] != "ok":
                        continue
                    for option in (o["image"] for family in record.get("c1", []) for o in family["options"]):
                        if option not in sources:
                            sources[option] = shard / option
                            checks.append(pool.submit(_checked, sources[option], Path(option).stem))
                    programs = {p["program_id"]: p for p in record["programs"]}
                    for execution in record["executions"]:
                        tables.add("executions", _execution(obs, execution, programs), dataset)
                    for task, row in _questions(record):
                        tables.add(task, row, dataset)
        tables.close()
        for check in checks:
            check.result()
    validate(tables)
    for relative, source in sources.items():  # links: the shards stay intact until the commit is recorded
        (staging / relative).parent.mkdir(parents=True, exist_ok=True)
        os.link(source, staging / relative)
    card = json.loads((out / CARD).read_text()) if (out / CARD).exists() else {}
    card.update(schema="seen-meta.v2", gt_contract=GT_CONFIG, definitions=DEFINITIONS)
    for dataset in datasets:
        card.setdefault("datasets", {})[dataset] = {
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "observation_status": {k.split("/")[1]: v for k, v in statuses.items() if k.startswith(f"{dataset}/")},
            "rows": {k.split("/")[0]: v for k, v in sorted(tables.counts.items()) if k.endswith(f"/{dataset}")},
            "collection_runs": {k: v for k, v in contracts.items() if k.startswith(f"{dataset}-")}}
    files = [r.as_posix() for dataset in datasets for r in dataset_files(dataset) if (staging / r).exists()]
    return {"state": "prepared", "datasets": datasets, "card": card, "files": files,
            "replaced": [r for r in files if (out / r).exists()],
            "shards": [] if partial else [str(s.resolve()) for s in shards],
            "links": [] if partial else [str(s.absolute()) for s in shards if s.is_symlink()]}  # the links themselves


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="export unfinished or partial runs (smoke tests)")
    parser.add_argument("--recover", action="store_true", help="finish or discard an interrupted export")
    parser.add_argument("--reuse", type=Path, nargs="+", help="carry --datasets over from these meta roots "
                        "(an observation from the last root that has it)")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    if args.recover:
        print(recover(args.output))
        return
    if args.reuse:
        args.output.mkdir(parents=True, exist_ok=True)
        card = reuse(args.reuse, args.output, args.datasets)
        print(json.dumps({d: card["datasets"][d]["rows"] for d in args.datasets}))
        return
    if args.work is None:
        parser.error("--work is required")
    print(json.dumps({k: v["rows"] for k, v in export(args.work, args.output, args.partial)["datasets"].items()}))


if __name__ == "__main__":
    main()
