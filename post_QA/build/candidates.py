"""Step 1, candidates: the GT rows of a meta root, per task, with the fields the selection balances.

Every row of tasks/<T>/<dataset>.parquet is a question with its GT; nothing is recomputed. Hard filters
of the benchmark become a `bench` flag (the SFT applies only its own) and initial images failing the image
checks (dark pixels, confirmed rendering defects) are flagged per observation. The 10^7-row tables are trimmed to fit in
memory: within each base group (observation, bench flag and the GROUPS fields) one row is kept for every
value of every other balanced field, so no value of a balanced field is lost for any observation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


LENGTH_EDGES = (2, 3, 5, 7, 11)  # public command count: 1, 2, 3-4, 5-6, 7-10, 11+
FORWARD_EDGES = (0.5, 1.0, 2.0, 4.0)  # total forward metres
TURN_EDGES = (1, 31, 61, 91)  # total turned degrees: none, up to 30, 60, 90, more
HORIZONTAL = ("front", "front-right", "right", "rear-right", "rear", "rear-left", "left", "front-left")
VERTICAL = ("above", "level", "below")
POSITIONS = {"F2": ("F2/first", "F2/last"), "F3+": ("F3+/first", "F3+/middle", "F3+/last")}  # A2, by forward count
ACTIONS = ("length", "forward_bin", "turn_bin", "turn_side")  # what a program does
GROUPS = {  # base groups of the trimming: answers (hard cells) and fields balanced with them
    "A1": ["collision", "radius_m", "representation"],
    "A2": ["position", "longest", "representation"],
    "A3": ["category", "representation"],
    "A4": ["label", "representation"],
    "B1": ["bin", "representation"],
    "B2": ["label", "representation"],
    "B3": ["bin", "representation"],
    "C1": ["representation"],
}
FIELDS = ("start",) + ACTIONS  # the other balanced fields: one row per value is kept in each base group
EXTRA = {"A2": ["radius_m"], "A3": ["radius_m"], "A4": ["fraction"], "B1": ["change"], "B3": ["fraction", "change"]}
COLUMNS = {
    "A1": ["radius_m", "collision"],
    "A2": ["radius_m", "first_collision_action"],
    "A3": ["radius_m", "category"],
    "A4": ["radii_m", "checkpoint_fraction", "horizontal", "vertical", "boundary_margin_deg"],
    "B1": ["radii_m", "distance_m", "initial_distance_m"],
    "B2": ["radii_m", "horizontal", "vertical", "boundary_margin_deg"],
    "B3": ["radii_m", "checkpoint_fraction", "distance_m", "initial_distance_m"],
    "C1": ["radii_m", "family_id", "options", "min_block_l1"],
}


def card(meta: Path) -> dict:
    return json.loads((meta / "dataset_card.json").read_text())


def tasks_of(meta_card: dict, dataset: str) -> list:
    return list(meta_card["gt_contract"]["tasks_by_dataset"][dataset])


def observations(meta: Path, dataset: str, spec: dict, quality) -> pd.DataFrame:
    """Observations that have GT (status ok) with position, camera and the image checks of their initial
    image (dark fraction from meta; reviewed defects excluded by hash without decoding PNGs)."""
    table = pq.read_table(meta / "observations" / f"{dataset}.parquet", columns=[
        "obs_id", "scene", "split", "status", "floor_y", "camera_position", "camera_yaw_rad", "camera_height_m",
        "hfov_deg", "vfov_deg", "width", "height", "image", "image_sha256", "dark_pixel_fraction"]).to_pandas()
    if (table["split"] != spec["split"]).any():
        raise ValueError(f"{dataset}: the {spec['name']} build only reads {spec['split']} observations")
    table = table[table["status"] == "ok"].set_index("obs_id")
    position = np.stack(table.pop("camera_position").to_numpy())
    table["x"], table["z"] = position[:, 0], position[:, 2]
    table["yaw_deg"] = np.degrees(table.pop("camera_yaw_rad")) % 360
    table["camera"] = [f"{h:g}m/{f:g}deg" for h, f in zip(table["camera_height_m"], table["hfov_deg"])]
    table["path"] = [str(meta / path) for path in table["image"]]
    quality.measured.update(zip(table["image"], table["dark_pixel_fraction"]))
    reviews = quality.reviews
    table["image_review"] = table["image_sha256"].map(reviews.status)
    table["image_review_reason"] = table["image_sha256"].map(reviews.reasons).fillna("unreviewed")
    for kind, strict in [("bench", True)] + ([("sft", False)] if "sft" in spec else []):
        rejected = [quality.rejection(path, sha, spec, strict)
                    for path, sha in zip(table["image"], table["image_sha256"])]
        table[f"{kind}_image"] = [r is None for r in rejected]
        table[f"{kind}_image_reason"] = [r[0] if r else None for r in rejected]
        table[f"{kind}_image_value"] = [r[1] if r else 0. for r in rejected]
    return table


def partition_scenes(dataset: str, split: str) -> list:
    partitions = json.loads((Path(__file__).resolve().parents[2] / "pipeline/assets/scene_partitions.v1.json").read_text())
    return sorted(s for s, v in partitions["datasets"][dataset].items() if v["partition"] == split)


def _rank(question_ids, seed: int) -> np.ndarray:
    """Deterministic, seed-dependent order of question ids (they are hex digests)."""
    mix = np.uint64(int.from_bytes(np.random.default_rng(seed).bytes(8), "big") >> 4)
    return np.array([int(q[-15:], 16) for q in question_ids], dtype=np.uint64) ^ mix


def _programs(actions: pa.ListArray) -> dict:
    """Per row: what the public commands do (start, command count, forward and turn bins, turn side)."""
    offsets = actions.offsets.to_numpy()
    flat, starts = actions.flatten(), offsets[:-1]
    forward = pc.equal(flat.field("type"), "forward").to_numpy(zero_copy_only=False)
    metres = np.nan_to_num(flat.field("m").to_numpy(zero_copy_only=False).astype(float))
    degrees = np.nan_to_num(flat.field("deg").to_numpy(zero_copy_only=False).astype(float))
    count, forward_m = np.diff(offsets), np.add.reduceat(metres, starts)
    turn_deg = np.add.reduceat(np.abs(degrees), starts)
    left, right = np.add.reduceat(degrees < 0, starts) > 0, np.add.reduceat(degrees > 0, starts) > 0
    return {"start": np.where(forward[starts], "forward", "turn"),
            "length": np.digitize(count, LENGTH_EDGES), "forward_bin": np.digitize(forward_m, FORWARD_EDGES),
            "turn_bin": np.digitize(turn_deg, TURN_EDGES),
            "turn_side": np.where(left & right, "both", np.where(left, "left", np.where(right, "right", "none"))),
            "_F": np.add.reduceat(forward.astype(np.int16), starts), "_flat": (forward, metres, offsets)}


def _a2_fields(answer, programs):
    """Which forward command the first collision falls in and whether it is the longest one.

    position: "<group>/<first|middle|last>", group "F2" (two forwards) or "F3+"; longest: "F<F>:True" or
    "F<F>:False" (F forwards) when the program has a unique longest forward, whatever the answer, else None
    (no length shortcut to control).
    """
    forward, metres, offsets = programs["_flat"]
    starts, count, F = offsets[:-1], np.diff(offsets), programs["_F"]
    k = starts + answer - 1
    cumulative = np.concatenate([[0], np.cumsum(forward)])
    ordinal = cumulative[k + 1] - cumulative[starts]
    top = np.maximum.reduceat(np.where(forward, metres, -np.inf), starts)
    unique = np.add.reduceat(forward & (np.abs(metres - np.repeat(top, count)) <= 1e-9), starts) == 1
    group = np.where(F == 2, "F2", "F3+")
    where = np.where(ordinal == 1, "first", np.where(ordinal == F, "last", "middle"))
    position = np.char.add(np.char.add(group, "/"), where)
    longest = np.char.add(np.char.add("F", F.astype(str)),
                          np.where(np.abs(metres[k] - top) <= 1e-9, ":True", ":False")).astype(object)
    longest[~unique] = None
    return position, group, longest


def _distance_fields(table, spec):
    distance = table["distance_m"].to_numpy()
    delta = distance - table["initial_distance_m"].to_numpy()
    bands = spec["distance_change_m"]
    change = np.where(np.abs(delta) < bands["unchanged"], "unchanged", np.where(
        np.abs(delta) < bands["small"], "small", np.where(delta > 0, "farther", "closer")))
    gate = spec["benchmark"]["distance_gate"]
    bench = ((np.abs(delta) >= np.maximum(gate["min_change_m"], gate["min_change_ratio"] *
                                          table["initial_distance_m"].to_numpy()))
             & (distance >= gate["min_distance_m"]))
    return {"change": change, "bin": np.digitize(distance, spec["distance_bins_m"]), "bench": bench}


def _rows(task, table, spec, radii):
    """The selection fields of one row group of a task table."""
    programs = _programs(table["actions"].combine_chunks())
    out = {"question_id": table["question_id"].to_numpy(zero_copy_only=False),
           "obs_id": table["obs_id"].to_numpy(zero_copy_only=False),
           "program_id": table["program_id"].to_numpy(zero_copy_only=False),
           "command_list": table["command_list"].to_numpy(),
           "representation": table["representation"].to_numpy(zero_copy_only=False),
           **{k: v for k, v in programs.items() if not k.startswith("_")}}
    bench = np.ones(len(table), bool)
    if task in ("A1", "A2", "A3"):
        out["radius_m"] = table["radius_m"].to_numpy()
    else:  # B/A4/C1 hold for every radius in radii_m: a bit mask over the GT radii
        listed = table["radii_m"].combine_chunks()
        bits = np.left_shift(1, np.searchsorted(radii, listed.flatten().to_numpy()))
        out["radii"] = np.add.reduceat(bits, listed.offsets.to_numpy()[:-1]).astype(np.int8)
    if task == "A1":
        out["collision"] = table["collision"].to_numpy()
    elif task == "A2":
        out["position"], out["fgroup"], out["longest"] = _a2_fields(table["first_collision_action"].to_numpy(),
                                                                    programs)
    elif task == "A3":
        out["category"] = table["category"].to_numpy(zero_copy_only=False)
    elif task == "C1":
        out["family_id"] = table["family_id"].to_numpy(zero_copy_only=False)
        out["options"] = table["options"].to_numpy(zero_copy_only=False)
        out["min_block_l1"] = table["min_block_l1"].to_numpy()
    else:
        if task in ("A4", "B3"):
            out["fraction"] = table["checkpoint_fraction"].to_numpy()
        if task in ("A4", "B2"):
            out["label"] = (pc.index_in(table["horizontal"], value_set=pa.array(HORIZONTAL)).to_numpy() * 3
                            + pc.index_in(table["vertical"], value_set=pa.array(VERTICAL)).to_numpy())
            bench &= table["boundary_margin_deg"].to_numpy() >= spec["benchmark"]["direction_margin_deg"]
        else:
            fields = _distance_fields(table, spec)
            bench &= fields.pop("bench")
            out.update(fields)
    out["bench"] = bench
    frame = pd.DataFrame(out)
    return frame[programs["_F"] >= 2] if task == "A2" else frame  # A2 needs two answerable forward commands


def _trim(frame, keys, fields):
    """Within each group of keys, the first row (in the seeded order) of every value of every field."""
    frame = frame.sort_values("_rank", kind="stable")
    kept = np.zeros(len(frame), bool)
    for field in fields:
        kept |= frame.groupby(keys + [field], sort=False, dropna=False).cumcount().to_numpy() == 0
    return frame[kept]


def radius_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """A1 rows of one observation, program and command list at adjacent radii, free then colliding."""
    rows = frame.sort_values(["obs_id", "program_id", "command_list", "radius_m"], kind="stable")
    key = rows["obs_id"] + ":" + rows["program_id"] + ":" + rows["command_list"].astype(str)
    hit = rows["collision"].to_numpy(bool)
    flip = np.zeros(len(rows), bool)
    flip[:-1] = (key.to_numpy()[:-1] == key.to_numpy()[1:]) & ~hit[:-1] & hit[1:]
    free, hit = rows[flip], rows[np.roll(flip, 1)]
    pairs = free[["obs_id", "program_id", "representation", "length", "start", "forward_bin", "turn_bin",
                  "turn_side", "_rank"]].copy()
    pairs["q_free"], pairs["q_collide"] = free["question_id"].to_numpy(), hit["question_id"].to_numpy()
    pairs["r_free"], pairs["r_collide"] = free["radius_m"].to_numpy(), hit["radius_m"].to_numpy()
    pairs["pair_id"] = pairs["q_free"] + "+" + pairs["q_collide"]
    return pairs


def candidates(meta: Path, dataset: str, task: str, spec: dict, radii: list):
    """Trimmed candidates of one task, their counts before trimming and (A1) the radius pairs."""
    source = pq.ParquetFile(meta / "tasks" / task / f"{dataset}.parquet")
    columns = ["question_id", "obs_id", "program_id", "command_list", "representation", "actions"] + COLUMNS[task]
    keys, fields = ["obs_id", "bench"] + GROUPS[task], FIELDS + tuple(EXTRA.get(task, ()))
    parts, pairs, rows, tail = [], [], 0, None
    for group in range(source.num_row_groups):
        frame = _rows(task, source.read_row_group(group, columns=columns), spec, radii)
        frame["_rank"] = _rank(frame["question_id"], spec["seed"])
        rows += len(frame)
        if task == "A1":  # an observation's rows are contiguous but may continue in the next row group
            whole = pd.concat([tail, frame], ignore_index=True)
            last = whole["obs_id"].to_numpy() == whole["obs_id"].iloc[-1]
            pairs.append(radius_pairs(whole[~last]))
            tail = whole[last]
        parts.append(_trim(frame, keys, fields))
    result = _trim(pd.concat(parts, ignore_index=True), keys, fields)
    stats = {"rows": rows, "kept": len(result), "observations": int(result["obs_id"].nunique())}
    if task != "A1":
        return result, None, stats
    pairs.append(radius_pairs(tail))
    pairs = _trim(pd.concat(pairs, ignore_index=True), ["obs_id", "r_free", "r_collide", "representation"], FIELDS)
    stats["pairs"] = len(pairs)
    return result, pairs, stats
