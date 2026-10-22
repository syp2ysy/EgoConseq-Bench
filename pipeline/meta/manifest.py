"""Seen meta inputs: everything needed to rebuild every observation and its GT.

inputs/scenes.jsonl          {dataset, scene_id, partition, source_split}
inputs/<dataset>.jsonl.gz    one observation per line, grouped by scene:
    {obs_id, dataset, scene_id, pose: {position, yaw_rad}, camera_height_m, hfov_deg,
     programs: [{program_id, atomic_actions, proposal_source,
                 commands: [{actions, representation, command_atomic_spans}]}],
     sources: [{origin, record_uid, pose_proposal}]}

A program is one atomic execution (0.25 m forwards / 15 deg turns); `commands` are the public
action lists that describe it (coarse, atomic or merged), each with half-open atom spans.
The fixed hyperparameters (radii, body, checkpoints, budgets) live in pipeline.config.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "seen-meta-inputs.v1"
ATOM_FORWARD_M, ATOM_TURN_DEG = .25, 15.


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_program(commands):
    """Atoms of public commands and the half-open atom span of every command."""
    atomic, spans = [], []
    for command in commands:
        forward = command["type"] == "forward"
        unit, value = (ATOM_FORWARD_M, command["m"]) if forward else (ATOM_TURN_DEG, command["deg"])
        count = round(abs(value) / unit)
        if count < 1 or not math.isclose(abs(value), count * unit, abs_tol=1e-9) or (forward and value <= 0):
            raise ValueError(f"command {command} is not a whole number of atoms")
        atom = {"type": "forward", "m": unit} if forward else {"type": "turn", "deg": math.copysign(unit, value)}
        spans.append([len(atomic), len(atomic) + count])
        atomic += [dict(atom) for _ in range(count)]
    return atomic, spans


def program(commands, representation, proposal_source):
    atomic, spans = atomic_program(commands)
    return {"program_id": digest(atomic)[:16], "atomic_actions": atomic, "proposal_source": proposal_source,
            "commands": [{"actions": list(commands), "representation": representation,
                          "command_atomic_spans": spans}]}


def merge_programs(programs):
    """One entry per atomic program; every public command list describing it is kept."""
    merged = {}
    for value in programs:
        entry = merged.setdefault(value["program_id"], {**value, "commands": []})
        for command in value["commands"]:
            if command not in entry["commands"]:
                entry["commands"].append(command)
    return list(merged.values())


def observation_id(dataset, scene_id, position, yaw_rad, camera_height_m, hfov_deg) -> str:
    """Only an identical input (same scene, exact position and heading, same camera) is the same observation."""
    key = [dataset, scene_id, [float(v) for v in position], float(yaw_rad) % (2 * math.pi),
           float(camera_height_m), float(hfov_deg)]
    return f"{dataset}-{digest(key)[:20]}"


def write_jsonl_gz(path, rows):
    path = Path(path)
    temporary = path.with_name("tmp." + path.name)
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)
