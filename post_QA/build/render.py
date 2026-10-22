"""Step 4, rendering: questions, answers and image references of the selection; checks and report.

Items follow the released format (ms-swift messages + images). Image paths are relative to the meta
root: initial images and C1 options are referenced where the meta export stored them; only the marked
target-point images are new, in marked/<dataset>/, named by the source image hash, the pixel and the
marker style, and rendered on every build (a file that differs from the rendering is replaced).
Presentation is assigned per set, independently of answers: units (m/cm, half and half by items) per
dataset and task, templates per task, C1 answer positions per dataset and, for B/A4/C1, the least used
legal radius; the two items of a radius pair share everything except the radius.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads
from PIL import Image, ImageDraw

from pipeline import config, io_utils
from post_QA import answers, templates as T
from post_QA.build import candidates as C
from post_QA.build.candidates import HORIZONTAL, VERTICAL
from post_QA.build.select import Views, split
from post_QA.categories import display_category_name

RADIUS = config.TARGET_POINT_MARKER_RADIUS_PX
MARKER = f"dot{RADIUS}"  # red disc in a white and a black ring, as pipeline.viz draws target points
FIELDS = {
    "A1": ["radius_m", "collision"], "A2": ["radius_m", "first_collision_action"], "A3": ["radius_m", "category"],
    "A4": ["radii_m", "point_id", "pixel_xy", "checkpoint_action", "checkpoint_fraction", "horizontal", "vertical",
           "boundary_margin_deg"],
    "B1": ["radii_m", "point_id", "pixel_xy", "distance_m", "initial_distance_m"],
    "B2": ["radii_m", "point_id", "pixel_xy", "horizontal", "vertical", "boundary_margin_deg"],
    "B3": ["radii_m", "point_id", "pixel_xy", "checkpoint_action", "checkpoint_fraction", "distance_m",
           "initial_distance_m"],
    "C1": ["radii_m", "options", "answer_index", "family_id"],
}
KEEP = ("cell", "start", "fgroup", "position", "longest", "label", "category", "bin", "change", "fraction",
        "length", "forward_bin", "turn_bin", "turn_side", "camera", "scene",
        "min_block_l1")  # selection fields kept per item


def _hash(*parts) -> bytes:
    return hashlib.sha256(":".join(map(str, parts)).encode()).digest()


def _value(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def fetch(meta, dataset, task, question_ids) -> pd.DataFrame:
    """Full GT rows of the selected questions, by meta question_id."""
    table = pads.dataset(meta / "tasks" / task / f"{dataset}.parquet").to_table(
        columns=["question_id", "actions"] + FIELDS[task], filter=pc.field("question_id").isin(list(question_ids)))
    return table.to_pandas().set_index("question_id")


def present(selection: pd.DataFrame, radii: list, seed: int) -> pd.DataFrame:
    """Units, templates, C1 answer positions and shown radii, keyed by item (pair_id for a pair)."""
    selection = selection.reset_index(drop=True)
    pair = selection["pair_id"].notna() if "pair_id" in selection else pd.Series(False, index=selection.index)
    selection["key"] = selection["pair_id"].where(pair, selection["question_id"]) if pair.any() else selection["question_id"]
    for column in ("unit", "template_id", "letter", "radius"):
        selection[column] = None
    for (kind, dataset, task), group in selection.groupby(["set", "dataset", "task"]):
        items = Counter()  # units by output items: a pair counts twice and is placed first
        for i in sorted(group.index, key=lambda i: (not pair[i], _hash(seed, "unit", group.at[i, "key"]))):
            unit = min(T.UNITS, key=lambda u: (items[u], _hash(seed, group.at[i, "key"], u)))
            items[unit] += 2 if pair[i] else 1
            selection.at[i, "unit"] = unit
        if task == "C1":
            letters = T.assign(group["key"], "ABCD", seed=seed, namespace=f"letter:{kind}:{dataset}")
            selection.loc[group.index, "letter"] = group["key"].map(letters)
        if task not in ("A1", "A2", "A3"):  # B/A4/C1 hold at every radius in radii_m: the least used one
            used = Counter()
            for i in sorted(group.index, key=lambda i: _hash(seed, "radius", group.at[i, "key"])):
                legal = [r for b, r in enumerate(radii) if int(group.at[i, "radii"]) >> b & 1]
                choice = min(legal, key=lambda r: (used[r], _hash(seed, group.at[i, "key"], r)))
                used[choice] += 1
                selection.at[i, "radius"] = choice
    for (kind, task), group in selection.groupby(["set", "task"]):
        ids = [t["id"] for t in T.QUESTION_TEMPLATES[task]]
        selection.loc[group.index, "template_id"] = group["key"].map(
            T.assign(group["key"], ids, seed=seed, namespace=f"template:{kind}:{task}"))
    return selection


def _inputs(task, gt, obs, radius):
    actions = [{"index": i + 1, "type": a["type"], **({"meters": a["m"]} if a["type"] == "forward" else
                                                      {"degrees": a["deg"]})} for i, a in enumerate(gt["actions"])]
    inputs = {"actions": actions, "robot": {"radius_m": float(radius)},
              "camera": {"hfov_deg": obs["hfov_deg"], "vfov_deg": obs["vfov_deg"],
                         "optical_center_height_m": obs["camera_height_m"]}}
    if task in ("A4", "B3"):
        inputs["checkpoint"] = {"action_index": int(gt["checkpoint_action"]), "fraction": float(gt["checkpoint_fraction"])}
    return inputs


def _answer(task, gt, unit, letter):
    if task == "A1":
        return answers.canonical_answer("A1", {"canonical_answer": "collision" if gt["collision"] else "no_collision"})
    if task == "A2":
        return answers.canonical_answer("A2", {"canonical_answer": f"action_{int(gt['first_collision_action'])}"})
    if task == "A3":
        return answers.canonical_answer("A3", {"canonical_answer": gt["category"]})
    if task in ("A4", "B2"):
        return answers.canonical_answer(task, {"horizontal_direction": gt["horizontal"],
                                               "vertical_direction": gt["vertical"]})
    if task in ("B1", "B3"):
        return answers.format_distance(float(gt["distance_m"]), unit)
    return letter


def item_id(task, question_id) -> str:
    return f"{task.lower()}-{question_id.split('-', 1)[1]}"


def marked_path(dataset, obs, pixel) -> str:
    return f"marked/{dataset}/{obs['image_sha256'][:20]}-{int(pixel[0])}-{int(pixel[1])}-{MARKER}.png"


def _draw(job):
    """The initial image (its hash verified) with the target point; the file is written unless it holds
    exactly this rendering. Returns the rendering's hash."""
    source, sha, pixel, path = job
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"{source} differs from its meta hash")
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
    draw, (x, y) = ImageDraw.Draw(image), pixel
    for r, color in ((RADIUS, (0, 0, 0)), (RADIUS - 2, (255, 255, 255)), (RADIUS - 4, (220, 24, 24))):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=config.QA_MARKED_PNG_COMPRESSION_LEVEL)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if not path.exists() or io_utils.sha256_file(path) != digest:
        path.parent.mkdir(parents=True, exist_ok=True)
        io_utils.atomic_write_binary(path, lambda stream: stream.write(data))
    return digest


def _versions(kind):
    """The radius pairs are a full-configuration diagnostic: hiding the radius leaves both prompts equal."""
    return [("full", T.ABLATIONS["full"])] if kind == "radius_pairs" else T.ABLATIONS.items()


def items(meta, selection: pd.DataFrame, obs: dict, workers=16):
    """Every item of the selection in every ablation version.

    Returns (items, per-item facts, marked image path -> hash of its rendering)."""
    out, facts, rows, jobs = defaultdict(list), [], {}, {}
    for (dataset, task), group in selection.groupby(["dataset", "task"]):
        ids = {q for column in ("question_id", "q_free", "q_collide") if column in group
               for q in group[column] if isinstance(q, str)}
        rows[dataset, task] = gt = fetch(meta, dataset, task, ids)
        if task in T.TARGET_TASKS:  # marked images, drawn once and in parallel
            for obs_id, question_id in zip(group["obs_id"], group["question_id"]):
                g, o = gt.loc[question_id], obs[dataset].loc[obs_id]
                path = marked_path(dataset, o, g["pixel_xy"])
                jobs[path] = (meta / o["image"], o["image_sha256"], [int(v) for v in g["pixel_xy"]], meta / path)
    with Pool(workers) as pool:
        marked = dict(zip(jobs, pool.map(_draw, list(jobs.values()), chunksize=64)))
    for (dataset, task), group in selection.groupby(["dataset", "task"]):
        gt = rows[dataset, task]
        for _, row in group.iterrows():
            o = obs[dataset].loc[row["obs_id"]]
            pair = isinstance(row.get("pair_id"), str)
            members = ([(row["q_free"], row["r_free"]), (row["q_collide"], row["r_collide"])] if pair else
                       [(row["question_id"], gt.at[row["question_id"], "radius_m"] if task in ("A1", "A2", "A3")
                         else row["radius"])])
            template = next(t for t in T.QUESTION_TEMPLATES[task] if t["id"] == row["template_id"])
            for question_id, radius in members:
                g = gt.loc[question_id]
                images = [marked_path(dataset, o, g["pixel_xy"]) if task in T.TARGET_TASKS else o["image"]]
                if task == "C1":
                    options = list(g["options"])
                    correct = options.pop(int(g["answer_index"]))
                    options.insert("ABCD".index(row["letter"]), correct)
                    images += options
                answer = _answer(task, g, row["unit"], row["letter"])
                inputs = _inputs(task, g, o, radius)
                for version, hide in _versions(row["set"]):
                    user = T.render_question(task, inputs, template, unit=row["unit"], hide=hide)
                    item = {"id": item_id(task, question_id), "task_id": task, "dataset": dataset, "images": images,
                            "messages": [{"role": "system", "content": T.SYSTEM_PROMPT},
                                         {"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                            "unit": row["unit"]}
                    if pair:
                        item["pair_id"] = row["pair_id"]
                    out[row["set"], version].append(item)
                facts.append({"set": row["set"], "dataset": dataset, "task": task, "id": item_id(task, question_id),
                              "question_id": question_id, "obs_id": row["obs_id"],
                              "pair_id": row["pair_id"] if pair else None, "program_id": row["program_id"],
                              "representation": row["representation"], "radius": float(radius), "unit": row["unit"],
                              "template_id": row["template_id"], "letter": row["letter"], "answer": answer,
                              "images": images, **{k: _value(row.get(k)) for k in KEEP},
                              "gt": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g.items()
                                     if k != "actions"},
                              "actions": [dict(a) for a in g["actions"]]})
    return out, facts, marked


def _derived(task, gt, actions):
    """Selection fields re-derived from the GT row: hard cell, first command and A2's longest flag."""
    fields = {"start": actions[0]["type"]}
    if task == "A1":
        fields["cell"] = f"collision={bool(gt['collision'])}"
    elif task == "A2":
        forwards = [i for i, a in enumerate(actions, 1) if a["type"] == "forward"]
        hit = int(gt["first_collision_action"])
        k = forwards.index(hit) if hit in forwards else -1
        group = "F2" if len(forwards) == 2 else "F3+"
        fields["cell"] = f"{group}/{'first' if k == 0 else 'last' if k == len(forwards) - 1 else 'middle'}"
        lengths = [actions[i - 1]["m"] for i in forwards]
        top = max(lengths)
        unique = sum(abs(m - top) <= 1e-9 for m in lengths) == 1
        fields["longest"] = f"F{len(forwards)}:{abs(actions[hit - 1]['m'] - top) <= 1e-9}" if unique else None
    elif task in ("A4", "B2"):
        fields["cell"] = f"label={HORIZONTAL.index(gt['horizontal']) * 3 + VERTICAL.index(gt['vertical'])}"
    return fields


def _item_ok(f, item, o, spec, quality):
    """One final item (full version) against its meta GT row, its observation and the spec."""
    task, gt, actions, unit = f["task"], f["gt"], f["actions"], item["unit"]
    text, answer = item["messages"][1]["content"], item["messages"][2]["content"]
    template = next(t for t in T.QUESTION_TEMPLATES[task] if t["id"] == f["template_id"])
    first = marked_path(f["dataset"], o, gt["pixel_xy"]) if task in T.TARGET_TASKS else o["image"]
    good = ([m["role"] for m in item["messages"]] == ["system", "user", "assistant"]
            and item["messages"][0]["content"] == T.SYSTEM_PROMPT
            and (item["task_id"], item["dataset"], unit, item.get("pair_id"))
            == (task, f["dataset"], f["unit"], f["pair_id"])
            and item["images"][0] == first and len(item["images"]) == (5 if task == "C1" else 1))
    if task == "A1":
        good &= answer == ("Collision." if gt["collision"] else "No collision.")
    elif task == "A2":
        hit = int(gt["first_collision_action"])
        good &= answer == f"Action {hit}." and actions[hit - 1]["type"] == "forward"
    elif task == "A3":
        good &= answer == display_category_name(gt["category"])
    elif task in ("A4", "B2"):
        good &= answers.parse_direction(answer) == answers.direction_axes(gt["horizontal"], gt["vertical"])
    elif task in ("B1", "B3"):
        good &= (answer.endswith(f" {unit}")
                 and abs(float(answers.parse_distance(answer, default_unit=unit)) - gt["distance_m"]) <= .005 + 1e-9)
    else:
        good &= (answer == f["letter"] and sorted(item["images"][1:]) == sorted(gt["options"])
                 and item["images"][1 + "ABCD".index(answer)] == gt["options"][gt["answer_index"]])
    steps = [f"({i}) forward {T.format_length(a['m'], unit)}" if a["type"] == "forward" else
             f"({i}) turn {'right' if a['deg'] >= 0 else 'left'} {T._number(abs(a['deg']))} "
             f"{'degree' if abs(a['deg']) == 1 else 'degrees'}"
             for i, a in enumerate(actions, 1)]
    lines = ["Actions:\n" + "\n".join(steps) + "\n\n",
             f"- Collision footprint radius: {T.format_length(f['radius'], unit)}\n",
             "- Camera optical-center height above the ground at your starting position: "
             + T.format_length(o["camera_height_m"], unit, height=True) + "\n",
             f"- Horizontal field of view: {T._number(o['hfov_deg'])}°\n",
             f"- Vertical field of view: {T._number(o['vfov_deg'])}°\n\n",
             "Question: " + template["question"].format(target=T.TARGET),
             T._DISTANCE_ANSWER[unit] if task in ("B1", "B3") else T.ANSWER_FORMATS[task]]
    if task in ("A4", "B3"):
        lines.append(f"Query moment: After completing all earlier actions and "
                     f"{T._number(100 * gt['checkpoint_fraction'])}% of the forward distance in action "
                     f"{gt['checkpoint_action']}.")
        good &= actions[gt["checkpoint_action"] - 1]["type"] == "forward"
    good &= all(line in text for line in lines) and text.count("<image>") == len(item["images"])
    good &= [T.COLLISION in text, T.FREE in text] == [T.PREMISES.get(task) == p for p in (T.COLLISION, T.FREE)]
    good &= ("marks a fixed surface point" in text) == (task in T.TARGET_TASKS)
    if task in ("A1", "A2", "A3"):
        good &= abs(f["radius"] - gt["radius_m"]) < 1e-9
    else:
        good &= any(abs(f["radius"] - r) < 1e-9 for r in gt["radii_m"])
    derived = _derived(task, gt, actions)
    if f["set"] != "benchmark":
        derived.pop("cell", None)
    good &= all(f[k] == v for k, v in derived.items())
    bench = spec["benchmark"]
    if f["set"] != "sft":  # benchmark gates
        if task in ("A4", "B2"):
            good &= gt["boundary_margin_deg"] >= bench["direction_margin_deg"]
        if task in ("B1", "B3"):
            gate = bench["distance_gate"]
            good &= (abs(gt["distance_m"] - gt["initial_distance_m"])
                     >= max(gate["min_change_m"], gate["min_change_ratio"] * gt["initial_distance_m"])
                     and gt["distance_m"] >= gate["min_distance_m"])
    strict = f["set"] != "sft"
    good &= quality.rejection(o["image"], o["image_sha256"], spec, strict) is None
    for path in item["images"][1:]:  # C1 options
        good &= quality.rejection(path, Path(path).stem, spec, strict) is None
    return bool(good)


def check(meta, selection, out, facts, marked, obs, spec, cells, excluded, quality, similar, workers=16):
    """The final items against the selection, the meta GT, the hard quotas, the image checks and contents,
    and (SFT) the evaluation images: quality(meta path) -> dark fraction; similar: obs_id -> (evaluation
    image, cosine) of the SFT observations compared. Sets eval_image and eval_cos of the SFT facts.

    Returns (problems, image hashes by path)."""
    problems = []
    fail = problems.append
    by_set = defaultdict(set)
    for f in facts:
        by_set[f["set"]].add(f["obs_id"])
    for a in by_set:
        for b in by_set:
            if a < b and by_set[a] & by_set[b]:
                fail(f"{a} and {b} share {len(by_set[a] & by_set[b])} observations")
    if by_set["sft"] & excluded:
        fail(f"the SFT uses {len(by_set['sft'] & excluded)} observations near the benchmark")

    expected = defaultdict(Counter)  # one item per selected question, two per radius pair
    for row in selection.to_dict("records"):
        pair = isinstance(row.get("pair_id"), str)
        for question_id in ([row["q_free"], row["q_collide"]] if pair else [row["question_id"]]):
            expected[row["set"]][item_id(row["task"], question_id)] += 1
    fact = {(f["set"], f["id"]): f for f in facts}
    kinds = sorted(set(expected) | {kind for kind, _ in out})
    for kind in kinds:
        ids = [r["id"] for r in out[kind, "full"]]
        if Counter(ids) != expected[kind] or len(set(ids)) != len(ids):
            fail(f"{kind}: the items are not the selected questions, once each")
        if Counter(f["id"] for f in facts if f["set"] == kind) != Counter(ids):
            fail(f"{kind}: the facts are not the items")
        for item in out[kind, "full"]:
            f = fact.get((kind, item["id"]))
            try:
                ok = f is not None and _item_ok(f, item, obs[f["dataset"]].loc[f["obs_id"]], spec, quality)
            except (ValueError, KeyError, IndexError, TypeError, StopIteration) as error:
                ok = False
                fail(f"{kind}/{item['id']}: {error!r}")
            if not ok:
                fail(f"{kind}/{item['id']}: the item disagrees with its GT, observation or the spec")
        for version, hide in _versions(kind):  # the same items, without the hidden parameter lines
            if version == "full":
                continue
            if [r["id"] for r in out[kind, version]] != ids:
                fail(f"{kind}/{version}: not the items of the full version")
                continue
            for r, base in zip(out[kind, version], out[kind, "full"]):
                user = T.remove_parameters(base["messages"][1]["content"], hide)
                expected_item = {**base, "messages": [base["messages"][0], {"role": "user", "content": user},
                                                      base["messages"][2]]}
                if r != expected_item or (hide and user == base["messages"][1]["content"]):
                    fail(f"{kind}/{version}/{r['id']}: not the full item without its parameter lines")
        pairs = defaultdict(list)
        for r in out[kind, "full"]:
            if "pair_id" in r:
                pairs[r["pair_id"]].append(r)
        for pair_id, rows in pairs.items():  # without the radius line the two prompts are equal
            prompts = [(r["messages"][0], T.remove_parameters(r["messages"][1]["content"], ("radius",))) for r in rows]
            if not (len(rows) == 2 and prompts[0] == prompts[1] and rows[0]["images"] == rows[1]["images"]
                    and {r["messages"][2]["content"] for r in rows} == {"Collision.", "No collision."}):
                fail(f"{kind}/{pair_id}: not a minimal radius pair")

    frame = pd.DataFrame([{k: f[k] for k in ("set", "dataset", "task", "scene", "obs_id", "program_id", "pair_id",
                                              "cell", "start", "unit", "letter")} for f in facts])
    units = frame.drop_duplicates(["set", "dataset", "task", "obs_id", "program_id", "cell"])  # a pair counts once
    counts = units.groupby(["set", "dataset", "task", "cell"]).size()
    for cell in cells:
        got = counts.get((cell["set"], cell["dataset"], cell["task"], cell["cell"]), 0)
        if got != cell["quota"]:
            fail(f"{cell['set']}/{cell['dataset']}/{cell['task']}/{cell['cell']}: {got} of {cell['quota']}")
    share = spec["benchmark"]["turn_first_min"]
    for (kind, dataset, task), g in units[units["set"] != "sft"].groupby(["set", "dataset", "task"]):
        if (g["start"] == "turn").sum() < math.ceil(share * len(g)):
            fail(f"{kind}/{dataset}/{task}: {(g['start'] == 'turn').sum()} of {len(g)} start with a turn")
    for dataset in obs:
        for task, total in spec["benchmark"]["totals"][dataset].items():
            got = ((units["set"] == "benchmark") & (units["dataset"] == dataset) & (units["task"] == task)).sum()
            if got != total:
                fail(f"benchmark/{dataset}/{task}: {got} of {total} questions")
        got = ((units["set"] == "radius_pairs") & (units["dataset"] == dataset)).sum()
        if got != spec["radius_pairs"]["pairs_per_dataset"]:
            fail(f"radius_pairs/{dataset}: {got} pairs")
        least = spec["benchmark"].get("scene_min_items", {}).get(dataset, 0)  # every usable scene of the split
        per_scene = Counter(units.loc[(units["set"] == "benchmark") & (units["dataset"] == dataset), "scene"])
        unusable = spec["benchmark"].get("unusable_scenes", {}).get(dataset, {})
        for scene in set(C.partition_scenes(dataset, spec["split"])) - set(unusable):
            if per_scene[scene] < least:
                fail(f"benchmark/{dataset}/{scene}: {per_scene[scene]} questions, fewer than {least}")
    for task in ("A4", "B2"):  # equal joint labels over the datasets
        total = sum(spec["benchmark"]["totals"][d].get(task, 0) for d in obs)
        got = Counter(units.loc[(units["set"] == "benchmark") & (units["task"] == task), "cell"])
        if total and dict(got) != {f"label={l}": n for l, n in split(total, range(24)).items()}:
            fail(f"benchmark/{task}: labels {dict(got)}")
    for (kind, dataset, task), g in frame.groupby(["set", "dataset", "task"]):
        n = Counter(g["unit"])
        if abs(n["m"] - n["cm"]) > (2 if g["pair_id"].notna().all() else 1):
            fail(f"{kind}/{dataset}/{task}: m/cm {n['m']}/{n['cm']}")
        if task == "C1":
            n = Counter(g["letter"])
            if max(n.values()) - min(n[x] for x in "ABCD") > 1:
                fail(f"{kind}/{dataset}/C1: answer positions {dict(n)}")

    threshold = spec["benchmark"]["dino"]["threshold"]
    for f in facts:  # each SFT initial image below the threshold to the pose-near evaluation images
        f["unmarked_image_sha256"] = ([obs[f["dataset"]].at[f["obs_id"], "image_sha256"]]
                                      + [Path(p).stem for p in f["images"][1:]])
        f["image_review"] = [quality.reviews.status(sha) for sha in f["unmarked_image_sha256"]]
        f["image_review_reason"] = [quality.reviews.reasons.get(sha, "unreviewed") for sha in f["unmarked_image_sha256"]]
        if f["set"] == "sft":
            f["eval_image"], f["eval_cos"] = similar.get(f["obs_id"], (None, None))
            if f["eval_cos"] is not None and f["eval_cos"] >= threshold:
                fail(f"sft/{f['id']}: {f['eval_cos']} similar to the evaluation image {f['eval_image']}")

    # _draw verifies marked images' sources while reading them. Other initial images are
    # themselves displayed and checked below, so no second read of marked sources is needed.
    paths = sorted({p for kind in kinds for r in out[kind, "full"] for p in r["images"]})
    with Pool(workers) as pool:
        hashes = dict(zip(paths, pool.map(io_utils.sha256_file, [meta / p for p in paths], chunksize=64)))
    for kind in kinds:  # contents: the initial image of the observation, the rendering, the C1 future views
        for r in out[kind, "full"]:
            f, first = fact.get((kind, r["id"])), r["images"][0]
            if f is None:
                continue
            if hashes[first] != (marked.get(first) if r["task_id"] in T.TARGET_TASKS
                                 else obs[f["dataset"]].at[f["obs_id"], "image_sha256"]):
                fail(f"{kind}/{r['id']}: the first image is not the expected content")
            if any(hashes[p] != p.rsplit("/", 1)[1].removesuffix(".png") for p in r["images"][1:]):
                fail(f"{kind}/{r['id']}: a C1 option image differs from its content hash")
    return problems, hashes


def _runs(actions) -> str:
    kinds = "".join("F" if a["type"] == "forward" else "T" for a in actions)
    runs = "".join(k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k)
    return runs if len(runs) <= 4 else runs[:4] + "+"


def _close_views(obs_ids, where, conflict) -> float:
    """Share of observations with another observation of the same set in a close view: one scan each way, so
    both observations of a close pair count."""
    close = set()
    for order in (obs_ids, obs_ids[::-1]):
        views = Views(where, conflict["distance_m"], conflict["yaw_deg"], conflict["floor_m"])
        for oid in order:
            if views.conflicts(oid):
                close.add(oid)
            views.add(oid)
    return round(len(close) / max(len(obs_ids), 1), 3)


def losses(cands, pairs, obs) -> dict:
    """Benchmark supply in observations before and after the image checks of the initial image: per dataset,
    task and hard-cell value, per camera, and of the radius pairs."""
    out = {}
    for d, tasks in cands.items():
        ok = obs[d]["bench_image"]
        for task, frame in tasks.items():
            frame = frame[frame["bench"].to_numpy()]
            field = {"A1": "collision", "A2": "position", "A4": "label", "B2": "label"}.get(task)
            for value, part in frame.groupby(field) if field else [("all", frame)]:
                ids = part["obs_id"].unique()
                out[f"{d}/{task}/{value}"] = [len(ids), int(ok.loc[ids].sum())]
        for camera, part in obs[d].groupby("camera"):
            out[f"{d}/camera/{camera}"] = [len(part), int(part["bench_image"].sum())]
        ids = pairs[d]["obs_id"].unique()
        out[f"{d}/radius_pairs"] = [len(ids), int(ok.loc[ids].sum())]
    return out


def review_shortfalls(shortfalls, obs, cands, quality):
    """Expose unresolved review coverage separately from a selection shortfall.

    Counts cover the dataset/task candidate pool, not a proof that reviewing them fills the cell.
    A greedy shortfall never proves that the original supply was insufficient.
    """
    for short in shortfalls:
        task, hashes = short["task"], []
        datasets = [short["dataset"]] if "dataset" in short else [d for d in cands if task in cands[d]]
        for dataset in datasets:
            pool = cands[dataset][task]
            if short["set"] != "sft":
                pool = pool[pool["bench"]]
            hashes += list(obs[dataset].loc[pool["obs_id"].unique(), "image_sha256"])
            if task == "C1":
                hashes += [Path(p).stem for d, p in quality.considered if d == dataset]
        gaps = {s: n for s, n in quality.reviews.summary(hashes).items() if s == "unreviewed"}
        short.update(reason="selection_shortfall",
                     review_gaps=gaps, review_gap_scope="dataset/task candidates")


def report(facts, obs, spec, extra) -> dict:
    """Distributions per set, dataset and task: scenes, views, actions, parameters and GT; baselines."""
    frame = pd.DataFrame(facts)
    everything = pd.concat(obs.values())
    where = {oid: (s, x, z, f, y) for oid, s, x, z, f, y in zip(
        everything.index, everything["scene"], everything["x"], everything["z"], everything["floor_y"],
        everything["yaw_deg"])}
    result = {"sets": {}, "items": {}, **extra}
    for kind, g in frame.groupby("set"):
        observations = list(dict.fromkeys(g["obs_id"]))
        result["sets"][kind] = {"items": len(g), "observations": len(observations), "scenes": int(g["scene"].nunique()),
                                "close_view_share": _close_views(observations, where, spec["benchmark"]["conflict"])}
    broad = set(spec["a3_broad_categories"])
    for (kind, dataset, task), g in frame.groupby(["set", "dataset", "task"]):
        scenes, answer_counts = Counter(g["scene"]), Counter(g["answer"])
        entry = {
            "items": len(g), "observations": int(g["obs_id"].nunique()), "scenes": len(scenes),
            "image_reviews": dict(Counter({sha: label for hashes, labels in zip(
                g["unmarked_image_sha256"], g["image_review"]) for sha, label in zip(hashes, labels)}.values())),
            "max_scene_share": round(max(scenes.values()) / len(g), 3),
            "frequency_baseline": round(answer_counts.most_common(1)[0][1] / len(g), 3),
            "unit": dict(Counter(g["unit"])), "radius": dict(Counter(g["radius"])), "camera": dict(Counter(g["camera"])),
            "representation": dict(Counter(g["representation"])),
            "commands": dict(sorted(Counter(len(a) for a in g["actions"]).items())),
            "forward_bin": dict(sorted(Counter(g["forward_bin"]).items())),
            "turn_bin": dict(sorted(Counter(g["turn_bin"]).items())), "turn_side": dict(Counter(g["turn_side"])),
            "start_turn_share": round(float((g["start"] == "turn").mean()), 3),
            "pattern": dict(Counter(_runs(a) for a in g["actions"]).most_common(12)),
            "program_repeats": int(len(g) - g["program_id"].nunique()),
        }
        if task in ("B1", "B3"):
            d = np.array([x["distance_m"] for x in g["gt"]])
            d0 = np.array([x["initial_distance_m"] for x in g["gt"]])
            entry.update(distance_bin=dict(sorted(Counter(g["bin"]).items())), change=dict(Counter(g["change"])),
                         initial_distance_baseline_within_0_25m=round(float(np.mean(np.abs(d - d0) <= .25)), 3))
        else:
            entry["answers"] = dict(answer_counts.most_common())
        if task == "A2":
            tied = g["longest"].isna()
            F = g.loc[~tied, "longest"].str.split(":")
            entry.update(position=dict(Counter(g["position"])), tied_lengths=int(tied.sum()),
                         longest_share_by_F={f: [round(float((s == "True").mean()), 3), round(1 / int(f[1:]), 3), len(s)]
                                             for f, s in F.str[1].groupby(F.str[0])},
                         chance_1_over_n=round(float(np.mean([1 / len(a) for a in g["actions"]])), 3),
                         chance_1_over_forwards=round(float(np.mean(
                             [1 / sum(x["type"] == "forward" for x in a) for a in g["actions"]])), 3))
        if task == "A3":
            entry["broad_category_share"] = round(float(g["answer"].isin(broad).mean()), 3)
        if task in ("A4", "B3"):
            entry["fraction"] = dict(Counter(g["fraction"]))
        if task == "C1":  # the least different option sets, to look at
            entry["least_different"] = [[gt["family_id"], round(v, 4)] for v, gt in sorted(
                zip(g["min_block_l1"], g["gt"]), key=lambda x: x[0])[:5]]
        if kind == "sft":
            cos = g["eval_cos"].dropna()
            entry["eval_similarity"] = {"compared": len(cos), "max": math.floor(cos.max() * 1e4) / 1e4 if len(cos) else None,  # shown cut, not rounded up
                                        "at_least_0.8": int((cos >= .8).sum())}
        result["items"][f"{kind}/{dataset}/{task}"] = entry
    return result
