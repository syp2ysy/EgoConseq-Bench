"""Steps 2 and 3, selection: the benchmark and its radius-pair diagnostic set, then the SFT from the rest.

Deterministic: candidates are visited in their seeded order (`_rank`) and every pick comes from the
stratum (combination of balanced fields) furthest below its targets. Answers that must be balanced are
hard cells, and every dataset and task has a minimum share of programs starting with a turn; scenes,
views, actions, parameters and the remaining GT fields are soft targets, availability checks and caps.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
from pathlib import Path

import numpy as np
import pandas as pd

from post_QA.build.candidates import ACTIONS, POSITIONS

UNIFORM = None  # target share spread evenly over the values present
LABEL_TASKS = ("A4", "B2")


def split(total: int, values) -> dict:
    """Integer quotas over values that differ by at most one, the remainder going to the first values."""
    values = list(values)
    base, extra = divmod(int(total), len(values))
    return {v: base + (i < extra) for i, v in enumerate(values)}


def capped(counts: dict, cap: float) -> dict:
    """Shares proportional to counts, none above cap (its excess spread over the others)."""
    shares = {k: v / sum(counts.values()) for k, v in counts.items()}
    fixed = set()
    while len(fixed) * cap < 1:
        rest = {k: v for k, v in shares.items() if k not in fixed}
        scale = (1 - cap * len(fixed)) / sum(rest.values())
        over = {k for k, v in rest.items() if v * scale > cap}
        if not over:
            return {**{k: v * scale for k, v in rest.items()}, **dict.fromkeys(fixed, cap)}
        fixed |= over
    return dict.fromkeys(shares, 1 / len(shares))


def actions(frame: pd.DataFrame, shares: dict) -> dict:
    """Targets for what the programs do, each field jointly with the representation: its values equally
    often within a representation, the representations at their shares (they differ in motion scale:
    the coarse programs are the long ones, so plain uniform fields would contradict the shares)."""
    goal = {}
    for field in ACTIONS:
        goal["representation", field] = {(rep, v): shares.get(rep, 0) / len(values) for rep, values
                                         in frame.groupby("representation")[field].unique().items() for v in values}
    return goal


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class Row:
    """One candidate, read lazily from the column arrays of its pool."""
    __slots__ = ("cols", "i")

    def __init__(self, cols, i):
        self.cols, self.i = cols, i

    def __getitem__(self, key):
        return self.cols[key][self.i]

    def to_dict(self):
        return {k: v[self.i] for k, v in self.cols.items()}


class Strata:
    """A pool grouped by balanced fields; `best` is the non-empty stratum furthest below its targets
    (ties go to the stratum whose next row comes first in the seeded order).

    A field is a column or a tuple of columns balanced jointly. Target shares are normalised over the
    values present; rows without a value in a field (None) are neutral for it.
    """

    def __init__(self, frame: pd.DataFrame, targets: dict):
        frame = frame.sort_values("_rank", kind="stable").reset_index(drop=True)
        self.cols = {c: frame[c].to_numpy() for c in frame.columns}
        self.dims = list(targets)
        columns = list(dict.fromkeys(c for dim in self.dims for c in (dim if isinstance(dim, tuple) else (dim,))))
        groups = ({} if not len(frame) else frame.groupby(columns, sort=True, dropna=False).indices if columns
                  else {(): np.arange(len(frame))})
        at = {c: i for i, c in enumerate(columns)}
        keys = [k if isinstance(k, tuple) else (k,) for k in groups]
        self.queues = [deque(v) for v in groups.values()]  # ascending positions = seeded order
        self.head = np.array([q[0] for q in self.queues], float)
        self.alive = np.ones(len(keys), bool)
        self.index = np.zeros((len(keys), len(self.dims)), int)
        self.positions, self.shares, self.counts = [], [], []
        for d, dim in enumerate(self.dims):
            values = [self._value(dim, lambda c: k[at[c]]) for k in keys]
            present = sorted({v for v in values if v is not None})
            goal = targets[dim] or dict.fromkeys(present, 1.)
            share = np.array([float(goal.get(v, 0.)) for v in present] + [0.])
            self.shares.append(share / max(share.sum(), 1e-12))
            self.counts.append(np.zeros(len(present) + 1))  # the last slot holds rows without a value
            self.positions.append({v: i for i, v in enumerate(present)})
            self.index[:, d] = [self.positions[d].get(v, -1) % (len(present) + 1) for v in values]

    @staticmethod
    def _value(dim, get):
        value = tuple(get(c) for c in (dim if isinstance(dim, tuple) else (dim,)))
        return None if any(_missing(v) for v in value) else value if isinstance(dim, tuple) else value[0]

    def best(self):
        if not self.alive.any():
            return None
        score = np.zeros(len(self.queues))
        for d, (share, counts) in enumerate(zip(self.shares, self.counts)):
            want = share * (counts[:-1].sum() + 1)
            gain = np.append(np.where(want[:-1] > 0, (want[:-1] - counts[:-1]) / np.maximum(want[:-1], 1e-12), -1.), 0.)
            score += gain[self.index[:, d]]
        score = np.where(self.alive, score, -np.inf)
        top = np.flatnonzero(score >= score.max() - 1e-9)
        return int(top[np.argmin(self.head[top])])

    def take(self, s):
        for d, counts in enumerate(self.counts):
            counts[self.index[s, d]] += 1

    def count(self, row):
        """Count a pick made from another pool of the same cell."""
        for d, dim in enumerate(self.dims):
            self.counts[d][self.positions[d].get(self._value(dim, lambda c: row[c]), -1)] += 1

    def next(self, s, available, prefer=None, lookahead=64):
        """First available row of stratum s, preferring `prefer` within the lookahead; unavailable rows go."""
        queue = self.queues[s]
        while queue and not available(Row(self.cols, queue[0])):
            queue.popleft()
        if not queue:
            self.alive[s], self.head[s] = False, np.inf
            return None
        row = None
        if prefer is not None:
            for position in range(min(lookahead, len(queue))):
                candidate = Row(self.cols, queue[position])
                if prefer(candidate) and available(candidate):
                    del queue[position]
                    row = candidate
                    break
        row = row or Row(self.cols, queue.popleft())
        self.head[s] = queue[0] if queue else np.inf
        return row


class Views:
    """Chosen observations on a grid per scene; a view conflicts within `distance` on the same floor and
    within `yaw` degrees of heading."""

    def __init__(self, where, distance, yaw, floor):
        self.where, self.distance, self.yaw, self.floor = where, distance, yaw, floor
        self.grid = defaultdict(list)

    def _cell(self, oid):
        scene, x, z, _, _ = self.where[oid]
        return scene, math.floor(x / self.distance), math.floor(z / self.distance)

    def add(self, oid):
        self.grid[self._cell(oid)].append(self.where[oid][1:])

    def conflicts(self, oid):
        scene, i, j = self._cell(oid)
        _, x, z, floor, yaw = self.where[oid]
        return any((a - x) ** 2 + (b - z) ** 2 < self.distance ** 2 and abs(f - floor) < self.floor
                   and abs((w - yaw + 180) % 360 - 180) < self.yaw
                   for di in (-1, 0, 1) for dj in (-1, 0, 1) for a, b, f, w in self.grid.get((scene, i + di, j + dj), ()))


def label_quotas(totals: dict, supply: dict):
    """Benchmark A4/B2: equal joint labels over all datasets, rarest labels placed first, each slot to a
    dataset that still has a spare observation of the label when one does (another task or a near view can take
    one).

    Returns the quotas per (dataset, label) and the labels whose equal share cannot be met.
    """
    per_label = split(sum(totals.values()), range(24))
    left, quotas, short = dict(totals), Counter(), {}
    for label in sorted(range(24), key=lambda l: (sum(supply[d][l] for d in totals), l)):
        for _ in range(per_label[label]):
            options = [d for d in totals if left[d] > 0 and supply[d][label] > quotas[d, label]]
            if not options:
                short[label] = per_label[label] - sum(quotas[d, label] for d in totals)
                break
            spare = [d for d in options if supply[d][label] - quotas[d, label] >= 2]  # keeps one observation spare
            d = max(spare or options, key=lambda d: (left[d] / totals[d], -list(totals).index(d)))
            quotas[d, label] += 1
            left[d] -= 1
    return quotas, short


def turn_quotas(cells, share):
    """Per cell, the picks starting with a turn that are taken first, so that the cells together hold at
    least `share` of them where the supply allows."""
    need = math.ceil(share * sum(q for _, q, _ in cells))
    supply = [min(q, int(pool.loc[pool["start"] == "turn", "obs_id"].nunique())) for pool, q, _ in cells]
    quotas = [min(math.ceil(share * q), s) for (_, q, _), s in zip(cells, supply)]
    for i in sorted(range(len(cells)), key=lambda i: supply[i] - quotas[i], reverse=True):
        extra = min(need - sum(quotas), supply[i] - quotas[i])
        if extra > 0:
            quotas[i] += extra
    return quotas


class Cell:
    """`quota` picks of one set, dataset and task from `frame`; `turns` of them, starting with a turn,
    are taken first. The soft targets count the picks of every cell of its set, dataset and task (group)."""

    def __init__(self, kind, dataset, task, name, frame, quota, turns, goal, cap, extra=None, on_pick=None):
        self.kind, self.dataset, self.task, self.name = kind, dataset, task, name
        self.frame, self.quota, self.turns, self.goal, self.cap = frame, quota, turns, goal, cap
        self.extra, self.on_pick = extra, on_pick
        self.supply = frame["obs_id"].nunique() / max(quota, 1)
        self.picks, self.scenes, self.group = [], Counter(), [self]

    def turned(self):
        return sum(row["start"] == "turn" for row in self.picks)


class Selector:
    """Selections of one build. obs: dataset -> observations; cands: dataset -> task -> candidates;
    pairs: dataset -> A1 radius pairs; dino: images.Dino; quality(meta path) -> dark fraction;
    caps: (dataset, task, label) -> the most A4/B2 questions of that label to ask of that dataset."""

    def __init__(self, obs, cands, pairs, spec, dino, quality, caps=None, *, prepare_reviews=False):
        self.obs, self.cands, self.pairs, self.spec, self.caps = obs, cands, pairs, spec, caps or {}
        self.dino, self.quality, self.similar = dino, quality, {}
        self.prepare_reviews = prepare_reviews
        everything = pd.concat(obs.values())
        self.where = {oid: (s, x, z, f, y) for oid, s, x, z, f, y in zip(
            everything.index, everything["scene"], everything["x"], everything["z"], everything["floor_y"],
            everything["yaw_deg"])}
        c = spec["benchmark"]["conflict"]
        self.taken = Views(self.where, c["distance_m"], c["yaw_deg"], c["floor_m"])  # benchmark and pairs
        self.used, self.rejected = set(), set()
        self.features, self.admitted = {}, defaultdict(list)  # DINOv2 features; (dataset, scene) -> obs_ids
        self.rows, self.shortfalls, self.cells, self.log = [], [], [], []

    def _pose(self, a, b):
        """Distances (inf on another floor) and heading differences between the observations a and b."""
        p, q = np.array([self.where[o][1:] for o in a]), np.array([self.where[o][1:] for o in b])
        dist = np.hypot(p[:, None, 0] - q[:, 0], p[:, None, 1] - q[:, 1])
        dist[np.abs(p[:, None, 2] - q[:, 2]) >= self.spec["benchmark"]["conflict"]["floor_m"]] = np.inf
        return dist, np.abs((p[:, None, 3] - q[:, 3] + 180) % 360 - 180)

    def _near(self, a, b):
        """Pose-near pairs of a and b: same floor, within the near-view distance and heading."""
        view = self.spec["benchmark"]["dino"]["near_view"]
        dist, turn = self._pose(a, b)
        return (dist <= view["distance_m"]) & (turn <= view["yaw_deg"])

    def _admit(self, dataset, oid):
        """Admit an evaluation observation unless the DINOv2 cosine of its initial image reaches the threshold
        to an admitted one of the same dataset and scene in a pose-near view."""
        self.features[oid] = feature = self.dino.embed(self.obs[dataset].at[oid, "path"])
        others = self.admitted[dataset, self.where[oid][0]]
        if others:
            near = [self.features[o] for o, n in zip(others, self._near([oid], others)[0]) if n]
            if near and float((self.dino.torch.stack(near) @ feature).max()) >= self.spec["benchmark"]["dino"]["threshold"]:
                return False
        others.append(oid)
        return True

    def goal(self, task, frame, *, benchmark):
        """Soft targets: actions, parameters and the GT fields that are not hard cells."""
        goal = {**actions(frame, self.spec["benchmark" if benchmark else "sft"]["representation"]), "camera": UNIFORM}
        if benchmark:  # the minimum of programs starting with a turn is taken first; the rest keeps this share
            first = self.spec["benchmark"]["turn_first_min"]
            goal["start"] = {"turn": first, "forward": 1 - first}
        if task in ("A1", "A2", "A3"):
            goal["radius_m"] = UNIFORM
        if task == "A2":  # forward counts at their supply; the colliding forward the unique longest at chance, 1/F
            known = frame.dropna(subset=["longest"])
            known = known.assign(F=known["longest"].str.split(":").str[0])
            forwards = Counter(known.drop_duplicates(["obs_id", "F"])["F"])
            goal["longest"] = {f"{F}:{flag}": n * (1 / int(F[1:]) if flag else 1 - 1 / int(F[1:]))
                               for F, n in forwards.items() for flag in (True, False)}
            groups = Counter(frame.drop_duplicates(["obs_id", "fgroup"])["fgroup"])
            if not benchmark:  # positions equal within each group (benchmark: hard cells)
                goal["position"] = {name: groups[g] / len(p) for g, p in POSITIONS.items() for name in p}
        if task == "A3":  # natural category shares by observation, frequent categories capped
            goal["category"] = capped(Counter(frame.drop_duplicates(["obs_id", "category"])["category"]),
                                      self.spec["a3_category_cap"])
        if task in ("A4", "B3"):
            goal["fraction"] = UNIFORM
        if task in ("B1", "B3"):
            goal.update(bin=UNIFORM, change=UNIFORM)
        if not benchmark:  # the SFT balances answers softly; the benchmark uses hard cells
            goal.update({"A1": {"collision": UNIFORM}, "A4": {"label": UNIFORM}, "B2": {"label": UNIFORM}}.get(task, {}))
        return goal

    def _decorate(self, dataset, frame):
        o = self.obs[dataset].loc[frame["obs_id"]]
        return frame.assign(camera=o["camera"].to_numpy(), scene=o["scene"].to_numpy())

    def _options_ok(self, dataset, row, strict):
        """Every C1 option passes the image checks (the initial image passed with its observation);
        otherwise the family is not used."""
        self.quality.considered.update((dataset, p) for p in row["options"])
        for path in row["options"]:
            rejected = self.quality.rejection(path, Path(path).stem, self.spec, strict)
            if rejected and rejected[0] == "unreviewed" and self.prepare_reviews:
                # Only provisional option selection permits missing reviews; final checks never do.
                dark = self.quality(path)
                limit = self.spec["benchmark" if strict else "sft"]["max_dark_fraction"]
                rejected = ("dark", dark) if dark > limit else None
            if rejected:
                if not strict:
                    self.log.append({"kind": "family", "dataset": dataset, "id": row["family_id"], "image": path,
                                     "reason": rejected[0], "value": rejected[1], "eval_image": None})
                return False
        return True

    def _eligible(self, dataset, task):
        frame = self.cands[dataset][task]
        return frame[frame["bench"].to_numpy() & self.obs[dataset].loc[frame["obs_id"], "bench_image"].to_numpy()]

    # -- benchmark and the radius-pair diagnostic set ----------------------------------------------
    def _cell(self, kind, dataset, task, name, frame, quota, turns, goal=None, **hooks):
        frame = self._decorate(dataset, frame)
        cap = math.ceil(self.spec["benchmark"]["scene_cap_factor"] * quota / max(1, frame["scene"].nunique()))
        return Cell(kind, dataset, task, name, frame, quota, turns, goal or self.goal(task, frame, benchmark=True),
                    cap, **hooks)

    def _fill(self, cell, n, *, turn_only=False, cap=None):
        """Up to n more picks of a cell (only programs starting with a turn if turn_only): unused scenes
        first, then up to the scene cap; no observation twice, no two close views, no DINO near duplicate.
        Returns the number of picks."""
        start = len(cell.picks)
        if n <= 0:
            return 0
        strata = Strata(cell.frame[cell.frame["start"].to_numpy() == "turn"] if turn_only else cell.frame, cell.goal)
        for other in cell.group:
            for row in other.picks:
                strata.count(row)
        cap = cell.cap if cap is None else cap

        def available(row):
            oid = row["obs_id"]
            return (oid not in self.used and oid not in self.rejected and cell.scenes[row["scene"]] < cap
                    and not self.taken.conflicts(oid) and (cell.extra is None or cell.extra(row)))
        while len(cell.picks) < start + n and (s := strata.best()) is not None:
            row = strata.next(s, available, prefer=lambda r: cell.scenes[r["scene"]] == 0)
            if row is None or (cell.task == "C1" and not self._options_ok(cell.dataset, row, True)):
                continue
            oid = row["obs_id"]
            if not self._admit(cell.dataset, oid):
                self.rejected.add(oid)
                continue
            self.used.add(oid)
            self.taken.add(oid)
            cell.scenes[row["scene"]] += 1
            cell.picks.append(row.to_dict())
            strata.take(s)
            if cell.on_pick:
                cell.on_pick(row)
        return len(cell.picks) - start

    def _run(self, cells):
        """Fill the cells, scarcest first: the programs starting with a turn, then the rest, then (a cell
        still short) beyond the scene cap. Turn picks a cell cannot make go to the other cells of its set,
        dataset and task, whose minimum share is what counts."""
        share = self.spec["benchmark"]["turn_first_min"]
        order = sorted(cells, key=lambda c: (c.supply, c.dataset, c.task, c.name))
        groups = defaultdict(list)
        for c in cells:
            groups[c.kind, c.dataset, c.task].append(c)
            c.group = groups[c.kind, c.dataset, c.task]
        for c in order:
            self._fill(c, c.turns, turn_only=True)
        for group in groups.values():
            need = math.ceil(share * sum(c.quota for c in group)) - sum(c.turned() for c in group)
            for c in sorted(group, key=lambda c: -c.supply):
                need -= self._fill(c, min(need, c.quota - len(c.picks)), turn_only=True)
        for c in order:
            self._fill(c, c.quota - len(c.picks))
        for c in order:
            if len(c.picks) < c.quota:
                self._fill(c, c.quota - len(c.picks), cap=math.inf)
        for c in cells:
            self.cells.append({"set": c.kind, "dataset": c.dataset, "task": c.task, "cell": c.name, "quota": c.quota})
            self.rows += [{**row, "set": c.kind, "dataset": c.dataset, "task": c.task, "cell": c.name} for row in c.picks]
            if len(c.picks) < c.quota:
                self.shortfalls.append({"set": c.kind, "dataset": c.dataset, "task": c.task, "cell": c.name,
                                        "quota": c.quota, "picked": len(c.picks)})
        for (kind, dataset, task), group in groups.items():
            missing = math.ceil(share * sum(c.quota for c in group)) - sum(c.turned() for c in group)
            if missing > 0:
                self.shortfalls.append({"set": kind, "dataset": dataset, "task": task, "cell": "turn_first",
                                        "missing": missing})

    def benchmark(self):
        totals, share = self.spec["benchmark"]["totals"], self.spec["benchmark"]["turn_first_min"]
        groups = defaultdict(list)  # (dataset, task) -> cells (pool, quota, name)
        for task in LABEL_TASKS:
            mine = {d: t[task] for d, t in totals.items() if task in t and d in self.cands}
            pools = {d: self._eligible(d, task) for d in mine}
            supply = {d: Counter(p.drop_duplicates(["obs_id", "label"])["label"]) for d, p in pools.items()}
            for (d, t, label), most in self.caps.items():
                if t == task and d in supply:
                    supply[d][label] = min(supply[d][label], most)
            quotas, short = label_quotas(mine, supply)
            for label, missing in short.items():
                self.shortfalls.append({"set": "benchmark", "task": task, "cell": f"label={label}", "missing": missing})
            for (d, label), q in sorted(quotas.items()):
                groups[d, task].append((pools[d][pools[d]["label"] == label], q, f"label={label}"))
        for d, tasks in totals.items():
            for task, quota in tasks.items():
                if d not in self.cands or task in LABEL_TASKS:
                    continue
                pool = self._eligible(d, task)
                if task == "A1":
                    groups[d, task] += [(pool[pool["collision"] == v], q, f"collision={v}")
                                        for v, q in split(quota, [True, False]).items()]
                elif task == "A2":  # forward-count groups by supply; positions equal within each group
                    n = pool.drop_duplicates(["obs_id", "fgroup"])["fgroup"].value_counts()
                    sizes = {"F2": round(quota * n.get("F2", 0) / max(n.sum(), 1))}
                    sizes["F3+"] = quota - sizes["F2"]
                    for g, names in POSITIONS.items():
                        groups[d, task] += [(pool[pool["position"] == name], q, name)
                                            for name, q in split(sizes[g], names).items()]
                else:
                    groups[d, task].append((pool, quota, "all"))
        self._run([self._cell("benchmark", d, task, name, pool, q, t)
                   for (d, task), group in groups.items()
                   for (pool, q, name), t in zip(group, turn_quotas(group, share))])

    def radius_pairs(self):
        spec, programs, cells = self.spec["radius_pairs"], Counter(), []
        bench = self.spec["benchmark"]
        for d, pairs in self.pairs.items():
            pool = pairs[self.obs[d].loc[pairs["obs_id"], "bench_image"].to_numpy()]
            pool = pool.assign(radii_pair=[f"{a:g}-{b:g}" for a, b in zip(pool["r_free"], pool["r_collide"])])
            goal = {**actions(pool, bench["representation"]), "camera": UNIFORM, "radii_pair": UNIFORM,
                    "start": {"turn": bench["turn_first_min"], "forward": 1 - bench["turn_first_min"]}}
            quota = spec["pairs_per_dataset"]
            turns = turn_quotas([(pool, quota, "pairs")], bench["turn_first_min"])[0]
            cells.append(self._cell("radius_pairs", d, "A1", "pairs", pool, quota, turns, goal,
                                    extra=lambda row: programs[row["program_id"]] < spec["program_cap"],
                                    on_pick=lambda row: programs.update([row["program_id"]])))
        self._run(cells)

    # -- SFT ---------------------------------------------------------------------------------------
    def exclusions(self):
        """Observations the SFT may not use; the evaluation observations (benchmark and pair set) are fixed.
        In this order, the first reason logged: the image checks; within exclusion_m of an evaluation
        observation (same scene and floor, any heading); the same initial image; a DINOv2 cosine of at least
        the threshold to the initial image of a pose-near evaluation observation (as for the benchmark).
        self.similar: obs_id -> (evaluation image, cosine) of every observation compared."""
        spec, threshold = self.spec["sft"], self.spec["benchmark"]["dino"]["threshold"]
        view = self.spec["benchmark"]["dino"]["near_view"]
        everything = pd.concat([o.assign(dataset=d) for d, o in self.obs.items()])
        evaluation = everything.loc[sorted(self.used)]
        rest = everything.drop(index=evaluation.index)
        bad = rest[~rest["sft_image"]]
        out = [(oid, r.sft_image_reason, r.sft_image_value, None)
               for oid, r in bad.iterrows()]
        rest = rest[rest["sft_image"]]
        same = rest["image_sha256"].map(dict(zip(evaluation["image_sha256"], evaluation["image"]))).dropna()
        out += [(oid, "same_image", 1., image) for oid, image in same.items()]
        rest = rest.drop(index=same.index)
        scenes, compared = evaluation.groupby(["dataset", "scene"]).indices, []
        for scene, q in rest.groupby(["dataset", "scene"]).indices.items():
            if scene not in scenes:
                continue
            a, b = rest.iloc[q], evaluation.iloc[scenes[scene]]
            dist, turn = self._pose(a.index, b.index)
            nearest = dist.argmin(axis=1)
            close = dist[np.arange(len(a)), nearest] < spec["exclusion_m"]
            out += [(a.index[i], "spatial", dist[i, nearest[i]], b["image"].iloc[nearest[i]])
                    for i in np.flatnonzero(close)]
            mask = (dist <= view["distance_m"]) & (turn <= view["yaw_deg"]) & ~close[:, None]
            rows = np.flatnonzero(mask.any(axis=1))
            if len(rows):
                compared.append((a.iloc[rows], b, mask[rows]))
        if compared:  # on the GPU: the pose-near candidates' features against the evaluation features of the build
            torch = self.dino.torch
            ids = pd.concat([a for a, _, _ in compared])
            feature = dict(zip(ids.index, self.dino.features(ids["path"])))
            for a, b, mask in compared:
                cos = torch.stack([feature[i] for i in a.index]) @ torch.stack([self.features[i] for i in b.index]).T
                cos[~torch.from_numpy(mask).to(cos.device)] = -1
                best, j = cos.max(dim=1)
                for oid, c, k in zip(a.index, best.tolist(), j.tolist()):
                    self.similar[oid] = (b["image"].iloc[k], c)
                    if c >= threshold:
                        out.append((oid, "dino", c, b["image"].iloc[k]))
        self.log += [{"kind": "observation", "dataset": everything.at[oid, "dataset"], "id": oid,
                      "image": everything.at[oid, "image"], "reason": reason, "value": float(value), "eval_image": image}
                     for oid, reason, value, image in out]
        return set(self.used) | {oid for oid, *_ in out}

    def sft(self):
        spec, c = self.spec["sft"], self.spec["benchmark"]["conflict"]
        excluded = self.exclusions()
        tasks = sorted({t for cands in self.cands.values() for t in cands})
        plan = {}
        for task in tasks:  # split over the datasets by the observations left after the exclusions
            sizes = {d: cands[task].loc[~cands[task]["obs_id"].isin(excluded).to_numpy(), "obs_id"].nunique()
                     for d, cands in self.cands.items() if task in cands}
            for d, n in sizes.items():
                plan[d, task] = round(spec["total"] / len(tasks) * n / sum(sizes.values()))
        place = {oid: (s, round(f), math.floor(x / spec["exclusion_m"]), math.floor(z / spec["exclusion_m"]))
                 for oid, (s, x, z, f, _) in self.where.items()}
        views = Views(self.where, c["distance_m"], c["yaw_deg"], c["floor_m"])  # distinct views of the SFT
        per_obs, keys, near, picked = Counter(), set(), Counter(), Counter()
        items = defaultdict(list)  # (dataset, task) -> the items picked, counted by every later round

        def accept(d, task, row, n=1):
            oid, values = row["obs_id"], row.to_dict()
            if not per_obs[oid]:
                views.add(oid)
            per_obs[oid] += n
            near[place[oid]] += n
            keys.add((oid, task, row["program_id"]))
            picked[d, task] += n
            self.rows.append({**values, "set": "sft", "dataset": d, "task": task, "cell": "sft"})
            items[d, task] += ([{**values, "radius_m": values["r_free"], "collision": False},  # a pair: two items
                                {**values, "radius_m": values["r_collide"], "collision": True}] if n == 2 else [values])

        def pool(d, frame):
            return self._decorate(d, frame[~frame["obs_id"].isin(excluded).to_numpy()])

        for d, cands in self.cands.items():  # C1 options measured ahead in parallel; any others when reached
            if "C1" in cands:
                frame = pool(d, cands["C1"]).sort_values("_rank").head(round(1.5 * plan[d, "C1"]))
                self.quality.prefetch(o for row in frame["options"] for o in row
                                      if self.quality.reviews.eligible(Path(o).stem)
                                      or self.prepare_reviews and self.quality.reviews.status(Path(o).stem) == "unreviewed")

        for d, pairs in self.pairs.items():  # radius pairs first: units of two on new views
            goal = 2 * round(spec["radius_pair_share_of_a1"] * plan[d, "A1"] / 2)
            frame = pool(d, pairs)
            strata = Strata(frame, {**actions(frame, spec["representation"]), "camera": UNIFORM})
            while picked[d, "A1"] < goal and (s := strata.best()) is not None:
                row = strata.next(s, lambda r: not per_obs[r["obs_id"]] and not views.conflicts(r["obs_id"])
                                  and near[place[r["obs_id"]]] + 2 <= spec["neighbourhood_cap"], lookahead=0)
                if row is not None:
                    strata.take(s)
                    accept(d, "A1", row, 2)

        def run(limit, cap, goals, distinct):
            streams = {key: Strata(frame, self.goal(key[1], frame, benchmark=False))
                       for key in goals for frame in [pool(key[0], self.cands[key[0]][key[1]])]}
            for key, strata in streams.items():  # balance the whole set so far, not only this round
                for values in items[key]:
                    strata.count(values)
            active = True
            while active:
                active = False
                for (d, task), strata in streams.items():
                    def available(row, task=task):
                        oid = row["obs_id"]
                        return (per_obs[oid] < limit and near[place[oid]] < cap
                                and (oid, task, row["program_id"]) not in keys
                                and not (distinct and views.conflicts(oid)))
                    for _ in range(1000):  # tasks take turns so that each reaches new places
                        if picked[d, task] >= goals[d, task] or (s := strata.best()) is None:
                            break
                        active = True  # a row is used or a stratum runs out: the round goes on until neither
                        row = strata.next(s, available, lookahead=0)
                        if row is None or (task == "C1" and not self._options_ok(d, row, False)):
                            continue
                        strata.take(s)
                        accept(d, task, row)

        run(1, spec["neighbourhood_cap"], plan, True)  # new views first
        run(spec["per_observation"], 2 * spec["neighbourhood_cap"], plan, False)  # then other tasks and programs
        missing = spec["total"] - sum(picked.values())
        full = {k: v for k, v in plan.items() if picked[k] >= v}
        if missing > 0 and full:  # a short task's remainder goes to the others, in proportion
            run(spec["per_observation"], 2 * spec["neighbourhood_cap"],
                {k: v + round(missing * v / sum(full.values())) for k, v in full.items()}, False)
        for key, goal in plan.items():
            if picked[key] < goal:
                self.shortfalls.append({"set": "sft", "dataset": key[0], "task": key[1], "quota": goal,
                                        "picked": picked[key]})
        return excluded
