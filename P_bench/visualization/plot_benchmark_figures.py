#!/usr/bin/env python3
"""Editable paper figures, computed from the current benchmark QA and metadata.

Run: python visualization/plot_benchmark_figures.py
Only writes figure artifacts under --output; never modifies benchmark data.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]
# Edit names, ordering, and colors here. IDs are supplementary, never sole labels.
TASKS = {
    "A1": ("Collision occurrence", "contact"),
    "A2": ("First-collision action", "contact"),
    "A3": ("First-contact object category", "contact"),
    "A4": ("Intermediate target direction", "relations"),
    "B1": ("Final target distance", "relations"),
    "B2": ("Final target direction", "relations"),
    "B3": ("Intermediate target distance", "relations"),
    "C1": ("Future-view selection", "view"),
}
GROUPS = {
    "contact": ("Body–scene contact", "#BB6146"),
    "relations": ("Target-relative spatial relations", "#307F9F"),
    "view": ("Future-view discrimination", "#8270AF"),
}
TASK_COLORS = ["#B75940", "#D17B5D", "#E4A68C", "#287894", "#4593AF",
               "#6EACC2", "#A0CAD8", "#8F7ABA"]
SPLITS = ("seen", "unseen")
SPLIT_COLORS = {"seen": "#307F9F", "unseen": "#C97F54"}
SOURCES = {"b1k": "BEHAVIOR-1K", "r2r": "R2R", "gs": "InteriorGS"}
HORIZONTAL = ("front", "front-right", "right", "rear-right", "rear",
              "rear-left", "left", "front-left")
VERTICAL = ("above", "level", "below")
INK, MUTED = "#243547", "#647383"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#C3CCD3", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "medium",
    "axes.titlesize": 13, "axes.titlepad": 14,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def label(task, width=29):
    return fill(TASKS[task][0], width) + f" ({task})"


def load_data(benchmark):
    rows, hashes = {}, {}
    for split in SPLITS:
        qp = benchmark / "benchmark" / split / "QA.json"
        ip = benchmark / "metadata" / split / "record_index.json"
        qa = json.loads(qp.read_text())
        index = json.loads(ip.read_text())
        # Require metadata to match this QA release, not a stale record index.
        digest = hashlib.sha256(qp.read_bytes()).hexdigest()
        if index.get("qa_sha256") != digest:
            raise ValueError(f"{split}: record index does not match QA.json")
        by_id = {r["item_id"]: r for r in index["items"]}
        assert len(by_id) == len(index["items"]), "Duplicate metadata IDs"
        assert len({q["id"] for q in qa}) == len(qa), "Duplicate QA IDs"
        rows[split] = []
        for q in qa:
            r = by_id[q["id"]]
            assert q["task_id"] in TASKS and q["task_id"] == r["task_id"]
            assert q["dataset"] == r["dataset"]
            answer = next(m["content"] for m in q["messages"] if m["role"] == "assistant")
            rows[split].append({**r, "answer": answer})
        hashes[split] = {"QA.json": digest,
                         "record_index.json": hashlib.sha256(ip.read_bytes()).hexdigest()}
    return rows, hashes


def summarize(rows, hashes):
    data = {"input_sha256": hashes, "splits": {}, "tasks": {},
            "units": {"counts": "QA items unless explicitly stated otherwise",
                      "distance_m": "Published GT: 3D camera-optical-center to fixed target point",
                      "action_length": "Full given sequence: Forward and Turn each count as one action",
                      "query_fraction": "Distance fraction within the specified Forward action"}}
    for task, (name, group) in TASKS.items():
        data["tasks"][task] = {"name": name, "group": GROUPS[group][0]}
    for split, rr in rows.items():
        distances = {}
        for task in ("B1", "B3"):
            values = []
            for r in rr:
                if r["task_id"] == task:
                    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*m\s*", r["answer"])
                    if not match:
                        raise ValueError(f"Unexpected distance GT: {r['answer']}")
                    values.append(float(match[1]))
            distances[task] = values
        config_counts = Counter((r["usage"]["body_radius_m"], r["usage"]["camera_height_m"],
                                 r["usage"]["hfov_deg"]) for r in rr)
        assert set(config_counts) <= {(r, h, f) for r in (.15, .2, .25)
                                     for h in (.5, 1., 1.5) for f in (79., 110.)}
        assert set(r["usage"]["action_length"] for r in rr) <= set(range(1, 7))
        task_counts = Counter(r["task_id"] for r in rr)
        directions = {}
        for task in ("A4", "B2"):
            for r in rr:
                if r["task_id"] == task:
                    rel = r["usage"]["target_point"]["relation"]
                    expected = rel["horizontal_direction"]
                    if rel["vertical_direction"] != "level":
                        expected += " and " + rel["vertical_direction"]
                    actual = r["answer"].strip().rstrip(".").lower().replace("behind", "rear")
                    assert actual == expected, f"Direction GT mismatch: {r['item_id']}"
            cc = Counter((r["usage"]["target_point"]["relation"]["horizontal_direction"],
                          r["usage"]["target_point"]["relation"]["vertical_direction"])
                         for r in rr if r["task_id"] == task)
            matrix = [[cc[h, v] for h in HORIZONTAL] for v in VERTICAL]
            assert sum(map(sum, matrix)) == task_counts[task]
            directions[task] = matrix
        d = {
            "qa_count": len(rr),
            "scene_count": len({(r["dataset"], r["scene_id"]) for r in rr}),
            "task_counts": dict(task_counts),
            "source_counts": dict(Counter(r["dataset"] for r in rr)),
            "source_task_counts": {src: {t: sum(r["dataset"] == src and r["task_id"] == t for r in rr)
                                          for t in TASKS} for src in SOURCES},
            "action_length_counts": dict(sorted(Counter(r["usage"]["action_length"] for r in rr).items())),
            "configurations": [{"radius_m": r, "height_m": h, "horizontal_fov_deg": f, "count": n}
                               for (r, h, f), n in sorted(config_counts.items())],
            "query_fraction_counts": {t: {str(f): sum(r["task_id"] == t and
                                        r["usage"].get("checkpoint", {}).get("fraction") == f for r in rr)
                                          for f in (.25, .5, .75)} for t in ("A4", "B3")},
            "distance_gt_m": distances,
            "direction_counts": directions,
            "collision_labels": dict(Counter(r["answer"].strip().rstrip(".") for r in rr if r["task_id"] == "A1")),
            "future_view_labels": dict(Counter(r["answer"].strip() for r in rr if r["task_id"] == "C1")),
        }
        assert sum(d["task_counts"].values()) == sum(d["source_counts"].values()) == len(rr)
        assert sum(c["count"] for c in d["configurations"]) == len(rr)
        for t in ("A4", "B3"):
            assert sum(d["query_fraction_counts"][t].values()) == task_counts[t]
        assert sum(d["collision_labels"].values()) == task_counts["A1"]
        assert sum(d["future_view_labels"].values()) == task_counts["C1"]
        data["splits"][split] = d
    data["qa_count"] = sum(d["qa_count"] for d in data["splits"].values())
    data["scene_count"] = len({(r["dataset"], r["scene_id"]) for rr in rows.values() for r in rr})
    return data


def save(fig, name, out, dpi):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def box(ax, x, y, w, h, color, face="white", linewidth=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.009",
                               linewidth=linewidth, edgecolor=color, facecolor=face))


def heatmap(ax, values, cmap, vmax):
    """Use vector cells, so SVG/PDF heatmaps remain editable shapes."""
    nr, nc = values.shape
    ax.pcolormesh(np.arange(nc+1)-.5, np.arange(nr+1)-.5, values,
                  cmap=cmap, vmin=0, vmax=vmax, edgecolors="white", linewidth=.45)
    ax.set(xlim=(-.5, nc-.5), ylim=(nr-.5, -.5))


def taxonomy(out, dpi):
    fig, ax = plt.subplots(figsize=(18, 10.5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    inputs = [(.02, "Current egocentric observation", "Visual evidence of the static scene"),
              (.35, "Body and camera configuration", "Footprint radius · camera height · field of view"),
              (.68, "Given motion sequence", "Forward distances · turn angles and directions")]
    for x, title, desc in inputs:
        box(ax, x, .87, .30, .11, "#CAD3DC", "#F6F8FA")
        ax.text(x+.15, .947, title, ha="center", va="center", fontsize=12.3, weight="medium")
        ax.text(x+.15, .903, desc, ha="center", va="center", fontsize=9.6, color=MUTED)
        ax.add_patch(FancyArrowPatch((x+.15, .86), (.5, .822), arrowstyle="-", color="#A9B5BF", lw=1))
    ax.text(.5, .79, "Spatial prediction under ego-motion", fontsize=19, ha="center", va="center")
    ax.text(.5, .75, "Predict before execution · no intermediate execution observations", fontsize=11, color=MUTED, ha="center")

    regions = [("contact", .02, .27), ("relations", .32, .41), ("view", .76, .22)]
    for group, x, w in regions:
        name, color = GROUPS[group]
        box(ax, x, .14, w, .55, color, "#FCFCFD")
        ax.text(x+w/2, .648, fill(name, 28), color=color, fontsize=14, weight="medium", ha="center", va="center")

    def card(task, x, y, w, h, question, answer):
        color = GROUPS[TASKS[task][1]][1]
        box(ax, x, y, w, h, "#DCE2E7")
        width = 25 if w < .21 else 33
        ax.text(x+.012, y+h-.014, label(task, width), fontsize=11.1, weight="medium", color=color, va="top", linespacing=1.2)
        ax.text(x+.012, y+.045 if h < .15 else y+.054,
                fill(question, 33 if w < .21 else 43), fontsize=9.5, va="bottom", linespacing=1.3)
        ax.text(x+.012, y+.016, answer, fontsize=8.7, color=MUTED, va="bottom")

    card("A1", .032, .462, .246, .121, "Will the body collide with any object along the given motion?", "Answer: Collision / No collision")
    card("A2", .032, .317, .246, .121, "At which listed action does the first collision occur?", "Answer: action index")
    card("A3", .032, .172, .246, .121, "What category of object is contacted first?", "Answer: object category")
    ax.text(.155, .603, "Entire motion · collision given for A2 / A3", fontsize=8.6, ha="center", color=MUTED)

    ax.text(.42, .592, "INTERMEDIATE POSITION", fontsize=9.2, ha="center", color=MUTED)
    ax.text(.63, .592, "FINAL POSITION", fontsize=9.2, ha="center", color=MUTED)
    card("B3", .333, .371, .182, .19, "How far is the fixed target from the camera at the query position?", "Answer: 3D distance in meters")
    card("B1", .538, .371, .182, .19, "How far is the fixed target from the camera after all actions?", "Answer: 3D distance in meters")
    card("A4", .333, .172, .182, .178, "In which direction is the fixed target at the query position?", "Answer: horizontal + vertical direction")
    card("B2", .538, .172, .182, .178, "In which direction is the fixed target after all actions?", "Answer: horizontal + vertical direction")

    card("C1", .772, .36, .196, .215, "Which candidate image matches the camera view after all actions?", "Answer: one of four candidate views")
    # Candidate rectangles express alternatives, not successive video frames.
    for j in range(4):
        xx = .787 + j*.044
        box(ax, xx, .275, .032, .053, GROUPS["view"][1], "#F1EDF8")
        ax.text(xx+.016, .301, "ABCD"[j], ha="center", va="center", color=GROUPS["view"][1])
    ax.text(.87, .237, "Alternative final views", ha="center", fontsize=10, color=MUTED)
    ax.text(.87, .194, "Recognition from candidates", ha="center", fontsize=9, color=MUTED)
    ax.text(.5, .094, "Intermediate queries: after earlier actions and 25%, 50%, or 75% of one specified Forward action.", ha="center", fontsize=10, color=MUTED)
    ax.text(.5, .055, "Target relations use the camera frame at the queried pose. Relation and future-view sequences are collision-free.", ha="center", fontsize=10, color=MUTED)
    save(fig, "01_task_taxonomy", out, dpi)


def task_distribution(data, out, dpi):
    fig = plt.figure(figsize=(18, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=.56)
    ax = fig.add_subplot(gs[0, 0]); bars = fig.add_subplot(gs[0, 1])
    counts = [sum(data["splits"][s]["task_counts"][t] for s in SPLITS) for t in TASKS]
    total = sum(counts)
    groups = [sum(n for n, t in zip(counts, TASKS) if TASKS[t][1] == g) for g in GROUPS]
    wedges, _ = ax.pie(counts, radius=1, startangle=90, counterclock=False,
                       colors=TASK_COLORS, wedgeprops={"width": .32, "edgecolor": "white", "linewidth": 1.5})
    inner, _ = ax.pie(groups, radius=.68, startangle=90, counterclock=False,
                      colors=[v[1] for v in GROUPS.values()], wedgeprops={"width": .31, "edgecolor": "white", "linewidth": 1.5})
    for w, (group, (name, color)), n in zip(inner, GROUPS.items(), groups):
        angle = np.deg2rad((w.theta1+w.theta2)/2)
        ax.text(.525*np.cos(angle), .525*np.sin(angle), f"{100*n/total:.1f}%", ha="center", va="center", fontsize=12, color="white")
    ax.text(0, .055, f"{total:,}", fontsize=24, ha="center", va="center")
    ax.text(0, -.11, "QA items", fontsize=10, color=MUTED, ha="center")
    for wedge, t, n in zip(wedges, TASKS, counts):
        angle = np.deg2rad((wedge.theta1+wedge.theta2)/2)
        xx, yy = np.cos(angle), np.sin(angle)
        side = 1 if xx > 0 else -1
        ax.annotate(f"{label(t, 24)}\n{n:,} · {100*n/total:.1f}%", xy=(xx, yy), xytext=(side*1.2, yy*1.23),
                    ha="left" if side > 0 else "right", va="center", fontsize=9.7,
                    arrowprops={"arrowstyle": "-", "color": "#AAB7C1", "lw": .9,
                                "connectionstyle": f"angle,angleA=0,angleB={np.rad2deg(angle)}"})
    ax.set(xlim=(-1.9, 1.9), ylim=(-1.55, 1.55), aspect="equal")
    ax.set_title("Task composition", loc="center", pad=22)
    ax.legend(handles=[Patch(facecolor=color, label=name) for name, color in GROUPS.values()],
              loc="upper center", bbox_to_anchor=(.5, .045), frameon=False, fontsize=10,
              handlelength=1.25, labelspacing=.65)
    ys = np.arange(len(TASKS))
    for offset, split in zip((-.18, .18), SPLITS):
        vals = [data["splits"][split]["task_counts"][t] for t in TASKS]
        b = bars.barh(ys+offset, vals, height=.32, color=SPLIT_COLORS[split], label=split.title())
        bars.bar_label(b, padding=4, fontsize=10)
    bars.set_yticks(ys, [label(t, 26) for t in TASKS], fontsize=10)
    bars.invert_yaxis(); bars.set_xlim(0, max(counts)*.95)
    bars.set_xlabel("Number of QA items")
    bars.set_title("Question counts by evaluation split", loc="left")
    bars.legend(frameon=False, ncol=2, loc="lower right")
    bars.grid(axis="x", alpha=.14); bars.set_axisbelow(True)
    fig.text(.5, -.035, f"Inner ring: prediction families. Outer ring: eight tasks. Every percentage uses all {total:,} QA items as its denominator.", ha="center", fontsize=10, color=MUTED)
    save(fig, "02_task_distribution", out, dpi)


def conditions(data, out, dpi):
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=.78, wspace=.52)
    cmap = LinearSegmentedColormap.from_list("configs", ["#F4F8FA", "#B6D1DB", "#307F9F"])
    max_n = max(c["count"] for s in SPLITS for c in data["splits"][s]["configurations"])
    heights, radii = [.5, 1, 1.5], [.15, .2, .25]
    for col, (split, fov) in enumerate((s, f) for s in SPLITS for f in (79, 110)):
        ax = fig.add_subplot(gs[0, col])
        cc = {(c["radius_m"], c["height_m"], c["horizontal_fov_deg"]): c["count"]
              for c in data["splits"][split]["configurations"]}
        arr = np.array([[cc.get((r, h, fov), 0) for r in radii] for h in heights])
        heatmap(ax, arr, cmap, max_n)
        for (i, j), n in np.ndenumerate(arr):
            ax.text(j, i, str(n), ha="center", va="center", color="white" if n/max_n > .68 else INK, fontsize=12)
        ax.set_xticks(range(3), [f"{r:.2f}" for r in radii]); ax.set_yticks(range(3), [f"{h:.1f}" for h in heights])
        ax.set_xlabel("Footprint radius (m)"); ax.set_ylabel("Camera height (m)")
        ax.set_title(f"{split.title()} · horizontal FOV {fov}°", fontsize=11.5)
    ax = fig.add_subplot(gs[1, :2])
    xx = np.arange(1, 7)
    for offset, split in zip((-.18, .18), SPLITS):
        d = data["splits"][split]
        nn = [d["action_length_counts"].get(int(x), 0) for x in xx]
        bars = ax.bar(xx+offset, np.array(nn)/d["qa_count"]*100, .34, color=SPLIT_COLORS[split], label=f"{split.title()} (N={d['qa_count']:,})")
        ax.bar_label(bars, labels=[str(n) for n in nn], padding=3, fontsize=9)
    ax.set(xticks=xx, xlabel="Actions in the full given sequence", ylabel="QA items within each split (%)", ylim=(0, 24))
    ax.set_title("Given motion length", loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[1, 2:])
    xx = np.arange(3)
    for offset, split in zip((-.18, .18), SPLITS):
        d = data["splits"][split]
        nn = [d["source_counts"][src] for src in SOURCES]
        bars = ax.bar(xx+offset, np.array(nn)/d["qa_count"]*100, .34, color=SPLIT_COLORS[split], label=split.title())
        ax.bar_label(bars, labels=[f"{n:,}" for n in nn], padding=3, fontsize=10)
    ax.set(xticks=xx, xticklabels=list(SOURCES.values()), ylabel="QA items within each split (%)", ylim=(0, 48))
    ax.set_title("Scene sources", loc="left"); ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    nconf = [len(data["splits"][s]["configurations"]) for s in SPLITS]
    fig.text(.5, .025, f"Configuration cells show QA counts on a shared color scale; {nconf[0]} / {nconf[1]} combinations in Seen / Unseen. Bar labels show counts; bar heights show within-split percentages.", ha="center", fontsize=10, color=MUTED)
    save(fig, "03_configuration_and_motion", out, dpi)


def queries(data, out, dpi):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5))
    fig.subplots_adjust(hspace=.58, wspace=.27, bottom=.12)
    max_d = max(v for s in SPLITS for vals in data["splits"][s]["distance_gt_m"].values() for v in vals)
    edges = np.arange(0, np.ceil(max_d)+1, 1.)
    for ax, task in zip(axs[0], ("B1", "B3")):
        for split in SPLITS:
            vals = data["splits"][split]["distance_gt_m"][task]
            counts, _ = np.histogram(vals, bins=edges)
            assert int(counts.sum()) == len(vals)
            med = np.median(vals)
            ax.stairs(100*counts/len(vals), edges, color=SPLIT_COLORS[split], linewidth=1.8,
                      label=f"{split.title()} · N={len(vals)} · median {med:.3f} m")
            ax.axvline(med, color=SPLIT_COLORS[split], linestyle="--", linewidth=1, alpha=.8)
        ax.set_title(label(task, 50), loc="left")
        ax.set(xlabel="Camera-to-target 3D distance (m)", ylabel="QA items within task and split (%)", xlim=(0, edges[-1]))
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    for ax, task in zip(axs[1], ("A4", "B3")):
        for offset, split in zip((-.18, .18), SPLITS):
            d = data["splits"][split]
            nn = [d["query_fraction_counts"][task][str(f)] for f in (.25, .5, .75)]
            bars = ax.bar(np.arange(3)+offset, np.array(nn)/sum(nn)*100, .34, color=SPLIT_COLORS[split], label=split.title())
            ax.bar_label(bars, labels=[str(n) for n in nn], padding=3, fontsize=11)
        ax.set_title(label(task, 50) + "\nQuery progress", loc="left")
        ax.set(xticks=range(3), xticklabels=["25%", "50%", "75%"], xlabel="Distance completed within the specified Forward action",
               ylabel="QA items within task and split (%)", ylim=(0, 49))
        ax.legend(frameon=False, ncol=2, loc="upper left")
        ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    fig.text(.5, .025, "Distances use published ground-truth answers and common 1 m bins. Query fractions refer to one Forward action, not total trajectory time or length.", ha="center", fontsize=10, color=MUTED)
    save(fig, "04_distance_and_query_progress", out, dpi)


def labels_and_sources(data, out, dpi):
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1], wspace=.48, hspace=.4)
    cmap = LinearSegmentedColormap.from_list("sources", ["#FFFFFF", "#D9E4EC", "#5E809A"])
    vmax = max(n for s in SPLITS for src in SOURCES for n in data["splits"][s]["source_task_counts"][src].values())
    for col, split in enumerate(SPLITS):
        ax = fig.add_subplot(gs[0, col]); d = data["splits"][split]
        arr = np.array([[d["source_task_counts"][src][t] for src in SOURCES] for t in TASKS])
        heatmap(ax, arr, cmap, vmax)
        ax.set_xticks(range(3), list(SOURCES.values()), fontsize=10)
        ax.set_yticks(range(8), [label(t, 33) for t in TASKS], fontsize=9.3)
        ax.set_title(f"{split.title()} · source coverage by task", loc="left")
        for (i, j), n in np.ndenumerate(arr):
            ax.text(j, i, str(n), ha="center", va="center", fontsize=11, color="white" if n/vmax > .7 else INK)
    for col, (task, key, categories) in enumerate((
        ("A1", "collision_labels", ["Collision", "No collision"]),
        ("C1", "future_view_labels", ["A", "B", "C", "D"]),
    )):
        ax = fig.add_subplot(gs[1, col])
        for offset, split in zip((-.18, .18), SPLITS):
            d = data["splits"][split]
            nn = [d[key].get(c, 0) for c in categories]
            bars = ax.bar(np.arange(len(categories))+offset, np.array(nn)/sum(nn)*100, .34, color=SPLIT_COLORS[split], label=split.title())
            ax.bar_label(bars, labels=[str(n) for n in nn], padding=3, fontsize=11)
        ax.set_title(label(task, 50) + " · answer distribution", loc="left", fontsize=12)
        ax.set(xticks=range(len(categories)), xticklabels=categories, ylabel="QA items within task and split (%)")
        ax.set_ylim(0, 64 if task == "A1" else 34)
        ax.legend(frameon=False, ncol=2, loc="upper right")
        ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    fig.text(.5, .025, "Heatmap cells are QA counts (shared scale). Zero indicates no samples from that source for that task. Bar labels are counts, not model predictions.", ha="center", fontsize=10, color=MUTED)
    save(fig, "05_source_and_answer_coverage", out, dpi)


def directions(data, out, dpi):
    fig, axs = plt.subplots(2, 2, figsize=(17, 7.8))
    fig.subplots_adjust(hspace=.64, wspace=.2, bottom=.15)
    cmap = LinearSegmentedColormap.from_list("directions", ["#FFFFFF", "#D5E8EF", "#307F9F"])
    all_pcts = []
    for split in SPLITS:
        for task in ("A4", "B2"):
            d = data["splits"][split]
            all_pcts.extend((np.array(d["direction_counts"][task])/d["task_counts"][task]*100).ravel())
    vmax = max(all_pcts)
    for row, task in enumerate(("A4", "B2")):
        for col, split in enumerate(SPLITS):
            ax = axs[row, col]; d = data["splits"][split]
            arr = np.array(d["direction_counts"][task]); pct = 100*arr/arr.sum()
            heatmap(ax, pct, cmap, vmax)
            ax.set_xticks(range(8), [s.replace("-", "-\n").capitalize() for s in HORIZONTAL], fontsize=9.3)
            ax.set_yticks(range(3), ["Above", "At camera level", "Below"], fontsize=10)
            ax.set_title(f"{split.title()} · {label(task, 50)}", loc="left", fontsize=12)
            for (i, j), n in np.ndenumerate(arr):
                ax.text(j, i, f"{n}\n{pct[i,j]:.1f}%", ha="center", va="center", fontsize=9.5,
                        color="white" if pct[i,j]/vmax > .7 else INK)
    fig.text(.5, .055, "Each cell shows count and percentage within its task and split. Colors share the same percentage scale across all four panels.", ha="center", fontsize=10, color=MUTED)
    fig.text(.5, .02, "Directions are defined in the camera frame at the queried pose. The 24 answer categories do not represent equal solid angles.", ha="center", fontsize=10, color=MUTED)
    save(fig, "06_target_direction_coverage", out, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "output/figures/paper_20260911")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    rows, hashes = load_data(args.benchmark)
    data = summarize(rows, hashes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figure_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    with (args.output / "task_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "task_name", "prediction_family", "seen", "unseen", "total"])
        for t, (name, group) in TASKS.items():
            ns = [data["splits"][s]["task_counts"][t] for s in SPLITS]
            writer.writerow([t, name, GROUPS[group][0], *ns, sum(ns)])
    taxonomy(args.output, args.dpi)
    for fn in (task_distribution, conditions, queries, labels_and_sources, directions):
        fn(data, args.output, args.dpi)
    print(f"Rendered 6 editable figures: {data['qa_count']:,} QA items; {data['scene_count']} scenes.")
    print(f"SVG / PDF / PNG and source statistics: {args.output}")


if __name__ == "__main__":
    main()
