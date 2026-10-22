"""Static page of published releases for reading and reviewing their items (not part of the build).

    python -m scripts.review_page data/benchmark --out data/analysis/benchmark_review

The static UI lives in scripts/review_ui/. It displays multiple cases per row, full questions,
images and answers, filterable summaries, and optional review records. Cached items.js contains the
original prompts, images, answers and evidence; published datasets are never modified. Browser records
retain the release-scoped localStorage keys (review:<release.json sha256>) and JSON import/export format.
The stratified sample is for finding systematic issues, not estimating an error rate.

Refresh just the layout, reusing previously generated items.js:
    python -m scripts.review_page --out data/analysis/benchmark_review --layout-only
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from post_QA.build import images
from post_QA import answers
from scripts import review_stats as stats

SPREAD = ("scene", "representation", "unit", "camera", "answer")
FACTS = {
    "A1": ["collision", "depth_view_collision", "corridor_samples", "corridor_coverage", "corridor_out_of_frame",
           "corridor_occluded"],
    "A2": ["first_collision_action", "depth_view_contact_command", "contact_progress_m", "command_length_m"],
    "A3": ["category", "source_category", "visible_points"],
    "A4": ["azimuth_deg", "elevation_deg", "boundary_margin_deg", "planar_range_m", "horizontal", "vertical",
           "checkpoint_action", "checkpoint_fraction"],
    "B1": ["distance_m", "initial_distance_m"],
    "C1": ["min_block_l1"],
}
FACTS.update(B2=FACTS["A4"][:6], B3=FACTS["B1"] + ["checkpoint_action", "checkpoint_fraction"])
UI_ASSETS = Path(__file__).with_name("review_ui")


def spread(rows: pd.DataFrame, n: int) -> list:
    """n rows, each next one showing the most field values not yet shown (ties in hashed order)."""
    rows = rows.assign(_h=[hashlib.sha256(i.encode()).hexdigest() for i in rows["id"]]).sort_values("_h")
    left, shown, picked = rows.to_dict("records"), {f: set() for f in SPREAD}, []
    while left and len(picked) < n:
        best = max(left, key=lambda r: sum(r[f] not in shown[f] for f in SPREAD))
        left.remove(best)
        picked.append(best)
        for f in SPREAD:
            shown[f].add(best[f])
    return picked


def sample(rows: pd.DataFrame, per_cell: int) -> set:
    ids = set()
    for (kind, _, _), cell in rows.groupby(["set", "task", "dataset"]):
        if kind == "radius_pairs":
            pairs = {r["pair_id"] for r in spread(cell.drop_duplicates("pair_id"), per_cell // 2)}
            ids |= set(cell.loc[cell["pair_id"].isin(pairs), "id"])
        else:
            ids |= {r["id"] for r in spread(cell, per_cell)}
    return ids


def facts(meta: Path, rows: pd.DataFrame) -> dict:
    """question_id -> the GT row of its meta task table (the FACTS columns)."""
    def read(entry):
        (task, dataset), group = entry
        table = pq.read_table(meta / "tasks" / task / f"{dataset}.parquet",
                              columns=["question_id", "actions"] + FACTS[task], use_threads=False,
                              filters=[("question_id", "in", sorted(set(group["question_id"])))])
        return table.to_pylist()
    with ThreadPoolExecutor(max_workers=3) as pool:
        return {r["question_id"]: r for batch in pool.map(read, rows.groupby(["task", "dataset"])) for r in batch}


def observations(meta, rows):
    columns = ["obs_id", "image", "image_sha256", "camera_height_m", "hfov_deg", "vfov_deg", "width", "height",
               "valid_depth_ratio", "visible_floor_ratio"]
    out = {}
    for dataset, group in rows.groupby("dataset"):
        table = pq.read_table(meta / "observations" / f"{dataset}.parquet", columns=columns,
                              filters=[("obs_id", "in", sorted(set(group["obs_id"])))])
        out.update(((dataset, r["obs_id"]), r) for r in table.to_pylist())
    return out


def describe(task, f, quality) -> list:
    lines = []
    if task == "A1":
        agree = "与 GT 一致" if f["depth_view_collision"] == f["collision"] else "与 GT 不一致"
        lines.append(f"初始深度图预测：{'碰撞' if f['depth_view_collision'] else '不碰撞'}（{agree}）")
        lines.append("路径地面：0 个采样点，可见性未知" if f["corridor_samples"] == 0 else
                     f"路径地面：{f['corridor_samples']} 个采样点，可见 {f['corridor_coverage']:.0%}，"
                     f"出画面 {f['corridor_out_of_frame']}，被遮挡 {f['corridor_occluded']}")
    elif task == "A2":
        view = f["depth_view_contact_command"]
        lines.append(f"初始深度图预测：{'不碰撞' if view is None else f'第 {view} 个命令'}"
                     f"（{'与 GT 一致' if view == f['first_collision_action'] else '与 GT 不一致'}）")
        lines.append(f"碰撞点：第 {f['first_collision_action']} 个命令前进 {f['contact_progress_m']:.2f} / "
                     f"{f['command_length_m']:.2f} m 处（距该命令末端 {f['command_length_m'] - f['contact_progress_m']:.2f} m）")
    elif task == "A3":
        lines.append(f"类别 {f['category']}（资产类别 {f['source_category']}），接触区可见点 {f['visible_points']}")
    elif task in ("A4", "B2"):
        lines.append(f"方位 {f['azimuth_deg']:+.1f}°，俯仰 {f['elevation_deg']:+.1f}°，距方向分界 "
                     f"{f['boundary_margin_deg']:.1f}°，水平距离 {f['planar_range_m']:.2f} m")
    elif task in ("B1", "B3"):
        lines.append(f"距离 {f['distance_m']:.2f} m（初始 {f['initial_distance_m']:.2f} m，变化 "
                     f"{f['distance_m'] - f['initial_distance_m']:+.2f} m）")
    else:
        lines.append(f"选项间最小块差 L1 {f['min_block_l1']:.3f}")
    return lines + [f"初始图：暗像素 {quality[0]:.1%}，边缘密度 {quality[1]:.3f}"]


def answer_value(row, gt) -> str:
    """The value the answer filter shows: the answer, for B1/B3 its distance bin."""
    if row["task"] not in ("B1", "B3"):
        return str(row["answer"]).rstrip(".")
    return stats.distance_bin(gt["distance_m"])


def write_layout(out: Path):
    """Copy the small static UI without rewriting published questions or cached review data."""
    out.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js", "stats.js"):
        (out / name).write_bytes((UI_ASSETS / name).read_bytes())


def release_record(path: Path, manifest: dict) -> dict:
    """Page summary of one release partition. A release moved to a new directory keeps its original review
    key (source_release_sha256), so stored reviews still apply; a fresh build is keyed by its selection."""
    metadata = path / manifest["metadata_dir"]
    report_path = metadata / "report.json"
    selection_sha256 = stats.sha256(metadata / "selection.parquet")
    return {"name": ", ".join(k for k, v in manifest["sets"].items() if v["scoring"] == "items"),
            "path": str(path), "prompt_version": manifest["prompt_version"],
            "review_key": manifest.get("source_release_sha256", selection_sha256),
            "release_sha256": stats.sha256(path / "release.json"), "image_checks": manifest["image_checks"],
            "report": json.loads(report_path.read_bytes()) if report_path.exists() else {},
            "selection_sha256": selection_sha256,
            "report_sha256": stats.sha256(report_path) if report_path.exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("releases", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument("--layout-only", action="store_true", help="Refresh the UI using existing items.js; no meta reads")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.layout_only:
        if not (out / "items.js").is_file():
            parser.error("--layout-only requires an existing items.js in --out")
        cached = json.loads((out / "items.js").read_text().removeprefix("const DATA = ").rstrip(";\n"))
        if "statistics" not in cached:
            parser.error("cached items predate the statistics schema; rerun with the release directories once")
        write_layout(out)
        print(out / "index.html", "layout refreshed; items.js unchanged")
        return
    if not args.releases:
        parser.error("at least one release is required unless --layout-only is used")
    releases, items, systems = [], [], Counter()
    partitions = []
    for path in args.releases:
        manifest = json.loads((path / "release.json").read_text())
        partitions.extend((path, part) for part in manifest["partitions"].values())
    for r, (path, manifest) in enumerate(partitions):
        print(f"Reading {path}: published selection and stored task measurements", flush=True)
        metadata = path / manifest["metadata_dir"]
        release = release_record(path, manifest)
        releases.append(release)
        meta = (path / manifest["image_root"]).resolve()
        qa = {i["id"]: i for s in manifest["sets"].values() for i in json.loads((path / s["dir"] / "QA.json").read_text())}
        rows = pd.read_parquet(metadata / "selection.parquet")
        rows = rows[rows["id"].isin(qa)]
        if len(rows) != len(qa) or not rows["id"].is_unique:
            raise ValueError(f"selection and public QA IDs differ in {path}")
        release["available_tasks"] = {d: [t for t in stats.TASKS
                                         if (table_path := meta / "tasks" / t / f"{d}.parquet").exists()
                                         and pq.ParquetFile(table_path).metadata.num_rows > 0]
                                      for d in sorted(rows["dataset"].unique())}
        release["dataset_supply"] = {}
        for dataset in release["available_tasks"]:
            supply = pq.read_table(meta / "observations" / f"{dataset}.parquet", columns=["scene", "status"]).to_pandas()
            ok = supply[supply["status"] == "ok"]
            release["dataset_supply"][dataset] = {"observations": len(supply), "captured": len(ok),
                                                  "captured_scenes": int(ok["scene"].nunique())}
        fact, picked = facts(meta, rows), sample(rows, args.per_cell)
        obs = observations(meta, rows)
        paths = sorted({o["image"] for o in obs.values()} | {p for item in qa.values() for p in item["images"][1:]})
        print(f"Measuring {len(paths):,} distinct original/option images once", flush=True)
        quality = dict(zip(paths, images.qualities(meta / p for p in paths)))
        for row in rows.sort_values(["set", "dataset", "task", "scene", "pair_id", "radius", "id"]).to_dict("records"):
            item = qa[row["id"]]
            gt, observation = fact[row["question_id"]], obs[(row["dataset"], row["obs_id"])]
            measured = stats.measurements(row["task"], gt, row["answer"], row["unit"])
            measured.update(dark=quality[observation["image"]][0], edges=quality[observation["image"]][1],
                            camera_height_m=observation["camera_height_m"], hfov_deg=observation["hfov_deg"],
                            vfov_deg=observation["vfov_deg"], width=observation["width"], height=observation["height"],
                            valid_depth_ratio=observation["valid_depth_ratio"], visible_floor_ratio=observation["visible_floor_ratio"],
                            prompt_words=len(item["messages"][1]["content"].split()),
                            option_quality=[quality[p] for p in item["images"][1:]])
            systems[item["messages"][0]["content"]] += 1
            items.append({"r": r, "id": row["id"], "set": row["set"], "dataset": row["dataset"], "task": row["task"],
                          "domain": stats.DOMAIN_FOR[row["task"]], "program": row["program_id"], "m": measured,
                          "raw_hash": observation["image_sha256"], "image_hashes": list(row["image_sha256"]),
                          "scene": row["scene"], "obs": row["obs_id"], "rep": row["representation"],
                          "unit": row["unit"], "radius": row["radius"], "camera": row["camera"],
                          "template": row["template_id"], "pair": row["pair_id"], "sample": row["id"] in picked,
                          "answer": answer_value(row, gt), "gt": item["messages"][2]["content"],
                          "system": item["messages"][0]["content"], "prompt": item["messages"][1]["content"],
                          "images": [os.path.relpath(meta / p, out) for p in item["images"]],
                          "facts": describe(row["task"], gt, quality[observation["image"]])})
    order = [s for s, _ in systems.most_common()]
    for item in items:
        item["sys"] = order.index(item.pop("system"))
    out.mkdir(parents=True, exist_ok=True)
    snapshot = stats.snapshot(items, releases)
    snapshot["generator_sha256"] = {str(p): stats.sha256(p) for p in
                                    [Path(__file__), Path(stats.__file__), Path(images.__file__), Path(answers.__file__),
                                     *sorted(UI_ASSETS.glob("*"))] if p.is_file()}
    (out / "stats.json").write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    (out / "items.js").write_text("const DATA = " + json.dumps(
        {"releases": releases, "systems": order, "items": items, "statistics": snapshot},
        ensure_ascii=False, allow_nan=False) + ";\n", encoding="utf-8")
    write_layout(out)
    print(out / "index.html", len(items), "items,", sum(i["sample"] for i in items), "in the review sample")


if __name__ == "__main__":
    main()
