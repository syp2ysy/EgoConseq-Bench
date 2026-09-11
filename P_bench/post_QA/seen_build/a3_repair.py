"""Scene-owned B1K A3 replay, with patches limited to task_outputs.A3."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from pipeline import io_utils


def apply_updates(record, updates):
    """Change A3 only; do not normalize, reinstall cases, or refresh surfaces."""
    updated = copy.deepcopy(record)
    cases = {case["case_id"]: case for case in updated["cases"]}
    for case_id, output in updates.items():
        if output is None:
            del cases[case_id]["task_outputs"]["A3"]
        else:
            cases[case_id]["task_outputs"]["A3"] = copy.deepcopy(output)
    return updated


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan(seen_records, unseen_records, work, targets=None):
    from post_QA.seen_build.collect import DEFAULT_B1K_MANIFEST

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, root in (("seen", seen_records), ("unseen", unseen_records)):
        root = Path(root).resolve()
        path = root / "b1k/records.jsonl"
        meta = json.loads(path.with_name("run_meta.json").read_text())
        source = meta.get("scene_source") or {
            "manifest": (meta.get("expansion") or {}).get("source_manifest")
                        or str(DEFAULT_B1K_MANIFEST), "split": "train"}
        by_scene = defaultdict(list)
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = json.loads(line)
                case_ids = [c["case_id"] for c in record["cases"]
                            if "A3" in c.get("task_outputs", {}) and
                            (targets is None or c["case_id"] in targets.get(record["record_uid"], set()))]
                count = len(case_ids)
                if count:
                    row = {
                        "record_uid": record["record_uid"], "byte_offset": offset,
                        "source_record_sha256": hashlib.sha256(line).hexdigest(),
                        "cases": count}
                    if targets is not None:
                        row["case_ids"] = case_ids
                    by_scene[record["scene_id"]].append(row)
        for scene_id, rows in by_scene.items():
            job = {"split": split, "scene_id": scene_id, "source_path": str(path),
                   "scene_source": source, "rows": rows}
            target = work / f"{split}-{scene_id}.job.json"
            if target.exists() and json.loads(target.read_text()) != job:
                raise ValueError(f"A3 repair source changed; finish the existing job: {target}")
            io_utils.atomic_write_json(target, job)
            jobs.append(target)
    return sorted(jobs, key=lambda p: (
        "hall_arch_wood" not in p.name, "hotel_gym_spa" not in p.name,
        -sum(r["cases"] for r in json.loads(p.read_text())["rows"])))


def replay_record(sim, record, case_ids=None):
    """Replay existing A3 actions against geometry and the original-pose mask."""
    import numpy as np
    from pipeline import actions, rollout
    from post_QA.seen_build.collect import _build_frame

    frame = _build_frame(sim, record)
    semantic = sim.semantic_index
    mask_ids = set(map(int, np.unique(frame.pts_sem)))
    groups = defaultdict(list)
    for case in record["cases"]:
        if "A3" in case.get("task_outputs", {}) and (case_ids is None or case["case_id"] in case_ids):
            groups[case["body_radius_m"]].append(case)
    updates, issues = {}, []
    for radius, cases in groups.items():
        nav = sim.proposal_nav(record["pose"]["position"], record["pose"]["yaw_rad"], radius_m=radius)
        programs = [actions.parse_actions(case["actions"]) for case in cases]
        traces = rollout.physical_path_traces(nav, programs)
        for case, program, trace in zip(cases, programs, traces):
            case_id, old = case["case_id"], case["task_outputs"]["A3"]
            if not trace.collision:
                issues.append({"case_id": case_id, "reason": "collision_changed"})
                continue
            index, _ = actions.contact_action_index(program, trace.first_contact_arc_m)
            if case.get("first_collision_action_index_1based") not in (None, index + 1):
                issues.append({"case_id": case_id, "reason": "collision_action_changed"})
                continue
            pose = actions.pose_at_arc(program, trace.first_contact_arc_m)
            hit = nav.closest_obstacle(pose)
            component = hit["obstacle_identity"]
            instance_id = semantic.instance_id_for_component(component)
            labels = semantic.category_layers(instance_id)
            detail = {"case_id": case_id, "instance_id": instance_id,
                      "source_category": labels["raw"], "contact_component": component}
            if instance_id not in mask_ids:
                issues.append({**detail, "reason": "contact_not_in_initial_mask"})
                continue
            # A3 requires the object to be visible, not its eventual contact face.
            # The actual collision mesh, not neighbouring depth-point votes, owns GT.
            if labels["raw"] == "ceilings":
                issues.append({**detail, "reason": "structural_asset_needs_review"})
                continue
            output = {**old, "instance_id": instance_id, "category": labels["machine"],
                      "source_category": labels["raw"], "contact_component": component}
            if labels["machine"] not in {c["id"] for c in output.get("choices", [])}:
                output["choices"] = [*output.get("choices", []),
                                     {"id": labels["machine"], "text": labels["raw"].replace("_", " ")}]
            updates[case_id] = output
    return updates, issues


def worker(job_path):
    from post_QA.seen_build.collect import make_scene_opener

    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    result_path = job_path.with_suffix(".results.jsonl")
    finished = set()
    if result_path.exists():
        with result_path.open() as stream:
            finished = {json.loads(line)["record_uid"] for line in stream}
    records = []
    with Path(job["source_path"]).open("rb") as stream:
        for row in job["rows"]:
            if row["record_uid"] in finished:
                continue
            stream.seek(row["byte_offset"])
            line = stream.readline()
            if hashlib.sha256(line).hexdigest() != row["source_record_sha256"]:
                raise ValueError(f"A3 repair record changed: {row['record_uid']}")
            record = json.loads(line)
            records.append(({**row, "scene_source": job["scene_source"],
                             "observation_profile": record.get("observation_profile", {})}, record))
    if not records:
        return
    sim = make_scene_opener()("b1k", job["scene_id"], records)
    try:
        with result_path.open("a", buffering=1) as output:
            for index, (row, record) in enumerate(records, 1):
                updates, issues = replay_record(sim, record, row.get("case_ids"))
                output.write(_json({"record_uid": row["record_uid"],
                    "source_record_sha256": row["source_record_sha256"],
                    "updates": updates, "issues": issues}) + "\n")
                print(f"[a3] {job['scene_id']} {index}/{len(records)}: "
                      f"{len(updates)} verified, {len(issues)} review", flush=True)
    finally:
        sim.close()


def run_jobs(jobs, *, gpus, python, data_root):
    """Four independent scene processes; a scene is never split across GPUs."""
    from queue import Queue
    from pipeline.b1k_process import wait_isolated_process

    queue = Queue()
    for job in jobs:
        queue.put(job)
    root = Path(__file__).resolve().parents[2]

    def run_gpu(gpu):
        from queue import Empty
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                OMNIGIBSON_DATA_PATH=str(data_root), OMNIGIBSON_APPDATA_PATH=str(data_root / "appdata"),
                OMNIGIBSON_GPU_ID=str(gpu), OMNIGIBSON_HEADLESS="True", OMNI_KIT_ACCEPT_EULA="YES",
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
            env.pop("CUDA_VISIBLE_DEVICES", None)
            command = [str(python), "-u", str(root / "scripts/build_seen_benchmark.py"),
                       "repair-a3", "--job", str(job)]
            with job.with_suffix(".log").open("ab") as log:
                child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = wait_isolated_process(child, timeout_s=21600)
            if code:
                raise RuntimeError(f"A3 scene failed ({code}): {job.with_suffix('.log')}")
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_gpu, gpus))


def publish(args, jobs, *, drop_review=False):
    """Publish A3 updates or withdrawals; preserve all other record fields."""
    from contextlib import ExitStack
    from datetime import datetime, timezone
    import tempfile
    from post_QA.categories import a3_answer
    from pipeline.candidate_sources import canonical_sha256

    audit_path = Path(args.work).parent / "a3_category_audit.json"
    previous = json.loads(audit_path.read_text()) if drop_review else {}
    withdrawn = previous.get("review", [])
    if drop_review and not withdrawn:
        return previous
    targets = defaultdict(dict)
    for row in withdrawn:
        targets[row["record_uid"]][row["case_id"]] = None
    results, review = {}, []
    for job_path in jobs:
        job = json.loads(job_path.read_text())
        planned = {row["record_uid"]: row for row in job["rows"]}
        with job_path.with_suffix(".results.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                uid = row["record_uid"]
                if row["source_record_sha256"] != planned[uid]["source_record_sha256"]:
                    raise ValueError(f"A3 result does not match its source: {uid}")
                results[uid] = row
                review.extend({"record_uid": uid, "scene_id": job["scene_id"],
                               "split": job["split"], **issue} for issue in row["issues"])
        if set(planned) - set(results):
            raise ValueError(f"A3 scene has unfinished records: {job_path}")
    benchmark, sft = Path(args.benchmark_root).resolve(), Path(args.sft_root).resolve()
    report = {"revision": "a3-instance-category-v1", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "verified_cases": sum(len(r["updates"]) for r in results.values()),
              "review_cases": len(review), "review": review, "datasets": {}, "benchmark": {}}
    if drop_review:
        report.update(verified_cases=previous["verified_cases"],
                      removed_cases=len(withdrawn), removed=withdrawn)
    bindings, answers, pending = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix=".a3-publish-", dir=Path(args.work)) as temporary:
        stage = Path(temporary)

        def destination(path):
            path = Path(path).resolve()
            if path not in pending:
                relative = path.relative_to(benchmark) if path.is_relative_to(benchmark) else Path("files") / str(len(pending))
                pending[path] = stage / relative
                pending[path].parent.mkdir(parents=True, exist_ok=True)
            return pending[path]

        def put(path, value):
            destination(path).write_text(_json(value) + "\n")

        for split, root in (("seen", args.seen_records), ("unseen", args.unseen_records)):
            root = Path(root).resolve()
            source = root / "b1k/records.jsonl"
            ranges, changed, count = [], 0, 0
            with source.open("rb") as incoming, destination(source).open("wb") as output:
                for line in incoming:
                    record = json.loads(line)
                    uid = record["record_uid"]
                    row = results.get(uid)
                    updates = targets.get(uid, {}) if drop_review else row["updates"] if row else {}
                    if row and hashlib.sha256(line).hexdigest() != row["source_record_sha256"]:
                        raise ValueError(f"Record changed during A3 replay: {uid}")
                    updated = apply_updates(record, updates)
                    # This comparison is the mutation boundary, not a new record eligibility gate.
                    restored = copy.deepcopy(updated)
                    old_cases = {c["case_id"]: c for c in record["cases"]}
                    for case in restored["cases"]:
                        if case["case_id"] in updates:
                            case["task_outputs"]["A3"] = old_cases[case["case_id"]]["task_outputs"]["A3"]
                    if restored != record:
                        raise ValueError(f"A3 update changed another field: {uid}")
                    changed += updated != record
                    payload = ((_json(updated) + "\n").encode() if updated != record else line)
                    offset = output.tell()
                    output.write(payload)
                    count += 1
                    scene = record["scene_id"]
                    if not ranges or ranges[-1]["scene_id"] != scene:
                        ranges.append({"scene_id": scene, "start_byte": offset,
                                       "end_byte": output.tell(), "record_count": 0})
                    ranges[-1]["end_byte"] = output.tell()
                    ranges[-1]["record_count"] += 1
                    bindings[uid] = {"byte_offset": offset,
                                     "record_sha256": canonical_sha256(updated),
                                     "split": split}
                    for case_id, a3 in updates.items():
                        answers[(uid, case_id)] = a3_answer(updated, a3) if a3 is not None else None
            digest = io_utils.sha256_file(destination(source))
            meta_path = source.with_name("run_meta.json")
            meta = json.loads(meta_path.read_text())
            meta.update(records_sha256=digest, scene_byte_ranges=ranges, a3_category_revision=report["revision"])
            put(meta_path, meta)
            meta_digest = io_utils.sha256_file(destination(meta_path))
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest["datasets"]:
                if entry["dataset"] == "b1k":
                    entry.update(records_sha256=digest, run_meta_sha256=meta_digest)
            put(manifest_path, manifest)
            report["datasets"][split] = {"records": count, "changed_records": changed,
                "records_sha256": digest, "run_meta_sha256": meta_digest}

        frozen_path = benchmark / "metadata/frozen.json"
        frozen = json.loads(frozen_path.read_text())
        retired_images = set()
        for split in ("seen", "unseen"):
            root = benchmark / "metadata" / split
            qa_path = benchmark / "benchmark" / split / "QA.json"
            index_path = root / "record_index.json"
            qa, index = json.loads(qa_path.read_text()), json.loads(index_path.read_text())
            items = {item["id"]: item for item in qa}
            changed, removed_ids = 0, set()
            for entry in index["items"]:
                uid = entry["record_uid"]
                if uid not in bindings:
                    continue
                bind = bindings[uid]
                entry["record_sha256"] = bind["record_sha256"]
                entry["source"].update(byte_offset=bind["byte_offset"],
                    records_sha256=report["datasets"][split]["records_sha256"],
                    run_meta_sha256=report["datasets"][split]["run_meta_sha256"])
                key = (uid, entry["usage"]["outcome_id"])
                if entry["task_id"] == "A3" and key in answers:
                    answer = answers[key]
                    if answer is None:
                        removed_ids.add(entry["item_id"])
                        continue
                    entry["usage"]["contact_category"] = answer
                    for message in items[entry["item_id"]]["messages"]:
                        if message["role"] == "assistant":
                            changed += message["content"] != answer
                            message["content"] = answer
            old_images = {p for item in qa for p in item["images"]}
            qa = [item for item in qa if item["id"] not in removed_ids]
            index["items"] = [e for e in index["items"] if e["item_id"] not in removed_ids]
            index["record_count"] = len(index["items"])
            put(qa_path, qa)
            index["qa_sha256"] = io_utils.sha256_file(destination(qa_path))
            put(index_path, index)
            frozen["splits"][split].update(qa_sha256=index["qa_sha256"],
                record_index_sha256=io_utils.sha256_file(destination(index_path)))
            report["benchmark"][split] = {"qa_items": len(qa), "a3_answers_changed": changed,
                                          "a3_answers_removed": len(removed_ids)}
            if drop_review:
                images = {p for item in qa for p in item["images"]}
                frozen["splits"][split].update(qa_count=len(qa), task_totals=dict(Counter(q["task_id"] for q in qa)))
                if images != old_images:
                    image_hash = hashlib.sha256()
                    for path in sorted(images):
                        image_hash.update(f"{path}\t{io_utils.sha256_file(qa_path.parent / path)}\n".encode())
                    frozen["splits"][split].update(image_count=len(images), images_sha256=image_hash.hexdigest())
                    retired_images.update(qa_path.parent / path for path in old_images - images)
            report_path = root / "report.json"
            selection_report = json.loads(report_path.read_text())
            selection_report["a3_category_revision"] = report["revision"]
            buckets = selection_report["answer_buckets"]
            for key in list(buckets):
                if "/A3/" in key:
                    del buckets[key]
            buckets.update(Counter(
                f"{q['dataset']}/A3/" + next(m["content"] for m in q["messages"] if m["role"] == "assistant")
                for q in qa if q["task_id"] == "A3"))
            if removed_ids:
                entries = index["items"]
                totals = Counter(f"{e['dataset']}/{e['task_id']}" for e in entries)
                turns = Counter(f"{e['dataset']}/{e['task_id']}" for e in entries if e["usage"]["starts_with"] == "turn")
                selection_report.update(total=len(qa), dataset_task_totals=dict(totals),
                    action_lengths=dict(Counter(f"{e['dataset']}/{e['task_id']}/L{e['usage']['action_length']}" for e in entries)),
                    scenes=dict(Counter(f"{e['dataset']}/{e['scene_id']}" for e in entries)),
                    turn_first={key: {"count": turns[key], "fraction": round(turns[key] / count, 6)} for key, count in totals.items()},
                    action_signatures={task: len({_json(e["usage"]["actions"]) for e in entries if e["task_id"] == task}) for task in {q["task_id"] for q in qa}},
                    physical_parameters={name: dict(Counter(str(e["usage"][name]) for e in entries)) for name in ("body_radius_m", "camera_height_m", "hfov_deg")})
                for item_id in removed_ids:
                    selection_report.get("image_quality", {}).pop(item_id, None)
                if "prompts" in selection_report:
                    selection_report["prompts"]["templates"] = dict(Counter(e["usage"]["template_id"] for e in entries))
                if "parameters" in selection_report:
                    selection_report["parameters"].update(qa_items=len(qa), height_distribution_m=dict(Counter(str(e["usage"]["camera_height_m"]) for e in entries)))
                if "visual_selection" in selection_report:
                    selection_report["visual_selection"]["scope"] = "Original image selection before A3 withdrawals; retained images are an unchanged subset."
            put(report_path, selection_report)
        frozen["a3_category_revision"] = report["revision"]
        if drop_review:
            frozen["total"] = sum(s["qa_count"] for s in frozen["splits"].values())
            # Preserve previously held-out record IDs even after withdrawing a QA.
            frozen["a3_withdrawal"] = {"removed_qa": sum(v["a3_answers_removed"] for v in report["benchmark"].values()),
                                       "record_exclusions_preserved": True}
        put(frozen_path, frozen)

        selection_path = sft / "metadata/selection.jsonl"
        changed, a3_counts, answer_counts = 0, Counter(), Counter()
        removed_rows, removed_sft_ids = [], set()
        kept_records, kept_images, old_sft_images = set(), set(), set()
        task_counts, dataset_counts = Counter(), Counter()
        with selection_path.open() as incoming, destination(selection_path).open("w") as output:
            for line in incoming:
                row = json.loads(line)
                uid = row["record_uid"]
                if uid in bindings:
                    row["byte_offset"] = bindings[uid]["byte_offset"]
                key = (uid, row["case_id"])
                old_sft_images.update(row.get("images", []))
                if row["task_id"] == "A3" and key in answers and answers[key] is None:
                    removed_rows.append(row)
                    removed_sft_ids.add(row["id"])
                    continue
                if row["task_id"] == "A3":
                    if key in answers:
                        changed += row["answer"] != answers[key]
                        row["answer"] = row["answer_bucket"] = answers[key]
                    a3_counts[row["answer"]] += 1
                answer_counts[f"{row['task_id']}/{row['answer']}"] += 1
                task_counts[row["task_id"]] += 1
                dataset_counts[row["dataset"]] += 1
                kept_records.add(uid)
                kept_images.update(row.get("images", []))
                output.write(_json(row) + "\n")
        # Preserve every prompt byte: only replace A3 assistant messages in each view.
        for path in sorted((sft / "json").glob("*.jsonl")):
            with ExitStack() as stack:
                metadata = stack.enter_context(selection_path.open())
                incoming = stack.enter_context(path.open())
                output = stack.enter_context(destination(path).open("w"))
                from itertools import zip_longest
                for line, metadata_line in zip_longest(incoming, metadata):
                    if line is None or metadata_line is None:
                        raise ValueError(f"SFT view and selection lengths differ: {path}")
                    row, item = json.loads(metadata_line), json.loads(line)
                    if item["id"] != row["id"]:
                        raise ValueError(f"SFT view and selection IDs differ: {path}: {row['id']}")
                    if item["id"] in removed_sft_ids:
                        continue
                    if row["task_id"] == "A3" and (row["record_uid"], row["case_id"]) in answers:
                        for message in item["messages"]:
                            if message["role"] == "assistant":
                                message["content"] = answers[(row["record_uid"], row["case_id"])]
                        output.write(_json(item) + "\n")
                    else:
                        output.write(line)
        summary_path = sft / "metadata/summary.json"
        summary = json.loads(summary_path.read_text())
        summary["a3_category_revision"] = report["revision"]
        summary["distributions"]["answer"] = dict(sorted(answer_counts.items()))
        if drop_review:
            summary.update(qa_items=sum(task_counts.values()), task_totals=dict(task_counts),
                           dataset_totals=dict(dataset_counts), covered_records=len(kept_records), image_files=len(kept_images))
            for row in removed_rows:
                keys = {"dataset_task": f"{row['dataset']}/A3", "length": f"A3/L{row['action_length']}",
                        "start": f"{row['dataset']}/A3/{row['starts_with']}", "template": row["template_id"]}
                for kind, key in keys.items():
                    counter = summary["distributions"].get(kind, {})
                    if key in counter:
                        counter[key] -= 1
                        if not counter[key]:
                            del counter[key]
                if "parameters" in summary:
                    summary["parameters"]["height_distribution_m"][str(row["inputs"]["camera"]["optical_center_height_m"])] -= 1
            if "parameters" in summary:
                summary["parameters"]["qa_items"] = summary["qa_items"]
            if "sampling" in summary:
                summary["sampling"]["task_shortfalls"]["A3"] = max(0, summary["sampling"]["target_per_task"] - task_counts["A3"])
            summary["a3_withdrawal"] = {"qa_removed": len(removed_rows), "record_support_and_capacity_scope": "original SFT selection before A3 withdrawals"}
            retired_images.update(sft / path for path in old_sft_images - kept_images)
        summary["exclude_index_sha256"] = io_utils.sha256_file(destination(frozen_path))
        for source in summary["sources"]:
            if Path(source["path"]).parent.name == "b1k":
                source["sha256"] = report["datasets"]["seen"]["records_sha256"]
        summary["prompts"].update(full_sha256=io_utils.sha256_file(destination(sft / "json/full.jsonl")),
                                 selection_sha256=io_utils.sha256_file(destination(selection_path)))
        put(summary_path, summary)
        report["sft"] = {"a3_answers_changed": changed, "a3_answers": dict(a3_counts),
                         "a3_answers_removed": len(removed_rows), "qa_items": sum(task_counts.values())}
        put(audit_path, report)
        from visualization.benchmark_browser import build_combined_benchmark_browser
        build_combined_benchmark_browser(stage)
        destination(benchmark / "index.html")
        for path in retired_images:
            if not any(path.is_relative_to(base) for base in (sft / "images", benchmark / "benchmark/seen/images", benchmark / "benchmark/unseen/images")):
                raise ValueError(f"Not an exported image: {path}")
            pending[path] = None
        backups, replaced = {}, []
        try:
            for path, new in pending.items():
                backup = stage / f"old-{len(backups)}"
                if path.exists():
                    os.link(path, backup)
                    backups[path] = backup
                if new is None:
                    path.unlink()
                else:
                    os.replace(new, path)
                replaced.append(path)
        except BaseException:
            for path in reversed(replaced):
                if path in backups:
                    os.replace(backups[path], path)
                else:
                    path.unlink()
            raise
    # Finished source bindings are obsolete after publication. Keep the small
    # audit, not stale jobs/results that cannot bind to the updated JSONL.
    for job in jobs:
        job.with_suffix(".results.jsonl").unlink()
        job.with_suffix(".log").unlink(missing_ok=True)
        job.unlink()
    return report


def run(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "status.json"
    if getattr(args, "drop_unresolved", False):
        report = publish(args, [], drop_review=True)
        report["phase"] = "complete_with_exclusions"
        io_utils.atomic_write_json(status_path, {key: value for key, value in report.items() if key not in ("review", "removed")})
        return {key: value for key, value in report.items() if key not in ("review", "removed")}
    targets = None
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("phase") in {"complete", "complete_with_exclusions"}:
            return previous
        if previous.get("targets") is not None:
            targets = {uid: set(values) for uid, values in previous["targets"].items()}
        if previous.get("phase") == "updated_with_review":
            audit = json.loads((work.parent / "a3_category_audit.json").read_text())
            targets = defaultdict(set)
            for row in audit["review"]:
                targets[row["record_uid"]].add(row["case_id"])
    context = {"targets": {uid: sorted(values) for uid, values in targets.items()}} if targets is not None else {}
    io_utils.atomic_write_json(status_path, {"phase": "planning", **context})
    jobs = plan(args.seen_records, args.unseen_records, work, targets=targets)
    io_utils.atomic_write_json(status_path, {"phase": "replaying", "scenes": len(jobs), "gpus": args.gpus, **context})
    run_jobs(jobs, gpus=args.gpus, python=args.b1k_python, data_root=args.b1k_data_root)
    io_utils.atomic_write_json(status_path, {"phase": "publishing", **context})
    report = publish(args, jobs)
    report["phase"] = "updated_with_review" if report["review_cases"] else "complete"
    if work.exists():
        io_utils.atomic_write_json(status_path, {key: value for key, value in report.items() if key != "review"})
    return {key: value for key, value in report.items() if key != "review"}
