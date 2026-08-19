"""Read-only inventory of historical B1K scene-authority failures."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


_PROGRESS_SCHEMAS = frozenset({
    "b1k-source-manifest-shard-progress.v1",
    "b1k-source-manifest-shard-progress.v2",
})


def _failure_category(error: str) -> str:
    normalized = str(error).lower()
    if "empty clearance-conditioned free space" in normalized:
        return "empty_clearance"
    if "position drift" in normalized:
        return "reset_position_drift"
    if "orientation drift" in normalized:
        return "reset_orientation_drift"
    if "joint" in normalized and "drift" in normalized:
        return "reset_joint_drift"
    if "triangle authority drift" in normalized:
        return "triangle_authority_drift"
    if "timeoutexpired" in normalized or "timed out" in normalized:
        return "timeout"
    if "no such file or directory" in normalized:
        return "missing_fragment"
    return "other"


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read B1K audit input {path}: {error}") \
            from error
    if not isinstance(value, dict):
        raise ValueError(f"B1K audit input is not an object: {path}")
    return value


def audit_authority_failures(root: Path) -> dict:
    """Return a deterministic report without modifying ``root`` or its files."""
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"B1K authority audit root is not a directory: {root}")
    events = []

    def child_evidence(child, *, path: Path, parent_kind: str,
                       parent_index: int, scene_id: str) -> dict:
        if (not isinstance(child, dict) or child.get("schema") !=
                "b1k-scene-authority-failure.v1" or
                str(child.get("scene_id") or "") != scene_id or
                not str(child.get("error") or "")):
            raise ValueError(f"B1K child authority failure is invalid: {path}")
        error = str(child["error"])
        return {
            "source_kind": "child_failure",
            "source_path": path.relative_to(resolved).as_posix(),
            "parent_kind": parent_kind,
            "parent_index": parent_index,
            "scene_id": scene_id,
            "error_type": str(child.get("error_type") or ""),
            "error": error,
            "category": _failure_category(error),
        }

    for path in sorted(resolved.rglob("shard-progress.json")):
        value = _read_json_object(path)
        if value.get("schema") not in _PROGRESS_SCHEMAS:
            continue
        relative = path.relative_to(resolved).as_posix()
        progress_events_by_scene = {}
        attempts = value.get("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError(f"B1K authority attempts are invalid: {path}")
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise ValueError(f"B1K authority attempt is invalid: {path}")
            if attempt.get("status") != "failed":
                continue
            error = str(attempt.get("error") or "")
            scene_id = str(attempt.get("scene_id") or "")
            if not scene_id or not error:
                raise ValueError(
                    f"B1K failed authority attempt is incomplete: {path}")
            rows = [{
                "source_kind": "shard_attempt",
                "source_path": relative,
                "attempt_index": attempt_index,
                "scene_id": scene_id,
                "returncode": attempt.get("returncode"),
                "error": error,
                "category": _failure_category(error),
            }]
            child = attempt.get("child_failure")
            if child is not None:
                rows.append(child_evidence(
                    child, path=path, parent_kind="shard_attempt",
                    parent_index=attempt_index, scene_id=scene_id))
            event = {
                "event_id": f"{relative}#attempt-{attempt_index}",
                "scene_id": scene_id,
                "source_directory": path.parent,
                "evidence": rows,
            }
            events.append(event)
            progress_events_by_scene.setdefault(scene_id, []).append(event)
        rows = value.get("failed_scenes")
        if not isinstance(rows, list):
            raise ValueError(f"B1K failed scene list is invalid: {path}")
        for failed_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"B1K failed scene row is invalid: {path}")
            error = str(row.get("error") or "")
            scene_id = str(row.get("scene_id") or "")
            if not scene_id or not error:
                raise ValueError(f"B1K failed scene row is incomplete: {path}")
            summary_evidence = [{
                "source_kind": "shard_progress",
                "source_path": relative,
                "failed_scene_index": failed_index,
                "scene_id": scene_id,
                "returncode": row.get("returncode"),
                "error": error,
                "category": _failure_category(error),
            }]
            child = row.get("child_failure")
            if child is None:
                pass
            else:
                summary_evidence.append(child_evidence(
                    child, path=path, parent_kind="shard_progress",
                    parent_index=failed_index, scene_id=scene_id))
            matching = progress_events_by_scene.get(scene_id, [])
            if matching:
                matching[-1]["evidence"].extend(summary_evidence)
            else:
                event = {
                    "event_id": f"{relative}#failed-scene-{failed_index}",
                    "scene_id": scene_id,
                    "source_directory": path.parent,
                    "evidence": summary_evidence,
                }
                events.append(event)
                progress_events_by_scene.setdefault(scene_id, []).append(event)

    for path in sorted(resolved.rglob("*.failure.json")):
        value = _read_json_object(path)
        if value.get("schema") != "b1k-scene-authority-failure.v1":
            raise ValueError(f"B1K authority failure schema is invalid: {path}")
        error = str(value.get("error") or "")
        scene_id = str(value.get("scene_id") or "")
        if not scene_id or not error:
            raise ValueError(f"B1K authority failure is incomplete: {path}")
        row = {
            "source_kind": "failure_file",
            "source_path": path.relative_to(resolved).as_posix(),
            "scene_id": scene_id,
            "error_type": str(value.get("error_type") or ""),
            "error": error,
            "category": _failure_category(error),
        }
        matching = [
            event for event in events
            if event["scene_id"] == scene_id and
            event["source_directory"] == path.parent
        ]
        if matching:
            matching[-1]["evidence"].append(row)
        else:
            events.append({
                "event_id": path.relative_to(resolved).as_posix(),
                "scene_id": scene_id,
                "source_directory": path.parent,
                "evidence": [row],
            })

    priority = {
        "failure_file": 0,
        "child_failure": 1,
        "shard_attempt": 2,
        "shard_progress": 3,
    }
    failures = []
    for event in sorted(events, key=lambda value: value["event_id"]):
        rows = sorted(event["evidence"], key=lambda row: (
            priority[row["source_kind"]], row["source_path"],
            row.get("attempt_index", -1),
            row.get("failed_scene_index", -1)))
        primary = min(rows, key=lambda row: (
            priority[row["source_kind"]], row["source_path"],
            row.get("attempt_index", -1)))
        failures.append({
            "event_id": event["event_id"],
            "scene_id": event["scene_id"],
            "category": primary["category"],
            "error": primary["error"],
            **({"error_type": primary["error_type"]}
               if "error_type" in primary else {}),
            "evidence": rows,
        })
    evidence_count = sum(len(event["evidence"]) for event in events)
    counts = Counter(row["category"] for row in failures)
    return {
        "schema": "b1k-authority-failure-audit.v1",
        "root": str(resolved),
        "failure_event_count": len(failures),
        "evidence_record_count": evidence_count,
        "unique_failed_scene_count": len({
            row["scene_id"] for row in failures}),
        "category_counts": dict(sorted(counts.items())),
        "failures": failures,
    }
