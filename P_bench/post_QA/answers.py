"""Canonical answers and fixed distance/aggregation rules for the evaluator."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import math
import re
from typing import Iterable, Mapping

from pipeline import config
from post_QA import templates


_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def parse_distance(response: object) -> Decimal | None:
    """Read one explicit quantity; unrecognized prose must be extracted by the LLM."""
    raw = ("" if response is None else str(response)).replace("−", "-")
    final = re.search(r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=]\s*([^\n]+)\s*$", raw, re.I)
    text = _text(final.group(1) if final else raw).strip("`*\"'").removesuffix(".")
    if not text or text in {"nan", "inf", "infinity", "+inf", "-inf", "+infinity", "-infinity"}:
        return None
    number = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?"
    unit = r"millimeters?|millimetres?|mm|centimeters?|centimetres?|cm|meters?|metres?|m"
    match = re.fullmatch(
        r"(?:(?:the\s+)?(?:camera-to-(?:point|target)\s+)?distance\s*(?:is|:|=)\s*)?"
        r"(?:(?:approximately|about|around|roughly|~|≈)\s*)?"
        rf"({number})\s*({unit})?(?:\s+away)?", text)
    if match is None:
        raise ValueError("expected one distance with m/cm/mm, not a range or explanation")
    value = Decimal(match.group(1))
    units = match.group(2) or "m"
    scale = Decimal("0.01") if units.startswith("c") else (
        Decimal("0.001") if units.startswith("milli") or units == "mm" else Decimal(1))
    value *= scale
    return value if math.isfinite(float(value)) else None


def score_distance(response: object, ground_truth: str) -> dict:
    """The same inclusive 0.25 m and 0.5 m rules apply after either extraction method."""
    reference = Decimal(ground_truth.strip().removesuffix(" m"))
    if not reference.is_finite() or reference < 0:
        raise ValueError("invalid distance ground truth")
    try:
        value = parse_distance(response)
    except ValueError:
        return {"status": "needs_review", "reason": "distance_not_an_unambiguous_scalar"}
    error = None if value is None else abs(value - reference)
    valid = value is not None and value >= 0
    correct = valid and error <= Decimal(str(config.B1_OPEN_ABSOLUTE_TOLERANCE_M))
    return {"status": "scored", "correct": correct, "score": float(correct),
            "correct_at_0.5m": valid and error <= Decimal("0.5"),
            "prediction_m": None if value is None else float(value),
            "absolute_error_m": None if error is None else float(error),
            "reason": "distance_tolerance" if valid else "invalid_distance"}


def _message(item: Mapping[str, object], role: str) -> str:
    values = [
        str(message.get("content") or "")
        for message in item.get("messages") or []
        if message.get("role") == role
    ]
    if len(values) != 1:
        raise ValueError(f"benchmark item needs one {role} message")
    return values[0]


def canonical_answer(task_id: str, private: Mapping[str, object]) -> str:
    """Render the concise assistant answer stored in the public JSON."""
    canonical = private.get("canonical_answer")
    if task_id == "A1":
        if canonical == "collision":
            return "Collision."
        if canonical == "no_collision":
            return "No collision."
    elif task_id == "A2":
        match = re.fullmatch(r"action_(\d+)", str(canonical))
        if match:
            return f"Action {int(match.group(1))}."
    elif task_id == "A3" and str(canonical or "").strip():
        return str(canonical).strip()
    elif task_id in {"A4", "B2"}:
        values = {
            "front": "Front.", "front-right": "Front-right.",
            "right": "Right.", "rear-right": "Rear-right.",
            "rear": "Behind.", "rear-left": "Rear-left.",
            "left": "Left.", "front-left": "Front-left.",
        }
        horizontal = private.get("horizontal_direction")
        vertical = private.get("vertical_direction")
        if horizontal is None:
            parts = str(canonical).split("|", 1)
            horizontal = parts[0]
            vertical = parts[1] if len(parts) == 2 else "level"
        if horizontal in values and vertical in {"above", "level", "below"}:
            base = values[str(horizontal)].removesuffix(".")
            suffix = "" if vertical == "level" else f" and {vertical}"
            return f"{base}{suffix}."
    elif task_id in {"B1", "B3"}:
        precise = Decimal(str(private["camera_to_target_distance_m"]))
        decimals = int(config.B1_OPEN_DISPLAY_DECIMALS)
        display = precise.quantize(
            Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
        return f"{display:.{decimals}f} m"
    elif task_id == "C1" and str(canonical).upper() in "ABCD":
        return str(canonical).upper()
    raise ValueError(f"invalid {task_id} canonical answer")


def _ordinal(value: str) -> int | None:
    if value in _ORDINALS:
        return _ORDINALS[value]
    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", value)
    return None if match is None else int(match.group(1))


def _actions(question: str) -> list[tuple[int, str]]:
    return [
        (int(index), _text(action))
        for index, action in re.findall(
            r"\((\d+)\)\s*([^,\n]+)", question)
    ]


DIRECTION_AXES = ("front_back", "left_right", "up_down")


def summarize_evaluation(rows: Iterable[dict]) -> dict:
    """Whole-question accuracy and axis credit stay separate; pending is not wrong."""
    rows = list(rows)

    def metrics(group, task):
        total = len(group)
        done = [row for row in group if row["status"] == "scored"]
        complete = total > 0 and len(done) == total
        correct = sum(row["correct"] for row in done)
        accuracy = correct / total if complete else None
        result = {"total": total, "scored": len(done), "pending": total - len(done),
                  "correct": correct, "accuracy": accuracy,
                  "truncated_responses": sum(r.get("response_finish_reason") == "length" for r in group)}
        if task == "A1":
            positive = [row for row in done if row["gt_collision"]]
            result["collision_recall"] = (
                sum(row["correct"] for row in positive) / len(positive)
                if complete and positive else None)
        if task in {"A4", "B2"}:
            result.update(joint_accuracy=accuracy,
                axis_accuracy=sum(row["score"] for row in done) / total if complete else None,
                by_axis={axis: sum(row["axis_correct"][axis] for row in done) / total
                         if complete else None for axis in DIRECTION_AXES})
        if task in {"B1", "B3"}:
            result["accuracy_at_0.25m"] = accuracy
            result["accuracy_at_0.5m"] = (
                sum(r["correct_at_0.5m"] for r in done) / total if complete else None)
        return result

    tasks = {task: metrics([r for r in rows if r["task_id"] == task], task)
             for task in templates.TASK_IDS}
    tasks["A2"]["by_start"] = {
        start: metrics([r for r in rows if r["task_id"] == "A2"
                        and r.get("starts_with") == start], "A2")
        for start in ("forward", "turn")}

    def mean(values):
        return sum(values) / len(values) if values and all(v is not None for v in values) else None

    radar = {task: tasks[task]["accuracy"] for task in ("A1", "A2", "A3", "C1")}
    radar["direction"] = mean([tasks[t]["axis_accuracy"] for t in ("A4", "B2")])
    radar["distance"] = mean([tasks[t]["accuracy"] for t in ("B1", "B3")])
    return {"tasks": tasks, "radar": radar,
            "macro_joint_accuracy": mean([t["accuracy"] for t in tasks.values() if t["total"]]),
            "complete": all(r["status"] == "scored" for r in rows)}
