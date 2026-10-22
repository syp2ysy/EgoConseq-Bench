"""Canonical answers and fixed distance/aggregation rules for the evaluator."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import math
import re
from typing import Iterable, Mapping

from pipeline import config
from post_QA import categories


DIRECTION_AXES = {
    "front_back": ("front", "aligned", "rear"),
    "left_right": ("left", "aligned", "right"),
    "up_down": ("above", "level", "below"),
}
DIRECTION_WORDS = ("front", "rear", "left", "right", "above", "level", "below")
HORIZONTAL_AXES = {
    "front": ("front", "aligned"), "front-left": ("front", "left"),
    "left": ("aligned", "left"), "rear-left": ("rear", "left"),
    "rear": ("rear", "aligned"), "rear-right": ("rear", "right"),
    "right": ("aligned", "right"), "front-right": ("front", "right"),
}
DIRECTION_SYNONYMS = {"forward": "front", "ahead": "front", "behind": "rear",
                      "back": "rear", "backward": "rear", "up": "above", "down": "below",
                      "upper": "above", "top": "above", "lower": "below", "bottom": "below"}
COMPASS_WORDS = {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"}
DISTANCE_UNITS = {
    **dict.fromkeys(("m", "meter", "meters", "metre", "metres"), Decimal("1")),
    **dict.fromkeys(("cm", "centimeter", "centimeters", "centimetre", "centimetres"), Decimal("0.01")),
    **dict.fromkeys(("mm", "millimeter", "millimeters", "millimetre", "millimetres"), Decimal("0.001")),
    **dict.fromkeys(("km", "kilometer", "kilometers", "kilometre", "kilometres"), Decimal("1000")),
    **dict.fromkeys(("ft", "foot", "feet"), Decimal("0.3048")),
    **dict.fromkeys(("in", "inch", "inches"), Decimal("0.0254")),
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", ("" if value is None else str(value)).strip().casefold())


def direction_axes(horizontal=None, vertical=None) -> dict:
    """V8 labels determine neutral components; missing components stay absent."""
    result = dict(zip(("front_back", "left_right"), HORIZONTAL_AXES[horizontal])) if horizontal else {}
    if vertical:
        result["up_down"] = vertical
    return result


def unknown_direction_axes(text: str) -> set:
    """Explicit conflicts/compass words block axes; they never select a direction."""
    text = re.sub(r"\b(above|below) (?:the )?camera level\b", r"\1", _text(text))
    words = {DIRECTION_SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", text)}
    unknown = {axis for axis, values in DIRECTION_AXES.items() if len(words & set(values)) > 1}
    if words & COMPASS_WORDS:
        unknown.update(("front_back", "left_right"))
    if "upright" in words and not words & {"above", "below", "level"}:
        unknown.add("up_down")  # Posture alone does not state target elevation.
    return unknown


def normalize_direction(words: Iterable[str], unknown_axes: Iterable[str] = ()) -> dict:
    """Map stated canonical words to axes, with the same defaults for rules and LLM."""
    words = set(words)
    unknown_axes = set(unknown_axes)
    fb, lr, ud = (words & set(values) for values in
                  (("front", "rear"), ("left", "right"), ("above", "level", "below")))
    result = {}
    if fb or lr:
        for axis, values in (("front_back", fb), ("left_right", lr)):
            if len(values) <= 1 and axis not in unknown_axes:
                result[axis] = next(iter(values), "aligned")
    if "up_down" not in unknown_axes:
        if len(ud) == 1:
            result["up_down"] = next(iter(ud))
        elif not ud and result:
            result["up_down"] = "level"
    return result


def parse_direction(text: str) -> dict:
    """Normalize stated direction words; never turn numeric angles into labels."""
    text = _text(text).strip(" .,'\"“”")
    text = text.replace("_", "-")
    text = re.sub(r"\b(?:straight|dead) ahead\b", "front", text)
    text = re.sub(r"(?<!\w)['\"]|['\"](?!\w)", "", text)
    text = re.sub(r"\b(?:slightly|approximately|roughly|simply)\s+", "", text)
    text = re.sub(r"^(?:the )?(?:marked point|dot|point|target) (?:is|lies) ", "", text)
    text = re.sub(r" (?:of|relative to) (?:the |your )?(?:camera|robot)$", "", text)
    text = re.sub(r"\bat (?:the )?camera level\b", "level", text)
    corner = re.fullmatch(r"(?:towards? |in |at )?(?:the )?(upper|top|lower|bottom)[ -](left|right) "
                          r"corner of (?:the |your )?(?:camera['’]s )?(?:image|view|field of view)", text)
    if corner:
        return direction_axes("front-" + corner[2], DIRECTION_SYNONYMS[corner[1]])
    if re.fullmatch(r"(?:in |at )?(?:the )?center of (?:the |your )?(?:camera'?s? )?(?:view|field of view)", text):
        return direction_axes("front", "level")
    words = {DIRECTION_SYNONYMS.get(w, w) for w in re.split(r"[\s,\-]+", text) if w}
    words -= {"and", "to", "the", "in", "on", "directly"}
    if words and not words - (set(DIRECTION_WORDS) | {"upright"} | COMPASS_WORDS):
        return normalize_direction(words & set(DIRECTION_WORDS), unknown_direction_axes(text))
    raise ValueError("expected a stated camera-relative direction")


def parse_distance(response: object, *, default_unit: str = "m") -> Decimal | None:
    """Read one explicit quantity; unrecognized prose must be extracted by the LLM."""
    raw = ("" if response is None else str(response)).replace("−", "-")
    text = _text(raw).strip("`*\"'").removesuffix(".")
    if not text or text in {"nan", "inf", "infinity", "+inf", "-inf", "+infinity", "-infinity"}:
        return None
    number = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?"
    unit = "|".join(DISTANCE_UNITS)
    match = re.fullmatch(
        r"(?:(?:the\s+)?(?:final\s+)?(?:camera-to-(?:point|target)\s+)?distance\s*(?:is|:|=)\s*)?"
        r"(?:(?:approximately|about|around|roughly|~|≈)\s*)?"
        rf"({number})\s*({unit})?(?:\s+away)?", text)
    if match is None:
        raise ValueError("expected one distance with a known unit, not a range or explanation")
    value = Decimal(match.group(1)) * DISTANCE_UNITS[match.group(2) or default_unit]
    return value if math.isfinite(float(value)) else None


def requested_distance_unit(question: str) -> str:
    """Read the fixed answer instruction; legacy questions default to meters."""
    return "cm" if re.search(r"\bAnswer in centimeters\b", question, re.I) else "m"


def format_distance(meters, unit: str = "m") -> str:
    """Keep the same physical answer precision in meters and centimeters."""
    if unit not in {"m", "cm"}:
        raise ValueError("distance output unit must be m or cm")
    decimals = int(config.B1_OPEN_DISPLAY_DECIMALS)
    display = Decimal(str(meters)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
    if unit == "cm":
        return f"{display * 100:.{max(0, decimals - 2)}f} cm"
    return f"{display:.{decimals}f} m"


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
    if task_id in {"A2", "A3"} and canonical == "no_collision":
        return "No collision."
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
        return categories.display_category_name(canonical)
    elif task_id in {"A4", "B2"}:
        values = {
            "front": "Front.", "front-right": "Front-right.",
            "right": "Right.", "rear-right": "Rear-right.",
            "rear": "Rear.", "rear-left": "Rear-left.",
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
        return format_distance(private["camera_to_target_distance_m"])
    elif task_id == "C1" and str(canonical).upper() in "ABCD":
        return str(canonical).upper()
    raise ValueError(f"invalid {task_id} canonical answer")


def _ordinal(value: str) -> int | None:
    if value in _ORDINALS:
        return _ORDINALS[value]
    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", value)
    return None if match is None else int(match.group(1))


def _actions(question: str) -> list[tuple[int, str]]:
    return [(int(index), _text(action)) for index, action in re.findall(r"\((\d+)\)\s*([^,\n]+)", question)]
