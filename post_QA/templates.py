"""Prompts shared by the benchmark and SFT exports: the released v8 task wording, the v2 motion model.

Tasks are grouped by what they predict: collision events (A1 whether, A2 at which action, A3 with
what), spatial relations of a marked point (A4 direction at a checkpoint, B1 distance at the end,
B2 direction at the end, B3 distance at a checkpoint) and the visual outcome (C1 final view).
A2/A3 are asked given that a collision occurs and A4/B1/B2/B3/C1 given a collision-free sequence;
only A1 asks whether a collision occurs.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from pipeline import actions as action_geometry

PROMPT_VERSION = "abc1-prompts-v11-fixed-floor"
SYSTEM_PROMPT = (
    "You are a mobile robot in a static scene. The first image shows your "
    "camera view before any action.\n\n"
    "Follow the listed motions exactly, without changing the path. "
    "Actions are sequential: Forward moves along your current heading by "
    "the stated distance; Turn rotates you and your camera in place by "
    "the stated angle and direction.\n\n"
    "Your body is a vertical cylinder. It spans from 5 cm above "
    "the ground at your starting position to 10 cm above the camera and stays at that ground height "
    "for the whole motion; obstacles entirely below 5 cm or entirely above the body are not touched. "
    "The camera is centered above the footprint, faces your heading, and remains level.\n\n"
    "Your body starts without touching any obstacle. Turning in place does not change the space "
    "occupied by the cylinder. Motion stops at the first contact with the scene. Contact exactly "
    "at the end of a forward command counts as contact during that command."
)
TASK_IDS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1")
DOMAINS = {"collision events": ("A1", "A2", "A3"), "spatial relations": ("A4", "B1", "B2", "B3"),
           "visual outcome": ("C1",)}
UNITS = ("m", "cm")
TARGET = "the marked point"
TARGET_TASKS = ("A4", "B1", "B2", "B3")
COLLISION, FREE = "A collision occurs during this sequence.", "The action sequence is collision-free."
PREMISES = {"A2": COLLISION, "A3": COLLISION, **dict.fromkeys(("A4", "B1", "B2", "B3", "C1"), FREE)}
DIRECTION_CONVENTION = action_geometry.DIRECTION_CONVENTION
_DIRECTION_ANSWER = ("Answer with the horizontal direction, adding 'and above' or 'and below' "
                     "if applicable. At camera level, give only the horizontal direction.")
_DISTANCE_ANSWER = {"m": "Answer in meters to two decimal places.",
                    "cm": "Answer in centimeters, rounded to the nearest centimeter."}
ANSWER_FORMATS = {
    "A1": "Answer only: Collision or No collision.",
    "A2": "Use the action numbers shown. Answer only: Action <number>.",
    "A3": "Answer with the object category only.",
    "A4": _DIRECTION_ANSWER,
    "B2": _DIRECTION_ANSWER,
    "C1": "Answer with one letter only: A, B, C, or D.",
}
PARAMETER_LINES = {"radius": ("- Collision footprint radius:",),
                   "height": ("- Camera optical-center height above the ground at your starting position:",),
                   "fov": ("- Horizontal field of view:", "- Vertical field of view:")}
ABLATIONS = {"full": (), "no_radius": ("radius",), "no_height": ("height",), "no_fov": ("fov",),
             "no_parameters": ("radius", "height", "fov")}


def _load_templates() -> dict[str, tuple[dict[str, str], ...]]:
    raw = json.loads(Path(__file__).with_name("templates.json").read_text(encoding="utf-8"))
    if set(raw) != set(TASK_IDS):
        raise ValueError("templates must cover every task")
    for task, rows in raw.items():
        ids, questions = [r["id"] for r in rows], [r["question"] for r in rows]
        if len(rows) < 10 or len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
            raise ValueError(f"{task} needs at least 10 templates with unique IDs and questions")
        if any(("{target}" in q) != (task in TARGET_TASKS) or "<image>" in q for q in questions):
            raise ValueError(f"{task}: a template misses {{target}} or repeats the shared setup")
    return {task: tuple(rows) for task, rows in raw.items()}


QUESTION_TEMPLATES = _load_templates()


def assign(keys: Iterable[str], choices, *, seed, namespace: str) -> dict:
    """Spread choices over keys in a fixed, answer-independent order; counts differ by at most one."""
    keys = sorted(set(keys), key=lambda k: hashlib.sha256(f"{seed}:{namespace}:{k}".encode()).digest())
    choices = sorted(choices, key=lambda c: hashlib.sha256(f"{seed}:{namespace}:choice:{c}".encode()).digest())
    return {key: choices[i % len(choices)] for i, key in enumerate(keys)}


def _number(value: object) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def format_length(meters, unit, *, height=False):
    value = Decimal(str(round(float(meters), 6))) * (100 if unit == "cm" else 1)
    number = format(value, "f")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    if height and unit == "m" and "." not in number:
        number += ".0"
    return f"{number} {unit}"


def remove_parameters(question: str, hide: Iterable[str]) -> str:
    """Delete the configuration lines of the hidden parameters; nothing else changes."""
    prefixes = tuple(p for name in hide for p in PARAMETER_LINES[name])
    if not prefixes:
        return question
    before, configuration = question.split("Configuration:\n", 1)
    configuration, after = configuration.split("\n\n", 1)
    lines = [line for line in configuration.splitlines() if not line.startswith(prefixes)]
    return before + ("Configuration:\n" + "\n".join(lines) + "\n\n" if lines else "") + after


def render_question(task_id: str, inputs: Mapping[str, object], template: Mapping[str, str], *,
                    unit: str = "m", hide: Iterable[str] = ()) -> str:
    """The user message of one item; every length (radius, height, forwards, answer) uses one unit."""
    if template not in QUESTION_TEMPLATES[task_id] or unit not in UNITS:
        raise ValueError("unknown template or unit")
    camera, robot = inputs["camera"], inputs["robot"]
    actions = []
    for action in inputs["actions"]:
        if action["type"] == "forward":
            text = "forward " + format_length(action["meters"], unit)
        else:
            angle = abs(action['degrees'])
            text = f"turn {'right' if action['degrees'] >= 0 else 'left'} {_number(angle)} {'degree' if angle == 1 else 'degrees'}"
        actions.append(f"({int(action['index'])}) {text}")
    setup = ["- Collision footprint radius: " + format_length(robot["radius_m"], unit),
             "- Camera optical-center height above the ground at your starting position: "
             + format_length(camera["optical_center_height_m"], unit, height=True),
             f"- Horizontal field of view: {_number(camera['hfov_deg'])}°",
             f"- Vertical field of view: {_number(camera['vfov_deg'])}°"]
    question = ("Your initial view:\n<image>\n"
                + ("\nThe dot's center marks a fixed surface point in the scene.\n" if task_id in TARGET_TASKS else "")
                + "Configuration:\n" + "\n".join(setup) + "\n\nActions:\n" + "\n".join(actions)
                + ("\n\n" + DIRECTION_CONVENTION if task_id in ("A4", "B2") else "") + "\n\n")
    if task_id in PREMISES:
        question += PREMISES[task_id] + "\n\n"
    if task_id in ("A4", "B3"):
        checkpoint = inputs["checkpoint"]
        question += (f"Query moment: After completing all earlier actions and {_number(100 * float(checkpoint['fraction']))}% "
                     f"of the forward distance in action {int(checkpoint['action_index'])}.\n\n")
    if task_id in ("B1", "B3"):
        question += "Measure the 3D straight-line distance from your camera's optical center to the marked point.\n\n"
    question += "Question: " + template["question"].format(target=TARGET)
    if task_id == "C1":
        question += ("\n\nCandidate images:\nOptions A–D are alternative final views, not consecutive frames. "
                     "Exactly one candidate is correct.\n"
                     + "\n".join(f"{label}. <image>" for label in "ABCD"))
    question += "\n\n" + (_DISTANCE_ANSWER[unit] if task_id in ("B1", "B3") else ANSWER_FORMATS[task_id])
    return remove_parameters(question, hide)
