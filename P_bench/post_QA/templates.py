"""Natural-language prompts shared by benchmark and training exports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable, Mapping, Protocol

from pipeline import actions as action_geometry


PROMPT_VERSION = "abc1-prompts-v8-clear-questions"
SYSTEM_PROMPT = (
    "You are a mobile robot in a static scene. The first image shows your "
    "camera view before any action.\n\n"
    "Follow the listed motions exactly, without changing the path. "
    "Actions are sequential: Forward moves along your current heading by "
    "the stated distance; Turn rotates you and your camera in place by "
    "the stated angle and direction.\n\n"
    "Your collision footprint is circular. The camera is centered above "
    "the footprint, faces your heading, and remains level.\n\n"
    "Assume continuous ground."
)
TASK_IDS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1")
INTERNAL_TO_TASK_ID = {
    "A1_collision": "A1",
    "A2_collision_step_grounding": "A2",
    "A3_contact_object": "A3",
    "A4_checkpoint_direction": "A4",
    "B1_endpoint_distance": "B1",
    "B2_endpoint_direction": "B2",
    "B3_checkpoint_distance": "B3",
    "C1_future_view_selection": "C1",
}


def _load_templates() -> dict[str, tuple[dict[str, str], ...]]:
    path = Path(__file__).with_name("templates.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != set(TASK_IDS):
        raise ValueError("ABC1 template tasks are incomplete")
    result = {}
    task_fields = {
        "A4": {"{target}"},
        "B1": {"{target}"},
        "B2": {"{target}"},
        "B3": {"{target}"},
    }
    for task_id in TASK_IDS:
        rows = tuple(dict(row) for row in raw[task_id])
        ids = [str(row.get("id")) for row in rows]
        if not rows or not all(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{task_id} template IDs must be unique")
        for row in rows:
            question = str(row.get("question") or "")
            fields = task_fields.get(task_id, set())
            if any(
                    field not in question for field in fields):
                raise ValueError(f"{row['id']} is missing a public input")
            if "<image>" in question:
                raise ValueError(f"{row['id']} duplicates the shared setup")
        result[task_id] = rows
    return result


QUESTION_TEMPLATES = _load_templates()

DIRECTION_CONVENTION = action_geometry.DIRECTION_CONVENTION
_DIRECTION_ANSWER = (
    "Answer with the horizontal direction, adding 'and above' or 'and below' "
    "if applicable. At camera level, give only the horizontal direction.")
ANSWER_FORMATS = {
    "A1": "Answer only: Collision or No collision.",
    "A2": "Use the action numbers shown. Answer only: Action <number>.",
    "A3": "Answer with the object category only.",
    "A4": _DIRECTION_ANSWER,
    "B1": "Answer in meters to two decimal places.",
    "B2": _DIRECTION_ANSWER,
    "B3": "Answer in meters to two decimal places.",
    "C1": "Answer with one letter only: A, B, C, or D.",
}


def public_task_id(task_id: str) -> str:
    """Map an internal projection task name to its public A1--C1 ID."""
    if task_id in TASK_IDS:
        return task_id
    try:
        return INTERNAL_TO_TASK_ID[task_id]
    except KeyError as error:
        raise ValueError(f"unknown QA task: {task_id}") from error


def _number(value: object) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


class _RandomChoice(Protocol):
    def choice(self, values):
        ...


def choose_template(task_id: str,
                    rng: _RandomChoice) -> Mapping[str, str]:
    """Randomly choose a template for ``task_id``."""
    return rng.choice(QUESTION_TEMPLATES[public_task_id(task_id)])


def balanced_template_ids(task_ids: Iterable[str], *, seed: int) -> list[str]:
    """Shuffle a balanced template multiset, independently of labels or actions."""
    by_task = defaultdict(list)
    for index, task in enumerate(task_ids):
        by_task[task].append(index)
    result = [""] * sum(map(len, by_task.values()))
    for task, positions in sorted(by_task.items()):
        bank = [row["id"] for row in QUESTION_TEMPLATES[task]]
        rng = random.Random(f"{seed}:templates:{task}")
        rng.shuffle(bank)
        assigned = [bank[i % len(bank)] for i in range(len(positions))]
        rng.shuffle(assigned)
        for index, name in zip(positions, assigned):
            result[index] = name
    return result


def remove_parameters(question: str, hide_params: Iterable[str]) -> str:
    """Delete selected configuration lines without changing the saved question."""
    hidden = set(hide_params)
    if not hidden:
        return question
    prefixes = tuple(prefix for parameter, prefix in (
        ("radius", "- Collision footprint radius:"),
        ("height", "- Camera optical-center height relative to the local ground reference:"),
        ("fov", "- Horizontal field of view:"),
        ("fov", "- Vertical field of view:"),
    ) if parameter in hidden)
    before, configuration = question.split("Configuration:\n", 1)
    configuration, after = configuration.split("\n\n", 1)
    lines = [line for line in configuration.splitlines() if not line.startswith(prefixes)]
    configuration = "Configuration:\n" + "\n".join(lines) + "\n\n" if lines else ""
    return before + configuration + after


def render_question(task_id: str, inputs: Mapping[str, object],
                    template: Mapping[str, str], *,
                    hide_params: Iterable[str] = ()) -> str:
    """Render a complete public prompt from one selected template."""
    task_id = public_task_id(task_id)
    if template not in QUESTION_TEMPLATES[task_id]:
        raise ValueError("template does not belong to task")
    camera = inputs["camera"]
    robot = inputs["robot"]
    actions = "\n".join(
        f"({int(action['index'])}) {action['text']}"
        for action in inputs["actions"])
    values = {
        "height_m": f"{float(camera['optical_center_height_m']):.1f}",
        "hfov_deg": _number(camera["hfov_deg"]),
        "vfov_deg": _number(camera["vfov_deg"]),
        "radius_m": _number(robot["radius_m"]),
        "actions": actions,
        "target": str(inputs.get("target") or "the target"),
    }
    query_moment = ""
    if task_id in {"A4", "B3"}:
        checkpoint = inputs["checkpoint"]
        query_moment = (
            "Query moment: After completing all earlier actions and "
            f"{_number(100 * float(checkpoint['fraction']))}% of the forward "
            f"distance in action {int(checkpoint['action_index'])}.\n\n")
    task_question = str(template["question"]).format(**values)
    direction_convention = ""
    if task_id in {"A4", "B2"}:
        direction_convention = "\n\n" + DIRECTION_CONVENTION
    target_note = ""
    if task_id in {"A4", "B1", "B2", "B3"}:
        target_note = (
            "\nThe dot's center marks a fixed surface point in the scene.\n")
    setup = [f"- Collision footprint radius: {values['radius_m']} m",
             "- Camera optical-center height relative to the local ground reference: "
             f"{values['height_m']} m",
             f"- Horizontal field of view: {values['hfov_deg']}°",
             f"- Vertical field of view: {values['vfov_deg']}°"]
    question = (
        "Your initial view:\n<image>\n"
        f"{target_note}"
        "Configuration:\n"
        + "\n".join(setup) + "\n\n"
        "Actions:\n"
        f"{actions}"
        f"{direction_convention}\n\n"
    )
    if task_id in {"A2", "A3"}:
        question += "A collision occurs during this sequence.\n\n"
    elif task_id in {"A4", "B1", "B2", "B3", "C1"}:
        question += "The action sequence is collision-free.\n\n"
    question += query_moment
    if task_id in {"B1", "B3"}:
        question += "Measure the 3D straight-line distance from your camera's optical center to the marked point.\n\n"
    question += f"Question: {task_question}"
    if task_id == "C1":
        question += (
            "\n\nCandidate images:\n"
            "Options A–D are alternative final views, not consecutive frames.\n"
        ) + "\n".join(
            f"{label}. <image>" for label in "ABCD")
    question += "\n\n" + ANSWER_FORMATS[task_id]
    return remove_parameters(question, hide_params)


def template_catalog() -> list[dict]:
    """Return all complete prompts in stable task/file order."""
    return [
        {"task_id": task_id, **dict(row)}
        for task_id in TASK_IDS
        for row in QUESTION_TEMPLATES[task_id]
    ]
