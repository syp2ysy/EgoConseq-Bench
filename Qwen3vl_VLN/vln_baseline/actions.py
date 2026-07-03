import math
import random
import re
from typing import Any, List, Optional, Tuple


STOP = 0
FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

FORWARD_DISTANCE_CM = 25
TURN_ANGLE_DEG = 15

ACTION_ID_TO_TEXT = {
    STOP: "stop",
    FORWARD: f"forward {FORWARD_DISTANCE_CM} cm",
    TURN_LEFT: f"turn left {TURN_ANGLE_DEG} degree",
    TURN_RIGHT: f"turn right {TURN_ANGLE_DEG} degree",
}

ACTION_TEXT_TO_ID = {
    "stop": STOP,
    "forward": FORWARD,
    "turn left": TURN_LEFT,
    "left": TURN_LEFT,
    "turn right": TURN_RIGHT,
    "right": TURN_RIGHT,
}


def action_id_to_text(action_id: int, repeats: int = 1) -> str:
    if action_id not in ACTION_ID_TO_TEXT:
        raise ValueError(f"Invalid action id: {action_id}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if action_id == STOP:
        return ACTION_ID_TO_TEXT[action_id]
    if action_id == FORWARD:
        return f"forward {FORWARD_DISTANCE_CM * repeats} cm"
    if action_id == TURN_LEFT:
        return f"turn left {TURN_ANGLE_DEG * repeats} degree"
    if action_id == TURN_RIGHT:
        return f"turn right {TURN_ANGLE_DEG * repeats} degree"
    return ACTION_ID_TO_TEXT[action_id]


def build_action_segments(
    actions: List[int],
    start: int,
    max_segments: int,
    max_repeats_per_segment: int = 3,
) -> Tuple[List[str], int]:
    segments: List[str] = []
    idx = start
    while idx < len(actions) and len(segments) < max_segments:
        action_id = actions[idx]
        if action_id == STOP:
            segments.append(action_id_to_text(action_id))
            idx += 1
            break

        repeats = 1
        while (
            idx + repeats < len(actions)
            and actions[idx + repeats] == action_id
            and repeats < max_repeats_per_segment
        ):
            repeats += 1

        segments.append(action_id_to_text(action_id, repeats=repeats))
        idx += repeats
    return segments, idx - start


def _combine_last_segment(action_text: str, next_action_text: str) -> Optional[str]:
    split_idx = action_text.rfind(", ")
    prefix = action_text[: split_idx + 2] if split_idx >= 0 else ""
    last_segment = action_text[split_idx + 2 :] if split_idx >= 0 else action_text
    last_number = re.search(r"-?\d+", last_segment)
    next_number = re.search(r"-?\d+", next_action_text)
    if last_number is None or next_number is None:
        return None

    value = int(last_number.group()) + int(next_number.group())
    if "forward" in last_segment and "forward" in next_action_text:
        if value <= 3 * FORWARD_DISTANCE_CM:
            return f"{prefix}forward {value} cm"
        return None
    if "turn left" in last_segment and "turn left" in next_action_text:
        if value <= 3 * TURN_ANGLE_DEG:
            return f"{prefix}turn left {value} degree"
        return None
    if "turn right" in last_segment and "turn right" in next_action_text:
        if value <= 3 * TURN_ANGLE_DEG:
            return f"{prefix}turn right {value} degree"
        return None
    return None


def build_navida_random_action_chunks(
    actions: List[int],
    max_segments: int,
    merge_probability: float = 0.7,
    rng: Optional[Any] = None,
) -> List[Tuple[int, str, int]]:
    if not actions:
        return []
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    if rng is None:
        rng = random.Random()

    chunks: List[Tuple[int, str, int]] = []
    start = 0
    answer = action_id_to_text(actions[0])
    last_action = actions[0]

    for idx in range(1, len(actions)):
        action_id = actions[idx]
        action_text = action_id_to_text(action_id)
        combined = None
        if action_id == last_action and rng.random() <= merge_probability:
            combined = _combine_last_segment(answer, action_text)

        if combined is not None:
            answer = combined
        elif answer.count(",") < max_segments - 1:
            answer = f"{answer}, {action_text}"
        else:
            chunks.append((start, answer, idx - start))
            start = idx
            answer = action_text
        last_action = action_id

    chunks.append((start, answer, len(actions) - start))
    return chunks


def format_action_segments(actions: List[int], start: int, max_segments: int) -> Tuple[str, int]:
    segments, consumed = build_action_segments(actions, start, max_segments)
    return ", ".join(segments), consumed


def parse_single_action(text: str) -> Tuple[Optional[int], Optional[float]]:
    normalized = text.strip().lower()
    if not normalized:
        return None, None
    if "stop" in normalized:
        return STOP, None

    number_match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    numeric = float(number_match.group()) if number_match else None

    if "forward" in normalized:
        return FORWARD, numeric if numeric is not None else float(FORWARD_DISTANCE_CM)
    if "left" in normalized:
        return TURN_LEFT, numeric if numeric is not None else float(TURN_ANGLE_DEG)
    if "right" in normalized:
        return TURN_RIGHT, numeric if numeric is not None else float(TURN_ANGLE_DEG)
    return None, None


def parse_action_sequence(output: str) -> List[Tuple[Optional[int], Optional[float]]]:
    answer_match = re.search(r"<answer>(.*?)</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        output = answer_match.group(1)
    parts = [part for part in re.split(r",|\n", output) if part.strip()]
    return [parse_single_action(part) for part in parts]


def expand_action_text(action_id: Optional[int], numeric: Optional[float]) -> List[int]:
    if action_id is None:
        return []
    if action_id == STOP:
        return [STOP]
    if action_id == FORWARD:
        value = numeric if numeric is not None else float(FORWARD_DISTANCE_CM)
        count = max(1, min(3, int(round(value / FORWARD_DISTANCE_CM))))
        return [FORWARD] * count
    if action_id in (TURN_LEFT, TURN_RIGHT):
        value = numeric if numeric is not None else float(TURN_ANGLE_DEG)
        count = max(1, min(3, int(round(value / TURN_ANGLE_DEG))))
        return [action_id] * count
    return []


def parse_and_expand_actions(
    output: str,
    max_segments: int,
    rng: random.Random,
) -> Tuple[List[int], bool]:
    pending_actions: List[int] = []
    used_fallback = False
    for action_id, numeric in parse_action_sequence(output)[:max_segments]:
        expanded = expand_action_text(action_id, numeric)
        if not expanded:
            expanded = [rng.randint(FORWARD, TURN_RIGHT)]
            used_fallback = True
        pending_actions.extend(expanded)
        if STOP in expanded:
            break
    if not pending_actions:
        pending_actions = [rng.randint(FORWARD, TURN_RIGHT)]
        used_fallback = True
    return pending_actions, used_fallback


def sanitize_metric(value: float) -> float:
    return 0.0 if math.isnan(value) or math.isinf(value) else float(value)
