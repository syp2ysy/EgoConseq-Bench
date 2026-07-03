from vln_baseline.actions import (
    ACTION_ID_TO_TEXT,
    build_navida_random_action_chunks,
    build_action_segments,
    expand_action_text,
    parse_action_sequence,
    parse_and_expand_actions,
)


def test_action_id_to_text_uses_navida_primitives():
    assert ACTION_ID_TO_TEXT[0] == "stop"
    assert ACTION_ID_TO_TEXT[1] == "forward 25 cm"
    assert ACTION_ID_TO_TEXT[2] == "turn left 15 degree"
    assert ACTION_ID_TO_TEXT[3] == "turn right 15 degree"


def test_build_action_segments_groups_consecutive_primitives_and_returns_consumed():
    segments, consumed = build_action_segments([1, 1, 1, 1, 2, 2, 0], start=0, max_segments=3)
    assert segments == [
        "forward 75 cm",
        "forward 25 cm",
        "turn left 30 degree",
    ]
    assert consumed == 6


def test_build_action_segments_stops_at_stop():
    segments, consumed = build_action_segments([1, 2, 0, 1], start=2, max_segments=3)
    assert segments == [
        "stop",
    ]
    assert consumed == 1


def test_navida_random_chunks_can_leave_repeated_actions_unmerged():
    class NoMergeRandom:
        def random(self):
            return 1.0

    chunks = build_navida_random_action_chunks(
        [1, 1, 1, 1, 0],
        max_segments=3,
        merge_probability=0.7,
        rng=NoMergeRandom(),
    )

    assert chunks == [
        (0, "forward 25 cm, forward 25 cm, forward 25 cm", 3),
        (3, "forward 25 cm, stop", 2),
    ]


def test_parse_action_sequence_accepts_text_and_numbers():
    parsed = parse_action_sequence("Forward 25 cm, turn LEFT 15 degree, stop")
    assert parsed == [(1, 25.0), (2, 15.0), (0, None)]


def test_expand_action_text_clamps_to_three_primitives():
    assert expand_action_text(1, 75.0) == [1, 1, 1]
    assert expand_action_text(1, 125.0) == [1, 1, 1]
    assert expand_action_text(2, 30.0) == [2, 2]
    assert expand_action_text(0, None) == [0]


def test_parse_and_expand_actions_executes_only_requested_segments():
    class FixedRandom:
        def randint(self, low, high):
            return low

    actions, used_fallback = parse_and_expand_actions(
        "forward 75 cm, turn left 30 degree, turn right 45 degree",
        max_segments=2,
        rng=FixedRandom(),
    )
    assert actions == [1, 1, 1, 2, 2]
    assert used_fallback is False
