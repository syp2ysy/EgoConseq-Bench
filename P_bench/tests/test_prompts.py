"""
Tests for egoconseq.tasks.prompts: o5_prompt()
TDD: new single-robot binary prompt tests.

Design change (new O5):
- ONE robot per question, cylindrical chassis, given DIAMETER in metres.
- Natural binary judgment prompt, not an MCQ/options list.
- No two-robot phrasing, no height mention.
"""
from egoconseq.tasks.prompts import o5_prompt


def test_o5_prompt_contains_cylindrical_chassis():
    """Prompt must describe a cylindrical chassis robot (圆柱形底盘)."""
    s = o5_prompt(diameter_m=0.80, horizon_m=1.0)
    assert "圆柱形底盘" in s, f"'圆柱形底盘' not in prompt: {s!r}"


def test_o5_prompt_contains_diameter_and_unit():
    """The diameter value and '米' must appear in the prompt."""
    s = o5_prompt(diameter_m=0.8, horizon_m=1.86)
    assert "米" in s, f"'米' not in prompt: {s!r}"
    # diameter formatted as .1f → "0.8"
    assert "0.8" in s, f"diameter '0.8' not in prompt: {s!r}"


def test_o5_prompt_asks_binary_contact_judgment():
    """Prompt must ask a contact/no-contact judgment without listing options."""
    s = o5_prompt(diameter_m=0.80, horizon_m=1.0)
    assert "会接触" in s, f"'会接触' not in prompt: {s!r}"
    assert "判断" in s, f"'判断' not in prompt: {s!r}"


def test_o5_prompt_no_two_robot_phrasing():
    """Prompt must NOT contain two-robot phrasing."""
    s = o5_prompt(diameter_m=0.80, horizon_m=1.0)
    assert "小机器人" not in s, f"'小机器人' (two-robot phrasing) found in prompt: {s!r}"
    assert "大机器人" not in s, f"'大机器人' (two-robot phrasing) found in prompt: {s!r}"
    assert "两个" not in s, f"'两个' (two robots) found in prompt: {s!r}"


def test_o5_prompt_not_multiple_choice_format():
    """Prompt must not look like an MCQ or slash-separated option list."""
    s = o5_prompt(diameter_m=0.80, horizon_m=1.0)
    assert " / " not in s
    assert "A." not in s and "B." not in s and "C." not in s
    assert "选项" not in s


def test_o5_prompt_no_height():
    """Prompt must NOT mention 高 (height) — only diameter is the embodiment variable."""
    s = o5_prompt(diameter_m=0.80, horizon_m=1.0)
    assert "高" not in s, f"'高' (height) found in prompt — forbidden: {s!r}"


def test_o5_prompt_contains_forward_movement():
    """Prompt must describe forward movement."""
    s = o5_prompt(diameter_m=0.80, horizon_m=1.0)
    assert "前" in s, f"'前' (forward direction) not in prompt: {s!r}"


def test_o5_prompt_parametric_values():
    """Different parameter values should produce different prompts."""
    s1 = o5_prompt(diameter_m=0.40, horizon_m=1.0)
    s2 = o5_prompt(diameter_m=0.80, horizon_m=1.86)
    assert s1 != s2, "prompt should change with different parameters"
    # s2 should contain 0.8 and 1.9 (horizon rounded to .1f)
    assert "0.8" in s2, f"diameter 0.8 not in s2: {s2!r}"
    assert "1.9" in s2, f"horizon 1.86→1.9 not in s2: {s2!r}"


def test_o5_prompt_horizon_in_text():
    """The horizon value must appear in the prompt."""
    s = o5_prompt(diameter_m=0.40, horizon_m=2.5)
    assert "2.5" in s, f"horizon 2.5 not in prompt: {s!r}"
