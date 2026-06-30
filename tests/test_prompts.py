"""
Tests for egoconseq.tasks.prompts: o5_prompt()
TDD: these tests must FAIL before implementation, PASS after.
"""
from egoconseq.tasks.prompts import o5_prompt


def test_o5_prompt_contains_both_widths():
    """Both body widths must appear in the prompt text."""
    s = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.0)
    assert "0.20" in s or "0.2" in s, f"small width 0.20 not in prompt: {s!r}"
    assert "0.80" in s or "0.8" in s, f"large width 0.80 not in prompt: {s!r}"


def test_o5_prompt_contains_metric_horizon():
    """The fixed metric horizon must appear with 米 (meters)."""
    s = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.5)
    assert "1.5" in s, f"horizon 1.5 not in prompt: {s!r}"
    assert "米" in s, f"'米' (meters) not in prompt: {s!r}"


def test_o5_prompt_no_height():
    """Prompt must NOT mention 高 (height) — only width is the embodiment variable (design §10)."""
    s = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.0)
    assert "高" not in s, f"'高' (height) found in prompt — forbidden by design §10: {s!r}"


def test_o5_prompt_three_options_present():
    """All three answer options must appear in the prompt."""
    s = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.0)
    assert "两个都不接触" in s, f"'两个都不接触' not in prompt: {s!r}"
    assert "只有小机器人不接触" in s, f"'只有小机器人不接触' not in prompt: {s!r}"
    assert "两个都会接触" in s, f"'两个都会接触' not in prompt: {s!r}"


def test_o5_prompt_mentions_body_contact_and_width():
    """Prompt must mention 身体接触 (body contact) and 宽 (width)."""
    s = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.0)
    assert "接触" in s, f"'接触' (contact) not in prompt: {s!r}"
    assert "宽" in s, f"'宽' (width) not in prompt: {s!r}"


def test_o5_prompt_mentions_forward_movement():
    """Prompt must describe forward movement (向正前方 or similar)."""
    s = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.0)
    assert "前" in s, f"'前' (forward direction) not in prompt: {s!r}"


def test_o5_prompt_parametric_values():
    """Different parameter values should produce different prompts."""
    s1 = o5_prompt(width_small_m=0.20, width_large_m=0.80, horizon_m=1.0)
    s2 = o5_prompt(width_small_m=0.30, width_large_m=1.00, horizon_m=2.0)
    assert s1 != s2, "prompt should change with different parameters"
    assert "0.30" in s2 or "0.3" in s2
    assert "1.00" in s2 or "1.0" in s2
    assert "2.0" in s2
