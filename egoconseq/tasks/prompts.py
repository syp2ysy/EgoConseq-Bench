"""
egoconseq.tasks.prompts
=======================
Chinese-language prompt builders for EgoConseq-Bench operations.

Design constraints:
- §6:  O5 action horizon is a FIXED physical path in metres, identical across
       both bodies.  The prompt must emphasise this shared metric distance.
- §10: Width is the ONLY embodiment variable mentioned; height must NOT appear.
"""

from __future__ import annotations


def o5_prompt(
    width_small_m: float,
    width_large_m: float,
    horizon_m: float,
) -> str:
    """Return a Chinese VQA question for the O5 footprint-counterfactual task.

    The question presents two robots with different body widths but an
    IDENTICAL fixed metric path (horizon_m metres forward).  The model must
    decide which robot's body contacts an obstacle.

    Parameters
    ----------
    width_small_m:
        Body width (metres) of the narrow robot.
    width_large_m:
        Body width (metres) of the wide robot.
    horizon_m:
        The shared fixed forward distance both robots travel (metres).
        This is the RED-LINE constraint (design §6): it must NOT equal any
        body-width derived value — it is the same physical path length for
        both bodies.

    Returns
    -------
    str
        Chinese prompt string.  Guaranteed to contain both width values,
        the metric horizon with "米", the three answer options, and the
        words "接触" and "宽".  Guaranteed NOT to contain "高".
    """
    prompt = (
        f"图中展示了一段走廊场景。"
        f"现有两个机器人，身体宽度分别为 {width_small_m:.2f} 米（小机器人）"
        f"和 {width_large_m:.2f} 米（大机器人），"
        f"两者起始位置和朝向完全相同，且都向正前方移动 {horizon_m} 米。\n\n"
        f"请根据图像中可见的障碍物和通道宽度，判断哪个机器人的身体会与障碍物发生接触。\n\n"
        f"请从以下三个选项中选择一个作答：\n"
        f"A. 两个都不接触\n"
        f"B. 只有小机器人不接触（即大机器人会接触，小机器人不会）\n"
        f"C. 两个都会接触\n\n"
        f"注意：两个机器人行进路径长度相同（均为 {horizon_m} 米），"
        f"唯一的差别是身体宽度（宽窄不同）。请只回答选项字母（A/B/C）。"
    )
    return prompt
