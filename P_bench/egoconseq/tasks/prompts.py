"""
egoconseq.tasks.prompts
=======================
Chinese-language prompt builders for EgoConseq-Bench operations.

Design constraints:
- §6:  O5 action horizon is a FIXED physical path in metres.
       The prompt must emphasise this shared metric distance.
- §10: Width (diameter) is the ONLY embodiment variable mentioned;
       height must NOT appear.
- New O5 design: ONE robot per question, natural binary contact judgment.
"""

from __future__ import annotations


def o5_prompt(
    diameter_m: float,
    horizon_m: float,
) -> str:
    """Return a Chinese VQA question for the O5 single-robot binary contact task.

    The question describes ONE cylindrical-chassis robot with the given
    chassis diameter (metres), moving forward a fixed metric distance
    (horizon_m metres).  The model must judge whether contact will occur.

    Parameters
    ----------
    diameter_m:
        Chassis diameter (metres) of the robot.  This is the ONLY
        embodiment variable (design §10): height must NOT appear.
    horizon_m:
        The fixed forward distance the robot travels (metres).
        This is the RED-LINE constraint (design §6): it is a physical
        path length, NOT derived from body width.

    Returns
    -------
    str
        Chinese prompt string.  Guaranteed to contain "圆柱形底盘", the
        diameter value with "米", "会接触", and "前".
        Guaranteed NOT to contain "高", "小机器人", "大机器人", or "两个".
    """
    prompt = (
        f"图中是一个圆柱形底盘的机器人，底盘直径约 {diameter_m:.1f} 米。\n"
        f"它从当前位置朝正前方移动约 {horizon_m:.1f} 米。\n"
        f"只看这张第一视角图，判断它的身体在当前可见的局部空间内会不会与障碍物发生接触。\n"
        f"请直接给出这个动作是否会接触的结论。"
    )
    return prompt
