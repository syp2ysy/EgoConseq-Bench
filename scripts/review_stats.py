"""Read-only statistics for any QA release. No simulator, GT generation or selection rules.

The UI receives per-question measurements for interactive filtering. stats.json also retains
release-scoped summaries and the original build reports, with their scope made explicit.
"""

from collections import Counter
import hashlib
import math

from post_QA.answers import parse_distance

TASKS = {"A1": "是否碰撞", "A2": "首次碰撞动作", "A3": "首次接触类别",
         "B3": "中途距离", "B1": "终点距离", "A4": "中途方向", "B2": "终点方向", "C1": "未来视图辨别"}
DOMAINS = {
    "collision": {"name": "碰撞事件预测", "short": "碰撞事件", "tasks": ["A1", "A2", "A3"], "color": "#23624d"},
    "spatial": {"name": "自运动下的空间关系更新", "short": "空间关系", "tasks": ["B3", "B1", "A4", "B2"], "color": "#6266a5"},
    "visual": {"name": "动作条件下的视觉后果辨别", "short": "视觉后果", "tasks": ["C1"], "color": "#ad633d"},
}
DOMAIN_FOR = {t: d for d, info in DOMAINS.items() for t in info["tasks"]}
DISTANCE_EDGES = [0, 1, 2, 4, 8]
SOURCES = {
    "vsi": {"name": "VSI-Bench · 图 5", "url": "https://arxiv.org/html/2412.14171",
            "method": "分层任务组成与按来源分组的输入跨度分布"},
    "mmsi": {"name": "MMSI-Bench · 图 3 / 表 2", "url": "https://arxiv.org/html/2505.23764",
             "method": "类别组成；题数、独立图像数和每题图像数分别报告"},
    "robo": {"name": "RoboSpatial · 表 7", "url": "https://arxiv.org/html/2411.16537",
             "method": "按来源、划分分别统计场景、图像与各任务题量"},
    "action": {"name": "ActionEQA · 官方数据表", "url": "https://github.com/ActionEQA/ActionEQA_TMLR_2026",
               "method": "独立图像、episode 和问题数分开；来源与任务设置交叉统计"},
    "gqa": {"name": "GQA · 图 10 / 13", "url": "https://cs.stanford.edu/people/dorarad/gqa/gqaPaper.pdf",
            "method": "条件答案分布和语义操作长度；保留类别长尾"},
    "clevr": {"name": "CLEVR · 图 3 / 7", "url": "https://arxiv.org/pdf/1612.06890",
              "method": "题型、语言长度、独立问题；区分题面长度与有效问题规模"},
    "mmbench": {"name": "MMBench · 图 2 / 4", "url": "https://arxiv.org/html/2307.06281",
                "method": "分层分类；正确答案与模型选择的位置分布"},
    "video": {"name": "Video-MME · 图 2 / 表 2", "url": "https://arxiv.org/html/2405.21075",
              "method": "输入跨度与任务的交叉分布；文本字段长度"},
    "ec": {"name": "ECBench · 图 3", "url": "https://arxiv.org/html/2501.05031",
           "method": "场景、数据来源与问题类别分别统计"},
    "open": {"name": "OpenEQA · 官方统计与基线", "url": "https://open-eqa.github.io/",
             "method": "题型、来源和评测设置；视觉与无图基线分开报告"},
    "spar": {"name": "SPAR · §3.3–3.4", "url": "https://arxiv.org/html/2503.22976",
             "method": "生成、筛选、人工验证与最终 benchmark 分开报告"},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance_bin(distance):
    """Fixed display bins in metres, independent of mutable build specs and rendered units."""
    for lo, hi in zip(DISTANCE_EDGES, DISTANCE_EDGES[1:]):
        if distance < hi:
            return f"{lo:g}–{hi:g} m"
    return f"≥ {DISTANCE_EDGES[-1]:g} m"


def measurements(task, gt, answer, unit):
    """Derive descriptive features only from stored actions and stored GT; never recompute GT."""
    actions = gt["actions"]
    lengths = [a["m"] for a in actions if a["type"] == "forward"]
    turns = [a["deg"] for a in actions if a["type"] == "turn"]
    runs = []
    for a in actions:
        name = "前进" if a["type"] == "forward" else "转向"
        if not runs or runs[-1] != name:
            runs.append(name)
    out = dict(commands=len(actions), forwards=len(lengths), turns=len(turns),
               forward_m=sum(lengths), turn_abs_deg=sum(map(abs, turns)), turn_net_deg=sum(turns),
               start=actions[0]["type"], pattern=" → ".join(runs),
               turn_side="双向" if any(t < 0 for t in turns) and any(t > 0 for t in turns) else
               "左转" if any(t < 0 for t in turns) else "右转" if any(t > 0 for t in turns) else "无转向",
               program_text=" → ".join(f"前进 {a['m']:g} m" if a["type"] == "forward" else
                                         f"{'右' if a['deg'] > 0 else '左'}转 {abs(a['deg']):g}°" for a in actions))
    # This is the recorded query prefix, not a claim about minimum reasoning steps.
    index, fraction = None, 1.
    if task == "A2":
        index = gt["first_collision_action"]
        fraction = gt["contact_progress_m"] / gt["command_length_m"]
        ordinal = sum(a["type"] == "forward" for a in actions[:index])
        unique_max = sum(math.isclose(x, max(lengths), rel_tol=0, abs_tol=1e-9) for x in lengths) == 1
        out.update(collision_action=index, forward_ordinal=ordinal,
                   position="首次" if ordinal == 1 else "末次" if ordinal == len(lengths) else "中间",
                   longest_hit=(math.isclose(lengths[ordinal - 1], max(lengths), rel_tol=0, abs_tol=1e-9)
                                if unique_max else None),
                   command_end_margin_m=gt["command_length_m"] - gt["contact_progress_m"])
    elif task in ("A4", "B3"):
        index, fraction = gt["checkpoint_action"], gt["checkpoint_fraction"]
        out.update(checkpoint_action=index, checkpoint_fraction=fraction)
    elif task in ("B1", "B2", "C1"):
        index = len(actions)
    if index is not None:
        prefix = actions[:index]
        out.update(prefix_commands=index,
                   prefix_forward_m=sum(a["m"] * (fraction if i == index - 1 else 1)
                                        for i, a in enumerate(prefix) if a["type"] == "forward"),
                   prefix_turn_deg=sum(abs(a["deg"]) for a in prefix if a["type"] == "turn"))
    for field in ("collision", "depth_view_collision", "corridor_samples", "corridor_coverage", "corridor_out_of_frame",
                  "corridor_occluded", "visible_points", "category", "source_category", "horizontal", "vertical",
                  "azimuth_deg", "elevation_deg", "boundary_margin_deg", "planar_range_m", "min_block_l1"):
        if field in gt:
            out[field] = gt[field]
    if task == "A1":
        out["depth_agree"] = gt["depth_view_collision"] == gt["collision"]
    if task == "A2":
        out["depth_agree"] = gt["depth_view_contact_command"] == gt["first_collision_action"]
    if task in ("B1", "B3"):
        d, d0 = gt["distance_m"], gt["initial_distance_m"]
        delta = d - d0
        out.update(distance_m=d, initial_distance_m=d0, distance_delta_m=delta,
                   relative_change=abs(delta) / d0 if d0 else None,
                   distance_answer_m=float(parse_distance(answer, default_unit=unit)),
                   distance_change="基本不变" if abs(delta) < .05 else "小变化" if abs(delta) < .3 else
                   "明显变远" if delta > 0 else "明显变近")
    return out


def summary(items):
    """Exact, unrounded denominators; images are identified by recorded content hashes."""
    programs = Counter(i["program"] for i in items)
    scene_counts = Counter((i["dataset"], i["scene"]) for i in items)
    return {
        "questions": len(items), "observations": len({(i["dataset"], i["obs"]) for i in items}),
        "scenes": len(scene_counts), "initial_images": len({i["raw_hash"] for i in items}),
        "input_images": len({h for i in items for h in i["image_hashes"]}),
        "image_references": sum(len(i["images"]) for i in items),
        "pairs": len({i["pair"] for i in items if i["pair"]}),
        "programs": len(programs), "program_repeat_excess": len(items) - len(programs),
        "top10_program_questions": sum(n for _, n in programs.most_common(10)),
        "max_scene_questions": max(scene_counts.values(), default=0),
        "tasks": dict(Counter(i["task"] for i in items)), "datasets": dict(Counter(i["dataset"] for i in items)),
        "domains": dict(Counter(i["domain"] for i in items)),
    }


def snapshot(items, releases):
    """Exportable audit snapshot. Fine-grained histograms are rendered from the same measurements."""
    sets = {}
    for r, release in enumerate(releases):
        for kind in sorted({i["set"] for i in items if i["r"] == r}):
            rows = [i for i in items if i["r"] == r and i["set"] == kind]
            sets[f"{r}/{kind}"] = {"release": release["name"], **summary(rows)}
    overlaps = []
    for a in range(len(releases)):
        for b in range(a + 1, len(releases)):
            groups = [[i for i in items if i["r"] == r and i["set"] == "benchmark"] for r in (a, b)]
            scenes = [{(i["dataset"], i["scene"]) for i in g} for g in groups]
            hashes = [{h for i in g for h in i["image_hashes"]} for g in groups]
            overlaps.append(dict(a=a, b=b, scenes=len(scenes[0] & scenes[1]), images=len(hashes[0] & hashes[1])))
    return {"schema": "pbench.review-statistics.v1", "taxonomy": {"tasks": TASKS, "domains": DOMAINS},
            "sources": SOURCES, "sets": sets, "overlaps": overlaps,
            "notes": ["题数按公开 QA 计；参数消融版本不重复计数；半径对照另列题数与对数。",
                      "独立初始图按未打点原图哈希去重；输入图像数包含打点图和未来选项，按内容哈希去重。",
                      "动作长度统计公开命令；查询前缀不等同于最少推理步数。",
                      "距离展示固定按米分档；m/cm 是表达单位，不改变物理量。",
                      "构建报告是该 release 的历史记录；供给各格可能共享观测，禁止直接求和。",
                      "图像统计及审阅记录不证明全部题目可回答；没有运行无图模型或人类作答基线。"],
            "releases": releases}
