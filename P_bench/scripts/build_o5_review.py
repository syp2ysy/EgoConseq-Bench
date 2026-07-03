"""Build a lightweight human-check review panel for O5 cases.

Reads data/demo/o5/manifest.jsonl and writes data/demo/o5/review.html.
The HTML references image files by relative path instead of embedding base64,
so the page stays small and browser-friendly.

Usage:
    python scripts/build_o5_review.py [--manifest path] [--out path]
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq.manifest import Case, read_jsonl


LABEL_COLORS = {
    "contact": "#ffd6d6",
    "no_contact": "#d8f5dd",
    "INVALID": "#eeeeee",
}

GROUP_SORT_KEY = {
    "flip": 0,
    "all_no_contact": 1,
    "all_contact": 2,
}


def format_float(v, decimals: int = 3) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def gt_display(label: str) -> str:
    if label == "contact":
        return "会接触"
    if label == "no_contact":
        return "不会接触"
    return label


def gt_reason(d_safe_m: object, horizon_m: object, label: str) -> str:
    d = format_float(d_safe_m)
    h = format_float(horizon_m)
    if label == "contact":
        return f"因为 d_safe={d} < H={h}"
    if label == "no_contact":
        return f"因为 d_safe={d} >= H={h}"
    return f"d_safe={d}, H={h}"


def relative_image_path(path: str | None, html_dir: Path) -> str:
    """Return a browser-friendly path relative to the HTML file."""
    if not path:
        return ""
    try:
        return os.path.relpath(path, start=html_dir)
    except ValueError:
        return path


def topdown_path_for_case(case: Case) -> str | None:
    if not case.image_path:
        return None
    image_path = Path(case.image_path)
    return str(image_path.parent.parent / "topdown" / image_path.name)


def case_sort_key(case: Case) -> tuple:
    group_kind = case.tags.get("group_kind", "") if case.tags else ""
    radius = case.body.get("radius_m", 0.0) if case.body else 0.0
    return (
        GROUP_SORT_KEY.get(group_kind, 99),
        case.group_id or "",
        float(radius or 0.0),
        case.case_id,
    )


def build_html(cases: list[Case], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_dir = out.parent

    sorted_cases = sorted(cases, key=case_sort_key)
    label_counts = Counter(c.answer.get("label", "INVALID") if c.answer else "INVALID" for c in cases)
    group_ids = {c.group_id or c.case_id for c in cases}
    group_kind_by_id = {}
    for c in cases:
        gid = c.group_id or c.case_id
        group_kind_by_id.setdefault(gid, c.tags.get("group_kind", "unknown") if c.tags else "unknown")
    group_kind_counts = Counter(group_kind_by_id.values())

    rows: list[str] = []
    prev_group_id = None
    for idx, case in enumerate(sorted_cases, start=1):
        group_id = case.group_id or case.case_id
        group_kind = case.tags.get("group_kind", "unknown") if case.tags else "unknown"
        geometry_tag = case.tags.get("geometry_tag", "") if case.tags else ""
        label = case.answer.get("label", "INVALID") if case.answer else "INVALID"
        answer_type = case.answer.get("answer_type", "") if case.answer else ""
        bg = LABEL_COLORS.get(label, "#ffffff")
        radius_m = case.body.get("radius_m") if case.body else None
        diameter_m = case.body.get("diameter_m") if case.body else None
        horizon_m = case.action.get("horizon_m") if case.action else None
        evidence = case.tags.get("gt_evidence", {}) if case.tags else {}
        rgb_src = relative_image_path(case.image_path, html_dir)
        topdown_src = relative_image_path(topdown_path_for_case(case), html_dir)
        reason = gt_reason(evidence.get("d_safe_m", case.d_safe_visible_m), horizon_m, label)

        if group_id != prev_group_id:
            rows.append(
                '<tr class="group-row">'
                f'<td colspan="11">group {html.escape(group_id)}'
                f' | kind={html.escape(group_kind)}</td></tr>'
            )
            prev_group_id = group_id

        rows.append(
            '<tr style="background:{bg}">'.format(bg=bg)
            + f"<td>{idx}</td>"
            + f'<td><img class="rgb" src="{html.escape(rgb_src)}" alt="RGB"></td>'
            + f"<td><div class=\"q\">{html.escape(case.question or '')}</div></td>"
            + (
                "<td>"
                f"<b>GT: {html.escape(gt_display(label))}</b><br>"
                f"<span>{html.escape(label)} / {html.escape(answer_type)}</span>"
                "</td>"
            )
            + f"<td>{format_float(radius_m)}<br><span>diam {format_float(diameter_m)}</span></td>"
            + f"<td>{format_float(case.d_safe_visible_m)}</td>"
            + f"<td>{format_float(case.d_safe_navmesh_m)}</td>"
            + f"<td>{format_float(horizon_m)}</td>"
            + (
                "<td>"
                f"{html.escape(reason)}<br>"
                "规则: 会接触 iff d_safe &lt; H"
                "</td>"
            )
            + f'<td><img class="topdown" src="{html.escape(topdown_src)}" alt="topdown"></td>'
            + (
                "<td>"
                f"<code>{html.escape(case.case_id)}</code><br>"
                f"{html.escape(case.scene_id or '')}<br>"
                f"<span>{html.escape(geometry_tag)}</span>"
                "</td>"
            )
            + "</tr>"
        )

    summary_bits = [
        f"<b>Total cases:</b> {len(cases)}",
        f"<b>Groups:</b> {len(group_ids)}",
    ]
    for label, count in sorted(label_counts.items()):
        summary_bits.append(f"<b>{html.escape(label)}:</b> {count}")
    for kind, count in sorted(group_kind_counts.items()):
        summary_bits.append(f"<b>{html.escape(kind)} groups:</b> {count}")

    text = f"""\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>EgoConseq-Bench O5 Review</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; background: #f3f3f3; margin: 0; padding: 12px; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  .summary {{ background: white; border: 1px solid #ccc; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; }}
  .summary span {{ margin-right: 16px; display: inline-block; }}
  table {{ border-collapse: collapse; width: 100%; background: white; }}
  th {{ background: #222; color: white; text-align: left; padding: 7px; position: sticky; top: 0; }}
  td {{ padding: 6px 7px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  img.rgb {{ width: 220px; max-height: 170px; object-fit: contain; border: 1px solid #bbb; background: #fafafa; }}
  img.topdown {{ width: 200px; max-height: 200px; object-fit: contain; border: 1px solid #bbb; background: #fafafa; }}
  .q {{ width: 280px; white-space: pre-wrap; line-height: 1.35; }}
  .group-row td {{ background: #e7eaf0; color: #333; font-weight: bold; font-size: 12px; }}
  code {{ font-size: 11px; }}
  span {{ color: #666; font-size: 11px; }}
</style>
</head>
<body>
<h1>EgoConseq-Bench O5 Human-Check Review</h1>
<div class="summary">{''.join(f'<span>{part}</span>' for part in summary_bits)}
<br><br>每行是一道单机器人二值题；同一个 group 内只有底盘直径不同。GT rule: contact iff d_safe &lt; H.</div>
<table>
<tr>
  <th>#</th><th>RGB</th><th>Question</th><th>GT</th><th>Body</th>
  <th>d_safe depth</th><th>d_safe nav</th><th>H</th><th>GT evidence</th>
  <th>Top-down</th><th>Case</th>
</tr>
{os.linesep.join(rows)}
</table>
</body>
</html>
"""
    out.write_text(text, encoding="utf-8")
    print(f"[build_o5_review] Review panel written: {out_path}  ({len(cases)} cases)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None,
                   help="Path to manifest.jsonl (default: data/demo/o5/manifest.jsonl)")
    p.add_argument("--out", default=None,
                   help="Output path for review.html (default: data/demo/o5/review.html)")
    return p.parse_args()


def main():
    args = parse_args()

    manifest_path = args.manifest or os.path.join(_REPO_ROOT, "data", "demo", "o5", "manifest.jsonl")
    out_path = args.out or os.path.join(_REPO_ROOT, "data", "demo", "o5", "review.html")

    if not os.path.isfile(manifest_path):
        print(f"[build_o5_review] ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    cases = read_jsonl(manifest_path)
    print(f"[build_o5_review] Loaded {len(cases)} cases from {manifest_path}")
    build_html(cases, out_path)


if __name__ == "__main__":
    main()
