"""Build human-check review panel for O5 cases.

Reads data/demo/o5/manifest.jsonl and writes data/demo/o5/review.html:
  - One row per case
  - Columns: RGB image | Question | GT label | d_safe_small / d_safe_large / H | Top-down | geometry_tag
  - small_only cases shown FIRST
  - Color-coded by label

Usage:
    python scripts/build_o5_review.py [--manifest path] [--out path]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from egoconseq.manifest import read_jsonl, Case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_data_uri(img_path: str | None) -> str:
    """Convert a local image file to a base64 data URI for inline HTML."""
    if img_path is None or not os.path.isfile(img_path):
        return "data:image/png;base64,"  # empty

    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    # Determine extension
    ext = Path(img_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png"}.get(ext, "image/png")
    return f"data:{mime};base64,{b64}"


LABEL_COLORS = {
    "small_only": "#c8f5c8",   # light green
    "both":       "#c8e8f5",   # light blue
    "neither":    "#f5e8c8",   # light amber
    "INVALID":    "#f5c8c8",   # light red
}

LABEL_SORT_KEY = {
    "small_only": 0,
    "both":       1,
    "neither":    2,
    "INVALID":    3,
}


def format_float(v, decimals: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_HEADER = """\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>EgoConseq-Bench O5 Human-Check Review</title>
<style>
  body { font-family: Arial, sans-serif; font-size: 13px; background: #f0f0f0; margin: 0; padding: 10px; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  .summary { background: #fff; padding: 8px 12px; border-radius: 6px; margin-bottom: 12px;
             border: 1px solid #ccc; font-size: 13px; }
  table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
          overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }
  th { background: #333; color: white; padding: 8px 10px; font-size: 12px; text-align: left; }
  td { padding: 6px 8px; vertical-align: top; border-bottom: 1px solid #eee; font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  img.rgb { width: 200px; height: auto; border-radius: 4px; border: 1px solid #ccc; }
  img.topdown { width: 180px; height: auto; border-radius: 4px; border: 1px solid #ccc; }
  .label { font-weight: bold; font-size: 13px; padding: 3px 7px; border-radius: 4px;
           display: inline-block; }
  .q { font-size: 11px; color: #333; max-width: 260px; white-space: pre-wrap; }
  .meta { font-size: 11px; color: #555; }
  .case-id { font-family: monospace; font-size: 10px; color: #888; }
  .geo-tag { background: #eee; border-radius: 3px; padding: 2px 5px; font-size: 10px; }
  .separator { background: #dde; font-size: 11px; color: #556; font-style: italic; }
</style>
</head>
<body>
<h1>EgoConseq-Bench O5 — Human-Check Review Panel</h1>
"""


def build_html(cases: list[Case], out_path: str) -> None:
    """Build the review HTML file."""
    # Sort: small_only first, then both, then neither, then INVALID
    def sort_key(c):
        lbl = c.answer.get("label", "INVALID") if c.answer else "INVALID"
        return (LABEL_SORT_KEY.get(lbl, 99), c.case_id)

    sorted_cases = sorted(cases, key=sort_key)

    # Count by label
    label_counts: dict[str, int] = {}
    for c in cases:
        lbl = c.answer.get("label", "INVALID") if c.answer else "INVALID"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    total = len(cases)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HTML_HEADER)

        # Summary box
        f.write('<div class="summary">\n')
        f.write(f"<b>Total cases:</b> {total} &nbsp;&nbsp; ")
        for lbl, color in LABEL_COLORS.items():
            cnt = label_counts.get(lbl, 0)
            if cnt == 0:
                continue
            pct = cnt / total * 100 if total > 0 else 0.0
            f.write(
                f'<span class="label" style="background:{color}">{lbl}</span>'
                f' {cnt} ({pct:.1f}%) &nbsp;&nbsp; '
            )
        f.write(
            "<br><br>"
            "<b>Instructions:</b> For each row verify that:<br>"
            "  (1) The top-down plot shows the correct gap (small disk ≤ H, large disk crosses H or vice-versa).<br>"
            "  (2) The RGB image shows the relevant corridor / obstacle.<br>"
            "  (3) The GT label is consistent with d_safe_small / d_safe_large / H values.<br>"
            "  Mark any suspect cases with the case_id for the QA log."
        )
        f.write("</div>\n")

        f.write('<table>\n')
        f.write(
            "<tr><th>#</th><th>RGB Image</th><th>Question</th>"
            "<th>GT Label</th>"
            "<th>d_safe_small (m)</th><th>d_safe_large (m)</th><th>H (m)</th>"
            "<th>Top-Down Evidence</th><th>Geometry Tag</th><th>Case ID / Scene</th></tr>\n"
        )

        prev_label_group = None
        row_num = 0
        for case in sorted_cases:
            row_num += 1
            lbl = case.answer.get("label", "INVALID") if case.answer else "INVALID"
            bg = LABEL_COLORS.get(lbl, "#fff")

            # Group separator
            if lbl != prev_label_group:
                f.write(
                    f'<tr class="separator"><td colspan="10">'
                    f'&nbsp;↓ &nbsp;<b>{lbl}</b> cases ({label_counts.get(lbl, 0)} total)'
                    f'</td></tr>\n'
                )
                prev_label_group = lbl

            # Geometry values
            d_small = None
            d_large = None
            horizon = None
            if case.action:
                horizon = case.action.get("horizon_m")
            # We need to recover d_safe per body from the case body + action
            # They were stored in the generation but not as direct Case fields.
            # Read from tags if present, else use d_safe_visible_m heuristic.
            # The generation stored d_safe_visible_m = d_vis_small in the case.
            d_small = case.d_safe_visible_m  # small body d_safe
            # For large body: stored in tags or we can parse from the answer
            # We stored geometry_tag in tags; check if extras stored
            geo_tag = case.tags.get("geometry_tag", "") if case.tags else ""

            # Recover d_safe_large from the label + horizon + d_small
            # This is not stored directly; we need to reconstruct from the known
            # label relationship. However for the review panel, the topdown plot
            # shows both visually. Mark as "see topdown".
            d_large_str = "see top-down"

            # Image URIs (inline base64)
            rgb_uri = img_to_data_uri(case.image_path)
            # Topdown path: same directory, same case_id but in topdown/
            td_path = None
            if case.image_path:
                img_dir = os.path.dirname(case.image_path)
                td_dir = os.path.join(os.path.dirname(img_dir), "topdown")
                case_base = os.path.basename(case.image_path)
                td_path = os.path.join(td_dir, case_base)
            td_uri = img_to_data_uri(td_path)

            # Scene
            scene = case.scene_id or ""

            # Pose
            pose_str = ""
            if case.pose:
                pose_str = f"x={case.pose[0]:.2f} z={case.pose[2]:.2f} yaw={math.degrees(case.pose[3]):.0f}°"

            f.write(f'<tr style="background:{bg}">\n')
            f.write(f'  <td>{row_num}</td>\n')
            f.write(
                f'  <td><img class="rgb" src="{rgb_uri}" alt="RGB" '
                f'onerror="this.alt=\'[no image]\'"></td>\n'
            )
            q_text = (case.question or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            f.write(f'  <td><div class="q">{q_text}</div></td>\n')
            f.write(
                f'  <td><span class="label" style="background:{bg}">{lbl}</span></td>\n'
            )
            f.write(f'  <td class="meta">{format_float(d_small)}</td>\n')
            f.write(f'  <td class="meta">{d_large_str}</td>\n')
            f.write(f'  <td class="meta">{format_float(horizon)}</td>\n')
            f.write(
                f'  <td><img class="topdown" src="{td_uri}" alt="top-down" '
                f'onerror="this.alt=\'[no topdown]\'"></td>\n'
            )
            f.write(
                f'  <td><span class="geo-tag">{geo_tag}</span></td>\n'
            )
            f.write(
                f'  <td class="case-id">{case.case_id}<br>'
                f'<span style="color:#aaa">{scene}</span><br>'
                f'<span style="color:#aaa">{pose_str}</span>'
                f'</td>\n'
            )
            f.write('</tr>\n')

        f.write('</table>\n')
        f.write('<p style="font-size:11px;color:#888;margin-top:10px;">'
                'Generated by build_o5_review.py — EgoConseq-Bench Task 18</p>\n')
        f.write('</body>\n</html>\n')

    print(f"[build_o5_review] Review panel written: {out_path}  ({total} cases)")


# ---------------------------------------------------------------------------
# Need math import for degrees conversion
# ---------------------------------------------------------------------------
import math


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
