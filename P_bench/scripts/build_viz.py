"""Static HTML review page from a collection dir (records.jsonl + debug/ images).

    python scripts/build_viz.py data/conseq/smoke_p2
    python -m http.server 8766 --directory data/conseq/smoke_p2   # open review.html

Self-contained (relative image paths, inline CSS/JS filtering). No server needed
to render; a static file server is only needed so the browser can load images.
"""

import argparse
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import record


VERDICT_COLOR = {"keep_agree": "#2e7d32", "keep_depth": "#2e7d32", "keep_visible": "#2e7d32",
                 "review_depth_hole": "#f9a825",
                 "discard_noise": "#c62828"}


def _fmt_act(a):
    if a["type"] == "turn":
        d = a["deg"]
        return ("L" if d < 0 else "R") + f"{abs(d):g}"
    return "F" + f"{a['m']:g}"


def _outcome_row(oc, out_dir):
    acts = " ".join(_fmt_act(a) for a in oc["actions"])
    c = oc["contact"]
    hit = f"{c['category']} ({c['vote_fraction']:.0%})" if c and not c["unattributed"] else \
          ("unattributed" if c else "—")
    ve = oc["view_exit"]
    nv = oc.get("nav_check")
    verdict = nv["verdict"] if nv else "n/a"
    color = VERDICT_COLOR.get(verdict, "#666")
    img = os.path.join("debug", f"ev_{oc['outcome_id']}.png")
    has_img = os.path.exists(os.path.join(out_dir, img))

    collided = "✓" if oc["collided"] else "✗"
    exit_txt = "✓" if not ve["end_in_fov"] else "✗"
    hc = int(bool(oc.get("human_check")))
    data_attrs = (f'data-collided="{int(oc["collided"])}" data-verdict="{verdict}" '
                  f'data-hc="{hc}"')
    body = f"""
    <tr class="oc" {data_attrs}>
      <td class="mono">{html.escape(acts)}</td>
      <td>{collided} {f'@{oc["first_contact_arc_m"]:.2f}m' if oc['collided'] else ''}</td>
      <td>{html.escape(hit)}</td>
      <td>{oc['pose_end_full']['heading_deg']:+.0f}°</td>
      <td>exit:{exit_txt}</td>
      <td><span class="badge" style="background:{color}">{verdict}</span></td>
      <td>{'<a href="'+img+'" target="_blank">evidence</a>' if has_img else ''}</td>
    </tr>"""
    if has_img:
        body += f'<tr class="ev"><td colspan="7"><img src="{img}" loading="lazy"></td></tr>'
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    recs = list(record.read_records(os.path.join(args.out_dir, "records.jsonl")))
    scenes = sorted({r["scene_id"] for r in recs})

    cards = []
    for r in recs:
        inv = ", ".join(f"{k}×{v}" for k, v in sorted(r["category_inventory"].items()))
        overlay = os.path.join("debug", f"overlay_{r['frame_id']}.png")
        overlay = overlay if os.path.exists(os.path.join(args.out_dir, overlay)) else r["image_path"]
        rows = "".join(_outcome_row(oc, args.out_dir) for oc in r["outcomes"])
        cards.append(f"""
        <div class="frame" data-scene="{r['scene_id']}">
          <h3>{r['frame_id']} <small>({r['n_nonstructural']} objects)</small></h3>
          <div class="cols">
            <img class="rgb" src="{r['image_path']}">
            <img class="rgb" src="{overlay}">
          </div>
          <div class="inv">{html.escape(inv)}</div>
          <table><tr><th>actions</th><th>collide</th><th>hit</th><th>head</th>
            <th>view</th><th>verdict</th><th></th></tr>{rows}</table>
        </div>""")

    scene_opts = "".join(f'<option value="{s}">{s}</option>' for s in scenes)
    page = f"""<!doctype html><meta charset="utf-8"><title>EgoConseq review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:12px;background:#fafafa}}
.frame{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;margin:10px 0}}
.cols{{display:flex;gap:8px}} .rgb{{width:48%;border:1px solid #ccc}}
.inv{{font-size:12px;color:#555;margin:6px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border-bottom:1px solid #eee;padding:3px 6px;text-align:left}}
.mono{{font-family:monospace}} .badge{{color:#fff;padding:1px 6px;border-radius:3px;font-size:11px}}
.ev img{{max-width:100%;border:1px solid #ccc;margin:4px 0}}
#bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ccc;z-index:9}}
</style>
<div id="bar">
  scene <select id="scene"><option value="">all</option>{scene_opts}</select>
  <label><input type="checkbox" id="collonly">collided only</label>
  <label><input type="checkbox" id="hconly">human-check only</label>
  <span id="count"></span>
</div>
{''.join(cards)}
<script>
function apply(){{
  const s=scene.value, co=collonly.checked, hc=hconly.checked; let n=0;
  document.querySelectorAll('.frame').forEach(f=>{{
    const sok = !s || f.dataset.scene===s;
    f.style.display = sok ? '' : 'none';
    f.querySelectorAll('tr.oc').forEach(r=>{{
      const ok = sok && (!co || r.dataset.collided==='1') && (!hc || r.dataset.hc==='1');
      r.style.display = ok?'':'none';
      if(r.nextElementSibling && r.nextElementSibling.classList.contains('ev'))
        r.nextElementSibling.style.display = ok?'':'none';
      if(ok) n++;
    }});
  }});
  count.textContent = n+' outcomes';
}}
scene.onchange=apply; collonly.onchange=apply; hconly.onchange=apply; apply();
</script>"""
    out = os.path.join(args.out_dir, "review.html")
    open(out, "w").write(page)
    print(f"wrote {out} ({len(recs)} frames, {sum(len(r['outcomes']) for r in recs)} outcomes)")


if __name__ == "__main__":
    main()
