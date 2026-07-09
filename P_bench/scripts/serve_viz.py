"""Interactive dataset review + live explore (stdlib http.server; Habitat single-threaded).

    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
        python scripts/serve_viz.py --port 8767 --dataset data/conseq/v2

Dataset mode: browse scene -> frame -> outcome (with filters). Each outcome shows an
obstacle-accurate top-down (rebuilt from the frame pose), a body-circle step-through
along the path, and a step-by-step reasoning trace. Live mode: sample a frame and
judge an ad-hoc action sequence through the SAME detail view.

A single SimSession is held at a time (Habitat is not thread-safe); rebuilt frames'
obstacle points are cached by frame id.
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import viz, perception, config
from pipeline import actions as A
from pipeline import record as REC
from pipeline.actions import parse_actions
from pipeline.body import Cylinder
from pipeline.consequence import judge
from pipeline.frame import build_frame
from pipeline.sim import SimSession
from pipeline.scene_pool import discover_semantic_scenes


def _downsample(a, n=3000):
    if a.shape[0] > n:
        a = a[np.linspace(0, a.shape[0] - 1, n).astype(int)]
    return a


def _obstacles_from_frame(fr):
    om = perception.obstacle_mask(fr.pts, fr.floor_y)
    ob = _downsample(fr.pts[om][:, [0, 2]])
    return np.asarray(ob, float).round(3).tolist()


def _objects_payload(objects):
    return [{"xz": [float(o["ground_xy_centroid"][0]), float(o["ground_xy_centroid"][1])],
             "cat": o["category"], "structural": bool(o["is_structural"]),
             "px": [float(o["centroid_px"][0]), float(o["centroid_px"][1])]}
            for o in objects]


def geometry_payload(objects, oc, obstacles_xz, rgb_url):
    """Assemble the render JSON consumed by the browser (dataset + live share it)."""
    acts = parse_actions(oc["actions"])
    path = [[float(x), float(z), float(np.degrees(h)), float(arc)]
            for (x, z, h, arc) in A.sample_path(acts, config.MARCH_STEP_M)]
    id2s = {o["instance_id"]: bool(o["is_structural"]) for o in objects}
    rels = [dict(r, structural=id2s.get(r["instance_id"], False))
            for r in oc["object_relations"]]
    c = oc.get("contact")
    return {
        "rgb": rgb_url,
        "obstacles_xz": obstacles_xz,
        "objects": _objects_payload(objects),
        "path": path,
        "cone_half_deg": config.FOV_HALF_DEG,
        "body_radius_m": float(oc["body"]["radius_m"]),
        "total_forward_m": float(oc["total_forward_m"]),
        "collided": bool(oc["collided"]),
        "first_contact_arc_m": oc["first_contact_arc_m"],
        "contact_action_index": oc.get("contact_action_index"),
        "contact_xz": (c["xy"] if c else None),
        "contact_px": (c.get("pixel") if c else None),
        "contact_id": (c["instance_id"] if c and not c.get("unattributed") else None),
        "pose_exec": oc["pose_end_exec"], "pose_full": oc["pose_end_full"],
        "objects_rel": rels,
        "reasoning": viz.reasoning_trace(oc, objects),
        "nav_check": oc.get("nav_check"),
        "actions": oc["actions"],
    }


class State:
    def __init__(self, img_dir, dataset=None):
        self.scenes = discover_semantic_scenes()
        self.by_id = {os.path.basename(os.path.dirname(s)): s for s in self.scenes}
        self.sim = None
        self.scene_id = None
        self.frames = {}          # live sampled frames
        self.img_dir = img_dir
        os.makedirs(img_dir, exist_ok=True)
        # dataset
        self.ds_dir = dataset
        self.by_frame = {}
        self.by_scene = {}
        self.obst = {}            # frame_id -> obstacle xz (cached rebuild)
        if dataset:
            for r in REC.read_records(os.path.join(dataset, "records.jsonl")):
                self.by_frame[r["frame_id"]] = r
                self.by_scene.setdefault(r["scene_id"], []).append(r["frame_id"])

    # ---- shared Habitat ----
    def open_scene(self, scene_id):
        if self.scene_id == scene_id and self.sim is not None:
            return
        if self.sim is not None:
            self.sim.close()
        self.sim = SimSession(self.by_id[scene_id])
        self.scene_id = scene_id
        self.frames.clear()

    # ---- dataset browse (pure records) ----
    def ds_scenes(self):
        return [{"scene": s, "n_frames": len(f)} for s, f in sorted(self.by_scene.items())]

    def ds_frames(self, scene):
        out = []
        for fid in self.by_scene.get(scene, []):
            r = self.by_frame[fid]
            out.append({"frame_id": fid, "rgb": "/dsimg/" + r["image_path"],
                        "n_nonstructural": r["n_nonstructural"],
                        "inventory": r["category_inventory"]})
        return out

    def ds_outcomes(self, frame_id):
        r = self.by_frame.get(frame_id)
        if r is None:
            return None
        out = []
        for oc in r["outcomes"]:
            c = oc.get("contact")
            out.append({"outcome_id": oc["outcome_id"], "seq_len": oc["seq_len"],
                        "radius_m": oc["body"]["radius_m"], "actions": oc["actions"],
                        "collided": oc["collided"], "arc": oc["first_contact_arc_m"],
                        "act_i": oc.get("contact_action_index"),
                        "hit": (c["category"] if c and not c.get("unattributed") else None),
                        "verdict": (oc["nav_check"]["verdict"] if oc.get("nav_check") else None),
                        "human_check": bool(oc.get("human_check"))})
        return out

    def _obstacles(self, r):
        fid = r["frame_id"]
        if fid in self.obst:
            return self.obst[fid]
        self.open_scene(r["scene_id"])
        pos = np.asarray(r["pose"]["position"], float)
        fr = build_frame(self.sim, pos, float(r["pose"]["yaw_rad"]),
                         frame_id=fid, scene_id=r["scene_id"], scene_glb=self.by_id[r["scene_id"]])
        self.obst[fid] = _obstacles_from_frame(fr)
        return self.obst[fid]

    def ds_render(self, frame_id, outcome_id):
        r = self.by_frame.get(frame_id)
        if r is None:
            return None
        oc = next((o for o in r["outcomes"] if o["outcome_id"] == outcome_id), None)
        if oc is None:
            return None
        obstacles = self._obstacles(r)
        return geometry_payload(r["objects"], oc, obstacles, "/dsimg/" + r["image_path"])

    # ---- live explore ----
    def sample(self, scene_id, seed):
        self.open_scene(scene_id)
        self.sim.recompute_navmesh(0.25)
        pose = self.sim.sample_random_pose(np.random.default_rng(seed), [0.25])
        if pose is None:
            return None
        pos, yaw = pose
        fid = f"{scene_id}-s{seed}"
        fr = build_frame(self.sim, pos, yaw, frame_id=fid, scene_id=scene_id,
                         scene_glb=self.by_id[scene_id])
        self.frames[fid] = fr
        Image.fromarray(fr.rgb).save(os.path.join(self.img_dir, fid + "_rgb.png"))
        return {"frame_id": fid, "rgb": f"/img/{fid}_rgb.png",
                "inventory": fr.category_inventory}

    def live_render(self, frame_id, radius, actions):
        fr = self.frames.get(frame_id)
        if fr is None:
            return None
        self.sim.recompute_navmesh(radius)
        acts = parse_actions(actions)
        nav = self.sim.nav(fr.position, fr.yaw_rad)
        oc = judge(fr, Cylinder(radius_m=radius), acts, nav=nav, geodesic=False)
        oc["outcome_id"] = f"{frame_id}-live"
        obstacles = _obstacles_from_frame(fr)
        return geometry_payload(fr.objects, oc, obstacles, f"/img/{frame_id}_rgb.png")


def _safe_join(root, rel):
    p = os.path.normpath(os.path.join(root, rel.lstrip("/")))
    return p if os.path.abspath(p).startswith(os.path.abspath(root)) else None


def make_handler(state, page):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=float).encode())

        def _file(self, fp, ctype):
            if fp and os.path.exists(fp):
                return self._send(200, open(fp, "rb").read(), ctype)
            return self._send(404, b"")

        def do_GET(self):
            u = urlparse(self.path)
            p, q = u.path, parse_qs(u.query)
            if p in ("/", "/index.html"):
                return self._send(200, page.encode(), "text/html")
            if p == "/api/ds/scenes":
                return self._json(state.ds_scenes())
            if p == "/api/ds/frames":
                return self._json(state.ds_frames(q.get("scene", [""])[0]))
            if p == "/api/ds/outcomes":
                r = state.ds_outcomes(q.get("frame", [""])[0])
                return self._json(r) if r is not None else self._send(404, b"")
            if p == "/api/scenes":
                return self._json(sorted(state.by_id))
            if p.startswith("/dsimg/") and state.ds_dir:
                return self._file(_safe_join(state.ds_dir, p[len("/dsimg/"):]), "image/png")
            if p.startswith("/img/"):
                return self._file(os.path.join(state.img_dir, os.path.basename(p)), "image/png")
            return self._send(404, b"")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/api/ds/render":
                    r = state.ds_render(req["frame_id"], req["outcome_id"])
                elif self.path == "/api/frame":
                    r = state.sample(req["scene"], int(req.get("seed", 0)))
                elif self.path == "/api/judge":
                    r = state.live_render(req["frame_id"], float(req["radius"]), req["actions"])
                else:
                    return self._send(404, b"")
                return self._json(r) if r is not None else self._json({"error": "failed"}, 400)
            except Exception as e:
                return self._json({"error": repr(e)}, 500)
    return H


PAGE = r"""<!doctype html><meta charset="utf-8"><title>EgoConseq review</title>
<style>
:root{--bd:#dcdcdc;--mut:#666;--hl:#e8f0fe}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;background:#fafafa;color:#222;font-size:13px}
#top{display:flex;gap:10px;align-items:center;padding:8px 12px;background:#fff;border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:5}
#top b{font-size:15px}
.tab{cursor:pointer;padding:3px 10px;border:1px solid var(--bd);border-radius:14px}
.tab.on{background:#1a73e8;color:#fff;border-color:#1a73e8}
#wrap{display:grid;grid-template-columns:180px 1fr 1fr;gap:10px;padding:10px;height:calc(100vh - 47px)}
.col{overflow:auto;background:#fff;border:1px solid var(--bd);border-radius:8px;padding:8px}
.fr{display:flex;gap:6px;align-items:center;padding:4px;border-radius:6px;cursor:pointer}
.fr:hover{background:var(--hl)} .fr.on{background:#d2e3fc}
.fr img{width:64px;height:48px;object-fit:cover;border:1px solid var(--bd);border-radius:3px}
.fr small{color:var(--mut)}
#rgbwrap{position:relative;width:100%} #rgb{width:100%;display:block;border:1px solid var(--bd);border-radius:4px}
#rgbov{position:absolute;left:0;top:0;width:100%;height:100%}
.flt{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0;align-items:center}
.flt select,.flt label{font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{border-bottom:1px solid #eee;padding:2px 4px;text-align:left}
.oc{cursor:pointer} .oc:hover{background:var(--hl)} .oc.on{background:#d2e3fc}
.mono{font-family:ui-monospace,monospace}
.badge{color:#fff;padding:0 5px;border-radius:3px;font-size:10px}
#svg{width:100%;height:auto;border:1px solid var(--bd);border-radius:6px;background:#fcfcfc}
#ctrl{display:flex;gap:8px;align-items:center;margin:6px 0}
#ctrl input[type=range]{flex:1}
.rz{background:#fff}
#reason{white-space:pre-wrap;font-size:11px;line-height:1.4;margin-top:8px}
#reason .ln{padding:0 4px;border-radius:3px}
#reason .hit{background:#fde7e7} #reason .now{background:#fff3cd}
#relWrap{max-height:260px;overflow:auto;margin-top:6px}
#relWrap th{position:sticky;top:0;background:#fff}
tr.struct td{color:#aaa} tr.hitrow td{background:#fde7e7}
.k{color:var(--mut)}
.hidden{display:none}
#live{padding:8px;background:#fff;border:1px solid var(--bd);border-radius:8px;margin-bottom:6px}
.act{font-family:ui-monospace,monospace;margin:2px 0}
</style>

<div id="top">
  <b>EgoConseq</b>
  <span class="tab on" id="tabDs" onclick="setMode('ds')">Dataset</span>
  <span class="tab" id="tabLive" onclick="setMode('live')">Live</span>
  <span id="dssel"></span>
  <span id="status" class="k"></span>
</div>

<div id="wrap">
  <div class="col" id="cFrames"></div>
  <div class="col" id="cMid">
    <div id="live" class="hidden">
      <b>Live</b> scene <select id="lscene"></select> seed <input id="lseed" type="number" value="0" style="width:56px">
      <button onclick="liveSample()">sample</button> <span id="lfid" class="k"></span>
      <div id="lacts"></div>
      r <input id="lrad" type="number" value="0.20" step="0.05" style="width:60px">
      <button onclick="addAct('turn')">+turn</button>
      <button onclick="addAct('forward')">+fwd</button>
      <button onclick="liveRun()"><b>Run</b></button>
    </div>
    <div id="rgbwrap"><img id="rgb"><svg id="rgbov" viewBox="0 0 640 480" preserveAspectRatio="none"></svg></div>
    <div id="invLine" class="k" style="margin:4px 0"></div>
    <div class="flt" id="filters">
      len <select id="fLen"></select>
      r <select id="fRad"></select>
      <label><input type="checkbox" id="fColl">collided</label>
      verdict <select id="fVer"></select>
      <label><input type="checkbox" id="fHc">human-check</label>
      <span id="ocCount" class="k"></span>
    </div>
    <table id="ocTbl"><tbody id="ocBody"></tbody></table>
  </div>
  <div class="col" id="cDetail">
    <svg id="svg" viewBox="0 0 400 400"></svg>
    <div id="ctrl" class="hidden">
      <button id="play" onclick="togglePlay()">▶</button>
      <input id="arc" type="range" min="0" max="1" step="0.01" value="1" oninput="onScrub()">
      <span id="arcLbl" class="k"></span>
    </div>
    <div id="reason"></div>
    <div id="relWrap"><table id="relTbl"></table></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const VCOL={keep_agree:'#2e7d32',keep_depth:'#2e7d32',keep_visible:'#2e7d32',
            review_depth_hole:'#f9a825',discard_noise:'#c62828'};
let MODE='ds', SCENE=null, FRAME=null, OCS=[], DATA=null, playing=null, HASDS=false;

// ---------- boot ----------
fetch('/api/ds/scenes').then(r=>r.json()).then(s=>{
  HASDS = s && s.length>0;
  if(HASDS){
    $('dssel').innerHTML='scene <select id="scene"></select>';
    $('scene').innerHTML=s.map(x=>`<option value="${x.scene}">${x.scene} (${x.n_frames})</option>`).join('');
    $('scene').onchange=()=>selScene($('scene').value);
    selScene(s[0].scene);
  } else { setMode('live'); }
});
fetch('/api/scenes').then(r=>r.json()).then(s=>{
  if($('lscene')) $('lscene').innerHTML=s.map(x=>`<option>${x}</option>`).join('');
});

function setMode(m){
  MODE=m;
  $('tabDs').classList.toggle('on',m==='ds'); $('tabLive').classList.toggle('on',m==='live');
  $('live').classList.toggle('hidden',m!=='live');
  $('filters').classList.toggle('hidden',m!=='ds');
  $('cFrames').classList.toggle('hidden',m!=='ds');
  if(m==='live' && $('lacts').children.length===0){addAct('turn');addAct('forward');}
}

// ---------- dataset: frames ----------
function selScene(s){
  SCENE=s;
  fetch('/api/ds/frames?scene='+encodeURIComponent(s)).then(r=>r.json()).then(fs=>{
    $('cFrames').innerHTML=fs.map(f=>
      `<div class="fr" data-f="${f.frame_id}" onclick="selFrame('${f.frame_id}','${f.rgb}')">
         <img src="${f.rgb}" loading="lazy">
         <div><div class="mono" style="font-size:10px">${f.frame_id.slice(-14)}</div>
         <small>${f.n_nonstructural} obj</small></div></div>`).join('');
    if(fs.length) selFrame(fs[0].frame_id, fs[0].rgb);
  });
}
function selFrame(fid,rgb){
  FRAME=fid;
  document.querySelectorAll('#cFrames .fr').forEach(e=>e.classList.toggle('on',e.dataset.f===fid));
  $('rgb').src=rgb; $('rgbov').innerHTML='';
  fetch('/api/ds/outcomes?frame='+encodeURIComponent(fid)).then(r=>r.json()).then(o=>{
    OCS=o; buildFilters(o); renderOcs();
  });
}
function buildFilters(o){
  const lens=[...new Set(o.map(x=>x.seq_len))].sort();
  const rads=[...new Set(o.map(x=>x.radius_m))].sort();
  const vers=[...new Set(o.map(x=>x.verdict))].filter(Boolean).sort();
  const opt=(a)=>['<option value="">all</option>'].concat(a.map(v=>`<option>${v}</option>`)).join('');
  if(!$('fLen').dataset.built){
    $('fLen').innerHTML=opt(lens); $('fRad').innerHTML=opt(rads); $('fVer').innerHTML=opt(vers);
    ['fLen','fRad','fColl','fVer','fHc'].forEach(id=>$(id).onchange=renderOcs);
    $('fLen').dataset.built='1';
  }
}
function actStr(acts){return acts.map(a=>a.type==='turn'?((a.deg<0?'↰':'↱')+Math.abs(a.deg)):('↑'+a.m)).join(' ');}
function renderOcs(){
  const fl=$('fLen').value, fr=$('fRad').value, fv=$('fVer').value,
        co=$('fColl').checked, hc=$('fHc').checked;
  let n=0;
  const rows=OCS.filter(o=>
      (!fl||o.seq_len==fl)&&(!fr||o.radius_m==fr)&&(!fv||o.verdict===fv)&&
      (!co||o.collided)&&(!hc||o.human_check)).map(o=>{
    n++;
    const vb=o.verdict?`<span class="badge" style="background:${VCOL[o.verdict]||'#888'}">${o.verdict.replace('_','·')}</span>`:'';
    const act=(o.collided&&o.act_i!=null)?' @act'+(o.act_i+1):'';
    const hit=o.collided?((o.hit?'hit '+o.hit:'unattr')+act+' @'+o.arc.toFixed(2)+'m'):'—';
    return `<tr class="oc" data-o="${o.outcome_id}" onclick="selOc('${o.outcome_id}',this)">
      <td>L${o.seq_len}</td><td class="mono">${actStr(o.actions)}</td>
      <td>r${o.radius_m}</td><td>${o.collided?'✓':'✗'} ${hit}</td>
      <td>${vb} ${o.human_check?'⚑':''}</td></tr>`;
  }).join('');
  $('ocBody').innerHTML=`<tr><th>len</th><th>actions</th><th>body</th><th>collision</th><th>verdict</th></tr>`+rows;
  $('ocCount').textContent=n+' outcomes';
}
function selOc(oid,tr){
  document.querySelectorAll('#ocBody .oc').forEach(e=>e.classList.remove('on'));
  if(tr) tr.classList.add('on');
  $('status').textContent=' rendering…';
  fetch('/api/ds/render',{method:'POST',body:JSON.stringify({frame_id:FRAME,outcome_id:oid})})
    .then(r=>r.json()).then(d=>{ $('status').textContent=''; if(d.error){alert(d.error);return;} drawDetail(d); });
}

// ---------- live ----------
function addAct(type){
  const d=document.createElement('div'); d.className='act';
  if(type==='turn'){
    d.innerHTML=`turn <select class="dir"><option value="R">right</option><option value="L">left</option></select>`
      +` <input class="mag" type="number" step="15" min="0" value="30" style="width:64px">° `
      +`<button onclick="this.parentNode.remove()">x</button>`;
  }else{
    d.innerHTML=`forward <input class="mag" type="number" step="0.1" min="0" value="1.0" style="width:64px">m `
      +`<button onclick="this.parentNode.remove()">x</button>`;
  }
  d.dataset.type=type; $('lacts').appendChild(d);
}
function getActs(){
  return [...$('lacts').children].map(d=>{
    const mag=+d.querySelector('.mag').value;
    if(d.dataset.type==='turn'){const dir=d.querySelector('.dir').value; return {type:'turn',deg:(dir==='L'?-1:1)*Math.abs(mag)};}
    return {type:'forward',m:mag};
  });
}
let LFID=null;
function liveSample(){
  $('lfid').textContent=' sampling…';
  fetch('/api/frame',{method:'POST',body:JSON.stringify({scene:$('lscene').value,seed:+$('lseed').value})})
   .then(r=>r.json()).then(d=>{ if(d.error){$('lfid').textContent=' '+d.error;return;}
     LFID=d.frame_id; $('lfid').textContent=' '+LFID; $('rgb').src=d.rgb+'?t='+Date.now(); $('rgbov').innerHTML='';
     $('invLine').textContent=Object.entries(d.inventory).map(([k,v])=>`${k}×${v}`).join('  '); });
}
function liveRun(){
  if(!LFID){alert('sample a frame first');return;}
  fetch('/api/judge',{method:'POST',body:JSON.stringify({frame_id:LFID,radius:+$('lrad').value,actions:getActs()})})
   .then(r=>r.json()).then(d=>{ if(d.error){alert(d.error);return;} drawDetail(d); });
}

// ---------- detail: rgb overlay + topdown + reasoning ----------
function drawDetail(d){
  DATA=d;
  if(d.rgb) $('rgb').src=d.rgb+(d.rgb.includes('?')?'':'?t='+Date.now());
  // rgb overlay: object centroids + contact pixel
  let ov=d.objects.filter(o=>!o.structural).map(o=>
    `<circle cx="${o.px[0]}" cy="${o.px[1]}" r="4" fill="none" stroke="#00f" stroke-width="2"/>`).join('');
  if(d.contact_px) ov+=`<circle cx="${d.contact_px[0]}" cy="${d.contact_px[1]}" r="9" fill="none" stroke="red" stroke-width="3"/>`;
  $('rgbov').innerHTML=ov;
  // reasoning
  $('reason').innerHTML=d.reasoning.map(s=>{
    const cls=s.includes('COLLISION')?'ln hit':'ln'; return `<div class="${cls}">${s}</div>`;}).join('');
  renderRel(d);
  // topdown + scrub
  const tf=d.total_forward_m||0;
  $('ctrl').classList.remove('hidden');
  const sl=$('arc'); sl.max=Math.max(tf,0.001); sl.value=sl.max;
  onScrub();
}
function renderRel(d){
  const which=d.collided?'exec':'full';
  const list=d.objects_rel.filter(r=>r[which] && !r.structural)
    .sort((a,b)=>a[which].dist_centroid_m-b[which].dist_centroid_m);
  const rows=list.map(r=>{
      const w=r[which], dc=r.delta_full.d_centroid_m;
      const cls=(r.instance_id===d.contact_id)?'hitrow':'';
      return `<tr class="${cls}"><td>${r.category}${r.instance_id===d.contact_id?' ⟵':''}</td>
        <td>${w.dist_centroid_m.toFixed(2)}</td><td>${w.dist_nearest_m.toFixed(2)}</td>
        <td>${w.bearing_deg.toFixed(0)}°</td><td>${w.in_fov?'✓':''}</td>
        <td style="color:${dc<0?'#2e7d32':'#c62828'}">${dc>=0?'+':''}${dc.toFixed(2)}</td></tr>`;
    }).join('');
  $('relTbl').innerHTML=`<tr><th>object (${list.length})</th><th>cen</th><th>near</th><th>bear</th><th>fov</th><th>Δcen</th></tr>`+rows;
}

// world(x=right,z=fwd) -> svg(px). Fit content into 400x400 with equal scale, z up.
let PROJ=null;
function fit(d){
  let xs=[0], zs=[0];
  d.obstacles_xz.forEach(p=>{xs.push(p[0]);zs.push(p[1]);});
  d.objects.forEach(o=>{xs.push(o.xz[0]);zs.push(o.xz[1]);});
  d.path.forEach(p=>{xs.push(p[0]);zs.push(p[1]);});
  let x0=Math.min(...xs),x1=Math.max(...xs),z0=Math.min(...zs),z1=Math.max(...zs);
  const m=0.4; x0-=m;x1+=m;z0-=m;z1+=m;
  const W=400,H=400, s=Math.min(W/(x1-x0),H/(z1-z0));
  const ox=(W-(x1-x0)*s)/2, oz=(H-(z1-z0)*s)/2;
  PROJ={x0,z0,s,ox,oz,W,H};
}
function P(x,z){return [PROJ.ox+(x-PROJ.x0)*PROJ.s, PROJ.H-(PROJ.oz+(z-PROJ.z0)*PROJ.s)];}
function bodyAt(d,arc){
  const pa=d.path; if(!pa.length) return [0,0,0];
  let best=pa[0];
  for(const p of pa){ if(p[3]<=arc+1e-9) best=p; else break; }
  return best; // [x,z,heading,arc]
}
function onScrub(){
  if(!DATA) return;
  const arc=+$('arc').value; drawTop(DATA,arc);
  $('arcLbl').textContent=`arc ${arc.toFixed(2)} / ${(+$('arc').max).toFixed(2)} m`;
}
function drawTop(d,arc){
  fit(d);
  const r=d.body_radius_m*PROJ.s;
  let g='';
  // view cone
  const R=Math.max(...[3, ...d.path.map(p=>Math.hypot(p[0],p[1]))]);
  const ch=d.cone_half_deg*Math.PI/180;
  [-1,1].forEach(s=>{const e=P(R*Math.sin(s*ch),R*Math.cos(s*ch)),o=P(0,0);
    g+=`<line x1="${o[0]}" y1="${o[1]}" x2="${e[0]}" y2="${e[1]}" stroke="#9db8e0" stroke-dasharray="4" stroke-width="1"/>`;});
  // obstacles
  g+='<g fill="#c4c4c4">'+d.obstacles_xz.map(p=>{const q=P(p[0],p[1]);return `<circle cx="${q[0].toFixed(1)}" cy="${q[1].toFixed(1)}" r="1.2"/>`;}).join('')+'</g>';
  // path
  if(d.path.length){g+='<polyline fill="none" stroke="#1a9c46" stroke-width="2" points="'+d.path.map(p=>{const q=P(p[0],p[1]);return q[0].toFixed(1)+','+q[1].toFixed(1);}).join(' ')+'"/>';}
  // objects
  d.objects.forEach(o=>{const q=P(o.xz[0],o.xz[1]);
    g+=`<circle cx="${q[0]}" cy="${q[1]}" r="${o.structural?2:3.5}" fill="${o.structural?'#bbb':'#111'}"/>`;
    if(!o.structural) g+=`<text x="${q[0]+4}" y="${q[1]+3}" font-size="9" fill="#333">${o.cat}</text>`;});
  // contact
  if(d.contact_xz){const q=P(d.contact_xz[0],d.contact_xz[1]);
    g+=`<path d="M${q[0]-5} ${q[1]-5}L${q[0]+5} ${q[1]+5}M${q[0]-5} ${q[1]+5}L${q[0]+5} ${q[1]-5}" stroke="red" stroke-width="2.5"/>`;}
  // origin triangle
  const o0=P(0,0); g+=`<path d="M${o0[0]} ${o0[1]-6}L${o0[0]-5} ${o0[1]+4}L${o0[0]+5} ${o0[1]+4}Z" fill="#1a73e8"/>`;
  // body circle at current arc
  const b=bodyAt(d,arc), bc=P(b[0],b[1]);
  const past=d.collided && d.first_contact_arc_m!=null && arc>=d.first_contact_arc_m-1e-6;
  const col=past?'#c62828':'#1a73e8';
  g+=`<circle cx="${bc[0]}" cy="${bc[1]}" r="${r.toFixed(1)}" fill="none" stroke="${col}" stroke-width="2"/>`;
  // heading tick
  const hh=b[2]*Math.PI/180, tip=P(b[0]+0.3*Math.sin(hh), b[1]+0.3*Math.cos(hh));
  g+=`<line x1="${bc[0]}" y1="${bc[1]}" x2="${tip[0]}" y2="${tip[1]}" stroke="${col}" stroke-width="1.5"/>`;
  // live distance to contact / nearest target
  const tgt=targetXZ(d);
  if(tgt){const t=P(tgt[0],tgt[1]);
    g+=`<line x1="${bc[0]}" y1="${bc[1]}" x2="${t[0]}" y2="${t[1]}" stroke="#00bcd4" stroke-dasharray="3" stroke-width="1"/>`;
    const dd=Math.hypot(b[0]-tgt[0],b[1]-tgt[1]);
    g+=`<text x="${(bc[0]+t[0])/2}" y="${(bc[1]+t[1])/2}" font-size="10" fill="#0097a7">${dd.toFixed(2)}m</text>`;}
  $('svg').innerHTML=g;
}
function targetXZ(d){
  if(d.contact_xz) return d.contact_xz;
  const ns=d.objects.filter(o=>!o.structural);
  if(!ns.length) return null;
  return ns.reduce((a,b)=>Math.hypot(a.xz[0],a.xz[1])<Math.hypot(b.xz[0],b.xz[1])?a:b).xz;
}
function togglePlay(){
  if(playing){clearInterval(playing);playing=null;$('play').textContent='▶';return;}
  $('play').textContent='⏸';
  const sl=$('arc'); if(+sl.value>=+sl.max) sl.value=0;
  playing=setInterval(()=>{
    let v=+sl.value+ (+sl.max)/60;
    if(v>=+sl.max){v=+sl.max; clearInterval(playing);playing=null;$('play').textContent='▶';}
    sl.value=v; onScrub();
  },50);
}
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--img-dir", default="data/conseq/serve_imgs")
    ap.add_argument("--dataset", default=None, help="a collection dir (records.jsonl) to review")
    args = ap.parse_args()
    state = State(args.img_dir, dataset=args.dataset)
    httpd = HTTPServer(("0.0.0.0", args.port), make_handler(state, PAGE))
    ds = f"  dataset={args.dataset} ({len(state.by_frame)} frames)" if args.dataset else ""
    print(f"serving on http://localhost:{args.port}  ({len(state.by_id)} scenes){ds}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
