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


def _uniq_action_seqs(outcomes):
    """Distinct action lists across a frame's stored outcomes (order-stable)."""
    seen, seqs = set(), []
    for oc in outcomes:
        key = tuple((a["type"], a.get("deg", a.get("m"))) for a in oc["actions"])
        if key not in seen:
            seen.add(key); seqs.append(oc["actions"])
    return seqs


class State:
    """Multi-dataset (GS + HM3D) review; robot-body panel re-renders/re-judges each
    frame at a chosen camera height + chassis radius. One Habitat session at a time."""

    def __init__(self, img_dir, datasets):
        # datasets: list of (name, dir, backend)
        self.img_dir = img_dir
        os.makedirs(img_dir, exist_ok=True)
        self.dsets = {}
        for name, d, backend in datasets:
            by_frame, by_scene = {}, {}
            for r in REC.read_records(os.path.join(d, "records.jsonl")):
                by_frame[r["frame_id"]] = r
                by_scene.setdefault(r["scene_id"], []).append(r["frame_id"])
            if backend == "gs":
                from pipeline.gs_sim import GsSimSession, discover_gs_scenes
                Session = GsSimSession
                by_id = {os.path.basename(s.rstrip("/")): s for s in discover_gs_scenes()}
                kw = {}
            else:
                Session = SimSession
                by_id = {os.path.basename(os.path.dirname(s)): s
                         for s in discover_semantic_scenes()}
                kw = {"heights": list(config.RENDER_HEIGHTS)}
            self.dsets[name] = dict(dir=d, backend=backend, by_frame=by_frame,
                                    by_scene=by_scene, by_id=by_id, Session=Session, kw=kw)
        self.active = datasets[0][0]
        self.sim = None
        self.open_key = None
        self.fcache = {}          # (active, frame_id, htag) -> Frame

    # ---- dataset / scene ----
    @property
    def ds(self):
        return self.dsets[self.active]

    def ds_names(self):
        return [{"name": n, "backend": self.dsets[n]["backend"]} for n in self.dsets]

    def set_active(self, name):
        if name in self.dsets and name != self.active:
            self.active = name
            if self.sim is not None:
                self.sim.close()
            self.sim = None; self.open_key = None
        return {"active": self.active, "backend": self.ds["backend"], "radii": self.radii()}

    def radii(self):
        rs = set()
        for r in self.ds["by_frame"].values():
            for oc in r["outcomes"]:
                rs.add(round(oc["body"]["radius_m"], 3))
        return sorted(rs)

    def _open(self, scene_id):
        key = (self.active, scene_id)
        if self.open_key == key and self.sim is not None:
            return
        if self.sim is not None:
            self.sim.close()
        ds = self.ds
        self.sim = ds["Session"](ds["by_id"][scene_id], **ds["kw"])
        self.open_key = key

    def _get_frame(self, fid, cam_h):
        """Rebuild + cache a Frame at camera height cam_h; also save its rgb PNG."""
        htag = config.height_tag(cam_h)
        ck = (self.active, fid, htag)
        if ck in self.fcache:
            return self.fcache[ck], htag
        ds = self.ds
        r = ds["by_frame"][fid]
        self._open(r["scene_id"])
        pos = np.asarray(r["pose"]["position"], float)
        fr = build_frame(self.sim, pos, float(r["pose"]["yaw_rad"]), cam_h=cam_h,
                         frame_id=fid, scene_id=r["scene_id"],
                         scene_glb=ds["by_id"][r["scene_id"]])
        Image.fromarray(fr.rgb).save(os.path.join(self.img_dir, f"{fid}_h{htag}.png"))
        self.fcache[ck] = fr
        return fr, htag

    # ---- browse (pure records) ----
    def ds_scenes(self):
        return [{"scene": s, "n_frames": len(f)} for s, f in sorted(self.ds["by_scene"].items())]

    def ds_frames(self, scene):
        ds = self.ds
        out = []
        for fid in ds["by_scene"].get(scene, []):
            r = ds["by_frame"][fid]
            out.append({"frame_id": fid, "rgb": "/dsimg/" + r["image_path"],
                        "n_nonstructural": r["n_nonstructural"],
                        "inventory": r["category_inventory"]})
        return out

    # ---- robot-body: re-judge a frame at (height, radius) ----
    def rejudge(self, fid, height, radius):
        ds = self.ds
        r = ds["by_frame"].get(fid)
        if r is None:
            return None
        fr, htag = self._get_frame(fid, height)
        self.sim.recompute_navmesh(radius)
        nav = self.sim.nav(fr.position, fr.yaw_rad)
        out = []
        for i, acts in enumerate(_uniq_action_seqs(r["outcomes"])):
            oc = judge(fr, Cylinder(radius_m=radius), parse_actions(acts), nav=nav, geodesic=False)
            c = oc.get("contact")
            out.append({"idx": i, "actions": acts, "seq_len": len(acts),
                        "radius_m": radius, "collided": oc["collided"],
                        "arc": oc["first_contact_arc_m"],
                        "act_i": oc.get("contact_action_index"),
                        "hit": (c["category"] if c and not c.get("unattributed") else None),
                        "verdict": (oc["nav_check"]["verdict"] if oc.get("nav_check") else None)})
        return {"rgb": f"/img/{fid}_h{htag}.png", "outcomes": out,
                "inventory": fr.category_inventory,
                "camera_height": height, "radius_m": radius}

    def ds_render(self, fid, actions, height, radius):
        ds = self.ds
        r = ds["by_frame"].get(fid)
        if r is None:
            return None
        fr, htag = self._get_frame(fid, height)
        self.sim.recompute_navmesh(radius)
        nav = self.sim.nav(fr.position, fr.yaw_rad)
        oc = judge(fr, Cylinder(radius_m=radius), parse_actions(actions), nav=nav, geodesic=False)
        oc["outcome_id"] = f"{fid}-h{htag}-r{int(radius*100)}"
        obstacles = _obstacles_from_frame(fr)
        pl = geometry_payload(fr.objects, oc, obstacles, f"/img/{fid}_h{htag}.png")
        pl["camera_height"] = height
        return pl

    # ---- live explore (uses default height) ----
    def sample(self, scene_id, seed):
        self._open(scene_id)
        self.sim.recompute_navmesh(0.25)
        pose = self.sim.sample_random_pose(np.random.default_rng(seed), [0.25])
        if pose is None:
            return None
        pos, yaw = pose
        fid = f"{scene_id}-s{seed}"
        fr = build_frame(self.sim, pos, yaw, frame_id=fid, scene_id=scene_id,
                         scene_glb=self.ds["by_id"][scene_id])
        self.fcache[(self.active, fid, "live")] = fr
        Image.fromarray(fr.rgb).save(os.path.join(self.img_dir, fid + "_rgb.png"))
        return {"frame_id": fid, "rgb": f"/img/{fid}_rgb.png",
                "inventory": fr.category_inventory}

    def live_render(self, frame_id, radius, actions):
        fr = self.fcache.get((self.active, frame_id, "live"))
        if fr is None:
            return None
        self.sim.recompute_navmesh(radius)
        oc = judge(fr, Cylinder(radius_m=radius), parse_actions(actions),
                   nav=self.sim.nav(fr.position, fr.yaw_rad), geodesic=False)
        oc["outcome_id"] = f"{frame_id}-live"
        return geometry_payload(fr.objects, oc, _obstacles_from_frame(fr),
                                f"/img/{frame_id}_rgb.png")


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
            if p == "/api/ds/names":
                return self._json({"datasets": state.ds_names(), "active": state.active,
                                   "radii": state.radii(),
                                   "heights": list(config.RENDER_HEIGHTS)})
            if p == "/api/ds/scenes":
                return self._json(state.ds_scenes())
            if p == "/api/ds/frames":
                return self._json(state.ds_frames(q.get("scene", [""])[0]))
            if p == "/api/scenes":
                return self._json(sorted(state.ds["by_id"]))
            if p.startswith("/dsimg/"):
                return self._file(_safe_join(state.ds["dir"], p[len("/dsimg/"):]), "image/png")
            if p.startswith("/img/"):
                return self._file(os.path.join(state.img_dir, os.path.basename(p)), "image/png")
            return self._send(404, b"")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/api/ds/set":
                    r = state.set_active(req["name"])
                elif self.path == "/api/ds/rejudge":
                    r = state.rejudge(req["frame_id"], float(req["height"]), float(req["radius"]))
                elif self.path == "/api/ds/render":
                    r = state.ds_render(req["frame_id"], req["actions"],
                                        float(req["height"]), float(req["radius"]))
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


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_viz.html"), encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--img-dir", default="data/conseq/serve_imgs")
    ap.add_argument("--dataset", default=None, help="single collection dir (records.jsonl)")
    ap.add_argument("--backend", choices=["hm3d", "gs"], default="hm3d")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="name=dir:backend ...  (multi-dataset; overrides --dataset)")
    args = ap.parse_args()
    if args.datasets:
        datasets = []
        for spec in args.datasets:
            name, rest = spec.split("=", 1)
            d, backend = rest.rsplit(":", 1)
            datasets.append((name, d, backend))
    elif args.dataset:
        datasets = [("default", args.dataset, args.backend)]
    else:
        ap.error("need --datasets or --dataset")
    state = State(args.img_dir, datasets)
    httpd = HTTPServer(("0.0.0.0", args.port), make_handler(state, PAGE))
    print(f"serving on http://localhost:{args.port}  datasets={[d[0] for d in datasets]}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
