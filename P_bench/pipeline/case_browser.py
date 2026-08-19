"""Deterministic static case-browser projection and rendering.

The browser is an internal, read-only view over validated candidate artifacts.
It may present A/B answers as open text, but it never changes the benchmark
items or their scorer contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from pipeline import io_utils


BROWSER_SCHEMA = "egoconseq.case-browser.v1"
C1_TASK_ID = "C1_future_view_selection"
CHECKPOINT_DIRECTION_TASK_ID = "A4_checkpoint_direction"
BROWSER_DIAGNOSTIC_TASK_IDS = (CHECKPOINT_DIRECTION_TASK_ID,)


@dataclass(frozen=True)
class BrowserSource:
    """One already-validated candidate artifact used by the browser."""

    label: str
    artifact_root: Path
    benchmark: dict
    scene_by_case_id: Mapping[str, str] = field(default_factory=dict)
    technical_by_case_id: Mapping[str, dict] = field(default_factory=dict)


def browser_source_from_validated_artifact(
        label: str, artifact_root: Path, benchmark: dict) -> BrowserSource:
    """Join browser metadata after the caller validated the artifact.

    ``benchmark`` must be the value returned by
    :func:`candidate_preview.validate_preview_artifact`.  Keeping this path
    explicit lets the candidate compiler validate once and then render,
    instead of repeating expensive B geometry checks solely for the report.
    """
    if not str(label).strip():
        raise ValueError("case-browser source label must be non-empty")
    artifact_root = Path(artifact_root)
    atoms = {
        str(row["id"]): row for row in io_utils.read_jsonl(
            artifact_root / "private" / "atoms.jsonl", require_dict=True)
    }
    contexts = {
        str(row["record_sha256"]): row["context"]
        for row in io_utils.read_jsonl(
            artifact_root / "private" / "record_contexts.jsonl",
            require_dict=True)
    }
    answers = {
        str(row["id"]): row for row in io_utils.read_jsonl(
            artifact_root / "private" / "answers.jsonl", require_dict=True)
    }
    scenes = {}
    technical = {}
    for case in benchmark["cases"]:
        case_id = str(case["public"]["id"])
        atom_ref = str(case["atom_ref"])
        atom = atoms[atom_ref]
        record_sha256 = str(atom["record_sha256"])
        context = contexts[record_sha256]
        source = context.get("source") or {}
        scene_id = context.get("scene_id") or source.get("scene_id")
        if not scene_id:
            raise ValueError(
                f"case-browser source lacks authenticated scene: {case_id}")
        scenes[case_id] = str(scene_id)
        answer = answers[case_id]
        selection = answer.get("selection_certificate") or {}
        outcome = atom.get("outcome") or {}
        row = {
            "record_sha256": record_sha256,
            "frame_id": str(atom["frame_id"]),
            "outcome_id": str(atom["outcome_id"]),
            "outcome_sha256": str(atom["outcome_sha256"]),
            "base_rollout_key": outcome.get("base_rollout_key"),
        }
        if selection.get("family_id") is not None:
            row["family_id"] = str(selection["family_id"])
        technical[case_id] = row
    return BrowserSource(
        label=str(label), artifact_root=artifact_root,
        benchmark=benchmark, scene_by_case_id=scenes,
        technical_by_case_id=technical)


def _asset_path(source: BrowserSource, relative: object,
                report_root: Path) -> str:
    absolute = source.artifact_root / str(relative)
    return os.path.relpath(absolute, report_root)


def _answer_display(case: dict) -> str:
    answer = str(case["canonical_answer"])
    choices = case["public"].get("choices") or []
    by_id = {str(choice.get("id")): choice for choice in choices}
    selected = by_id.get(answer)
    if selected is None:
        return answer
    return str(selected.get("text", selected.get("id", answer)))


def _browser_case(source: BrowserSource, case: dict,
                  report_root: Path) -> dict:
    public = case["public"]
    case_id = str(public["id"])
    task_id = str(public["task_id"])
    model_input = dict(public.get("model_input") or {})
    initial_rgb = model_input.pop("initial_rgb")
    initial_sha256 = model_input.pop("initial_rgb_sha256")
    result = {
        "id": case_id,
        "task_id": task_id,
        "head": str(public["result_head"]),
        "source_label": source.label,
        "scene_id": source.scene_by_case_id.get(case_id, "unknown"),
        "question": str(public["question"]),
        "response_mode": (
            "image_mcq" if task_id == C1_TASK_ID else "open_text"),
        "answer": {
            "raw": str(case["canonical_answer"]),
            "display": _answer_display(case),
        },
        "initial_image": {
            "path": _asset_path(source, initial_rgb, report_root),
            "sha256": str(initial_sha256),
        },
        "model_input": model_input,
        "atom_ref": str(case["atom_ref"]),
        "oracle_ref": dict(case.get("oracle_ref") or {}),
        "technical": dict(source.technical_by_case_id.get(case_id) or {}),
    }
    if task_id == C1_TASK_ID:
        choices = public.get("choices") or []
        answer = str(case["canonical_answer"])
        if len(choices) != 4 or any("image" not in choice for choice in choices):
            raise ValueError(
                f"C1 case {case_id} must have exactly four image choices")
        choice_ids = [str(choice["id"]) for choice in choices]
        if len(set(choice_ids)) != 4:
            raise ValueError(f"C1 case {case_id} repeats an image choice id")
        if answer not in choice_ids:
            raise ValueError(
                f"C1 case {case_id} answer is outside image choices")
        result["choices"] = [{
            "id": str(choice["id"]),
            "path": _asset_path(source, choice["image"], report_root),
            "sha256": str(choice["image_sha256"]),
            "is_correct": str(choice["id"]) == answer,
        } for choice in choices]
    return result


def build_browser_payload(sources: Sequence[BrowserSource], *,
                          report_root: Path) -> dict:
    """Project validated candidate cases into a compact browser payload."""
    report_root = Path(report_root)
    coverage: dict[str, int] = {}
    cases = []
    seen_case_ids = set()
    for source in sources:
        for task_id, count in source.benchmark["coverage"].items():
            coverage[task_id] = coverage.get(task_id, 0) + int(count)
        for case in source.benchmark["cases"]:
            case_id = str(case["public"]["id"])
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case id: {case_id}")
            seen_case_ids.add(case_id)
            cases.append(_browser_case(source, case, report_root))
    return {
        "schema": BROWSER_SCHEMA,
        "candidate_only": True,
        "headline_eligible": False,
        "coverage": coverage,
        "sources": [{
            "label": source.label,
            "case_count": len(source.benchmark["cases"]),
        } for source in sources],
        "cases": cases,
    }


_STYLE = r"""
:root {
  color-scheme: light;
  --ink: #18201f;
  --muted: #68716f;
  --line: #dfe3df;
  --paper: #f4f6f3;
  --surface: #ffffff;
  --chrome: #15201e;
  --chrome-2: #21302d;
  --a: #c85b2d;
  --b: #276a9a;
  --c: #7353a6;
  --good: #1b7553;
  --warn: #ad6928;
  --shadow: 0 12px 32px rgba(21, 32, 30, .10);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { background: var(--paper); }
body { margin: 0; color: var(--ink); background: var(--paper); }
button, input, select { font: inherit; }
button { color: inherit; }
.topbar {
  display: grid; grid-template-columns: minmax(260px, 1fr) auto;
  gap: 28px; padding: 22px clamp(18px, 3vw, 44px); color: #f7faf8;
  background: linear-gradient(115deg, var(--chrome), #1e302c 62%, #263a36);
  border-bottom: 1px solid rgba(255,255,255,.12);
}
.brand { display: flex; align-items: flex-start; gap: 14px; }
.brand-mark {
  width: 38px; height: 38px; display: grid; place-items: center; flex: none;
  border: 1px solid rgba(255,255,255,.35); border-radius: 11px;
  font: 800 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
  background: rgba(255,255,255,.08);
}
.brand h1 { margin: 0; font-size: 20px; letter-spacing: -.02em; }
.brand p { margin: 5px 0 0; color: #b9c8c3; font-size: 12px; }
.status-stack { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; }
.status-pill {
  align-self: flex-start; padding: 6px 9px; border-radius: 999px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
  color: #dbe6e2; font-size: 11px; letter-spacing: .02em;
}
.status-pill--warn { color: #ffe0af; border-color: rgba(255,194,100,.42); }
.workspace-nav {
  position: sticky; top: 0; z-index: 30; padding: 0 clamp(14px, 3vw, 44px);
  background: rgba(255,255,255,.96); border-bottom: 1px solid var(--line);
  box-shadow: 0 4px 18px rgba(21,32,30,.05); backdrop-filter: blur(12px);
}
.task-tabs { display: flex; align-items: stretch; gap: 2px; overflow-x: auto; }
.task-tab {
  min-width: max-content; padding: 13px 13px 11px; border: 0;
  border-bottom: 3px solid transparent; background: transparent; cursor: pointer;
  color: #59615f; font-weight: 720; font-size: 13px;
}
.task-tab span { margin-left: 6px; color: #8a9290; font: 600 11px ui-monospace, monospace; }
.task-tab:hover { background: #f4f6f3; color: var(--ink); }
.task-tab[aria-pressed="true"] { color: var(--ink); border-color: var(--active, var(--ink)); }
.task-tab[data-head="A"] { --active: var(--a); }
.task-tab[data-head="B"] { --active: var(--b); }
.task-tab[data-head="C"] { --active: var(--c); }
.toolbar-wrap { padding: 18px clamp(14px, 3vw, 44px) 0; }
.toolbar {
  display: grid; grid-template-columns: minmax(220px, 1.5fr) repeat(7, minmax(108px, .55fr)) auto;
  gap: 9px; align-items: end; padding: 13px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 3px 16px rgba(21,32,30,.04);
}
.field { min-width: 0; }
.field label { display: block; margin: 0 0 5px; color: #707876; font-size: 10px; font-weight: 760; letter-spacing: .08em; text-transform: uppercase; }
.field input, .field select {
  width: 100%; height: 36px; border: 1px solid #d8ddda; border-radius: 8px;
  background: #fbfcfb; color: var(--ink); padding: 0 9px; outline: none;
}
.field input:focus, .field select:focus { border-color: #507b70; box-shadow: 0 0 0 3px rgba(59,112,98,.12); }
.reset-button, .icon-button, .pager button, .drawer-button {
  border: 1px solid #d8ddda; border-radius: 8px; background: #fff; cursor: pointer;
}
.reset-button { height: 36px; padding: 0 12px; }
.reset-button:hover, .icon-button:hover, .pager button:hover, .drawer-button:hover { background: #f0f3f0; }
.result-bar {
  display: flex; justify-content: space-between; align-items: center; gap: 14px;
  padding: 17px clamp(18px, 3vw, 44px) 10px; color: var(--muted); font-size: 12px;
}
.result-bar strong { color: var(--ink); font-size: 14px; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; }
.legend i { width: 8px; height: 8px; display: inline-block; margin-right: 5px; border-radius: 50%; }
.legend .a { background: var(--a); } .legend .b { background: var(--b); } .legend .c { background: var(--c); }
.case-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  align-items: start; gap: 16px; padding: 0 clamp(14px, 3vw, 44px) 22px;
}
.case-card {
  min-width: 0; overflow: hidden; background: var(--surface); border: 1px solid var(--line);
  border-top: 3px solid var(--task-color); border-radius: 13px; box-shadow: 0 5px 22px rgba(21,32,30,.055);
  transition: transform .16s ease, box-shadow .16s ease;
}
.case-card:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
.case-card[data-head="A"] { --task-color: var(--a); }
.case-card[data-head="B"] { --task-color: var(--b); }
.case-card[data-head="C"] { --task-color: var(--c); }
.case-card--c1 { grid-column: span 2; }
.card-head { display: flex; align-items: center; gap: 8px; padding: 11px 13px 9px; }
.task-code { color: var(--task-color); font: 800 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }
.case-id { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #858d8a; font: 11px ui-monospace, monospace; }
.source-chip { margin-left: auto; color: #66706d; background: #f1f3f1; padding: 3px 6px; border-radius: 5px; font-size: 10px; }
.image-button { display: block; position: relative; width: 100%; padding: 0; border: 0; background: #e7ebe7; cursor: zoom-in; overflow: hidden; }
.initial-frame { aspect-ratio: 4 / 3; }
.image-button img { display: block; width: 100%; height: 100%; object-fit: cover; }
.image-role {
  position: absolute; left: 8px; bottom: 8px; padding: 4px 7px; border-radius: 5px;
  color: white; background: rgba(15,22,21,.76); font: 700 10px ui-monospace, monospace;
}
.card-content { padding: 13px; }
.question { margin: 0 0 11px; font: 660 15px/1.42 Georgia, "Times New Roman", serif; letter-spacing: -.005em; }
.action-timeline { display: flex; gap: 5px; align-items: stretch; overflow-x: auto; padding: 2px 0 10px; }
.primitive {
  min-width: max-content; display: grid; grid-template-columns: 18px auto; align-items: center;
  gap: 5px; padding: 6px 7px; border-radius: 7px; background: #f1f3f1; color: #49524f;
  font: 650 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.primitive b { display: grid; place-items: center; width: 18px; height: 18px; border-radius: 50%; background: #fff; color: #7a827f; }
.primitive--answer { color: #753714; background: #ffe2ca; box-shadow: inset 0 0 0 1px #f0ad7b; }
.answer-block { display: flex; gap: 10px; align-items: baseline; padding: 10px 11px; border-left: 3px solid var(--task-color); background: #f6f8f6; border-radius: 0 8px 8px 0; }
.answer-block small { color: #78817e; font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.answer-block strong { overflow-wrap: anywhere; font-size: 15px; }
.gt-hidden .answer-block strong, .gt-hidden .choice-correct { filter: blur(8px); user-select: none; }
.meta-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 11px; }
.meta-chip { padding: 4px 6px; border: 1px solid #e0e4e1; border-radius: 5px; color: #66706d; background: #fafbfa; font: 10px ui-monospace, monospace; }
.card-actions { display: flex; justify-content: flex-end; padding: 0 13px 13px; }
.drawer-button { padding: 7px 10px; font-size: 11px; }
.c1-layout { display: grid; grid-template-columns: minmax(230px, .86fr) minmax(320px, 1.14fr); gap: 10px; padding: 0 13px 12px; }
.c1-initial, .choice-image { border-radius: 9px; }
.c1-choices { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 7px; }
.choice-image { aspect-ratio: 4 / 3; outline: 1px solid rgba(255,255,255,.5); }
.choice-label { position: absolute; left: 7px; top: 7px; padding: 3px 6px; border-radius: 5px; background: rgba(18,26,24,.75); color: white; font: 750 10px ui-monospace, monospace; }
.c1-choices .image-role { display: none; }
.choice-correct { outline: 3px solid var(--good); outline-offset: -3px; }
.choice-correct::after { content: "GT"; position: absolute; right: 7px; top: 7px; padding: 3px 6px; color: #fff; background: var(--good); border-radius: 5px; font: 800 10px ui-monospace, monospace; }
.empty-state { grid-column: 1 / -1; padding: 56px 20px; text-align: center; color: var(--muted); border: 1px dashed #c9cfcb; border-radius: 14px; background: rgba(255,255,255,.55); }
.pager { display: flex; justify-content: center; align-items: center; gap: 8px; padding: 4px 20px 44px; }
.pager button { min-width: 38px; height: 34px; padding: 0 10px; }
.pager button:disabled { opacity: .4; cursor: default; }
.pager span { min-width: 110px; text-align: center; color: var(--muted); font: 11px ui-monospace, monospace; }
.dialog-backdrop { position: fixed; inset: 0; z-index: 70; background: rgba(11,17,16,.46); opacity: 0; pointer-events: none; transition: opacity .18s ease; }
.dialog-backdrop.is-open { opacity: 1; pointer-events: auto; }
.case-drawer {
  position: fixed; inset: 0 0 0 auto; z-index: 80; width: min(720px, 94vw); overflow-y: auto;
  background: #fbfcfb; box-shadow: -24px 0 60px rgba(10,18,16,.26); transform: translateX(104%);
  transition: transform .2s ease; outline: none;
}
.case-drawer.is-open { transform: translateX(0); }
.drawer-head { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 10px; padding: 13px 16px; border-bottom: 1px solid var(--line); background: rgba(251,252,251,.96); backdrop-filter: blur(10px); }
.drawer-head strong { flex: 1; }
.icon-button { width: 34px; height: 34px; }
.drawer-body { padding: 16px 18px 44px; }
.drawer-image { border-radius: 10px; overflow: hidden; }
.drawer-c1 { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 9px; margin-top: 10px; }
.detail-section { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
.detail-section h3 { margin: 0 0 9px; font-size: 12px; text-transform: uppercase; letter-spacing: .09em; color: #6f7875; }
.detail-grid { display: grid; grid-template-columns: minmax(120px,.38fr) 1fr; gap: 7px 12px; margin: 0; font-size: 12px; }
.detail-grid dt { color: #7a8380; }
.detail-grid dd { margin: 0; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.lightbox { position: fixed; inset: 0; z-index: 110; display: none; background: rgba(5,9,8,.96); color: white; }
.lightbox.is-open { display: grid; grid-template-rows: auto 1fr auto; }
.lightbox-toolbar { display: flex; align-items: center; gap: 7px; padding: 11px 14px; background: rgba(18,25,23,.88); }
.lightbox-toolbar .icon-button { color: white; border-color: #53615d; background: #25312e; }
.lightbox-title { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font: 12px ui-monospace, monospace; }
.lightbox-stage { position: relative; overflow: hidden; display: grid; place-items: center; cursor: grab; }
.lightbox-stage.is-dragging { cursor: grabbing; }
.lightbox-image { display: block; max-width: 92vw; max-height: 80vh; user-select: none; transform-origin: center; will-change: transform; }
.lightbox-foot { padding: 9px 14px; color: #aebbb7; text-align: center; font: 11px ui-monospace, monospace; }
@media (max-width: 1180px) {
  .toolbar { grid-template-columns: repeat(4, minmax(130px,1fr)); }
  .field--search { grid-column: span 2; }
}
@media (max-width: 760px) {
  .topbar { grid-template-columns: 1fr; gap: 12px; padding: 17px 16px; }
  .status-stack { justify-content: flex-start; }
  .toolbar { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .field--search { grid-column: span 2; }
  .case-grid { grid-template-columns: 1fr; }
  .case-card--c1 { grid-column: auto; }
  .c1-layout { grid-template-columns: 1fr; }
  .case-drawer { width: 100vw; }
  .result-bar { align-items: flex-start; flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
"""


_DATASET_STYLE = r"""
.dataset-rail {
  padding: 14px clamp(14px, 3vw, 44px) 0; background: var(--paper);
}
.dataset-switcher {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 9px; max-width: 1080px;
}
.dataset-switch {
  position: relative; display: grid; grid-template-columns: 1fr auto;
  gap: 4px 12px; align-items: center; min-height: 64px; padding: 11px 13px;
  border: 1px solid #d9dedb; border-radius: 12px; background: #fff;
  text-align: left; cursor: pointer; box-shadow: 0 3px 14px rgba(21,32,30,.04);
}
.dataset-switch:hover { border-color: #9eaaa6; background: #fbfcfb; }
.dataset-switch[aria-pressed="true"] {
  border-color: #426f64; box-shadow: inset 0 0 0 1px #426f64,
    0 7px 20px rgba(34,85,72,.11); background: #f7fbf9;
}
.dataset-switch strong { font-size: 13px; letter-spacing: -.01em; }
.dataset-switch b {
  grid-row: span 2; color: #304641; font: 760 18px/1 ui-monospace, monospace;
}
.dataset-switch small { color: #77817e; font-size: 10px; }
.dataset-switch .pending { color: var(--warn); font-weight: 760; }
@media (max-width: 760px) {
  .dataset-switcher { grid-template-columns: 1fr; }
}
"""


_SCRIPT = r"""
(() => {
  'use strict';
  const payload = JSON.parse(document.getElementById('case-browser-data').textContent);
  const TASK_LABELS = {
    A1_collision: 'A1 · 碰撞判断',
    A2_collision_step_grounding: 'A2 · 碰撞动作',
    A3_contact_object: 'A3 · 接触对象',
    B1_endpoint_distance: 'B1 · 终点距离',
    B2_endpoint_direction: 'B2 · 终点方向',
    C1_future_view_selection: 'C1 · 终点视野'
  };
  const ALL = 'all';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const make = (tag, className, text) => {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined && text !== null) value.textContent = String(text);
    return value;
  };
  const state = {
    task: ALL, query: '', source: '', scene: '', primitives: '', radius: '',
    fov: '', answer: '', sort: 'artifact', page: 1, pageSize: null,
    showGT: true, selectedCase: null
  };
  let lastFocus = null;
  let currentRows = [];
  let lightboxItems = [];
  let lightboxIndex = 0;
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let dragging = false;
  let dragStart = null;

  function actionsText(actions) {
    return (actions || []).map((action, index) => {
      if (action.type === 'forward') return `${index + 1}: 前进 ${action.m} m`;
      const side = Number(action.deg) >= 0 ? '右转' : '左转';
      return `${index + 1}: ${side} ${Math.abs(Number(action.deg))}°`;
    }).join(' → ');
  }

  function targetText(target) {
    if (!target) return '';
    if (typeof target === 'string') return target;
    for (const key of ['display_name', 'label', 'name', 'category', 'reference_text']) {
      if (target[key]) return String(target[key]);
    }
    return '';
  }

  function searchText(row) {
    if (row._searchText) return row._searchText;
    row._searchText = [row.id, row.task_id, row.source_label, row.scene_id,
      row.question, row.answer.display, actionsText(row.model_input.actions),
      targetText(row.model_input.target), row.oracle_ref.frame_id,
      row.oracle_ref.outcome_id].filter(Boolean).join(' ').toLowerCase();
    return row._searchText;
  }

  function parseHash() {
    const params = new URLSearchParams(location.hash.replace(/^#/, ''));
    if (params.has('task')) state.task = params.get('task') || ALL;
    if (params.has('q')) state.query = params.get('q') || '';
    for (const key of ['source', 'scene', 'primitives', 'radius', 'fov', 'answer', 'sort']) {
      if (params.has(key)) state[key] = params.get(key) || '';
    }
    if (params.has('page')) state.page = Math.max(1, Number(params.get('page')) || 1);
    if (params.has('size')) state.pageSize = [12,24,48].includes(Number(params.get('size'))) ? Number(params.get('size')) : null;
    if (params.has('gt')) state.showGT = params.get('gt') !== 'hidden';
    if (params.has('case')) state.selectedCase = params.get('case');
  }

  function writeHash() {
    const params = new URLSearchParams();
    if (state.task !== ALL) params.set('task', state.task);
    if (state.query) params.set('q', state.query);
    for (const key of ['source', 'scene', 'primitives', 'radius', 'fov', 'answer']) {
      if (state[key]) params.set(key, state[key]);
    }
    if (state.sort !== 'artifact') params.set('sort', state.sort);
    if (state.page > 1) params.set('page', String(state.page));
    if (state.pageSize) params.set('size', String(state.pageSize));
    if (!state.showGT) params.set('gt', 'hidden');
    if (state.selectedCase) params.set('case', state.selectedCase);
    history.replaceState(null, '', `#${params.toString()}`);
  }

  function option(select, value, label) {
    const item = make('option', '', label);
    item.value = String(value);
    select.append(item);
  }

  function populateSelect(id, values, allLabel) {
    const select = $(id);
    select.replaceChildren();
    option(select, '', allLabel);
    values.forEach(value => option(select, value, value));
  }

  function discreteValues(field) {
    return [...new Set(payload.cases.map(row => field(row)).filter(value => value !== '' && value !== null && value !== undefined))]
      .sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
  }

  function currentPageSize() {
    return state.pageSize || (state.task === 'C1_future_view_selection' ? 12 : 24);
  }

  function baseFiltered(ignoreTask = false) {
    const query = state.query.trim().toLowerCase();
    return payload.cases.filter(row => {
      if (!ignoreTask && state.task !== ALL && row.task_id !== state.task) return false;
      if (query && !searchText(row).includes(query)) return false;
      if (state.source && row.source_label !== state.source) return false;
      if (state.scene && row.scene_id !== state.scene) return false;
      if (state.primitives && String((row.model_input.actions || []).length) !== state.primitives) return false;
      if (state.radius && String(row.model_input.body_radius_m) !== state.radius) return false;
      if (state.fov && String(row.model_input.hfov_deg) !== state.fov) return false;
      if (state.answer && row.answer.display !== state.answer) return false;
      return true;
    });
  }

  function sortRows(rows) {
    const values = rows.slice();
    if (state.sort === 'scene') values.sort((a,b) => a.scene_id.localeCompare(b.scene_id) || a.id.localeCompare(b.id));
    if (state.sort === 'actions') values.sort((a,b) => (a.model_input.actions || []).length - (b.model_input.actions || []).length || a.id.localeCompare(b.id));
    if (state.sort === 'answer') values.sort((a,b) => a.answer.display.localeCompare(b.answer.display, undefined, {numeric:true}) || a.id.localeCompare(b.id));
    return values;
  }

  function primitiveTimeline(row) {
    const timeline = make('div', 'action-timeline');
    const answerIndex = row.task_id === 'A2_collision_step_grounding' ? Number(row.answer.raw.replace('action_', '')) : null;
    (row.model_input.actions || []).forEach((action, index) => {
      const primitive = make('div', `primitive${answerIndex === index + 1 ? ' primitive--answer' : ''}`);
      primitive.append(make('b', '', index + 1));
      if (action.type === 'forward') primitive.append(make('span', '', `F ${action.m} m`));
      else primitive.append(make('span', '', `${Number(action.deg) >= 0 ? 'R' : 'L'} ${Math.abs(Number(action.deg))}°`));
      timeline.append(primitive);
    });
    return timeline;
  }

  function imageButton(image, role, row, extraClass = '') {
    const button = make('button', `image-button ${extraClass}`.trim());
    button.type = 'button';
    button.setAttribute('aria-label', `放大 ${role}`);
    const img = make('img');
    img.src = image.path;
    img.alt = `${row.id} · ${role}`;
    // Only the current page exists in the DOM, so eager loading here means
    // 24 A/B frames or 60 C1 frames rather than the full 2,989-image atlas.
    // This avoids persistent grey placeholders in offline file:// viewing.
    img.loading = 'eager';
    img.decoding = 'async';
    button.append(img, make('span', 'image-role', role));
    button.addEventListener('click', event => {
      event.stopPropagation();
      openLightbox(row, role);
    });
    return button;
  }

  function chip(value) { return make('span', 'meta-chip', value); }

  function metadata(row) {
    const model = row.model_input;
    const line = make('div', 'meta-row');
    line.append(chip(row.scene_id), chip(`r=${model.body_radius_m} m`),
      chip(`h=${model.camera_height_above_visible_floor_m} m`),
      chip(`HFOV ${model.hfov_deg}°`));
    const target = targetText(model.target);
    if (target) line.append(chip(`target: ${target}`));
    return line;
  }

  function answerBlock(row) {
    const block = make('div', 'answer-block');
    block.append(make('small', '', 'Reference GT'), make('strong', '', row.answer.display));
    return block;
  }

  function cardHeader(row) {
    const head = make('div', 'card-head');
    head.append(make('span', 'task-code', row.task_id.split('_')[0]),
      make('span', 'case-id', row.id), make('span', 'source-chip', row.source_label));
    return head;
  }

  function openCard(row) {
    const card = make('article', 'case-card');
    card.dataset.head = row.head;
    card.dataset.caseId = row.id;
    card.append(cardHeader(row), imageButton(row.initial_image, 'initial RGB', row, 'initial-frame'));
    const content = make('div', 'card-content');
    content.append(make('h2', 'question', row.question), primitiveTimeline(row), answerBlock(row), metadata(row));
    card.append(content);
    const actions = make('div', 'card-actions');
    const details = make('button', 'drawer-button', '查看完整证据 →');
    details.type = 'button';
    details.addEventListener('click', () => openDrawer(row));
    actions.append(details); card.append(actions);
    return card;
  }

  function c1Card(row) {
    const card = make('article', 'case-card case-card--c1');
    card.dataset.head = row.head; card.dataset.caseId = row.id;
    card.append(cardHeader(row));
    const content = make('div', 'card-content');
    content.append(make('h2', 'question', row.question), primitiveTimeline(row), metadata(row));
    card.append(content);
    const layout = make('div', 'c1-layout');
    layout.append(imageButton(row.initial_image, 'initial RGB', row, 'c1-initial'));
    const choices = make('div', 'c1-choices');
    row.choices.forEach(choice => {
      const button = imageButton(choice, choice.id, row,
        `choice-image${choice.is_correct ? ' choice-correct' : ''}`);
      button.append(make('span', 'choice-label', choice.id));
      choices.append(button);
    });
    layout.append(choices); card.append(layout);
    const actions = make('div', 'card-actions');
    const details = make('button', 'drawer-button', '查看完整证据 →');
    details.type = 'button'; details.addEventListener('click', () => openDrawer(row));
    actions.append(details); card.append(actions);
    return card;
  }

  function updateAnswerFilter(rows) {
    const select = $('#filter-answer');
    const values = [...new Set(rows.map(row => row.answer.display))]
      .sort((a,b) => a.localeCompare(b, undefined, {numeric:true}));
    const prior = state.answer;
    select.replaceChildren(); option(select, '', values.length > 32 ? 'GT 类别过多（用搜索）' : '全部 GT');
    if (values.length <= 32) values.forEach(value => option(select, value, value));
    select.disabled = values.length > 32;
    if (!values.includes(prior) || values.length > 32) state.answer = '';
    select.value = state.answer;
  }

  function updateTaskCounts() {
    const rows = baseFiltered(true);
    const counts = {};
    rows.forEach(row => { counts[row.task_id] = (counts[row.task_id] || 0) + 1; });
    $$('.task-tab').forEach(button => {
      const task = button.dataset.taskFilter;
      const count = task === ALL ? rows.length : (counts[task] || 0);
      $('span', button).textContent = count;
      button.setAttribute('aria-pressed', String(state.task === task));
    });
  }

  function render() {
    document.body.classList.toggle('gt-hidden', !state.showGT);
    $('#toggle-gt').textContent = state.showGT ? '隐藏 GT' : '显示 GT';
    const taskRows = baseFiltered(true).filter(row => state.task === ALL || row.task_id === state.task);
    updateAnswerFilter(taskRows);
    currentRows = sortRows(baseFiltered());
    const pageSize = currentPageSize();
    const pages = Math.max(1, Math.ceil(currentRows.length / pageSize));
    state.page = Math.min(Math.max(1, state.page), pages);
    const start = (state.page - 1) * pageSize;
    const visible = currentRows.slice(start, start + pageSize);
    const grid = $('#case-grid'); grid.replaceChildren();
    if (!visible.length) {
      const empty = make('div', 'empty-state');
      empty.append(make('strong', '', '没有符合条件的 case'), make('p', '', '请减少筛选条件，或点击“重置”。'));
      grid.append(empty);
    } else {
      const fragment = document.createDocumentFragment();
      visible.forEach(row => fragment.append(row.response_mode === 'image_mcq' ? c1Card(row) : openCard(row)));
      grid.append(fragment);
    }
    $('#result-count').textContent = `${currentRows.length.toLocaleString()} 个 case`;
    $('#result-range').textContent = currentRows.length ? `显示 ${start + 1}–${Math.min(start + pageSize, currentRows.length)}` : '显示 0';
    $('#page-label').textContent = `${state.page} / ${pages}`;
    $('#page-prev').disabled = state.page <= 1;
    $('#page-next').disabled = state.page >= pages;
    $('#page-size').value = String(pageSize);
    updateTaskCounts(); writeHash();
  }

  function detailGrid(values) {
    const list = make('dl', 'detail-grid');
    values.forEach(([key, value]) => {
      list.append(make('dt', '', key), make('dd', '', value === undefined || value === null || value === '' ? '—' : typeof value === 'object' ? JSON.stringify(value) : value));
    });
    return list;
  }

  function openDrawer(row) {
    lastFocus = document.activeElement;
    state.selectedCase = row.id; writeHash();
    const drawer = $('#case-drawer');
    $('#drawer-title').textContent = `${row.task_id.split('_')[0]} · ${row.id}`;
    const body = $('#drawer-body'); body.replaceChildren();
    body.append(imageButton(row.initial_image, 'initial RGB', row, 'drawer-image'),
      make('h2', 'question', row.question), primitiveTimeline(row), answerBlock(row));
    if (row.choices) {
      const choiceGrid = make('div', 'drawer-c1');
      row.choices.forEach(choice => choiceGrid.append(imageButton(choice, choice.id, row,
        `choice-image${choice.is_correct ? ' choice-correct' : ''}`)));
      body.append(choiceGrid);
    }
    const publicSection = make('section', 'detail-section');
    publicSection.append(make('h3', '', '公开输入'), detailGrid([
      ['Scene', row.scene_id], ['Source', row.source_label],
      ['Actions', actionsText(row.model_input.actions)],
      ['Target', targetText(row.model_input.target)],
      ['Body radius', `${row.model_input.body_radius_m} m`],
      ['Camera height', `${row.model_input.camera_height_above_visible_floor_m} m`],
      ['HFOV / VFOV', `${row.model_input.hfov_deg}° / ${row.model_input.vfov_deg}°`]
    ]));
    const authority = make('section', 'detail-section');
    authority.append(make('h3', '', 'GT 与绑定'), detailGrid([
      ['Readable GT', row.answer.display], ['Raw answer', row.answer.raw],
      ['Atom ref', row.atom_ref], ['Frame ID', row.oracle_ref.frame_id],
      ['Outcome ID', row.oracle_ref.outcome_id], ['Oracle refs', row.oracle_ref],
      ['Technical', row.technical]
    ]));
    body.append(publicSection, authority);
    $('#dialog-backdrop').classList.add('is-open');
    drawer.classList.add('is-open'); drawer.setAttribute('aria-hidden', 'false'); drawer.focus();
  }

  function closeDrawer() {
    const drawer = $('#case-drawer');
    drawer.classList.remove('is-open'); drawer.setAttribute('aria-hidden', 'true');
    $('#dialog-backdrop').classList.remove('is-open');
    state.selectedCase = null; writeHash();
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function caseImages(row) {
    return [{...row.initial_image, role: 'initial RGB'}, ...(row.choices || []).map(choice => ({...choice, role: choice.id}))];
  }

  function applyTransform() {
    $('#lightbox-image').style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    $('#zoom-value').textContent = `${Math.round(zoom * 100)}%`;
  }

  function showLightboxItem() {
    const item = lightboxItems[lightboxIndex];
    const image = $('#lightbox-image'); image.src = item.path; image.alt = item.role;
    $('#lightbox-title').textContent = `${item.role} · ${item.sha256 || ''}`;
    $('#lightbox-index').textContent = `${lightboxIndex + 1} / ${lightboxItems.length}`;
    zoom = 1; panX = 0; panY = 0; applyTransform();
  }

  function openLightbox(row, role) {
    lastFocus = document.activeElement;
    lightboxItems = caseImages(row);
    lightboxIndex = Math.max(0, lightboxItems.findIndex(item => item.role === role));
    $('#image-lightbox').classList.add('is-open');
    $('#image-lightbox').setAttribute('aria-hidden', 'false');
    showLightboxItem(); $('#lightbox-close').focus();
  }

  function closeLightbox() {
    $('#image-lightbox').classList.remove('is-open');
    $('#image-lightbox').setAttribute('aria-hidden', 'true');
    $('#lightbox-image').removeAttribute('src');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function stepLightbox(delta) {
    lightboxIndex = (lightboxIndex + delta + lightboxItems.length) % lightboxItems.length;
    showLightboxItem();
  }

  function bindFilters() {
    populateSelect('#filter-source', discreteValues(row => row.source_label), '全部来源');
    populateSelect('#filter-scene', discreteValues(row => row.scene_id), '全部场景');
    populateSelect('#filter-primitives', discreteValues(row => (row.model_input.actions || []).length), '全部长度');
    populateSelect('#filter-radius', discreteValues(row => row.model_input.body_radius_m), '全部半径');
    populateSelect('#filter-fov', discreteValues(row => row.model_input.hfov_deg), '全部 HFOV');
    $('#filter-query').value = state.query;
    for (const key of ['source', 'scene', 'primitives', 'radius', 'fov', 'sort']) {
      const input = $(`#filter-${key}`); if (input) input.value = state[key];
    }
    $$('.task-tab').forEach(button => button.addEventListener('click', () => {
      state.task = button.dataset.taskFilter; state.page = 1; state.answer = ''; render();
    }));
    $('#filter-query').addEventListener('input', event => { state.query = event.target.value; state.page = 1; render(); });
    for (const key of ['source', 'scene', 'primitives', 'radius', 'fov', 'answer', 'sort']) {
      $(`#filter-${key}`).addEventListener('change', event => { state[key] = event.target.value; state.page = 1; render(); });
    }
    $('#page-size').addEventListener('change', event => { state.pageSize = Number(event.target.value); state.page = 1; render(); });
    $('#reset-filters').addEventListener('click', () => {
      Object.assign(state, {task: ALL, query: '', source: '', scene: '', primitives: '', radius: '', fov: '', answer: '', sort: 'artifact', page: 1, pageSize: null});
      bindStateToControls(); render();
    });
    $('#toggle-gt').addEventListener('click', () => { state.showGT = !state.showGT; render(); });
    $('#page-prev').addEventListener('click', () => { state.page -= 1; render(); scrollTo({top: 0, behavior: 'smooth'}); });
    $('#page-next').addEventListener('click', () => { state.page += 1; render(); scrollTo({top: 0, behavior: 'smooth'}); });
  }

  function bindStateToControls() {
    $('#filter-query').value = state.query;
    for (const key of ['source', 'scene', 'primitives', 'radius', 'fov', 'sort']) {
      $(`#filter-${key}`).value = state[key];
    }
  }

  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#dialog-backdrop').addEventListener('click', closeDrawer);
  $('#lightbox-close').addEventListener('click', closeLightbox);
  $('#lightbox-prev').addEventListener('click', () => stepLightbox(-1));
  $('#lightbox-next').addEventListener('click', () => stepLightbox(1));
  $('#zoom-in').addEventListener('click', () => { zoom = Math.min(6, zoom + .25); applyTransform(); });
  $('#zoom-out').addEventListener('click', () => { zoom = Math.max(.5, zoom - .25); applyTransform(); });
  $('#zoom-fit').addEventListener('click', () => { zoom = 1; panX = 0; panY = 0; applyTransform(); });
  $('#lightbox-stage').addEventListener('wheel', event => {
    event.preventDefault(); zoom = Math.max(.5, Math.min(6, zoom + (event.deltaY < 0 ? .2 : -.2))); applyTransform();
  }, {passive: false});
  $('#lightbox-stage').addEventListener('pointerdown', event => {
    dragging = true; dragStart = {x: event.clientX - panX, y: event.clientY - panY};
    event.currentTarget.setPointerCapture(event.pointerId); event.currentTarget.classList.add('is-dragging');
  });
  $('#lightbox-stage').addEventListener('pointermove', event => {
    if (!dragging) return; panX = event.clientX - dragStart.x; panY = event.clientY - dragStart.y; applyTransform();
  });
  $('#lightbox-stage').addEventListener('pointerup', event => {
    dragging = false; event.currentTarget.classList.remove('is-dragging');
  });
  $('#lightbox-image').addEventListener('dblclick', () => { zoom = zoom === 1 ? 2 : 1; panX = 0; panY = 0; applyTransform(); });
  document.addEventListener('keydown', event => {
    if ($('#image-lightbox').classList.contains('is-open')) {
      if (event.key === 'Escape') closeLightbox();
      if (event.key === 'ArrowLeft') stepLightbox(-1);
      if (event.key === 'ArrowRight') stepLightbox(1);
      return;
    }
    if (event.key === 'Escape' && $('#case-drawer').classList.contains('is-open')) closeDrawer();
    if (event.key === '/' && !['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) {
      event.preventDefault(); $('#filter-query').focus();
    }
  });
  window.addEventListener('hashchange', () => { parseHash(); bindStateToControls(); render(); });

  parseHash(); bindFilters(); render();
  if (state.selectedCase) {
    const selected = payload.cases.find(row => row.id === state.selectedCase);
    if (selected) openDrawer(selected);
  }
})();
"""


_CHECKPOINT_DIRECTION_SCRIPT = r"""
(() => {
  'use strict';
  const payload = JSON.parse(document.getElementById('case-browser-data').textContent);
  const cases = new Map(payload.cases
    .filter(row => row.task_id === 'A4_checkpoint_direction')
    .map(row => [row.id, row]));
  const grid = document.getElementById('case-grid');
  function decorate() {
    grid.querySelectorAll('.case-card[data-case-id]').forEach(card => {
      const row = cases.get(card.dataset.caseId);
      if (!row || card.dataset.a4Decorated) return;
      const checkpoint = row.model_input.checkpoint;
      const primitives = card.querySelectorAll('.primitive');
      const selected = primitives[Number(checkpoint.action_index) - 1];
      if (selected) selected.classList.add('primitive--answer');
      const metadata = card.querySelector('.meta-row');
      if (metadata) {
        const chip = document.createElement('span');
        chip.className = 'meta-chip';
        chip.textContent = `checkpoint: action ${checkpoint.action_index} · ${Number(checkpoint.fraction) * 100}%`;
        metadata.append(chip);
      }
      card.dataset.a4Decorated = 'true';
    });
  }
  new MutationObserver(decorate).observe(grid, {childList: true});
  decorate();
})();
"""


_DATASET_SCRIPT = r"""
(() => {
  'use strict';
  const select = document.getElementById('filter-source');
  const buttons = Array.from(document.querySelectorAll('.dataset-switch'));
  if (!select || !buttons.length) return;
  const payloadNode = document.getElementById('case-browser-data');
  const datasetPayload = payloadNode ? JSON.parse(payloadNode.textContent) : {};
  const datasets = Array.isArray(datasetPayload.datasets) ? datasetPayload.datasets : [];
  datasets.forEach(dataset => {
    if (!dataset.label || Array.from(select.options).some(option => option.value === dataset.label)) return;
    const option = document.createElement('option');
    option.value = dataset.label; option.textContent = dataset.label; select.append(option);
  });
  function decorateEmptyState() {
    const dataset = datasets.find(row =>
      row.label === select.value && Number(row.case_count) === 0);
    const empty = document.querySelector('#case-grid .empty-state');
    if (!dataset || !empty) return;
    const title = empty.querySelector('strong');
    const note = empty.querySelector('p');
    if (title) title.textContent = `${dataset.label} 暂无认证 QA`;
    if (note && dataset.availability_note) note.textContent = dataset.availability_note;
  }
  function syncDatasetButtons() {
    buttons.forEach(button => button.setAttribute(
      'aria-pressed', String(button.dataset.datasetSource === select.value)));
  }
  buttons.forEach(button => button.addEventListener('click', () => {
    select.value = button.dataset.datasetSource;
    select.dispatchEvent(new Event('change', {bubbles: true}));
    syncDatasetButtons(); decorateEmptyState();
  }));
  select.addEventListener('change', () => {
    syncDatasetButtons(); setTimeout(decorateEmptyState);
  });
  window.addEventListener('hashchange', () => setTimeout(() => {
    syncDatasetButtons(); decorateEmptyState();
  }));
  document.addEventListener('click', () => setTimeout(decorateEmptyState));
  document.addEventListener('input', () => setTimeout(decorateEmptyState));
  const reset = document.getElementById('reset-filters');
  if (reset) reset.addEventListener('click', () => setTimeout(syncDatasetButtons));
  const hashSource = new URLSearchParams(location.hash.replace(/^#/, '')).get('source');
  if (hashSource && datasets.some(dataset => dataset.label === hashSource)) {
    select.value = hashSource;
    select.dispatchEvent(new Event('change', {bubbles: true}));
  }
  syncDatasetButtons(); decorateEmptyState();
})();
"""


def _safe_json_for_html(value: object) -> str:
    """Serialize JSON without permitting data to terminate its script tag."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        sort_keys=True).replace("<", "\\u003c").replace(
            ">", "\\u003e").replace("&", "\\u0026")


def _task_buttons(payload: dict) -> str:
    counts = payload["coverage"]
    task_ids = tuple(counts)
    values = [
        ('all', '全部任务', sum(counts.values()), '', 'true'),
    ]
    labels = {
        "A1_collision": "A1 碰撞",
        "A2_collision_step_grounding": "A2 碰撞动作",
        "A3_contact_object": "A3 接触对象",
        "A4_checkpoint_direction": "A4 过程方位",
        "B1_endpoint_distance": "B1 终点距离",
        "B2_endpoint_direction": "B2 终点方向",
        "C1_future_view_selection": "C1 终点视野",
    }
    for task_id in task_ids:
        values.append((
            task_id, labels.get(task_id, task_id), counts[task_id],
            task_id[:1], 'false'))
    return "".join(
        f'<button class="task-tab" type="button" data-head="{head}" '
        f'data-task-filter="{html.escape(task_id)}" '
        f'aria-pressed="{pressed}">{html.escape(label)}'
        f'<span>{count}</span></button>'
        for task_id, label, count, head, pressed in values)


def _dataset_switcher(payload: dict) -> str:
    """Render the opt-in dataset rail over the existing source filter."""
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset browser payload has no datasets")
    values = [{
        "id": "all",
        "label": "全部数据集",
        "case_count": len(payload["cases"]),
        "status": "candidate",
        "coverage": payload["coverage"],
        "source": "",
    }]
    values.extend({**dataset, "source": dataset["label"]}
                  for dataset in datasets)
    buttons = []
    for value in values:
        c1_count = int(value["coverage"].get(C1_TASK_ID, 0))
        a4_count = int(value["coverage"].get(
            CHECKPOINT_DIRECTION_TASK_ID, 0))
        status = str(value.get("status") or "candidate")
        detail = "全部任务"
        if value["id"] != "all":
            detail = (
                '<span class="pending">C1 Pending</span>'
                if c1_count == 0 else f"C1 {c1_count}")
            if a4_count:
                detail += f" · A4 {a4_count}"
            if status == "pilot":
                detail = f"Pilot · {detail}"
            elif status == "collecting":
                detail = "收集中 · 暂无认证 QA"
            elif status == "unavailable":
                detail = "当前协议不可用"
        buttons.append(
            '<button class="dataset-switch" type="button" '
            f'data-dataset-id="{html.escape(str(value["id"]))}" '
            f'data-dataset-source="{html.escape(str(value["source"]))}" '
            f'aria-pressed="{str(value["id"] == "all").lower()}">'
            f'<strong>{html.escape(str(value["label"]))}</strong>'
            f'<b>{int(value["case_count"]):,}</b>'
            f'<small>{detail}</small></button>')
    return (
        '<section class="dataset-rail" aria-label="切换数据集">'
        '<div id="dataset-switcher" class="dataset-switcher">'
        + "".join(buttons) + '</div></section>\n')


def render_case_browser_payload(
        payload: dict, report_root: Path, *, dataset_mode: bool = False
        ) -> Path:
    """Write one dependency-free browser from an already projected payload."""
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    source_text = " · ".join(
        f'{source["label"]}={source["case_count"]}'
        for source in payload["sources"])
    dataset_nav = _dataset_switcher(payload) if dataset_mode else ""
    style_extra = _DATASET_STYLE if dataset_mode else ""
    script_extra = _DATASET_SCRIPT if dataset_mode else ""
    source_filter_label = "数据集" if dataset_mode else "来源"
    has_checkpoint_direction = bool(
        payload["coverage"].get(CHECKPOINT_DIRECTION_TASK_ID))
    browser_script = _SCRIPT + (
        _CHECKPOINT_DIRECTION_SCRIPT if has_checkpoint_direction else "")
    diagnostic_status = (
        '\n    <span class="status-pill status-pill--warn">'
        'A4 diagnostic</span>' if has_checkpoint_direction else '')
    document = f'''<!doctype html>
<html lang="zh-CN" data-browser-schema="{BROWSER_SCHEMA}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>EgoConseq · 全量 Case Browser</title>
<style>{_STYLE}{style_extra}</style>
</head>
<body>
<header class="topbar">
  <div class="brand"><div class="brand-mark">EC</div><div>
    <h1>EgoConseq · 全量 Case Browser</h1>
    <p>{html.escape(source_text)} · 共 {len(payload["cases"]):,} 个 case</p>
  </div></div>
  <div class="status-stack">
    <span class="status-pill">只读 GT atlas</span>
    <span class="status-pill">candidate-only</span>{diagnostic_status}
  </div>
</header>
{dataset_nav}<nav class="workspace-nav" aria-label="按任务筛选"><div class="task-tabs">
{_task_buttons(payload)}
</div></nav>
<section class="toolbar-wrap" aria-label="筛选和排序"><div class="toolbar">
  <div class="field field--search"><label for="filter-query">搜索</label><input id="filter-query" type="search" placeholder="Case ID、场景、问题、动作、目标或 GT（快捷键 /）"></div>
  <div class="field"><label for="filter-source">{source_filter_label}</label><select id="filter-source"></select></div>
  <div class="field"><label for="filter-scene">场景</label><select id="filter-scene"></select></div>
  <div class="field"><label for="filter-primitives">动作数</label><select id="filter-primitives"></select></div>
  <div class="field"><label for="filter-radius">Body radius</label><select id="filter-radius"></select></div>
  <div class="field"><label for="filter-fov">HFOV</label><select id="filter-fov"></select></div>
  <div class="field"><label for="filter-answer">GT</label><select id="filter-answer"></select></div>
  <div class="field"><label for="filter-sort">排序</label><select id="filter-sort"><option value="artifact">Artifact 顺序</option><option value="scene">Scene</option><option value="actions">动作长度</option><option value="answer">GT</option></select></div>
  <button id="reset-filters" class="reset-button" type="button">重置</button>
</div></section>
<section class="result-bar"><div><strong id="result-count">0 个 case</strong> · <span id="result-range">显示 0</span></div><div class="legend"><span><i class="a"></i>A 轨迹交互</span><span><i class="b"></i>B 终点关系</span><span><i class="c"></i>C 终点观察</span><button id="toggle-gt" class="reset-button" type="button">隐藏 GT</button><label>每页 <select id="page-size"><option>12</option><option selected>24</option><option>48</option></select></label></div></section>
<main id="case-grid" class="case-grid" aria-live="polite"></main>
<div class="pager"><button id="page-prev" type="button">← 上一页</button><span id="page-label">1 / 1</span><button id="page-next" type="button">下一页 →</button></div>
<div id="dialog-backdrop" class="dialog-backdrop"></div>
<aside id="case-drawer" class="case-drawer" role="dialog" aria-modal="true" aria-hidden="true" tabindex="-1" aria-labelledby="drawer-title">
  <div class="drawer-head"><strong id="drawer-title">Case</strong><button id="drawer-close" class="icon-button" type="button" aria-label="关闭详情">×</button></div>
  <div id="drawer-body" class="drawer-body"></div>
</aside>
<div id="image-lightbox" class="lightbox" role="dialog" aria-modal="true" aria-hidden="true" aria-label="图片放大查看">
  <div class="lightbox-toolbar"><button id="lightbox-close" class="icon-button" type="button" aria-label="关闭大图">×</button><div id="lightbox-title" class="lightbox-title"></div><button id="lightbox-prev" class="icon-button" type="button" aria-label="上一张">←</button><button id="zoom-out" class="icon-button" type="button" aria-label="缩小">−</button><button id="zoom-fit" class="icon-button" type="button">适应</button><span id="zoom-value">100%</span><button id="zoom-in" class="icon-button" type="button" aria-label="放大">＋</button><button id="lightbox-next" class="icon-button" type="button" aria-label="下一张">→</button></div>
  <div id="lightbox-stage" class="lightbox-stage"><img id="lightbox-image" class="lightbox-image" alt=""></div>
  <div id="lightbox-index" class="lightbox-foot"></div>
</div>
<template id="case-template"><div class="action-timeline"></div></template>
<script id="case-browser-data" type="application/json">{_safe_json_for_html(payload)}</script>
<script>{browser_script}{script_extra}</script>
</body></html>'''
    output = report_root / "index.html"
    io_utils.atomic_write_text(output, document)
    return output


def render_case_browser(sources: Sequence[BrowserSource],
                        report_root: Path) -> Path:
    """Write a dependency-free, offline browser for every source case."""
    report_root = Path(report_root)
    payload = build_browser_payload(sources, report_root=report_root)
    return render_case_browser_payload(payload, report_root)
