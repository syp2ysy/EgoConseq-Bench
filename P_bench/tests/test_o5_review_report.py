"""Tests for O5 review/report scripts under the single-robot group schema."""

from pathlib import Path

from egoconseq.manifest import Case
from scripts.build_o5_review import build_html
from scripts.run_baselines_report import cases_to_o5_groups, summarize_cases


def make_case(
    case_id: str,
    group_id: str,
    radius_m: float,
    label: str,
    image_path: str,
    d_safe_m: float,
    horizon_m: float,
    group_kind: str,
) -> Case:
    return Case(
        case_id=case_id,
        operation_id="O5",
        readout_tag="binary_contact",
        group_id=group_id,
        scene_id="scene-a",
        image_path=image_path,
        question=(
            f"图中是一个圆柱形底盘的机器人，底盘直径约 {2 * radius_m:.1f} 米。\n"
            f"它从当前位置朝正前方移动约 {horizon_m:.1f} 米。\n"
            "请直接判断它的身体会不会发生接触。"
        ),
        body={"radius_m": radius_m, "diameter_m": 2 * radius_m},
        action={
            "type": "forward",
            "horizon_m": horizon_m,
            "horizon_reference": "metric_fixed",
        },
        d_safe_visible_m=d_safe_m,
        d_safe_navmesh_m=d_safe_m + 0.1,
        answer={
            "answer_type": "binary_contact",
            "options": ["contact", "no_contact"],
            "label": label,
        },
        tags={
            "group_kind": group_kind,
            "geometry_tag": "narrow-gap" if group_kind == "flip" else group_kind,
            "gt_evidence": {
                "d_safe_m": d_safe_m,
                "horizon_m": horizon_m,
                "rule": "contact iff d_safe < H",
            },
        },
    )


def test_review_html_uses_relative_images_and_shows_binary_gt(tmp_path):
    img_dir = tmp_path / "img"
    topdown_dir = tmp_path / "topdown"
    img_dir.mkdir()
    topdown_dir.mkdir()
    (img_dir / "c-small.png").write_bytes(b"not-a-real-png")
    (topdown_dir / "c-small.png").write_bytes(b"not-a-real-png")

    case = make_case(
        case_id="c-small",
        group_id="g1",
        radius_m=0.10,
        label="no_contact",
        image_path=str(img_dir / "c-small.png"),
        d_safe_m=2.0,
        horizon_m=1.4,
        group_kind="flip",
    )

    out_path = tmp_path / "review.html"
    build_html([case], str(out_path))
    html = out_path.read_text(encoding="utf-8")

    assert "data:image" not in html
    assert 'src="img/c-small.png"' in html
    assert 'src="topdown/c-small.png"' in html
    assert "binary_contact" in html
    assert "GT: 不会接触" in html
    assert "因为 d_safe=2.000 &gt;= H=1.400" in html
    assert "两个" not in html
    assert "会接触 / 不会接触" not in html


def test_report_groups_cases_by_counterfactual_group():
    cases = [
        make_case("g1-small", "g1", 0.10, "no_contact", "img/a.png", 2.0, 1.4, "flip"),
        make_case("g1-large", "g1", 0.40, "contact", "img/a.png", 0.6, 1.4, "flip"),
        make_case("g2-small", "g2", 0.10, "no_contact", "img/b.png", 2.0, 1.0, "all_no_contact"),
        make_case("g2-large", "g2", 0.40, "no_contact", "img/b.png", 1.8, 1.0, "all_no_contact"),
    ]

    groups = cases_to_o5_groups(cases)
    summary = summarize_cases(cases)

    assert len(groups) == 2
    assert [len(group) for group in groups] == [2, 2]
    assert groups[0][0]["radius_m"] == 0.10
    assert {case["gt"] for case in groups[0]} == {"contact", "no_contact"}
    assert summary["case_label_counts"]["contact"] == 1
    assert summary["case_label_counts"]["no_contact"] == 3
    assert summary["group_kind_counts"]["flip"] == 1
    assert summary["group_kind_counts"]["all_no_contact"] == 1
