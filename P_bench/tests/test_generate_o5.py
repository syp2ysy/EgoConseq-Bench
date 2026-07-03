"""Tests for O5 demo generation helpers.

The generator must emit single-robot binary-contact cases.  Counterfactual
behavior lives in a group of cases with the same image/pose/action horizon and
different chassis diameters, not in one question that compares two robots.
"""

from scripts.generate_o5 import (
    R_LARGE,
    R_SMALL,
    build_o5_group_cases,
    choose_horizon,
    clear_o5_output_dirs,
    save_topdown_o5,
)


def test_choose_horizon_prefers_flip_when_target_needs_it():
    result = choose_horizon(
        d_safe_small=2.0,
        d_safe_large=0.6,
        target_counts={
            "flip": (10, 0),
            "all_no_contact": (10, 10),
            "all_contact": (10, 10),
        },
    )

    assert result is not None
    horizon_m, group_kind = result
    assert group_kind == "flip"
    assert d_safe_margin_ok(2.0, horizon_m, R_SMALL)
    assert d_safe_margin_ok(0.6, horizon_m, R_LARGE)


def test_build_o5_group_cases_emits_one_binary_case_per_radius():
    cases = build_o5_group_cases(
        group_id="g-o5",
        case_prefix="O5-test",
        scene_id="scene-a",
        pose=[1.0, 0.0, 2.0, 0.5],
        horizon_m=1.4,
        group_kind="flip",
        d_vis_small=2.0,
        d_vis_large=0.6,
        d_nav_small=2.1,
        d_nav_large=0.7,
    )

    assert cases is not None
    assert len(cases) == 2
    assert {c.group_id for c in cases} == {"g-o5"}
    assert {c.readout_tag for c in cases} == {"binary_contact"}
    assert {c.answer["answer_type"] for c in cases} == {"binary_contact"}
    assert {c.answer["label"] for c in cases} == {"contact", "no_contact"}
    assert sorted(c.body["diameter_m"] for c in cases) == [0.2, 0.8]
    assert {c.action["horizon_m"] for c in cases} == {1.4}
    assert {c.action["horizon_reference"] for c in cases} == {"metric_fixed"}

    for case in cases:
        assert "小机器人" not in case.question
        assert "大机器人" not in case.question
        assert "两个" not in case.question
        assert "高" not in case.question
        assert case.tags["group_kind"] == "flip"
        assert case.tags["radii_in_group_m"] == [R_SMALL, R_LARGE]
        assert case.tags["gt_evidence"]["rule"] == "contact iff d_safe < H"


def test_build_o5_group_cases_rejects_ambiguous_member():
    cases = build_o5_group_cases(
        group_id="g-o5",
        case_prefix="O5-test",
        scene_id="scene-a",
        pose=[1.0, 0.0, 2.0, 0.5],
        horizon_m=1.0,
        group_kind="flip",
        d_vis_small=1.05,
        d_vis_large=0.6,
        d_nav_small=1.05,
        d_nav_large=0.6,
    )

    assert cases is None


def test_save_topdown_o5_plots_only_current_robot(monkeypatch, tmp_path):
    """Each top-down evidence image belongs to one question, so draw one body."""
    calls = []

    def fake_plot_body_path(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("scripts.generate_o5._plot_body_path", fake_plot_body_path)

    save_topdown_o5(
        out_path=str(tmp_path / "td.png"),
        obstacle_pts=[],
        path_samples=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.0)],
        d_safe_m=1.2,
        horizon_m=1.0,
        radius_m=0.4,
        gt_label="no_contact",
        title="single case",
    )

    assert len(calls) == 1


def test_clear_o5_output_dirs_removes_stale_pngs(tmp_path):
    img_dir = tmp_path / "img"
    topdown_dir = tmp_path / "topdown"
    img_dir.mkdir()
    topdown_dir.mkdir()
    (img_dir / "old.png").write_bytes(b"old")
    (topdown_dir / "old.png").write_bytes(b"old")
    (topdown_dir / "keep.txt").write_text("keep", encoding="utf-8")

    clear_o5_output_dirs(str(img_dir), str(topdown_dir))

    assert not (img_dir / "old.png").exists()
    assert not (topdown_dir / "old.png").exists()
    assert (topdown_dir / "keep.txt").exists()


def d_safe_margin_ok(d_safe_m: float, horizon_m: float, radius_m: float) -> bool:
    return abs(d_safe_m - horizon_m) >= radius_m
