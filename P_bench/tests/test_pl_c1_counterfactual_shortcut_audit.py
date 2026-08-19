import pytest

from pipeline import c1_counterfactual_shortcut_audit
from tests.test_pl_task3_authority import _counterfactual_artifact


def _rows():
    return [
        {
            "source_dataset": "r2r", "scene_id": "s1",
            "answer_position": 1,
            "initial_similarity_prediction": 1,
            "visual_medoid_prediction": 2,
            "motion_prediction": 1,
        },
        {
            "source_dataset": "r2r", "scene_id": "s1",
            "answer_position": 3,
            "initial_similarity_prediction": 1,
            "visual_medoid_prediction": 3,
            "motion_prediction": 3,
        },
        {
            "source_dataset": "gs", "scene_id": "s2",
            "answer_position": 2,
            "initial_similarity_prediction": 2,
            "visual_medoid_prediction": 2,
            "motion_prediction": 4,
        },
    ]


def test_counterfactual_audit_reports_three_baselines_deterministically():
    report = c1_counterfactual_shortcut_audit.audit_rows(
        _rows(), resamples=40, seed=17)

    assert report["schema"] == \
        "egoconseq.c1-counterfactual-shortcut-audit.v1"
    assert report["item_count"] == 3
    assert report["scene_count"] == 2
    assert report["baselines"]["initial_to_option_similarity"][
        "accuracy"] == 2 / 3
    assert report["baselines"]["options_only_visual_medoid"][
        "accuracy"] == 2 / 3
    assert report["baselines"]["motion_only"]["accuracy"] == 2 / 3
    assert set(report["dataset_strata"]) == {"gs", "r2r"}
    assert report["dataset_strata"]["gs"]["item_count"] == 1
    assert report["baselines"]["initial_to_option_similarity"][
        "scene_clustered_95pct_ci"] == [0.5, 1.0]
    assert c1_counterfactual_shortcut_audit.audit_rows(
        _rows(), resamples=40, seed=17) == report


def test_counterfactual_audit_clusters_scene_ids_within_dataset():
    rows = [
        {
            "source_dataset": dataset, "scene_id": "shared-scene",
            "answer_position": answer,
            "initial_similarity_prediction": prediction,
            "visual_medoid_prediction": prediction,
            "motion_prediction": prediction,
        }
        for dataset, answer, prediction in (
            ("r2r", 1, 1), ("gs", 2, 1))
    ]

    report = c1_counterfactual_shortcut_audit.audit_rows(
        rows, resamples=20, seed=4)

    assert report["scene_count"] == 2
    assert report["baselines"]["initial_to_option_similarity"][
        "scene_clustered_95pct_ci"] is not None


def test_counterfactual_audit_rejects_invalid_predictions():
    rows = _rows()
    rows[0] = dict(rows[0], visual_medoid_prediction=5)
    with pytest.raises(ValueError, match="prediction"):
        c1_counterfactual_shortcut_audit.audit_rows(
            rows, resamples=10, seed=1)


def test_motion_only_uses_leave_one_out_not_self_fitted_action_labels():
    prepared = [
        {"source_dataset": "r2r", "scene_id": "s1",
         "answer_position": 1, "action_key": "unique-a"},
        {"source_dataset": "r2r", "scene_id": "s2",
         "answer_position": 2, "action_key": "unique-b"},
    ]

    predictions = c1_counterfactual_shortcut_audit._motion_predictions(
        prepared)

    assert predictions == [2, 1]


def test_counterfactual_audit_reads_a_validated_artifact(tmp_path):
    artifact, authority = _counterfactual_artifact(tmp_path)

    report = c1_counterfactual_shortcut_audit.audit_artifact(
        artifact, resamples=10, seed=1,
        expected_source_authority=authority)

    assert report["item_count"] >= 1
    assert report["scene_count"] == 1
    assert set(report["baselines"]) == {
        "initial_to_option_similarity",
        "options_only_visual_medoid",
        "motion_only",
    }


def test_counterfactual_audit_refuses_output_inside_benchmark(tmp_path):
    artifact, authority = _counterfactual_artifact(tmp_path)

    with pytest.raises(ValueError, match="outside benchmark"):
        c1_counterfactual_shortcut_audit.write_artifact_audit(
            artifact, artifact / "audit.json", resamples=10, seed=1,
            expected_source_authority=authority)
