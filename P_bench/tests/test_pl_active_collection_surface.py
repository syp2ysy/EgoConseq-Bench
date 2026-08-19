"""The public collector exposes only the active three-dataset pipeline."""

from pipeline import capability_contracts, dataset_contracts
from pipeline.collection_cli import build_parser


def test_collection_cli_omits_retired_experiment_modes():
    help_text = build_parser().format_help()
    for option in (
            "--pose-pool-only", "--pose-pool-in",
            "--pose-pool-min-accepted-per-scene",
            "--pose-pool-max-batches", "--body-paired-core",
            "--family-plan", "--family-plan-poses-only"):
        assert option not in help_text


def test_registered_collection_datasets_are_exactly_the_active_three():
    assert dataset_contracts.main_collection_datasets() == (
        "r2r", "gs", "b1k")
    assert capability_contracts.DATASET_ORDER == ("r2r", "b1k", "gs")
