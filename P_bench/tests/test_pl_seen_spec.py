from post_QA.seen_build import spec


def test_cli_spec_changes_existing_selector_quotas_without_stale_cache(tmp_path):
    import json
    from post_QA.seen_build import selection
    from scripts.build_seen_benchmark import build_arg_parser

    original = spec.SPEC_PATH
    value = json.loads(original.read_text())
    value["benchmark_id"] = "small-unseen"
    value["dataset_task_totals"]["gs"]["A2"] = 4
    value["a2_start_totals"]["gs"] = {"3": {"forward": 4, "turn": 0}}
    path = tmp_path / "small.json"
    path.write_text(json.dumps(value))
    selection._a2_slot_counts("gs")  # Already imported and cached under seen.
    args = build_arg_parser().parse_args([
        "compile", "--spec", str(path), "--output", str(tmp_path / "qa")])
    try:
        spec.configure(args.spec)
        quotas = selection.all_slot_quotas()
        assert sum(n for slot, n in quotas.items() if slot[:2] == ("gs", "A2")) == 4
        assert selection._slot_quota(("gs", "A2", 3, "forward", 1, "shortest")) == 1
        from post_QA.seen_build import artifact
        qa = tmp_path / "qa.json"
        qa.write_text("[]")
        index = artifact.write_index([], qa, tmp_path / "record_index.json")
        assert index["benchmark"] == "small-unseen"
    finally:
        spec.configure(original)


def test_gs_never_exposes_a3_or_b1():
    assert spec.supported_tasks("gs") == (
        "A1", "A2", "A4", "B2", "C1")
    assert spec.DATASET_TASK_TOTALS["gs"]["A3"] == 0
    assert spec.DATASET_TASK_TOTALS["gs"]["B1"] == 0


def test_a2_never_assigns_turn_first_to_l3():
    for dataset in spec.DATASETS:
        starts = spec.A2_START_TOTALS[dataset]
        assert starts[3]["turn"] == 0
        assert starts[3]["forward"] > 0
