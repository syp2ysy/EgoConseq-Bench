import json
from pathlib import Path

from post_QA.seen_build import records_update as update


def test_records_chain_repairs_and_restarts_without_building_qa(tmp_path, monkeypatch):
    work = tmp_path / "legacy"
    (work / "tmp").mkdir(parents=True)
    output = tmp_path / "updated"
    collections = []
    monkeypatch.setattr(update, "_collect", lambda *args, **kwargs: collections.append(args[0]))

    def materialize(plan, work, target):
        target.mkdir(parents=True)
        (target / "manifest.json").write_text('{"record_count": 2}')

    monkeypatch.setattr(update.release, "materialize_records", materialize)
    monkeypatch.setattr(update.inventory, "sources_from_manifest", lambda path: path)

    def plan(sources, path, **kwargs):
        path.parent.mkdir(parents=True)
        path.write_text("plan")

    monkeypatch.setattr(update.inventory, "write_plan", plan)
    monkeypatch.setattr(update.selection, "enumerate_candidates", lambda path, **kwargs: path)
    monkeypatch.setattr(update.supply, "analyze_candidates", lambda path, **kwargs: {
        "feasible": path.name != "records-000",
        "shortfall_total": 1 if path.name == "records-000" else 0})

    def forbidden(*args, **kwargs):
        raise AssertionError("records update must never build or publish benchmark")

    for name in ("compile_records", "materialize_benchmark"):
        monkeypatch.setattr(update.release, name, forbidden)
    options = dict(python=Path("python"), b1k_python=Path("behavior/python"),
                   b1k_data_root=tmp_path / "assets")
    result = update.update_records(work / "plan.jsonl", work, output, **options)
    assert result["phase"] == "complete"
    assert (output / "current").resolve().name == "records-001"
    assert not (work / "tmp").exists()
    assert not list((output / "work").iterdir())
    assert len(collections) == 2
    assert update.update_records(work / "plan.jsonl", work, output, **options) == result
    assert len(collections) == 2
    assert json.loads((output / "supply.json").read_text())["shortfall_total"] == 0


def test_explicit_benchmark_output_chains_compile_after_record_merge(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(update, "_collect", lambda *a, **k: events.append("collect"))

    def materialize(plan, work, target):
        events.append("merge")
        target.mkdir(parents=True)
        (target / "manifest.json").write_text('{"record_count": 2}')

    monkeypatch.setattr(update.release, "materialize_records", materialize)
    monkeypatch.setattr(update.release, "compile_records",
                        lambda *a, **k: events.append("compile") or {"qa_items": 2})
    result = update.update_records(
        tmp_path / "plan.jsonl", tmp_path / "work", tmp_path / "updated",
        python=Path("python"), b1k_python=Path("behavior/python"),
        b1k_data_root=tmp_path / "assets", benchmark_output=tmp_path / "benchmark")
    assert events == ["collect", "merge", "compile"]
    assert result["phase"] == "complete"
    assert result["benchmark"] == str(tmp_path / "benchmark")


def test_surface_refinement_switches_once_without_supply_or_benchmark(
        tmp_path, monkeypatch):
    output = tmp_path / "seen_updates"
    current = output / "versions" / "records-002"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text('{"record_count": 3}')
    (output / "current").symlink_to(current.relative_to(output),
                                     target_is_directory=True)
    work = output / "work" / "surfaces"
    calls = []

    monkeypatch.setattr(
        update.inventory, "sources_from_manifest",
        lambda root: [("gs", Path(root) / "gs/records.jsonl")])

    def plan(_sources, path, **kwargs):
        calls.append(("plan", kwargs["seed"]))
        path.parent.mkdir(parents=True)
        path.write_text("plan")
        return {"records": 3}

    monkeypatch.setattr(update.inventory, "write_surface_plan", plan)
    monkeypatch.setattr(
        update, "_collect",
        lambda plan, work_dir, logs, **kwargs:
            calls.append(("collect", plan, work_dir, logs, kwargs["seed"])))

    def materialize(_plan, _work, target):
        calls.append(("materialize", target.name))
        target.mkdir(parents=True)
        (target / "manifest.json").write_text('{"record_count": 3}')
        return {"record_count": 3, "surface": {"target_records": 2}}

    monkeypatch.setattr(update.release, "materialize_records", materialize)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("surface refinement must not enter supply/build loops")

    monkeypatch.setattr(update.supply, "analyze_candidates", forbidden)
    monkeypatch.setattr(update.selection, "enumerate_candidates", forbidden)
    for name in ("compile_records", "materialize_benchmark"):
        monkeypatch.setattr(update.release, name, forbidden)

    result = update.refine_surfaces(
        current, output, work,
        python=Path("python"), b1k_python=Path("behavior/python"),
        b1k_data_root=tmp_path / "b1k", seed=20260906)

    assert result["phase"] == "complete"
    assert (output / "current").resolve().name == "records-003"
    assert sorted(path.name for path in output.joinpath("versions").iterdir()) == [
        "records-002", "records-003"]
    assert not work.exists()
    assert [call[0] for call in calls] == ["plan", "collect", "materialize"]
