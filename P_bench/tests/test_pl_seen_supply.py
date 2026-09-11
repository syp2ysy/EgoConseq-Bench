from dataclasses import replace

from post_QA.seen_build import selection, supply
from tests.test_pl_seen_selection import _candidate


def test_supply_reports_every_slot_shortfall_at_once():
    rows = [
        _candidate(0, starts_with="forward"),
        replace(_candidate(1, starts_with="forward"),
                task_id="A4", record_id="same-record"),
    ]
    rows[0] = replace(rows[0], record_id="same-record")
    quotas = {
        ("gs", "A1", "forward", "no_collision"): 1,
        ("gs", "A1", "turn", "collision"): 1,
        ("gs", "A4", "forward"): 1,
    }

    report = supply.analyze_candidates(rows, quotas=quotas, seed=3)

    assert report["feasible"] is False
    deficits = {
        tuple(row["slot"]): row["shortfall"]
        for row in report["slots"] if row["shortfall"]
    }
    assert deficits == {
        ("gs", "A1", "turn", "collision"): 1,
        ("gs", "A4", "forward"): 1,
    }


def test_full_spec_has_no_a2_l3_turn_slot():
    quotas = selection.all_slot_quotas()

    assert not any(
        slot[:4] == (dataset, "A2", 3, "turn")
        for dataset in ("b1k", "gs", "r2r")
        for slot in quotas)
