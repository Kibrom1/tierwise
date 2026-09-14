"""Replay: what different cuts would have done to a log already written."""

import json

import pytest

from tierwise import ReplayResult, Thresholds, replay
from tierwise.cli import main

BASE_TS = 1_700_000_000.0

EASY = {"category": "typo", "file_count": 1, "lines_changed": 3}            # 0.00
MID = {"category": "feature", "file_count": 3, "lines_changed": 90,
       "dependency_depth": 1, "requires_context": True}                     # 0.47
HARD = {"category": "architecture", "file_count": 30, "lines_changed": 2000,
        "dependency_depth": 6, "requires_context": True, "ambiguity": 0.9}  # 1.00


def write(path, rows, mode="w"):
    """rows: (signals, tier, source, outcome, cost)."""
    with path.open(mode, encoding="utf-8") as handle:
        for index, (signals, tier, source, outcome, cost) in enumerate(rows):
            handle.write(json.dumps({
                "event": "routing_decision", "decision_id": f"d{index}", "tier": tier,
                "source": source, "attempt": 1, "confidence": 0.9, "signals": signals,
            }) + "\n")
            if outcome:
                handle.write(json.dumps({
                    "event": "task_outcome", "decision_id": f"d{index}",
                    "outcome": outcome, "cost_usd": cost, "timestamp": BASE_TS + index,
                }) + "\n")


@pytest.fixture
def log(tmp_path):
    return tmp_path / "routing.jsonl"


def test_unchanged_thresholds_change_nothing(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 5)
    result = replay(log, Thresholds())
    assert result.considered == 5
    assert result.changed == 0
    assert result.tier_before == result.tier_after == {"medium": 5}


def test_lowering_a_cut_routes_work_higher(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 5)
    result = replay(log, Thresholds(low_medium=0.20, medium_high=0.40))
    assert result.changed == 5
    assert result.tier_after == {"high": 5}


def test_raising_a_cut_routes_work_lower(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 5)
    result = replay(log, Thresholds(low_medium=0.55, medium_high=0.80))
    assert result.tier_after == {"low": 5}


def test_it_reports_failures_that_would_have_gone_higher(log):
    write(log, [(MID, "medium", "heuristic", "insufficient", 0.02)] * 4
               + [(EASY, "low", "heuristic", "success", 0.004)] * 4)
    result = replay(log, Thresholds(low_medium=0.20, medium_high=0.40))

    assert result.failures_total == 4
    assert result.failures_routed_higher == 4
    assert result.successes_total == 4


def test_it_reports_successes_that_would_have_gone_lower(log):
    """The savings on offer, and the risk taken to get them."""
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 6)
    result = replay(log, Thresholds(low_medium=0.55, medium_high=0.80))
    assert result.successes_routed_lower == 6
    assert result.failures_total == 0


def test_cost_delta_uses_the_costs_in_the_log(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 3
               + [(HARD, "high", "heuristic", "success", 0.10)] * 3)
    # Push the medium work up into high: three steps at +0.08 each.
    result = replay(log, Thresholds(low_medium=0.20, medium_high=0.40))
    assert result.estimated_cost_delta == pytest.approx(0.24)


def test_no_cost_data_means_no_estimate(log):
    write(log, [(MID, "medium", "heuristic", "success", None)] * 5)
    assert replay(log, Thresholds()).estimated_cost_delta is None


# -- what it refuses to replay ----------------------------------------------

def test_classifier_decisions_are_not_re_derived(log):
    """The classifier's verdict is not in the log, so it cannot be replayed."""
    write(log, [(MID, "high", "llm", "success", 0.02)] * 4)
    result = replay(log, Thresholds())
    assert result.considered == 0
    assert result.skipped_classifier == 4


def test_hints_floors_and_retries_are_skipped(log):
    write(log, [(MID, "high", "hint", "success", 0.1)] * 2
               + [(MID, "medium", "floor", "success", 0.02)] * 2
               + [(EASY, "low", "heuristic", "success", 0.004)] * 2)
    result = replay(log, Thresholds())
    assert result.considered == 2
    assert result.skipped_not_from_cuts == 4


def test_events_without_signals_are_skipped(log):
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "event": "routing_decision", "decision_id": "d0", "tier": "low",
            "source": "heuristic", "attempt": 1,
        }) + "\n")
    result = replay(log, Thresholds())
    assert result.considered == 0
    assert result.skipped_no_signals == 1


def test_missing_log_is_not_an_error(tmp_path):
    assert replay(tmp_path / "nope.jsonl", Thresholds()).considered == 0


def test_rows_are_optional(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 3)
    assert replay(log, Thresholds()).rows == []
    rows = replay(log, Thresholds(low_medium=0.20, medium_high=0.40), keep_rows=True).rows
    assert len(rows) == 3
    assert rows[0].was == "medium" and rows[0].would_be == "high"
    assert rows[0].changed is True


def test_result_serialises(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 2)
    payload = replay(log, Thresholds(), keep_rows=True).to_dict(include_rows=True)
    json.dumps(payload)
    assert payload["considered"] == 2
    assert len(payload["rows"]) == 2


# -- CLI ---------------------------------------------------------------------

def test_cli_replay_reports_the_shift(log, capsys):
    write(log, [(MID, "medium", "heuristic", "insufficient", 0.02)] * 3)
    assert main(["replay", str(log), "--low-medium", "0.20", "--medium-high", "0.40"]) == 0
    out = capsys.readouterr().out
    assert "3 decisions replayed, 3 would change tier" in out
    assert "would have been routed higher" in out
    assert "not that they would have succeeded there" in out


def test_cli_replay_json(log, capsys):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)] * 3)
    assert main(["replay", str(log), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["considered"] == 3
    assert payload["thresholds"]["low_medium"] == pytest.approx(0.30)


def test_cli_replay_rejects_impossible_cuts(log):
    write(log, [(MID, "medium", "heuristic", "success", 0.02)])
    with pytest.raises(ValueError):
        main(["replay", str(log), "--low-medium", "0.60", "--medium-high", "0.65"])
