import json

import pytest

from tierwise import ThresholdTuner, Thresholds
from tierwise.telemetry import JsonlSink


def write_log(path, rows):
    """rows: (outcome, tier, attempt, source) -> a decision + its outcome."""
    with path.open("w", encoding="utf-8") as handle:
        for index, (outcome, tier, attempt, source) in enumerate(rows):
            decision_id = f"d{index}"
            handle.write(json.dumps({
                "event": "routing_decision", "decision_id": decision_id,
                "session_id": "s1", "step_index": index, "tier": tier,
                "source": source, "confidence": 0.8, "attempt": attempt,
            }) + "\n")
            if outcome is not None:
                handle.write(json.dumps({
                    "event": "task_outcome", "decision_id": decision_id,
                    "outcome": outcome, "cost_usd": 0.001,
                }) + "\n")


@pytest.fixture
def log(tmp_path):
    return tmp_path / "routing.jsonl"


def test_not_enough_samples_changes_nothing(log):
    write_log(log, [("success", "low", 1, "heuristic")] * 5)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.adjusted is False
    assert "insufficient samples" in result.reason
    assert Thresholds.load().version == 0


def test_high_failure_rate_routes_higher(log):
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 15
                   + [("success", "low", 1, "heuristic")] * 15)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.adjusted is True
    assert result.failure_rate == pytest.approx(0.5)
    # Lower cuts mean more tasks clear them, i.e. more tasks route higher.
    assert result.after["low_medium"] < before.low_medium
    assert result.after["medium_high"] < before.medium_high
    # And the classifier gets consulted more readily.
    assert result.after["llm_fallback"] > before.llm_fallback


def test_sustained_success_routes_cheaper(log):
    write_log(log, [("success", "low", 1, "heuristic")] * 30)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.adjusted is True
    assert result.after["low_medium"] > before.low_medium
    assert result.after["llm_fallback"] < before.llm_fallback


def test_failure_rate_in_band_is_left_alone(log):
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 3
                   + [("success", "low", 1, "heuristic")] * 27)
    result = ThresholdTuner(min_samples=20, target_failure_rate=0.10).tune(log)
    assert result.adjusted is False
    assert result.direction == "unchanged"
    assert result.failure_rate == pytest.approx(0.10)


def test_tuned_thresholds_persist_for_the_next_run(log):
    """The loop only closes if what it learns outlives the process."""
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 30)
    ThresholdTuner(min_samples=20).tune(log)

    reloaded = Thresholds.load()
    assert reloaded.version == 1
    assert reloaded.low_medium < Thresholds().low_medium
    assert reloaded.tuned_from == str(log)


def test_dry_run_reports_without_persisting(log):
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 30)
    result = ThresholdTuner(min_samples=20).tune(log, apply=False)
    assert result.adjusted is True
    assert Thresholds.load().version == 0


def test_repeated_tuning_accumulates(log):
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 30)
    tuner = ThresholdTuner(min_samples=20)
    tuner.tune(log)
    first = Thresholds.load().low_medium
    tuner.tune(log)
    second = Thresholds.load().low_medium
    assert second < first
    assert Thresholds.load().version == 2


def test_guardrails_stop_the_walk(log):
    """Auto-applying means the floor has to actually hold."""
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 30)
    tuner = ThresholdTuner(min_samples=20, step=0.05, floor=0.10)
    for _ in range(40):
        tuner.tune(log)
    final = Thresholds.load()
    final.validate()
    assert final.low_medium >= 0.10
    assert final.medium_high - final.low_medium >= Thresholds.MIN_GAP


def test_at_the_limit_it_reports_no_change(log):
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 30)
    tuner = ThresholdTuner(min_samples=20, step=0.05, floor=0.10)
    for _ in range(40):
        tuner.tune(log)
    assert tuner.tune(log).reason == "already at a guardrail limit"


def test_errors_are_not_counted_as_tier_failures(log):
    """A failed API call says nothing about whether the tier was right."""
    write_log(log, [("error", "low", 1, "heuristic")] * 20
                   + [("success", "low", 1, "heuristic")] * 20)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.samples == 20
    assert result.failure_rate == pytest.approx(0.0)


def test_hints_floors_and_retries_are_excluded(log):
    write_log(log, [("insufficient", "low", 1, "hint")] * 10
                   + [("insufficient", "low", 1, "floor")] * 10
                   + [("insufficient", "medium", 2, "escalation")] * 10
                   + [("success", "low", 1, "heuristic")] * 20)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.samples == 20
    assert result.failures == 0


def test_decisions_without_outcomes_are_ignored(log):
    write_log(log, [(None, "low", 1, "heuristic")] * 50
                   + [("success", "low", 1, "heuristic")] * 5)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.samples == 5
    assert result.adjusted is False


def test_per_tier_breakdown_is_reported(log):
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 12
                   + [("success", "high", 1, "heuristic")] * 12)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.per_tier["low"] == {"total": 12, "failures": 12}
    assert result.per_tier["high"] == {"total": 12, "failures": 0}


def test_missing_log_is_not_an_error(tmp_path):
    result = ThresholdTuner().tune(tmp_path / "nope.jsonl")
    assert result.adjusted is False
    assert result.samples == 0


def test_corrupt_lines_are_skipped(log):
    write_log(log, [("success", "low", 1, "heuristic")] * 25)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n\n[1,2,3]\n")
    assert ThresholdTuner(min_samples=20).tune(log).samples == 25


def test_tuning_event_is_recorded(tmp_path, log):
    audit = tmp_path / "audit.jsonl"
    write_log(log, [("insufficient", "low", 1, "heuristic")] * 30)
    ThresholdTuner(min_samples=20).tune(log, telemetry=JsonlSink(audit))
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "threshold_tuning"
    assert events[0]["before"]["low_medium"] != events[0]["after"]["low_medium"]
