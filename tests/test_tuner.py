import json

import pytest

from tierwise import ThresholdTuner, Thresholds
from tierwise.telemetry import JsonlSink

BASE_TS = 1_700_000_000.0


def write_log(path, rows, mode="w", start=0):
    """rows: (outcome, tier, attempt, source) -> a decision plus its outcome.

    Outcome timestamps increase, because the watermark is a timestamp.
    """
    with path.open(mode, encoding="utf-8") as handle:
        for offset, (outcome, tier, attempt, source) in enumerate(rows):
            index = start + offset
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
                    "timestamp": BASE_TS + index,
                }) + "\n")


def append_log(path, rows):
    existing = sum(1 for line in path.read_text().splitlines() if '"routing_decision"' in line)
    write_log(path, rows, mode="a", start=existing)


FAIL_LOW = ("insufficient", "low", 1, "heuristic")
PASS_LOW = ("success", "low", 1, "heuristic")
FAIL_MED = ("insufficient", "medium", 1, "heuristic")
PASS_MED = ("success", "medium", 1, "heuristic")


@pytest.fixture
def log(tmp_path):
    return tmp_path / "routing.jsonl"


def by_name(result):
    return {a.name: a for a in result.adjustments}


# -- watermark: evidence is spent when it is acted on ------------------------

def test_the_same_evidence_is_not_spent_twice(log):
    """Re-running on an unchanged log must move nothing.

    Without a watermark the tuner re-read the whole log every run and stepped
    again on outcomes it had already acted on -- a nightly job would walk the
    cuts to the floor on one bad week and keep walking.
    """
    write_log(log, [FAIL_LOW] * 30)
    tuner = ThresholdTuner(min_samples=20)

    first = tuner.tune(log)
    assert first.adjusted is True
    after_first = Thresholds.load().low_medium

    for _ in range(5):
        again = tuner.tune(log)
        assert again.adjusted is False
        assert again.reason == "no new outcomes since the last tuning"
        assert again.samples == 0

    assert Thresholds.load().low_medium == pytest.approx(after_first)
    assert Thresholds.load().version == 1


def test_new_evidence_moves_it_again(log):
    write_log(log, [FAIL_LOW] * 30)
    tuner = ThresholdTuner(min_samples=20)
    tuner.tune(log)
    first = Thresholds.load().low_medium

    append_log(log, [FAIL_LOW] * 30)
    second = tuner.tune(log)

    assert second.adjusted is True
    assert second.samples == 30            # only the new outcomes
    assert Thresholds.load().low_medium < first


def test_watermark_is_persisted_and_advances(log):
    write_log(log, [FAIL_LOW] * 30)
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.watermark_before is None
    assert result.watermark_after == pytest.approx(BASE_TS + 29)
    assert Thresholds.load().tuned_through == pytest.approx(BASE_TS + 29)


def test_unspent_evidence_is_kept_for_next_time(log):
    """Evidence that did not move anything must still be there next run."""
    write_log(log, [FAIL_LOW] * 10)          # below min_samples
    tuner = ThresholdTuner(min_samples=20)
    assert tuner.tune(log).adjusted is False
    assert Thresholds.load().tuned_through is None

    append_log(log, [FAIL_LOW] * 10)         # now 20 together
    second = tuner.tune(log)
    assert second.samples == 20
    assert second.adjusted is True


def test_a_dry_run_does_not_spend_the_evidence(log):
    write_log(log, [FAIL_LOW] * 30)
    tuner = ThresholdTuner(min_samples=20)
    assert tuner.tune(log, apply=False).adjusted is True
    assert Thresholds.load().tuned_through is None
    assert tuner.tune(log).adjusted is True   # still available


def test_window_keeps_only_recent_outcomes(log):
    write_log(log, [PASS_LOW] * 40 + [FAIL_LOW] * 25)
    result = ThresholdTuner(min_samples=20, window=25).tune(log, apply=False)
    assert result.samples == 25
    assert result.failure_rate == pytest.approx(1.0)


# -- attribution: each cut moves on its own tier -----------------------------

def test_low_tier_failures_move_only_the_low_cut(log):
    """All failures landing in `low` says nothing about the medium/high cut."""
    write_log(log, [FAIL_LOW] * 30)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)
    cuts = by_name(result)

    assert result.after["low_medium"] < before.low_medium
    assert result.after["medium_high"] == pytest.approx(before.medium_high)
    assert cuts["low_medium"].direction == "tightened"
    assert cuts["low_medium"].evidence == "tier=low"
    assert cuts["medium_high"].direction == "insufficient samples"
    assert cuts["medium_high"].samples == 0


def test_medium_tier_failures_move_only_the_high_cut(log):
    write_log(log, [FAIL_MED] * 30)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.after["medium_high"] < before.medium_high
    assert result.after["low_medium"] == pytest.approx(before.low_medium)


def test_each_cut_answers_to_its_own_evidence(log):
    """Failing at low while succeeding at medium moves them opposite ways."""
    write_log(log, [FAIL_LOW] * 25 + [PASS_MED] * 25)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)
    cuts = by_name(result)

    assert cuts["low_medium"].direction == "tightened"
    assert cuts["medium_high"].direction == "relaxed"
    assert result.after["low_medium"] < before.low_medium
    assert result.after["medium_high"] > before.medium_high


def test_a_boundary_without_enough_evidence_stays_put(log):
    write_log(log, [FAIL_LOW] * 25 + [FAIL_MED] * 5)
    result = ThresholdTuner(min_samples=20).tune(log)
    cuts = by_name(result)

    assert cuts["low_medium"].moved is True
    assert cuts["medium_high"].moved is False
    assert cuts["medium_high"].samples == 5
    assert cuts["medium_high"].direction == "insufficient samples"


def test_high_tier_failures_move_no_cut(log):
    """Nothing sits above high, so its failures are not a routing problem.

    Not the tier cuts, and not the fallback cut either: consulting the
    classifier could not have produced a higher route than the one taken.
    """
    write_log(log, [("insufficient", "high", 1, "heuristic")] * 30)
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.adjusted is False
    assert result.per_tier["high"] == {"total": 30, "failures": 30}
    assert Thresholds.load().version == 0


def test_fallback_cut_moves_opposite_to_the_tier_cuts(log):
    """Lowering a tier cut routes higher; raising the fallback cut consults
    the classifier more. Same signal, opposite directions."""
    write_log(log, [FAIL_LOW] * 30)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.after["llm_fallback"] > before.llm_fallback
    assert by_name(result)["llm_fallback"].evidence == "source=heuristic, tier<high"


def test_sustained_success_routes_cheaper(log):
    write_log(log, [PASS_LOW] * 30)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)

    assert result.adjusted is True
    assert result.after["low_medium"] > before.low_medium
    assert result.after["llm_fallback"] < before.llm_fallback


def test_failure_rate_in_band_is_left_alone(log):
    write_log(log, [FAIL_LOW] * 3 + [PASS_LOW] * 27)
    result = ThresholdTuner(min_samples=20, target_failure_rate=0.10).tune(log)
    assert result.adjusted is False
    assert result.direction == "unchanged"
    assert by_name(result)["low_medium"].failure_rate == pytest.approx(0.10)


# -- guardrails --------------------------------------------------------------

def test_not_enough_samples_changes_nothing(log):
    write_log(log, [PASS_LOW] * 5)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.adjusted is False
    assert "short of samples" in result.reason
    assert Thresholds.load().version == 0


def test_guardrails_stop_the_walk(log):
    """Auto-applying means the floor has to actually hold under fresh evidence."""
    tuner = ThresholdTuner(min_samples=20, step=0.05, floor=0.10)
    write_log(log, [FAIL_LOW] * 25)
    for _ in range(40):
        tuner.tune(log)
        append_log(log, [FAIL_LOW] * 25)

    final = Thresholds.load()
    final.validate()
    assert final.low_medium >= 0.10
    assert final.medium_high - final.low_medium >= Thresholds.MIN_GAP


def test_at_the_limit_it_reports_no_change(log):
    tuner = ThresholdTuner(min_samples=20, step=0.05, floor=0.10)
    write_log(log, [FAIL_LOW] * 25)
    for _ in range(40):
        tuner.tune(log)
        append_log(log, [FAIL_LOW] * 25)

    assert tuner.tune(log).reason == "already at a guardrail limit"


# -- what counts as evidence at all ------------------------------------------

def test_errors_are_not_counted_as_tier_failures(log):
    """A failed API call says nothing about whether the tier was right."""
    write_log(log, [("error", "low", 1, "heuristic")] * 20 + [PASS_LOW] * 20)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.samples == 20
    assert result.failure_rate == pytest.approx(0.0)


def test_hints_floors_and_retries_are_excluded(log):
    write_log(log, [("insufficient", "low", 1, "hint")] * 10
                   + [("insufficient", "low", 1, "floor")] * 10
                   + [("insufficient", "medium", 2, "escalation")] * 10
                   + [PASS_LOW] * 20)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.samples == 20
    assert result.failures == 0


def test_decisions_without_outcomes_are_ignored(log):
    write_log(log, [(None, "low", 1, "heuristic")] * 50 + [PASS_LOW] * 5)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.samples == 5
    assert result.adjusted is False


def test_per_tier_breakdown_is_reported(log):
    write_log(log, [FAIL_LOW] * 12 + [("success", "high", 1, "heuristic")] * 12)
    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.per_tier["low"] == {"total": 12, "failures": 12}
    assert result.per_tier["high"] == {"total": 12, "failures": 0}


def test_missing_log_is_not_an_error(tmp_path):
    result = ThresholdTuner().tune(tmp_path / "nope.jsonl")
    assert result.adjusted is False
    assert result.samples == 0


def test_corrupt_lines_are_skipped(log):
    write_log(log, [PASS_LOW] * 25)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n\n[1,2,3]\n")
    assert ThresholdTuner(min_samples=20).tune(log).samples == 25


def test_tuning_event_is_recorded(tmp_path, log):
    audit = tmp_path / "audit.jsonl"
    write_log(log, [FAIL_LOW] * 30)
    ThresholdTuner(min_samples=20).tune(log, telemetry=JsonlSink(audit))
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "threshold_tuning"
    assert events[0]["before"]["low_medium"] != events[0]["after"]["low_medium"]
    assert events[0]["adjustments"][0]["evidence"] == "tier=low"


def test_dry_run_reports_without_persisting(log):
    write_log(log, [FAIL_LOW] * 30)
    result = ThresholdTuner(min_samples=20).tune(log, apply=False)
    assert result.adjusted is True
    assert Thresholds.load().version == 0


# -- attribution: a cut answers only for decisions it made --------------------

FAIL_MED_LLM = ("insufficient", "medium", 1, "llm")
FAIL_LOW_HOLD = ("insufficient", "low", 1, "cache_hold")


def test_classifier_failures_do_not_move_the_cuts(log):
    """The classifier chose these tiers, not the cuts.

    Counting them let a stub classifier's failures at medium tighten
    medium_high, which pushed correctly-scored medium work to the top tier.
    """
    write_log(log, [FAIL_MED_LLM] * 30 + [FAIL_LOW_HOLD] * 30)
    before = Thresholds()
    result = ThresholdTuner(min_samples=20).tune(log)
    cuts = by_name(result)

    assert result.after["low_medium"] == pytest.approx(before.low_medium)
    assert result.after["medium_high"] == pytest.approx(before.medium_high)
    assert cuts["low_medium"].samples == 0
    assert cuts["medium_high"].samples == 0
