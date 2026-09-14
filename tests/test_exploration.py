"""Exploration routing, and targets derived from cost.

Outcomes are only ever observed at the tier actually used, so
under-provisioning is measurable and over-provisioning is not. Exploration is
what buys the missing half of the evidence.
"""

import json

import pytest

from tierwise import (
    JsonlSink,
    Outcome,
    Router,
    RouterConfig,
    RoutingSession,
    Source,
    TaskSignals,
    ThresholdTuner,
    Thresholds,
    Tier,
)

BASE_TS = 1_700_000_000.0
MID = dict(category="feature", file_count=3, lines_changed=90,
           dependency_depth=1, requires_context=True)      # routes medium
EASY = dict(category="typo", file_count=1, lines_changed=2)  # routes low


def always(value):
    return lambda: value


# -- the Router side ---------------------------------------------------------

def test_exploration_is_off_by_default():
    """It trades quality for evidence, so nobody gets it without asking."""
    router = Router(rng=always(0.0))
    assert router.config.exploration_rate == 0.0
    assert router.route(TaskSignals(**MID)).source is not Source.EXPLORATION


def test_exploration_routes_one_tier_below():
    router = Router(config=RouterConfig(exploration_rate=1.0), rng=always(0.0))
    decision = router.route(TaskSignals(**MID))

    assert decision.source is Source.EXPLORATION
    assert decision.tier is Tier.LOW
    assert decision.explored_from is Tier.MEDIUM
    assert decision.confidence == 0.0
    assert "explored down from medium" in decision.rationale


def test_rate_gates_exploration():
    signals = TaskSignals(**MID)
    config = RouterConfig(exploration_rate=0.1)
    assert Router(config=config, rng=always(0.05)).route(signals).explored_from is Tier.MEDIUM
    assert Router(config=config, rng=always(0.5)).route(signals).explored_from is None


def test_never_explores_below_low():
    """Nothing sits under `low`, so there is nothing to learn there."""
    router = Router(config=RouterConfig(exploration_rate=1.0), rng=always(0.0))
    decision = router.route(TaskSignals(**EASY))
    assert decision.tier is Tier.LOW
    assert decision.source is Source.HEURISTIC


def test_never_explores_against_a_hint():
    """A hint is a person saying what the task needs, not a guess to test."""
    router = Router(config=RouterConfig(exploration_rate=1.0), rng=always(0.0))
    decision = router.route(TaskSignals(**MID, tier_hint=Tier.HIGH))
    assert decision.source is Source.HINT
    assert decision.tier is Tier.HIGH


def test_never_explores_below_a_floor():
    router = Router(config=RouterConfig(exploration_rate=1.0), rng=always(0.0))
    decision = router.route(TaskSignals(**EASY, min_tier=Tier.MEDIUM))
    assert decision.tier is Tier.MEDIUM
    assert decision.source is Source.FLOOR


def test_a_failed_exploration_escalates_back():
    """The safety net that makes exploring affordable."""
    session = RoutingSession(
        router=Router(config=RouterConfig(exploration_rate=1.0), rng=always(0.0))
    )
    explored = session.route_step(TaskSignals(**MID))
    assert explored.tier is Tier.LOW

    recovered = session.mark_outcome(Outcome.INSUFFICIENT)
    assert recovered is not None
    assert recovered.tier is Tier.MEDIUM        # back to the recommendation


def test_exploration_is_visible_in_the_log(tmp_path):
    log = tmp_path / "explore.jsonl"
    session = RoutingSession(router=Router(
        config=RouterConfig(exploration_rate=1.0), rng=always(0.0),
        telemetry=JsonlSink(log),
    ))
    session.route_step(TaskSignals(**MID))
    session.mark_outcome(Outcome.SUCCESS, cost_usd=0.004)

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["source"] == "exploration"
    assert events[0]["explored_from"] == "medium"
    assert events[0]["tier"] == "low"


# -- the tuner side ----------------------------------------------------------

def write(path, rows, mode="w", start=0):
    """rows: (outcome, tier, source, explored_from, cost)."""
    with path.open(mode, encoding="utf-8") as handle:
        for offset, (outcome, tier, source, explored_from, cost) in enumerate(rows):
            index = start + offset
            handle.write(json.dumps({
                "event": "routing_decision", "decision_id": f"d{index}", "tier": tier,
                "source": source, "explored_from": explored_from, "attempt": 1,
                "confidence": 0.8,
            }) + "\n")
            handle.write(json.dumps({
                "event": "task_outcome", "decision_id": f"d{index}", "outcome": outcome,
                "cost_usd": cost, "timestamp": BASE_TS + index,
            }) + "\n")


def normal(outcome, tier, cost=None):
    return (outcome, tier, "heuristic", None, cost)


def downgrade(outcome, tier, came_from, cost=None):
    return (outcome, tier, "exploration", came_from, cost)


def by_name(result):
    return {a.name: a for a in result.adjustments}


@pytest.fixture
def log(tmp_path):
    return tmp_path / "routing.jsonl"


def test_explorations_stay_out_of_the_ordinary_failure_rate(log):
    """A deliberate downgrade failing is not evidence the tier cut is wrong."""
    write(log, [normal("success", "low")] * 20
               + [downgrade("insufficient", "low", "medium")] * 20)
    result = ThresholdTuner(min_samples=20).tune(log, apply=False)

    assert result.samples == 40
    assert result.explorations == 20
    assert result.per_tier["low"] == {"total": 20, "failures": 0}
    assert result.failure_rate == pytest.approx(0.0)


def test_successful_downgrades_relax_a_cut_that_would_not_have_moved(log):
    """The evidence ordinary outcomes can never supply."""
    baseline = [normal("insufficient", "low")] * 2 + [normal("success", "low")] * 18
    write(log, baseline)                                   # rate 0.10 == target
    assert ThresholdTuner(min_samples=20).tune(log, apply=False).adjusted is False

    write(log, baseline + [downgrade("success", "low", "medium")] * 20)
    result = ThresholdTuner(min_samples=20).tune(log, apply=False)
    cut = by_name(result)["low_medium"]

    assert cut.direction == "relaxed"
    assert cut.basis == "downgrades that succeeded"
    assert cut.explore_success_rate == pytest.approx(1.0)
    assert result.after["low_medium"] > Thresholds().low_medium


def test_failed_downgrades_do_not_relax(log):
    write(log, [normal("insufficient", "low")] * 2 + [normal("success", "low")] * 18
               + [downgrade("insufficient", "low", "medium")] * 20)
    assert ThresholdTuner(min_samples=20).tune(log, apply=False).adjusted is False


def test_failures_outrank_successful_downgrades(log):
    """Safety first: real failures at this tier beat evidence of headroom."""
    write(log, [normal("insufficient", "low")] * 10 + [normal("success", "low")] * 10
               + [downgrade("success", "low", "medium")] * 20)
    cut = by_name(ThresholdTuner(min_samples=20).tune(log, apply=False))["low_medium"]

    assert cut.direction == "tightened"
    assert cut.basis == "failures at this tier"


def test_downgrades_are_credited_to_the_right_cut(log):
    """A downgrade from high says something about the medium/high cut only."""
    write(log, [downgrade("success", "medium", "high")] * 20)
    cuts = by_name(ThresholdTuner(min_samples=20).tune(log, apply=False))

    assert cuts["medium_high"].explore_samples == 20
    assert cuts["low_medium"].explore_samples == 0
    assert cuts["medium_high"].direction == "relaxed"


def test_exploration_alone_is_enough_evidence(log):
    """No ordinary outcomes at all, but 20 downgrades that worked."""
    write(log, [downgrade("success", "low", "medium")] * 20)
    result = ThresholdTuner(min_samples=20).tune(log, apply=False)
    assert result.adjusted is True
    assert by_name(result)["low_medium"].samples == 0


# -- cost-derived targets ----------------------------------------------------

def test_target_is_fixed_until_you_say_what_failure_costs(log):
    write(log, [normal("success", "low", 0.004)] * 20)
    cut = by_name(ThresholdTuner(min_samples=20).tune(log, apply=False))["low_medium"]
    assert cut.target_source == "fixed"
    assert cut.target == pytest.approx(0.10)


def test_target_is_derived_from_observed_costs(log):
    write(log, [normal("success", "low", 0.004)] * 20
               + [normal("success", "medium", 0.020)] * 20)
    tuner = ThresholdTuner(min_samples=20, rework_cost_usd=0.05)
    cut = by_name(tuner.tune(log, apply=False))["low_medium"]

    # (0.020 - 0.004) / (0.020 + 0.05)
    assert cut.target_source == "cost-derived"
    assert cut.target == pytest.approx(0.2286, abs=1e-4)


def test_expensive_rework_tightens_the_target(log):
    """If a failure costs a lot, tolerate fewer of them."""
    rows = ([normal("success", "low", 0.004)] * 20
            + [normal("success", "medium", 0.020)] * 20)
    write(log, rows)
    cheap = by_name(ThresholdTuner(min_samples=20, rework_cost_usd=0.01)
                    .tune(log, apply=False))["low_medium"].target
    dear = by_name(ThresholdTuner(min_samples=20, rework_cost_usd=5.0)
                   .tune(log, apply=False))["low_medium"].target
    assert dear < cheap


def test_each_boundary_gets_its_own_target(log):
    write(log, [normal("success", "low", 0.004)] * 20
               + [normal("success", "medium", 0.020)] * 20
               + [normal("success", "high", 0.100)] * 20)
    cuts = by_name(ThresholdTuner(min_samples=20, rework_cost_usd=0.05).tune(log, apply=False))
    assert cuts["low_medium"].target != cuts["medium_high"].target
    assert cuts["low_medium"].target_source == "cost-derived"
    assert cuts["medium_high"].target_source == "cost-derived"


def test_missing_cost_data_falls_back(log):
    write(log, [normal("success", "low", None)] * 20)
    cut = by_name(ThresholdTuner(min_samples=20, rework_cost_usd=0.05)
                  .tune(log, apply=False))["low_medium"]
    assert cut.target_source == "fixed (no cost data)"


def test_derived_target_is_clamped(log):
    """A free failure implies 'always try cheap', which is not a policy."""
    write(log, [normal("success", "low", 0.0001)] * 20
               + [normal("success", "medium", 10.0)] * 20)
    cut = by_name(ThresholdTuner(min_samples=20, rework_cost_usd=0.0)
                  .tune(log, apply=False))["low_medium"]
    assert cut.target <= 0.50


def test_cost_per_success_is_reported(log):
    write(log, [normal("success", "low", 0.01)] * 10
               + [normal("insufficient", "low", 0.01)] * 10)
    result = ThresholdTuner(min_samples=20).tune(log, apply=False)
    assert result.cost_per_success == pytest.approx(0.02)   # $0.20 spent, 10 successes
    assert result.mean_cost_by_tier["low"] == pytest.approx(0.01)


def test_cost_per_success_is_none_without_cost_data(log):
    write(log, [normal("success", "low")] * 20)
    assert ThresholdTuner(min_samples=20).tune(log, apply=False).cost_per_success is None


def test_costs_come_from_every_priced_outcome(log):
    """What a tier costs is a property of the model, not of how it was chosen.

    In a real loop the top tier is often reached only by escalation -- rows the
    failure rate deliberately ignores. If cost statistics ignored them too, the
    medium/high target could never be derived.
    """
    write(log, [normal("success", "low", 0.004)] * 20
               + [normal("insufficient", "medium", 0.020)] * 20
               + [("success", "high", "escalation", None, 0.100)] * 10)
    tuner = ThresholdTuner(min_samples=20, rework_cost_usd=0.05)
    result = tuner.tune(log, apply=False)

    assert result.mean_cost_by_tier["high"] == pytest.approx(0.100)
    assert by_name(result)["medium_high"].target_source == "cost-derived"
    # ...while the escalation rows still stay out of the failure rate
    assert "high" not in result.per_tier
