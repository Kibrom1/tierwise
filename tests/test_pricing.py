"""Switching cost: the half of the decision complexity scoring cannot see."""

import pytest

from tierwise import (
    ModelPrice,
    Router,
    RouterConfig,
    Source,
    SwitchContext,
    TaskSignals,
    Tier,
    breakeven_output_tokens,
    evaluate_switch,
    price_for,
    step_cost,
)

MODELS = {Tier.LOW: "claude-haiku-4-5-20251001",
          Tier.MEDIUM: "claude-sonnet-5",
          Tier.HIGH: "claude-opus-5"}
model_for = MODELS.__getitem__

EASY = dict(category="typo", file_count=1, lines_changed=2)     # routes low
HARD = dict(category="architecture", file_count=30, lines_changed=2000,
            dependency_depth=6, requires_context=True, ambiguity=0.9)


# -- the arithmetic ----------------------------------------------------------

def test_cache_read_is_a_tenth_of_base_input():
    """100k tokens at $5/MTok is $0.50 of base input: $0.05 read, $0.625 write."""
    opus = price_for("claude-opus-5")
    assert step_cost(opus, 100_000, 0, cache_hit=True) == pytest.approx(0.05)
    assert step_cost(opus, 100_000, 0, cache_hit=False) == pytest.approx(0.625)


def test_dropping_a_tier_against_a_warm_cache_costs_more():
    """The finding that makes this module necessary."""
    opus, haiku = price_for("claude-opus-5"), price_for("claude-haiku-4-5-20251001")
    stay = step_cost(opus, 100_000, 500, cache_hit=True)
    move = step_cost(haiku, 100_000, 500, cache_hit=False)
    assert move > stay
    assert move == pytest.approx(0.1275)    # 0.125 cache write + 0.0025 output
    assert stay == pytest.approx(0.0625)    # 0.05 cache read + 0.0125 output


def test_a_big_context_holds_the_downgrade_back():
    verdict = evaluate_switch(
        Tier.HIGH, Tier.LOW,
        SwitchContext(cached_tokens=100_000, expected_output_tokens=500,
                      incumbent=Tier.HIGH),
        model_for,
    )
    assert verdict.switch is False
    assert "costs $" in verdict.reason
    assert verdict.saving < 0


def test_a_small_context_lets_it_through():
    verdict = evaluate_switch(
        Tier.HIGH, Tier.LOW,
        SwitchContext(cached_tokens=2_000, expected_output_tokens=500,
                      incumbent=Tier.HIGH),
        model_for,
    )
    assert verdict.switch is True
    assert verdict.saving > 0


def test_large_output_outruns_the_cache_penalty():
    ctx = SwitchContext(cached_tokens=100_000, expected_output_tokens=8_000,
                        incumbent=Tier.HIGH)
    assert evaluate_switch(Tier.HIGH, Tier.LOW, ctx, model_for).switch is True


def test_breakeven_is_reported_in_output_tokens():
    point = breakeven_output_tokens(Tier.HIGH, Tier.LOW, 100_000, model_for)
    # penalty 100k x (1 x 1.25 - 5 x 0.1) / M = $0.075; saving $20/MTok
    assert point == pytest.approx(3_750, rel=1e-6)


def test_no_warm_cache_means_nothing_to_forfeit():
    ctx = SwitchContext(cached_tokens=100_000, expected_output_tokens=100,
                        incumbent=Tier.HIGH, cache_warm=False)
    assert evaluate_switch(Tier.HIGH, Tier.LOW, ctx, model_for).switch is True


def test_upgrades_are_never_gated_on_cost():
    """A step that needs a better model needs it whatever the cache is doing."""
    ctx = SwitchContext(cached_tokens=1_000_000, expected_output_tokens=1,
                        incumbent=Tier.LOW)
    verdict = evaluate_switch(Tier.LOW, Tier.HIGH, ctx, model_for)
    assert verdict.switch is True
    assert "quality" in verdict.reason


def test_unknown_prices_do_not_pretend():
    ctx = SwitchContext(cached_tokens=100_000, expected_output_tokens=10,
                        incumbent=Tier.HIGH)
    verdict = evaluate_switch(Tier.HIGH, Tier.LOW, ctx,
                              lambda tier: "some-other-providers-model")
    assert verdict.switch is True
    assert "no price" in verdict.reason


def test_custom_prices_are_accepted():
    flat = {"a": ModelPrice(1.0, 1.0), "b": ModelPrice(1.0, 1.0)}
    ctx = SwitchContext(cached_tokens=50_000, expected_output_tokens=1_000,
                        incumbent=Tier.HIGH)
    verdict = evaluate_switch(Tier.HIGH, Tier.LOW, ctx,
                              {Tier.HIGH: "a", Tier.LOW: "b"}.__getitem__,
                              prices=flat.get)
    assert verdict.switch is False        # identical prices, so the write is pure loss


def test_negative_token_counts_are_rejected():
    with pytest.raises(ValueError):
        SwitchContext(cached_tokens=-1)


# -- through the Router ------------------------------------------------------

def test_the_router_holds_a_downgrade_that_would_cost_more():
    router = Router()
    decision = router.route(
        TaskSignals(**EASY),
        context=SwitchContext(cached_tokens=100_000, expected_output_tokens=400,
                              incumbent=Tier.HIGH),
    )
    assert decision.tier is Tier.HIGH
    assert decision.source is Source.CACHE_HOLD
    assert "held at high" in decision.rationale


def test_without_a_context_nothing_changes():
    assert Router().route(TaskSignals(**EASY)).tier is Tier.LOW


def test_the_gate_can_be_turned_off():
    router = Router(config=RouterConfig(respect_switching_cost=False))
    decision = router.route(
        TaskSignals(**EASY),
        context=SwitchContext(cached_tokens=100_000, expected_output_tokens=400,
                              incumbent=Tier.HIGH),
    )
    assert decision.tier is Tier.LOW


def test_a_cold_cache_routes_normally():
    decision = Router().route(
        TaskSignals(**EASY),
        context=SwitchContext(cached_tokens=100_000, incumbent=Tier.HIGH,
                              cache_warm=False),
    )
    assert decision.tier is Tier.LOW


def test_the_router_never_holds_back_an_upgrade():
    decision = Router().route(
        TaskSignals(**HARD),
        context=SwitchContext(cached_tokens=500_000, expected_output_tokens=10,
                              incumbent=Tier.LOW),
    )
    assert decision.tier is Tier.HIGH


def test_a_held_decision_is_not_then_explored_downward():
    """Exploration exists to measure headroom, not to re-lose the cache."""
    router = Router(config=RouterConfig(exploration_rate=1.0,
                                        respect_switching_cost=True),
                    rng=lambda: 0.0)
    decision = router.route(
        TaskSignals(**EASY),
        context=SwitchContext(cached_tokens=100_000, expected_output_tokens=400,
                              incumbent=Tier.HIGH),
    )
    assert decision.source is Source.CACHE_HOLD
    assert decision.tier is Tier.HIGH


def test_the_hold_is_visible_in_the_log():
    from tierwise import TelemetryLog
    log = TelemetryLog()
    Router(telemetry=log).route(
        TaskSignals(**EASY),
        context=SwitchContext(cached_tokens=100_000, expected_output_tokens=400,
                              incumbent=Tier.HIGH),
    )
    assert log.events[0]["source"] == "cache_hold"
