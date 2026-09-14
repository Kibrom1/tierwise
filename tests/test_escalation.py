import pytest

from tierwise.escalation import EscalationPolicy, Outcome
from tierwise.mapping import DEFAULT_MODELS, ModelMap
from tierwise.models import Source, TaskSignals, Tier
from tierwise.router import Router, RouterConfig


@pytest.fixture
def router():
    return Router(model_map=ModelMap(dict(DEFAULT_MODELS)))


@pytest.fixture
def low_decision(router):
    return router.route(TaskSignals(category="typo", lines_changed=1))


def test_success_does_not_escalate(router, low_decision):
    assert router.report_outcome(low_decision, Outcome.SUCCESS) is None


def test_insufficient_output_bumps_one_tier(router, low_decision):
    escalated = router.report_outcome(low_decision, Outcome.INSUFFICIENT)
    assert escalated is not None
    assert escalated.tier is Tier.MEDIUM
    assert escalated.model == DEFAULT_MODELS[Tier.MEDIUM]
    assert escalated.source is Source.ESCALATION
    assert escalated.escalated_from is Tier.LOW
    assert escalated.attempt == 2


def test_rejection_escalates_too(router, low_decision):
    assert router.report_outcome(low_decision, Outcome.REJECTED) is not None


def test_call_errors_do_not_escalate(router, low_decision):
    """A failed API call is an infrastructure problem, not a tier problem."""
    assert router.report_outcome(low_decision, Outcome.ERROR) is None


def test_outcome_accepts_a_string(router, low_decision):
    assert router.report_outcome(low_decision, "insufficient") is not None


def test_attempts_are_capped(router, low_decision):
    first = router.report_outcome(low_decision, Outcome.INSUFFICIENT)
    assert router.report_outcome(first, Outcome.INSUFFICIENT) is None


def test_higher_cap_allows_a_second_bump():
    router = Router(config=RouterConfig(escalation=EscalationPolicy(max_attempts=3)))
    decision = router.route(TaskSignals(category="typo", lines_changed=1))
    first = router.report_outcome(decision, Outcome.INSUFFICIENT)
    second = router.report_outcome(first, Outcome.INSUFFICIENT)
    assert second.tier is Tier.HIGH
    assert second.attempt == 3


def test_top_tier_has_nowhere_to_escalate():
    router = Router(config=RouterConfig(escalation=EscalationPolicy(max_attempts=5)))
    decision = router.route(TaskSignals(tier_hint=Tier.HIGH))
    assert router.report_outcome(decision, Outcome.INSUFFICIENT) is None


def test_escalation_preserves_the_original_signals(router, low_decision):
    escalated = router.report_outcome(low_decision, Outcome.INSUFFICIENT)
    assert escalated.signals is low_decision.signals


def test_tier_bump_is_clamped():
    assert Tier.HIGH.bumped() is Tier.HIGH
    assert Tier.LOW.bumped(5) is Tier.HIGH
    assert Tier.HIGH.bumped(-5) is Tier.LOW
