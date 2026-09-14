import json

import pytest

from tierwise import (
    JsonlSink,
    Outcome,
    Router,
    RouterConfig,
    RoutingSession,
    TaskSignals,
    Tier,
)
from tierwise.escalation import EscalationPolicy

EASY = dict(category="typo", file_count=1, lines_changed=2)
HARD = dict(category="architecture", file_count=30, lines_changed=2000,
            dependency_depth=6, requires_context=True, ambiguity=0.9)


def test_each_step_is_routed_independently():
    """The point of the inner loop: one hard step must not tax the rest."""
    session = RoutingSession()
    tiers = [
        session.route_step(TaskSignals(**EASY)).tier,
        session.route_step(TaskSignals(**HARD)).tier,
        session.route_step(TaskSignals(**EASY)).tier,
    ]
    assert tiers == [Tier.LOW, Tier.HIGH, Tier.LOW]


def test_session_de_escalates_after_a_hard_step():
    session = RoutingSession()
    session.route_step(TaskSignals(**HARD))
    session.route_step(TaskSignals(**EASY))
    assert session.summary()["de_escalations"] == 1


def test_steps_are_numbered_and_tagged_with_the_session():
    session = RoutingSession()
    first = session.route_step(TaskSignals(**EASY))
    second = session.route_step(TaskSignals(**EASY))
    assert (first.step_index, second.step_index) == (0, 1)
    assert first.session_id == second.session_id == session.session_id
    assert first.decision_id != second.decision_id


def test_sessions_do_not_share_ids():
    assert RoutingSession().session_id != RoutingSession().session_id


def test_mark_outcome_returns_an_escalation_when_earned():
    session = RoutingSession()
    session.route_step(TaskSignals(**EASY))
    escalated = session.mark_outcome(Outcome.INSUFFICIENT)
    assert escalated is not None
    assert escalated.tier is Tier.MEDIUM
    assert escalated.session_id == session.session_id
    assert escalated.step_index == 0  # same step, second attempt
    assert session.tier_history == [Tier.LOW, Tier.MEDIUM]


def test_mark_outcome_returns_none_on_success():
    session = RoutingSession()
    session.route_step(TaskSignals(**EASY))
    assert session.mark_outcome("success") is None


def test_mark_outcome_before_any_step_is_an_error():
    with pytest.raises(RuntimeError):
        RoutingSession().mark_outcome(Outcome.SUCCESS)


def test_mark_outcome_can_target_an_earlier_step():
    session = RoutingSession()
    first = session.route_step(TaskSignals(**EASY))
    session.route_step(TaskSignals(**EASY))
    escalated = session.mark_outcome(Outcome.INSUFFICIENT, decision=first)
    assert escalated.step_index == 0


def test_outcome_is_logged_as_its_own_event(tmp_path):
    """Outcomes must survive the process, so the tuner can read them later."""
    log = tmp_path / "loop.jsonl"
    session = RoutingSession(router=Router(telemetry=JsonlSink(log)))
    decision = session.route_step(TaskSignals(**EASY))
    session.mark_outcome(Outcome.SUCCESS, cost_usd=0.0012)

    events = [json.loads(line) for line in log.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds == ["routing_decision", "task_outcome"]
    outcome = events[1]
    assert outcome["decision_id"] == decision.decision_id
    assert outcome["outcome"] == "success"
    assert outcome["cost_usd"] == pytest.approx(0.0012)


def test_escalation_is_logged_too(tmp_path):
    log = tmp_path / "loop.jsonl"
    session = RoutingSession(router=Router(telemetry=JsonlSink(log)))
    session.route_step(TaskSignals(**EASY))
    session.mark_outcome(Outcome.INSUFFICIENT)

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["event"] for e in events] == [
        "routing_decision", "task_outcome", "routing_decision",
    ]
    assert events[2]["attempt"] == 2
    assert events[2]["source"] == "escalation"


def test_summary_reports_the_loop_shape():
    session = RoutingSession(
        router=Router(config=RouterConfig(escalation=EscalationPolicy(max_attempts=2)))
    )
    session.route_step(TaskSignals(**EASY))
    session.mark_outcome(Outcome.INSUFFICIENT)
    session.route_step(TaskSignals(**HARD))

    summary = session.summary()
    assert summary["steps"] == 2
    assert summary["decisions"] == 3
    assert summary["escalations"] == 1
    assert summary["tier_counts"]["low"] == 1
    assert summary["session_id"] == session.session_id
    assert "low_medium" in summary["thresholds"]


def test_router_is_reusable_across_sessions():
    """Router holds no per-task state, so one instance serves many sessions."""
    router = Router()
    a, b = RoutingSession(router=router), RoutingSession(router=router)
    a.route_step(TaskSignals(**EASY))
    b.route_step(TaskSignals(**HARD))
    assert a.step_index == b.step_index == 1
    assert a.tier_history == [Tier.LOW]
    assert b.tier_history == [Tier.HIGH]
