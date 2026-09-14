"""TaskRunner: the loop running itself, with the two judgements injected."""

import json

import pytest

from tierwise import (
    Attempt,
    JsonlSink,
    Outcome,
    Router,
    RouterConfig,
    RoutingSession,
    Source,
    StepResult,
    TaskRunner,
    TaskSignals,
    TelemetryLog,
    Tier,
    as_outcome,
    make_anthropic_executor,
)
from tierwise.escalation import EscalationPolicy

EASY = dict(category="typo", file_count=1, lines_changed=2)            # -> low
HARD = dict(category="architecture", file_count=30, lines_changed=2000,
            dependency_depth=6, requires_context=True, ambiguity=0.9)  # -> high


def executor_needing(tier: Tier, log=None):
    """Succeeds only at or above `tier` -- what a test suite tells you."""
    def executor(decision):
        if log is not None:
            log.append(decision.tier)
        return {"tier": decision.tier, "ok": decision.tier >= tier}
    return executor


def verify(result):
    return result["ok"]


# -- the mechanism -----------------------------------------------------------

def test_a_step_that_works_runs_once():
    runner = TaskRunner(executor_needing(Tier.LOW), verify)
    outcome = runner.run(TaskSignals(**EASY))

    assert outcome.ok is True
    assert outcome.tier is Tier.LOW
    assert outcome.escalated is False
    assert len(outcome.attempts) == 1


def test_a_step_that_fails_escalates_and_retries_itself():
    tried = []
    runner = TaskRunner(executor_needing(Tier.MEDIUM, tried), verify)
    outcome = runner.run(TaskSignals(**EASY))

    assert outcome.ok is True
    assert tried == [Tier.LOW, Tier.MEDIUM]     # called twice, on its own
    assert outcome.tiers == [Tier.LOW, Tier.MEDIUM]
    assert outcome.escalated is True


def test_it_stops_at_the_attempt_cap():
    runner = TaskRunner(
        executor_needing(Tier.HIGH), verify,
        router=Router(config=RouterConfig(escalation=EscalationPolicy(max_attempts=2))),
    )
    outcome = runner.run(TaskSignals(**EASY))

    assert outcome.ok is False
    assert outcome.tiers == [Tier.LOW, Tier.MEDIUM]   # not three


def test_a_raised_cap_lets_it_climb_further():
    runner = TaskRunner(
        executor_needing(Tier.HIGH), verify,
        router=Router(config=RouterConfig(escalation=EscalationPolicy(max_attempts=3))),
    )
    outcome = runner.run(TaskSignals(**EASY))

    assert outcome.ok is True
    assert outcome.tiers == [Tier.LOW, Tier.MEDIUM, Tier.HIGH]


def test_the_top_tier_failing_ends_it():
    runner = TaskRunner(lambda d: {"ok": False}, verify)
    outcome = runner.run(TaskSignals(**HARD))

    assert outcome.ok is False
    assert outcome.tiers == [Tier.HIGH]


def test_run_all_keeps_one_session():
    runner = TaskRunner(executor_needing(Tier.LOW), verify)
    results = runner.run_all([TaskSignals(**EASY), TaskSignals(**HARD), TaskSignals(**EASY)])

    assert [r.tier for r in results] == [Tier.LOW, Tier.HIGH, Tier.LOW]
    assert runner.summary()["de_escalations"] == 1


# -- the injected judgements -------------------------------------------------

def test_verify_accepts_bools_outcomes_and_names():
    assert as_outcome(True) is Outcome.SUCCESS
    assert as_outcome(False) is Outcome.INSUFFICIENT
    assert as_outcome(Outcome.REJECTED) is Outcome.REJECTED
    assert as_outcome("rejected") is Outcome.REJECTED
    assert as_outcome("ERROR") is Outcome.ERROR


def test_a_rejecting_verifier_escalates_too():
    tried = []
    runner = TaskRunner(
        executor_needing(Tier.MEDIUM, tried),
        lambda r: Outcome.SUCCESS if r["ok"] else Outcome.REJECTED,
    )
    assert runner.run(TaskSignals(**EASY)).tiers == [Tier.LOW, Tier.MEDIUM]


def test_the_executor_sees_the_whole_decision():
    seen = {}

    def executor(decision):
        seen["model"] = decision.model
        seen["rationale"] = decision.rationale
        return {"ok": True}

    TaskRunner(executor, verify).run(TaskSignals(**EASY))
    assert seen["model"]
    assert "complexity score" in seen["rationale"]


def test_a_thrown_error_is_not_treated_as_a_tier_problem():
    """Escalating would spend a bigger model on something it cannot fix."""
    calls = []

    def executor(decision):
        calls.append(decision.tier)
        raise ConnectionError("no route to host")

    outcome = TaskRunner(executor, verify).run(TaskSignals(**EASY))

    assert outcome.ok is False
    assert calls == [Tier.LOW]                       # no retry at a higher tier
    assert outcome.attempts[0].outcome is Outcome.ERROR
    assert isinstance(outcome.attempts[0].error, ConnectionError)


def test_cost_is_collected_when_a_cost_fn_is_given():
    runner = TaskRunner(
        executor_needing(Tier.MEDIUM), verify,
        cost_fn=lambda r: 0.01 if r["tier"] is Tier.LOW else 0.05,
    )
    outcome = runner.run(TaskSignals(**EASY))
    assert outcome.total_cost_usd == pytest.approx(0.06)


# -- it still feeds the loops it is part of ----------------------------------

def test_running_automatically_still_records_everything(tmp_path):
    log = tmp_path / "auto.jsonl"
    runner = TaskRunner(
        executor_needing(Tier.MEDIUM), verify,
        router=Router(telemetry=JsonlSink(log)),
        cost_fn=lambda r: 0.01,
    )
    runner.run(TaskSignals(**EASY))

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["event"] for e in events] == [
        "routing_decision", "task_outcome", "routing_decision", "task_outcome",
    ]
    assert events[1]["outcome"] == "insufficient"
    assert events[2]["source"] == "escalation"
    assert events[3]["outcome"] == "success"


def test_it_honours_exploration():
    tried = []
    runner = TaskRunner(
        executor_needing(Tier.LOW, tried), verify,
        router=Router(config=RouterConfig(exploration_rate=1.0), rng=lambda: 0.0),
    )
    outcome = runner.run(TaskSignals(
        category="feature", file_count=3, lines_changed=90,
        dependency_depth=1, requires_context=True,
    ))
    assert outcome.attempts[0].decision.source is Source.EXPLORATION
    assert outcome.ok is True          # the downgrade worked: real evidence


def test_an_existing_session_can_be_supplied():
    log = TelemetryLog()
    session = RoutingSession(router=Router(telemetry=log))
    runner = TaskRunner(executor_needing(Tier.LOW), verify, session=session)
    runner.run(TaskSignals(**EASY))

    assert runner.session is session
    assert session.step_index == 1
    assert len(log) == 2


def test_session_and_router_are_mutually_exclusive():
    with pytest.raises(ValueError):
        TaskRunner(executor_needing(Tier.LOW), verify,
                   session=RoutingSession(), router=Router())


# -- the optional Anthropic executor -----------------------------------------

def test_anthropic_executor_calls_the_routed_model():
    seen = {}

    class Messages:
        def create(self, **kwargs):
            seen.update(kwargs)
            return {"ok": True}

    class Client:
        messages = Messages()

    executor = make_anthropic_executor(client=Client(), max_tokens=64, system="be terse")
    runner = TaskRunner(executor, verify)
    outcome = runner.run(TaskSignals(description="fix a typo", **EASY))

    assert outcome.ok is True
    assert seen["model"] == outcome.model
    assert seen["max_tokens"] == 64
    assert seen["system"] == "be terse"
    assert seen["messages"][0]["content"] == "fix a typo"


def test_anthropic_executor_takes_a_custom_prompt():
    seen = {}

    class Messages:
        def create(self, **kwargs):
            seen.update(kwargs)
            return {"ok": True}

    class Client:
        messages = Messages()

    executor = make_anthropic_executor(
        client=Client(), prompt_fn=lambda d: f"[{d.tier.value}] {d.signals.description}"
    )
    TaskRunner(executor, verify).run(TaskSignals(description="fix a typo", **EASY))
    assert seen["messages"][0]["content"] == "[low] fix a typo"
