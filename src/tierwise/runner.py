"""TaskRunner -- run the loop, don't just advise it.

The router decides and the caller executes. That split is deliberate: owning
the model call would mean owning auth, retries, streaming, tool use and rate
limits, which is an SDK's job and not this library's.

What *was* left to the caller unnecessarily is the plumbing between those two
things -- call the model, check the result, escalate, retry at the new tier,
record the outcome, stop at the cap. That is the same ten lines in every
integration, it is where the loop gets subtly wrong (a forgotten
``mark_outcome`` silently disables all learning), and none of it is
application-specific.

So the runner automates the mechanism and keeps the two judgements injected:

* ``executor`` -- makes the call. You still own the transport.
* ``verify``   -- says whether the result is good. You still own the standard.

Which means the loop is exactly as good as ``verify``. A lazy check ("the
response was non-empty") makes an automatic loop that escalates on noise and
teaches the tuner from it. Wire it to the thing you would actually trust: your
tests, your linter, your reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Union

from .escalation import Outcome
from .models import RoutingDecision, TaskSignals, Tier
from .router import Router
from .session import RoutingSession

#: Makes the call for a decision. Gets the whole decision, not just the model
#: id, so it can see the tier and why it was chosen.
Executor = Callable[[RoutingDecision], Any]

#: Judges a result. True/False, an Outcome, or an outcome name.
Verdict = Union[bool, Outcome, str]
Verifier = Callable[[Any], Verdict]

CostFn = Callable[[Any], Optional[float]]


def as_outcome(verdict: Verdict) -> Outcome:
    if isinstance(verdict, Outcome):
        return verdict
    if isinstance(verdict, bool):
        return Outcome.SUCCESS if verdict else Outcome.INSUFFICIENT
    return Outcome(str(verdict).strip().lower())


@dataclass
class Attempt:
    """One try at one step."""

    decision: RoutingDecision
    outcome: Outcome
    result: Any = None
    cost_usd: Optional[float] = None
    error: Optional[BaseException] = None

    @property
    def tier(self) -> Tier:
        return self.decision.tier

    @property
    def model(self) -> str:
        return self.decision.model


@dataclass
class StepResult:
    """What happened to one step, across however many tiers it took."""

    ok: bool
    result: Any = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def decision(self) -> Optional[RoutingDecision]:
        return self.attempts[-1].decision if self.attempts else None

    @property
    def tier(self) -> Optional[Tier]:
        return self.attempts[-1].tier if self.attempts else None

    @property
    def model(self) -> Optional[str]:
        return self.attempts[-1].model if self.attempts else None

    @property
    def escalated(self) -> bool:
        return len(self.attempts) > 1

    @property
    def total_cost_usd(self) -> float:
        return sum(a.cost_usd or 0.0 for a in self.attempts)

    @property
    def tiers(self) -> list[Tier]:
        return [a.tier for a in self.attempts]


class TaskRunner:
    """Routes, calls, checks, escalates and records -- without being asked twice."""

    def __init__(
        self,
        executor: Executor,
        verify: Verifier,
        session: Optional[RoutingSession] = None,
        router: Optional[Router] = None,
        cost_fn: Optional[CostFn] = None,
    ) -> None:
        if session is not None and router is not None:
            raise ValueError("pass a session or a router, not both")
        self.executor = executor
        self.verify = verify
        self.cost_fn = cost_fn
        self.session = session if session is not None else RoutingSession(router=router)

    def run(self, signals: TaskSignals) -> StepResult:
        """Run one step to a conclusion, escalating if the tier proves too low."""
        decision = self.session.route_step(signals)
        attempts: list[Attempt] = []

        while True:
            try:
                result = self.executor(decision)
            except Exception as exc:  # noqa: BLE001
                # Infrastructure, not tier. Escalating would spend a more
                # expensive model on a problem a bigger model cannot fix, and
                # would teach the tuner that the tier was wrong when it wasn't.
                attempts.append(Attempt(decision=decision, outcome=Outcome.ERROR, error=exc))
                self.session.mark_outcome(Outcome.ERROR, decision=decision)
                return StepResult(ok=False, attempts=attempts)

            outcome = as_outcome(self.verify(result))
            cost = self.cost_fn(result) if self.cost_fn else None
            attempts.append(
                Attempt(decision=decision, outcome=outcome, result=result, cost_usd=cost)
            )

            retry = self.session.mark_outcome(outcome, cost_usd=cost, decision=decision)

            if outcome is Outcome.SUCCESS:
                return StepResult(ok=True, result=result, attempts=attempts)
            if retry is None:
                # Nothing left to escalate to, or the attempt cap is reached.
                return StepResult(ok=False, result=result, attempts=attempts)
            decision = retry

    def run_all(self, steps: Iterable[TaskSignals]) -> list[StepResult]:
        """Run a whole task, one step at a time, in one session."""
        return [self.run(signals) for signals in steps]

    def summary(self) -> dict[str, Any]:
        return self.session.summary()


def make_anthropic_executor(
    prompt_fn: Optional[Callable[[RoutingDecision], str]] = None,
    max_tokens: int = 1024,
    system: Optional[str] = None,
    client: Any = None,
    api_key: Optional[str] = None,
) -> Executor:
    """An executor that calls the Anthropic Messages API at the routed model.

    Convenience for the common case, not a general SDK wrapper: streaming, tool
    use and anything else belong in an executor of your own, which is a
    two-line function.
    """
    if client is None:
        import anthropic

        import os

        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def default_prompt(decision: RoutingDecision) -> str:
        return decision.signals.description

    build = prompt_fn or default_prompt

    def executor(decision: RoutingDecision) -> Any:
        kwargs: dict[str, Any] = {
            "model": decision.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": build(decision)}],
        }
        if system is not None:
            kwargs["system"] = system
        return client.messages.create(**kwargs)

    return executor
