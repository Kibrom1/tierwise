"""RoutingSession -- the inner loop.

An agent working a task takes many steps, and they are not equally hard. A
router that picks one tier per task re-creates the problem it exists to solve,
one level down: every step of the task pays for the hardest step in it.

A session routes each step independently. The tier can climb for one gnarly step
and drop straight back for the next, because nothing carries the previous tier
forward -- only the history, which is there to be read, not obeyed.

The Router stays stateless; all per-task state lives here.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from .escalation import Outcome
from .models import RoutingDecision, TaskSignals, Tier
from .router import Router
from .telemetry import build_outcome_event


class RoutingSession:
    def __init__(
        self,
        router: Optional[Router] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.router = router or Router()
        self.session_id = session_id or uuid.uuid4().hex
        self.step_index = 0
        self.decisions: list[RoutingDecision] = []

    # -- the loop ------------------------------------------------------------

    def route_step(self, signals: TaskSignals) -> RoutingDecision:
        """Route the next step. Re-entrant: call it once per step."""
        decision = self.router.route(
            signals, session_id=self.session_id, step_index=self.step_index
        )
        self.decisions.append(decision)
        self.step_index += 1
        return decision

    def mark_outcome(
        self,
        outcome: Outcome | str,
        cost_usd: Optional[float] = None,
        decision: Optional[RoutingDecision] = None,
    ) -> Optional[RoutingDecision]:
        """Report what happened on a step.

        Records the outcome for the outer loop and returns a re-route when the
        step earned an escalation, or None when it did not. Defaults to the most
        recent step; pass ``decision`` to report on an earlier one.
        """
        target = decision or self.last_decision
        if target is None:
            raise RuntimeError("mark_outcome called before any route_step")

        outcome = Outcome(outcome) if isinstance(outcome, str) else outcome
        self.router.telemetry.emit(build_outcome_event(target, outcome.value, cost_usd))

        escalated = self.router.report_outcome(target, outcome)
        if escalated is not None:
            self.decisions.append(escalated)
        return escalated

    # -- introspection -------------------------------------------------------

    @property
    def last_decision(self) -> Optional[RoutingDecision]:
        return self.decisions[-1] if self.decisions else None

    @property
    def tier_history(self) -> list[Tier]:
        return [d.tier for d in self.decisions]

    def summary(self) -> dict[str, Any]:
        tiers = self.tier_history
        escalations = [d for d in self.decisions if d.escalated_from is not None]
        return {
            "session_id": self.session_id,
            "steps": self.step_index,
            "decisions": len(self.decisions),
            "tier_history": [t.value for t in tiers],
            "escalations": len(escalations),
            "de_escalations": sum(
                1 for a, b in zip(tiers, tiers[1:]) if b.rank < a.rank
            ),
            "tier_counts": {
                tier.value: sum(1 for t in tiers if t is tier) for tier in Tier
            },
            "thresholds": {
                "low_medium": self.router.thresholds.low_medium,
                "medium_high": self.router.thresholds.medium_high,
                "llm_fallback": self.router.confidence_threshold,
                "version": self.router.thresholds.version,
            },
        }
