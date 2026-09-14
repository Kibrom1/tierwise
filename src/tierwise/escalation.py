"""Escalation policy.

Routing cheap is only safe if there is a way back up. When a tier's attempt
comes back unusable, the policy bumps one tier and hands back a new decision --
capped, so a failing task cannot walk itself to the top tier repeatedly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .models import RoutingDecision, Source, Tier


class Outcome(str, enum.Enum):
    """What happened when the routed model ran."""

    SUCCESS = "success"
    INSUFFICIENT = "insufficient"   # output was wrong, shallow, or incomplete
    ERROR = "error"                 # the call itself failed
    REJECTED = "rejected"           # a human turned the result down


ESCALATING_OUTCOMES = {Outcome.INSUFFICIENT, Outcome.REJECTED}


@dataclass
class EscalationPolicy:
    """Decides whether a failed attempt earns a higher tier."""

    max_attempts: int = 2
    steps: int = 1

    def should_escalate(self, decision: RoutingDecision, outcome: Outcome) -> bool:
        if isinstance(outcome, str):
            outcome = Outcome(outcome)
        if outcome not in ESCALATING_OUTCOMES:
            return False
        if decision.attempt >= self.max_attempts:
            return False
        return decision.tier is not Tier.HIGH

    def escalate(
        self,
        decision: RoutingDecision,
        outcome: Outcome,
        model_for_tier,
    ) -> RoutingDecision | None:
        """Return the next decision, or None if escalation is not warranted."""
        if isinstance(outcome, str):
            outcome = Outcome(outcome)
        if not self.should_escalate(decision, outcome):
            return None

        next_tier = decision.tier.bumped(self.steps)
        return RoutingDecision(
            tier=next_tier,
            model=model_for_tier(next_tier),
            confidence=1.0,
            source=Source.ESCALATION,
            rationale=(
                f"attempt {decision.attempt} at {decision.tier.value} returned "
                f"{outcome.value}; escalated to {next_tier.value}"
            ),
            signals=decision.signals,
            attempt=decision.attempt + 1,
            escalated_from=decision.tier,
        )
