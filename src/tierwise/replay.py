"""Replay a decision log against different thresholds.

The tuner applies changes automatically, which is only reasonable if you can
see what a change would have done before trusting it. Every routing_decision
event carries the full signals it was made from, so the counterfactual is
already in the log: re-score each decision, cut it at candidate thresholds, and
compare.

What replay can and cannot tell you is worth being exact about. It can say how
many decisions would land in a different tier, and -- where outcomes were
reported -- how many *failures* would have been routed higher and how many
*successes* would have been routed lower. It cannot say whether those rerouted
tasks would then have succeeded: that outcome was never observed. A failure
moved up to a bigger model is a plausible fix, not a proven one.

Decisions that did not come from the cuts are excluded rather than guessed at:
an engineer's hint, a min_tier floor and an escalation retry would be unchanged
by new thresholds, and a decision the classifier made cannot be re-derived
without calling it again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .heuristics import score_task, tier_for_score
from .models import TaskSignals, Tier
from .thresholds import Thresholds
from .tuner import EventSource, _events_from

FAILURE_OUTCOMES = {"insufficient", "rejected"}
SUCCESS_OUTCOMES = {"success"}

#: Sources whose tier the cuts did not decide.
NOT_FROM_CUTS = {"hint", "floor", "escalation"}

_SIGNAL_FIELDS = set(TaskSignals().to_dict())


@dataclass
class ReplayRow:
    decision_id: str
    score: float
    was: str
    would_be: str
    outcome: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.was != self.would_be

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "changed": self.changed}


@dataclass
class ReplayResult:
    thresholds: dict[str, float]
    considered: int = 0
    skipped_not_from_cuts: int = 0
    skipped_classifier: int = 0
    skipped_no_signals: int = 0
    changed: int = 0
    tier_before: dict[str, int] = field(default_factory=dict)
    tier_after: dict[str, int] = field(default_factory=dict)
    #: Failures that the candidate cuts would have sent to a higher tier. A
    #: plausible fix, not a proven one -- the rerouted outcome was never seen.
    failures_routed_higher: int = 0
    failures_total: int = 0
    #: Successes the candidate cuts would have sent lower: the savings on offer,
    #: and the risk taken to get them.
    successes_routed_lower: int = 0
    successes_total: int = 0
    estimated_cost_delta: Optional[float] = None
    rows: list[ReplayRow] = field(default_factory=list)

    def to_dict(self, include_rows: bool = False) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "rows"}
        if include_rows:
            data["rows"] = [r.to_dict() for r in self.rows]
        return data


def _signals_from_event(raw: Any) -> Optional[TaskSignals]:
    if not isinstance(raw, dict):
        return None
    try:
        return TaskSignals(**{k: v for k, v in raw.items() if k in _SIGNAL_FIELDS})
    except (TypeError, ValueError):
        return None


def replay(
    log: EventSource,
    thresholds: Optional[Thresholds] = None,
    thresholds_path: str | Path | None = None,
    keep_rows: bool = False,
) -> ReplayResult:
    """Re-cut a decision log at candidate thresholds and report the difference."""
    candidate = thresholds if thresholds is not None else Thresholds.load(thresholds_path)
    events = list(_events_from(log))

    outcomes: dict[str, dict[str, Any]] = {}
    costs: dict[str, list[float]] = {}
    for event in events:
        if event.get("event") != "task_outcome":
            continue
        key = event.get("decision_id")
        if key:
            outcomes[key] = event

    for event in events:
        if event.get("event") != "routing_decision":
            continue
        key = event.get("decision_id")
        cost = outcomes.get(key, {}).get("cost_usd") if key else None
        if cost is not None:
            costs.setdefault(str(event.get("tier")), []).append(float(cost))
    mean_cost = {tier: sum(v) / len(v) for tier, v in costs.items() if v}

    result = ReplayResult(thresholds={
        "low_medium": candidate.low_medium,
        "medium_high": candidate.medium_high,
    })
    delta = 0.0
    priced = False

    for event in events:
        if event.get("event") != "routing_decision":
            continue
        if event.get("attempt", 1) != 1:
            result.skipped_not_from_cuts += 1
            continue
        source = event.get("source")
        if source in NOT_FROM_CUTS:
            result.skipped_not_from_cuts += 1
            continue
        if source == "llm":
            # The classifier's verdict cannot be re-derived from the log.
            result.skipped_classifier += 1
            continue

        signals = _signals_from_event(event.get("signals"))
        if signals is None:
            result.skipped_no_signals += 1
            continue

        was = str(event.get("tier"))
        score = score_task(signals).value
        would_be = tier_for_score(score, candidate).value

        result.considered += 1
        result.tier_before[was] = result.tier_before.get(was, 0) + 1
        result.tier_after[would_be] = result.tier_after.get(would_be, 0) + 1
        if was != would_be:
            result.changed += 1

        if was in mean_cost and would_be in mean_cost:
            delta += mean_cost[would_be] - mean_cost[was]
            priced = True

        outcome = outcomes.get(str(event.get("decision_id")), {}).get("outcome")
        outcome = str(outcome).lower() if outcome else None
        if outcome in FAILURE_OUTCOMES:
            result.failures_total += 1
            if Tier(would_be) > Tier(was):
                result.failures_routed_higher += 1
        elif outcome in SUCCESS_OUTCOMES:
            result.successes_total += 1
            if Tier(would_be) < Tier(was):
                result.successes_routed_lower += 1

        if keep_rows:
            result.rows.append(ReplayRow(
                decision_id=str(event.get("decision_id")), score=round(score, 4),
                was=was, would_be=would_be, outcome=outcome,
            ))

    if priced:
        result.estimated_cost_delta = round(delta, 6)
    return result
