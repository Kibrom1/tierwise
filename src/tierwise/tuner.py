"""ThresholdTuner -- the outer loop.

Telemetry that nothing reads is decoration. The tuner closes the circuit: it
reads the decision log, joins each decision to the outcome reported for it, and
moves the thresholds that produced those decisions.

It reads the *persisted* log across sessions rather than one session's memory,
and it writes the thresholds back to disk. Both halves matter -- a tuner that
learns from a single in-flight task and forgets at process exit never
accumulates enough evidence to be right, and never gets to apply it.

Two things keep it honest:

* **A watermark.** Evidence is consumed when it is acted on. Re-running against
  the same log moves nothing, because a threshold change is a response to new
  outcomes, not to the accumulated past being read again.
* **Attribution.** Each cut moves on the failures of the tier it governs. All
  the failures landing in ``low`` says the low/medium cut is wrong; it says
  nothing about the medium/high cut, and moving both would push medium-tier
  work to the top tier for no reason.

Failure is treated asymmetrically on purpose. Under-provisioning shows up as a
bad outcome and is measurable; over-provisioning is invisible in outcome data --
the top tier never fails a task the cheap tier could have done. So the loop
tightens on observed failures and relaxes on their sustained absence, which is
the only evidence that the cheap side has room.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .telemetry import TelemetryLog, TelemetrySink, build_tuning_event, read_events
from .thresholds import Thresholds

#: Outcomes that mean the tier was too low. ERROR is excluded deliberately: a
#: failed API call is an infrastructure problem and says nothing about tier.
FAILURE_OUTCOMES = {"insufficient", "rejected"}
SUCCESS_OUTCOMES = {"success"}

#: A decision log: a path to JSONL, an in-memory TelemetryLog, or any iterable
#: of event dicts.
EventSource = Union[str, Path, TelemetryLog, Iterable[dict]]

#: Bounds on a cost-derived target, so an odd cost ratio cannot produce a
#: target of 0.99 (never tighten) or 0.001 (never relax).
MIN_TARGET = 0.02
MAX_TARGET = 0.50

TIGHTEN = "tightened"
RELAX = "relaxed"
UNCHANGED = "unchanged"
INSUFFICIENT = "insufficient samples"


def _events_from(source: EventSource) -> Iterable[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return read_events(source)
    if isinstance(source, TelemetryLog):
        return list(source.events)
    return list(source)


def _describe(source: EventSource) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return type(source).__name__


@dataclass
class BoundaryAdjustment:
    """What one threshold did, and on whose evidence."""

    name: str
    evidence: str
    samples: int
    failures: int
    failure_rate: Optional[float]
    direction: str
    before: float
    after: float
    #: The failure rate this boundary is steering toward, and where it came
    #: from: a hand-set constant, or the break-even rate implied by what the
    #: two tiers actually cost.
    target: Optional[float] = None
    target_source: str = "fixed"
    #: Exploration evidence: downgrades from the tier above this cut.
    explore_samples: int = 0
    explore_successes: int = 0
    explore_success_rate: Optional[float] = None
    #: Which evidence drove the move.
    basis: Optional[str] = None

    @property
    def moved(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "moved": self.moved}


@dataclass
class TuningResult:
    adjusted: bool
    reason: str
    samples: int = 0
    failures: int = 0
    failure_rate: Optional[float] = None
    direction: Optional[str] = None
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    per_tier: dict[str, dict[str, int]] = field(default_factory=dict)
    adjustments: list[BoundaryAdjustment] = field(default_factory=list)
    watermark_before: Optional[float] = None
    watermark_after: Optional[float] = None
    explorations: int = 0
    #: What the loop is actually trying to minimise. Tier is only a proxy for
    #: it; this is the number to watch across runs.
    cost_per_success: Optional[float] = None
    mean_cost_by_tier: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["adjustments"] = [a.to_dict() for a in self.adjustments]
        return data


class ThresholdTuner:
    """Moves the cuts, within guardrails, from observed outcomes.

    Guardrails are not optional decoration on an auto-applying loop -- they are
    what keeps a bad labelling week from walking the thresholds somewhere they
    cannot walk back from: a minimum sample count per boundary, one bounded
    step per run, hard floor and ceiling, an enforced gap between the tier cuts,
    a watermark so the same evidence is never spent twice, and a recorded
    tuning event for every change.
    """

    def __init__(
        self,
        step: float = 0.03,
        min_samples: int = 20,
        target_failure_rate: float = 0.10,
        floor: float = 0.10,
        ceiling: float = 0.90,
        window: Optional[int] = None,
        rework_cost_usd: Optional[float] = None,
    ) -> None:
        self.step = step
        #: Required *per boundary*, not in total -- each cut moves on its own
        #: evidence, so each needs its own sample before it earns a move.
        self.min_samples = min_samples
        self.target_failure_rate = target_failure_rate
        self.floor = floor
        self.ceiling = ceiling
        #: Keep only the most recent N outcomes. Routing quality is not
        #: stationary; a regime from six months ago should not outvote last
        #: week. None keeps everything since the watermark.
        self.window = window
        #: What one failed step costs you beyond the model call -- rework,
        #: review time, the delay. Set it and each boundary steers to the
        #: break-even failure rate implied by its own tiers' observed costs
        #: instead of a hand-picked constant. Left None, target_failure_rate
        #: applies: the tuner will not invent a number only you can know.
        self.rework_cost_usd = rework_cost_usd

    # -- data ----------------------------------------------------------------

    @staticmethod
    def collect(
        events: Iterable[dict[str, Any]], since: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Join decisions to their outcomes, newest evidence only.

        Only first attempts by an inferring source count. An escalated retry is
        the loop already working, and a decision that came from an engineer hint
        or a min_tier floor was never the classifier's call to get wrong.

        ``since`` is the watermark: outcomes at or before it have already been
        acted on.
        """
        decisions: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, dict[str, Any]] = {}

        for event in events:
            kind = event.get("event")
            key = event.get("decision_id")
            if not key:
                continue
            if kind == "routing_decision":
                decisions[key] = event
            elif kind == "task_outcome":
                outcomes[key] = event

        scored = []
        for key, outcome in outcomes.items():
            decision = decisions.get(key)
            if decision is None:
                continue
            if decision.get("attempt", 1) != 1:
                continue
            if decision.get("source") in {"hint", "floor", "escalation"}:
                continue
            label = str(outcome.get("outcome", "")).lower()
            if label not in FAILURE_OUTCOMES | SUCCESS_OUTCOMES:
                continue
            timestamp = float(outcome.get("timestamp") or 0.0)
            if since is not None and timestamp <= since:
                continue
            scored.append({
                "tier": decision.get("tier"),
                "source": decision.get("source"),
                "confidence": decision.get("confidence"),
                "explored_from": decision.get("explored_from"),
                "outcome": label,
                "failed": label in FAILURE_OUTCOMES,
                "cost_usd": outcome.get("cost_usd"),
                "timestamp": timestamp,
            })
        scored.sort(key=lambda row: row["timestamp"])
        return scored

    # -- tuning --------------------------------------------------------------

    def tune(
        self,
        log: EventSource,
        thresholds: Optional[Thresholds] = None,
        apply: bool = True,
        thresholds_path: str | Path | None = None,
        telemetry: Optional[TelemetrySink] = None,
    ) -> TuningResult:
        """Read the log, decide per boundary, and (by default) persist.

        ``log`` is a path to a JSONL decision log, a TelemetryLog, or any
        iterable of event dicts.
        """
        current = thresholds if thresholds is not None else Thresholds.load(thresholds_path)
        events = list(_events_from(log))
        scored = self.collect(events, since=current.tuned_through)
        if self.window is not None:
            scored = scored[-self.window:]

        # Exploration rows are deliberate downgrades. They say nothing about
        # the tier they ran at and everything about the tier above it, so they
        # are held apart from the ordinary failure rate.
        explored = [r for r in scored if r["source"] == "exploration"]
        normal = [r for r in scored if r["source"] != "exploration"]

        per_tier: dict[str, dict[str, int]] = {}
        for row in normal:
            bucket = per_tier.setdefault(str(row["tier"]), {"total": 0, "failures": 0})
            bucket["total"] += 1
            bucket["failures"] += int(row["failed"])

        failures = sum(1 for row in normal if row["failed"])
        overall_rate = (failures / len(normal)) if normal else None
        # What a tier costs is a property of the model, not of how the routing
        # decision was reached -- so this reads every priced outcome, including
        # escalations and hints that the failure rate deliberately ignores.
        # In a run where the top tier is only ever reached by escalation, those
        # are the only prices it has.
        cost_by_tier = self._mean_cost_by_tier(events)
        before = self._snapshot(current)

        low_target, low_target_source = self._target_for("low", "medium", cost_by_tier)
        high_target, high_target_source = self._target_for("medium", "high", cost_by_tier)

        # Each cut answers to the evidence from the tier it governs, plus any
        # downgrades from the tier above it.
        adjustments = [
            self._decide(
                "low_medium", "tier=low", current.low_medium,
                [r for r in normal if r["tier"] == "low"],
                [r for r in explored if r["explored_from"] == "medium"],
                low_target, low_target_source,
            ),
            self._decide(
                "medium_high", "tier=medium", current.medium_high,
                [r for r in normal if r["tier"] == "medium"],
                [r for r in explored if r["explored_from"] == "high"],
                high_target, high_target_source,
            ),
            # The fallback cut governs when a confident heuristic call is
            # trusted, so it answers to how often confident calls are wrong --
            # excluding the top tier, where consulting the classifier could not
            # have produced a higher route anyway.
            self._decide(
                "llm_fallback", "source=heuristic, tier<high", current.llm_fallback,
                [r for r in normal
                 if r["source"] == "heuristic" and r["tier"] != "high"],
                [], self.target_failure_rate, "fixed", invert=True,
            ),
        ]

        proposed = current.copy()
        proposed.low_medium, proposed.medium_high, proposed.llm_fallback = (
            adjustments[0].after, adjustments[1].after, adjustments[2].after
        )
        self._enforce_gap(proposed, adjustments)
        after = self._snapshot(proposed)

        result = TuningResult(
            adjusted=False,
            reason="",
            samples=len(scored),
            failures=failures,
            failure_rate=round(overall_rate, 4) if overall_rate is not None else None,
            before=before,
            after=after,
            per_tier=per_tier,
            adjustments=adjustments,
            watermark_before=current.tuned_through,
            watermark_after=current.tuned_through,
            explorations=len(explored),
            cost_per_success=self._cost_per_success(scored),
            mean_cost_by_tier={k: round(v, 6) for k, v in cost_by_tier.items()},
        )

        if not scored:
            result.reason = "no new outcomes since the last tuning"
            result.direction = UNCHANGED
            return result

        if before == after:
            moved_any = any(a.direction in {TIGHTEN, RELAX} for a in adjustments)
            result.reason = (
                "already at a guardrail limit" if moved_any
                else "every boundary within its acceptable band or short of samples"
            )
            result.direction = UNCHANGED
            return result

        moved = [a for a in adjustments if a.moved]
        result.adjusted = True
        result.direction = ", ".join(f"{a.name} {a.direction}" for a in moved)
        result.reason = "thresholds adjusted from observed outcomes"

        if apply:
            # Spend the evidence: everything up to here has now been acted on.
            proposed.tuned_through = max(row["timestamp"] for row in scored)
            proposed.save(thresholds_path, tuned_from=_describe(log))
            result.after = self._snapshot(proposed)
            result.watermark_after = proposed.tuned_through
            if telemetry is not None:
                telemetry.emit(build_tuning_event(result.to_dict()))

        return result

    # -- internals -----------------------------------------------------------

    def _decide(
        self,
        name: str,
        evidence: str,
        value: float,
        rows: list[dict[str, Any]],
        explore_rows: list[dict[str, Any]],
        target: float,
        target_source: str = "fixed",
        invert: bool = False,
    ) -> BoundaryAdjustment:
        """Move one cut on its own evidence, or report why it did not.

        Precedence is safety first: observed failures at this tier outrank any
        evidence that the cheaper side has room.

        ``invert`` flips the direction of the move: lowering a tier cut routes
        higher, but *raising* the fallback cut consults the classifier more, so
        the two respond to the same signal in opposite directions.
        """
        failures = sum(1 for r in rows if r["failed"])
        rate = (failures / len(rows)) if rows else None
        enough = len(rows) >= self.min_samples

        explore_successes = sum(1 for r in explore_rows if not r["failed"])
        explore_rate = (
            explore_successes / len(explore_rows) if explore_rows else None
        )
        explored_enough = len(explore_rows) >= self.min_samples

        detail = dict(
            name=name, evidence=evidence, samples=len(rows), failures=failures,
            failure_rate=round(rate, 4) if rate is not None else None,
            target=round(target, 4), target_source=target_source,
            explore_samples=len(explore_rows), explore_successes=explore_successes,
            explore_success_rate=round(explore_rate, 4) if explore_rate is not None else None,
        )

        if not enough and not explored_enough:
            return BoundaryAdjustment(
                direction=INSUFFICIENT, before=value, after=value, **detail
            )

        if enough and rate > target:
            direction, basis = TIGHTEN, "failures at this tier"
            delta = self.step if invert else -self.step
        elif explored_enough and explore_rate >= 1 - target:
            # Downgrades from the tier above keep succeeding: the recommendation
            # was over-provisioned. This is the only *direct* evidence of that
            # -- ordinary outcomes can never supply it.
            direction, basis = RELAX, "downgrades that succeeded"
            delta = -self.step if invert else self.step
        elif enough and rate < target / 2:
            direction, basis = RELAX, "sustained success at this tier"
            delta = -self.step if invert else self.step
        else:
            direction, basis, delta = UNCHANGED, None, 0.0

        moved = min(max(value + delta, self.floor), self.ceiling)
        return BoundaryAdjustment(
            direction=direction, basis=basis, before=value, after=moved, **detail
        )

    # -- cost ----------------------------------------------------------------

    @staticmethod
    def _mean_cost_by_tier(events: list[dict[str, Any]]) -> dict[str, float]:
        """Mean observed cost per tier, from every priced outcome in the log."""
        decisions = {
            e["decision_id"]: e for e in events
            if e.get("event") == "routing_decision" and e.get("decision_id")
        }
        totals: dict[str, list[float]] = {}
        for event in events:
            if event.get("event") != "task_outcome":
                continue
            cost = event.get("cost_usd")
            decision = decisions.get(event.get("decision_id"))
            if cost is None or decision is None:
                continue
            totals.setdefault(str(decision.get("tier")), []).append(float(cost))
        return {tier: sum(v) / len(v) for tier, v in totals.items() if v}

    @staticmethod
    def _cost_per_success(rows: list[dict[str, Any]]) -> Optional[float]:
        """What the loop is actually minimising, when cost is recorded."""
        spend = sum(float(r["cost_usd"]) for r in rows if r.get("cost_usd") is not None)
        successes = sum(1 for r in rows if not r["failed"])
        if not successes or not spend:
            return None
        return round(spend / successes, 6)

    def _target_for(
        self, cheap: str, expensive: str, cost_by_tier: dict[str, float]
    ) -> tuple[float, str]:
        """The failure rate at which trying the cheaper tier stops paying.

        Trying cheap first costs ``c_cheap`` always, plus -- when it fails --
        the expensive run anyway and whatever the failure itself cost you. It
        breaks even against going straight to expensive at

            p* = (c_exp - c_cheap) / (c_exp + rework)

        With no rework cost that rate is close to 1: if a failure were free,
        you would always try cheap first. Rework is the term that makes the
        answer sane, and it is the one number the log cannot tell us -- so
        without it we do not guess, we fall back to the fixed target.
        """
        if self.rework_cost_usd is None:
            return self.target_failure_rate, "fixed"

        c_cheap = cost_by_tier.get(cheap)
        c_exp = cost_by_tier.get(expensive)
        if not c_cheap or not c_exp or c_exp <= c_cheap:
            return self.target_failure_rate, "fixed (no cost data)"

        target = (c_exp - c_cheap) / (c_exp + self.rework_cost_usd)
        return min(max(target, MIN_TARGET), MAX_TARGET), "cost-derived"

    @staticmethod
    def _enforce_gap(proposed: Thresholds, adjustments: list[BoundaryAdjustment]) -> None:
        """Keep the tier cuts apart, yielding whichever one has no evidence."""
        gap = Thresholds.MIN_GAP
        if proposed.medium_high - proposed.low_medium >= gap:
            return

        low_moved, high_moved = adjustments[0].moved, adjustments[1].moved
        if low_moved and not high_moved:
            proposed.low_medium = proposed.medium_high - gap
            adjustments[0].after = proposed.low_medium
        else:
            proposed.medium_high = proposed.low_medium + gap
            adjustments[1].after = proposed.medium_high

    @staticmethod
    def _snapshot(thresholds: Thresholds) -> dict[str, float]:
        return {
            "low_medium": round(thresholds.low_medium, 4),
            "medium_high": round(thresholds.medium_high, 4),
            "llm_fallback": round(thresholds.llm_fallback, 4),
        }
