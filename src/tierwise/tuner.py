"""ThresholdTuner -- the outer loop.

Telemetry that nothing reads is decoration. The tuner closes the circuit: it
reads the decision log, joins each decision to the outcome reported for it, and
moves the thresholds that produced those decisions.

It reads the *persisted* log across sessions rather than one session's memory,
and it writes the thresholds back to disk. Both halves matter -- a tuner that
learns from a single in-flight task and forgets at process exit never
accumulates enough evidence to be right, and never gets to apply it.

Failure is treated asymmetrically on purpose. Under-provisioning shows up as a
bad outcome and is measurable; over-provisioning is invisible in outcome data --
Opus never fails a task Haiku could have done. So the loop tightens on observed
failures and relaxes on their sustained absence, which is the only evidence that
the cheap side has room.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .telemetry import TelemetrySink, build_tuning_event, read_events
from .thresholds import Thresholds

#: Outcomes that mean the tier was too low. ERROR is excluded deliberately: a
#: failed API call is an infrastructure problem and says nothing about tier.
FAILURE_OUTCOMES = {"insufficient", "rejected"}
SUCCESS_OUTCOMES = {"success"}


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThresholdTuner:
    """Moves the cuts, within guardrails, from observed outcomes.

    Guardrails are not optional decoration on an auto-applying loop -- they are
    what keeps a bad labelling week from walking the thresholds somewhere they
    cannot walk back from: a minimum sample count before acting, one bounded
    step per run, hard floor and ceiling, an enforced gap between the tier cuts,
    and a recorded tuning event for every change.
    """

    def __init__(
        self,
        step: float = 0.03,
        min_samples: int = 20,
        target_failure_rate: float = 0.10,
        floor: float = 0.10,
        ceiling: float = 0.90,
    ) -> None:
        self.step = step
        self.min_samples = min_samples
        self.target_failure_rate = target_failure_rate
        self.floor = floor
        self.ceiling = ceiling

    # -- data ----------------------------------------------------------------

    @staticmethod
    def collect(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Join decisions to their outcomes.

        Only first attempts by an inferring source count. An escalated retry is
        the loop already working, and a decision that came from an engineer hint
        or a min_tier floor was never the classifier's call to get wrong.
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
            scored.append({
                "tier": decision.get("tier"),
                "source": decision.get("source"),
                "confidence": decision.get("confidence"),
                "outcome": label,
                "failed": label in FAILURE_OUTCOMES,
                "cost_usd": outcome.get("cost_usd"),
            })
        return scored

    # -- tuning --------------------------------------------------------------

    def tune(
        self,
        log_path: str | Path,
        thresholds: Optional[Thresholds] = None,
        apply: bool = True,
        thresholds_path: str | Path | None = None,
        telemetry: Optional[TelemetrySink] = None,
    ) -> TuningResult:
        """Read the log, decide, and (by default) persist the new thresholds."""
        current = thresholds if thresholds is not None else Thresholds.load(thresholds_path)
        scored = self.collect(read_events(log_path))

        per_tier: dict[str, dict[str, int]] = {}
        for row in scored:
            bucket = per_tier.setdefault(str(row["tier"]), {"total": 0, "failures": 0})
            bucket["total"] += 1
            bucket["failures"] += int(row["failed"])

        if len(scored) < self.min_samples:
            return TuningResult(
                adjusted=False,
                reason=f"insufficient samples ({len(scored)}/{self.min_samples})",
                samples=len(scored),
                failures=sum(1 for r in scored if r["failed"]),
                before=self._snapshot(current),
                after=self._snapshot(current),
                per_tier=per_tier,
            )

        failures = sum(1 for row in scored if row["failed"])
        failure_rate = failures / len(scored)
        proposed = current.copy()

        if failure_rate > self.target_failure_rate:
            # Too many under-provisioned steps: make it easier to land in a
            # higher tier, and consult the classifier more readily.
            proposed.low_medium = max(current.low_medium - self.step, self.floor)
            proposed.medium_high = max(
                current.medium_high - self.step, proposed.low_medium + Thresholds.MIN_GAP
            )
            proposed.llm_fallback = min(current.llm_fallback + self.step, self.ceiling)
            direction = "tightened (failure rate above target -- routing higher)"
        elif failure_rate < self.target_failure_rate / 2:
            # Sustained success is the only evidence that the cheap side has
            # room. Give some back.
            proposed.medium_high = min(current.medium_high + self.step, self.ceiling)
            proposed.low_medium = min(
                current.low_medium + self.step, proposed.medium_high - Thresholds.MIN_GAP
            )
            proposed.llm_fallback = max(current.llm_fallback - self.step, self.floor)
            direction = "relaxed (failure rate well under target -- routing cheaper)"
        else:
            return TuningResult(
                adjusted=False,
                reason="failure rate within the acceptable band",
                samples=len(scored),
                failures=failures,
                failure_rate=round(failure_rate, 4),
                direction="unchanged",
                before=self._snapshot(current),
                after=self._snapshot(current),
                per_tier=per_tier,
            )

        before = self._snapshot(current)
        after = self._snapshot(proposed)

        if before == after:
            return TuningResult(
                adjusted=False,
                reason="already at a guardrail limit",
                samples=len(scored),
                failures=failures,
                failure_rate=round(failure_rate, 4),
                direction=direction,
                before=before,
                after=after,
                per_tier=per_tier,
            )

        result = TuningResult(
            adjusted=True,
            reason="thresholds adjusted from observed outcomes",
            samples=len(scored),
            failures=failures,
            failure_rate=round(failure_rate, 4),
            direction=direction,
            before=before,
            after=after,
            per_tier=per_tier,
        )

        if apply:
            proposed.save(thresholds_path, tuned_from=str(log_path))
            result.after = self._snapshot(proposed)
            if telemetry is not None:
                telemetry.emit(build_tuning_event(result.to_dict()))

        return result

    @staticmethod
    def _snapshot(thresholds: Thresholds) -> dict[str, float]:
        return {
            "low_medium": round(thresholds.low_medium, 4),
            "medium_high": round(thresholds.medium_high, 4),
            "llm_fallback": round(thresholds.llm_fallback, 4),
        }
