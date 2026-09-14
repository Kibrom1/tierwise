"""The router: signals in, model out.

Precedence, highest first:

1. An explicit engineer tier hint. Engineers usually know when a task is hairy,
   and that signal is cheaper and more accurate than anything inferred.
2. The heuristic scorer, when it is confident.
3. The LLM classifier, for the ambiguous middle only.

A ``min_tier`` floor is applied last, so a task can be pinned above whatever the
classifiers concluded.

The Router itself is stateless -- one instance is safe to share across
concurrent tasks. Per-task state (step index, tier history, the decision an
outcome refers to) belongs to RoutingSession.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import heuristics
from .escalation import EscalationPolicy, Outcome
from .llm_classifier import Classifier, default_classifier
from .mapping import ModelMap
from .models import RoutingDecision, Source, TaskSignals, Tier
from .telemetry import NullSink, TelemetrySink, build_event
from .thresholds import Thresholds

# Kept for callers that imported it before thresholds became tunable.
DEFAULT_CONFIDENCE_THRESHOLD = Thresholds().llm_fallback


@dataclass
class RouterConfig:
    #: None means "use the tuned Thresholds value" -- set it to pin a threshold
    #: that the outer loop cannot move.
    confidence_threshold: Optional[float] = None
    trust_hints: bool = True
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)

    #: Fraction of eligible decisions deliberately routed one tier *below* the
    #: recommendation, to find out whether the cheaper tier would have done.
    #:
    #: Outcomes are only ever observed at the tier actually used, so
    #: under-provisioning is measurable and over-provisioning is not: the top
    #: tier never fails work a cheap tier could have handled. Without
    #: exploration the loop can only infer that the cheap side has room from the
    #: *absence* of failures. This buys that evidence directly, and pays for it
    #: in occasional avoidable retries -- so it is off by default.
    #:
    #: A downgrade that fails escalates straight back up, which is the safety
    #: net that makes exploring affordable at all.
    exploration_rate: float = 0.0


class Router:
    def __init__(
        self,
        model_map: Optional[ModelMap] = None,
        classifier: Optional[Classifier] = None,
        config: Optional[RouterConfig] = None,
        telemetry: Optional[TelemetrySink] = None,
        thresholds: Optional[Thresholds] = None,
        rng: Optional[Callable[[], float]] = None,
    ) -> None:
        self.model_map = model_map or ModelMap.from_env()
        self.classifier = classifier or default_classifier()
        self.config = config or RouterConfig()
        # Not `telemetry or NullSink()`: a sink that defines __len__ (such as
        # an empty TelemetryLog) is falsy, and would be silently discarded.
        self.telemetry = telemetry if telemetry is not None else NullSink()
        #: Loaded from disk when not supplied, so thresholds the tuner wrote in
        #: an earlier run are in force from the first decision of this one.
        self.thresholds = thresholds if thresholds is not None else Thresholds.load()
        #: Injectable so exploration is testable without flaky randomness.
        self.rng = rng if rng is not None else random.random

    @property
    def confidence_threshold(self) -> float:
        if self.config.confidence_threshold is not None:
            return self.config.confidence_threshold
        return self.thresholds.llm_fallback

    # -- public API ----------------------------------------------------------

    def route(
        self,
        signals: TaskSignals,
        session_id: Optional[str] = None,
        step_index: Optional[int] = None,
    ) -> RoutingDecision:
        """Route one task -- or one step of one -- to a tier and model."""
        started = time.perf_counter()

        if self.config.trust_hints and signals.tier_hint is not None:
            tier, confidence, source, rationale = (
                signals.tier_hint,
                1.0,
                Source.HINT,
                f"engineer hint: {signals.tier_hint.value}",
            )
        else:
            classification, _score = heuristics.classify(signals, self.thresholds)
            if classification.confidence >= self.confidence_threshold:
                tier, confidence, source, rationale = (
                    classification.tier,
                    classification.confidence,
                    Source.HEURISTIC,
                    classification.rationale,
                )
            else:
                fallback = self.classifier.classify(signals)
                tier, confidence, source, rationale = (
                    fallback.tier,
                    fallback.confidence,
                    fallback.source,
                    f"{classification.rationale}; ambiguous -> {fallback.rationale}",
                )

        floored_tier, rationale = self._apply_floor(tier, signals, rationale)
        if floored_tier is not tier:
            tier, source, confidence = floored_tier, Source.FLOOR, 1.0

        explored_from = None
        if self._should_explore(tier, signals, source):
            explored_from = tier
            tier = tier - 1
            source = Source.EXPLORATION
            rationale = (
                f"{rationale}; explored down from {explored_from.value} "
                f"to measure whether {tier.value} suffices"
            )
            confidence = 0.0

        decision = RoutingDecision(
            tier=tier,
            model=self.model_map.model_for(tier),
            confidence=confidence,
            source=source,
            rationale=rationale,
            signals=signals,
            session_id=session_id,
            step_index=step_index,
            explored_from=explored_from,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.telemetry.emit(build_event(decision, elapsed_ms))
        return decision

    def report_outcome(
        self, decision: RoutingDecision, outcome: Outcome | str
    ) -> RoutingDecision | None:
        """Feed back what happened. Returns a re-route if one is warranted."""
        started = time.perf_counter()
        escalated = self.config.escalation.escalate(
            decision,
            Outcome(outcome) if isinstance(outcome, str) else outcome,
            self.model_map.model_for,
        )
        if escalated is not None:
            escalated.session_id = decision.session_id
            escalated.step_index = decision.step_index
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.telemetry.emit(build_event(escalated, elapsed_ms))
        return escalated

    # -- internals -----------------------------------------------------------

    def _should_explore(self, tier: Tier, signals: TaskSignals, source: Source) -> bool:
        """Never explore against someone's explicit instruction.

        A tier hint or a min_tier floor is a person saying what this task needs;
        quietly routing below it to gather data would be spending their task on
        our curiosity.
        """
        if self.config.exploration_rate <= 0:
            return False
        if tier is Tier.LOW:
            return False            # nothing below to try
        if source is Source.FLOOR or signals.min_tier is not None:
            return False
        if signals.tier_hint is not None and self.config.trust_hints:
            return False
        return self.rng() < self.config.exploration_rate

    @staticmethod
    def _apply_floor(
        tier: Tier, signals: TaskSignals, rationale: str
    ) -> tuple[Tier, str]:
        floor = signals.min_tier
        if floor is not None and tier < floor:
            return floor, f"{rationale}; raised to min_tier={floor.value}"
        return tier, rationale
