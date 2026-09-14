"""The router: signals in, model out.

Precedence, highest first:

1. An explicit engineer tier hint. Engineers usually know when a task is hairy,
   and that signal is cheaper and more accurate than anything inferred.
2. The heuristic scorer, when it is confident.
3. The LLM classifier, for the ambiguous middle only.

A ``min_tier`` floor is applied last, so a task can be pinned above whatever the
classifiers concluded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from . import heuristics
from .escalation import EscalationPolicy, Outcome
from .llm_classifier import Classifier, default_classifier
from .mapping import ModelMap
from .models import RoutingDecision, Source, TaskSignals, Tier
from .telemetry import NullSink, TelemetrySink, build_event

# Heuristic confidence below this is treated as ambiguous and handed to the LLM.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class RouterConfig:
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    trust_hints: bool = True
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)


class Router:
    def __init__(
        self,
        model_map: Optional[ModelMap] = None,
        classifier: Optional[Classifier] = None,
        config: Optional[RouterConfig] = None,
        telemetry: Optional[TelemetrySink] = None,
    ) -> None:
        self.model_map = model_map or ModelMap.from_env()
        self.classifier = classifier or default_classifier()
        self.config = config or RouterConfig()
        self.telemetry = telemetry or NullSink()

    # -- public API ----------------------------------------------------------

    def route(self, signals: TaskSignals) -> RoutingDecision:
        """Route one task to a tier and model."""
        started = time.perf_counter()

        if self.config.trust_hints and signals.tier_hint is not None:
            tier, confidence, source, rationale = (
                signals.tier_hint,
                1.0,
                Source.HINT,
                f"engineer hint: {signals.tier_hint.value}",
            )
        else:
            classification, _score = heuristics.classify(signals)
            if classification.confidence >= self.config.confidence_threshold:
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

        decision = RoutingDecision(
            tier=tier,
            model=self.model_map.model_for(tier),
            confidence=confidence,
            source=source,
            rationale=rationale,
            signals=signals,
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
            decision, Outcome(outcome) if isinstance(outcome, str) else outcome,
            self.model_map.model_for,
        )
        if escalated is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.telemetry.emit(build_event(escalated, elapsed_ms))
        return escalated

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _apply_floor(
        tier: Tier, signals: TaskSignals, rationale: str
    ) -> tuple[Tier, str]:
        floor = signals.min_tier
        if floor is not None and tier.rank < floor.rank:
            return floor, f"{rationale}; raised to min_tier={floor.value}"
        return tier, rationale
