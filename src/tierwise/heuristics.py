"""Heuristic complexity scoring.

The scorer turns task signals into a 0.0-1.0 complexity score, then maps that
score onto a tier. Confidence is derived from how far the score sits from the
nearest tier boundary: a score parked on a boundary is exactly the ambiguous
case that should fall through to the LLM classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import Classification, Source, TaskSignals, Tier
from .thresholds import DEFAULT_THRESHOLDS, Thresholds

# Default boundaries on the normalized 0..1 complexity score. These are the
# starting point only -- the live values come from a Thresholds instance, which
# the ThresholdTuner rewrites from observed outcomes.
LOW_MEDIUM_BOUNDARY = DEFAULT_THRESHOLDS.low_medium
MEDIUM_HIGH_BOUNDARY = DEFAULT_THRESHOLDS.medium_high
BOUNDARY_MARGIN = DEFAULT_THRESHOLDS.boundary_margin

# Category priors. Deliberately small: category is a weak signal on its own and
# is only here to break ties, not to drive the decision.
CATEGORY_PRIORS: dict[str, float] = {
    "typo": -0.10,
    "docstring": -0.10,
    "comment": -0.10,
    "formatting": -0.10,
    "rename": -0.05,
    "unit_test": 0.00,
    "bugfix": 0.05,
    "refactor": 0.08,
    "integration_test": 0.08,
    "feature": 0.12,
    "migration": 0.15,
    "architecture": 0.20,
    "security_review": 0.20,
    "debug_production": 0.22,
}

# Normalizer. Below the theoretical max (~15) so a task does not have to max out
# every signal to reach the top tier, but high enough that an ordinary multi-file
# refactor stays out of it.
_MAX_RAW = 13.0


@dataclass
class Score:
    """A scored task: the raw contributions plus the normalized result."""

    value: float
    contributions: dict[str, float]

    def top_drivers(self, n: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(
            (item for item in self.contributions.items() if abs(item[1]) > 1e-9),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return ranked[:n]


def _bucket(value: int, thresholds: tuple[int, ...], weights: tuple[float, ...]) -> float:
    for threshold, weight in zip(thresholds, weights):
        if value <= threshold:
            return weight
    return weights[-1]


def score_task(signals: TaskSignals) -> Score:
    """Score a task's complexity on a normalized 0.0-1.0 scale."""
    contributions: dict[str, float] = {
        "file_count": _bucket(signals.file_count, (1, 3, 10), (0.0, 1.0, 2.0, 3.0)),
        "lines_changed": _bucket(signals.lines_changed, (20, 100, 400), (0.0, 1.0, 2.0, 3.0)),
        "dependency_depth": _bucket(signals.dependency_depth, (0, 2), (0.0, 1.0, 2.0)),
        "requires_context": 1.5 if signals.requires_context else 0.0,
        "ambiguity": signals.ambiguity * 3.0,
        "greenfield": -0.5 if signals.is_greenfield else 0.0,
    }

    prior = CATEGORY_PRIORS.get((signals.category or "").strip().lower())
    contributions["category_prior"] = (prior or 0.0) * _MAX_RAW

    raw = sum(contributions.values())
    normalized = max(0.0, min(1.0, raw / _MAX_RAW))
    return Score(value=normalized, contributions=contributions)


def has_evidence(signals: TaskSignals) -> bool:
    """Did the caller tell the scorer anything beyond a free-text description?

    Every structural signal defaults to its "simplest" value, so a task that
    arrives with nothing but a description scores 0.0 -- and 0.0 sits as far
    from a cut as a score can, which the distance-based confidence reads as
    certainty. That made the least-informed decision the most confident one:
    "migrate billing to a new payment provider" went to the cheapest tier at
    confidence 1.00, and the classifier -- the only layer that reads the
    description -- was never asked. A recognised category counts as evidence,
    if weak; an unrecognised one does not.
    """
    if signals.metadata.get("measured"):
        return True     # zeros read off a real diff are a measurement
    if (signals.category or "").strip().lower() in CATEGORY_PRIORS:
        return True
    return bool(
        signals.file_count != 1
        or signals.lines_changed
        or signals.dependency_depth
        or signals.requires_context
        or signals.ambiguity
        or signals.is_greenfield
    )


def tier_for_score(value: float, thresholds: Optional[Thresholds] = None) -> Tier:
    t = thresholds or DEFAULT_THRESHOLDS
    if value < t.low_medium:
        return Tier.LOW
    if value < t.medium_high:
        return Tier.MEDIUM
    return Tier.HIGH


def confidence_for_score(value: float, thresholds: Optional[Thresholds] = None) -> float:
    """Confidence = normalized distance to the nearest tier boundary.

    Continuous by construction: a score sitting on a cut is confidence 0, and it
    climbs smoothly from there. That matters for the outer loop -- if confidence
    could only take a handful of discrete values, moving a threshold by a small
    step would either change nothing or change everything.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    distance = min(abs(value - t.low_medium), abs(value - t.medium_high))
    # boundary_margin away from a cut is where confidence reaches ~0.5;
    # anything further scales up toward 1.0.
    return max(0.0, min(1.0, distance / (t.boundary_margin * 2)))


def classify(
    signals: TaskSignals, thresholds: Optional[Thresholds] = None
) -> tuple[Classification, Score]:
    """Classify a task heuristically, returning the verdict and its score."""
    t = thresholds or DEFAULT_THRESHOLDS
    score = score_task(signals)
    tier = tier_for_score(score.value, t)
    confidence = confidence_for_score(score.value, t)

    drivers = score.top_drivers()
    if not has_evidence(signals):
        # Nothing to be confident about. Zero confidence hands the call to the
        # classifier, which reads the description; the heuristic tier stays in
        # the rationale for the record.
        confidence = 0.0
        driver_text = "no signals supplied, description only"
    elif drivers:
        driver_text = ", ".join(f"{name}={weight:+.2f}" for name, weight in drivers)
    else:
        driver_text = "no strong signals"
    rationale = f"complexity score {score.value:.2f} -> {tier.value} (drivers: {driver_text})"

    return (
        Classification(
            tier=tier,
            confidence=confidence,
            rationale=rationale,
            source=Source.HEURISTIC,
        ),
        score,
    )
