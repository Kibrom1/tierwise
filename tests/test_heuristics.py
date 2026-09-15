import pytest

from tierwise.heuristics import (
    BOUNDARY_MARGIN,
    LOW_MEDIUM_BOUNDARY,
    MEDIUM_HIGH_BOUNDARY,
    classify,
    confidence_for_score,
    score_task,
    tier_for_score,
)
from tierwise.models import Source, TaskSignals, Tier


def test_trivial_task_scores_low():
    signals = TaskSignals(description="fix a typo", category="typo", file_count=1, lines_changed=2)
    classification, score = classify(signals)
    assert classification.tier is Tier.LOW
    assert score.value < LOW_MEDIUM_BOUNDARY
    assert classification.source is Source.HEURISTIC


def test_sprawling_ambiguous_task_scores_high():
    signals = TaskSignals(
        description="rework how billing state is persisted",
        category="architecture",
        file_count=25,
        lines_changed=1200,
        dependency_depth=5,
        requires_context=True,
        ambiguity=0.8,
    )
    classification, score = classify(signals)
    assert classification.tier is Tier.HIGH
    assert score.value >= MEDIUM_HIGH_BOUNDARY


def test_ordinary_feature_lands_medium():
    signals = TaskSignals(
        description="add a filter param to the reports endpoint",
        category="feature",
        file_count=3,
        lines_changed=90,
        dependency_depth=1,
        requires_context=True,
    )
    classification, _ = classify(signals)
    assert classification.tier is Tier.MEDIUM


def test_category_alone_does_not_decide_tier():
    """Same category, very different context -> different tiers."""
    small = TaskSignals(category="unit_test", file_count=1, lines_changed=15)
    large = TaskSignals(
        category="unit_test",
        file_count=14,
        lines_changed=600,
        dependency_depth=4,
        requires_context=True,
        ambiguity=0.5,
    )
    assert classify(small)[0].tier is not classify(large)[0].tier


def test_greenfield_reduces_score():
    base = dict(category="feature", file_count=3, lines_changed=120)
    brownfield, _ = classify(TaskSignals(**base))
    greenfield, _ = classify(TaskSignals(**base, is_greenfield=True))
    assert score_task(TaskSignals(**base, is_greenfield=True)).value < score_task(
        TaskSignals(**base)
    ).value
    assert greenfield.tier.rank <= brownfield.tier.rank


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, Tier.LOW),
        (LOW_MEDIUM_BOUNDARY - 0.01, Tier.LOW),
        (LOW_MEDIUM_BOUNDARY, Tier.MEDIUM),
        (MEDIUM_HIGH_BOUNDARY - 0.01, Tier.MEDIUM),
        (MEDIUM_HIGH_BOUNDARY, Tier.HIGH),
        (1.0, Tier.HIGH),
    ],
)
def test_tier_boundaries(value, expected):
    assert tier_for_score(value) is expected


def test_confidence_collapses_at_boundaries():
    assert confidence_for_score(LOW_MEDIUM_BOUNDARY) == pytest.approx(0.0)
    assert confidence_for_score(MEDIUM_HIGH_BOUNDARY) == pytest.approx(0.0)
    assert confidence_for_score(0.0) > 0.5
    assert confidence_for_score(1.0) > 0.5


def test_confidence_at_margin_is_about_half():
    at_margin = LOW_MEDIUM_BOUNDARY + BOUNDARY_MARGIN
    assert confidence_for_score(at_margin) == pytest.approx(0.5, abs=0.01)


def test_score_is_clamped_and_drivers_ranked():
    signals = TaskSignals(
        file_count=999, lines_changed=99999, dependency_depth=99,
        requires_context=True, ambiguity=1.0, category="architecture",
    )
    score = score_task(signals)
    assert 0.0 <= score.value <= 1.0
    drivers = score.top_drivers(3)
    assert len(drivers) == 3
    assert abs(drivers[0][1]) >= abs(drivers[-1][1])


def test_invalid_signals_rejected():
    with pytest.raises(ValueError):
        TaskSignals(ambiguity=1.5)
    with pytest.raises(ValueError):
        TaskSignals(file_count=-1)
    with pytest.raises(ValueError):
        TaskSignals(lines_changed=-3)


# -- evidence: no signals is not the same as a simple task ---------------------

from tierwise import Router, TaskSignals as _Signals, Tier as _Tier  # noqa: E402
from tierwise.heuristics import classify as _classify, has_evidence  # noqa: E402


def test_description_only_is_not_confident():
    """Score 0.0 used to read as confidence 1.00 -- certainty from nothing."""
    signals = _Signals(description="migrate billing to a new payment provider")
    assert not has_evidence(signals)
    classification, _ = _classify(signals)
    assert classification.confidence == 0.0
    assert "description only" in classification.rationale


def test_description_only_goes_to_the_classifier():
    decision = Router().route(_Signals(description="migrate billing to a new payment provider"))
    assert decision.source.value == "llm"
    # The offline stub resolves upward rather than guessing cheap.
    assert decision.tier is not _Tier.LOW


@pytest.mark.parametrize("signals", [
    _Signals(category="typo"),
    _Signals(lines_changed=2),
    _Signals(file_count=4),
    _Signals(requires_context=True),
    _Signals(ambiguity=0.2),
    _Signals(is_greenfield=True),
    _Signals(metadata={"measured": True}),
])
def test_any_real_signal_counts_as_evidence(signals):
    assert has_evidence(signals)


def test_an_unknown_category_is_not_evidence():
    assert not has_evidence(_Signals(category="billing"))
