import json

import pytest

from tierwise.escalation import EscalationPolicy, Outcome
from tierwise.llm_classifier import StubClassifier
from tierwise.mapping import DEFAULT_MODELS, ModelMap
from tierwise.models import Classification, RoutingDecision, Source, TaskSignals, Tier
from tierwise.router import Router, RouterConfig
from tierwise.telemetry import JsonlSink


class RecordingClassifier:
    """Stand-in for a live LLM classifier."""

    def __init__(self, tier=Tier.HIGH, confidence=0.9):
        self.tier = tier
        self.confidence = confidence
        self.calls = []

    def classify(self, signals):
        self.calls.append(signals)
        return Classification(
            tier=self.tier,
            confidence=self.confidence,
            rationale="recorded",
            source=Source.LLM,
        )


@pytest.fixture
def classifier():
    return RecordingClassifier()


@pytest.fixture
def router(classifier):
    return Router(model_map=ModelMap(dict(DEFAULT_MODELS)), classifier=classifier)


def test_hint_wins_over_inference(router, classifier):
    signals = TaskSignals(
        description="rename a local variable",
        category="rename",
        file_count=1,
        lines_changed=1,
        tier_hint=Tier.HIGH,
    )
    decision = router.route(signals)
    assert decision.tier is Tier.HIGH
    assert decision.source is Source.HINT
    assert decision.confidence == 1.0
    assert classifier.calls == []


def test_hints_can_be_distrusted(classifier):
    router = Router(classifier=classifier, config=RouterConfig(trust_hints=False))
    signals = TaskSignals(category="typo", file_count=1, lines_changed=1, tier_hint=Tier.HIGH)
    decision = router.route(signals)
    assert decision.source is not Source.HINT
    assert decision.tier is Tier.LOW


def test_confident_heuristic_skips_the_classifier(router, classifier):
    signals = TaskSignals(description="fix a typo", category="typo", lines_changed=1)
    decision = router.route(signals)
    assert decision.source is Source.HEURISTIC
    assert decision.tier is Tier.LOW
    assert classifier.calls == []


def test_ambiguous_task_falls_through_to_classifier(classifier):
    """A score parked on a tier boundary must reach the LLM layer."""
    router = Router(
        classifier=classifier,
        config=RouterConfig(confidence_threshold=1.01),  # force every task to be ambiguous
    )
    decision = router.route(TaskSignals(category="feature", file_count=3, lines_changed=90))
    assert len(classifier.calls) == 1
    assert decision.source is Source.LLM
    assert decision.tier is Tier.HIGH


def test_min_tier_floors_the_decision(router):
    signals = TaskSignals(category="typo", lines_changed=1, min_tier=Tier.MEDIUM)
    decision = router.route(signals)
    assert decision.tier is Tier.MEDIUM
    assert decision.source is Source.FLOOR
    assert "min_tier" in decision.rationale


def test_min_tier_never_lowers_a_decision(router):
    signals = TaskSignals(
        category="architecture", file_count=30, lines_changed=2000,
        dependency_depth=6, requires_context=True, ambiguity=0.9,
        min_tier=Tier.LOW,
    )
    assert router.route(signals).tier is Tier.HIGH


def test_model_map_resolves_and_env_overrides(monkeypatch):
    monkeypatch.setenv("TIERWISE_MODEL_LOW", "my-cheap-model")
    mapping = ModelMap.from_env()
    assert mapping.model_for(Tier.LOW) == "my-cheap-model"
    assert mapping.model_for(Tier.HIGH) == DEFAULT_MODELS[Tier.HIGH]
    assert set(mapping.as_dict()) == {"low", "medium", "high"}


def test_stub_classifier_resolves_upward_without_network():
    stub = StubClassifier()
    result = stub.classify(TaskSignals(category="feature", file_count=3, lines_changed=90))
    assert result.source is Source.LLM
    assert result.tier is not Tier.LOW
    assert "stub" in result.rationale


def test_stub_classifier_does_not_exceed_high():
    stub = StubClassifier()
    result = stub.classify(
        TaskSignals(category="architecture", file_count=40, lines_changed=5000,
                    dependency_depth=8, requires_context=True, ambiguity=1.0)
    )
    assert result.tier is Tier.HIGH


def test_telemetry_records_one_event_per_decision(tmp_path, classifier):
    log = tmp_path / "routing.jsonl"
    router = Router(classifier=classifier, telemetry=JsonlSink(log))
    router.route(TaskSignals(category="typo", lines_changed=1))
    router.route(TaskSignals(category="architecture", file_count=20, lines_changed=900,
                             dependency_depth=5, requires_context=True, ambiguity=0.9))

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 2
    assert {e["event"] for e in events} == {"routing_decision"}
    assert all("elapsed_ms" in e and "timestamp" in e for e in events)
    assert events[0]["tier"] == "low"
    assert events[1]["tier"] == "high"


def test_decision_serializes_cleanly(router):
    decision = router.route(TaskSignals(description="x", category="typo", lines_changed=1))
    payload = decision.to_dict()
    json.dumps(payload)  # must be JSON-safe
    assert payload["model"] == DEFAULT_MODELS[Tier.LOW]
    assert payload["signals"]["tier_hint"] is None


def test_tier_hint_accepts_a_string():
    signals = TaskSignals(tier_hint="high", min_tier="low")
    assert signals.tier_hint is Tier.HIGH
    assert signals.min_tier is Tier.LOW
