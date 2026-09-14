"""Covers the pieces ported from the alternative implementation:
an injectable call_fn, Tier ordering/arithmetic, and the in-memory log.
"""

import json

import pytest

from tierwise import (
    LLMClassifier,
    Outcome,
    Router,
    RoutingSession,
    Source,
    StubClassifier,
    TaskSignals,
    TelemetryLog,
    ThresholdTuner,
    Thresholds,
    Tier,
    make_anthropic_call_fn,
)
from tierwise.llm_classifier import AnthropicClassifier

EASY = dict(category="typo", file_count=1, lines_changed=2)


# -- Tier ordering and arithmetic -------------------------------------------

def test_tier_orders_by_rank_not_by_string():
    """"medium" sorts after "high" as text -- ordering must not inherit that."""
    assert Tier.LOW < Tier.MEDIUM < Tier.HIGH
    assert Tier.HIGH > Tier.MEDIUM
    assert Tier.MEDIUM <= Tier.MEDIUM
    assert sorted([Tier.HIGH, Tier.LOW, Tier.MEDIUM]) == [Tier.LOW, Tier.MEDIUM, Tier.HIGH]


def test_tier_arithmetic_clamps():
    assert Tier.LOW + 1 is Tier.MEDIUM
    assert Tier.MEDIUM - 1 is Tier.LOW
    assert Tier.HIGH + 1 is Tier.HIGH
    assert Tier.LOW - 5 is Tier.LOW
    assert Tier.LOW + 99 is Tier.HIGH


def test_tier_arithmetic_rejects_non_ints():
    with pytest.raises(TypeError):
        Tier.LOW + 1.5
    with pytest.raises(TypeError):
        Tier.LOW + "1"


def test_tier_values_stay_strings_for_the_log():
    """The tuner parses logs written by earlier runs; humans read them too."""
    assert Tier.HIGH.value == "high"
    assert json.loads(json.dumps({"tier": Tier.HIGH.value}))["tier"] == "high"


# -- injectable call_fn ------------------------------------------------------

def test_call_fn_is_injectable():
    calls = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return '{"tier": "high", "confidence": 0.88, "rationale": "sprawling"}'

    result = LLMClassifier(call_fn=fake).classify(TaskSignals(**EASY))
    assert result.tier is Tier.HIGH
    assert result.confidence == pytest.approx(0.88)
    assert result.source is Source.LLM
    assert "sprawling" in result.rationale
    assert len(calls) == 1
    assert "Respond with JSON only" in calls[0]


def test_prompt_carries_the_signals():
    prompt = LLMClassifier.build_prompt(
        TaskSignals(description="rework billing", category="architecture", file_count=12)
    )
    assert "rework billing" in prompt
    assert '"file_count": 12' in prompt


def test_fenced_json_is_parsed():
    fenced = '```json\n{"tier": "medium", "confidence": 0.6}\n```'
    result = LLMClassifier(call_fn=lambda _: fenced).classify(TaskSignals(**EASY))
    assert result.tier is Tier.MEDIUM


def test_prose_around_json_is_tolerated():
    noisy = 'Sure!\n{"tier": "low", "confidence": 0.9}\nHope that helps.'
    assert LLMClassifier(call_fn=lambda _: noisy).classify(TaskSignals(**EASY)).tier is Tier.LOW


def test_transport_failure_degrades_to_the_fallback():
    def boom(_: str) -> str:
        raise ConnectionError("no route to host")

    result = LLMClassifier(call_fn=boom).classify(TaskSignals(**EASY))
    assert "unavailable (ConnectionError)" in result.rationale
    assert "stub" in result.rationale


def test_unparseable_response_degrades():
    result = LLMClassifier(call_fn=lambda _: "I'd say medium?").classify(TaskSignals(**EASY))
    assert "unavailable" in result.rationale


def test_missing_call_fn_degrades_rather_than_raising():
    assert LLMClassifier().classify(TaskSignals(**EASY)).tier is not None


def test_custom_fallback_is_used():
    class Always(StubClassifier):
        def classify(self, signals):
            result = super().classify(signals)
            result.rationale = "custom fallback"
            return result

    result = LLMClassifier(call_fn=lambda _: "junk", fallback=Always()).classify(
        TaskSignals(**EASY)
    )
    assert "custom fallback" in result.rationale


def test_anthropic_classifier_builds_its_call_fn_lazily():
    """Constructing one without the SDK or a key must not raise."""
    classifier = AnthropicClassifier()
    assert classifier.call_fn is None
    result = classifier.classify(TaskSignals(**EASY))   # degrades, does not raise
    assert result.tier is not None


def test_make_anthropic_call_fn_with_an_injected_client():
    class Block:
        text = '{"tier": "high", "confidence": 0.91}'

    class Response:
        content = [Block()]

    class Client:
        def __init__(self):
            self.kwargs = None

        class messages:  # noqa: N801
            pass

    client = Client()

    class Messages:
        def create(self, **kwargs):
            client.kwargs = kwargs
            return Response()

    client.messages = Messages()

    call_fn = make_anthropic_call_fn(client=client, model="my-model", max_tokens=64)
    raw = call_fn("hello")
    assert json.loads(raw)["tier"] == "high"
    assert client.kwargs["model"] == "my-model"
    assert client.kwargs["max_tokens"] == 64

    result = LLMClassifier(call_fn=call_fn).classify(TaskSignals(**EASY))
    assert result.tier is Tier.HIGH


def test_router_accepts_an_injected_llm_classifier():
    classifier = LLMClassifier(call_fn=lambda _: '{"tier": "high", "confidence": 0.9}')
    router = Router(classifier=classifier)
    router.config.confidence_threshold = 1.01   # force the ambiguous path
    assert router.route(TaskSignals(**EASY)).tier is Tier.HIGH


# -- in-memory TelemetryLog --------------------------------------------------

def test_memory_log_collects_events():
    log = TelemetryLog()
    session = RoutingSession(router=Router(telemetry=log))
    session.route_step(TaskSignals(**EASY))
    session.mark_outcome(Outcome.SUCCESS, cost_usd=0.002)

    assert len(log) == 2
    assert [e["event"] for e in log] == ["routing_decision", "task_outcome"]
    assert len(log.of_kind("routing_decision")) == 1


def test_memory_log_dump_matches_jsonl():
    log = TelemetryLog()
    RoutingSession(router=Router(telemetry=log)).route_step(TaskSignals(**EASY))
    lines = log.dump().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "routing_decision"


def test_empty_memory_log_dumps_empty():
    assert TelemetryLog().dump() == ""


def test_memory_log_writes_to_disk(tmp_path):
    log = TelemetryLog()
    RoutingSession(router=Router(telemetry=log)).route_step(TaskSignals(**EASY))
    path = log.write(tmp_path / "nested" / "out.jsonl")
    assert path.exists()
    assert json.loads(path.read_text().splitlines()[0])["event"] == "routing_decision"


def test_tuner_reads_an_in_memory_log():
    log = TelemetryLog()
    session = RoutingSession(router=Router(telemetry=log))
    for _ in range(25):
        session.route_step(TaskSignals(**EASY))
        session.mark_outcome(Outcome.INSUFFICIENT)

    result = ThresholdTuner(min_samples=20).tune(log)
    assert result.adjusted is True
    assert result.samples == 25
    assert Thresholds.load().tuned_from == "TelemetryLog"


def test_tuner_reads_a_plain_iterable_of_events():
    events = []
    for index in range(25):
        events.append({"event": "routing_decision", "decision_id": f"d{index}",
                       "tier": "low", "source": "heuristic", "attempt": 1})
        events.append({"event": "task_outcome", "decision_id": f"d{index}",
                       "outcome": "success"})
    result = ThresholdTuner(min_samples=20).tune(events)
    assert result.samples == 25
    assert result.adjusted is True


def test_an_empty_sink_is_not_mistaken_for_no_sink():
    """TelemetryLog defines __len__, so an empty one is falsy.

    Regression: `telemetry or NullSink()` silently dropped it, and the first
    events of every run vanished.
    """
    log = TelemetryLog()
    assert not log                      # falsy while empty
    router = Router(telemetry=log)
    assert router.telemetry is log
    router.route(TaskSignals(**EASY))
    assert len(log) == 1
