"""LLM classifier fallback.

Only ambiguous tasks reach this layer -- the ones the heuristic scorer landed
too close to a tier boundary to call. The default implementation is a stub that
makes no network calls, so the router runs with no API key.

The live path is split in two: ``LLMClassifier`` owns the prompt and the
parsing, and a ``call_fn`` -- a plain ``str -> str`` callable -- owns the
transport. Injecting the transport means the classifier is testable with a
three-line fake instead of a mocked SDK, and swapping providers is a new
factory rather than a new classifier.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional, Protocol

from .models import Classification, Source, TaskSignals, Tier

CallFn = Callable[[str], str]

CLASSIFIER_PROMPT = """You are a task complexity classifier for a code-assistant router.

Classify the task below into exactly one tier:
- "low": mechanical, self-contained, no reasoning about existing logic required.
- "medium": ordinary feature or bugfix work spanning a few files, some context needed.
- "high": ambiguous scope, cross-cutting changes, deep dependency chains, or
  design decisions with lasting consequences.

Task signals (JSON):
{signals}

Respond with JSON only, no prose:
{{"tier": "low|medium|high", "confidence": 0.0-1.0, "rationale": "one sentence"}}
"""


class Classifier(Protocol):
    def classify(self, signals: TaskSignals) -> Classification: ...


class StubClassifier:
    """Offline fallback classifier.

    Ambiguous tasks are, by construction, sitting on a tier boundary. Rather
    than inventing a verdict, the stub resolves upward to the safer tier and
    reports low confidence, so the choice is visible in telemetry instead of
    silently masquerading as a real classification.
    """

    name = "stub"

    def __init__(self, confidence: float = 0.5) -> None:
        self.confidence = confidence

    def classify(self, signals: TaskSignals) -> Classification:
        from .heuristics import classify as heuristic_classify

        heuristic, score = heuristic_classify(signals)
        tier = heuristic.tier if heuristic.tier is Tier.HIGH else heuristic.tier + 1
        return Classification(
            tier=tier,
            confidence=self.confidence,
            rationale=(
                f"stub classifier: ambiguous at score {score.value:.2f}, "
                f"resolved {heuristic.tier.value} -> {tier.value} (no live classifier configured)"
            ),
            source=Source.LLM,
        )


class LLMClassifier:
    """Prompt, call, parse -- with the call injected.

    Any failure (transport, SDK, unparseable response) degrades to the fallback
    classifier rather than propagating: a classifier outage should make routing
    less clever, never take it down.
    """

    name = "llm"

    def __init__(
        self,
        call_fn: Optional[CallFn] = None,
        fallback: Optional[Classifier] = None,
        name: Optional[str] = None,
    ) -> None:
        self.call_fn = call_fn
        self.fallback = fallback if fallback is not None else StubClassifier()
        if name is not None:
            self.name = name

    def _get_call_fn(self) -> CallFn:
        if self.call_fn is None:
            raise RuntimeError("no call_fn configured")
        return self.call_fn

    def classify(self, signals: TaskSignals) -> Classification:
        try:
            raw = self._get_call_fn()(self.build_prompt(signals))
            parsed = _parse_json_object(raw)
            tier = Tier(str(parsed["tier"]).strip().lower())
            confidence = float(parsed.get("confidence", 0.7))
            rationale = str(parsed.get("rationale", "")).strip() or "llm classification"
            return Classification(
                tier=tier,
                confidence=confidence,
                rationale=f"{self.name}: {rationale}",
                source=Source.LLM,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the route
            result = self.fallback.classify(signals)
            result.rationale = f"{self.name} unavailable ({type(exc).__name__}); {result.rationale}"
            return result

    @staticmethod
    def build_prompt(signals: TaskSignals) -> str:
        return CLASSIFIER_PROMPT.format(
            signals=json.dumps(signals.to_dict(), ensure_ascii=False, indent=2)
        )


class AnthropicClassifier(LLMClassifier):
    """LLMClassifier wired to the Anthropic Messages API.

    The call_fn is built on first use, not in ``__init__``, so constructing one
    without the SDK installed or without a key is not an error -- it degrades on
    the first classification like any other transport failure.
    """

    name = "anthropic"

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 256,
        client: Any = None,
        fallback: Optional[Classifier] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(call_fn=None, fallback=fallback)
        self.model = model or os.environ.get("TIERWISE_CLASSIFIER_MODEL", "claude-haiku-4-5")
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client = client

    def _get_call_fn(self) -> CallFn:
        if self.call_fn is None:
            self.call_fn = make_anthropic_call_fn(
                api_key=self.api_key,
                model=self.model,
                max_tokens=self.max_tokens,
                client=self._client,
            )
        return self.call_fn


def make_anthropic_call_fn(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    client: Any = None,
) -> CallFn:
    """Build a ``str -> str`` transport backed by the Anthropic Messages API.

    The ``anthropic`` import is deferred into this function so the rest of the
    package has no hard dependency on the SDK for anyone who only wants the
    heuristic layer.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    target_model = model or os.environ.get("TIERWISE_CLASSIFIER_MODEL", "claude-haiku-4-5")

    def call_fn(prompt: str) -> str:
        response = client.messages.create(
            model=target_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in getattr(response, "content", []))

    return call_fn


def _parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in classifier response: {text[:200]!r}")
    return json.loads(match.group(0))


def default_classifier() -> Classifier:
    """Pick a classifier from the environment. Stubbed unless asked otherwise."""
    choice = os.environ.get("TIERWISE_CLASSIFIER", "stub").strip().lower()
    if choice in {"anthropic", "live"}:
        return AnthropicClassifier()
    return StubClassifier()
