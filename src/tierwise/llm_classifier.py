"""LLM classifier fallback.

Only ambiguous tasks reach this layer -- the ones the heuristic scorer landed
too close to a tier boundary to call. The default implementation is a stub that
makes no network calls, so the router runs with no API key. Swap in
AnthropicClassifier (or anything matching the Classifier protocol) for live
classification.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Protocol

from .models import Classification, Source, TaskSignals, Tier

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
        tier = heuristic.tier if heuristic.tier is Tier.HIGH else heuristic.tier.bumped()
        return Classification(
            tier=tier,
            confidence=self.confidence,
            rationale=(
                f"stub classifier: ambiguous at score {score.value:.2f}, "
                f"resolved {heuristic.tier.value} -> {tier.value} (no live classifier configured)"
            ),
            source=Source.LLM,
        )


class AnthropicClassifier:
    """Live classifier backed by the Anthropic Messages API.

    Requires the ``anthropic`` extra and ANTHROPIC_API_KEY. Any failure --
    missing SDK, missing key, bad response -- degrades to the stub rather than
    taking the router down.
    """

    name = "anthropic"

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 256,
        client: Any = None,
        fallback: Optional[Classifier] = None,
    ) -> None:
        self.model = model or os.environ.get("TIERWISE_CLASSIFIER_MODEL", "claude-haiku-4-5")
        self.max_tokens = max_tokens
        self._client = client
        self.fallback = fallback or StubClassifier()

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so the base install stays dependency-free

            self._client = anthropic.Anthropic()
        return self._client

    def classify(self, signals: TaskSignals) -> Classification:
        try:
            client = self._get_client()
            prompt = CLASSIFIER_PROMPT.format(
                signals=json.dumps(signals.to_dict(), ensure_ascii=False, indent=2)
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                getattr(block, "text", "") for block in getattr(response, "content", [])
            )
            parsed = _parse_json_object(text)
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


def _parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
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
