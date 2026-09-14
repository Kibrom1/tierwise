"""Core data types for TierWise routing."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class Tier(str, enum.Enum):
    """Complexity tiers, ordered low -> high."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return _TIER_ORDER.index(self)

    @classmethod
    def from_rank(cls, rank: int) -> "Tier":
        rank = max(0, min(rank, len(_TIER_ORDER) - 1))
        return _TIER_ORDER[rank]

    def bumped(self, steps: int = 1) -> "Tier":
        return Tier.from_rank(self.rank + steps)


_TIER_ORDER: list[Tier] = [Tier.LOW, Tier.MEDIUM, Tier.HIGH]


class Source(str, enum.Enum):
    """Where a routing decision came from."""

    HINT = "hint"
    HEURISTIC = "heuristic"
    LLM = "llm"
    ESCALATION = "escalation"
    FLOOR = "floor"


@dataclass
class TaskSignals:
    """Everything TierWise knows about a task before routing it.

    Category alone is a weak complexity signal, so the fields that carry the
    most weight are the contextual ones: how much code is in play, how deep the
    dependency chain runs, whether existing logic has to be understood, and how
    ambiguous the request is.
    """

    description: str = ""
    category: Optional[str] = None
    file_count: int = 1
    lines_changed: int = 0
    dependency_depth: int = 0
    requires_context: bool = False
    ambiguity: float = 0.0
    is_greenfield: bool = False
    tier_hint: Optional[Tier] = None
    min_tier: Optional[Tier] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.tier_hint, str):
            self.tier_hint = Tier(self.tier_hint)
        if isinstance(self.min_tier, str):
            self.min_tier = Tier(self.min_tier)
        if self.file_count < 0:
            raise ValueError("file_count must be >= 0")
        if self.lines_changed < 0:
            raise ValueError("lines_changed must be >= 0")
        if self.dependency_depth < 0:
            raise ValueError("dependency_depth must be >= 0")
        if not 0.0 <= self.ambiguity <= 1.0:
            raise ValueError("ambiguity must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tier_hint"] = self.tier_hint.value if self.tier_hint else None
        data["min_tier"] = self.min_tier.value if self.min_tier else None
        return data


@dataclass
class Classification:
    """A tier guess from one classifier, with how sure it is."""

    tier: Tier
    confidence: float
    rationale: str
    source: Source

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class RoutingDecision:
    """The final answer: which model to call, and why."""

    tier: Tier
    model: str
    confidence: float
    source: Source
    rationale: str
    signals: TaskSignals
    attempt: int = 1
    escalated_from: Optional[Tier] = None

    # Loop identity. A decision belongs to a step of a session, and outcomes
    # reported later are joined back to it by decision_id.
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: Optional[str] = None
    step_index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "step_index": self.step_index,
            "tier": self.tier.value,
            "model": self.model,
            "confidence": round(self.confidence, 3),
            "source": self.source.value,
            "rationale": self.rationale,
            "attempt": self.attempt,
            "escalated_from": self.escalated_from.value if self.escalated_from else None,
            "signals": self.signals.to_dict(),
        }
