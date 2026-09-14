"""Tunable thresholds, and where they persist.

These are the knobs the outer loop moves. They live in a dataclass rather than
as module constants precisely so the ThresholdTuner can rewrite them, and they
persist to disk so what one session learns is still there for the next one --
a feedback loop whose output dies with the process is not a loop.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_PATH = Path.home() / ".tierwise" / "thresholds.json"
ENV_PATH = "TIERWISE_THRESHOLDS"


def resolve_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get(ENV_PATH)
    return Path(override) if override else DEFAULT_PATH


@dataclass
class Thresholds:
    """Every cut the router makes, in one tunable object."""

    # Complexity-score cuts between tiers.
    low_medium: float = 0.30
    medium_high: float = 0.70
    # Score distance from a cut within which confidence reads ~0.5.
    boundary_margin: float = 0.07
    # Heuristic confidence below which the LLM classifier is consulted.
    llm_fallback: float = 0.50

    # Provenance, so a tuned file is auditable.
    version: int = 0
    updated_at: Optional[float] = None
    tuned_from: Optional[str] = None

    MIN_GAP = 0.15

    def validate(self) -> None:
        if not 0.0 < self.low_medium < 1.0:
            raise ValueError("low_medium must be strictly between 0 and 1")
        if not 0.0 < self.medium_high < 1.0:
            raise ValueError("medium_high must be strictly between 0 and 1")
        if self.medium_high - self.low_medium < self.MIN_GAP:
            raise ValueError(
                f"tier cuts must stay at least {self.MIN_GAP} apart "
                f"(got {self.low_medium} / {self.medium_high})"
            )
        if self.boundary_margin <= 0:
            raise ValueError("boundary_margin must be > 0")
        if not 0.0 <= self.llm_fallback <= 1.0:
            raise ValueError("llm_fallback must be between 0 and 1")

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Thresholds":
        """Load tuned thresholds, falling back to defaults.

        A missing or unreadable file is not an error: routing must keep working
        on a machine that has never been tuned.
        """
        target = resolve_path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()

        known = {f for f in cls().to_dict()}
        loaded = cls(**{k: v for k, v in raw.items() if k in known})
        try:
            loaded.validate()
        except ValueError:
            return cls()
        return loaded

    def save(self, path: str | Path | None = None, tuned_from: str | None = None) -> Path:
        self.validate()
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.version += 1
        self.updated_at = time.time()
        if tuned_from is not None:
            self.tuned_from = tuned_from
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "Thresholds":
        return Thresholds(**self.to_dict())


DEFAULT_THRESHOLDS = Thresholds()
