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

    #: Watermark: outcomes at or before this timestamp have already been acted
    #: on. Without it the tuner re-reads the whole log every run and moves the
    #: cuts again on evidence it already consumed -- a nightly job would walk
    #: routing to the floor on one bad week and keep walking.
    tuned_through: Optional[float] = None

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
        on a machine that has never been tuned. If nothing has been tuned yet
        and `tierwise.toml`/`tierwise.json` sets `quality_floor`, that preset is
        the starting point instead of the hardcoded "balanced" cuts -- but only
        ever the *starting* point: a `thresholds.json` that already exists (this
        machine has been tuned) wins over the config file every time, because
        the tuner's evidence is worth more than a preset guess.
        """
        target = resolve_path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls._default_from_config()

        known = {f for f in cls().to_dict()}
        loaded = cls(**{k: v for k, v in raw.items() if k in known})
        try:
            loaded.validate()
        except ValueError:
            return cls()
        return loaded

    @classmethod
    def _default_from_config(cls) -> "Thresholds":
        """`cls()` unless `tierwise.toml`/`tierwise.json` names a quality_floor."""
        from .config import load_config  # local import: avoid a load-time cycle

        floor = load_config().get("quality_floor")
        if not floor:
            return cls()
        try:
            return cls.from_quality_floor(floor)
        except ValueError:
            # An unknown preset in config shouldn't break routing -- `models`
            # commands surface bad config loudly elsewhere; here, fall back.
            return cls()

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

    @classmethod
    def from_quality_floor(cls, name: str) -> "Thresholds":
        """Build thresholds from a named preset instead of raw cut values.

        Raises ValueError on an unknown name -- a typo'd preset should fail
        loudly, not silently fall back to "balanced".
        """
        try:
            low_medium, medium_high = QUALITY_FLOOR_PRESETS[name]
        except KeyError:
            known = ", ".join(sorted(QUALITY_FLOOR_PRESETS))
            raise ValueError(f"unknown quality floor {name!r} (known: {known})") from None
        thresholds = cls(low_medium=low_medium, medium_high=medium_high)
        thresholds.validate()
        return thresholds


#: Named presets for the two complexity cuts, for callers who want a knob
#: simpler than raw floats. These set `low_medium`/`medium_high` only --
#: `boundary_margin` and `llm_fallback` stay at their defaults, and a tuned
#: `thresholds.json` still wins once one exists (see `Thresholds.load`).
#: "balanced" is exactly `Thresholds()`'s starting cuts.
QUALITY_FLOOR_PRESETS: dict[str, tuple[float, float]] = {
    "strict": (0.20, 0.55),
    "balanced": (0.30, 0.70),
    "lenient": (0.40, 0.80),
}


def quality_floor_names() -> tuple[str, ...]:
    return tuple(QUALITY_FLOOR_PRESETS)


DEFAULT_THRESHOLDS = Thresholds()
