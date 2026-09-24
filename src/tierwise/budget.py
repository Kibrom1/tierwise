"""A spend ceiling for the proxy -- a safety rail, not governance.

Explicitly narrower than LiteLLM's multi-tenant budget/rate limiting (see
delivery-decision.md's scope note): this is one developer's own machine and
one number -- "stop letting TierWise touch routing once I've spent $X this
period." It never blocks a request. A request that would work without
TierWise always still goes through; the ceiling only turns off *enforcement*
(the reroute), falling back to shadow-only (log what would have routed,
change nothing) until the period rolls over. Spend below the ceiling is
untouched -- this only ever removes TierWise's own effect on cost, never
adds a new way for a request to fail.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_PATH = Path.home() / ".tierwise" / "budget.json"
ENV_PATH = "TIERWISE_BUDGET"

#: Calendar precision is not the point here -- a rolling window is. "monthly"
#: is treated as 30 days, not billing-cycle-aware; good enough for a personal
#: safety rail, and it never needs a timezone.
PERIOD_SECONDS = {
    "daily": 24 * 60 * 60,
    "weekly": 7 * 24 * 60 * 60,
    "monthly": 30 * 24 * 60 * 60,
}


def resolve_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get(ENV_PATH)
    return Path(override) if override else DEFAULT_PATH


@dataclass
class BudgetState:
    """Spend tracked against a configured ceiling for one rolling period.

    `limit_usd=None` (the default) means no ceiling at all -- `record_spend`
    and `exceeded` are then no-ops, so a caller that never sets a budget pays
    no cost for carrying this around.
    """

    limit_usd: Optional[float] = None
    period: str = "monthly"
    spent_usd: float = 0.0
    period_started_at: Optional[float] = None

    def validate(self) -> None:
        if self.limit_usd is not None and self.limit_usd <= 0:
            raise ValueError("limit_usd must be > 0")
        if self.period not in PERIOD_SECONDS:
            raise ValueError(f"period must be one of {sorted(PERIOD_SECONDS)}")
        if self.spent_usd < 0:
            raise ValueError("spent_usd must be >= 0")

    def _roll_if_expired(self, now: float) -> None:
        if self.period_started_at is None:
            self.period_started_at = now
            return
        if now - self.period_started_at >= PERIOD_SECONDS[self.period]:
            self.period_started_at = now
            self.spent_usd = 0.0

    def record_spend(self, usd: float, now: Optional[float] = None) -> None:
        """Add a real, observed cost. Negative or zero amounts are ignored."""
        if usd <= 0 or self.limit_usd is None:
            return
        now = time.time() if now is None else now
        self._roll_if_expired(now)
        self.spent_usd += usd

    def exceeded(self, now: Optional[float] = None) -> bool:
        if self.limit_usd is None:
            return False
        now = time.time() if now is None else now
        self._roll_if_expired(now)
        return self.spent_usd >= self.limit_usd

    def remaining(self, now: Optional[float] = None) -> Optional[float]:
        if self.limit_usd is None:
            return None
        now = time.time() if now is None else now
        self._roll_if_expired(now)
        return max(0.0, self.limit_usd - self.spent_usd)

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BudgetState":
        """Load the persisted ceiling and spend, or return an unset one.

        A missing or unreadable file is not an error -- routing must keep
        working on a machine that has never configured a budget.
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

    def save(self, path: str | Path | None = None) -> Path:
        self.validate()
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
