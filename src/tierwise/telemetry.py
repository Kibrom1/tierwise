"""Routing telemetry.

Every decision is emitted as one event so routing quality can be tuned against
real outcomes later. Sinks are intentionally dumb; aggregation belongs
downstream.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .models import RoutingDecision


class TelemetrySink(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...


class NullSink:
    """Drops everything. The default, so nothing writes to disk uninvited."""

    def emit(self, event: dict[str, Any]) -> None:  # noqa: D102
        return None


class JsonlSink:
    """Appends one JSON object per line to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


class StderrSink:
    """Useful when debugging routing decisions interactively."""

    def emit(self, event: dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False), file=sys.stderr)


def build_event(decision: "RoutingDecision", elapsed_ms: float) -> dict[str, Any]:
    event = decision.to_dict()
    event["event"] = "routing_decision"
    event["timestamp"] = time.time()
    event["elapsed_ms"] = round(elapsed_ms, 3)
    return event


def build_outcome_event(
    decision: "RoutingDecision",
    outcome: str,
    cost_usd: Optional[float] = None,
) -> dict[str, Any]:
    """An outcome is a separate append-only event keyed to the decision.

    Mutating the original event in place would be tidier to read, but it only
    works while the process is alive -- and the whole point of recording
    outcomes is that a later run can learn from them.
    """
    return {
        "event": "task_outcome",
        "timestamp": time.time(),
        "decision_id": decision.decision_id,
        "session_id": decision.session_id,
        "step_index": decision.step_index,
        "tier": decision.tier.value,
        "source": decision.source.value,
        "attempt": decision.attempt,
        "outcome": outcome,
        "cost_usd": cost_usd,
    }


def build_tuning_event(result: dict[str, Any]) -> dict[str, Any]:
    """Record a threshold change, so every tuned value is traceable."""
    return {"event": "threshold_tuning", "timestamp": time.time(), **result}


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL log, skipping lines that are not valid JSON objects."""
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event
