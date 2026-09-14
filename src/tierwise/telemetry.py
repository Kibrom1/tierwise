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
from typing import Any, Protocol, TYPE_CHECKING

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
