#!/usr/bin/env python3
"""Summarize a routing.jsonl decision log by tier.

Zero new dependencies -- reads the JSONL telemetry that `tierwise serve` (or
any Router(telemetry=JsonlSink(...))) already writes, and reports request
counts and decision sources per tier. This is the personal-account
equivalent of querying a spend table: no Postgres needed for one person's
usage, just this log.

Usage:
    python scripts/spend_summary.py routing.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def summarize(path: str) -> None:
    by_tier: Counter[str] = Counter()
    by_source: defaultdict[str, Counter[str]] = defaultdict(Counter)
    total = 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") != "routing_decision":
                continue
            total += 1
            tier = event.get("tier", "unknown")
            by_tier[tier] += 1
            by_source[tier][event.get("source", "unknown")] += 1

    if not total:
        print(f"no routing_decision events found in {path}")
        return

    print(f"{total} routing decisions in {path}\n")
    for tier in ("low", "medium", "high"):
        count = by_tier.get(tier, 0)
        if not count:
            continue
        pct = 100 * count / total
        sources = ", ".join(f"{src}={n}" for src, n in by_source[tier].items())
        print(f"  {tier:<7} {count:>5}  ({pct:5.1f}%)   sources: {sources}")

    unknown = total - sum(by_tier.get(t, 0) for t in ("low", "medium", "high"))
    if unknown:
        print(f"  {'other':<7} {unknown:>5}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: spend_summary.py <routing.jsonl>", file=sys.stderr)
        sys.exit(1)
    summarize(sys.argv[1])
