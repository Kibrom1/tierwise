# Running TierWise on a personal Anthropic account

This is the setup for trying TierWise on your own account before rolling it
out further — no separate proxy project, no LiteLLM, no Postgres. TierWise
already ships the pieces that a bolt-on router would otherwise duplicate
(`tierwise serve`'s proxy, JSONL telemetry, the tuner); this doc just wires
them together for one person's use.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## 2. Set your model names

`tierwise.toml` at the repo root already pins:

```toml
[models]
low = "claude-haiku-4-5-20251001"
medium = "claude-sonnet-5"
high = "claude-opus-5"
```

Check what's active with `tierwise models` — anything it reports as `default`
means this file isn't being picked up (run from inside the repo, or set
`TIERWISE_CONFIG` to its path).

## 3. Point your client at the routing proxy — shadow mode first

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # your personal key
tierwise serve --telemetry routing.jsonl
```

In another shell (or your app / agent / editor's model config):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
```

Shadow mode is the default: every request is logged with the tier TierWise
*would* have routed to, and forwarded to Anthropic unchanged. Nothing about
your actual model usage changes yet — you're collecting evidence.

Use Claude (CLI, SDK, whatever you normally use) for a while, then check what
it would have done:

```bash
python scripts/spend_summary.py routing.jsonl
```

```
42 routing decisions in routing.jsonl

  low        18  ( 42.9%)   sources: heuristic=18
  medium     19  ( 45.2%)   sources: heuristic=17, llm=2
  high        5  ( 11.9%)   sources: heuristic=5
```

## 4. Turn it on

Once the shadow log looks right:

```bash
tierwise serve --telemetry routing.jsonl --enforce
```

Requests now actually get routed to the tier TierWise picks, not just logged.

## 5. Close the loop (optional, once you have real outcomes)

If you have a way to tell whether a step's output was good enough (tests
passing, a lint pass, you rejecting a diff), report it and let thresholds
learn:

```python
from tierwise import Outcome, Router, RoutingSession, TaskSignals

session = RoutingSession(router=Router())
decision = session.route_step(TaskSignals(category="bugfix", file_count=2, lines_changed=40))
# ... call decision.model, check the result ...
session.mark_outcome(Outcome.SUCCESS)  # or INSUFFICIENT / REJECTED / ERROR
```

Then, on a cadence (nightly / weekly, not after every task):

```bash
tierwise tune routing.jsonl --dry-run   # see what would change first
tierwise tune routing.jsonl             # apply
tierwise thresholds                     # what's in force now
```

## Why no Postgres / LiteLLM here

An earlier pass at this (before finding TierWise already existed) built a
LiteLLM proxy plus a Postgres spend-log container. TierWise's own proxy
covers the routing, and `routing.jsonl` + `scripts/spend_summary.py` covers
spend visibility for a single account. A SQL-backed telemetry sink is worth
adding to `tierwise.telemetry` if this grows to multiple people sharing one
log (see the roadmap in the README) — `NullSink` / `JsonlSink` / `StderrSink`
/ `TelemetryLog` are the sinks that exist today.
