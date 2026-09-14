# TierWise

Route coding tasks to the cheapest model tier that can actually do them.

Over-provisioning is the default failure mode of LLM-assisted engineering: a
one-line typo fix and a cross-cutting billing rewrite both get sent to the most
expensive model available. TierWise scores a task's real complexity and routes
it to a `low` / `medium` / `high` tier, with a way back up when the cheap tier
turns out to be wrong.

Design premise: **task category is a weak complexity signal.** "Write a unit
test" can be fifteen lines or a six-hour slog through mocked dependencies. What
actually predicts complexity is context — file count, lines changed, dependency
depth, whether existing logic has to be understood, and how underspecified the
request is. Category is kept only as a small tie-breaking prior.

## Install

```bash
pip install -e .              # core, zero runtime dependencies
pip install -e ".[anthropic]" # + live LLM classifier fallback
pip install -e ".[dev]"       # + pytest
```

Python 3.10+.

## Quickstart

```console
$ tierwise route "fix a typo in the README" --category typo --lines 2
tier:       low
model:      claude-haiku-4-5
source:     heuristic
confidence: 1.00
why:        complexity score 0.00 -> low (drivers: category_prior=-1.30)

$ tierwise route "add a filter param to the reports endpoint" \
    --category feature --files 3 --lines 90 --depth 1 --needs-context
tier:       medium
model:      claude-sonnet-4-5
source:     heuristic
confidence: 1.00
why:        complexity score 0.47 -> medium (drivers: category_prior=+1.56, requires_context=+1.50, file_count=+1.00)

$ tierwise route "redesign billing persistence" --category architecture \
    --files 30 --lines 2000 --depth 6 --needs-context --ambiguity 0.9
tier:       high
model:      claude-opus-4-5
source:     heuristic
confidence: 1.00
why:        complexity score 1.00 -> high (drivers: file_count=+3.00, lines_changed=+3.00, ambiguity=+2.70)
```

A task that lands near a tier boundary is *ambiguous*, and only those tasks pay
for a classifier call:

```console
$ tierwise route "refactor the auth middleware" --category refactor \
    --files 4 --lines 150 --depth 2 --needs-context --ambiguity 0.3
tier:       high
model:      claude-opus-4-5
source:     llm
confidence: 0.50
why:        complexity score 0.65 -> medium (...); ambiguous -> stub classifier:
            ambiguous at score 0.65, resolved medium -> high (no live classifier configured)
```

Other commands:

```bash
tierwise explain --category refactor --files 4 --lines 150   # score breakdown
tierwise models                                              # current tier -> model map
tierwise route ... --json                                    # full decision as JSON
tierwise route ... --model-only                              # just the model id, for scripts
tierwise route ... --telemetry routing.jsonl                 # append the decision to a log
```

## Library use

```python
from tierwise import Outcome, Router, TaskSignals, Tier

router = Router()

decision = router.route(TaskSignals(
    description="migrate the reporting job to the new scheduler",
    category="migration",
    file_count=6,
    lines_changed=310,
    dependency_depth=3,
    requires_context=True,
    ambiguity=0.4,
))

print(decision.tier, decision.model, decision.source, decision.rationale)

# ... run the task against decision.model, then feed back what happened.
retry = router.report_outcome(decision, Outcome.INSUFFICIENT)
if retry:
    print("escalated to", retry.model)
```

## How a decision is made

Precedence, highest first:

1. **Engineer tier hint** (`tier_hint` / `--hint`). Engineers usually know when a
   task is hairy, and that signal is cheaper and more accurate than anything
   inferred. Set `RouterConfig(trust_hints=False)` to ignore hints.
2. **Heuristic scorer**, when its confidence clears the threshold (default 0.5).
   No network call, sub-millisecond.
3. **LLM classifier**, for the ambiguous middle only — scores parked near a tier
   boundary, where the heuristic genuinely cannot tell.

A `min_tier` floor is applied last, so a task can be pinned above whatever the
classifiers concluded.

### Signals

| Signal | Effect |
| --- | --- |
| `file_count` | 1 / 2–3 / 4–10 / >10 → 0, 1, 2, 3 |
| `lines_changed` | <20 / <100 / <400 / ≥400 → 0, 1, 2, 3 |
| `dependency_depth` | 0 / 1–2 / >2 → 0, 1, 2 |
| `requires_context` | +1.5 — existing logic must be understood first |
| `ambiguity` (0.0–1.0) | ×3.0 — the heaviest single signal |
| `is_greenfield` | −0.5 — no existing behaviour to preserve |
| `category` | small prior, −0.10 (typo) to +0.22 (debug_production) |

The raw sum is normalized to 0.0–1.0 and cut at `0.30` (low/medium) and `0.70`
(medium/high). Confidence is the distance from the nearest boundary: a score
sitting on a cut is confidence 0, which is exactly what routes it to the
classifier. All constants live at the top of `heuristics.py`.

### Escalation

Routing cheap is only safe if there is a way back up. `report_outcome` takes
what actually happened and returns a re-route when one is warranted:

- `INSUFFICIENT` / `REJECTED` → bump one tier
- `SUCCESS` → nothing
- `ERROR` → nothing; a failed API call is an infrastructure problem, not a tier
  problem

Capped at `EscalationPolicy(max_attempts=2)` so a hopeless task cannot walk
itself to the top tier repeatedly.

## The LLM classifier fallback

The default is `StubClassifier`: **no network, no API key**. Ambiguous tasks sit
on a boundary by construction, so rather than inventing a verdict the stub
resolves upward to the safer tier and reports confidence 0.5 — visible in
telemetry instead of silently masquerading as a real classification.

To use live classification:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
export TIERWISE_CLASSIFIER=anthropic
export TIERWISE_CLASSIFIER_MODEL=claude-haiku-4-5   # optional
```

`AnthropicClassifier` degrades to the stub on any failure — missing SDK, missing
key, unparseable response — so a classifier outage never takes routing down.
Anything with a `classify(signals) -> Classification` method can be passed as
`Router(classifier=...)`.

## Configuration

Model IDs change. Nothing in the routing logic needs editing when they do:

| Variable | Default |
| --- | --- |
| `TIERWISE_MODEL_LOW` | `claude-haiku-4-5` |
| `TIERWISE_MODEL_MEDIUM` | `claude-sonnet-4-5` |
| `TIERWISE_MODEL_HIGH` | `claude-opus-4-5` |
| `TIERWISE_CLASSIFIER` | `stub` (`anthropic` for live) |
| `TIERWISE_CLASSIFIER_MODEL` | `claude-haiku-4-5` |

**Check the defaults against current model IDs before relying on them** — or
pass an explicit `ModelMap(...)`.

## Telemetry

Every decision emits one event, so routing quality can be tuned against real
outcomes later. Sinks are deliberately dumb; aggregation belongs downstream.

```python
from tierwise import JsonlSink, Router
router = Router(telemetry=JsonlSink("routing.jsonl"))
```

```json
{"tier": "low", "model": "claude-haiku-4-5", "confidence": 1.0, "source": "heuristic",
 "rationale": "complexity score 0.00 -> low (...)", "attempt": 1, "escalated_from": null,
 "signals": {...}, "event": "routing_decision", "timestamp": 1773450000.0, "elapsed_ms": 0.07}
```

`NullSink` (default, writes nothing), `JsonlSink`, and `StderrSink` ship in the
box.

## Layout

```
src/tierwise/
  models.py          Tier, TaskSignals, Classification, RoutingDecision
  heuristics.py      complexity scoring, tier boundaries, confidence
  llm_classifier.py  Classifier protocol, StubClassifier, AnthropicClassifier
  mapping.py         tier -> model, env-overridable
  escalation.py      outcomes and the bump-one-tier policy
  telemetry.py       decision events and sinks
  router.py          orchestration and precedence
  cli.py             route / explain / models
tests/               45 tests, no network
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Roadmap

- Signal extraction from a real diff or working tree, so callers stop supplying
  counts by hand
- Tuning the boundaries against logged telemetry rather than by judgement
- Per-tier cost accounting in the decision event
