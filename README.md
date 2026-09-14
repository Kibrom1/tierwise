# TierWise

Route each step of a task to the cheapest model tier that can actually do it —
and let the thresholds learn from what happens next.

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
tierwise thresholds                                          # thresholds in force
tierwise tune routing.jsonl [--dry-run]                      # close the outer loop
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

## The two loops

Routing is not one decision per task. It is a decision per *step*, and a slower
decision about how to decide.

### Inner loop — per-step routing

An agent working a task takes many steps, and they are not equally hard. A
router that picks one tier per task re-creates the problem it exists to solve,
one level down: every step pays for the hardest step in the task.

`RoutingSession` routes each step independently. The tier climbs for one gnarly
step and drops straight back for the next — nothing carries the previous tier
forward.

```python
from tierwise import JsonlSink, Outcome, Router, RoutingSession, TaskSignals

session = RoutingSession(router=Router(telemetry=JsonlSink("loop.jsonl")))

for step in agent_steps:
    decision = session.route_step(TaskSignals(...))
    result = run_step(step, model=decision.model)
    session.mark_outcome(Outcome.SUCCESS if result.ok else Outcome.INSUFFICIENT,
                         cost_usd=result.cost)

print(session.summary()["tier_history"])   # ['low', 'high', 'low', 'low']
```

`mark_outcome` returns a re-route when the step earned an escalation, and `None`
when it did not — so the caller retries on a higher tier without restarting the
task.

`Router` holds no per-task state, so one instance serves any number of
concurrent sessions.

### Outer loop — thresholds learn from outcomes

`ThresholdTuner` reads the persisted decision log, joins each decision to the
outcome reported for it, and moves the cuts those decisions came from.

```bash
tierwise tune loop.jsonl              # apply and persist
tierwise tune loop.jsonl --dry-run    # report the proposed change only
tierwise thresholds                   # what is in force right now
```

```
tuner: tightened (failure rate above target -- routing higher)
  samples: 34  failure_rate: 0.8824
  before: {'low_medium': 0.30, 'medium_high': 0.70, 'llm_fallback': 0.50}
  after:  {'low_medium': 0.27, 'medium_high': 0.67, 'llm_fallback': 0.53}
```

Two properties make this a loop rather than a report:

- **Outcomes persist.** `mark_outcome` appends a separate `task_outcome` event
  keyed to the decision id, rather than mutating an event in memory. A tuner
  that learns only from the task currently in flight never accumulates enough
  evidence to be right.
- **Thresholds persist.** Tuned values are written to
  `~/.tierwise/thresholds.json` (override with `TIERWISE_THRESHOLDS`) and loaded
  by the next `Router`. Learning that dies at process exit is not learning.

**Failure is treated asymmetrically, on purpose.** Under-provisioning shows up
in outcome data; over-provisioning does not — the top tier never fails a task
the cheap tier could have done. So the loop tightens on observed failures and
relaxes only on their sustained absence, which is the sole evidence that the
cheap side has room.

Because tuning auto-applies, the guardrails are load-bearing: a minimum sample
count before acting, one bounded step per run (0.03), a hard floor and ceiling,
an enforced gap between the tier cuts, and a recorded `threshold_tuning` event
for every change. Outcomes from engineer hints, `min_tier` floors, and
escalation retries are excluded from the failure rate — none of those were the
classifier's call to get wrong. `Outcome.ERROR` is excluded too.

## How a single decision is made

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
classifier.

Confidence is **continuous**, not bucketed. That matters for the outer loop: if
confidence could only take a handful of discrete values, moving a threshold by a
small step would either change nothing or change everything, and a tuner that
can only overshoot is worse than none.

These are starting values, not constants — the live ones come from a
`Thresholds` instance that the tuner rewrites.

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

The live path splits prompt/parsing from transport. `LLMClassifier` owns the
first; a `call_fn` — a plain `str -> str` callable — owns the second. Testing it
takes a three-line fake instead of a mocked SDK, and a new provider is a new
factory rather than a new classifier:

```python
from tierwise import LLMClassifier, Router, make_anthropic_call_fn

# in a test
router = Router(classifier=LLMClassifier(
    call_fn=lambda prompt: '{"tier": "high", "confidence": 0.9}'
))

# in production
router = Router(classifier=LLMClassifier(
    call_fn=make_anthropic_call_fn(model="claude-haiku-4-5")
))
```

Any failure — transport, missing SDK, unparseable response — degrades to the
fallback classifier rather than propagating. `AnthropicClassifier` builds its
`call_fn` on first use, so constructing one without the SDK installed is not an
error either.

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
| `TIERWISE_THRESHOLDS` | `~/.tierwise/thresholds.json` |

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

`NullSink` (default, writes nothing), `JsonlSink`, `StderrSink`, and
`TelemetryLog` (in memory) ship in the box. The tuner accepts any of them:

```python
log = TelemetryLog()
session = RoutingSession(router=Router(telemetry=log))
...
ThresholdTuner().tune(log)          # a log object
ThresholdTuner().tune("loop.jsonl") # a path
ThresholdTuner().tune(events)       # any iterable of event dicts
```

`TelemetryLog` is for tests and single-process runs. It is not a substitute for
`JsonlSink` in anything long-lived — the outer loop learns from history, and
history a process forgets on exit is not history.

## Layout

```
src/tierwise/
  models.py          Tier (ordered, with clamped +/-), TaskSignals,
                     Classification, RoutingDecision
  heuristics.py      complexity scoring, tier boundaries, confidence
  llm_classifier.py  Classifier protocol, StubClassifier, LLMClassifier,
                     make_anthropic_call_fn
  mapping.py         tier -> model, env-overridable
  escalation.py      outcomes and the bump-one-tier policy
  telemetry.py       decision events and sinks
  router.py          orchestration and precedence (stateless)
  session.py         RoutingSession — the inner loop
  thresholds.py      the tunable cuts, and where they persist
  tuner.py           ThresholdTuner — the outer loop
  cli.py             route / explain / models / thresholds / tune
tests/               108 tests, no network
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Roadmap

- Signal extraction from a real diff or working tree, so callers stop supplying
  counts by hand — the biggest remaining gap, since the router currently assumes
  someone already knows the file and line counts
- Per-tier cost accounting rolled up from `cost_usd`, so the loop can optimize
  spend directly instead of using tier as a proxy for it
- A tuner that fits the cuts from the score distribution rather than nudging
  them a fixed step at a time
