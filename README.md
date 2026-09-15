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

**Site:** [`docs/`](docs/) is the landing page — one self-contained `index.html`,
no build step, no dependencies. Deployed on Vercel; `vercel.json` points the
project's output directory at `docs/`, so a push to `main` redeploys it.

**Not writing the calls yourself?** `tierwise serve` puts the router in the
request path, so any client that accepts a base URL is routed without a line of
its code changing:

```bash
tierwise serve                                   # reports only; forwards unchanged
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
```

Shadow mode is the default: it decides what it *would* have routed, writes that
to the decision log, and forwards your request untouched. `--enforce` makes it
real once the log convinces you.

**New here?** [USAGE.md](USAGE.md) is the step-by-step guide for wiring this
into a Claude-based agent or CI, and `examples/claude_agent_loop.py` runs the
whole loop offline in one command.

## Install

```bash
pip install tierwise          # once released; see Releasing below
pip install -e .              # core, zero runtime dependencies
pip install -e ".[anthropic]" # + live LLM classifier fallback
pip install -e ".[dev]"       # + pytest
```

Python 3.10+.

## Quickstart

```console
$ tierwise route "fix a typo in the README" --category typo --lines 2
tier:       low
model:      claude-haiku-4-5-20251001
source:     heuristic
confidence: 1.00
why:        complexity score 0.00 -> low (drivers: category_prior=-1.30)

$ tierwise route "add a filter param to the reports endpoint" \
    --category feature --files 3 --lines 90 --depth 1 --needs-context
tier:       medium
model:      claude-sonnet-5
source:     heuristic
confidence: 1.00
why:        complexity score 0.47 -> medium (drivers: category_prior=+1.56, requires_context=+1.50, file_count=+1.00)

$ tierwise route "redesign billing persistence" --category architecture \
    --files 30 --lines 2000 --depth 6 --needs-context --ambiguity 0.9
tier:       high
model:      claude-opus-5
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
model:      claude-opus-5
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

### Letting it run the loop

`TaskRunner` does the plumbing — route, call, check, escalate, retry at the new
tier, record the outcome, stop at the cap — while the two judgements stay
yours:

```python
from tierwise import TaskRunner, make_anthropic_executor

runner = TaskRunner(
    executor=make_anthropic_executor(),        # you own the call
    verify=lambda response: tests_pass(),      # you own the standard
    cost_fn=lambda response: price(response),
)

for step in agent_steps:
    result = runner.run(TaskSignals(...))
    print(result.ok, result.tier, result.tiers, result.total_cost_usd)
```

It still never makes the model call itself; `executor` does, and swapping in
streaming, tool use or another provider is a two-line function.

**The loop is exactly as good as `verify`.** A lazy check — "the response was
non-empty" — gives you an automatic loop that escalates on noise and then
teaches the tuner from it. Wire it to something you would actually trust: your
tests, your linter, your reviewer. A thrown exception is recorded as
`ERROR` and does *not* escalate, because a bigger model cannot fix a
connection.

The manual form below is the same loop written out, for when you want the
pieces separately.

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
tierwise replay loop.jsonl --low-medium 0.20 --medium-high 0.50
```

`replay` answers the question that makes auto-applied tuning safe to trust:
what would these cuts have done to work you have already routed?

```
cuts 0.20 / 0.50   4 decisions replayed, 2 would change tier

  low        2 ->    2     0
  medium     2 ->    0   -2
  high       0 ->    2   +2

  of 1 failed steps, 1 would have been routed higher
  estimated spend change: +0.1500 (at the per-tier costs in this log)
```

The trade in one view. Note what it does *not* claim: those steps were never
run at the new tier, so this says where they would have gone, not that they
would have succeeded there. Decisions the classifier made are excluded rather
than guessed at — its verdict is not in the log.

```
thresholds adjusted from observed outcomes
  low_medium    tier=low                   n=12  fail=0.00  0.30 -> 0.33   relaxed
  medium_high   tier=medium                n=6   fail=1.00  0.70 -> 0.67   tightened
  llm_fallback  source=heuristic, tier<high n=18  fail=0.33  0.50 -> 0.53   tightened
```

**Each cut moves on the failures of the tier it governs.** Above, work routed
`low` never failed and work routed `medium` always did, so the two cuts move in
opposite directions from their own evidence. Tuning both on one aggregate rate
would push medium-tier work to the top tier on the strength of failures that
happened somewhere else entirely. Failures at `high` move nothing: there is no
higher tier, so they are not a routing problem.

**Evidence is spent when it is acted on.** A watermark (`tuned_through`) rides
along in the thresholds file, and only outcomes newer than it count. Re-running
against an unchanged log adjusts nothing and says so, which is what makes the
tuner safe on a cron — without it, every run would step again on the same
history and a single bad week would walk routing to the floor.

Two further properties make this a loop rather than a report:

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
cheap side has room — unless you turn on exploration.

### Exploration — the half of the evidence outcomes cannot give you

Outcomes are only ever observed at the tier actually used. A step routed
`medium` that succeeds looks identical whether or not `low` would have done, so
no amount of ordinary logging reveals over-provisioning. Exploration buys that
evidence directly: a small fraction of eligible decisions are routed one tier
*below* the recommendation, and what happens is recorded.

```python
Router(config=RouterConfig(exploration_rate=0.05))   # off by default
```

A downgrade that succeeds is direct evidence the recommendation was too
expensive; one that fails escalates straight back up, which is the safety net
that makes exploring affordable. Explorations are logged as their own source
with the tier they came down from, and they never touch the ordinary failure
rate — a deliberate downgrade failing says nothing about the cut.

It never explores against an instruction: not below a `min_tier` floor, not
against a `tier_hint`, and never below `low`. Those are people saying what the
task needs, and spending their task on our curiosity is not a trade the loop
gets to make.

Safety keeps precedence: real failures at a tier outrank any evidence that the
cheaper side has room.

### What rate should it steer to?

`target_failure_rate` defaults to 0.10 — a hand-picked number. Tell it what a
failure actually costs you and each boundary derives its own target instead:

```bash
tierwise tune routing.jsonl --rework-cost 0.50
```

Trying cheap first always costs the cheap run, plus — when it fails — the
expensive run anyway and whatever the failure itself cost. That breaks even
against going straight to expensive at

```
p* = (c_expensive - c_cheap) / (c_expensive + rework)
```

with the tier costs read from `cost_usd` in the log. Rework is the one term the
log cannot supply, so without it nothing is guessed and the fixed target
applies. `cost_per_success` is reported on every run: tier is only a proxy, and
that is the number the loop is really trying to move.

Because tuning auto-applies, the guardrails are load-bearing: a minimum sample
count **per boundary**, one bounded step per run (0.03), a hard floor and
ceiling, an enforced gap between the tier cuts, a watermark so no outcome is
counted twice, an optional `window` for when routing quality is not stationary,
and a recorded `threshold_tuning` event for every change. Outcomes from engineer hints, `min_tier` floors, and
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
   boundary, where the heuristic genuinely cannot tell — and for tasks that
   arrive with **no signals at all**. The scorer never reads `description`, so a
   description-only task has nothing to score; it gets confidence 0 and goes to
   the classifier, which does read it. (Offline, the stub resolves such a task
   one tier *up* from low rather than guessing cheap.) Signals built from a diff
   are marked `metadata["measured"]`, so a genuinely empty diff still counts as
   evidence of a small change.

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
    call_fn=make_anthropic_call_fn(model="claude-haiku-4-5-20251001")
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
export TIERWISE_CLASSIFIER_MODEL=claude-haiku-4-5-20251001   # optional
```

`AnthropicClassifier` degrades to the stub on any failure — missing SDK, missing
key, unparseable response — so a classifier outage never takes routing down.
Anything with a `classify(signals) -> Classification` method can be passed as
`Router(classifier=...)`.

## Configuration

Model names belong in the project, not in three environment variables set in
three places. Put them in `tierwise.toml` next to your code and commit it —
it is found by walking up from the working directory, so the CLI in CI, the
tuning cron and every developer resolve the same names:

```toml
[models]
low = "claude-haiku-4-5-20251001"
medium = "claude-sonnet-5"
high = "claude-opus-5"
```

`tierwise.json` with the same shape works too, and needs no TOML parser (which
Python 3.10 lacks). Any provider: nothing here is parsed or validated, the
strings come back verbatim as `decision.model`.

Precedence: an explicit `ModelMap` > environment > config file > built-in
defaults. `tierwise models` shows which applied:

```console
$ tierwise models
low     gpt-5-mini                   [config]
medium  llama-4-70b                  [config]
high    claude-opus-5              [env]

config file: /srv/app/tierwise.json
```

**The built-in defaults are a convenience, not a recommendation** — Anthropic
IDs, correct when written and certain to age, and wrong by construction for any
other provider. They are labelled `default` precisely so a built-in never reads
as a choice you made.

| Variable | Default |
| --- | --- |
| `TIERWISE_CONFIG` | nearest `tierwise.toml` / `tierwise.json` |
| `TIERWISE_MODEL_LOW` / `_MEDIUM` / `_HIGH` | see above |
| `TIERWISE_CLASSIFIER` | `stub` (`anthropic` for live) |
| `TIERWISE_CLASSIFIER_MODEL` | `claude-haiku-4-5-20251001` |
| `TIERWISE_THRESHOLDS` | `~/.tierwise/thresholds.json` |

A config file that exists but cannot be parsed raises `ConfigError` rather than
falling back — silently ignoring it would route against models nobody chose.

## Telemetry

Every decision emits one event, so routing quality can be tuned against real
outcomes later. Sinks are deliberately dumb; aggregation belongs downstream.

```python
from tierwise import JsonlSink, Router
router = Router(telemetry=JsonlSink("routing.jsonl"))
```

```json
{"tier": "low", "model": "claude-haiku-4-5-20251001", "confidence": 1.0, "source": "heuristic",
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
  config.py          project config discovery
  mapping.py         tier -> model, with provenance, env-overridable
  escalation.py      outcomes and the bump-one-tier policy
  telemetry.py       decision events and sinks
  router.py          orchestration and precedence (stateless)
  runner.py          TaskRunner — runs the loop around your executor
  session.py         RoutingSession — the inner loop
  thresholds.py      the tunable cuts, and where they persist
  tuner.py           ThresholdTuner — the outer loop
  diff.py            git diff -> TaskSignals
  replay.py          re-cut a log at candidate thresholds
  proxy.py           the routing proxy: re-route without touching client code
  cli.py             route / explain / models / thresholds / tune / replay / serve
docs/                the landing page (single self-contained index.html)
examples/            runnable agent-loop walkthrough
tests/               208 tests, no network
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on 3.10 through 3.13 for every push and pull request
(`.github/workflows/tests.yml`). It needs no network and no API key.

## Releasing

`.github/workflows/release.yml` builds, runs the suite against the built wheel,
and publishes on a version tag:

```bash
git tag v0.8.0 && git push origin v0.8.0
```

It uses PyPI trusted publishing, so no token is stored anywhere. Configure the
publisher once at <https://pypi.org/manage/account/publishing/> — this
repository, workflow `release.yml`, environment `pypi`.

## Roadmap

- Signal extraction from a real diff or working tree, so callers stop supplying
  counts by hand — the biggest remaining gap, since the router currently assumes
  someone already knows the file and line counts
- Per-tier cost accounting rolled up from `cost_usd`, so the loop can optimize
  spend directly instead of using tier as a proxy for it
- A tuner that fits the cuts from the score distribution rather than nudging
  them a fixed step at a time
