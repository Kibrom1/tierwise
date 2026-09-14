# Using TierWise with Claude

A guide for someone already calling Claude to do engineering work — through the
Anthropic SDK, a coding agent, or CI — who wants each step to run on the
cheapest model that can actually do it.

## Install

Not on PyPI. Two real options:

```bash
pip install -e /path/to/tierwise                              # local / vendored
pip install "tierwise @ git+https://github.com/Kibrom1/tierwise.git"
```

Zero runtime dependencies, so it drops into an existing project without
dragging anything in. The `[anthropic]` extra is only needed for the live
classifier fallback; `[dev]` pulls pytest.

## Your first decision

Three lines. This is the complete minimum:

```python
from tierwise import Router, TaskSignals, Tier

router = Router()                                  # once, at startup
decision = router.route(TaskSignals(category="bugfix", file_count=4, lines_changed=150))

client.messages.create(model=decision.model, ...)  # your existing call, unchanged
```

What comes back for a few shapes:

```
minimal     -> claude-haiku-4-5   | low    | heuristic
with hint   -> claude-opus-4-5    | hint
with floor  -> claude-sonnet-4-5  | floor
```

`decision.model` is the only field you strictly need. `.tier`, `.confidence`,
`.source` and `.rationale` are there for logging, and for arguing with the
decision later.

Two shortcuts that skip signal estimation entirely:

```python
TaskSignals(description="migrate payments schema", tier_hint=Tier.HIGH)  # you already know
TaskSignals(category="typo", lines_changed=2, min_tier=Tier.MEDIUM)      # floor, not a pin
```

Non-Python callers shell out — this is the whole CI integration:

```bash
MODEL=$(tierwise route "add a filter param" --category feature --files 3 --lines 90 --model-only)
# claude-sonnet-4-5
```

### Adopt it in stages

Those three lines are already a useful integration. Everything else is opt-in:

| Stage | You add | You get |
| --- | --- | --- |
| 1 | `Router.route()` | per-task routing, today |
| 2 | `RoutingSession.route_step()` | per-step routing inside an agent loop |
| 3 | `mark_outcome()` | escalation when a step comes back wrong |
| 4 | `tierwise tune` | thresholds that move with observed outcomes |

Do not skip to stage 4. The tuner is only as good as your definition of
failure — without a signal you trust, it trains on noise.

## The one thing to understand first

**TierWise decides. You execute.** It never calls a model. `route()` hands back
a decision whose `.model` is a string; your code makes the call.

```
your orchestrator                TierWise
─────────────────                ────────
  describe the step   ──────▶    route_step(signals) ──▶ decision.model
  call Claude with it
  check the result
  report what happened ─────▶    mark_outcome(...)   ──▶ escalate? re-route
                                                        log it for tuning
```

That means three things are yours to supply. If any is missing, TierWise has
nothing to work with:

| You supply | TierWise gives back |
| --- | --- |
| `TaskSignals` — how big/ambiguous the step is | a tier and a model id |
| The actual model call | — |
| An outcome label — did it work? | an escalation, and evidence for tuning |

## See the whole loop

```bash
python examples/claude_agent_loop.py --tune
```

No API key needed — it runs offline. Real output:

```
Routing one task, step by step:

  step 0: Read the failing test and report what it asser low    tests pass
  step 1: Trace the bug through the session and storage  high   tests pass
  step 2: Rework the retry policy so the fix holds under medium tests fail (step needs high)
          escalated ->                                   high   tests pass
  step 3: Update the changelog entry                     low    tests pass

  tiers used:      ['low', 'high', 'medium', 'high', 'low']
  escalations:     1
  de-escalations:  2

  illustrative spend:  $0.215
  every step at high:  $0.376

Outer loop:
  tightened (failure rate above target -- routing higher)
  samples=4 failure_rate=0.25
  cuts 0.30/0.70 -> 0.27/0.67
```

Four things happened there, and they are the whole product:

1. **Step 1 got `high`, step 3 got `low`.** One task, different tiers per step.
2. **Step 2 was routed `medium`, failed, and escalated to `high`** — without
   restarting the task.
3. **Step 3 dropped straight back to `low`.** A hard step does not tax the rest.
4. **The failure moved the thresholds.** The next run starts stricter.

`--live` runs the same loop against the real API.

## Wiring it into your own loop

```python
from tierwise import JsonlSink, Outcome, Router, RoutingSession, TaskSignals

session = RoutingSession(router=Router(telemetry=JsonlSink("routing.jsonl")))

for step in plan:
    decision = session.route_step(TaskSignals(
        description=step.text,
        category="refactor",
        file_count=len(step.files),
        lines_changed=step.estimated_lines,
        dependency_depth=step.depth,
        requires_context=step.touches_existing_logic,
        ambiguity=0.4,
    ))

    result = call_claude(model=decision.model, prompt=step.text)   # yours

    retry = session.mark_outcome(
        Outcome.SUCCESS if tests_pass() else Outcome.INSUFFICIENT,
        cost_usd=result.cost,
    )
    while retry is not None:                  # the step earned a higher tier
        result = call_claude(model=retry.model, prompt=step.text)
        retry = session.mark_outcome(
            Outcome.SUCCESS if tests_pass() else Outcome.INSUFFICIENT,
            cost_usd=result.cost, decision=retry,
        )
```

One `Router` serves any number of concurrent sessions — it holds no per-task
state. One `RoutingSession` per task.

## Filling in the signals

This is where most of the accuracy lives, and where most integrations are lazy.

Two of the six come free from a diff:

```python
from claude_agent_loop import signals_from_git_diff
signals = signals_from_git_diff("HEAD~1", category="refactor")
# -> 11 files, 586 lines, requires_context=True  -> routed: high
```

The other four are judgement your orchestrator has to make:

- **`requires_context`** (+1.5) — must existing logic be understood before
  writing anything? New file: no. Editing a function whose callers matter: yes.
- **`ambiguity`** 0.0–1.0 (×3.0, the heaviest signal) — how underspecified is
  the request? A ticket with acceptance criteria is 0.1; "make the checkout
  flow less confusing" is 0.9.
- **`dependency_depth`** — how far the change reaches past the files it edits.
- **`is_greenfield`** (−0.5) — no existing behaviour to preserve.

**If you already know the answer, say so.** `tier_hint=Tier.HIGH` is trusted
outright and skips inference entirely — it is the cheapest and most accurate
signal there is. `min_tier=Tier.MEDIUM` sets a floor without pinning the tier.

```python
TaskSignals(description="migrate the payments schema", tier_hint=Tier.HIGH)
```

Check a decision by hand before trusting it:

```bash
tierwise explain --category refactor --files 4 --lines 150 --depth 2 --needs-context
tierwise route "..." --category refactor --files 4 --lines 150 --json
```

## Defining "it worked"

`mark_outcome` is only as good as your definition of failure. Use a signal you
already have — a test run, a lint pass, a rejected diff, a human sending it
back.

| Outcome | Meaning | Effect |
| --- | --- | --- |
| `SUCCESS` | it worked | nothing; counts as evidence the tier was enough |
| `INSUFFICIENT` | wrong, shallow, incomplete | escalate one tier; counts as failure |
| `REJECTED` | a human turned it down | same as insufficient |
| `ERROR` | the call itself failed | **nothing** — infrastructure, not tier |

Getting `ERROR` right matters more than it looks. A rate limit or a timeout is
not evidence the model was too small, and counting it as one walks your
thresholds up and your bill with them.

If you never call `mark_outcome`, escalation never fires and the tuner sits at
"insufficient samples" forever. The inner loop still works; the outer one does
not exist.

## Tuning: cadence and expectations

```bash
tierwise tune routing.jsonl --dry-run    # see the proposed change
tierwise tune routing.jsonl              # apply and persist
tierwise thresholds                      # what is in force now
```

Run it **on a schedule against accumulated logs** — nightly, or per sprint — not
after every task. It moves at most 0.03 per run and needs 20 outcomes *per
boundary* before that boundary moves at all: a bad labelling day should not be
able to relocate your routing.

Re-running is safe. A watermark in the thresholds file records which outcomes
have already been acted on, so a second run over an unchanged log reports `no
new outcomes since the last tuning` and changes nothing. Evidence that did not
move anything — because it was in-band, or short of samples — is kept and
counted again next time.

Each cut answers to its own evidence: `low_medium` to failures of work routed
`low`, `medium_high` to work routed `medium`, and the fallback cut to confident
heuristic calls that went wrong below the top tier. Failures at `high` move
nothing, because there is no higher tier to route to. So a report like

```
low_medium    tier=low      n=12  fail=0.00  0.30 -> 0.33   relaxed
medium_high   tier=medium   n=6   fail=1.00  0.70 -> 0.67   tightened
```

is two independent conclusions, not one rate applied twice.

Tuned values persist to `~/.tierwise/thresholds.json` (`TIERWISE_THRESHOLDS`)
and load automatically in the next `Router`. Nothing else to wire.

### Two switches worth knowing

**Exploration.** By default the loop can only learn from failures, and failures
only tell you when a tier was too *small*. Nothing in ordinary outcome data ever
says a task could have run cheaper — a step routed `medium` that succeeded looks
the same whether or not `low` would have done. Turning on exploration routes a
small fraction of steps one tier below the recommendation and records what
happens:

```python
Router(config=RouterConfig(exploration_rate=0.05))   # 5% of eligible steps
```

A downgrade that fails escalates straight back up, so the cost of being wrong is
one extra call. It never explores against a `tier_hint` or below a `min_tier`
floor. Start at 0.05 and read `expl=n/successes` in the tuning report.

**Cost-derived targets.** `--rework-cost USD` tells the tuner what one failed
step costs you beyond the model call — rework, review, the delay. Each boundary
then steers to the break-even failure rate implied by its own tiers' observed
costs rather than a flat 0.10:

```bash
tierwise tune routing.jsonl --rework-cost 0.50
```

Cheap rework means tolerate more retries; expensive rework means tighten. It is
the one number the log cannot infer, so without it the fixed target stands.
Watch `cost_per_success` across runs — tier is a proxy, that is the real score.

What to expect: **it finds the expensive side faster than the cheap side**
(unless exploration is on).
Under-provisioning shows up as a failed step; over-provisioning never shows up
at all, because the top tier does not fail work a cheap tier could have done.
So the loop tightens quickly on real failures and gives budget back only after
a sustained run without them. That asymmetry is deliberate — the alternative is
a loop that cheerfully routes everything to the cheapest model because nothing
ever told it otherwise.

To roll back, delete the thresholds file; defaults resume.

## Before production

- **Check the model IDs.** `tierwise models` prints what it will call.
  Defaults are `claude-haiku-4-5` / `claude-sonnet-4-5` / `claude-opus-4-5` —
  verify against current IDs and override with `TIERWISE_MODEL_LOW` /
  `_MEDIUM` / `_HIGH`.
- **Use `JsonlSink`, not `TelemetryLog`,** for anything long-lived. The outer
  loop learns from history, and an in-memory log is not history.
- **Decide whether you want the LLM fallback live.** Default is a stub that
  resolves ambiguous cases upward and says so. `TIERWISE_CLASSIFIER=anthropic`
  makes it real; it only fires on genuine boundary cases.
- **Start with `--dry-run` tuning** for the first few cycles and read the
  proposed changes before letting them apply.
