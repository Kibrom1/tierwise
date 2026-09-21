# How a routing decision is made

This is the complete flow behind a single `Router.route()` call (equivalently,
one `tierwise route` / one request through `tierwise serve`). README's
"How a single decision is made" section is the short version; this is the
whole thing, including the two stages that come after the tier is chosen —
switching cost and exploration — which change what's actually returned.

Source of truth: `src/tierwise/router.py` (orchestration),
`src/tierwise/heuristics.py` (scoring), `src/tierwise/thresholds.py` (the
tunable cuts). If this doc and the code ever disagree, the code is right —
open an issue or fix this file.

## Surfaces that trigger this flow

Every one of these ends up calling the same `Router.route()` — none of them
have their own scoring logic:

| Surface | What it's for |
| --- | --- |
| `tierwise route` (CLI) | one-off, scriptable, CI |
| `Router` / `RoutingSession` (library) | wired into your own agent loop |
| `tierwise serve` (proxy) | routes real API calls with zero per-call ceremony — the only surface where a decision here actually changes which model answers a request |
| a Claude Skill (`tierwise-route`) | ask for a routing opinion in a Claude conversation in plain language, no CLI flags |

That last one is worth being precise about: the Skill estimates `TaskSignals`
from a plain-language description (a judgment call, not a measurement — see
`signals_from_diff()` for the measured version) and then runs the real CLI, so
the decision itself is made by this same flow. But it only ever *prints* a
recommendation — a Skill has no way to change which model is generating the
chat response it's running inside. Only the proxy changes what a request
actually gets served by.

## The flow, in order

```
tier_hint given?  ──yes──▶  use it, confidence 1.0, source=hint
       │no
       ▼
heuristic score the task ──▶ confidence >= llm_fallback threshold?
       │                              │yes
       │no                            ▼
       ▼                      use heuristic tier, source=heuristic
LLM classifier (stub or live)
       │
       ▼
use classifier tier, source=llm / stub
       │
       ▼
apply min_tier floor (if set)         ──▶ source=floor when it actually raises the tier
       │
       ▼
apply switching-cost hold (if a SwitchContext was passed and this is a
downgrade from an incumbent tier with a warm cache)  ──▶ source=cache_hold
       │                                                  when the hold fires
       ▼
exploration roll (if exploration_rate > 0, eligible, and the dice say so)
       │                                                  ──▶ source=exploration,
       ▼                                                      confidence forced to 0
final RoutingDecision: tier, model, confidence, source, rationale
```

Only one of `hint` / `heuristic` / `llm` / `stub` decides the *tier* at the
top of the flow. Everything below the floor step can still override that
tier afterward (raise it for a floor, hold it for cache economics, or lower
it for exploration) — the `source` on the final decision tells you which
stage actually won.

### 1. Engineer tier hint

`TaskSignals(tier_hint=Tier.HIGH)` (or `--hint high` on the CLI) is trusted
outright: confidence 1.0, no scoring, no classifier call. This is the
cheapest and most accurate signal there is — you already know. Disable with
`RouterConfig(trust_hints=False)` if you want hints ignored (e.g. auditing
what the heuristic alone would have said).

### 2. Heuristic scorer

No hint → the task is scored by `heuristics.score_task()`. Every task gets
scored here, including ones with no real signals — the "no evidence" case
below is a *confidence* outcome of this step, not a step it skips.

**The score.** A weighted sum of contributions, normalized by dividing by
`_MAX_RAW = 13.0` and clamped to `[0.0, 1.0]`:

| Signal | Contribution |
| --- | --- |
| `file_count` | 0.0 / 1.0 / 2.0 / 3.0 for 1, 2–3, 4–10, >10 files |
| `lines_changed` | 0.0 / 1.0 / 2.0 / 3.0 for <20, <100, <400, ≥400 lines |
| `dependency_depth` | 0.0 / 1.0 / 2.0 for 0, 1–2, >2 |
| `requires_context` | +1.5 flat if true |
| `ambiguity` (0.0–1.0) | ×3.0 — the single heaviest signal |
| `is_greenfield` | −0.5 flat if true |
| `category` | a prior from `CATEGORY_PRIORS`, from −0.10 (`typo`) to +0.22
  (`debug_production`), scaled by `_MAX_RAW` |

Category priors are deliberately small — category is a weak, tie-breaking
signal, not a driver. An unrecognized category contributes nothing (and the
CLI warns about it).

**Score → tier.** Two tunable cuts, `low_medium` and `medium_high` (defaults
0.30 / 0.70): below `low_medium` is `low`, below `medium_high` is `medium`,
otherwise `high`.

**Score → confidence.** Confidence is the normalized *distance from the
nearest cut* — continuous, not bucketed:

```
distance   = min(|score - low_medium|, |score - medium_high|)
confidence = clamp(distance / (boundary_margin * 2), 0.0, 1.0)
```

`boundary_margin` (default 0.07) is how far from a cut confidence reaches
~0.5. A score sitting exactly on a cut has confidence 0 — the genuinely
ambiguous case. Continuity matters for tuning: with bucketed confidence, a
small threshold move would either change nothing or flip everything at once.

**Gate: is there actually evidence?** `has_evidence()` checks whether the
caller supplied anything beyond a bare description — a non-default
`file_count`, `lines_changed`, `dependency_depth`, `requires_context`,
`ambiguity`, `is_greenfield`, a recognized `category`, or signals pulled from
a real diff (`metadata["measured"]`, which counts even when the diff is
genuinely empty — a measured zero is still a measurement). If none of that is
present, confidence is forced to **0.0** regardless of what the score
happened to be — a bare description defaults every signal to its "simplest"
value and would otherwise score 0.0, the single most confident-looking score
possible, for the least-informed decision possible. Forcing confidence to 0
here is what sends a plain-language task to the classifier instead of
silently routing it to `low`.

**Gate: does confidence clear the threshold?** If heuristic confidence is
`>= thresholds.llm_fallback` (default 0.50), the heuristic's tier is used,
`source=heuristic`. Otherwise, step 3.

### 3. LLM classifier

Only reached for the ambiguous middle — low heuristic confidence, which
includes both boundary-adjacent scores and every no-evidence task. The
`Classifier` protocol is one method, `classify(signals) -> Classification`:

- **`StubClassifier`** (default, no network, no API key) resolves upward
  rather than guessing cheap, and says so in the rationale.
- **`LLMClassifier`** does the real thing: reads `description`, calls out via
  an injected `call_fn` (`make_anthropic_call_fn()` for the built-in
  Anthropic transport).

Selected by `TIERWISE_CLASSIFIER` (`stub` | `anthropic`).

### 4. `min_tier` floor

Applied after the tier is chosen, whatever chose it. If `signals.min_tier` is
set and the current tier is below it, the tier is raised to the floor and
`source` becomes `floor`, confidence 1.0. This is a floor, not a pin — it can
only push the tier up.

### 5. Switching-cost hold

Only relevant when the caller passes a `SwitchContext` (an in-progress
session with a warm cache on some `incumbent` tier) and the tier this
decision would otherwise land on is a **downgrade** from that incumbent.
Prompt caches are tied to a specific model, so dropping a tier mid-task can
forfeit a warm cache and eat a cache-write cost that outweighs the savings
from the cheaper tier. `evaluate_switch()` weighs that trade-off; if it says
the switch isn't worth it, the decision is held at the incumbent tier,
`source=cache_hold`, and the rationale records why. This only ever holds a
tier *up* (keeps the more expensive one) — it never blocks a tier that needs
to go up.

### 6. Exploration

Off by default (`RouterConfig.exploration_rate`, e.g. `0.05` for 5%).
Outcomes are only ever observed at the tier actually used, so failures reveal
under-provisioning but successes never reveal over-provisioning — a `medium`
task that succeeds looks identical whether or not `low` would have worked
too. Exploration deliberately routes a fraction of eligible decisions one
tier below the recommendation to measure that directly. Eligible means: tier
isn't already `low`, the tier wasn't set by a `floor` or `cache_hold`, and
there's no `tier_hint` or `min_tier` in play — exploration never overrides an
explicit instruction. A downgrade that then fails escalates straight back up
(see below), which is what makes exploring affordable: the cost of being
wrong is one extra call, and its confidence is reported as 0.

## After the decision: escalation

`route()` produces one decision; what happens next is the caller's job, fed
back through `mark_outcome()` / `report_outcome()`:

| Outcome | Effect |
| --- | --- |
| `SUCCESS` | nothing; counts as evidence the tier was enough |
| `INSUFFICIENT` | bump one tier and re-route, up to `max_attempts` (default 2) |
| `REJECTED` | same as `INSUFFICIENT` |
| `ERROR` | nothing — infrastructure, not tier; excluded from tuning too |

This is the inner loop (`RoutingSession`) — per-step, immediate, and separate
from the thresholds themselves moving over time.

## The thresholds are not constants

`low_medium`, `medium_high`, `boundary_margin`, and `llm_fallback` all live in
a `Thresholds` object (`src/tierwise/thresholds.py`), not as fixed numbers.
They:

- **Persist** to `~/.tierwise/thresholds.json` (override with
  `TIERWISE_THRESHOLDS`), so a value the outer loop learns survives past one
  process.
- **Load automatically** into a new `Router` — a fresh session inherits
  whatever the last `tierwise tune` run decided.
- **Move only via `tierwise tune`**, reading a decision log and adjusting the
  cuts from observed outcomes — see `USAGE.md`'s "Tuning: cadence and
  expectations" for the full mechanics (per-boundary attribution, the
  watermark that stops repeat runs from double-counting evidence, the
  0.03-per-run step cap, exploration, and cost-derived targets).
- Can be **pinned** by passing an explicit `Thresholds()` to `Router`, which
  skips loading the tuned file entirely.

So "how the decision is made" has a static half (the flow above, and the
scoring formula) and a moving half (exactly where the cuts and the fallback
threshold currently sit) — `tierwise thresholds` always shows what's live.

## Tier → model

Once a tier is settled, `ModelMap` resolves it to an actual model string —
`tierwise.toml` > `TIERWISE_MODEL_LOW`/`_MEDIUM`/`_HIGH` env vars > built-in
defaults (currently `claude-haiku-4-5-20251001` / `claude-sonnet-5` /
`claude-opus-5`). `tierwise models` shows the resolved mapping and marks
anything unset as a built-in rather than a deliberate choice. The router
never inspects or validates this string — whatever `ModelMap` returns is
handed back verbatim as `decision.model`.
