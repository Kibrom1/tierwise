"""Worked example: routing the steps of a Claude coding agent with TierWise.

Runs offline by default -- no API key, no network -- so you can see the whole
loop before wiring anything up:

    python examples/claude_agent_loop.py

With the SDK installed and ANTHROPIC_API_KEY set, the same loop makes real
calls:

    python examples/claude_agent_loop.py --live

What it shows, in order:

0. TaskRunner driving the whole thing: it routes, calls, checks, escalates and
   records on its own. The two judgements stay yours -- `executor` makes the
   call, `verify` decides whether the result is good.
1. One task, five steps of unequal difficulty, each routed on its own.
2. A step that fails at the tier it was routed to, and escalates.
3. What that cost, against the bill for sending every step to the top tier.
4. The outer loop reading the log those steps just wrote.

The one thing to take away: TierWise never makes the model call itself. It runs
the loop around one you supply. Everything below the `Executor` line is yours to
replace.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Protocol

from tierwise import (
    JsonlSink,
    Outcome,
    Router,
    RouterConfig,
    RoutingSession,
    Source,
    TaskRunner,
    TaskSignals,
    ThresholdTuner,
    Thresholds,
    Tier,
)

LOG = "agent-loop.jsonl"

# Illustrative only -- placeholders for the per-step cost your own accounting
# would report. Replace with real figures before drawing any conclusions.
ILLUSTRATIVE_STEP_COST = {Tier.LOW: 0.004, Tier.MEDIUM: 0.019, Tier.HIGH: 0.094}


# --------------------------------------------------------------------------
# The task: what an agent would actually work through.
# --------------------------------------------------------------------------

@dataclass
class Step:
    instruction: str
    signals: TaskSignals
    #: Tier below which this step genuinely cannot be done. The offline stub
    #: uses it to decide whether a step fails; in real life your tests do.
    needs: Tier = Tier.LOW


TASK = [
    Step(
        "Read the failing test and report what it asserts",
        TaskSignals(category="unit_test", file_count=1, lines_changed=8),
        needs=Tier.LOW,
    ),
    Step(
        "Trace the bug through the session and storage layers",
        TaskSignals(category="bugfix", file_count=9, lines_changed=340,
                    dependency_depth=4, requires_context=True, ambiguity=0.6),
        needs=Tier.HIGH,
    ),
    Step(
        "Rework the retry policy so the fix holds under concurrency",
        TaskSignals(category="refactor", file_count=4, lines_changed=160,
                    dependency_depth=2, requires_context=True),
        # Routed at medium on its signals, but genuinely needs high: the step
        # fails, and the loop escalates it rather than restarting the task.
        needs=Tier.HIGH,
    ),
    Step(
        # Large diff, trivial work. Its signals say medium; it only ever needed
        # low. Nothing in ordinary outcome data can reveal that -- a step routed
        # medium that succeeds looks identical whether or not low would have
        # done. Only an exploration finds it.
        "Rename sessionId to session_id across the package",
        TaskSignals(category="rename", file_count=20, lines_changed=600,
                    dependency_depth=3),
        needs=Tier.LOW,
    ),
    Step(
        "Update the changelog entry",
        TaskSignals(category="docstring", file_count=1, lines_changed=6),
        needs=Tier.LOW,
    ),
]


# --------------------------------------------------------------------------
# Executor -- everything below this line is yours. TierWise picks the model;
# running it is your orchestrator's job.
# --------------------------------------------------------------------------

@dataclass
class StepResult:
    ok: bool
    cost_usd: float
    detail: str


class Executor(Protocol):
    def run(self, model: str, tier: Tier, step: Step) -> StepResult: ...


class StubExecutor:
    """Offline stand-in. A step succeeds when it was routed at or above the
    tier it actually needed -- which is exactly what a test suite tells you."""

    def run(self, model: str, tier: Tier, step: Step) -> StepResult:
        ok = tier >= step.needs
        return StepResult(
            ok=ok,
            cost_usd=ILLUSTRATIVE_STEP_COST[tier],
            detail="tests pass" if ok else f"tests fail (step needs {step.needs.value})",
        )


class AnthropicExecutor:
    """Real calls. Substitute your own success check for `ok`."""

    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.Anthropic()

    def run(self, model: str, tier: Tier, step: Step) -> StepResult:
        response = self.client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": step.instruction}],
        )
        text = "".join(getattr(b, "text", "") for b in response.content)
        usage = getattr(response, "usage", None)
        # Whatever your pipeline already knows: did the tests pass, was the diff
        # accepted, did a human send it back? That is the `ok` you want here.
        return StepResult(
            ok=bool(text.strip()),
            cost_usd=ILLUSTRATIVE_STEP_COST[tier],
            detail=f"{getattr(usage, 'output_tokens', '?')} output tokens",
        )


# --------------------------------------------------------------------------
# Signals: where the numbers come from.
# --------------------------------------------------------------------------

def signals_from_git_diff(ref: str = "HEAD~1", category: Optional[str] = None) -> TaskSignals:
    """Build TaskSignals from a real diff -- a starting point, not a rule.

    file_count and lines_changed come straight from `git diff --numstat`.
    requires_context, ambiguity and dependency_depth are judgement calls your
    orchestrator has to make; the defaults here are conservative.
    """
    out = subprocess.run(
        ["git", "diff", "--numstat", ref],
        capture_output=True, text=True, check=True,
    ).stdout

    files, lines = 0, 0
    for row in out.splitlines():
        parts = row.split("\t")
        if len(parts) != 3:
            continue
        files += 1
        added, removed = parts[0], parts[1]
        lines += sum(int(v) for v in (added, removed) if v.isdigit())

    return TaskSignals(
        category=category,
        file_count=max(files, 1),
        lines_changed=lines,
        requires_context=files > 1,
    )


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------

def run_task(executor: Executor, explore: float = 0.0, rng=None,
             quiet: bool = False) -> tuple[float, float]:
    session = RoutingSession(router=Router(
        telemetry=JsonlSink(LOG),
        config=RouterConfig(exploration_rate=explore),
        rng=rng,
    ))

    # The runner needs to know which step a decision belongs to. Escalation
    # preserves the signals object, so identity is a reliable key.
    by_signals = {id(step.signals): step for step in TASK}

    runner = TaskRunner(
        executor=lambda decision: executor.run(
            decision.model, decision.tier, by_signals[id(decision.signals)]
        ),
        verify=lambda result: result.ok,          # your tests go here
        cost_fn=lambda result: result.cost_usd,
        session=session,
    )

    def say(line: str) -> None:
        if not quiet:
            print(line)

    spent = 0.0
    for step, outcome in zip(TASK, runner.run_all([s.signals for s in TASK])):
        spent += outcome.total_cost_usd
        for index, attempt in enumerate(outcome.attempts):
            mark = " (explored)" if attempt.decision.source is Source.EXPLORATION else ""
            label = step.instruction[:46] if index == 0 else "escalated ->"
            say(f"  step {outcome.attempts[0].decision.step_index}: {label:<46} "
                f"{attempt.tier.value:<6} {attempt.result.detail}{mark}")

    summary = session.summary()
    say(f"\n  tiers used:      {summary['tier_history']}")
    say(f"  escalations:     {summary['escalations']}")
    say(f"  de-escalations:  {summary['de_escalations']}")

    all_high = len(TASK) * ILLUSTRATIVE_STEP_COST[Tier.HIGH]
    return spent, all_high


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="make real Anthropic calls (needs the SDK and ANTHROPIC_API_KEY)")
    parser.add_argument("--tune", action="store_true",
                        help="run the outer loop over the log afterwards")
    parser.add_argument("--runs", type=int, default=6,
                        help="repeats before tuning (default: 6) -- one task is not evidence")
    parser.add_argument("--explore", type=float, default=0.0, metavar="RATE",
                        help="fraction of steps routed one tier below the "
                             "recommendation, to measure whether cheaper would do")
    parser.add_argument("--rework-cost", type=float, default=None, metavar="USD",
                        help="what a failed step costs beyond the model call; "
                             "makes the tuning target cost-derived")
    args = parser.parse_args(argv)

    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
            return 1
        executor: Executor = AnthropicExecutor()
    else:
        executor = StubExecutor()

    rng = random.Random(7).random          # seeded, so the example is repeatable
    print("Routing one task, step by step:\n")
    spent, all_high = run_task(executor, explore=args.explore, rng=rng)

    print(f"\n  illustrative spend:  ${spent:.3f}")
    print(f"  every step at high:  ${all_high:.3f}")
    print(f"  (figures are placeholders -- substitute your own per-step costs)")

    if args.tune:
        # One task is four outcomes split across three tiers -- nowhere near
        # enough for any single boundary to have earned a move. Repeat first.
        for _ in range(max(args.runs - 1, 0)):
            run_task(executor, explore=args.explore, rng=rng, quiet=True)

        print(f"\nOuter loop (after {args.runs} runs of the task):")
        result = ThresholdTuner(min_samples=5, rework_cost_usd=args.rework_cost).tune(LOG)
        print(f"  {result.reason}")
        if result.cost_per_success is not None:
            print(f"  cost per successful step: ${result.cost_per_success:.4f}")
        for adj in result.adjustments:
            arrow = f"{adj.before:.2f} -> {adj.after:.2f}"
            rate = "n/a" if adj.failure_rate is None else f"{adj.failure_rate:.2f}"
            explored = f"expl={adj.explore_samples}/{adj.explore_successes}"
            print(f"    {adj.name:<13} n={adj.samples:<3} fail={rate:<5} {explored:<12} "
                  f"target={adj.target:.2f} [{adj.target_source}]  "
                  f"{arrow:<14} {adj.direction}")
            if adj.basis:
                print(f"      on: {adj.basis}")

    print(f"\n  decision log: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
