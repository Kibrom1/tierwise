"""Worked example: routing the steps of a Claude coding agent with TierWise.

Runs offline by default -- no API key, no network -- so you can see the whole
loop before wiring anything up:

    python examples/claude_agent_loop.py

With the SDK installed and ANTHROPIC_API_KEY set, the same loop makes real
calls:

    python examples/claude_agent_loop.py --live

What it shows, in order:

1. One task, four steps of unequal difficulty, each routed on its own.
2. A step that fails at the tier it was routed to, and escalates.
3. What that cost, against the bill for sending every step to the top tier.
4. The outer loop reading the log those steps just wrote.

The one thing to take away: TierWise decides, you execute. It never makes the
model call. Everything below the `Executor` line is yours to replace.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Protocol

from tierwise import (
    JsonlSink,
    Outcome,
    Router,
    RoutingSession,
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

def run_task(executor: Executor) -> tuple[float, float]:
    session = RoutingSession(router=Router(telemetry=JsonlSink(LOG)))
    spent = 0.0

    for step in TASK:
        decision = session.route_step(step.signals)
        result = executor.run(decision.model, decision.tier, step)
        spent += result.cost_usd

        print(f"  step {decision.step_index}: {step.instruction[:46]:<46} "
              f"{decision.tier.value:<6} {result.detail}")

        outcome = Outcome.SUCCESS if result.ok else Outcome.INSUFFICIENT
        retry = session.mark_outcome(outcome, cost_usd=result.cost_usd)

        # A returned decision means the step earned a higher tier. Re-run it.
        while retry is not None:
            result = executor.run(retry.model, retry.tier, step)
            spent += result.cost_usd
            print(f"          {'escalated ->':<46} {retry.tier.value:<6} {result.detail}")
            retry = session.mark_outcome(
                Outcome.SUCCESS if result.ok else Outcome.INSUFFICIENT,
                cost_usd=result.cost_usd,
                decision=retry,
            )

    summary = session.summary()
    print(f"\n  tiers used:      {summary['tier_history']}")
    print(f"  escalations:     {summary['escalations']}")
    print(f"  de-escalations:  {summary['de_escalations']}")

    all_high = len(TASK) * ILLUSTRATIVE_STEP_COST[Tier.HIGH]
    return spent, all_high


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="make real Anthropic calls (needs the SDK and ANTHROPIC_API_KEY)")
    parser.add_argument("--tune", action="store_true",
                        help="run the outer loop over the log afterwards")
    args = parser.parse_args(argv)

    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
            return 1
        executor: Executor = AnthropicExecutor()
    else:
        executor = StubExecutor()

    print("Routing one task, step by step:\n")
    spent, all_high = run_task(executor)

    print(f"\n  illustrative spend:  ${spent:.3f}")
    print(f"  every step at high:  ${all_high:.3f}")
    print(f"  (figures are placeholders -- substitute your own per-step costs)")

    if args.tune:
        print("\nOuter loop:")
        before = Thresholds.load()
        result = ThresholdTuner(min_samples=4).tune(LOG)
        print(f"  {result.direction or result.reason}")
        print(f"  samples={result.samples} failure_rate={result.failure_rate}")
        print(f"  cuts {before.low_medium:.2f}/{before.medium_high:.2f}"
              f" -> {result.after['low_medium']:.2f}/{result.after['medium_high']:.2f}")

    print(f"\n  decision log: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
