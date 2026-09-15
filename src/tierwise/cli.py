"""Command line interface for TierWise."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .heuristics import CATEGORY_PRIORS, has_evidence
from .heuristics import classify as heuristic_classify
from .mapping import ModelMap
from .models import TaskSignals, Tier
from .router import Router, RouterConfig
from .telemetry import JsonlSink, NullSink, StderrSink
from .replay import replay
from .thresholds import Thresholds, resolve_path
from .tuner import ThresholdTuner


def _add_signal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("description", nargs="?", default="", help="what the task is")
    parser.add_argument("--category", help="task category, e.g. refactor, bugfix, unit_test")
    parser.add_argument("--files", type=int, default=1, dest="file_count",
                        help="number of files in play (default: 1)")
    parser.add_argument("--lines", type=int, default=0, dest="lines_changed",
                        help="estimated lines changed (default: 0)")
    parser.add_argument("--depth", type=int, default=0, dest="dependency_depth",
                        help="dependency depth the change reaches into (default: 0)")
    parser.add_argument("--needs-context", action="store_true", dest="requires_context",
                        help="existing logic has to be understood first")
    parser.add_argument("--ambiguity", type=float, default=0.0,
                        help="how underspecified the request is, 0.0-1.0 (default: 0.0)")
    parser.add_argument("--greenfield", action="store_true", dest="is_greenfield",
                        help="new code with no existing behaviour to preserve")
    parser.add_argument("--hint", choices=[t.value for t in Tier], dest="tier_hint",
                        help="engineer-supplied tier; trusted over inference")
    parser.add_argument("--min-tier", choices=[t.value for t in Tier], dest="min_tier",
                        help="floor the routed tier at this level")


def _signals_from_args(args: argparse.Namespace) -> TaskSignals:
    return TaskSignals(
        description=args.description,
        category=args.category,
        file_count=args.file_count,
        lines_changed=args.lines_changed,
        dependency_depth=args.dependency_depth,
        requires_context=args.requires_context,
        ambiguity=args.ambiguity,
        is_greenfield=args.is_greenfield,
        tier_hint=Tier(args.tier_hint) if args.tier_hint else None,
        min_tier=Tier(args.min_tier) if args.min_tier else None,
    )


def _warn_thin_signals(signals: TaskSignals) -> None:
    """Say so when the description is doing all the work.

    The heuristic scorer never reads the description, so a plain-language task
    with no flags has nothing to score. It is routed by the classifier instead
    -- which, unless a live one is configured, can only guess.
    """
    category = (signals.category or "").strip().lower()
    if category and category not in CATEGORY_PRIORS:
        known = ", ".join(sorted(CATEGORY_PRIORS))
        print(f"note: unknown category {signals.category!r} is ignored. Known: {known}",
              file=sys.stderr)
    if not has_evidence(signals):
        live = os.environ.get("TIERWISE_CLASSIFIER", "stub").strip().lower() in {"anthropic", "live"}
        how = ("the live classifier will read the description" if live else
               "with no live classifier this is a cautious guess; set "
               "TIERWISE_CLASSIFIER=anthropic to classify from the description")
        print("note: no signals given (--files, --lines, --depth, --needs-context, "
              f"--ambiguity, --category); {how}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tierwise",
        description="Route a coding task to the cheapest model tier that can handle it.",
    )
    parser.add_argument("--version", action="version", version=f"tierwise {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route = subparsers.add_parser("route", help="route a task and print the chosen model")
    _add_signal_args(route)
    route.add_argument("--json", action="store_true", help="emit the full decision as JSON")
    route.add_argument("--model-only", action="store_true", help="print only the model id")
    route.add_argument("--telemetry", metavar="PATH",
                       help="append the decision to a JSONL file ('-' for stderr)")
    route.add_argument("--threshold", type=float, default=None,
                       help="heuristic confidence below which the LLM classifier is used")

    explain = subparsers.add_parser("explain", help="show the heuristic score breakdown")
    _add_signal_args(explain)
    explain.add_argument("--json", action="store_true", help="emit the breakdown as JSON")

    models = subparsers.add_parser(
        "models", help="print the tier -> model mapping and where each name came from"
    )
    models.add_argument("--json", action="store_true", help="emit as JSON")
    subparsers.add_parser("thresholds", help="print the thresholds currently in force")

    tune = subparsers.add_parser(
        "tune", help="close the outer loop: retune thresholds from a decision log"
    )
    tune.add_argument("log", help="JSONL decision log written by --telemetry")
    tune.add_argument("--dry-run", action="store_true",
                      help="report the proposed change without persisting it")
    tune.add_argument("--min-samples", type=int, default=20,
                      help="outcomes required before any adjustment (default: 20)")
    tune.add_argument("--target-failure-rate", type=float, default=0.10,
                      help="failure rate the loop steers toward (default: 0.10)")
    tune.add_argument("--step", type=float, default=0.03,
                      help="maximum threshold movement per run (default: 0.03)")
    tune.add_argument("--rework-cost", type=float, default=None, metavar="USD",
                      help="what one failed step costs beyond the model call; "
                           "set it and each boundary steers to the break-even "
                           "failure rate implied by its tiers' observed costs")
    tune.add_argument("--window", type=int, default=None,
                      help="consider only the most recent N outcomes")

    rep = subparsers.add_parser(
        "replay", help="re-cut a decision log at different thresholds, without applying them"
    )
    rep.add_argument("log", help="JSONL decision log written by --telemetry")
    rep.add_argument("--low-medium", type=float, default=None,
                     help="candidate low/medium cut (default: the one in force)")
    rep.add_argument("--medium-high", type=float, default=None,
                     help="candidate medium/high cut (default: the one in force)")
    rep.add_argument("--json", action="store_true", help="emit the full result as JSON")
    return parser


def _make_router(args: argparse.Namespace) -> Router:
    telemetry = NullSink()
    path = getattr(args, "telemetry", None)
    if path == "-":
        telemetry = StderrSink()
    elif path:
        telemetry = JsonlSink(path)

    config = RouterConfig()
    if getattr(args, "threshold", None) is not None:
        config.confidence_threshold = args.threshold

    return Router(config=config, telemetry=telemetry)


def _cmd_route(args: argparse.Namespace) -> int:
    signals = _signals_from_args(args)
    if not args.model_only:
        _warn_thin_signals(signals)
    decision = _make_router(args).route(signals)
    if args.model_only:
        print(decision.model)
    elif args.json:
        print(json.dumps(decision.to_dict(), indent=2))
    else:
        print(f"tier:       {decision.tier.value}")
        print(f"model:      {decision.model}")
        print(f"source:     {decision.source.value}")
        print(f"confidence: {decision.confidence:.2f}")
        print(f"why:        {decision.rationale}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    signals = _signals_from_args(args)
    _warn_thin_signals(signals)
    classification, score = heuristic_classify(signals)
    if args.json:
        print(json.dumps({
            "score": round(score.value, 3),
            "tier": classification.tier.value,
            "confidence": round(classification.confidence, 3),
            "contributions": {k: round(v, 3) for k, v in score.contributions.items()},
            "rationale": classification.rationale,
        }, indent=2))
        return 0

    print(f"score:      {score.value:.3f}")
    print(f"tier:       {classification.tier.value}")
    print(f"confidence: {classification.confidence:.2f}")
    print("contributions:")
    for name, weight in sorted(score.contributions.items(), key=lambda i: -abs(i[1])):
        print(f"  {name:<18} {weight:+.2f}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    mapping = ModelMap.resolve()
    described = mapping.describe()

    if args.json:
        print(json.dumps(described, indent=2))
        return 0

    for tier in Tier:
        entry = described["models"][tier.value]
        print(f"{tier.value:<7} {entry['model']:<28} [{entry['from']}]")
    print()
    print(f"config file: {described['config_file'] or 'none found'}")
    if not described["configured"]:
        print("No model names configured -- these are built-in defaults, which age. "
              "Set yours in tierwise.toml or TIERWISE_MODEL_LOW/_MEDIUM/_HIGH.")
    return 0


def _cmd_thresholds(_args: argparse.Namespace) -> int:
    thresholds = Thresholds.load()
    payload = thresholds.to_dict()
    payload["path"] = str(resolve_path())
    payload["tuned"] = thresholds.version > 0
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    candidate = Thresholds.load()
    if args.low_medium is not None:
        candidate.low_medium = args.low_medium
    if args.medium_high is not None:
        candidate.medium_high = args.medium_high
    candidate.validate()

    result = replay(args.log, thresholds=candidate)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"cuts {candidate.low_medium:.2f} / {candidate.medium_high:.2f}"
          f"   {result.considered} decisions replayed, {result.changed} would change tier")
    print()
    for tier in Tier:
        before = result.tier_before.get(tier.value, 0)
        after = result.tier_after.get(tier.value, 0)
        move = after - before
        arrow = f"{move:+d}" if move else "  0"
        print(f"  {tier.value:<7} {before:>4} -> {after:>4}   {arrow}")
    print()
    if result.failures_total:
        print(f"  of {result.failures_total} failed steps, "
              f"{result.failures_routed_higher} would have been routed higher")
    if result.successes_total:
        print(f"  of {result.successes_total} successful steps, "
              f"{result.successes_routed_lower} would have been routed lower")
    if result.estimated_cost_delta is not None:
        print(f"  estimated spend change: {result.estimated_cost_delta:+.4f} "
              f"(at the per-tier costs in this log)")
    skipped = (result.skipped_not_from_cuts + result.skipped_classifier
               + result.skipped_no_signals)
    if skipped:
        print(f"\n  {skipped} decisions not replayable: "
              f"{result.skipped_not_from_cuts} not decided by the cuts, "
              f"{result.skipped_classifier} decided by the classifier, "
              f"{result.skipped_no_signals} missing signals")
    print("\n  Rerouted steps were never actually run at the new tier -- this says "
          "where they would\n  have gone, not that they would have succeeded there.")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    tuner = ThresholdTuner(
        step=args.step,
        min_samples=args.min_samples,
        target_failure_rate=args.target_failure_rate,
        rework_cost_usd=args.rework_cost,
        window=args.window,
    )
    result = tuner.tune(args.log, apply=not args.dry_run)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "route": _cmd_route,
        "explain": _cmd_explain,
        "models": _cmd_models,
        "thresholds": _cmd_thresholds,
        "tune": _cmd_tune,
        "replay": _cmd_replay,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
