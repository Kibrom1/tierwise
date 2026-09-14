"""Command line interface for TierWise."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .heuristics import classify as heuristic_classify
from .mapping import ModelMap
from .models import TaskSignals, Tier
from .router import Router, RouterConfig
from .telemetry import JsonlSink, NullSink, StderrSink


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

    subparsers.add_parser("models", help="print the current tier -> model mapping")
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
    decision = _make_router(args).route(_signals_from_args(args))
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


def _cmd_models(_args: argparse.Namespace) -> int:
    print(json.dumps(ModelMap.from_env().as_dict(), indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {"route": _cmd_route, "explain": _cmd_explain, "models": _cmd_models}
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
