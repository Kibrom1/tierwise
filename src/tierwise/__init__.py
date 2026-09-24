"""TierWise -- route coding tasks to the cheapest model tier that can do them.

Two loops:

* the inner loop -- ``RoutingSession`` routes every step of a multi-step task
  independently, so a tier can climb for one hard step and drop back for the
  next;
* the outer loop -- ``ThresholdTuner`` reads the persisted decision log, joins
  it to reported outcomes, and moves the thresholds those decisions came from.
"""

from .escalation import EscalationPolicy, Outcome
from .heuristics import classify as heuristic_classify, score_task
from .llm_classifier import (
    AnthropicClassifier,
    CallFn,
    Classifier,
    LLMClassifier,
    StubClassifier,
    default_classifier,
    make_anthropic_call_fn,
)
from .config import ConfigError, find_config, load_config
from .diff import (
    DiffStat,
    parse_name_status,
    parse_numstat,
    run_git_diff,
    signals_from_diff,
    signals_from_numstat,
    signals_from_stat,
)
from .mapping import DEFAULT_MODELS, FallbackMap, ModelMap
from .models import Classification, RoutingDecision, Source, TaskSignals, Tier
from .budget import BudgetState
from .pricing import (
    PRICES,
    ModelPrice,
    SwitchContext,
    SwitchVerdict,
    actual_cost,
    breakeven_output_tokens,
    evaluate_switch,
    price_for,
    step_cost,
)
from .proxy import (
    ENFORCE,
    SHADOW,
    ProxyPlan,
    ProxyRouter,
    conversation_key,
    serve,
    signals_from_payload,
)
from .replay import ReplayResult, ReplayRow, replay
from .router import Router, RouterConfig
from .runner import (
    Attempt,
    CostFn,
    Executor,
    StepResult,
    TaskRunner,
    Verifier,
    as_outcome,
    make_anthropic_executor,
)
from .session import RoutingSession
from .telemetry import (
    JsonlSink,
    NullSink,
    StderrSink,
    TelemetryLog,
    TelemetrySink,
    read_events,
)
from .thresholds import DEFAULT_THRESHOLDS, QUALITY_FLOOR_PRESETS, Thresholds, quality_floor_names
from .tuner import BoundaryAdjustment, ThresholdTuner, TuningResult

__version__ = "0.8.0"

__all__ = [
    "__version__",
    "actual_cost",
    "AnthropicClassifier",
    "as_outcome",
    "Attempt",
    "BoundaryAdjustment",
    "breakeven_output_tokens",
    "BudgetState",
    "CallFn",
    "Classification",
    "Classifier",
    "ConfigError",
    "conversation_key",
    "CostFn",
    "default_classifier",
    "DEFAULT_MODELS",
    "FallbackMap",
    "DEFAULT_THRESHOLDS",
    "QUALITY_FLOOR_PRESETS",
    "DiffStat",
    "ENFORCE",
    "EscalationPolicy",
    "evaluate_switch",
    "Executor",
    "find_config",
    "heuristic_classify",
    "JsonlSink",
    "LLMClassifier",
    "load_config",
    "make_anthropic_call_fn",
    "make_anthropic_executor",
    "ModelMap",
    "ModelPrice",
    "NullSink",
    "Outcome",
    "parse_name_status",
    "parse_numstat",
    "price_for",
    "quality_floor_names",
    "PRICES",
    "ProxyPlan",
    "ProxyRouter",
    "read_events",
    "replay",
    "ReplayResult",
    "ReplayRow",
    "Router",
    "RouterConfig",
    "RoutingDecision",
    "RoutingSession",
    "run_git_diff",
    "score_task",
    "serve",
    "SHADOW",
    "signals_from_diff",
    "signals_from_numstat",
    "signals_from_payload",
    "signals_from_stat",
    "Source",
    "StderrSink",
    "step_cost",
    "StepResult",
    "StubClassifier",
    "SwitchContext",
    "SwitchVerdict",
    "TaskRunner",
    "TaskSignals",
    "TelemetryLog",
    "TelemetrySink",
    "Thresholds",
    "ThresholdTuner",
    "Tier",
    "TuningResult",
    "Verifier",
]
