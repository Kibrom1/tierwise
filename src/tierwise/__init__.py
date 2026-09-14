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
from .mapping import DEFAULT_MODELS, ModelMap
from .models import Classification, RoutingDecision, Source, TaskSignals, Tier
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
from .thresholds import DEFAULT_THRESHOLDS, Thresholds
from .tuner import BoundaryAdjustment, ThresholdTuner, TuningResult

__version__ = "0.8.0"

__all__ = [
    "__version__",
    "AnthropicClassifier",
    "as_outcome",
    "Attempt",
    "BoundaryAdjustment",
    "CallFn",
    "Classification",
    "Classifier",
    "ConfigError",
    "CostFn",
    "default_classifier",
    "DEFAULT_MODELS",
    "DEFAULT_THRESHOLDS",
    "DiffStat",
    "EscalationPolicy",
    "Executor",
    "find_config",
    "heuristic_classify",
    "JsonlSink",
    "LLMClassifier",
    "load_config",
    "make_anthropic_call_fn",
    "make_anthropic_executor",
    "ModelMap",
    "NullSink",
    "Outcome",
    "parse_name_status",
    "parse_numstat",
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
    "signals_from_diff",
    "signals_from_numstat",
    "signals_from_stat",
    "Source",
    "StderrSink",
    "StepResult",
    "StubClassifier",
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
