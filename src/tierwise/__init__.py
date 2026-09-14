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
from .llm_classifier import AnthropicClassifier, Classifier, StubClassifier, default_classifier
from .mapping import DEFAULT_MODELS, ModelMap
from .models import Classification, RoutingDecision, Source, TaskSignals, Tier
from .router import Router, RouterConfig
from .session import RoutingSession
from .telemetry import JsonlSink, NullSink, StderrSink, TelemetrySink, read_events
from .thresholds import DEFAULT_THRESHOLDS, Thresholds
from .tuner import ThresholdTuner, TuningResult

__version__ = "0.2.0"

__all__ = [
    "AnthropicClassifier",
    "Classification",
    "Classifier",
    "DEFAULT_MODELS",
    "DEFAULT_THRESHOLDS",
    "EscalationPolicy",
    "JsonlSink",
    "ModelMap",
    "NullSink",
    "Outcome",
    "Router",
    "RouterConfig",
    "RoutingDecision",
    "RoutingSession",
    "Source",
    "StderrSink",
    "StubClassifier",
    "TaskSignals",
    "TelemetrySink",
    "ThresholdTuner",
    "Thresholds",
    "Tier",
    "TuningResult",
    "__version__",
    "default_classifier",
    "heuristic_classify",
    "read_events",
    "score_task",
]
