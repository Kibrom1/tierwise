"""TierWise -- route coding tasks to the cheapest model tier that can do them."""

from .escalation import EscalationPolicy, Outcome
from .heuristics import classify as heuristic_classify, score_task
from .llm_classifier import AnthropicClassifier, Classifier, StubClassifier, default_classifier
from .mapping import DEFAULT_MODELS, ModelMap
from .models import Classification, RoutingDecision, Source, TaskSignals, Tier
from .router import Router, RouterConfig
from .telemetry import JsonlSink, NullSink, StderrSink, TelemetrySink

__version__ = "0.1.0"

__all__ = [
    "AnthropicClassifier",
    "Classification",
    "Classifier",
    "DEFAULT_MODELS",
    "EscalationPolicy",
    "JsonlSink",
    "ModelMap",
    "NullSink",
    "Outcome",
    "Router",
    "RouterConfig",
    "RoutingDecision",
    "Source",
    "StderrSink",
    "StubClassifier",
    "TaskSignals",
    "TelemetrySink",
    "Tier",
    "__version__",
    "default_classifier",
    "heuristic_classify",
    "score_task",
]
