"""Tier -> model mapping.

Model IDs change often. Every default here is overridable by environment
variable or by passing an explicit ModelMap, so the routing logic never has to
be edited when a model name changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .models import Tier

DEFAULT_MODELS: dict[Tier, str] = {
    Tier.LOW: "claude-haiku-4-5",
    Tier.MEDIUM: "claude-sonnet-4-5",
    Tier.HIGH: "claude-opus-4-5",
}

ENV_KEYS: dict[Tier, str] = {
    Tier.LOW: "TIERWISE_MODEL_LOW",
    Tier.MEDIUM: "TIERWISE_MODEL_MEDIUM",
    Tier.HIGH: "TIERWISE_MODEL_HIGH",
}


@dataclass
class ModelMap:
    """Resolves a tier to a concrete model identifier."""

    models: dict[Tier, str]

    @classmethod
    def from_env(cls, base: dict[Tier, str] | None = None) -> "ModelMap":
        models = dict(base or DEFAULT_MODELS)
        for tier, key in ENV_KEYS.items():
            override = os.environ.get(key)
            if override:
                models[tier] = override
        return cls(models=models)

    def model_for(self, tier: Tier) -> str:
        try:
            return self.models[tier]
        except KeyError as exc:  # pragma: no cover - guarded by Tier enum
            raise KeyError(f"no model configured for tier {tier}") from exc

    def as_dict(self) -> dict[str, str]:
        return {tier.value: model for tier, model in self.models.items()}
