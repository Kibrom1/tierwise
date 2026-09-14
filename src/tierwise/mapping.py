"""Tier -> model mapping, and where each name came from.

Model IDs change, and they differ by provider. Nothing in the routing logic
knows or cares what the strings are: resolve them here, once, and the rest of
the package never needs editing when a model is renamed or a provider swapped.

Precedence, highest first:

1. an explicit ModelMap passed to the Router
2. TIERWISE_MODEL_LOW / _MEDIUM / _HIGH
3. the project config file (see config.py)
4. the built-in defaults

The built-in defaults are a convenience, not a recommendation: they are
Anthropic model IDs, correct when this was written and certain to age, and wrong
by construction for any other provider. Because a wrong model id fails loudly at
call time but looks authoritative in a listing, every resolved name carries its
origin, and ``tierwise models`` prints it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import ConfigError, load_config
from .models import Tier

DEFAULT_MODELS: dict[Tier, str] = {
    Tier.LOW: "claude-haiku-4-5-20251001",
    Tier.MEDIUM: "claude-sonnet-5",
    Tier.HIGH: "claude-opus-5",
}

ENV_KEYS: dict[Tier, str] = {
    Tier.LOW: "TIERWISE_MODEL_LOW",
    Tier.MEDIUM: "TIERWISE_MODEL_MEDIUM",
    Tier.HIGH: "TIERWISE_MODEL_HIGH",
}

EXPLICIT, ENV, CONFIG, DEFAULT = "explicit", "env", "config", "default"


@dataclass
class ModelMap:
    """Resolves a tier to a concrete model identifier."""

    models: dict[Tier, str]
    #: Where each name came from, so a guess never passes for a choice.
    origins: dict[Tier, str] = field(default_factory=dict)
    config_path: Optional[str] = None

    def __post_init__(self) -> None:
        for tier in self.models:
            self.origins.setdefault(tier, EXPLICIT)

    @classmethod
    def resolve(
        cls,
        base: dict[Tier, str] | None = None,
        config_path: str | Path | None = None,
    ) -> "ModelMap":
        """Build the map from config file, environment and defaults."""
        models = dict(base or DEFAULT_MODELS)
        origins = {tier: DEFAULT for tier in models}

        config = load_config(config_path)
        from_config = config.get("models") or {}
        if not isinstance(from_config, dict):
            raise ConfigError("[models] must be a table of tier -> model name")

        for key, value in from_config.items():
            try:
                tier = Tier(str(key).strip().lower())
            except ValueError as exc:
                raise ConfigError(
                    f"unknown tier {key!r} in [models]; expected one of "
                    f"{', '.join(t.value for t in Tier)}"
                ) from exc
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"[models].{key} must be a non-empty string")
            models[tier], origins[tier] = value, CONFIG

        for tier, key in ENV_KEYS.items():
            override = os.environ.get(key)
            if override:
                models[tier], origins[tier] = override, ENV

        return cls(models=models, origins=origins, config_path=config.get("__path__"))

    #: Kept for callers written before the config file existed.
    from_env = resolve

    def model_for(self, tier: Tier) -> str:
        try:
            return self.models[tier]
        except KeyError as exc:  # pragma: no cover - guarded by Tier enum
            raise KeyError(f"no model configured for tier {tier}") from exc

    def origin_for(self, tier: Tier) -> str:
        return self.origins.get(tier, EXPLICIT)

    def as_dict(self) -> dict[str, str]:
        return {tier.value: model for tier, model in self.models.items()}

    def describe(self) -> dict[str, Any]:
        """Everything needed to answer 'where did this model name come from?'"""
        return {
            "models": {
                tier.value: {"model": model, "from": self.origin_for(tier)}
                for tier, model in self.models.items()
            },
            "config_file": self.config_path,
            "configured": any(o != DEFAULT for o in self.origins.values()),
        }
