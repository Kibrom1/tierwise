"""Verify configured model names against what the provider actually serves.

mapping.py resolves tier -> model name from an explicit map, env vars,
tierwise.toml, or the built-in defaults, and says plainly that the defaults
are "a convenience, not a recommendation... correct when this was written and
certain to age." This module does the one narrow thing that catches that
decay before a real request does: ask the provider's own model-listing
endpoint whether each configured name still exists, and say which ones don't.

Deliberately not model *discovery*. A provider's list says what exists, not
which of those belong at which tier -- that judgement (cheap vs capable for
coding tasks) isn't in the API response, and guessing it from the list risks
silently routing to whatever came back. This only checks names already
configured; it never assigns new ones.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from .mapping import ModelMap
from .models import Tier

#: Per-provider defaults for reaching a model-listing endpoint. Anthropic and
#: OpenAI-compatible APIs (what tierwise serve already speaks to, per the
#: multi-client proxy work) both expose one; add a provider here rather than
#: hardcoding a new base URL at the call site.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "path": "/v1/models",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "path": "/v1/models",
        "env_key": "OPENAI_API_KEY",
    },
}


class VerifyError(RuntimeError):
    """The provider could not be asked at all -- no key, network, or bad response.

    Distinct from a model simply not being found: that is a normal, expected
    result (TierCheck.found=False), not an error. This is for the cases where
    the question itself couldn't be asked.
    """


@dataclass
class TierCheck:
    tier: Tier
    model: str
    found: bool


@dataclass
class VerifyResult:
    provider: str
    checks: list[TierCheck] = field(default_factory=list)

    @property
    def all_found(self) -> bool:
        return all(c.found for c in self.checks)

    @property
    def missing(self) -> list[TierCheck]:
        return [c for c in self.checks if not c.found]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "all_found": self.all_found,
            "checks": [
                {"tier": c.tier.value, "model": c.model, "found": c.found}
                for c in self.checks
            ],
        }


def _fetch_model_ids(
    provider: str,
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: float = 10.0,
) -> set[str]:
    if provider not in PROVIDER_DEFAULTS:
        raise VerifyError(
            f"unknown provider {provider!r} (known: {', '.join(sorted(PROVIDER_DEFAULTS))})"
        )
    defaults = PROVIDER_DEFAULTS[provider]
    key = api_key or os.environ.get(defaults["env_key"])
    if not key:
        raise VerifyError(
            f"no API key for {provider} -- set {defaults['env_key']} or pass one explicitly"
        )

    url = (base_url or defaults["base_url"]).rstrip("/") + defaults["path"]
    if provider == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"Authorization": f"Bearer {key}"}

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise VerifyError(f"{provider} returned {error.code}: {error.reason}") from error
    except urllib.error.URLError as error:
        raise VerifyError(f"could not reach {provider}: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise VerifyError(f"{provider} returned a response that was not valid JSON") from error

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise VerifyError(f"unexpected response shape from {provider}'s model list")

    return {
        entry["id"]
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }


def verify_models(
    model_map: ModelMap,
    provider: str = "anthropic",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> VerifyResult:
    """Check every tier's configured model name against the provider's own list.

    Raises VerifyError if the provider can't be reached or answers oddly --
    that's a setup problem, not a verification result. A model simply not
    being in the list is not an error; it is exactly what this exists to
    surface, and shows up as `found=False` on that tier's TierCheck.
    """
    known_ids = _fetch_model_ids(provider, base_url, api_key)
    checks = [
        TierCheck(tier=tier, model=model, found=model in known_ids)
        for tier, model in model_map.models.items()
    ]
    return VerifyResult(provider=provider, checks=checks)
