"""Verifying configured model names against a provider's own model list.

Deliberately not an integration test against a real API -- these mock
urllib.request.urlopen so the suite stays network-free (consistent with the
rest of tierwise, which has no runtime dependencies and no network in tests).
"""

import io
import json
import urllib.error

import pytest

from tierwise.mapping import ModelMap
from tierwise.models import Tier
from tierwise.verify import VerifyError, _fetch_model_ids, verify_models


def _model_map():
    return ModelMap(models={
        Tier.LOW: "claude-haiku-4-5-20251001",
        Tier.MEDIUM: "claude-sonnet-5",
        Tier.HIGH: "claude-opus-99-does-not-exist",
    })


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, payload=None, http_error=None, url_error=None):
    def fake_urlopen(request, timeout=None):
        if http_error is not None:
            raise http_error
        if url_error is not None:
            raise url_error
        return _FakeResponse(payload)

    monkeypatch.setattr("tierwise.verify.urllib.request.urlopen", fake_urlopen)


def test_verify_reports_found_and_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _patch_urlopen(monkeypatch, payload={"data": [
        {"id": "claude-haiku-4-5-20251001"},
        {"id": "claude-sonnet-5"},
    ]})

    result = verify_models(_model_map(), provider="anthropic")

    assert not result.all_found
    found = {c.tier: c.found for c in result.checks}
    assert found[Tier.LOW] is True
    assert found[Tier.MEDIUM] is True
    assert found[Tier.HIGH] is False
    assert [c.model for c in result.missing] == ["claude-opus-99-does-not-exist"]


def test_verify_all_found(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _patch_urlopen(monkeypatch, payload={"data": [
        {"id": "claude-haiku-4-5-20251001"},
        {"id": "claude-sonnet-5"},
        {"id": "claude-opus-99-does-not-exist"},
    ]})

    result = verify_models(_model_map(), provider="anthropic")
    assert result.all_found
    assert result.missing == []


def test_missing_api_key_raises_verify_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(VerifyError, match="ANTHROPIC_API_KEY"):
        verify_models(_model_map(), provider="anthropic")


def test_unknown_provider_raises():
    with pytest.raises(VerifyError, match="unknown provider"):
        _fetch_model_ids("made-up-provider", None, "sk-test")


def test_http_error_raises_verify_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    error = urllib.error.HTTPError(url="https://api.anthropic.com/v1/models",
                                    code=401, msg="Unauthorized", hdrs=None, fp=io.BytesIO(b""))
    _patch_urlopen(monkeypatch, http_error=error)

    with pytest.raises(VerifyError, match="401"):
        verify_models(_model_map(), provider="anthropic")


def test_unreachable_provider_raises_verify_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _patch_urlopen(monkeypatch, url_error=urllib.error.URLError("no route to host"))

    with pytest.raises(VerifyError, match="could not reach"):
        verify_models(_model_map(), provider="anthropic")


def test_response_without_data_field_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _patch_urlopen(monkeypatch, payload={"unexpected": "shape"})

    with pytest.raises(VerifyError, match="unexpected response shape"):
        verify_models(_model_map(), provider="anthropic")


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_urlopen(monkeypatch, payload={"data": [{"id": "claude-haiku-4-5-20251001"},
                                                   {"id": "claude-sonnet-5"},
                                                   {"id": "claude-opus-99-does-not-exist"}]})

    result = verify_models(_model_map(), provider="anthropic", api_key="sk-explicit")
    assert result.all_found


def test_openai_provider_uses_bearer_auth(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        return _FakeResponse({"data": [{"id": "gpt-5"}]})

    monkeypatch.setattr("tierwise.verify.urllib.request.urlopen", fake_urlopen)

    model_map = ModelMap(models={Tier.LOW: "gpt-5", Tier.MEDIUM: "gpt-5",
                                  Tier.HIGH: "gpt-5"})
    result = verify_models(model_map, provider="openai")
    assert result.all_found
    assert captured["headers"].get("Authorization") == "Bearer sk-oai"
    assert "api.openai.com" in captured["url"]
