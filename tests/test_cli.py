import json

import pytest

from tierwise.cli import main


def test_route_prints_human_output(capsys):
    assert main(["route", "fix a typo", "--category", "typo", "--lines", "1"]) == 0
    out = capsys.readouterr().out
    assert "tier:" in out and "low" in out
    assert "model:" in out


def test_route_model_only(capsys):
    assert main(["route", "--category", "typo", "--lines", "1", "--model-only"]) == 0
    out = capsys.readouterr().out.strip()
    assert out and "\n" not in out


def test_route_json_output(capsys):
    assert main(["route", "big rework", "--category", "architecture", "--files", "30",
                 "--lines", "2000", "--depth", "6", "--needs-context",
                 "--ambiguity", "0.9", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "high"
    assert payload["signals"]["file_count"] == 30


def test_route_respects_hint_and_min_tier(capsys):
    main(["route", "--category", "typo", "--hint", "high", "--json"])
    assert json.loads(capsys.readouterr().out)["source"] == "hint"

    main(["route", "--category", "typo", "--lines", "1", "--min-tier", "medium", "--json"])
    assert json.loads(capsys.readouterr().out)["tier"] == "medium"


def test_route_writes_telemetry(tmp_path, capsys):
    log = tmp_path / "events.jsonl"
    main(["route", "--category", "typo", "--lines", "1", "--telemetry", str(log)])
    capsys.readouterr()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 1 and events[0]["event"] == "routing_decision"


def test_explain_shows_contributions(capsys):
    assert main(["explain", "--category", "feature", "--files", "4", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "contributions" in payload
    assert payload["contributions"]["file_count"] == pytest.approx(2.0)
    assert 0.0 <= payload["score"] <= 1.0


def test_models_command_lists_all_tiers(capsys):
    assert main(["models", "--json"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert set(described["models"]) == {"low", "medium", "high"}
    assert described["configured"] is False


def test_models_command_marks_defaults_as_defaults(capsys):
    """A built-in model id must not read as a chosen one."""
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "[default]" in out
    assert "built-in defaults, which age" in out


def test_models_verify_reports_missing(capsys, monkeypatch):
    from tierwise.mapping import ModelMap
    from tierwise.models import Tier
    from tierwise.verify import TierCheck, VerifyResult

    def fake_verify_models(model_map, provider="anthropic", base_url=None):
        return VerifyResult(provider=provider, checks=[
            TierCheck(tier=Tier.LOW, model="claude-haiku-4-5-20251001", found=True),
            TierCheck(tier=Tier.MEDIUM, model="claude-sonnet-5", found=True),
            TierCheck(tier=Tier.HIGH, model="claude-opus-99-gone", found=False),
        ])

    monkeypatch.setattr("tierwise.cli.verify_models", fake_verify_models)

    assert main(["models", "--verify", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_found"] is False
    assert payload["checks"][2]["found"] is False


def test_models_verify_all_found_exits_zero(capsys, monkeypatch):
    from tierwise.models import Tier
    from tierwise.verify import TierCheck, VerifyResult

    def fake_verify_models(model_map, provider="anthropic", base_url=None):
        return VerifyResult(provider=provider, checks=[
            TierCheck(tier=t, model="m", found=True) for t in Tier
        ])

    monkeypatch.setattr("tierwise.cli.verify_models", fake_verify_models)

    assert main(["models", "--verify"]) == 0
    out = capsys.readouterr().out
    assert "all configured models exist" in out


def test_models_verify_reports_error_without_crashing(capsys, monkeypatch):
    from tierwise.verify import VerifyError

    def fake_verify_models(model_map, provider="anthropic", base_url=None):
        raise VerifyError("no API key for anthropic -- set ANTHROPIC_API_KEY")

    monkeypatch.setattr("tierwise.cli.verify_models", fake_verify_models)

    assert main(["models", "--verify"]) == 1
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out


def test_missing_subcommand_exits(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_route_warns_when_only_a_description_is_given(capsys):
    assert main(["route", "migrate billing to a new payment provider"]) == 0
    captured = capsys.readouterr()
    assert "no signals given" in captured.err
    assert "tier:" in captured.out


def test_route_warns_on_an_unknown_category(capsys):
    assert main(["route", "x", "--category", "billing", "--lines", "5"]) == 0
    assert "unknown category 'billing'" in capsys.readouterr().err


def test_model_only_output_stays_clean(capsys):
    assert main(["route", "just words", "--model-only"]) == 0
    assert capsys.readouterr().err == ""
