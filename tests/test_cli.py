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
    assert main(["models"]) == 0
    mapping = json.loads(capsys.readouterr().out)
    assert set(mapping) == {"low", "medium", "high"}


def test_missing_subcommand_exits(capsys):
    with pytest.raises(SystemExit):
        main([])
