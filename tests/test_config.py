"""Project config for model names: discovery, precedence, and provenance."""

import json

import pytest

from tierwise import FallbackMap, ModelMap, Router, TaskSignals, Tier
from tierwise.config import ConfigError, find_config, load_config
from tierwise.mapping import CONFIG, DEFAULT, ENV, EXPLICIT

TOML = """
[models]
low = "small-model"
medium = "mid-model"
high = "big-model"
"""

try:  # TOML parsing is 3.11+, or tomli
    import tomllib  # noqa: F401
    HAS_TOML = True
except ModuleNotFoundError:
    try:
        import tomli  # noqa: F401
        HAS_TOML = True
    except ModuleNotFoundError:
        HAS_TOML = False

needs_toml = pytest.mark.skipif(not HAS_TOML, reason="no TOML parser on this Python")


def write_json(directory, models):
    path = directory / "tierwise.json"
    path.write_text(json.dumps({"models": models}))
    return path


def write_json_table(directory, key, table):
    path = directory / "tierwise.json"
    path.write_text(json.dumps({key: table}))
    return path


# -- discovery ---------------------------------------------------------------

def test_config_is_found_by_walking_up(tmp_path, monkeypatch):
    write_json(tmp_path, {"low": "small-model"})
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.delenv("TIERWISE_CONFIG")

    assert find_config(nested) == tmp_path / "tierwise.json"


def test_no_config_anywhere_is_fine(tmp_path, monkeypatch):
    # cwd matters here: ModelMap.resolve() with no explicit path walks up from
    # it looking for tierwise.toml/.json. The repo root now legitimately has
    # a tierwise.toml (see PERSONAL_SETUP.md), so this test has to run from
    # somewhere that isn't inside the repo, not just unset the env override.
    monkeypatch.delenv("TIERWISE_CONFIG")
    monkeypatch.chdir(tmp_path)
    assert load_config(tmp_path / "nope.json") == {}
    assert ModelMap.resolve().origin_for(Tier.LOW) == DEFAULT


def test_env_var_points_at_a_specific_file(tmp_path, monkeypatch):
    path = write_json(tmp_path, {"low": "small-model"})
    monkeypatch.setenv("TIERWISE_CONFIG", str(path))
    assert find_config() == path
    assert ModelMap.resolve().model_for(Tier.LOW) == "small-model"


def test_a_missing_pointed_at_file_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(tmp_path / "gone.toml"))
    assert ModelMap.resolve().model_for(Tier.LOW)   # defaults still work


@needs_toml
def test_toml_is_read(tmp_path, monkeypatch):
    path = tmp_path / "tierwise.toml"
    path.write_text(TOML)
    monkeypatch.setenv("TIERWISE_CONFIG", str(path))

    mapping = ModelMap.resolve()
    assert mapping.as_dict() == {
        "low": "small-model", "medium": "mid-model", "high": "big-model",
    }


# -- precedence --------------------------------------------------------------

def test_config_beats_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(tmp_path, {"low": "small-model"})))
    mapping = ModelMap.resolve()

    assert mapping.model_for(Tier.LOW) == "small-model"
    assert mapping.origin_for(Tier.LOW) == CONFIG
    assert mapping.origin_for(Tier.HIGH) == DEFAULT      # untouched tiers fall through


def test_env_beats_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(tmp_path, {"low": "small-model"})))
    monkeypatch.setenv("TIERWISE_MODEL_LOW", "from-env")
    mapping = ModelMap.resolve()

    assert mapping.model_for(Tier.LOW) == "from-env"
    assert mapping.origin_for(Tier.LOW) == ENV


def test_an_explicit_map_beats_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(tmp_path, {"low": "small-model"})))
    monkeypatch.setenv("TIERWISE_MODEL_LOW", "from-env")

    router = Router(model_map=ModelMap({Tier.LOW: "pinned", Tier.MEDIUM: "m", Tier.HIGH: "h"}))
    assert router.model_map.model_for(Tier.LOW) == "pinned"
    assert router.model_map.origin_for(Tier.LOW) == EXPLICIT


def test_the_router_picks_up_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(
        tmp_path, {"low": "small-model", "medium": "mid", "high": "big"})))
    decision = Router().route(TaskSignals(category="typo", lines_changed=2))
    assert decision.model == "small-model"


# -- broken config is loud ---------------------------------------------------

def test_malformed_json_raises(tmp_path, monkeypatch):
    path = tmp_path / "tierwise.json"
    path.write_text("{not json")
    monkeypatch.setenv("TIERWISE_CONFIG", str(path))
    with pytest.raises(ConfigError):
        ModelMap.resolve()


def test_an_unknown_tier_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(tmp_path, {"cheap": "x"})))
    with pytest.raises(ConfigError) as excinfo:
        ModelMap.resolve()
    assert "cheap" in str(excinfo.value)


def test_an_empty_model_name_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(tmp_path, {"low": "  "})))
    with pytest.raises(ConfigError):
        ModelMap.resolve()


def test_models_must_be_a_table(tmp_path, monkeypatch):
    path = tmp_path / "tierwise.json"
    path.write_text(json.dumps({"models": ["low", "high"]}))
    monkeypatch.setenv("TIERWISE_CONFIG", str(path))
    with pytest.raises(ConfigError):
        ModelMap.resolve()


# -- provenance --------------------------------------------------------------

def test_describe_reports_where_each_name_came_from(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG", str(write_json(tmp_path, {"medium": "mid-model"})))
    monkeypatch.setenv("TIERWISE_MODEL_HIGH", "big-from-env")
    described = ModelMap.resolve().describe()

    assert described["models"]["low"]["from"] == DEFAULT
    assert described["models"]["medium"]["from"] == CONFIG
    assert described["models"]["high"]["from"] == ENV
    assert described["configured"] is True
    assert described["config_file"].endswith("tierwise.json")


def test_unconfigured_is_reported_as_such():
    described = ModelMap.resolve().describe()
    assert described["configured"] is False
    assert described["config_file"] is None
    assert all(e["from"] == DEFAULT for e in described["models"].values())


# -- FallbackMap ---------------------------------------------------------------

def test_fallback_map_defaults_to_empty():
    fallback = FallbackMap.resolve()
    assert fallback.models == {}
    assert fallback.for_tier(Tier.LOW) is None
    assert fallback.for_tier(None) is None


def test_fallback_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG",
                       str(write_json_table(tmp_path, "fallback_models", {"low": "backup-small"})))
    fallback = FallbackMap.resolve()

    assert fallback.for_tier(Tier.LOW) == "backup-small"
    assert fallback.origins[Tier.LOW] == CONFIG
    assert fallback.for_tier(Tier.HIGH) is None


def test_fallback_env_beats_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG",
                       str(write_json_table(tmp_path, "fallback_models", {"low": "from-config"})))
    monkeypatch.setenv("TIERWISE_FALLBACK_LOW", "from-env")
    fallback = FallbackMap.resolve()

    assert fallback.for_tier(Tier.LOW) == "from-env"
    assert fallback.origins[Tier.LOW] == ENV


def test_fallback_unknown_tier_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG",
                       str(write_json_table(tmp_path, "fallback_models", {"extreme": "x"})))
    with pytest.raises(ConfigError):
        FallbackMap.resolve()


def test_fallback_must_be_a_table(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG",
                       str(write_json_table(tmp_path, "fallback_models", "not-a-table")))
    with pytest.raises(ConfigError):
        FallbackMap.resolve()


def test_fallback_empty_name_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TIERWISE_CONFIG",
                       str(write_json_table(tmp_path, "fallback_models", {"low": ""})))
    with pytest.raises(ConfigError):
        FallbackMap.resolve()


def test_fallback_as_dict():
    fallback = FallbackMap(models={Tier.LOW: "backup"})
    assert fallback.as_dict() == {"low": "backup"}
