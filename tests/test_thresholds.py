import json

import pytest

from tierwise.thresholds import DEFAULT_THRESHOLDS, QUALITY_FLOOR_PRESETS, Thresholds, resolve_path


def test_defaults_are_valid():
    Thresholds().validate()
    assert DEFAULT_THRESHOLDS.low_medium < DEFAULT_THRESHOLDS.medium_high


def test_roundtrip_persists(isolated_thresholds):
    original = Thresholds(low_medium=0.25, medium_high=0.66, llm_fallback=0.6)
    path = original.save()
    assert path == isolated_thresholds
    loaded = Thresholds.load()
    assert loaded.low_medium == pytest.approx(0.25)
    assert loaded.medium_high == pytest.approx(0.66)
    assert loaded.llm_fallback == pytest.approx(0.6)
    assert loaded.version == 1


def test_save_bumps_version_and_stamps_time(isolated_thresholds):
    t = Thresholds()
    t.save()
    t.save()
    assert t.version == 2
    assert t.updated_at is not None


def test_missing_file_falls_back_to_defaults():
    assert Thresholds.load().to_dict() == Thresholds().to_dict()


def test_corrupt_file_falls_back_to_defaults(isolated_thresholds):
    isolated_thresholds.parent.mkdir(parents=True, exist_ok=True)
    isolated_thresholds.write_text("{not json")
    assert Thresholds.load().low_medium == pytest.approx(Thresholds().low_medium)


def test_invalid_persisted_values_are_rejected(isolated_thresholds):
    """A file that would break routing must not be trusted."""
    isolated_thresholds.parent.mkdir(parents=True, exist_ok=True)
    isolated_thresholds.write_text(json.dumps({"low_medium": 0.8, "medium_high": 0.2}))
    loaded = Thresholds.load()
    assert loaded.low_medium < loaded.medium_high


def test_unknown_keys_are_ignored(isolated_thresholds):
    isolated_thresholds.parent.mkdir(parents=True, exist_ok=True)
    isolated_thresholds.write_text(json.dumps({"low_medium": 0.28, "wat": 1}))
    assert Thresholds.load().low_medium == pytest.approx(0.28)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_medium": 0.5, "medium_high": 0.55},   # gap too small
        {"low_medium": 0.0},
        {"medium_high": 1.0},
        {"boundary_margin": 0.0},
        {"llm_fallback": 1.5},
    ],
)
def test_validate_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        Thresholds(**kwargs).validate()


def test_env_var_controls_path(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "custom.json"
    monkeypatch.setenv("TIERWISE_THRESHOLDS", str(target))
    assert resolve_path() == target
    Thresholds().save()
    assert target.exists()


def test_copy_is_independent():
    a = Thresholds()
    b = a.copy()
    b.low_medium = 0.1
    assert a.low_medium != b.low_medium


def test_quality_floor_presets_are_all_valid():
    for name in QUALITY_FLOOR_PRESETS:
        Thresholds.from_quality_floor(name).validate()


def test_balanced_preset_matches_plain_defaults():
    balanced = Thresholds.from_quality_floor("balanced")
    assert balanced.low_medium == pytest.approx(Thresholds().low_medium)
    assert balanced.medium_high == pytest.approx(Thresholds().medium_high)


def test_strict_routes_more_work_up_than_lenient():
    strict = Thresholds.from_quality_floor("strict")
    lenient = Thresholds.from_quality_floor("lenient")
    assert strict.low_medium < lenient.low_medium
    assert strict.medium_high < lenient.medium_high


def test_unknown_quality_floor_raises():
    with pytest.raises(ValueError, match="unknown quality floor"):
        Thresholds.from_quality_floor("nonexistent")


def test_quality_floor_from_config_is_the_untuned_default(tmp_path, monkeypatch):
    config = tmp_path / "tierwise.toml"
    config.write_text('quality_floor = "strict"\n')
    monkeypatch.setenv("TIERWISE_CONFIG", str(config))

    loaded = Thresholds.load()
    strict = Thresholds.from_quality_floor("strict")
    assert loaded.low_medium == pytest.approx(strict.low_medium)
    assert loaded.medium_high == pytest.approx(strict.medium_high)


def test_tuned_thresholds_win_over_quality_floor_config(tmp_path, monkeypatch, isolated_thresholds):
    config = tmp_path / "tierwise.toml"
    config.write_text('quality_floor = "strict"\n')
    monkeypatch.setenv("TIERWISE_CONFIG", str(config))

    tuned = Thresholds(low_medium=0.33, medium_high=0.77)
    tuned.save()

    loaded = Thresholds.load()
    assert loaded.low_medium == pytest.approx(0.33)
    assert loaded.medium_high == pytest.approx(0.77)


def test_unknown_quality_floor_in_config_falls_back_to_defaults(tmp_path, monkeypatch):
    config = tmp_path / "tierwise.toml"
    config.write_text('quality_floor = "made-up"\n')
    monkeypatch.setenv("TIERWISE_CONFIG", str(config))

    assert Thresholds.load().to_dict() == Thresholds().to_dict()
