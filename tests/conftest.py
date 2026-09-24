import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never discover a real tierwise.toml by walking up from the repo.

    Config discovery searches parent directories, so without this a config file
    anywhere above the checkout would change what the tests route to.
    """
    monkeypatch.setenv("TIERWISE_CONFIG", str(tmp_path / "absent.toml"))


@pytest.fixture(autouse=True)
def isolated_thresholds(tmp_path, monkeypatch):
    """Never read or write the developer's own tuned thresholds file.

    Thresholds persist to disk by design, so without this every test run on a
    tuned machine would route differently -- and a tuner test would overwrite
    real state.
    """
    monkeypatch.setenv("TIERWISE_THRESHOLDS", str(tmp_path / "thresholds.json"))
    return tmp_path / "thresholds.json"


@pytest.fixture(autouse=True)
def isolated_budget(tmp_path, monkeypatch):
    """Never read or write the developer's own spend ceiling/state.

    Same reasoning as isolated_thresholds: without this, a test run on a
    machine that has configured a real budget would inherit its spend, and a
    budget test would overwrite real state.
    """
    monkeypatch.setenv("TIERWISE_BUDGET", str(tmp_path / "budget.json"))
    return tmp_path / "budget.json"
