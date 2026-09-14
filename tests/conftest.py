import pytest


@pytest.fixture(autouse=True)
def isolated_thresholds(tmp_path, monkeypatch):
    """Never read or write the developer's own tuned thresholds file.

    Thresholds persist to disk by design, so without this every test run on a
    tuned machine would route differently -- and a tuner test would overwrite
    real state.
    """
    monkeypatch.setenv("TIERWISE_THRESHOLDS", str(tmp_path / "thresholds.json"))
    return tmp_path / "thresholds.json"
