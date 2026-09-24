"""The spend ceiling: a safety rail, not multi-tenant budgeting."""

import pytest

from tierwise.budget import BudgetState


def test_unset_ceiling_is_never_exceeded():
    budget = BudgetState()
    budget.record_spend(1_000_000)  # no-op: no limit configured
    assert not budget.exceeded()
    assert budget.remaining() is None
    assert budget.spent_usd == 0.0


def test_spend_accumulates_until_the_ceiling():
    budget = BudgetState(limit_usd=10.0)
    now = 1_000_000.0
    budget.record_spend(4.0, now=now)
    budget.record_spend(4.0, now=now + 1)
    assert not budget.exceeded(now=now + 2)
    assert budget.remaining(now=now + 2) == pytest.approx(2.0)

    budget.record_spend(2.5, now=now + 3)
    assert budget.exceeded(now=now + 4)
    assert budget.remaining(now=now + 4) == 0.0


def test_period_rolls_over_and_resets_spend():
    budget = BudgetState(limit_usd=10.0, period="daily")
    now = 1_000_000.0
    budget.record_spend(10.0, now=now)
    assert budget.exceeded(now=now + 10)

    day = 24 * 60 * 60
    assert not budget.exceeded(now=now + day + 1)
    assert budget.spent_usd == 0.0


def test_negative_or_zero_spend_is_ignored():
    budget = BudgetState(limit_usd=10.0)
    budget.record_spend(0)
    budget.record_spend(-5)
    assert budget.spent_usd == 0.0


def test_roundtrip_persists(isolated_budget):
    original = BudgetState(limit_usd=25.0, period="weekly", spent_usd=3.5,
                            period_started_at=1_700_000_000.0)
    path = original.save()
    assert path == isolated_budget

    loaded = BudgetState.load()
    assert loaded.limit_usd == pytest.approx(25.0)
    assert loaded.period == "weekly"
    assert loaded.spent_usd == pytest.approx(3.5)


def test_missing_file_falls_back_to_unset():
    loaded = BudgetState.load()
    assert loaded.limit_usd is None
    assert not loaded.exceeded()


def test_corrupt_file_falls_back_to_unset(isolated_budget):
    isolated_budget.parent.mkdir(parents=True, exist_ok=True)
    isolated_budget.write_text("{not json")
    assert BudgetState.load().limit_usd is None


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        BudgetState(limit_usd=0).validate()
    with pytest.raises(ValueError):
        BudgetState(limit_usd=-5).validate()


def test_invalid_period_rejected():
    with pytest.raises(ValueError):
        BudgetState(period="yearly").validate()
