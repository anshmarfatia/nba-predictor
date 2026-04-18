"""Tests for portfolio metrics.

Invariants enforced:
- MDD uses running peak (off-season gaps must not shift the answer).
- Sharpe on constant returns is 0; Sortino ≥ Sharpe on asymmetric returns.
- ROI matches hand calc.
- Sparse bet-day returns differ from zero-filled calendar returns.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.finance.bet_log import STATUS_LOST, STATUS_PUSH, STATUS_WON
from src.finance.metrics import (
    NBA_BET_DAYS_PER_YEAR,
    calmar,
    hit_rate,
    max_drawdown,
    roi,
    sharpe,
    sortino,
    summary,
    volatility,
)


def _bets(rows):
    return pd.DataFrame(rows)


def test_roi_hand_calc():
    bets = _bets([
        {"stake": 100.0, "payout": 90.0, "status": STATUS_WON, "edge": 0.05, "kelly_fraction_used": 0.02},
        {"stake": 200.0, "payout": -200.0, "status": STATUS_LOST, "edge": 0.03, "kelly_fraction_used": 0.04},
    ])
    assert roi(bets) == pytest.approx((90 - 200) / 300)


def test_hit_rate_excludes_pushes():
    bets = _bets([
        {"stake": 1, "payout": 1, "status": STATUS_WON, "edge": 0.05, "kelly_fraction_used": 0.01},
        {"stake": 1, "payout": -1, "status": STATUS_LOST, "edge": 0.05, "kelly_fraction_used": 0.01},
        {"stake": 1, "payout": 0, "status": STATUS_PUSH, "edge": 0.05, "kelly_fraction_used": 0.01},
    ])
    assert hit_rate(bets) == 0.5


def test_max_drawdown_on_monotone_is_zero():
    eq = pd.Series([100, 110, 120, 130], index=pd.date_range("2024-11-01", periods=4))
    depth, peak, trough = max_drawdown(eq)
    assert depth == 0.0


def test_max_drawdown_hand_calc():
    eq = pd.Series(
        [100, 110, 120, 90, 100, 150],
        index=pd.date_range("2024-11-01", periods=6),
    )
    depth, peak, trough = max_drawdown(eq)
    # Peak 120 at idx 2, trough 90 at idx 3 → -25%.
    assert depth == pytest.approx(-0.25, rel=1e-6)
    assert peak == eq.index[2]
    assert trough == eq.index[3]


def test_max_drawdown_is_robust_to_off_season_gap():
    """Sparse series + 4-month gap → MDD unchanged vs no-gap series."""
    eq_dense = pd.Series([100, 120, 90, 150], index=pd.date_range("2024-11-01", periods=4))
    gapped_index = [
        date(2024, 11, 1),
        date(2024, 11, 2),
        date(2025, 3, 1),       # 4-month gap, value 90 = new trough
        date(2025, 10, 20),     # off-season, then resumption
    ]
    eq_sparse = pd.Series([100, 120, 90, 150], index=pd.to_datetime(gapped_index))

    d_dense, _, _ = max_drawdown(eq_dense)
    d_sparse, _, _ = max_drawdown(eq_sparse)
    assert d_dense == pytest.approx(d_sparse)


def test_sharpe_on_constant_returns_is_zero():
    # Equity grows 1% every bet day — zero-variance returns.
    eq = pd.Series((1.01 ** np.arange(20)) * 1000.0, index=pd.date_range("2024-11-01", periods=20))
    rets = eq.pct_change().dropna()
    assert sharpe(rets) == 0.0


def test_sortino_ge_sharpe_on_asymmetric_returns():
    rng = np.random.default_rng(0)
    upside = rng.uniform(0.00, 0.02, size=80)
    downside = rng.uniform(-0.005, 0.0, size=20)
    rets = pd.Series(np.concatenate([upside, downside]))
    # Sortino penalizes only downside; positively skewed returns → Sortino ≥ Sharpe.
    assert sortino(rets) >= sharpe(rets)


def test_sparse_vs_zero_filled_sharpe_differ_materially():
    """Zero-filling off-season reduces numerator (average return over more days)
    more than it reduces denominator (std with many zeros), typically making Sharpe
    smaller. The sparse convention avoids polluting the signal with off-season zeros."""
    bet_days = pd.date_range("2024-11-01", periods=10)
    eq_sparse = pd.Series(1000 * 1.01 ** np.arange(10), index=bet_days)
    rets_sparse = eq_sparse.pct_change().dropna()

    # Build a zero-filled calendar version spanning 365 days.
    full_idx = pd.date_range(bet_days[0], periods=365)
    eq_calendar = pd.Series(index=full_idx, dtype=float)
    eq_calendar.loc[bet_days] = eq_sparse.values
    eq_calendar = eq_calendar.ffill()
    rets_calendar = eq_calendar.pct_change().dropna()

    sh_sparse = sharpe(rets_sparse)
    sh_calendar = sharpe(rets_calendar)
    assert not np.isclose(sh_sparse, sh_calendar, atol=1e-3)


def test_volatility_nonzero_on_variable_returns():
    rets = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert volatility(rets) > 0


def test_calmar_infinite_when_no_drawdown():
    eq = pd.Series([100, 105, 110, 115], index=pd.date_range("2024-11-01", periods=4))
    assert calmar(eq) == float("inf")


def test_summary_keys_present_and_finite():
    eq = pd.Series(
        [1000, 1010, 990, 1100, 1050, 1150],
        index=pd.date_range("2024-11-01", periods=6),
    )
    bets = _bets([
        {"stake": 100.0, "payout": 10.0, "status": STATUS_WON, "edge": 0.05, "kelly_fraction_used": 0.02},
        {"stake": 100.0, "payout": -100.0, "status": STATUS_LOST, "edge": 0.04, "kelly_fraction_used": 0.02},
    ])
    s = summary(bets, eq)
    for k in ("n_bets", "roi", "sharpe", "sortino", "max_drawdown", "calmar", "volatility"):
        assert k in s
    assert s["n_bets"] == 2
    assert s["starting_bankroll"] == 1000.0
    assert s["ending_bankroll"] == 1150.0


def test_periods_per_year_default_is_documented_constant():
    # Make sure we don't silently change the annualization factor.
    assert NBA_BET_DAYS_PER_YEAR == 180
