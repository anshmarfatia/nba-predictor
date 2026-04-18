"""Portfolio metrics for a betting backtest.

Hand-rolled in ~100 lines of pandas so the Sharpe / drawdown derivations are
visible to anyone reading the code. `empyrical` is unmaintained; `quantstats`
is heavy and noisy — neither pays rent here.

Two conventions matter for sports-betting equity curves and they are both
non-obvious; the functions below take them as defaults but expose them
explicitly so readers can swap.

1. **Sparse equity, bet-day returns.** NBA has a ~6-month off-season. An
   equity series indexed by calendar days and forward-filled across the
   gap produces zero-return rows for ~180 days every year, artificially
   shrinking volatility's denominator and inflating Sharpe. All inputs
   below assume equity is indexed by *bet days only* and returns are
   computed on that sparse index.

2. **Running peak for drawdown.** `equity.cummax()` carries the last peak
   forward across flat periods. That's what we want — the peak is always
   the running high *before* the trough, never a global max.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NBA_BET_DAYS_PER_YEAR = 180   # regular-season days when at least one game is played


# ---------------------------------------------------------------------------
# Bet-level statistics
# ---------------------------------------------------------------------------

def _empty(bets: pd.DataFrame, col: str) -> bool:
    """Bets DataFrames come in two flavors: empty-no-columns (no bets placed)
    and empty-with-columns (all filtered out). Treat both as 'nothing to measure'."""
    return bets.empty or col not in bets.columns


def roi(bets: pd.DataFrame) -> float:
    """Total P&L divided by total staked. 0 if no bets."""
    if _empty(bets, "stake"):
        return 0.0
    staked = bets["stake"].sum()
    return float(bets["payout"].fillna(0.0).sum() / staked) if staked else 0.0


def hit_rate(bets: pd.DataFrame) -> float:
    """Wins / (wins + losses). Pushes and voids excluded."""
    if _empty(bets, "status"):
        return float("nan")
    settled = bets[bets["status"].isin(("won", "lost"))]
    if settled.empty:
        return float("nan")
    return float((settled["status"] == "won").mean())


def avg_edge(bets: pd.DataFrame) -> float:
    if _empty(bets, "edge"):
        return 0.0
    return float(bets["edge"].mean())


def avg_kelly_used(bets: pd.DataFrame) -> float:
    if _empty(bets, "kelly_fraction_used"):
        return 0.0
    return float(bets["kelly_fraction_used"].mean())


# ---------------------------------------------------------------------------
# Time-series statistics
# ---------------------------------------------------------------------------

def bet_day_returns(equity: pd.Series) -> pd.Series:
    """pct_change on the sparse, bet-day-only equity series."""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().dropna()


def volatility(returns: pd.Series, periods_per_year: int = NBA_BET_DAYS_PER_YEAR) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


_ZERO_STD = 1e-12


def sharpe(
    returns: pd.Series,
    periods_per_year: int = NBA_BET_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Annualized Sharpe on bet-day returns.

    `rf` is a per-period risk-free rate (usually 0 for sports bets). For
    fractional-Kelly strategies Sharpe often under-sells growth because of
    fat right tails — report Calmar alongside.
    """
    if returns.empty:
        return 0.0
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std < _ZERO_STD:
        return 0.0
    excess = returns - rf
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino(
    returns: pd.Series,
    periods_per_year: int = NBA_BET_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Like Sharpe but only downside deviation in the denominator."""
    if returns.empty:
        return 0.0
    excess = returns - rf
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return float("inf")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Largest peak-to-trough drop on the running high.

    Returns (depth, peak_date, trough_date). `depth` is in [-1, 0]; 0 means
    the curve is monotonically non-decreasing.
    """
    if equity.empty:
        return 0.0, None, None
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough_date = drawdown.idxmin()
    depth = float(drawdown.loc[trough_date])
    peak_slice = running_peak.loc[:trough_date]
    peak_date = peak_slice[peak_slice == peak_slice.loc[trough_date]].index[0]
    return depth, peak_date, trough_date


def calmar(
    equity: pd.Series,
    periods_per_year: int = NBA_BET_DAYS_PER_YEAR,
) -> float:
    """Annualized return / |max drawdown|. Drawdown-normalized growth."""
    if len(equity) < 2:
        return 0.0
    n_periods = len(equity) - 1
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    if n_periods <= 0:
        return 0.0
    annualized = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return float("inf")
    return float(annualized / abs(mdd))


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------

def summary(
    bets: pd.DataFrame,
    equity: pd.Series,
    periods_per_year: int = NBA_BET_DAYS_PER_YEAR,
) -> dict:
    """Headline metrics for a backtest run. Everything is finite-safe."""
    returns = bet_day_returns(equity)
    mdd, peak, trough = max_drawdown(equity)
    starting = float(equity.iloc[0]) if not equity.empty else 0.0
    ending = float(equity.iloc[-1]) if not equity.empty else 0.0
    return {
        "n_bets": int(len(bets)),
        "n_bet_days": int(len(equity)),
        "starting_bankroll": starting,
        "ending_bankroll": ending,
        "total_staked": float(bets["stake"].sum()) if not bets.empty else 0.0,
        "total_pnl": ending - starting,
        "roi": roi(bets),
        "hit_rate": hit_rate(bets),
        "avg_edge": avg_edge(bets),
        "avg_kelly_used": avg_kelly_used(bets),
        "volatility": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": mdd,
        "max_drawdown_peak": peak,
        "max_drawdown_trough": trough,
        "calmar": calmar(equity, periods_per_year),
    }
