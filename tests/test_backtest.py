"""Tests for the backtest engine.

The tests focus on the three non-obvious invariants from the plan review:
1. No look-ahead (predicted_at > game_date must raise).
2. Simultaneous-Kelly normalization (Σfᵢ ≤ max_concurrent_exposure).
3. Single-bookmaker handling (consensus path is skipped).

Plus the usual: exact bankroll on a hand-computed fixture, zero bets when
no edge, home/away/best side selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.finance.backtest import (
    BacktestConfig,
    normalize_concurrent,
    run_backtest,
)
from src.finance.staking import FlatStake, FractionalKelly, FullKelly, ThresholdKelly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_game(
    game_id: str,
    game_date,
    home_id: int,
    away_id: int,
    model_prob: float,
    ml_home: float,
    ml_away: float,
    home_won: int,
    bookmaker: str = "kaggle_close",
):
    return {
        "pred": {
            "game_id": game_id,
            "game_date": pd.Timestamp(game_date),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "model_prob": model_prob,
            "predicted_at": pd.Timestamp(game_date) - pd.Timedelta(days=1),
        },
        "odds": {
            "game_date": pd.Timestamp(game_date),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "bookmaker": bookmaker,
            "ml_home": ml_home,
            "ml_away": ml_away,
        },
        "outcome": {"game_id": game_id, "home_won": home_won},
    }


def _from_games(games):
    preds = pd.DataFrame([g["pred"] for g in games])
    odds = pd.DataFrame([g["odds"] for g in games])
    outcomes = pd.DataFrame([g["outcome"] for g in games])
    return preds, odds, outcomes


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_look_ahead_in_predictions_raises():
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.70, -150, +130, 1)]
    preds, odds, outcomes = _from_games(games)
    # Poison a prediction with a post-game timestamp.
    preds.loc[0, "predicted_at"] = pd.Timestamp("2024-11-02")
    cfg = BacktestConfig(strategy=FractionalKelly(0.25))
    with pytest.raises(AssertionError):
        run_backtest(preds, odds, outcomes, cfg)


def test_zero_edge_produces_no_bets():
    # Fair 50/50 market with model_prob=0.5 → edge=0.
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.50, +100, +100, 1)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(strategy=FullKelly(), min_edge=0.0)
    result = run_backtest(preds, odds, outcomes, cfg)
    assert result.bets.empty


def test_min_edge_gate_filters_small_edges():
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.52, +100, +100, 1)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(strategy=FractionalKelly(0.25), min_edge=0.05)
    result = run_backtest(preds, odds, outcomes, cfg)
    assert result.bets.empty


def test_simultaneous_kelly_cap_caps_total_exposure_and_scales_proportionally():
    """Four concurrent bets each suggest a large Kelly — realized stakes must
    sum to the cap, preserving relative ratios."""
    games = [
        # 4 games same day, same model_prob and odds → same kelly each.
        _make_game(f"g{i}", "2024-11-01", 10 + i, 20 + i, 0.70, +100, -120, 1)
        for i in range(4)
    ]
    preds, odds, outcomes = _from_games(games)
    cap = 0.40
    cfg = BacktestConfig(
        strategy=FullKelly(),
        min_edge=0.0,
        max_concurrent_exposure=cap,
        starting_bankroll=10_000.0,
    )
    result = run_backtest(preds, odds, outcomes, cfg)

    assert len(result.bets) == 4
    # Total realized stake = cap * bankroll.
    total_stake = result.bets["stake"].sum()
    assert total_stake == pytest.approx(cap * 10_000.0, rel=1e-9)
    # Equal-Kelly bets → equal realized stakes.
    stakes = result.bets["stake"].to_numpy()
    assert np.allclose(stakes, stakes[0])


def test_simultaneous_kelly_cap_preserves_ratios_on_unequal_kelly():
    """Unequal Kelly suggestions → proportional scaling so the ratio is preserved."""
    games = [
        _make_game("g1", "2024-11-01", 1, 2, 0.70, +150, -180, 1),
        _make_game("g2", "2024-11-01", 3, 4, 0.55, +100, -120, 1),
    ]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(
        strategy=FullKelly(),
        min_edge=0.0,
        max_concurrent_exposure=0.10,   # tight cap
        starting_bankroll=10_000.0,
    )
    result = run_backtest(preds, odds, outcomes, cfg)
    assert len(result.bets) == 2
    f_suggested = result.bets["kelly_full"].to_numpy()
    f_used = result.bets["kelly_fraction_used"].to_numpy()
    # Ratio of suggested → ratio of used must be the same.
    assert f_used[0] / f_used[1] == pytest.approx(f_suggested[0] / f_suggested[1], rel=1e-9)
    assert f_used.sum() == pytest.approx(0.10, rel=1e-9)


def test_no_cap_when_under_exposure_limit():
    """Single small-Kelly bet → no haircut applied."""
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.55, +100, -120, 1)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(strategy=FractionalKelly(0.25), min_edge=0.0)
    result = run_backtest(preds, odds, outcomes, cfg)
    assert len(result.bets) == 1
    # 0.25x Kelly on this bet is far below the 0.50 cap — should be untouched.
    row = result.bets.iloc[0]
    assert row["kelly_fraction_used"] == pytest.approx(0.25 * row["kelly_full"])


def test_single_bookmaker_path_produces_bets():
    """With only one bookmaker, consensus() is skipped and the single line is used."""
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.65, +100, -120, 1, bookmaker="kaggle_close")]
    preds, odds, outcomes = _from_games(games)
    # Explicitly set bookmaker=None so the "auto-skip consensus" branch is exercised.
    cfg = BacktestConfig(strategy=FractionalKelly(0.25), bookmaker=None, min_edge=0.0)
    result = run_backtest(preds, odds, outcomes, cfg)
    assert len(result.bets) == 1
    # Bookmaker on the bet should be the single source, not "consensus".
    assert result.bets.iloc[0]["bookmaker"] == "kaggle_close"


def test_winning_bet_pnl_matches_payout_formula():
    # Bet the home side at +150, home wins → payout = stake * 1.5.
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.60, +150, -180, 1)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(
        strategy=FlatStake(unit=100.0), min_edge=0.0, starting_bankroll=10_000.0, side="home"
    )
    result = run_backtest(preds, odds, outcomes, cfg)
    assert len(result.bets) == 1
    assert result.bets.iloc[0]["payout"] == pytest.approx(100.0 * 1.5)
    # Equity ends at 10_000 + 150 = 10_150 (last bet-day value).
    assert result.equity.iloc[-1] == pytest.approx(10_150.0)


def test_losing_bet_loses_stake():
    # Bet the home side at +150, home loses → payout = -stake.
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.60, +150, -180, 0)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(
        strategy=FlatStake(unit=100.0), min_edge=0.0, starting_bankroll=10_000.0, side="home"
    )
    result = run_backtest(preds, odds, outcomes, cfg)
    assert result.bets.iloc[0]["payout"] == pytest.approx(-100.0)
    assert result.equity.iloc[-1] == pytest.approx(9_900.0)


def test_side_best_picks_positive_edge_side():
    # Model thinks away is way undervalued.
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.30, -150, +130, 0)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(strategy=FractionalKelly(0.25), min_edge=0.0, side="best")
    result = run_backtest(preds, odds, outcomes, cfg)
    assert len(result.bets) == 1
    assert result.bets.iloc[0]["side"] == "away"


def test_home_only_mode_skips_when_home_has_no_edge():
    games = [_make_game("g1", "2024-11-01", 1, 2, 0.30, -150, +130, 0)]
    preds, odds, outcomes = _from_games(games)
    cfg = BacktestConfig(strategy=FractionalKelly(0.25), min_edge=0.0, side="home")
    result = run_backtest(preds, odds, outcomes, cfg)
    assert result.bets.empty


def test_normalize_concurrent_is_idempotent_when_under_cap():
    assert normalize_concurrent([0.05, 0.05], cap=0.5) == [0.05, 0.05]


def test_normalize_concurrent_scales_proportionally_when_over_cap():
    out = normalize_concurrent([0.30, 0.60], cap=0.30)
    assert sum(out) == pytest.approx(0.30)
    assert out[0] / out[1] == pytest.approx(0.30 / 0.60)


def test_normalize_concurrent_handles_empty():
    assert normalize_concurrent([], cap=0.5) == []
