"""Join model predictions against market odds and compute edge.

This is the Phase-5 deliverable: for each game the model predicts, compare
the model-implied home-win probability against the de-vigged market-implied
probability, and flag games where the divergence exceeds a threshold.

Inputs (both DataFrames):
  predictions: game_id, game_date, home_team_id, away_team_id, model_prob
  odds:        game_date, home_team_id, away_team_id, bookmaker, ml_home, ml_away

Output: one row per (game, bookmaker) with `market_prob`, `edge`, `kelly_full`.

The `consensus` helper additionally averages probabilities across bookmakers
per game — more stable than any single book's line.
"""
from __future__ import annotations

import pandas as pd

from src.features.odds_math import (
    american_to_prob,
    devig_two_way,
    kelly_fraction,
)


def add_market_prob(odds: pd.DataFrame) -> pd.DataFrame:
    """Attach de-vigged home/away probabilities from moneyline columns."""
    out = odds.copy()
    raw_home = out["ml_home"].map(american_to_prob)
    raw_away = out["ml_away"].map(american_to_prob)
    fair = [devig_two_way(h, a) for h, a in zip(raw_home, raw_away)]
    out["market_prob_home"] = [f[0] for f in fair]
    out["market_prob_away"] = [f[1] for f in fair]
    out["vig"] = raw_home + raw_away - 1.0
    return out


def compare(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with odds snapshots and compute per-bookmaker edge.

    Match key is (game_date, home_team_id, away_team_id). Returns one row per
    (game, bookmaker). `edge` is signed: positive means the model gives the
    home team a higher win probability than the market does.
    """
    o = add_market_prob(odds)
    merged = predictions.merge(
        o,
        on=["game_date", "home_team_id", "away_team_id"],
        how="inner",
        suffixes=("", "_odds"),
    )
    merged["edge"] = merged["model_prob"] - merged["market_prob_home"]
    merged["kelly_full"] = [
        kelly_fraction(p, ml) if e > 0 else 0.0
        for p, ml, e in zip(merged["model_prob"], merged["ml_home"], merged["edge"])
    ]
    return merged


def consensus(compared: pd.DataFrame) -> pd.DataFrame:
    """Average across bookmakers to get one edge estimate per game."""
    agg = (
        compared.groupby(["game_id", "game_date", "home_team_id", "away_team_id"], as_index=False)
        .agg(
            model_prob=("model_prob", "first"),
            market_prob_home=("market_prob_home", "mean"),
            n_books=("bookmaker", "nunique"),
            best_ml_home=("ml_home", "max"),
        )
    )
    agg["edge"] = agg["model_prob"] - agg["market_prob_home"]
    return agg.sort_values("edge", ascending=False).reset_index(drop=True)
