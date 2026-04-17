"""Team Elo rating feature.

Chronologically walks through a matchup frame, maintaining one running rating
per team. For each game we attach the *pre-game* rating as a feature, then
update both ratings based on the outcome. Runs in O(n) with a single pass.

Design notes:
  - Default K=20 matches FiveThirtyEight's NBA Elo.
  - Home-court advantage is applied to the expected-score calculation only
    (the rating itself is location-independent).
  - Between seasons, ratings regress toward 1500 by `season_regression`.
    Raw continuity overweights past success; pure reset discards real signal.
  - Margin-of-victory multipliers are intentionally omitted for v1 — easy
    to add later if calibration shows systematic bias on blowouts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _walk_elo(
    matchup: pd.DataFrame,
    k: float,
    home_advantage: float,
    initial_rating: float,
    season_regression: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, float], dict[int, str]]:
    """Walk a matchup frame in date order. Returns (home_pre, away_pre,
    final_ratings, last_season_per_team). Shared between `add_elo` (training)
    and `final_elos` (prediction time)."""
    ratings: dict[int, float] = {}
    last_season: dict[int, str] = {}

    home_pre = np.empty(len(matchup), dtype=float)
    away_pre = np.empty(len(matchup), dtype=float)

    home_ids = matchup["home_team_id"].to_numpy()
    away_ids = matchup["away_team_id"].to_numpy()
    seasons = matchup["season"].to_numpy()
    home_won = matchup["home_won"].to_numpy()

    for i in range(len(matchup)):
        h, a, season = int(home_ids[i]), int(away_ids[i]), seasons[i]

        for tid in (h, a):
            if tid not in ratings:
                ratings[tid] = initial_rating
            elif last_season.get(tid) != season:
                ratings[tid] = (
                    ratings[tid] * (1 - season_regression)
                    + initial_rating * season_regression
                )
            last_season[tid] = season

        rh, ra = ratings[h], ratings[a]
        home_pre[i] = rh
        away_pre[i] = ra

        exp_h = 1.0 / (1.0 + 10 ** (-((rh + home_advantage) - ra) / 400.0))
        actual_h = float(home_won[i])

        ratings[h] = rh + k * (actual_h - exp_h)
        ratings[a] = ra + k * ((1.0 - actual_h) - (1.0 - exp_h))

    return home_pre, away_pre, ratings, last_season


def add_elo(
    matchup: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 100.0,
    initial_rating: float = 1500.0,
    season_regression: float = 0.25,
) -> pd.DataFrame:
    """Add `home_elo_pre`, `away_elo_pre`, and `elo_diff` to a matchup frame.

    The frame must contain `home_team_id`, `away_team_id`, `home_won`,
    `season`, and `game_date`. Rows are processed in date order.
    """
    if matchup.empty:
        return matchup.copy()

    out = matchup.sort_values(["game_date", "game_id"]).reset_index(drop=True).copy()
    home_pre, away_pre, _, _ = _walk_elo(out, k, home_advantage, initial_rating, season_regression)

    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff"] = home_pre - away_pre
    return out


def final_elos(
    matchup: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 100.0,
    initial_rating: float = 1500.0,
    season_regression: float = 0.25,
) -> tuple[dict[int, float], dict[int, str]]:
    """Return the POST-game rating + last-seen-season per team after walking
    the whole matchup frame. Used by predict.py to look up each team's current
    rating before scoring an upcoming game."""
    if matchup.empty:
        return {}, {}
    ordered = matchup.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    _, _, ratings, last_season = _walk_elo(
        ordered, k, home_advantage, initial_rating, season_regression
    )
    return ratings, last_season


def pre_game_elo(
    team_id: int,
    game_season: str,
    ratings: dict[int, float],
    last_season: dict[int, str],
    initial_rating: float = 1500.0,
    season_regression: float = 0.25,
) -> float:
    """Look up a team's pre-game Elo for an upcoming game. Applies the
    season-boundary regression if the team hasn't played this season yet."""
    r = ratings.get(team_id, initial_rating)
    if team_id in last_season and last_season[team_id] != game_season:
        r = r * (1 - season_regression) + initial_rating * season_regression
    return r
