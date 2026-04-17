"""Loaders for the matchup grain (one row per game).

The DB stores per-team-game rows in `team_games` and `team_games_advanced`.
This module joins them and pivots to a single home/away row per game,
ready for feature engineering and model training.

Typical flow:
    df = load_team_games(engine)         # per-team-game
    df = add_rolling(df, ...)            # still per-team-game
    matchup = to_matchup(df)             # one row per game, home_/away_ cols
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

_BASE_QUERY = """
SELECT
    g.game_id, g.team_id, g.season, g.season_type, g.game_date,
    g.team_abbreviation, g.matchup, g.is_home, g.wl,
    g.pts, g.plus_minus, g.fg_pct, g.fg3_pct, g.ft_pct,
    g.reb, g.ast, g.tov,
    a.off_rating, a.def_rating, a.net_rating,
    a.pace, a.ts_pct, a.efg_pct,
    a.ast_pct, a.oreb_pct, a.dreb_pct, a.tm_tov_pct, a.pie
FROM team_games g
LEFT JOIN team_games_advanced a USING (game_id, team_id)
WHERE g.season_type = :season_type
ORDER BY g.game_date, g.game_id, g.team_id
"""


def load_team_games(
    engine: Engine,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Load per-team-game rows joined with advanced stats."""
    df = pd.read_sql_query(
        text(_BASE_QUERY), engine, params={"season_type": season_type}
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["won"] = (df["wl"] == "W").astype("int8")
    return df


def to_matchup(df: pd.DataFrame, drop_neutral: bool = True) -> pd.DataFrame:
    """Collapse two team-game rows per game into one matchup row.

    Returns one row per `game_id` with `home_*` and `away_*` prefixed columns,
    plus shared columns (`game_id`, `season`, `season_type`, `game_date`).
    The target column is `home_won`.

    Neutral-site games (both teams `is_home=False`) are dropped by default —
    they break the home/away contract. ~5 per season historically.
    """
    if df.empty:
        return df.copy()

    if drop_neutral:
        has_home = df.groupby("game_id")["is_home"].transform("any")
        df = df.loc[has_home].copy()

    shared_cols = ["game_id", "season", "season_type", "game_date"]
    side_drop = ["is_home", "matchup", "season", "season_type", "game_date"]

    home = (
        df.loc[df["is_home"]]
        .drop(columns=side_drop, errors="ignore")
        .add_prefix("home_")
        .rename(columns={"home_game_id": "game_id"})
    )
    away = (
        df.loc[~df["is_home"]]
        .drop(columns=side_drop, errors="ignore")
        .add_prefix("away_")
        .rename(columns={"away_game_id": "game_id"})
    )

    base = df[shared_cols].drop_duplicates(subset="game_id")
    out = base.merge(home, on="game_id").merge(away, on="game_id")
    return out.sort_values("game_date").reset_index(drop=True)
