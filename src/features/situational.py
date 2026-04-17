"""Situational features: calendar, season-phase, and COVID flag.

Operates on per-team-game rows (pre-pivot). After the matchup pivot these
columns become `home_*` / `away_*` automatically.
"""
from __future__ import annotations

import pandas as pd

COVID_SEASONS: frozenset[str] = frozenset({"2019-20", "2020-21"})


def add_situational(df: pd.DataFrame, games_per_season: int = 82) -> pd.DataFrame:
    """Add `month`, `game_number`, `season_progress`, and `is_covid_season`.

    `game_number` is 1-indexed and per (team, season). `season_progress` is
    normalized to [0, 1] assuming an 82-game regular season (use a smaller
    denominator when loading lockout or bubble seasons separately).
    """
    if df.empty:
        return df.copy()

    out = df.sort_values(["team_id", "game_date", "game_id"]).copy()
    out["month"] = out["game_date"].dt.month.astype("int8")
    out["is_covid_season"] = out["season"].isin(COVID_SEASONS).astype("int8")
    out["game_number"] = out.groupby(["team_id", "season"]).cumcount().astype("int16") + 1
    out["season_progress"] = out["game_number"] / games_per_season
    return out
