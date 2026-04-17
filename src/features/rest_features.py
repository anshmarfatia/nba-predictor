"""Rest-related features.

`add_rest` operates on per-team-game rows (pre-pivot) and adds each team's
`days_rest` (days since their previous game in the same season) and a
`is_b2b` flag (rest == 1). The first game of a (team, season) has NaN rest.

`add_rest_diff` runs after the matchup pivot and adds `rest_diff`
(home_days_rest - away_days_rest), which the outline calls out as the
single most predictive form of the rest signal.
"""
from __future__ import annotations

import pandas as pd


def add_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add `days_rest` and `is_b2b` to a per-team-game DataFrame."""
    if df.empty:
        return df.copy()

    out = df.sort_values(["team_id", "game_date", "game_id"]).copy()
    out["days_rest"] = (
        out.groupby(["team_id", "season"])["game_date"].diff().dt.days
    )
    out["is_b2b"] = (out["days_rest"] == 1).astype("int8")
    return out


def add_rest_diff(matchup: pd.DataFrame) -> pd.DataFrame:
    """Add `rest_diff = home_days_rest - away_days_rest` to a matchup frame."""
    if matchup.empty:
        return matchup.copy()
    out = matchup.copy()
    out["rest_diff"] = out["home_days_rest"] - out["away_days_rest"]
    return out
