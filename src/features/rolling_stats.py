"""Leakage-safe rolling team-stat features.

Operates on per-team-game rows (the natural grain of `team_games`).
For each (team_id, season), computes shifted rolling means so the feature
at game t uses ONLY data from games [t-w, t-1].

Using `.shift(1)` before `.rolling()` is the critical discipline that the
project outline calls out as the #1 leakage risk in sports prediction.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_COLS: tuple[str, ...] = (
    "net_rating",
    "off_rating",
    "def_rating",
    "pace",
    "ts_pct",
    "efg_pct",
    "pts",
    "plus_minus",
)
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20)


def add_rolling(
    df: pd.DataFrame,
    cols: tuple[str, ...] = DEFAULT_COLS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    group_keys: tuple[str, ...] = ("team_id", "season"),
) -> pd.DataFrame:
    """Append shifted rolling-mean columns to `df`.

    For each (col, w) pair, adds `{col}_r{w}` = mean of the previous `w`
    games for that team within the same season. The first game of a
    (team, season) group has no prior data, so its rolling feature is NaN.

    Uses `min_periods=1` so partially-filled windows still produce a value
    (e.g. the 2nd game of the season yields a 1-game mean for `_r5`).
    Downstream code can filter on `game_number` if strict windows are needed.
    """
    if df.empty:
        return df.copy()

    out = df.sort_values(["team_id", "game_date", "game_id"]).copy()
    grouped = out.groupby(list(group_keys), sort=False)

    for col in cols:
        if col not in out.columns:
            continue
        for w in windows:
            out[f"{col}_r{w}"] = grouped[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return out
