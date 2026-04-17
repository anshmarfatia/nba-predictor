"""Roster availability features at the team-game grain.

The hypothesis: when a team's stars are missing, the team is significantly
weaker than its rolling team-level stats suggest. The team's last 10-game
NetRtg may include games where LeBron played; tonight he's out. A model
that doesn't see availability is blind to this.

Concretely, for each (team, game) we compute four features from the
PRIOR-game player log (everything `< game_date`) — the leakage-safe slice:

  - `rotation_avg_minutes_recent`: mean per-game minutes for the team's
    rotation (top-N by minutes) over the last `lookback` games. Drops when
    rotation players miss games — a smooth proxy for "team has been
    shorthanded recently."
  - `top1_active_recent`: fraction of the last `lookback` games in which
    the team's top player (by `lookback`-window minutes) actually played
    (minutes > 0). 1.0 = healthy, 0.0 = out for the entire window.
  - `top3_active_recent`: same but averaged across the top-3 rotation.
  - `rotation_minutes_share_recent`: fraction of total team minutes
    consumed by the rotation in the last `lookback` games. Stable at ~0.85
    when healthy; drops when starters are out and bench is filling minutes.

All four are computed from data strictly before each game, so they are
serving-safe: at prediction time we use the most recent `lookback` games
to project the same features.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_LOOKBACK = 10
DEFAULT_TOP_N = 8


def _team_rotation(player_window: pd.DataFrame, top_n: int) -> list[int]:
    """Identify a team's rotation as the top-N players by total minutes
    in the supplied window. Returns player_ids ordered most → least minutes."""
    if player_window.empty:
        return []
    totals = player_window.groupby("player_id")["minutes"].sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()


def _features_for_window(
    player_window: pd.DataFrame, top_n: int
) -> dict[str, float]:
    """Compute the 4 player-availability features for one (team, lookback)
    window. `player_window` must be the team's player_games rows over the
    chosen lookback period — strictly prior to the game being scored."""
    if player_window.empty:
        return {
            "rotation_avg_minutes_recent": 0.0,
            "top1_active_recent": 0.0,
            "top3_active_recent": 0.0,
            "rotation_minutes_share_recent": 0.0,
        }

    rotation = _team_rotation(player_window, top_n)
    rot_set = set(rotation)
    top3_set = set(rotation[:3])
    top1 = rotation[0] if rotation else None

    n_games = player_window["game_id"].nunique()
    if n_games == 0:
        return {
            "rotation_avg_minutes_recent": 0.0,
            "top1_active_recent": 0.0,
            "top3_active_recent": 0.0,
            "rotation_minutes_share_recent": 0.0,
        }

    rot_minutes_total = player_window.loc[
        player_window["player_id"].isin(rot_set), "minutes"
    ].sum()
    team_minutes_total = player_window["minutes"].sum()

    games_top1_active = (
        player_window[
            (player_window["player_id"] == top1) & (player_window["minutes"] > 0)
        ]["game_id"].nunique()
        if top1 is not None
        else 0
    )
    games_top3_active_count = (
        player_window[
            player_window["player_id"].isin(top3_set) & (player_window["minutes"] > 0)
        ]
        .groupby("game_id")["player_id"]
        .nunique()
        .sum()
    )

    return {
        "rotation_avg_minutes_recent": rot_minutes_total / max(len(rot_set), 1) / n_games,
        "top1_active_recent": games_top1_active / n_games,
        "top3_active_recent": games_top3_active_count / (3 * n_games),
        "rotation_minutes_share_recent": (
            rot_minutes_total / team_minutes_total if team_minutes_total > 0 else 0.0
        ),
    }


def add_player_features(
    team_games: pd.DataFrame,
    player_games: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Append player-availability columns to a per-team-game DataFrame.

    `team_games` must include `team_id`, `season`, `game_date`, `game_id`.
    `player_games` is the rows from the `player_games` table.

    The walk is per-(team, season): for each team-game, we slice the
    player_games table to the previous `lookback` games for that team, then
    compute the four features. Players' minutes from the CURRENT game are
    never included — that's the entire leakage-safety contract.
    """
    if team_games.empty:
        return team_games.copy()

    out = team_games.sort_values(["team_id", "game_date", "game_id"]).copy()
    pg = player_games.copy()
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    out["game_date"] = pd.to_datetime(out["game_date"])

    pg_by_team = {tid: g.sort_values("game_date") for tid, g in pg.groupby("team_id")}

    new_cols: dict[str, list[float]] = {
        "rotation_avg_minutes_recent": [],
        "top1_active_recent": [],
        "top3_active_recent": [],
        "rotation_minutes_share_recent": [],
    }

    for _, row in out.iterrows():
        tid = int(row["team_id"])
        season = row["season"]
        cutoff = row["game_date"]

        team_pg = pg_by_team.get(tid, pd.DataFrame())
        if team_pg.empty:
            for k in new_cols:
                new_cols[k].append(0.0)
            continue

        prior = team_pg[
            (team_pg["season"] == season) & (team_pg["game_date"] < cutoff)
        ]
        if prior.empty:
            for k in new_cols:
                new_cols[k].append(0.0)
            continue

        recent_game_ids = (
            prior.drop_duplicates("game_id")
            .sort_values("game_date")
            .tail(lookback)["game_id"]
            .tolist()
        )
        window = prior[prior["game_id"].isin(recent_game_ids)]
        feats = _features_for_window(window, top_n)
        for k, v in feats.items():
            new_cols[k].append(v)

    for k, vals in new_cols.items():
        out[k] = vals
    return out
