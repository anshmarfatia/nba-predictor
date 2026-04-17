"""Leakage-safety tests for feature builders.

The single most important invariant in this project: a feature for game t
must not use any data from game t. These tests enforce that for the rolling
feature builder.
"""
from __future__ import annotations

import pandas as pd

from src.features.rolling_stats import add_rolling


def _toy_df(values_team_a: list[float]) -> pd.DataFrame:
    """3 games for team A, 1 decoy for team B, all in one season."""
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1, 2],
            "season": ["2023-24"] * 4,
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-02"]
            ),
            "net_rating": values_team_a + [99.0],
        }
    )


def test_rolling_first_game_is_nan():
    """The first game of a (team, season) group has no prior data → NaN."""
    out = add_rolling(_toy_df([10.0, 20.0, 30.0]), cols=("net_rating",), windows=(2,))
    a = out[out["team_id"] == 1].sort_values("game_date")
    assert pd.isna(a["net_rating_r2"].iloc[0])


def test_rolling_uses_only_prior_games():
    """Rolling mean at game t must equal mean of games [t-w, t-1]."""
    out = add_rolling(_toy_df([10.0, 20.0, 30.0]), cols=("net_rating",), windows=(2,))
    a = out[out["team_id"] == 1].sort_values("game_date")
    feat = a["net_rating_r2"].tolist()

    assert feat[1] == 10.0
    assert feat[2] == (10.0 + 20.0) / 2


def test_rolling_does_not_leak_current_game():
    """Changing game t's raw value must NOT change game t's rolling feature.

    This is the core leakage invariant. If this fails, the entire model is
    compromised — the feature would "see" the game it is predicting.
    """
    base = add_rolling(_toy_df([10.0, 20.0, 30.0]), cols=("net_rating",), windows=(2,))
    perturbed = add_rolling(
        _toy_df([10.0, 20.0, 999.0]), cols=("net_rating",), windows=(2,)
    )

    base_a = base[base["team_id"] == 1].sort_values("game_date")
    pert_a = perturbed[perturbed["team_id"] == 1].sort_values("game_date")

    assert base_a["net_rating_r2"].iloc[2] == pert_a["net_rating_r2"].iloc[2]


def test_rolling_does_not_cross_seasons():
    """A team's first game of a new season should not inherit last season's window."""
    df = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1],
            "season": ["2022-23", "2022-23", "2023-24", "2023-24"],
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_date": pd.to_datetime(
                ["2023-04-01", "2023-04-03", "2023-10-25", "2023-10-27"]
            ),
            "net_rating": [10.0, 20.0, 100.0, 200.0],
        }
    )
    out = add_rolling(df, cols=("net_rating",), windows=(5,))
    g3_feat = out.loc[out["game_id"] == "g3", "net_rating_r5"].iloc[0]
    assert pd.isna(g3_feat)
