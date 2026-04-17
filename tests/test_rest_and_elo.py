"""Tests for rest_features and elo."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.rest_features import add_rest, add_rest_diff
from src.features.elo import add_elo


def test_rest_first_game_has_nan_and_b2b_zero():
    df = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "season": ["2023-24"] * 3,
            "game_id": ["g1", "g2", "g3"],
            "game_date": pd.to_datetime(["2023-10-25", "2023-10-26", "2023-10-29"]),
        }
    )
    out = add_rest(df).sort_values("game_date").reset_index(drop=True)

    assert pd.isna(out.loc[0, "days_rest"])
    assert out.loc[0, "is_b2b"] == 0
    assert out.loc[1, "days_rest"] == 1
    assert out.loc[1, "is_b2b"] == 1
    assert out.loc[2, "days_rest"] == 3
    assert out.loc[2, "is_b2b"] == 0


def test_rest_does_not_cross_seasons():
    df = pd.DataFrame(
        {
            "team_id": [1, 1],
            "season": ["2022-23", "2023-24"],
            "game_id": ["g1", "g2"],
            "game_date": pd.to_datetime(["2023-04-10", "2023-10-25"]),
        }
    )
    out = add_rest(df)
    assert pd.isna(out.loc[out["season"] == "2023-24", "days_rest"].iloc[0])


def test_rest_diff_sign():
    matchup = pd.DataFrame(
        {
            "game_id": ["g1"],
            "home_days_rest": [3.0],
            "away_days_rest": [1.0],
        }
    )
    out = add_rest_diff(matchup)
    assert out["rest_diff"].iloc[0] == 2.0


def _two_team_matchup(outcomes: list[int], dates: list[str], season: str = "2023-24") -> pd.DataFrame:
    """Build a minimal matchup frame: team 1 (home) vs team 2 (away) in each row."""
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(len(outcomes))],
            "season": [season] * len(outcomes),
            "game_date": pd.to_datetime(dates),
            "home_team_id": [1] * len(outcomes),
            "away_team_id": [2] * len(outcomes),
            "home_won": outcomes,
        }
    )


def test_elo_pre_game_rating_does_not_use_current_result():
    """The first game's pre-rating must equal the initial rating for both teams."""
    matchup = _two_team_matchup([1, 1], ["2023-10-25", "2023-10-27"])
    out = add_elo(matchup)
    assert out.loc[0, "home_elo_pre"] == 1500.0
    assert out.loc[0, "away_elo_pre"] == 1500.0


def test_elo_winner_gains_loser_loses():
    """Home wins g0 → at g1, home's pre-rating is above 1500 and away's below."""
    matchup = _two_team_matchup([1, 0], ["2023-10-25", "2023-10-27"])
    out = add_elo(matchup, k=20.0)
    assert out.loc[1, "home_elo_pre"] > 1500.0
    assert out.loc[1, "away_elo_pre"] < 1500.0


def test_elo_season_regression_pulls_toward_mean():
    """A team well above 1500 at season end should regress toward 1500 at next season."""
    dates = [f"2023-0{i}-15" for i in range(1, 6)] + ["2024-10-25"]
    matchup = _two_team_matchup([1, 1, 1, 1, 1, 0], dates, season="2022-23")
    matchup.loc[5, "season"] = "2023-24"

    out = add_elo(matchup, season_regression=0.25)
    end_of_s1 = out.loc[4, "home_elo_pre"]
    # After g4 (5th win), home rating goes up once more; then between seasons
    # regression: new_rating = old * 0.75 + 1500 * 0.25 → strictly closer to 1500
    start_of_s2 = out.loc[5, "home_elo_pre"]
    # g4's rating reflects 4 wins; g5's reflects 5 wins then regression
    # Just assert regression pulled it strictly toward 1500
    assert abs(start_of_s2 - 1500) < abs(end_of_s1 - 1500) + 1e-6
