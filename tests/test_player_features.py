"""Leakage-safety + correctness tests for player_features.

Same invariant as the rolling test: features for game t must not depend on
data from game t. Player availability is the highest-leakage-risk feature
in the project — if a star DNP shows up in their own game's "rotation
availability" feature, the model trivially learns to use it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.player_features import add_player_features


def _toy_team_games() -> pd.DataFrame:
    """3 games for team 1 in one season."""
    return pd.DataFrame({
        "team_id": [1, 1, 1],
        "season": ["2023-24"] * 3,
        "game_id": ["g1", "g2", "g3"],
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
    })


def _toy_player_games(g3_top_minutes: float) -> pd.DataFrame:
    """3 games × 2 players. Player 100 is unambiguously the star (40+ min vs
    15 min for player 200), so they remain the top-1 even if they miss a game.
    We vary their g3 minutes to test leakage."""
    return pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
        "player_id": [100, 200, 100, 200, 100, 200],
        "team_id": [1, 1, 1, 1, 1, 1],
        "season": ["2023-24"] * 6,
        "game_date": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03",
            "2024-01-05", "2024-01-05",
        ]),
        "minutes": [40.0, 15.0, 40.0, 15.0, g3_top_minutes, 15.0],
    })


def test_first_game_has_zero_features():
    """The first game of a season has no prior data → all features should be 0."""
    out = add_player_features(_toy_team_games(), _toy_player_games(35.0))
    g1 = out[out["game_id"] == "g1"].iloc[0]
    assert g1["top1_active_recent"] == 0.0
    assert g1["rotation_avg_minutes_recent"] == 0.0


def test_top1_active_increases_after_played_games():
    """g3's top1_active_recent should reflect g1+g2 (both fully played) → 1.0."""
    out = add_player_features(_toy_team_games(), _toy_player_games(35.0))
    g3 = out[out["game_id"] == "g3"].iloc[0]
    assert g3["top1_active_recent"] == 1.0


def test_does_not_leak_current_game():
    """The CRITICAL invariant: changing the star's minutes IN g3 must NOT
    change g3's player-availability features (which only see g1+g2)."""
    base = add_player_features(_toy_team_games(), _toy_player_games(35.0))
    perturbed = add_player_features(_toy_team_games(), _toy_player_games(0.0))

    base_g3 = base[base["game_id"] == "g3"].iloc[0]
    pert_g3 = perturbed[perturbed["game_id"] == "g3"].iloc[0]

    for col in (
        "rotation_avg_minutes_recent",
        "top1_active_recent",
        "top3_active_recent",
        "rotation_minutes_share_recent",
    ):
        assert base_g3[col] == pert_g3[col], f"Leakage in column {col}"


def test_star_absence_lowers_top1_active():
    """If the star (still clearly the top-1) misses g2, g3's top1_active_recent
    over the last 2 games should be 0.5 (played g1, missed g2)."""
    pg = _toy_player_games(40.0)
    # Make the star DNP in g2 — but they're still the top by total minutes (40 > 30).
    pg.loc[(pg["game_id"] == "g2") & (pg["player_id"] == 100), "minutes"] = 0.0

    out = add_player_features(_toy_team_games(), pg)
    g3 = out[out["game_id"] == "g3"].iloc[0]
    assert g3["top1_active_recent"] == 0.5
