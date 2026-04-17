"""Tests for odds math and market comparison.

Worked examples:
  -110 / -110  → 0.524 / 0.524, vig = 4.8% → de-vigged to 0.5 / 0.5
  +150 / -180  → 0.400 / 0.643, total 1.043 → de-vigged to ~0.384 / 0.616
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features.odds_math import (
    american_to_prob,
    devig_two_way,
    edge,
    kelly_fraction,
    moneyline_to_fair_prob,
    prob_to_american,
)
from src.ingest.team_map import resolve_id
from src.pipeline.market_compare import add_market_prob, compare, consensus


def test_american_to_prob_positive():
    assert american_to_prob(+150) == pytest.approx(0.4, abs=1e-6)


def test_american_to_prob_negative():
    assert american_to_prob(-200) == pytest.approx(2 / 3, abs=1e-6)


def test_american_to_prob_even_money():
    assert american_to_prob(+100) == pytest.approx(0.5, abs=1e-6)


def test_prob_to_american_roundtrip():
    for ml in (-250.0, -110.0, +120.0, +300.0):
        assert prob_to_american(american_to_prob(ml)) == pytest.approx(ml, abs=0.1)


def test_devig_sums_to_one():
    a, b = devig_two_way(0.55, 0.50)
    assert a + b == pytest.approx(1.0)
    assert a > b


def test_moneyline_to_fair_prob_symmetric():
    p_home, p_away = moneyline_to_fair_prob(-110, -110)
    assert p_home == pytest.approx(0.5, abs=1e-6)
    assert p_away == pytest.approx(0.5, abs=1e-6)


def test_kelly_zero_when_no_edge():
    # market at -110 implies 52.4%; model says exactly 52.4% → no edge → stake 0
    assert kelly_fraction(0.524, -110) < 0.01


def test_kelly_positive_when_edge_exists():
    # model gives 60% to a +100 line (market says 50%)
    f = kelly_fraction(0.60, +100)
    assert f == pytest.approx(0.20, abs=1e-6)  # (1 * 0.60 - 0.40) / 1


def test_edge_sign():
    assert edge(0.65, 0.55) == pytest.approx(0.10)
    assert edge(0.45, 0.55) == pytest.approx(-0.10)


def test_team_map_resolves_canonical_names():
    assert resolve_id("Boston Celtics") is not None
    assert resolve_id("Los Angeles Lakers") is not None


def test_team_map_resolves_aliases():
    # odds feeds sometimes spell LA Clippers without the comma/city prefix
    assert resolve_id("LA Clippers") == resolve_id("Los Angeles Clippers")


def test_team_map_unknown_returns_none():
    assert resolve_id("Fake Team") is None
    assert resolve_id("") is None


def _toy_odds_frame() -> pd.DataFrame:
    """Two bookmakers, one game. One book has better home price than the other."""
    return pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-01-15", "2026-01-15"]).date,
            "home_team_id": [1610612738, 1610612738],  # BOS
            "away_team_id": [1610612747, 1610612747],  # LAL
            "bookmaker": ["draftkings", "fanduel"],
            "ml_home": [-180, -175],
            "ml_away": [+155, +150],
        }
    )


def test_add_market_prob_devigs_correctly():
    o = add_market_prob(_toy_odds_frame())
    assert (o["market_prob_home"] + o["market_prob_away"]).round(6).eq(1.0).all()
    assert (o["vig"] > 0).all()  # books always charge vig


def test_compare_computes_edge_per_bookmaker():
    preds = pd.DataFrame(
        {
            "game_id": ["0022400100"],
            "game_date": pd.to_datetime(["2026-01-15"]).date,
            "home_team_id": [1610612738],
            "away_team_id": [1610612747],
            "model_prob": [0.70],
        }
    )
    out = compare(preds, _toy_odds_frame())
    assert len(out) == 2  # one row per bookmaker
    assert (out["edge"] > 0).all()  # model at 0.70 vs. market ~0.63 → positive edge
    assert "kelly_full" in out.columns


def test_consensus_averages_across_books():
    preds = pd.DataFrame(
        {
            "game_id": ["0022400100"],
            "game_date": pd.to_datetime(["2026-01-15"]).date,
            "home_team_id": [1610612738],
            "away_team_id": [1610612747],
            "model_prob": [0.70],
        }
    )
    out = consensus(compare(preds, _toy_odds_frame()))
    assert len(out) == 1
    assert out["n_books"].iloc[0] == 2
    assert not math.isnan(out["edge"].iloc[0])
