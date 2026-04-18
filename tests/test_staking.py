"""Tests for staking strategies.

The invariants: no-edge → no bet; fractional Kelly is a true scaling of
full Kelly; capped Kelly respects the ceiling; the factory round-trips.
"""
from __future__ import annotations

import pytest

from src.features.odds_math import kelly_fraction
from src.finance.staking import (
    CappedKelly,
    FixedFractional,
    FlatStake,
    FractionalKelly,
    FullKelly,
    ThresholdKelly,
    build,
)


def _call(strat, *, p=0.60, ml=100, bankroll=10_000, edge=0.05):
    return strat.size(model_prob=p, ml=ml, bankroll=bankroll, edge=edge)


def test_full_kelly_matches_odds_math_reference():
    dec = _call(FullKelly(), p=0.60, ml=100)
    assert dec.fraction == pytest.approx(kelly_fraction(0.60, 100))
    assert dec.kelly_full == pytest.approx(dec.fraction)


def test_fractional_kelly_is_multiplier_times_full():
    full = _call(FullKelly(), p=0.60, ml=100).fraction
    frac = _call(FractionalKelly(multiplier=0.25), p=0.60, ml=100).fraction
    assert frac == pytest.approx(0.25 * full)
    frac_half = _call(FractionalKelly(multiplier=0.5), p=0.60, ml=100).fraction
    assert frac_half == pytest.approx(0.5 * full)


def test_no_edge_means_zero_kelly():
    # American +100 implies 50% fair. p=0.5 → full Kelly = 0 → fractional also 0.
    assert _call(FullKelly(), p=0.50, ml=100).fraction == 0.0
    assert _call(FractionalKelly(0.25), p=0.50, ml=100).fraction == 0.0
    assert _call(CappedKelly(), p=0.50, ml=100).fraction == 0.0


def test_negative_edge_returns_zero_from_full_kelly():
    assert _call(FullKelly(), p=0.40, ml=100).fraction == 0.0


def test_threshold_kelly_skips_below_threshold():
    strat = ThresholdKelly(min_edge=0.05, multiplier=0.25)
    dec_low = _call(strat, p=0.60, ml=100, edge=0.01)
    assert dec_low.fraction == 0.0
    assert "below_threshold" in dec_low.rationale
    dec_high = _call(strat, p=0.60, ml=100, edge=0.10)
    assert dec_high.fraction > 0


def test_capped_kelly_respects_ceiling():
    strat = CappedKelly(multiplier=1.0, max_fraction=0.05)
    # Construct a high-Kelly scenario: big edge on +odds.
    dec = _call(strat, p=0.65, ml=150)
    assert dec.fraction <= 0.05 + 1e-12
    assert "capped" in dec.rationale


def test_flat_stake_is_dollar_invariant():
    strat = FlatStake(unit=100.0)
    assert _call(strat, bankroll=10_000).fraction == pytest.approx(0.01)
    assert _call(strat, bankroll=5_000).fraction == pytest.approx(0.02)


def test_flat_stake_caps_at_bankroll():
    strat = FlatStake(unit=10_000.0)
    dec = _call(strat, bankroll=5_000.0)
    assert dec.fraction == 1.0


def test_fixed_fractional_rejects_invalid_pct():
    with pytest.raises(ValueError):
        FixedFractional(pct=0.0)
    with pytest.raises(ValueError):
        FixedFractional(pct=1.0)


def test_fixed_fractional_ignores_edge():
    strat = FixedFractional(pct=0.02)
    dec_small = _call(strat, edge=0.001)
    dec_big = _call(strat, edge=0.20)
    # FixedFractional doesn't gate on edge — that's ThresholdKelly's job.
    assert dec_small.fraction == dec_big.fraction == 0.02


def test_factory_builds_every_registered_strategy():
    assert isinstance(build({"type": "flat", "unit": 50.0}), FlatStake)
    assert isinstance(build({"type": "fixed_fractional", "pct": 0.01}), FixedFractional)
    assert isinstance(build({"type": "full_kelly"}), FullKelly)
    assert isinstance(build({"type": "fractional_kelly", "multiplier": 0.25}), FractionalKelly)
    assert isinstance(build({"type": "threshold_kelly", "min_edge": 0.02}), ThresholdKelly)
    assert isinstance(build({"type": "capped_kelly", "multiplier": 0.25, "max_fraction": 0.05}), CappedKelly)


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError):
        build({"type": "martingale"})
