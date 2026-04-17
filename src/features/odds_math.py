"""Pure functions for converting sportsbook odds to probabilities.

American odds convention:
    +150  means "bet 100 to win 150"  → implied prob 100/(100+150) = 0.40
    -200  means "bet 200 to win 100"  → implied prob 200/(200+100) = 0.667

Bookmakers include a margin (the "vig" or "juice"), so the two sides of a
two-way market sum to >1.0. De-vigging rescales them to sum to 1 — a fair
probability estimate that strips out the house edge.
"""
from __future__ import annotations


def american_to_prob(odds: float) -> float:
    """Convert American moneyline odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def prob_to_american(p: float) -> float:
    """Inverse of `american_to_prob`. Returns positive or negative per convention."""
    if not 0 < p < 1:
        raise ValueError(f"probability must be in (0, 1), got {p}")
    if p < 0.5:
        return round((100.0 / p) - 100.0, 2)
    return round(-(p / (1 - p)) * 100.0, 2)


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    """Rescale a two-outcome market so probabilities sum to 1.

    This is the simplest de-vig method (proportional). More sophisticated
    methods (power, Shin) exist but only matter at the margins.
    """
    total = p_a + p_b
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return p_a / total, p_b / total


def moneyline_to_fair_prob(ml_home: float, ml_away: float) -> tuple[float, float]:
    """Convert two American moneylines into de-vigged fair probabilities.

    Returns (p_home_fair, p_away_fair), summing to 1.
    """
    return devig_two_way(american_to_prob(ml_home), american_to_prob(ml_away))


def edge(model_prob: float, market_prob: float) -> float:
    """Signed edge: positive means the model is more confident than the market."""
    return model_prob - market_prob


def kelly_fraction(model_prob: float, ml: float) -> float:
    """Full-Kelly stake as a fraction of bankroll for an American moneyline bet.

    Returns 0 if the bet has no edge (never risk money on a no-edge bet).
    Most practitioners use fractional Kelly (e.g. 0.25x) to control variance;
    that scaling is a caller concern.
    """
    if ml > 0:
        b = ml / 100.0
    else:
        b = 100.0 / -ml
    q = 1.0 - model_prob
    f = (b * model_prob - q) / b
    return max(f, 0.0)
