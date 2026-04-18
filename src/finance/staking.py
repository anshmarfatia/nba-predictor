"""Staking strategies: how much of the bankroll to risk per bet.

Strategies are `StakingStrategy` Protocol implementers — structural typing
lets the backtest accept ad-hoc lambdas and the concrete classes below
without an ABC. A `build(cfg)` factory maps config dicts to strategies for
dashboard / CLI boundaries.

Five concrete variants, all reusing `src.features.odds_math.kelly_fraction`:

    FlatStake(unit)               — fixed dollar stake (fractional of bankroll varies)
    FixedFractional(pct)          — fixed % of bankroll every bet
    FullKelly()                   — raw Kelly
    FractionalKelly(multiplier)   — Kelly × multiplier (default 0.25 — the project default)
    ThresholdKelly(min_edge, mul) — skip bets below min_edge, else fractional Kelly
    CappedKelly(mul, max_frac)    — fractional Kelly with a hard per-bet ceiling

None of these handle the *simultaneous-bet* constraint — that's the
backtest's job (see `src.finance.backtest.normalize_concurrent`). A
strategy reports what it would want to bet; the engine enforces portfolio
limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.features.odds_math import kelly_fraction


@dataclass(frozen=True)
class StakeDecision:
    """What the strategy proposes for a single bet, before any concurrent-bet haircut."""
    fraction: float        # fraction of bankroll the strategy wants to risk
    rationale: str         # short tag (e.g. 'kelly_0.25', 'below_threshold', 'flat')
    kelly_full: float      # raw full-Kelly fraction for auditing


class StakingStrategy(Protocol):
    name: str

    def size(
        self, *, model_prob: float, ml: float, bankroll: float, edge: float
    ) -> StakeDecision: ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlatStake:
    """Risk a fixed dollar amount per bet, independent of edge or bankroll.

    Fraction is `unit / bankroll` so it auto-adjusts as the bankroll evolves;
    the *dollar* stake is the invariant. Useful as a naive baseline.
    """
    unit: float
    name: str = "flat"

    def size(self, *, model_prob: float, ml: float, bankroll: float, edge: float) -> StakeDecision:
        k_full = kelly_fraction(model_prob, ml)
        if bankroll <= 0:
            return StakeDecision(fraction=0.0, rationale="broke", kelly_full=k_full)
        f = min(1.0, self.unit / bankroll)
        return StakeDecision(fraction=f, rationale="flat", kelly_full=k_full)


@dataclass(frozen=True)
class FixedFractional:
    """Always risk pct% of current bankroll — compounding, but not edge-aware."""
    pct: float
    name: str = "fixed_fractional"

    def __post_init__(self):
        if not 0 < self.pct < 1:
            raise ValueError(f"pct must be in (0, 1), got {self.pct}")

    def size(self, *, model_prob: float, ml: float, bankroll: float, edge: float) -> StakeDecision:
        k_full = kelly_fraction(model_prob, ml)
        return StakeDecision(fraction=self.pct, rationale=f"fixed_{self.pct:.3f}", kelly_full=k_full)


@dataclass(frozen=True)
class FullKelly:
    """Raw Kelly. Mathematically optimal under known p; aggressive in practice."""
    name: str = "full_kelly"

    def size(self, *, model_prob: float, ml: float, bankroll: float, edge: float) -> StakeDecision:
        k = kelly_fraction(model_prob, ml)
        return StakeDecision(fraction=k, rationale="kelly_1.00", kelly_full=k)


@dataclass(frozen=True)
class FractionalKelly:
    """Kelly × multiplier. Standard practitioner fix for parameter uncertainty.

    Thorp / MacLean-Ziemba argue 0.25x–0.5x captures ~90%+ of Kelly growth
    with a small fraction of its drawdown. 0.25 is the project default.
    """
    multiplier: float = 0.25
    name: str = "fractional_kelly"

    def __post_init__(self):
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {self.multiplier}")

    def size(self, *, model_prob: float, ml: float, bankroll: float, edge: float) -> StakeDecision:
        k = kelly_fraction(model_prob, ml)
        return StakeDecision(
            fraction=self.multiplier * k,
            rationale=f"kelly_{self.multiplier:.2f}",
            kelly_full=k,
        )


@dataclass(frozen=True)
class ThresholdKelly:
    """Fractional Kelly, but skip bets whose edge is below `min_edge`.

    Edge filtering is really a *gate*, not a stake-sizer — but it's
    convenient to package with the sizing policy since they always travel
    together in a deployed strategy.
    """
    min_edge: float = 0.02
    multiplier: float = 0.25
    name: str = "threshold_kelly"

    def size(self, *, model_prob: float, ml: float, bankroll: float, edge: float) -> StakeDecision:
        k = kelly_fraction(model_prob, ml)
        if edge < self.min_edge:
            return StakeDecision(fraction=0.0, rationale="below_threshold", kelly_full=k)
        return StakeDecision(
            fraction=self.multiplier * k,
            rationale=f"kelly_{self.multiplier:.2f}_gated_{self.min_edge:.2f}",
            kelly_full=k,
        )


@dataclass(frozen=True)
class CappedKelly:
    """Fractional Kelly with a hard per-bet ceiling.

    Even 0.25x Kelly can recommend a 15%+ stake on a big-edge +dog bet;
    institutional bankroll policy usually caps that. Max fraction 5% is a
    sensible default.
    """
    multiplier: float = 0.25
    max_fraction: float = 0.05
    name: str = "capped_kelly"

    def size(self, *, model_prob: float, ml: float, bankroll: float, edge: float) -> StakeDecision:
        k = kelly_fraction(model_prob, ml)
        proposed = self.multiplier * k
        capped = min(proposed, self.max_fraction)
        rationale = (
            f"kelly_{self.multiplier:.2f}_capped" if capped < proposed else f"kelly_{self.multiplier:.2f}"
        )
        return StakeDecision(fraction=capped, rationale=rationale, kelly_full=k)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "flat": FlatStake,
    "fixed_fractional": FixedFractional,
    "full_kelly": FullKelly,
    "fractional_kelly": FractionalKelly,
    "threshold_kelly": ThresholdKelly,
    "capped_kelly": CappedKelly,
}


def build(cfg: dict) -> StakingStrategy:
    """Construct a strategy from a config dict.

    Examples:
        build({"type": "fractional_kelly", "multiplier": 0.25})
        build({"type": "threshold_kelly", "min_edge": 0.03, "multiplier": 0.25})
        build({"type": "flat", "unit": 100.0})
    """
    cfg = dict(cfg)
    kind = cfg.pop("type", None)
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown strategy type {kind!r}. Options: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**cfg)
