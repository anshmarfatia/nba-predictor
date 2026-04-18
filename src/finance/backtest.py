"""Backtest engine: replay historical games with a staking strategy and
record P&L as bets resolve.

## The discipline

**No look-ahead.** Predictions come from walk-forward (model trained only on
prior seasons). An assertion in the day loop guarantees `predicted_at <=
game_date` even if a caller builds predictions some other way.

**Start-of-day bankroll for sizing.** Within a single game day, every bet is
sized off the *same* bankroll — the one the day opened with. This is what
makes the simultaneous-Kelly cap meaningful; sequential compounding within a
day would mask the concurrent-exposure problem.

**Concurrent-bet normalization.** Textbook Kelly assumes outcomes of bet i
are known before bet i+1. NBA reality: six games tip off in one 7–9 PM
window. Raw Kelly fractions across concurrent bets can easily sum to
> 100 % of bankroll → ruin on a bad night. When Σfᵢ > `max_concurrent_exposure`
(default 0.50), all stakes are scaled down proportionally so they sum to
the cap. Stored `kelly_fraction_used` is the *post-scaling* fraction.

**Single-bookmaker data.** If only one bookmaker is present for a day's
odds (typical for Kaggle closes), `consensus()` is skipped — no point
averaging one number. The single book's line is used directly.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

from src.features.odds_math import (
    american_to_prob,
    devig_two_way,
    kelly_fraction,
)
from src.finance import metrics as _metrics
from src.finance.bet_log import (
    STATUS_LOST,
    STATUS_OPEN,
    STATUS_WON,
    realize_pnl,
)
from src.finance.staking import StakeDecision, StakingStrategy
from src.pipeline.market_compare import consensus as _consensus

log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Config + Result
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    strategy: StakingStrategy
    starting_bankroll: float = 10_000.0
    min_edge: float = 0.02
    side: Literal["home", "away", "best"] = "best"
    bookmaker: str | None = None                 # None → consensus (skipped if single-book)
    rebalance: Literal["compound", "fixed_unit"] = "compound"
    max_concurrent_exposure: float = 0.5
    model_version: str = "v3"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class BacktestResult:
    bets: pd.DataFrame                 # one row per placed bet
    equity: pd.Series                  # sparse (bet-day) bankroll index
    summary: dict

    def summary_pretty(self) -> str:
        s = self.summary
        lines = [
            f"n_bets={s['n_bets']:<4d}  n_bet_days={s['n_bet_days']:<4d}",
            f"start={s['starting_bankroll']:.0f}  end={s['ending_bankroll']:.0f}  pnl={s['total_pnl']:+.0f}",
            f"ROI={s['roi']:+.2%}  hit={s['hit_rate']:.2%}  avg_edge={s['avg_edge']:+.3f}",
            f"sharpe={s['sharpe']:.2f}  sortino={s['sortino']:.2f}  "
            f"calmar={s['calmar']:.2f}  MDD={s['max_drawdown']:.2%}",
        ]
        return "\n  ".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _side_payload(
    model_prob_home: float,
    ml_home: float,
    ml_away: float,
    preferred_side: str,
) -> tuple[str, float, float, float]:
    """Pick the side to bet and compute (side, model_prob_side, ml_side, edge_side).

    With a de-vigged two-way market, exactly one side has positive edge
    (unless edge is zero for both). `preferred_side='best'` selects the
    positive-edge side, or the max-edge side if neither is positive.
    """
    p_home_raw = american_to_prob(ml_home)
    p_away_raw = american_to_prob(ml_away)
    p_home_fair, p_away_fair = devig_two_way(p_home_raw, p_away_raw)
    edge_home = model_prob_home - p_home_fair
    edge_away = (1.0 - model_prob_home) - p_away_fair

    if preferred_side == "home":
        return "home", model_prob_home, ml_home, edge_home
    if preferred_side == "away":
        return "away", 1.0 - model_prob_home, ml_away, edge_away
    # "best"
    if edge_home >= edge_away:
        return "home", model_prob_home, ml_home, edge_home
    return "away", 1.0 - model_prob_home, ml_away, edge_away


def _collapse_to_daily_odds(odds_today: pd.DataFrame, bookmaker: str | None) -> pd.DataFrame:
    """One row per (game_date, home_team_id, away_team_id). If a bookmaker is
    specified, filter to it; otherwise collapse across books (consensus mean
    if >1, passthrough if exactly 1)."""
    if odds_today.empty:
        return odds_today
    if bookmaker is not None:
        return odds_today[odds_today["bookmaker"] == bookmaker].copy()
    unique_books = odds_today["bookmaker"].nunique()
    if unique_books <= 1:
        return odds_today.copy()
    # Consensus = mean moneyline probability across books, then back to ml.
    # We only need ml_home/ml_away for downstream edge math — keep it simple.
    grp = odds_today.groupby(
        ["game_date", "home_team_id", "away_team_id"], as_index=False
    ).agg(ml_home=("ml_home", "mean"), ml_away=("ml_away", "mean"))
    grp["bookmaker"] = "consensus"
    return grp


def normalize_concurrent(
    fractions: list[float], cap: float
) -> list[float]:
    """Scale a list of suggested stake fractions so their sum ≤ cap.

    Proportional haircut — preserves relative sizing. Idempotent when Σ ≤ cap.
    """
    total = sum(fractions)
    if total <= cap or total == 0:
        return list(fractions)
    scale = cap / total
    return [f * scale for f in fractions]


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def run_backtest(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    outcomes: pd.DataFrame,
    cfg: BacktestConfig,
) -> BacktestResult:
    """Replay games in chronological order and return realized P&L.

    Expected columns:
      predictions: game_id, game_date, home_team_id, away_team_id, model_prob,
                   predicted_at (optional — checked for look-ahead if present)
      odds:        game_date, home_team_id, away_team_id, bookmaker, ml_home, ml_away
      outcomes:    game_id, home_won   (0/1)
    """
    _validate_predictions(predictions)
    outcomes_by_id = outcomes.set_index("game_id")["home_won"].to_dict()

    bankroll = float(cfg.starting_bankroll)
    bet_rows: list[dict] = []
    equity: dict[date, float] = {}

    game_dates = sorted(predictions["game_date"].unique())
    for d in game_dates:
        d_date = pd.Timestamp(d).date()
        day_preds = predictions[predictions["game_date"] == d]
        day_odds = odds[odds["game_date"] == d]
        if day_odds.empty:
            continue

        day_odds = _collapse_to_daily_odds(day_odds, cfg.bookmaker)
        joined = day_preds.merge(
            day_odds, on=["game_date", "home_team_id", "away_team_id"], how="inner"
        )
        if joined.empty:
            continue

        # Size every candidate bet off the *same* start-of-day bankroll.
        candidates = _size_candidates(joined, bankroll, cfg)
        if not candidates:
            continue

        # Simultaneous-Kelly cap applied proportionally across the day's bets.
        raw_fractions = [c["kelly_fraction_suggested"] for c in candidates]
        final_fractions = normalize_concurrent(raw_fractions, cfg.max_concurrent_exposure)

        day_pnl = 0.0
        for cand, f_final in zip(candidates, final_fractions):
            stake = f_final * bankroll
            if stake <= 0.0:
                continue
            won = bool(
                outcomes_by_id.get(cand["game_id"], None)
                == (1 if cand["side"] == "home" else 0)
            )
            payout = realize_pnl(stake, cand["entry_ml"], won)
            day_pnl += payout
            bet_rows.append({
                "placed_at": datetime.combine(d_date, datetime.min.time(), tzinfo=timezone.utc),
                "game_date": d_date,
                "game_id": cand["game_id"],
                "model_version": cfg.model_version,
                "bookmaker": cand["bookmaker"],
                "side": cand["side"],
                "entry_ml": cand["entry_ml"],
                "entry_market_prob": cand["entry_market_prob"],
                "model_prob": cand["model_prob_side"],
                "edge": cand["edge"],
                "stake": stake,
                "kelly_full": cand["kelly_full"],
                "kelly_fraction_used": f_final,
                "strategy": cfg.strategy.name,
                "bankroll_before": bankroll,
                "backtest_run_id": cfg.run_id,
                "status": STATUS_WON if won else STATUS_LOST,
                "payout": payout,
                "settled_at": datetime.combine(
                    d_date, datetime.max.time().replace(microsecond=0), tzinfo=timezone.utc
                ),
            })

        bankroll += day_pnl
        equity[d_date] = bankroll
        if bankroll <= 0:
            log.warning("Bankroll wiped on %s. Halting backtest.", d_date)
            break

    bets_df = pd.DataFrame(bet_rows)
    equity_s = pd.Series(equity, name="bankroll").sort_index()
    if equity_s.empty:
        equity_s = pd.Series(
            [cfg.starting_bankroll], index=[pd.Timestamp(game_dates[0]).date() if game_dates else date.today()],
            name="bankroll",
        )
    else:
        # Prepend starting bankroll one day before the first bet day so
        # equity-based metrics have a baseline to compare against.
        first = equity_s.index[0]
        equity_s = pd.concat([
            pd.Series([cfg.starting_bankroll], index=[first - pd.Timedelta(days=1)]),
            equity_s,
        ]).sort_index()

    return BacktestResult(
        bets=bets_df,
        equity=equity_s,
        summary=_metrics.summary(bets_df, equity_s),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _validate_predictions(predictions: pd.DataFrame) -> None:
    """Check no-look-ahead when `predicted_at` is present."""
    required = {"game_id", "game_date", "home_team_id", "away_team_id", "model_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    if "predicted_at" in predictions.columns:
        preds_date = pd.to_datetime(predictions["predicted_at"]).dt.date
        game_date = pd.to_datetime(predictions["game_date"]).dt.date
        bad = (preds_date > game_date).sum()
        if bad:
            raise AssertionError(
                f"Look-ahead detected in {bad} predictions (predicted_at > game_date). "
                "Use walk-forward predictions only."
            )


def _size_candidates(
    joined: pd.DataFrame, bankroll: float, cfg: BacktestConfig
) -> list[dict]:
    """Turn a day's joined rows into bet candidates with suggested fractions."""
    candidates: list[dict] = []
    for row in joined.itertuples(index=False):
        side, p_side, ml_side, edge_side = _side_payload(
            row.model_prob, row.ml_home, row.ml_away, cfg.side
        )
        if edge_side < cfg.min_edge:
            continue
        decision: StakeDecision = cfg.strategy.size(
            model_prob=p_side, ml=ml_side, bankroll=bankroll, edge=edge_side
        )
        if decision.fraction <= 0:
            continue
        p_home_raw = american_to_prob(row.ml_home)
        p_away_raw = american_to_prob(row.ml_away)
        p_home_fair, p_away_fair = devig_two_way(p_home_raw, p_away_raw)
        entry_market_prob = p_home_fair if side == "home" else p_away_fair
        candidates.append({
            "game_id": row.game_id,
            "bookmaker": getattr(row, "bookmaker", "consensus"),
            "side": side,
            "entry_ml": float(ml_side),
            "entry_market_prob": float(entry_market_prob),
            "model_prob_side": float(p_side),
            "edge": float(edge_side),
            "kelly_full": float(decision.kelly_full),
            "kelly_fraction_suggested": float(decision.fraction),
        })
    return candidates
