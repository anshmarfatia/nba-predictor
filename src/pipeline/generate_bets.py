"""CLI driver for the backtest engine.

Glues together:
  1. walk-forward predictions (leak-free per-game model probabilities)
  2. odds_snapshots (historical closes)
  3. outcomes from team_games
  4. staking strategy + config → run_backtest → persist to bet_log

Usage:
    # Create the bet_log table once.
    python -m src.pipeline.generate_bets --create-tables

    # Default run: 0.25x Kelly, 2% min edge, full date range in DB.
    python -m src.pipeline.generate_bets

    # Strategy comparison.
    python -m src.pipeline.generate_bets --strategy flat --unit 100
    python -m src.pipeline.generate_bets --strategy fractional_kelly --multiplier 0.25 --min-edge 0.03
    python -m src.pipeline.generate_bets --strategy threshold_kelly --min-edge 0.02 --multiplier 0.25

    # Scoped to a season.
    python -m src.pipeline.generate_bets --season 2023-24

The script only writes to `bet_log` when `--persist` is passed; default is
print-and-exit so dashboard exploration doesn't accidentally pollute the DB.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import engine
from src.features.matchup import load_team_games
from src.finance.backtest import BacktestConfig, run_backtest
from src.finance.bet_log import bulk_insert_bets, delete_run, ensure_table
from src.finance.staking import build as build_strategy
from src.models.walkforward import make_xgb_predict_fn, walk_forward_predictions
from src.models.xgboost_model import select_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate_bets")


FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"


def load_odds_from_db() -> pd.DataFrame:
    """Pull odds_snapshots into a DataFrame with the columns backtest expects."""
    sql = text("""
        SELECT game_date, home_team_id, away_team_id, bookmaker, ml_home, ml_away
        FROM odds_snapshots
        WHERE ml_home IS NOT NULL AND ml_away IS NOT NULL
    """)
    df = pd.read_sql_query(sql, engine)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_outcomes() -> pd.DataFrame:
    """home_won label per game from team_games."""
    raw = load_team_games(engine)
    home = raw[raw["is_home"]][["game_id", "won"]].rename(columns={"won": "home_won"})
    return home.drop_duplicates(subset="game_id")


def strategy_from_args(args: argparse.Namespace) -> dict:
    cfg: dict = {"type": args.strategy}
    if args.strategy == "flat":
        cfg["unit"] = args.unit
    elif args.strategy == "fixed_fractional":
        cfg["pct"] = args.pct
    elif args.strategy in ("fractional_kelly", "capped_kelly"):
        cfg["multiplier"] = args.multiplier
        if args.strategy == "capped_kelly":
            cfg["max_fraction"] = args.max_fraction
    elif args.strategy == "threshold_kelly":
        cfg["multiplier"] = args.multiplier
        cfg["min_edge"] = args.min_edge_strategy
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a betting backtest.")
    p.add_argument("--create-tables", action="store_true")
    p.add_argument("--strategy",
                   choices=("flat", "fixed_fractional", "full_kelly",
                            "fractional_kelly", "threshold_kelly", "capped_kelly"),
                   default="fractional_kelly")
    p.add_argument("--unit", type=float, default=100.0, help="flat stake unit")
    p.add_argument("--pct", type=float, default=0.02, help="fixed_fractional pct (0-1)")
    p.add_argument("--multiplier", type=float, default=0.25,
                   help="Kelly multiplier for fractional_kelly / threshold_kelly / capped_kelly")
    p.add_argument("--max-fraction", type=float, default=0.05,
                   help="capped_kelly hard ceiling per bet")
    p.add_argument("--min-edge-strategy", type=float, default=0.02,
                   help="threshold_kelly edge gate (strategy-level)")

    p.add_argument("--bankroll", type=float, default=10_000.0)
    p.add_argument("--min-edge", type=float, default=0.02,
                   help="Backtest-level edge gate (applied after the strategy's own gate).")
    p.add_argument("--side", choices=("home", "away", "best"), default="best")
    p.add_argument("--bookmaker", default=None,
                   help="Single bookmaker to bet into. None = consensus (auto-skipped if single-book).")
    p.add_argument("--max-concurrent-exposure", type=float, default=0.5)
    p.add_argument("--model-version", default="walkforward_xgb")

    p.add_argument("--initial-train", type=int, default=3)
    p.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    p.add_argument("--window", type=int, default=3)

    p.add_argument("--season", default=None,
                   help="Restrict to a single season (e.g. '2023-24').")
    p.add_argument("--predictions-source", type=Path, default=None,
                   help="Read predictions from a parquet (skip walk-forward). For the "
                        "meta-model output, combine with --prob-column meta_prob_lr to pick "
                        "which column from the side frame to use as home-level P(home).")
    p.add_argument("--prob-column", default="model_prob",
                   help="Column name to use as P(home_win) when --predictions-source is set. "
                        "Side-level frames expose meta_prob_lr / meta_prob_xgb via the side_to_home_preds shim.")
    p.add_argument("--persist", action="store_true",
                   help="Write bets to the bet_log table. Default is print-only.")
    p.add_argument("--replace-run", default=None,
                   help="Delete an existing backtest_run_id before writing (safe re-runs).")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Write summary dict to JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.create_tables:
        ensure_table()

    if args.predictions_source is not None:
        log.info("Reading predictions from %s (column: %s)", args.predictions_source, args.prob_column)
        src = pd.read_parquet(args.predictions_source)
        if "side" in src.columns:
            # Side-level meta frame — pull home rows and rename the chosen prob column.
            from src.models.meta_model import side_to_home_preds
            preds = side_to_home_preds(src, args.prob_column)
        else:
            preds = src.copy()
            if args.prob_column != "model_prob":
                preds = preds.rename(columns={args.prob_column: "model_prob"})
            preds = preds.dropna(subset=["model_prob"]).reset_index(drop=True)
        preds["game_date"] = pd.to_datetime(preds["game_date"])
        log.info("Loaded %d predictions from %s", len(preds), args.predictions_source)
    else:
        log.info("Loading features from %s", FEATURES_PATH)
        features_df = pd.read_parquet(FEATURES_PATH)
        feature_cols = select_features(features_df)
        log.info("Features: %d seasons, %d cols", features_df["season"].nunique(), len(feature_cols))

        log.info("Running walk-forward to produce leak-free predictions (%s, %d-season warmup)...",
                 args.mode, args.initial_train)
        preds = walk_forward_predictions(
            features_df,
            make_xgb_predict_fn(feature_cols),
            initial_train_seasons=args.initial_train,
            mode=args.mode,
            window=args.window,
        )
    if args.season:
        preds = preds[preds["fold"] == args.season].copy()
        if preds.empty:
            log.error("No predictions for season %s — did walk-forward cover it?", args.season)
            return 2
    log.info("Predictions: %d rows across %d fold(s)",
             len(preds), preds["fold"].nunique() if "fold" in preds.columns else 1)

    log.info("Loading odds from DB...")
    odds = load_odds_from_db()
    log.info("  odds_snapshots: %d rows, %d books", len(odds), odds["bookmaker"].nunique())
    if odds.empty:
        log.error("odds_snapshots is empty. Run `src.ingest.ingest_historical_odds` first.")
        return 2

    log.info("Loading outcomes...")
    outcomes = load_outcomes()
    log.info("  outcomes: %d games", len(outcomes))

    strat = build_strategy(strategy_from_args(args))
    log.info("Strategy: %s", strat)

    cfg = BacktestConfig(
        strategy=strat,
        starting_bankroll=args.bankroll,
        min_edge=args.min_edge,
        side=args.side,
        bookmaker=args.bookmaker,
        max_concurrent_exposure=args.max_concurrent_exposure,
        model_version=args.model_version,
    )

    result = run_backtest(preds, odds, outcomes, cfg)
    log.info("Backtest run_id=%s", cfg.run_id)
    log.info("Summary:\n  %s", result.summary_pretty())

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(
            {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in result.summary.items()},
            indent=2, default=str,
        ))
        log.info("Wrote summary to %s", args.output_json)

    if args.persist and not result.bets.empty:
        if args.replace_run:
            delete_run(args.replace_run)
        n = bulk_insert_bets(result.bets.to_dict("records"))
        log.info("Wrote %d bets to bet_log (run_id=%s)", n, cfg.run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
