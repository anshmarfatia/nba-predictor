"""CLI: load existing artifacts, compute chart inputs, save PNGs.

Artifacts consumed (fail gracefully if missing):
  - data/processed/features.parquet     (from src.pipeline.build_training_set)
  - data/processed/meta_side_predictions.parquet (from src.models.meta_model)
  - odds_snapshots                      (from src.ingest.ingest_historical_odds)
  - team_games                          (from src.ingest.ingest_games)

The CLI emits every chart into docs/assets/:
  model_comparison.png
  reliability_curves.png
  edge_bucket_{variant}.png     (one per variant with an edge signal)
  equity_curve_overlay.png
  drawdown_curve.png
  meta_lr_coefficients.png
  meta_xgb_feature_importance.png
  fold_performance.png

Results are consumed from the existing meta-model + backtest pipelines —
we do NOT retrain with different discipline. This module is reporting only.

Usage:
    python -m src.reporting.generate_figures
    python -m src.reporting.generate_figures --skip-heavy   # no backtests, faster
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import engine
from src.features.matchup import load_team_games
from src.finance.backtest import BacktestConfig, run_backtest
from src.finance.staking import FractionalKelly
from src.models.calibration import expected_calibration_error, reliability_table
from src.models.evaluate import evaluate
from src.models.walkforward import make_xgb_predict_fn, walk_forward_predictions
from src.models.xgboost_model import select_features
from src.models.meta_model import (
    META_FEATURES_MINIMAL,
    VARIANT_PROB_COLS,
    bootstrap_roi_ci,
    build_side_frame,
    fold_roi,
    nested_walk_forward,
    side_to_home_preds,
    train_meta_lr,
    train_meta_xgb,
)
from src.reporting import charts

log = logging.getLogger("generate_figures")

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
META_PREDS_PATH = ROOT / "data" / "processed" / "meta_side_predictions.parquet"
ASSETS_DIR = ROOT / "docs" / "assets"


# ---------------------------------------------------------------------------
# Artifact loading with graceful failures
# ---------------------------------------------------------------------------

@dataclass
class ArtifactBundle:
    features_df: pd.DataFrame
    base_preds: pd.DataFrame
    odds: pd.DataFrame
    outcomes: pd.DataFrame
    side_df: pd.DataFrame


def _load_odds() -> pd.DataFrame:
    df = pd.read_sql_query(
        text("SELECT game_date, home_team_id, away_team_id, bookmaker, ml_home, ml_away "
             "FROM odds_snapshots WHERE ml_home IS NOT NULL AND ml_away IS NOT NULL"),
        engine,
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _load_outcomes() -> pd.DataFrame:
    raw = load_team_games(engine)
    return (
        raw[raw["is_home"]][["game_id", "won"]]
        .rename(columns={"won": "home_won"})
        .drop_duplicates(subset="game_id")
    )


def _load_or_rebuild_meta(features_df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """If the side predictions parquet exists, reuse it; else rebuild end-to-end."""
    if META_PREDS_PATH.exists():
        log.info("Loading cached meta predictions: %s", META_PREDS_PATH)
        return pd.read_parquet(META_PREDS_PATH)
    log.info("Cache missing — rebuilding walk-forward predictions + meta-model...")
    feat_cols = select_features(features_df)
    base_preds = walk_forward_predictions(features_df, make_xgb_predict_fn(feat_cols))
    side_df = build_side_frame(base_preds, odds)
    for kind in ("lr", "xgb"):
        side_df = nested_walk_forward(side_df, META_FEATURES_MINIMAL, model_kind=kind)
    return side_df


def load_artifacts() -> ArtifactBundle:
    missing = []
    if not FEATURES_PATH.exists():
        missing.append(f"{FEATURES_PATH}  (run: python -m src.pipeline.build_training_set)")

    try:
        odds = _load_odds()
    except Exception as exc:
        raise SystemExit(f"Could not read odds_snapshots from the DB: {exc}\n"
                         "Run: python -m src.ingest.ingest_historical_odds --csv <csv>") from exc
    if odds.empty:
        missing.append("odds_snapshots is empty  (run: python -m src.ingest.ingest_historical_odds)")

    try:
        outcomes = _load_outcomes()
    except Exception as exc:
        raise SystemExit(f"Could not read team_games from the DB: {exc}\n"
                         "Run: python -m src.ingest.ingest_games") from exc

    if missing:
        raise SystemExit("Missing prerequisite artifacts:\n  - " + "\n  - ".join(missing))

    features_df = pd.read_parquet(FEATURES_PATH)
    side_df = _load_or_rebuild_meta(features_df, odds)
    # The cached parquet may have been built before walk-forward ran its full
    # course; verify the meta columns are present.
    for col in ("meta_prob_lr", "meta_prob_xgb"):
        if col not in side_df.columns:
            log.warning("Meta column %s not in cache — rebuilding side_df in-place.", col)
            side_df = build_side_frame(
                walk_forward_predictions(features_df, make_xgb_predict_fn(select_features(features_df))),
                odds,
            )
            for kind in ("lr", "xgb"):
                side_df = nested_walk_forward(side_df, META_FEATURES_MINIMAL, model_kind=kind)
            break

    base_preds = (
        side_df[side_df["side"] == "home"][
            ["game_id", "game_date", "home_team_id", "away_team_id",
             "home_won", "model_home_prob", "fold", "predicted_at"]
        ]
        .rename(columns={"model_home_prob": "model_prob"})
        .reset_index(drop=True)
    )
    return ArtifactBundle(features_df, base_preds, odds, outcomes, side_df)


# ---------------------------------------------------------------------------
# Compute chart inputs from artifacts
# ---------------------------------------------------------------------------

def compute_variant_metrics(side_df: pd.DataFrame, min_edge: float = 0.02) -> pd.DataFrame:
    """Aggregate metrics per variant on the same OOS scope (where meta
    predictions exist). ROI uses the same edge gate as the notebook."""
    # Scope: rows where at least one meta variant has a prediction.
    scope_mask = side_df[["meta_prob_lr", "meta_prob_xgb"]].notna().any(axis=1)
    scope = side_df.loc[scope_mask].copy()
    rows = []
    for name, col in VARIANT_PROB_COLS.items():
        if col not in scope.columns or scope[col].isna().all():
            rows.append({"variant": name, "n": 0, "log_loss": np.nan, "ece": np.nan, "roi": np.nan})
            continue
        sub = scope[scope[col].notna()]
        y = sub["side_won"].astype(int)
        p = sub[col].to_numpy()
        m = evaluate(y, p)
        ece = expected_calibration_error(reliability_table(y, p))
        # Market-vs-market has no edges → skip ROI for "market".
        if name == "market":
            roi_val = np.nan
        else:
            stats = fold_roi(sub, col, min_edge)
            roi_val = stats["roi"]
        rows.append({
            "variant": name, "n": int(len(sub)),
            "log_loss": m.log_loss, "ece": ece, "roi": roi_val,
        })
    return pd.DataFrame(rows)


def _run_strategy_backtest(
    side_df: pd.DataFrame, prob_col: str, odds: pd.DataFrame, outcomes: pd.DataFrame,
):
    preds = side_to_home_preds(side_df, prob_col)
    if preds.empty:
        return None
    cfg = BacktestConfig(
        strategy=FractionalKelly(multiplier=0.25),
        starting_bankroll=10_000.0,
        min_edge=0.02,
        side="best",
        bookmaker=None,
        max_concurrent_exposure=0.5,
        model_version=prob_col,
    )
    return run_backtest(preds, odds, outcomes, cfg)


def compute_equity_dict(
    side_df: pd.DataFrame, odds: pd.DataFrame, outcomes: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Run 0.25× Kelly backtest for each variant that has a signal,
    return {variant: equity Series}."""
    out: dict[str, pd.Series] = {}
    for name, col in [("base", "model_home_prob"),
                      ("meta_lr", "meta_prob_lr"),
                      ("meta_xgb", "meta_prob_xgb")]:
        result = _run_strategy_backtest(side_df, col, odds, outcomes)
        if result is not None and not result.equity.empty:
            out[name] = result.equity
    return out


def compute_fold_metrics(side_df: pd.DataFrame, min_edge: float = 0.02) -> pd.DataFrame:
    """Per-fold per-variant metrics — the consistency check."""
    rows = []
    scope = side_df[side_df[["meta_prob_lr", "meta_prob_xgb"]].notna().any(axis=1)]
    for name, col in VARIANT_PROB_COLS.items():
        if col not in scope.columns:
            continue
        for fold, grp in scope.groupby("fold"):
            sub = grp[grp[col].notna()]
            if sub.empty:
                continue
            y = sub["side_won"].astype(int)
            p = sub[col].to_numpy()
            m = evaluate(y, p)
            if name == "market":
                roi_val, hit, n_bets = np.nan, np.nan, 0
            else:
                stats = fold_roi(sub, col, min_edge)
                roi_val, hit, n_bets = stats["roi"], stats["win_rate"], stats["n"]
            rows.append({
                "variant": name, "fold": fold,
                "log_loss": m.log_loss, "roi": roi_val,
                "hit_rate": hit, "n_bets": n_bets,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LR coefficients + XGB feature importance
# ---------------------------------------------------------------------------

def fit_final_meta_lr(side_df: pd.DataFrame):
    """For the coefficient chart: refit LR on the union of training folds
    (all folds except the last). Mirrors the last nested WF fit without
    needing to re-enter the loop."""
    folds_sorted = sorted(side_df["fold"].unique())
    if len(folds_sorted) < 2:
        return None
    train = side_df[side_df["fold"].isin(folds_sorted[:-1])].copy()
    return train_meta_lr(train, META_FEATURES_MINIMAL)


def fit_final_meta_xgb(side_df: pd.DataFrame):
    folds_sorted = sorted(side_df["fold"].unique())
    if len(folds_sorted) < 3:
        return None, None, None
    # Use all but the last fold to train, and the next-to-last as inner-val.
    inner_val = side_df[side_df["fold"] == folds_sorted[-2]].copy()
    train = side_df[side_df["fold"].isin(folds_sorted[:-2])].copy()
    model = train_meta_xgb(train, inner_val, META_FEATURES_MINIMAL)
    # Permutation importance on the held-out last fold.
    last = side_df[side_df["fold"] == folds_sorted[-1]].copy()
    X = last[META_FEATURES_MINIMAL]
    y = last["side_won"].astype(int)
    return model, X, y


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all project figures.")
    p.add_argument("--skip-heavy", action="store_true",
                   help="Skip backtests (no equity/drawdown curves).")
    p.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_artifacts()
    side_df = bundle.side_df

    generated: list[Path] = []

    # 1) Model comparison
    metrics_df = compute_variant_metrics(side_df)
    log.info("\nVariant metrics:\n%s", metrics_df.round(4).to_string(index=False))
    out = args.assets_dir / "model_comparison.png"
    charts.plot_model_comparison(metrics_df, output_path=out)
    generated.append(out)

    # 2) Reliability curves
    scope = side_df[side_df[["meta_prob_lr", "meta_prob_xgb"]].notna().any(axis=1)]
    frames = {}
    for name, col in VARIANT_PROB_COLS.items():
        if col in scope.columns and not scope[col].isna().all():
            sub = scope[scope[col].notna()]
            frames[name] = (sub["side_won"].astype(int).to_numpy(), sub[col].to_numpy())
    out = args.assets_dir / "reliability_curves.png"
    charts.plot_reliability_curves(frames, output_path=out)
    generated.append(out)

    # 3) Edge-bucket performance (one chart per variant with an edge signal)
    for variant_name, prob_col in [("base", "model_home_prob"),
                                    ("meta_lr", "meta_prob_lr"),
                                    ("meta_xgb", "meta_prob_xgb")]:
        if prob_col not in scope.columns or scope[prob_col].isna().all():
            continue
        sub = scope[scope[prob_col].notna()].copy()
        sub["edge"] = sub[prob_col] - sub["side_market_prob"]
        bucket_df = charts.compute_edge_bucket_table(
            sub, edge_col="edge", won_col="side_won",
        )
        if bucket_df.empty:
            continue
        pretty = charts.VARIANT_PRETTY.get(variant_name, variant_name)
        out = args.assets_dir / f"edge_bucket_{variant_name}.png"
        charts.plot_edge_bucket_performance(
            bucket_df, title=f"Edge bucket — {pretty}", output_path=out,
        )
        generated.append(out)

    # 4–5) Equity + drawdown curves (requires backtest runs; heavy)
    if not args.skip_heavy:
        log.info("Running portfolio backtests for equity/drawdown charts...")
        equity_dict = compute_equity_dict(side_df, bundle.odds, bundle.outcomes)
        if equity_dict:
            out = args.assets_dir / "equity_curve_overlay.png"
            charts.plot_equity_curve(equity_dict, output_path=out)
            generated.append(out)
            out = args.assets_dir / "drawdown_curve.png"
            charts.plot_drawdown_curve(equity_dict, output_path=out)
            generated.append(out)
        else:
            log.warning("No backtests produced equity curves — skipping charts 4 & 5.")

    # 6) Meta-LR coefficients
    lr_model = fit_final_meta_lr(side_df)
    if lr_model is not None:
        coef = lr_model.named_steps["lr"].coef_[0]
        out = args.assets_dir / "meta_lr_coefficients.png"
        charts.plot_lr_coefficients(coef, META_FEATURES_MINIMAL, output_path=out)
        generated.append(out)
    else:
        log.warning("Too few folds for LR final fit — skipping chart 6.")

    # 7) Meta-XGB feature importance
    xgb_model, X, y = fit_final_meta_xgb(side_df)
    if xgb_model is not None:
        out = args.assets_dir / "meta_xgb_feature_importance.png"
        charts.plot_xgb_feature_importance(
            xgb_model, X, y, META_FEATURES_MINIMAL, output_path=out,
        )
        generated.append(out)
    else:
        log.warning("Too few folds for XGB final fit — skipping chart 7.")

    # 8) Fold performance
    fold_df = compute_fold_metrics(side_df)
    if not fold_df.empty:
        out = args.assets_dir / "fold_performance.png"
        charts.plot_fold_performance(fold_df, output_path=out)
        generated.append(out)

    log.info("\nGenerated %d chart file(s):", len(generated))
    for p in generated:
        log.info("  %s", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
