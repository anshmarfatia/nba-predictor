"""Meta-model: does the base model add residual alpha to the market prior?

The base model (walk-forward XGBoost on team features) is 65 % accurate but
has *zero* betting alpha vs. the closing line — the edge signal is
monotonically inverted. This module tests the natural next question: if we
take the market as a prior and train a second-stage model on
`(model_prob, market_prob, edge, logit features, ...)`, can we recover any
residual alpha?

Methodology discipline (enforced by tests + the protocol below):

    - Side-level dataset — two rows per game, one per side. Symmetric
      learning; no home-bias artifact. Home-level compat shim for backtest.
    - Market-only is the null hypothesis. If the meta doesn't beat
      `side_market_prob` OOS on both log-loss and ROI, no alpha was found.
    - ROI with bootstrap 95 % CI is the alpha criterion, not a fixed
      win-rate threshold. Win rate is a diagnostic only.
    - L2-regularized LR primary (C=0.5). XGBoost (depth 2, reg_lambda 5,
      early stopping) as a robustness check.
    - Logit features: logit(p) = log(p / (1-p)) with clipping at 1e-4.
    - No in-sample metric counts toward the alpha claim — only nested
      walk-forward out-of-sample results.
    - If calibration is applied, fit only on an inner-val slice carved from
      the training window; never the outer test fold.

Run:
    python -m src.models.meta_model            # both LR + XGB, default config
    python -m src.models.meta_model --kind lr  # LR only
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier

from db import engine
from src.features.matchup import load_team_games
from src.features.odds_math import american_to_prob, devig_two_way
from src.models.evaluate import evaluate
from src.models.walkforward import make_xgb_predict_fn, walk_forward_predictions
from src.models.xgboost_model import select_features

log = logging.getLogger("meta_model")


# ---------------------------------------------------------------------------
# Canonical schema — the dataset is side-level.
# ---------------------------------------------------------------------------

CANONICAL_SIDE_COLUMNS = [
    "game_id", "game_date", "season", "fold",
    "side", "side_team_id", "opponent_team_id",
    "side_model_prob", "side_market_prob", "side_moneyline",
    "edge", "abs_edge", "side_won",
    # game-level context (repeated across both sides):
    "home_team_id", "away_team_id", "home_won",
    "model_home_prob", "market_home_prob",
    "home_moneyline", "away_moneyline",
    "predicted_at",
]

_EPS = 1e-4


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add meta-model features in-place-ish (returns a new frame).

    Logit features match LR's functional form (bounded inputs break linear
    models near the tails; log-odds are linear and unbounded). Ratio /
    interaction features are deliberately *not* included in the minimal
    set — they're correlated with `edge` and regularization will just
    redistribute weight.
    """
    out = df.copy()
    out["logit_side_model_prob"] = _logit(out["side_model_prob"])
    out["logit_side_market_prob"] = _logit(out["side_market_prob"])
    out["logit_edge"] = out["logit_side_model_prob"] - out["logit_side_market_prob"]
    return out


META_FEATURES_MINIMAL = [
    "side_model_prob", "side_market_prob",
    "edge", "abs_edge",
    "logit_side_model_prob", "logit_side_market_prob", "logit_edge",
]


# ---------------------------------------------------------------------------
# Side-level dataset construction
# ---------------------------------------------------------------------------

def _collapse_odds_per_game(odds: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_date, home_team_id, away_team_id). With multiple
    bookmakers, average the moneylines on the probability scale, then back
    to American odds via the ml-ish placeholder. Simpler pragmatic choice:
    since Kaggle source is single-book, just dedupe."""
    if odds.empty:
        return odds
    if odds["bookmaker"].nunique() <= 1:
        return odds.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"]).copy()
    # Consensus across books on the implied-probability scale.
    odds = odds.copy()
    odds["p_home_raw"] = odds["ml_home"].map(american_to_prob)
    odds["p_away_raw"] = odds["ml_away"].map(american_to_prob)
    agg = (
        odds.groupby(["game_date", "home_team_id", "away_team_id"], as_index=False)
        .agg(p_home_raw=("p_home_raw", "mean"), p_away_raw=("p_away_raw", "mean"),
             ml_home=("ml_home", "mean"), ml_away=("ml_away", "mean"))
    )
    agg["bookmaker"] = "consensus"
    return agg


def build_side_frame(
    base_preds: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the canonical side-level meta frame.

    Output: exactly two rows per game (one for home, one for away) with
    CANONICAL_SIDE_COLUMNS plus the derived logit features. The target is
    `side_won`.
    """
    required_pred_cols = {"game_id", "game_date", "home_team_id", "away_team_id",
                          "home_won", "model_prob", "fold", "predicted_at"}
    missing = required_pred_cols - set(base_preds.columns)
    if missing:
        raise ValueError(f"base_preds missing columns: {sorted(missing)}")

    odds_one = _collapse_odds_per_game(odds)
    if "p_home_raw" not in odds_one.columns:
        odds_one = odds_one.copy()
        odds_one["p_home_raw"] = odds_one["ml_home"].map(american_to_prob)
        odds_one["p_away_raw"] = odds_one["ml_away"].map(american_to_prob)
    fair = [devig_two_way(h, a) for h, a in zip(odds_one["p_home_raw"], odds_one["p_away_raw"])]
    odds_one = odds_one.copy()
    odds_one["market_home_prob"] = [f[0] for f in fair]
    odds_one["market_away_prob"] = [f[1] for f in fair]

    merged = base_preds.merge(
        odds_one[["game_date", "home_team_id", "away_team_id",
                  "ml_home", "ml_away", "market_home_prob", "market_away_prob"]],
        on=["game_date", "home_team_id", "away_team_id"],
        how="inner",
    ).rename(columns={
        "ml_home": "home_moneyline",
        "ml_away": "away_moneyline",
        "model_prob": "model_home_prob",
    })
    merged["season"] = merged["fold"]   # convention: meta's season label = base's test fold

    # Build two rows per game.
    home_rows = pd.DataFrame({
        "game_id": merged["game_id"],
        "game_date": merged["game_date"],
        "season": merged["season"],
        "fold": merged["fold"],
        "side": "home",
        "side_team_id": merged["home_team_id"].astype(int),
        "opponent_team_id": merged["away_team_id"].astype(int),
        "side_model_prob": merged["model_home_prob"],
        "side_market_prob": merged["market_home_prob"],
        "side_moneyline": merged["home_moneyline"],
        "side_won": merged["home_won"].astype(int),
        "home_team_id": merged["home_team_id"].astype(int),
        "away_team_id": merged["away_team_id"].astype(int),
        "home_won": merged["home_won"].astype(int),
        "model_home_prob": merged["model_home_prob"],
        "market_home_prob": merged["market_home_prob"],
        "home_moneyline": merged["home_moneyline"],
        "away_moneyline": merged["away_moneyline"],
        "predicted_at": merged["predicted_at"],
    })
    away_rows = pd.DataFrame({
        "game_id": merged["game_id"],
        "game_date": merged["game_date"],
        "season": merged["season"],
        "fold": merged["fold"],
        "side": "away",
        "side_team_id": merged["away_team_id"].astype(int),
        "opponent_team_id": merged["home_team_id"].astype(int),
        "side_model_prob": 1.0 - merged["model_home_prob"],
        "side_market_prob": merged["market_away_prob"],
        "side_moneyline": merged["away_moneyline"],
        "side_won": (1 - merged["home_won"].astype(int)).astype(int),
        "home_team_id": merged["home_team_id"].astype(int),
        "away_team_id": merged["away_team_id"].astype(int),
        "home_won": merged["home_won"].astype(int),
        "model_home_prob": merged["model_home_prob"],
        "market_home_prob": merged["market_home_prob"],
        "home_moneyline": merged["home_moneyline"],
        "away_moneyline": merged["away_moneyline"],
        "predicted_at": merged["predicted_at"],
    })

    side_df = pd.concat([home_rows, away_rows], ignore_index=True)
    side_df["edge"] = side_df["side_model_prob"] - side_df["side_market_prob"]
    side_df["abs_edge"] = side_df["edge"].abs()
    side_df = side_df.sort_values(["game_date", "game_id", "side"]).reset_index(drop=True)
    side_df = add_meta_features(side_df)
    return side_df


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_meta_lr(
    train: pd.DataFrame, features: list[str], C: float = 0.5, seed: int = 42,
) -> Pipeline:
    """L2-regularized LogisticRegression in a StandardScaler pipeline.

    Coefficients are reported for transparency but NOT interpreted causally
    because `side_model_prob`, `side_market_prob`, and `edge` are highly
    collinear. Regularization redistributes weight in ways that don't
    correspond to individual importance.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty="l2", max_iter=1000, random_state=seed)),
    ])
    pipe.fit(train[features], train["side_won"].astype(int))
    return pipe


def train_meta_xgb(
    train: pd.DataFrame, val: pd.DataFrame, features: list[str], seed: int = 42,
) -> XGBClassifier:
    """Shallow, heavily regularized XGBoost. Robustness check for LR — the
    sample size (~5k side-rows per fold) does not justify a high-capacity
    model. Expect results close to LR."""
    model = XGBClassifier(
        max_depth=2,
        n_estimators=300,
        learning_rate=0.03,
        reg_lambda=5.0,
        reg_alpha=0.5,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        early_stopping_rounds=25,
    )
    model.fit(
        train[features], train["side_won"].astype(int),
        eval_set=[(val[features], val["side_won"].astype(int))],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Nested walk-forward
# ---------------------------------------------------------------------------

def _inner_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve the most recent training fold as inner validation. If only one
    train fold exists, split it 80/20 by game_date order."""
    folds = sorted(train_df["fold"].unique())
    if len(folds) >= 2:
        inner_val = train_df[train_df["fold"] == folds[-1]].copy()
        inner_train = train_df[train_df["fold"].isin(folds[:-1])].copy()
        return inner_train, inner_val
    dates = train_df["game_date"].sort_values().unique()
    cutoff = dates[int(len(dates) * 0.8)]
    inner_train = train_df[train_df["game_date"] < cutoff].copy()
    inner_val = train_df[train_df["game_date"] >= cutoff].copy()
    return inner_train, inner_val


@dataclass
class MetaFold:
    test_fold: str
    train_folds: list[str]
    n_train: int
    n_test: int


def nested_walk_forward(
    side_df: pd.DataFrame,
    features: list[str],
    model_kind: Literal["lr", "xgb"],
    *, initial_train_folds: int = 1,
) -> pd.DataFrame:
    """Run nested walk-forward over meta-model folds and emit OOS predictions.

    Each outer fold trains on all prior base-model folds (which are the
    meta-model's "seasons"), inner-val-splits for XGBoost early stopping /
    calibration hooks, tests on the outer fold.

    Returns the input `side_df` with an added column:
        meta_prob_{model_kind}: float, meta-model's P(side_won=1)
    for every row whose fold is ≥ initial_train_folds-th position. Earlier
    folds (warm-up) get NaN in that column.
    """
    folds_sorted = sorted(side_df["fold"].unique())
    if len(folds_sorted) <= initial_train_folds:
        raise ValueError(
            f"Need > {initial_train_folds} folds, got {len(folds_sorted)}: {folds_sorted}"
        )

    out = side_df.copy()
    meta_col = f"meta_prob_{model_kind}"
    out[meta_col] = np.nan

    for i in range(initial_train_folds, len(folds_sorted)):
        test_fold = folds_sorted[i]
        train_folds = folds_sorted[:i]
        train = out[out["fold"].isin(train_folds)]
        test = out[out["fold"] == test_fold]
        if train.empty or test.empty:
            continue

        if model_kind == "lr":
            model = train_meta_lr(train, features)
            p = model.predict_proba(test[features])[:, 1]
        elif model_kind == "xgb":
            inner_tr, inner_val = _inner_split(train)
            model = train_meta_xgb(inner_tr, inner_val, features)
            p = model.predict_proba(test[features])[:, 1]
        else:
            raise ValueError(f"Unknown model_kind {model_kind!r}")

        out.loc[test.index, meta_col] = p
        log.info(
            "fold=%s  train_folds=%s  n_train=%d  n_test=%d",
            test_fold, train_folds, len(train), len(test),
        )

    return out


# ---------------------------------------------------------------------------
# Evaluation — per-variant metrics + ROI with bootstrap CIs
# ---------------------------------------------------------------------------

VARIANT_PROB_COLS = {
    "base":   "side_model_prob",
    "market": "side_market_prob",
    "meta_lr":  "meta_prob_lr",
    "meta_xgb": "meta_prob_xgb",
}


def variant_metrics_table(side_df: pd.DataFrame) -> pd.DataFrame:
    """Log-loss / Brier / AUC / ECE on all OOS rows for each variant."""
    from src.models.calibration import expected_calibration_error, reliability_table
    rows = []
    # "OOS rows" = rows where at least one meta variant has a prediction.
    oos_mask = side_df[list(VARIANT_PROB_COLS.values())[2:]].notna().any(axis=1)
    scope = side_df[oos_mask]
    for name, col in VARIANT_PROB_COLS.items():
        sub = scope[scope[col].notna()].copy()
        if sub.empty:
            continue
        m = evaluate(sub["side_won"].astype(int), sub[col])
        ece = expected_calibration_error(reliability_table(sub["side_won"].astype(int), sub[col]))
        rows.append({
            "variant": name, "n": len(sub),
            "accuracy": m.accuracy, "log_loss": m.log_loss,
            "brier": m.brier, "auc": m.roc_auc, "ece": ece,
        })
    return pd.DataFrame(rows)


def realize_return(side_moneyline: float, won: bool) -> float:
    """Return-per-unit-staked. +1.5 on a +150 winner, -1.0 on a loser."""
    if not won:
        return -1.0
    if side_moneyline > 0:
        return side_moneyline / 100.0
    return 100.0 / abs(side_moneyline)


def fold_roi(df: pd.DataFrame, prob_col: str, min_edge: float) -> dict:
    """ROI and diagnostics for a single variant + fold's worth of side rows.

    Bet every side row where `prob_col - side_market_prob >= min_edge`.
    Uniform 1-unit stakes (ROI is invariant to stake so long as it's flat).
    """
    mask = (df[prob_col].notna()) & ((df[prob_col] - df["side_market_prob"]) >= min_edge)
    bets = df.loc[mask].copy()
    if bets.empty:
        return {"n": 0, "roi": 0.0, "win_rate": float("nan"),
                "avg_log_return": 0.0, "avg_edge": 0.0}
    returns = np.array([
        realize_return(ml, bool(w)) for ml, w in zip(bets["side_moneyline"], bets["side_won"])
    ])
    wins = (returns > 0).astype(int)
    roi = float(returns.mean())
    # Log-utility growth rate assumes fractional Kelly (stake < bankroll); on a
    # flat-stake loss the "return" is -1.0 and log1p(-1) = -inf. Floor at
    # -0.999999 so the diagnostic is finite and interpretable.
    log_returns = np.log1p(np.maximum(returns, -0.999999))
    return {
        "n": int(len(bets)),
        "roi": roi,
        "win_rate": float(wins.mean()),
        "avg_log_return": float(log_returns.mean()),
        "avg_edge": float((bets[prob_col] - bets["side_market_prob"]).mean()),
    }


def bootstrap_roi_ci(
    df: pd.DataFrame, prob_col: str, min_edge: float,
    n_boot: int = 1000, seed: int = 0,
) -> tuple[float, float, float]:
    """Game-level bootstrap on realized ROI. Resample at the game level so
    within-game side correlation is preserved."""
    mask = (df[prob_col].notna()) & ((df[prob_col] - df["side_market_prob"]) >= min_edge)
    bets = df.loc[mask].copy()
    if bets.empty:
        return 0.0, 0.0, 0.0
    bets = bets.copy()
    bets["return"] = [
        realize_return(ml, bool(w)) for ml, w in zip(bets["side_moneyline"], bets["side_won"])
    ]
    games = bets["game_id"].unique()
    rng = np.random.default_rng(seed)
    roi_samples = np.empty(n_boot, dtype=float)
    by_game = bets.groupby("game_id")["return"].apply(list).to_dict()
    for i in range(n_boot):
        picks = rng.choice(games, size=len(games), replace=True)
        pooled = np.concatenate([by_game[g] for g in picks])
        roi_samples[i] = pooled.mean()
    point = float(bets["return"].mean())
    lo = float(np.percentile(roi_samples, 2.5))
    hi = float(np.percentile(roi_samples, 97.5))
    return point, lo, hi


def roi_per_fold(side_df: pd.DataFrame, prob_col: str, min_edge: float) -> pd.DataFrame:
    """One row per outer test fold with ROI + bootstrap CI."""
    rows = []
    for fold_label, grp in side_df[side_df[prob_col].notna()].groupby("fold"):
        stats = fold_roi(grp, prob_col, min_edge)
        point, lo, hi = bootstrap_roi_ci(grp, prob_col, min_edge, n_boot=500)
        rows.append({
            "fold": fold_label, **stats, "roi_lo95": lo, "roi_hi95": hi,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Home-level compatibility shim for run_backtest
# ---------------------------------------------------------------------------

def side_to_home_preds(side_df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """Convert side-level meta frame back to home-level predictions for
    `run_backtest`. Keeps only `side == 'home'` rows and renames."""
    home = side_df[side_df["side"] == "home"].copy()
    home = home[[
        "game_id", "game_date", "home_team_id", "away_team_id",
        "home_won", "fold", "predicted_at", prob_col,
    ]].rename(columns={prob_col: "model_prob"})
    return home.dropna(subset=["model_prob"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_odds() -> pd.DataFrame:
    df = pd.read_sql_query(
        text("SELECT game_date, home_team_id, away_team_id, bookmaker, ml_home, ml_away "
             "FROM odds_snapshots WHERE ml_home IS NOT NULL AND ml_away IS NOT NULL"),
        engine,
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + evaluate the meta-model.")
    p.add_argument("--kind", choices=("lr", "xgb", "both"), default="both")
    p.add_argument("--min-edge", type=float, default=0.02,
                   help="Edge threshold for the ROI calculation (side-level).")
    p.add_argument("--initial-train-folds", type=int, default=1,
                   help="Number of warm-up base-model folds before emitting OOS meta preds.")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parents[2] / "data" / "processed" / "meta_side_predictions.parquet")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    features_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    features_df = pd.read_parquet(features_path)
    feature_cols = select_features(features_df)
    log.info("Running walk-forward to produce base-model OOS predictions...")
    base_preds = walk_forward_predictions(features_df, make_xgb_predict_fn(feature_cols))
    log.info("base_preds: %d rows across folds %s", len(base_preds), sorted(base_preds["fold"].unique()))

    odds = _load_odds()
    log.info("odds: %d rows, %d unique games", len(odds),
             odds[["game_date", "home_team_id", "away_team_id"]].drop_duplicates().shape[0])

    side_df = build_side_frame(base_preds, odds)
    log.info("side_df: %d rows (expected 2 per game), folds=%s",
             len(side_df), sorted(side_df["fold"].unique()))

    kinds = ("lr", "xgb") if args.kind == "both" else (args.kind,)
    for k in kinds:
        log.info("Training meta-%s with nested walk-forward...", k)
        side_df = nested_walk_forward(
            side_df, META_FEATURES_MINIMAL, model_kind=k,
            initial_train_folds=args.initial_train_folds,
        )

    # Metrics table
    log.info("\n=== Out-of-sample metrics (all variants) ===")
    metrics_tbl = variant_metrics_table(side_df)
    log.info("\n%s", metrics_tbl.to_string(index=False))

    # ROI with bootstrap CIs per variant
    log.info("\n=== Realized ROI at min_edge=%.3f (95 %% bootstrap CI) ===", args.min_edge)
    scope = side_df[side_df[["meta_prob_lr", "meta_prob_xgb"]].notna().any(axis=1)]
    for name, col in VARIANT_PROB_COLS.items():
        if col not in scope.columns or scope[col].isna().all():
            continue
        point, lo, hi = bootstrap_roi_ci(scope, col, args.min_edge, n_boot=2000)
        n = ((scope[col].notna()) & ((scope[col] - scope["side_market_prob"]) >= args.min_edge)).sum()
        log.info("  %-10s  n=%-5d  ROI=%+.4f  [%+.4f, %+.4f]", name, n, point, lo, hi)

    # Per-fold ROI for the primary LR variant
    if "meta_prob_lr" in side_df.columns and not side_df["meta_prob_lr"].isna().all():
        log.info("\n=== meta-LR ROI per fold ===")
        log.info("\n%s", roi_per_fold(side_df, "meta_prob_lr", args.min_edge).to_string(index=False))

    # Write predictions parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    side_df.to_parquet(args.output, index=False)
    log.info("Wrote side-level meta predictions to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
