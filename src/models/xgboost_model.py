"""XGBoost classifier on the full feature set.

Uses defaults that are reasonable for tabular sports data out of the box;
hyperparameter tuning via Optuna is a Phase-5 task per the outline. The
feature selector picks every rolling (`_r5/_r10/_r20`), Elo, and rest column
automatically so new features added to the pipeline show up here for free.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.models.evaluate import evaluate
from src.models.splits import by_season

log = logging.getLogger("xgb")

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


# Pre-game features that are safe to use (no outcome leakage).
# Anything not on this list — raw box-score stats, the target, IDs — is ignored.
EXPLICIT_FEATURES = (
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_is_b2b", "away_is_b2b",
    "home_game_number", "away_game_number",
    "home_season_progress", "away_season_progress",
    "home_month", "home_is_covid_season",
)
ROLLING_SUFFIXES = ("_r5", "_r10", "_r20")
PLAYER_FEATURE_SUFFIXES = (
    "_rotation_avg_minutes_recent",
    "_top1_active_recent",
    "_top3_active_recent",
    "_rotation_minutes_share_recent",
)


def select_features(df: pd.DataFrame) -> list[str]:
    """Explicit allow-list: named pre-game features + every rolling column +
    home/away player-availability columns when present."""
    rolling = [c for c in df.columns if any(c.endswith(s) for s in ROLLING_SUFFIXES)]
    player = [c for c in df.columns if any(c.endswith(s) for s in PLAYER_FEATURE_SUFFIXES)]
    named = [c for c in EXPLICIT_FEATURES if c in df.columns]
    return named + rolling + player


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    n_estimators: int = 2000,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 50,
) -> XGBClassifier:
    """Train XGBoost with early stopping on the val set.

    n_estimators is intentionally generous — early stopping picks the best
    iteration, so setting it too low would cap performance. The val set
    decides when to stop; the test set is never touched here.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        train_df[features], train_df["home_won"],
        eval_set=[(val_df[features], val_df["home_won"])],
        verbose=False,
    )
    return model


def top_importances(model: XGBClassifier, features: list[str], n: int = 15) -> pd.Series:
    return (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False)
        .head(n)
    )


def save_model(model: XGBClassifier, features: list[str], version: str) -> Path:
    """Persist the trained model plus a sidecar manifest (feature list + metadata).

    XGBoost's native JSON format is forward-compatible across versions. The
    sidecar is needed because the model alone doesn't remember feature names
    — predict.py uses the manifest to build the inference vector in the
    same column order the model saw at training time.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{version}.json"
    manifest_path = MODELS_DIR / f"{version}.manifest.json"

    model.save_model(model_path)
    manifest_path.write_text(json.dumps({
        "version": version,
        "features": features,
        "best_iteration": int(model.best_iteration),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    log.info("Saved model to %s and manifest to %s", model_path, manifest_path)
    return model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and optionally save the XGBoost model.")
    p.add_argument("--save", metavar="VERSION", help="Save trained model under this version name.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(path)
    split = by_season(df)
    log.info("Split sizes: %s", split.sizes())

    features = select_features(df)
    log.info("Training on %d features: %s ...", len(features), features[:8])

    model = train(split.train, split.val, features)
    log.info("Best iteration: %d (of %d max)", model.best_iteration, model.n_estimators)

    for name, frame in [("val", split.val), ("test", split.test)]:
        y_prob = model.predict_proba(frame[features])[:, 1]
        m = evaluate(frame["home_won"], y_prob)
        log.info("xgb [%s]  %s", name, m.pretty())

    log.info("Top feature importances:\n%s", top_importances(model, features).to_string())

    if args.save:
        save_model(model, features, args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
