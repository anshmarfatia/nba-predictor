"""Optuna-based hyperparameter tuning for the XGBoost NBA predictor.

Sweeps regularization + capacity hyperparameters using TPE (Bayesian)
optimization, scored on the 2022-23 validation log-loss. `n_estimators`
is intentionally left out of the search: every trial uses early stopping
so the tree count is decided per-trial by the eval_set dynamics — same
discipline training already uses.

After the study finishes, we retrain one clean model with the best params
(train-only, val-for-early-stopping — same contract v1 used) and evaluate
on the untouched 2023-24 test set. The test numbers are the ones to trust.

Usage:
    python -m src.models.tune --save v2 --trials 100
    python -m src.models.tune --save v2 --trials 50 --timeout 180
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.models.evaluate import evaluate
from src.models.splits import by_season
from src.models.xgboost_model import MODELS_DIR, save_model, select_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tune")


def build_classifier(params: dict) -> XGBClassifier:
    """Single place defining the non-search constants (objective, seed, etc.)."""
    return XGBClassifier(
        n_estimators=2000,
        early_stopping_rounds=50,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        **params,
    )


def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = build_classifier(params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, y_prob)

    return objective


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune XGBoost with Optuna and save the best model.")
    p.add_argument("--save", required=True,
                   help="Version name to save the tuned model (e.g. 'v2').")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--timeout", type=int, default=None,
                   help="Optional wall-clock cap in seconds.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(path)
    split = by_season(df)
    features = select_features(df)
    log.info("Split sizes: %s, features: %d", split.sizes(), len(features))

    X_train, y_train = split.train[features], split.train["home_won"]
    X_val, y_val = split.val[features], split.val["home_won"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    log.info("Running %d trials (timeout=%s)...", args.trials, args.timeout)
    study.optimize(
        make_objective(X_train, y_train, X_val, y_val),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    log.info("Best val log-loss: %.4f", study.best_value)
    log.info("Best params:")
    for k, v in study.best_params.items():
        log.info("  %s = %s", k, v)

    log.info("Retraining with best params...")
    final = build_classifier(study.best_params)
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log.info("Final best_iteration: %d", final.best_iteration)

    for name, frame in [("val", split.val), ("test", split.test)]:
        probs = final.predict_proba(frame[features])[:, 1]
        m = evaluate(frame["home_won"], probs)
        log.info("tuned [%s]  %s", name, m.pretty())

    save_model(final, features, args.save)
    params_path = MODELS_DIR / f"{args.save}.params.json"
    params_path.write_text(json.dumps({
        "best_params": study.best_params,
        "best_val_logloss": study.best_value,
        "n_trials": len(study.trials),
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    log.info("Saved best params to %s", params_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
