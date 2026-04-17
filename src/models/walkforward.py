"""Walk-forward cross-validation for time-series sports data.

A single train / val / test split (as in `splits.by_season`) tells you how
the model did on *one* holdout season. That can be lucky. Walk-forward
retrains season-by-season so every season after the initial window becomes
an out-of-sample test set, and the average / dispersion of metrics across
folds is the number to trust.

Two modes:
    expanding — fold N trains on seasons[0..N-1], tests on seasons[N]
    rolling   — fold N trains on the most recent W seasons, tests on seasons[N]

Rolling is useful when the league changes (pace revolution, rule changes);
expanding is the default since more data usually wins.

Usage:
    python -m src.models.walkforward                 # XGB, expanding, 3-season warmup
    python -m src.models.walkforward --mode rolling --window 3
    python -m src.models.walkforward --model logreg  # baseline logreg instead
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from src.models.baseline import MINIMAL_FEATURES, train_logreg
from src.models.evaluate import evaluate
from src.models.splits import Split
from src.models.xgboost_model import select_features, train as xgb_train

log = logging.getLogger("walkforward")


@dataclass(frozen=True)
class Fold:
    train_seasons: list[str]
    test_season: str
    train: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "test": len(self.test)}


def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_seasons: int = 3,
    mode: str = "expanding",
    window: int = 3,
    season_col: str = "season",
) -> Iterator[Fold]:
    """Yield folds that advance one season at a time.

    The first `initial_train_seasons` seasons form the warm-up training set;
    every subsequent season becomes a test fold in turn.
    """
    seasons = sorted(df[season_col].unique())
    if len(seasons) <= initial_train_seasons:
        raise ValueError(
            f"Need more than {initial_train_seasons} seasons, got {len(seasons)}: {seasons}"
        )
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}")

    for i in range(initial_train_seasons, len(seasons)):
        test_season = seasons[i]
        if mode == "expanding":
            train_seasons = seasons[:i]
        else:
            train_seasons = seasons[max(0, i - window):i]
        yield Fold(
            train_seasons=list(train_seasons),
            test_season=test_season,
            train=df[df[season_col].isin(train_seasons)].copy(),
            test=df[df[season_col] == test_season].copy(),
        )


def _inner_split(train: pd.DataFrame, season_col: str = "season") -> Split:
    """Carve the most recent training season off as inner-val for early stopping.

    Walk-forward's outer test set is untouched. Inside each fold, XGBoost still
    needs *some* held-out data to stop training at the right iteration — we use
    the last train season for that, the same discipline `by_season` uses.
    """
    seasons = sorted(train[season_col].unique())
    if len(seasons) < 2:
        raise ValueError(f"Inner split needs ≥2 training seasons, got {seasons}")
    inner_val_season = seasons[-1]
    return Split(
        train=train[train[season_col] < inner_val_season].copy(),
        val=train[train[season_col] == inner_val_season].copy(),
        test=pd.DataFrame(),
    )


def make_xgb_predict_fn(features: list[str]) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Factory that returns a predict_fn for XGBoost with inner early stopping."""
    def predict_fn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        inner = _inner_split(train)
        model = xgb_train(inner.train, inner.val, features)
        return model.predict_proba(test[features])[:, 1]
    return predict_fn


def make_logreg_predict_fn(features: list[str] = MINIMAL_FEATURES) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Factory for the minimal logistic-regression baseline."""
    def predict_fn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        fake_split = Split(train=train, val=train.head(0), test=train.head(0))
        model = train_logreg(fake_split, features=features)
        return model.predict_proba(test[features])[:, 1]
    return predict_fn


def walk_forward_evaluate(
    df: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    *,
    initial_train_seasons: int = 3,
    mode: str = "expanding",
    window: int = 3,
    season_col: str = "season",
    target: str = "home_won",
) -> pd.DataFrame:
    """Run predict_fn over each fold. Returns one row per test season."""
    rows = []
    for fold in walk_forward_splits(
        df,
        initial_train_seasons=initial_train_seasons,
        mode=mode,
        window=window,
        season_col=season_col,
    ):
        probs = predict_fn(fold.train, fold.test)
        m = evaluate(fold.test[target], probs)
        row = {
            "test_season": fold.test_season,
            "train_span": f"{fold.train_seasons[0]}..{fold.train_seasons[-1]}",
            "n_train": len(fold.train),
            "n_test": len(fold.test),
            **m.as_dict(),
        }
        rows.append(row)
        log.info("fold test=%s  train=%d  %s", fold.test_season, len(fold.train), m.pretty())
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.Series:
    """Mean and std across folds — report both. Std matters: a single lucky fold
    shouldn't anchor how good you think the model is."""
    cols = ["accuracy", "log_loss", "brier", "roc_auc"]
    agg = metrics[cols].agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward CV across NBA seasons.")
    p.add_argument("--model", choices=("xgb", "logreg"), default="xgb")
    p.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    p.add_argument("--initial-train", type=int, default=3,
                   help="Number of seasons in the initial training window.")
    p.add_argument("--window", type=int, default=3,
                   help="Rolling-window size (ignored for expanding mode).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(path)
    log.info("Loaded %d rows across %d seasons", len(df), df["season"].nunique())

    if args.model == "xgb":
        features = select_features(df)
        predict_fn = make_xgb_predict_fn(features)
        log.info("Model: XGBoost on %d features", len(features))
    else:
        predict_fn = make_logreg_predict_fn()
        log.info("Model: LogReg on %d features", len(MINIMAL_FEATURES))

    metrics = walk_forward_evaluate(
        df, predict_fn,
        initial_train_seasons=args.initial_train,
        mode=args.mode,
        window=args.window,
    )
    log.info("\nPer-fold metrics:\n%s", metrics.to_string(index=False))
    log.info("\nAggregate across %d folds:\n%s", len(metrics), summarize(metrics).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
