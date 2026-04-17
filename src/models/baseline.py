"""Baseline models: 'home team always wins' and a small logistic regression.

The home-wins baseline is the floor every real model must beat. Historically
the home team wins ~59% of regular-season games, and any model that doesn't
clear that number is worse than a coin flip with a thumb on the scale.

The LogReg uses a deliberately minimal feature set (Elo diff + rest diff + b2b
flags) to demonstrate how much signal those four features alone carry.
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import Metrics, evaluate
from src.models.splits import Split, by_season

log = logging.getLogger("baseline")

MINIMAL_FEATURES = ["elo_diff", "rest_diff", "home_is_b2b", "away_is_b2b"]


def home_wins_baseline(y_true: pd.Series) -> np.ndarray:
    """Trivially predict P(home wins) = 1.0 for every game."""
    return np.full(len(y_true), 0.99)  # clipped to avoid log-loss infinity


def train_logreg(split: Split, features: list[str] = MINIMAL_FEATURES) -> Pipeline:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
    X = split.train[features]
    y = split.train["home_won"]
    pipe.fit(X, y)
    return pipe


def report(name: str, y_true: pd.Series, y_prob: np.ndarray) -> Metrics:
    m = evaluate(y_true, y_prob)
    log.info("%-18s  %s", name, m.pretty())
    return m


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(path)
    split = by_season(df)
    log.info("Split sizes: %s", split.sizes())

    # 1. Home-wins baseline
    for name, frame in [("test", split.test), ("val", split.val)]:
        report(f"home-wins [{name}]", frame["home_won"], home_wins_baseline(frame["home_won"]))

    # 2. Minimal logistic regression
    model = train_logreg(split)
    log.info("LogReg coefficients: %s", dict(zip(MINIMAL_FEATURES, model.named_steps["lr"].coef_[0])))
    for name, frame in [("val", split.val), ("test", split.test)]:
        y_prob = model.predict_proba(frame[MINIMAL_FEATURES])[:, 1]
        report(f"logreg [{name}]", frame["home_won"], y_prob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
