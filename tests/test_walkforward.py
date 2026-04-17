"""Tests for walk-forward split logic.

The critical invariants: each fold's test set is entirely after its training
set, and the training set never contains the test season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.walkforward import (
    Fold,
    _inner_split,
    walk_forward_evaluate,
    walk_forward_splits,
)


def _toy_df(n_per_season: int = 100) -> pd.DataFrame:
    """Five fake seasons, each with n rows and a roughly balanced target."""
    rng = np.random.default_rng(0)
    rows = []
    for s in ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24"):
        for i in range(n_per_season):
            rows.append({
                "season": s,
                "game_id": f"{s}-{i}",
                "feat": rng.normal(),
                "home_won": int(rng.random() < 0.58),
            })
    return pd.DataFrame(rows)


def test_expanding_splits_advance_one_season():
    df = _toy_df()
    folds = list(walk_forward_splits(df, initial_train_seasons=2, mode="expanding"))
    assert [f.test_season for f in folds] == ["2021-22", "2022-23", "2023-24"]
    assert folds[0].train_seasons == ["2019-20", "2020-21"]
    assert folds[-1].train_seasons == ["2019-20", "2020-21", "2021-22", "2022-23"]


def test_rolling_splits_keep_fixed_window():
    df = _toy_df()
    folds = list(walk_forward_splits(df, initial_train_seasons=2, mode="rolling", window=2))
    for f in folds:
        assert len(f.train_seasons) <= 2
    assert folds[-1].train_seasons == ["2021-22", "2022-23"]


def test_no_overlap_between_train_and_test():
    """The single most important safety property: test season not in train."""
    df = _toy_df()
    for fold in walk_forward_splits(df, initial_train_seasons=2):
        assert fold.test_season not in set(fold.train["season"])
        assert set(fold.test["season"]) == {fold.test_season}


def test_too_few_seasons_raises():
    df = _toy_df()
    df = df[df["season"].isin(["2019-20", "2020-21"])]
    with pytest.raises(ValueError):
        list(walk_forward_splits(df, initial_train_seasons=2))


def test_invalid_mode_raises():
    df = _toy_df()
    with pytest.raises(ValueError):
        list(walk_forward_splits(df, initial_train_seasons=2, mode="backwards"))


def test_inner_split_uses_last_train_season_as_val():
    df = _toy_df()
    folds = list(walk_forward_splits(df, initial_train_seasons=3, mode="expanding"))
    inner = _inner_split(folds[0].train)
    assert set(inner.val["season"]) == {"2021-22"}
    assert "2021-22" not in set(inner.train["season"])


def test_walk_forward_evaluate_one_row_per_fold():
    df = _toy_df()

    def dumb_predict(train, test):
        # Constant 0.6 — accuracy will equal the base rate, metrics will be finite.
        return np.full(len(test), 0.6)

    metrics = walk_forward_evaluate(df, dumb_predict, initial_train_seasons=2)
    assert len(metrics) == 3
    assert set(metrics.columns) >= {"test_season", "accuracy", "log_loss", "brier", "roc_auc"}
    assert metrics["n_test"].gt(0).all()
