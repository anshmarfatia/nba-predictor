"""Time-based train / validation / test splits.

Sports data is a time series — random splits leak future information into the
training set and inflate every metric. The outline is emphatic on this.

Default split (per the project outline):
    train: seasons <= 2021-22
    val:   2022-23        (hyperparameter tuning)
    test:  2023-24        (touched once at the end)
    (2024-25 is held back as a "live simulation" period)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def by_season(
    df: pd.DataFrame,
    val_season: str = "2022-23",
    test_season: str = "2023-24",
    season_col: str = "season",
) -> Split:
    """Chronological split: train < val_season < test_season; rows after test_season are excluded."""
    seasons_sorted = sorted(df[season_col].unique())
    if val_season not in seasons_sorted or test_season not in seasons_sorted:
        raise ValueError(
            f"val ({val_season}) or test ({test_season}) season not in data: {seasons_sorted}"
        )
    train = df[df[season_col] < val_season].copy()
    val = df[df[season_col] == val_season].copy()
    test = df[df[season_col] == test_season].copy()
    return Split(train=train, val=val, test=test)
