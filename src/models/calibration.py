"""Calibration analysis: does P(home_win)=0.7 actually mean 70% win rate?

A 66%-accurate model with miscalibrated probabilities is nearly useless for
the betting-edge application the outline targets in Phase 5. Rank metrics
(accuracy, AUC) don't tell you whether the probabilities themselves are
trustworthy — reliability curves and Expected Calibration Error do.

Run:
    python -m src.models.calibration

Writes a reliability-diagram PNG to data/processed/.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.baseline import MINIMAL_FEATURES, train_logreg
from src.models.splits import by_season
from src.models.xgboost_model import select_features, train

log = logging.getLogger("calibration")


def reliability_table(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bucket predictions into equal-width bins and compare predicted vs actual."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "n": n,
            "mean_predicted": float(y_prob[mask].mean()) if n else np.nan,
            "actual_rate": float(y_true[mask].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Weighted mean |predicted - actual| across bins (classic ECE)."""
    valid = table[table["n"] > 0]
    if valid["n"].sum() == 0:
        return float("nan")
    weights = valid["n"] / valid["n"].sum()
    gaps = (valid["mean_predicted"] - valid["actual_rate"]).abs()
    return float((weights * gaps).sum())


def plot_reliability(
    tables: dict[str, pd.DataFrame],
    out_path: Path,
    title: str = "Reliability diagram — test set",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="perfectly calibrated")

    for name, tbl in tables.items():
        valid = tbl[tbl["n"] > 0]
        ece = expected_calibration_error(tbl)
        ax.plot(
            valid["mean_predicted"], valid["actual_rate"],
            marker="o", label=f"{name} (ECE={ece:.3f})",
        )

    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed home win rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(path)
    split = by_season(df)
    log.info("Split sizes: %s", split.sizes())

    # Train both models
    lr = train_logreg(split)
    xgb_features = select_features(df)
    xgb = train(split.train, split.val, xgb_features)
    log.info("XGBoost best iteration: %d", xgb.best_iteration)

    # Reliability tables on test
    y_true = split.test["home_won"].to_numpy()
    probs = {
        "LogReg (4 features)": lr.predict_proba(split.test[MINIMAL_FEATURES])[:, 1],
        "XGBoost (62 features)": xgb.predict_proba(split.test[xgb_features])[:, 1],
    }
    tables = {name: reliability_table(y_true, p) for name, p in probs.items()}

    for name, tbl in tables.items():
        log.info("\n=== %s ===\n%s\nECE=%.4f",
                 name, tbl.to_string(index=False), expected_calibration_error(tbl))

    out_png = Path(__file__).resolve().parents[2] / "data" / "processed" / "calibration_test.png"
    plot_reliability(tables, out_png)
    log.info("Reliability diagram: %s", out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
