"""Calibration analysis + post-hoc recalibration.

A 66%-accurate model with miscalibrated probabilities is nearly useless for
the betting-edge application the outline targets in Phase 5. Rank metrics
(accuracy, AUC) don't tell you whether the probabilities themselves are
trustworthy — reliability curves and Expected Calibration Error do.

This module does two things:

1. Measure calibration — `reliability_table`, `expected_calibration_error`,
   `plot_reliability`. Unchanged contract.

2. Fix calibration — `IsotonicCalibrator` and `PlattCalibrator` wrap the
   standard recipes. Fit on a held-out calibration set, then `.transform()`
   maps raw model probabilities to better-calibrated ones. Both persist to
   JSON (no pickle — format stays readable across sklearn versions).

Run:
    python -m src.models.calibration

Trains the XGBoost model, fits both calibrators on the val set, and reports
pre- vs post-calibration ECE / log-loss on the untouched test set. Also
writes a reliability-diagram PNG to data/processed/.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.baseline import MINIMAL_FEATURES, train_logreg
from src.models.evaluate import evaluate
from src.models.splits import by_season
from src.models.xgboost_model import select_features, train

log = logging.getLogger("calibration")

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------

_EPS = 1e-6


def _clip_probs(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


@dataclass
class IsotonicCalibrator:
    """Monotone step-function mapping raw → calibrated probabilities.

    Non-parametric; can fix arbitrary calibration shapes but needs enough
    data (a full NBA season of ~1200 games is comfortable). Prefer this
    over Platt when you have the data.
    """
    x_thresholds: np.ndarray | None = None
    y_thresholds: np.ndarray | None = None
    _reg: IsotonicRegression | None = None

    method: str = "isotonic"

    def fit(self, y_true, y_prob) -> "IsotonicCalibrator":
        y = np.asarray(y_true, dtype=int)
        p = _clip_probs(y_prob)
        reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        reg.fit(p, y)
        self._reg = reg
        self.x_thresholds = np.asarray(reg.X_thresholds_, dtype=float)
        self.y_thresholds = np.asarray(reg.y_thresholds_, dtype=float)
        return self

    def transform(self, y_prob) -> np.ndarray:
        p = _clip_probs(y_prob)
        if self._reg is not None:
            return self._reg.predict(p)
        # Loaded-from-disk path: reconstruct via linear interpolation between
        # the stored monotone thresholds. Matches IsotonicRegression.predict
        # for values inside the training range; clamps outside.
        if self.x_thresholds is None or self.y_thresholds is None:
            raise RuntimeError("IsotonicCalibrator is not fitted or loaded.")
        return np.interp(p, self.x_thresholds, self.y_thresholds)

    def save(self, path: Path) -> None:
        if self.x_thresholds is None or self.y_thresholds is None:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "method": "isotonic",
            "x_thresholds": self.x_thresholds.tolist(),
            "y_thresholds": self.y_thresholds.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        state = json.loads(Path(path).read_text())
        if state.get("method") != "isotonic":
            raise ValueError(f"Not an isotonic calibrator file: {state.get('method')}")
        return cls(
            x_thresholds=np.asarray(state["x_thresholds"], dtype=float),
            y_thresholds=np.asarray(state["y_thresholds"], dtype=float),
        )


@dataclass
class PlattCalibrator:
    """Sigmoid fit on logits: p_cal = sigmoid(a * logit(p_raw) + b).

    Two parameters, so it can only re-scale and shift — it can't fix
    non-sigmoidal shapes. Use when the calibration set is small.
    """
    coef: float | None = None
    intercept: float | None = None

    method: str = "platt"

    def fit(self, y_true, y_prob) -> "PlattCalibrator":
        y = np.asarray(y_true, dtype=int)
        p = _clip_probs(y_prob)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(logits, y)
        self.coef = float(lr.coef_[0][0])
        self.intercept = float(lr.intercept_[0])
        return self

    def transform(self, y_prob) -> np.ndarray:
        if self.coef is None or self.intercept is None:
            raise RuntimeError("PlattCalibrator is not fitted or loaded.")
        p = _clip_probs(y_prob)
        logits = np.log(p / (1.0 - p))
        z = self.coef * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: Path) -> None:
        if self.coef is None or self.intercept is None:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "method": "platt",
            "coef": self.coef,
            "intercept": self.intercept,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PlattCalibrator":
        state = json.loads(Path(path).read_text())
        if state.get("method") != "platt":
            raise ValueError(f"Not a Platt calibrator file: {state.get('method')}")
        return cls(coef=float(state["coef"]), intercept=float(state["intercept"]))


def load_calibrator(path: Path):
    """Dispatch to the right loader based on the stored method tag."""
    state = json.loads(Path(path).read_text())
    method = state.get("method")
    if method == "isotonic":
        return IsotonicCalibrator.load(path)
    if method == "platt":
        return PlattCalibrator.load(path)
    raise ValueError(f"Unknown calibrator method: {method!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(path)
    split = by_season(df)
    log.info("Split sizes: %s", split.sizes())

    lr = train_logreg(split)
    xgb_features = select_features(df)
    xgb = train(split.train, split.val, xgb_features)
    log.info("XGBoost best iteration: %d", xgb.best_iteration)

    # Raw probabilities on val (for calibrator fitting) and test (for scoring).
    # Using val for calibration has a small caveat: XGB already used it for
    # early stopping. For stricter separation, carve off a dedicated calib
    # slice of train — but with a single season of val and good discipline
    # elsewhere this is a standard, reasonable trade-off.
    val_probs_xgb = xgb.predict_proba(split.val[xgb_features])[:, 1]
    test_probs_xgb = xgb.predict_proba(split.test[xgb_features])[:, 1]
    test_probs_lr = lr.predict_proba(split.test[MINIMAL_FEATURES])[:, 1]

    y_val = split.val["home_won"].to_numpy()
    y_test = split.test["home_won"].to_numpy()

    iso = IsotonicCalibrator().fit(y_val, val_probs_xgb)
    platt = PlattCalibrator().fit(y_val, val_probs_xgb)
    test_probs_iso = iso.transform(test_probs_xgb)
    test_probs_platt = platt.transform(test_probs_xgb)

    probs = {
        "LogReg (4 features)": test_probs_lr,
        "XGBoost (raw)": test_probs_xgb,
        "XGBoost + Isotonic": test_probs_iso,
        "XGBoost + Platt": test_probs_platt,
    }
    tables = {name: reliability_table(y_test, p) for name, p in probs.items()}

    log.info("\n%-28s  %-9s  %-9s  %-9s  %-9s",
             "Model", "acc", "log_loss", "brier", "ECE")
    for name, p in probs.items():
        m = evaluate(y_test, p)
        ece = expected_calibration_error(tables[name])
        log.info("%-28s  %.4f    %.4f    %.4f    %.4f",
                 name, m.accuracy, m.log_loss, m.brier, ece)

    out_png = Path(__file__).resolve().parents[2] / "data" / "processed" / "calibration_test.png"
    plot_reliability(tables, out_png)
    log.info("Reliability diagram: %s", out_png)

    # Persist calibrators alongside the XGB model artifacts so predict.py /
    # the dashboard can apply them when present.
    iso_path = MODELS_DIR / "latest.isotonic.json"
    platt_path = MODELS_DIR / "latest.platt.json"
    iso.save(iso_path)
    platt.save(platt_path)
    log.info("Saved calibrators: %s, %s", iso_path, platt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
