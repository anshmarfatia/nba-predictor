"""Tests for the reporting/charts module.

These tests pin the *data-prep* behavior and that each chart function
writes a valid file when asked. We deliberately do NOT assert pixel
equality — chart layout can change with matplotlib versions and the
test suite stays stable if only aesthetics shift.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # never pop an interactive window in CI

import numpy as np
import pandas as pd
import pytest

from src.reporting import charts


# ---------------------------------------------------------------------------
# Drawdown math (pin the running-peak semantics)
# ---------------------------------------------------------------------------

def test_drawdown_on_monotone_equity_is_zero():
    eq = pd.Series([100, 110, 120, 130], index=pd.date_range("2024-11-01", periods=4))
    dd = charts.drawdown_series(eq)
    assert dd.max() == pytest.approx(0.0, abs=1e-12)
    assert dd.min() == pytest.approx(0.0, abs=1e-12)


def test_drawdown_is_negative_after_peak_loss():
    # Peak 120, trough 80 → -33.3% drawdown.
    eq = pd.Series([100, 120, 80, 90], index=pd.date_range("2024-11-01", periods=4))
    dd = charts.drawdown_series(eq)
    assert dd.min() == pytest.approx(-1.0 / 3.0, abs=1e-9)


def test_drawdown_running_peak_not_global_peak():
    # Peak 120 on day 2 → drop to 110 on day 3 is -8.3% from running peak,
    # even though day 3's 110 matches a later higher peak of 150.
    eq = pd.Series([100, 120, 110, 150], index=pd.date_range("2024-11-01", periods=4))
    dd = charts.drawdown_series(eq)
    assert dd.iloc[2] == pytest.approx(-10.0 / 120.0, abs=1e-9)
    assert dd.iloc[3] == pytest.approx(0.0)


def test_drawdown_handles_empty_series():
    dd = charts.drawdown_series(pd.Series(dtype=float))
    assert dd.empty


# ---------------------------------------------------------------------------
# Edge bucket assignment
# ---------------------------------------------------------------------------

def test_assign_edge_buckets_boundaries():
    # 0.02 is the lower edge of (0.02, 0.05] → lands in first bucket (0, 0.02].
    edges = pd.Series([0.01, 0.02, 0.03, 0.07, 0.10, 0.15, 0.30])
    buckets = charts.assign_edge_buckets(edges)
    # Boundary convention: right=True, include_lowest=False
    labels = [str(b) for b in buckets]
    # 0.01 in (0, 0.02]
    assert "0.02" in labels[0]
    # 0.30 in (0.2, 1.0]
    assert "1.0" in labels[-1]


def test_compute_edge_bucket_table_drops_nonpositive():
    df = pd.DataFrame({"edge": [-0.05, 0.0, 0.01, 0.03, 0.10],
                       "won": [1, 1, 1, 0, 1]})
    out = charts.compute_edge_bucket_table(df, edge_col="edge", won_col="won")
    # Only rows with edge > 0 count → 3 rows in 3 distinct buckets.
    assert out["n"].sum() == 3
    assert out["win_rate"].between(0, 1).all()


def test_compute_edge_bucket_table_empty_input():
    df = pd.DataFrame({"edge": [], "won": []})
    out = charts.compute_edge_bucket_table(df, "edge", "won")
    assert out.empty


# ---------------------------------------------------------------------------
# Chart-writing smoke tests
# ---------------------------------------------------------------------------

def _metrics_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {"variant": "base",     "n": 5102, "log_loss": 0.642, "ece": 0.012, "roi": -0.0295},
        {"variant": "market",   "n": 5102, "log_loss": 0.610, "ece": 0.010, "roi": np.nan},
        {"variant": "meta_lr",  "n": 5102, "log_loss": 0.612, "ece": 0.008, "roi": -0.1072},
        {"variant": "meta_xgb", "n": 5102, "log_loss": 0.624, "ece": 0.040, "roi": -0.0531},
    ])


def test_plot_model_comparison_writes_png(tmp_path: Path):
    path = tmp_path / "cmp.png"
    fig = charts.plot_model_comparison(_metrics_fixture(), output_path=path)
    assert path.exists() and path.stat().st_size > 2_000
    assert fig is not None


def test_plot_reliability_curves_handles_empty_bins(tmp_path: Path):
    # Construct a prob frame where some bins will be empty (all prob near 0.5).
    y_true = np.random.RandomState(0).randint(0, 2, size=200)
    y_prob = np.full(200, 0.5)
    frames = {"base": (y_true, y_prob)}
    path = tmp_path / "rel.png"
    fig = charts.plot_reliability_curves(frames, output_path=path)
    assert path.exists() and path.stat().st_size > 1_500
    assert fig is not None


def test_plot_reliability_curves_skips_empty_input(tmp_path: Path):
    frames = {"empty_var": (np.array([]), np.array([]))}
    path = tmp_path / "rel_empty.png"
    # Should not raise.
    fig = charts.plot_reliability_curves(frames, output_path=path)
    assert path.exists()
    assert fig is not None


def test_plot_edge_bucket_performance_writes_png(tmp_path: Path):
    bucket_df = pd.DataFrame({
        "bucket": pd.Categorical([
            pd.Interval(0.0, 0.02), pd.Interval(0.02, 0.05),
            pd.Interval(0.05, 0.08), pd.Interval(0.08, 0.12),
        ]),
        "n": [400, 300, 200, 50],
        "win_rate": [0.44, 0.42, 0.36, 0.30],
        "avg_edge": [0.012, 0.035, 0.063, 0.100],
    })
    path = tmp_path / "edge.png"
    fig = charts.plot_edge_bucket_performance(bucket_df, output_path=path)
    assert path.exists() and path.stat().st_size > 1_500
    assert fig is not None


def test_plot_equity_curve_writes_png(tmp_path: Path):
    idx = pd.date_range("2020-11-01", periods=30)
    equity_dict = {
        "base":    pd.Series(10000 * (1 + np.linspace(-0.4, 0.1, 30)), index=idx),
        "meta_lr": pd.Series(10000 * (1 - np.linspace(0, 0.6, 30)), index=idx),
    }
    path = tmp_path / "eq.png"
    fig = charts.plot_equity_curve(equity_dict, output_path=path)
    assert path.exists() and path.stat().st_size > 2_000


def test_plot_drawdown_curve_writes_png(tmp_path: Path):
    idx = pd.date_range("2020-11-01", periods=30)
    equity_dict = {
        "base":    pd.Series(np.linspace(10000, 13000, 30), index=idx),
        "meta_lr": pd.Series(np.linspace(10000, 4000, 30), index=idx),
    }
    path = tmp_path / "dd.png"
    fig = charts.plot_drawdown_curve(equity_dict, output_path=path)
    assert path.exists() and path.stat().st_size > 2_000


def test_plot_lr_coefficients_writes_png(tmp_path: Path):
    coef = np.array([0.8, -0.3, 0.5, -0.1, 0.2])
    feats = ["a", "b", "c", "d", "e"]
    path = tmp_path / "coef.png"
    fig = charts.plot_lr_coefficients(coef, feats, output_path=path)
    assert path.exists() and path.stat().st_size > 2_000


def test_plot_xgb_feature_importance_writes_png(tmp_path: Path):
    """Smoke test with an LR standing in for XGB — permutation_importance
    works on any estimator with predict_proba; the chart function only
    cares about the return shape."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(200) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    path = tmp_path / "fi.png"
    fig = charts.plot_xgb_feature_importance(
        model, X, y, ["a", "b", "c", "d"],
        output_path=path, n_repeats=3,
    )
    assert path.exists() and path.stat().st_size > 1_500


def test_plot_fold_performance_writes_png(tmp_path: Path):
    fold_df = pd.DataFrame([
        {"variant": "base",    "fold": "2020-21", "log_loss": 0.64, "roi": -0.05, "hit_rate": 0.45, "n_bets": 400},
        {"variant": "base",    "fold": "2021-22", "log_loss": 0.66, "roi": -0.02, "hit_rate": 0.47, "n_bets": 450},
        {"variant": "meta_lr", "fold": "2020-21", "log_loss": 0.61, "roi": -0.08, "hit_rate": 0.36, "n_bets": 300},
        {"variant": "meta_lr", "fold": "2021-22", "log_loss": 0.62, "roi": -0.16, "hit_rate": 0.28, "n_bets": 200},
    ])
    path = tmp_path / "folds.png"
    fig = charts.plot_fold_performance(fold_df, output_path=path)
    assert path.exists() and path.stat().st_size > 2_000


# ---------------------------------------------------------------------------
# CLI graceful-failure behavior
# ---------------------------------------------------------------------------

def test_generate_figures_fails_gracefully_without_features(monkeypatch, tmp_path):
    """If features.parquet is missing and odds are empty, the CLI should
    exit with a SystemExit and a clear message — not crash with a cryptic
    KeyError."""
    from src.reporting import generate_figures

    monkeypatch.setattr(generate_figures, "FEATURES_PATH", tmp_path / "nonexistent.parquet")
    monkeypatch.setattr(generate_figures, "_load_odds", lambda: pd.DataFrame())

    with pytest.raises(SystemExit):
        generate_figures.load_artifacts()
