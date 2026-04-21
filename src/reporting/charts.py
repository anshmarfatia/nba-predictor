"""Chart functions for the project's final writeup.

Each function takes prepared data (DataFrame / dict / Series), returns a
matplotlib Figure, and saves a PNG to `output_path` when provided. Chart
inputs are arranged by `generate_figures.py`; this module does only
plotting.

Design rules (kept consistent for a professional look):
  - matplotlib + a colorblind-safe seaborn palette.
  - Clear title + axis labels + legend.
  - "Lower is better" explicit in labels for log-loss / ECE / MDD.
  - Sample-size annotations where it matters (edge buckets, folds).
  - No 3D plots, no broken axes without explicit labels.
  - No chart hides the negative empirical finding — captions are the
    responsibility of the caller but the plots never mislead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")

VARIANT_COLORS = {
    "base":     "#4C72B0",
    "market":   "#55A868",
    "meta_lr":  "#C44E52",
    "meta_xgb": "#8172B2",
}
VARIANT_PRETTY = {
    "base":     "Base model",
    "market":   "Market (benchmark)",
    "meta_lr":  "Meta-LR",
    "meta_xgb": "Meta-XGB",
}


def _save(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Chart 1 — model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: log-loss, ECE, ROI per variant.

    Expects columns: `variant`, `log_loss`, `ece`, `roi` (ROI may be NaN).
    log-loss and ECE are "lower is better" (labeled on axis).
    """
    df = metrics_df.set_index("variant")
    variants = list(df.index)
    colors = [VARIANT_COLORS.get(v, "gray") for v in variants]
    labels = [VARIANT_PRETTY.get(v, v) for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, metric, title, lower_better in zip(
        axes,
        ("log_loss", "ece", "roi"),
        ("Log-loss\n(lower is better)", "ECE\n(lower is better)",
         "Realized ROI @ 2% edge\n(higher is better)"),
        (True, True, False),
    ):
        vals = df[metric].to_numpy() if metric in df.columns else np.full(len(variants), np.nan)
        bars = ax.bar(range(len(variants)), np.nan_to_num(vals, nan=0.0),
                      color=colors, edgecolor="white")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        if metric == "roi":
            ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.set_ylabel("ROI (fraction of stake)")
        else:
            ax.set_ylabel(metric)
        # Annotate bar values (even NaN-filled zeros → show "n/a").
        for i, v in enumerate(vals):
            if np.isnan(v):
                ax.text(i, 0, "n/a", ha="center", va="bottom", fontsize=9, color="gray")
            else:
                fmt = f"{v:+.2%}" if metric == "roi" else f"{v:.4f}"
                ax.text(i, v, fmt, ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)

    fig.suptitle("Model comparison — OOS metrics across 3 walk-forward folds",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — reliability curves
# ---------------------------------------------------------------------------

def plot_reliability_curves(
    prediction_frames: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path | None = None,
    n_bins: int = 10,
) -> plt.Figure:
    """Overlay reliability curves for each variant.

    `prediction_frames` is `{variant_name: (y_true, y_prob)}`. Internally
    reuses `reliability_table` and `expected_calibration_error` from
    `src.models.calibration` so binning matches the rest of the project.
    """
    from src.models.calibration import expected_calibration_error, reliability_table

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="perfect calibration")
    for name, (y_true, y_prob) in prediction_frames.items():
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        if len(y_true) == 0 or len(y_prob) == 0:
            continue
        tbl = reliability_table(y_true, y_prob, n_bins=n_bins)
        valid = tbl[tbl["n"] > 0]
        if valid.empty:
            continue
        ece = expected_calibration_error(tbl)
        color = VARIANT_COLORS.get(name, None)
        ax.plot(valid["mean_predicted"], valid["actual_rate"],
                marker="o", color=color, lw=1.8, markersize=6,
                label=f"{VARIANT_PRETTY.get(name, name)} (ECE={ece:.3f}, n={int(valid['n'].sum())})")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Reliability curves — OOS\ncurves hugging the diagonal are well-calibrated",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — edge-bucket performance
# ---------------------------------------------------------------------------

EDGE_BUCKETS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 1.0]


def assign_edge_buckets(edges: pd.Series) -> pd.Series:
    """Labeled bucket categorical for edge-bucket analysis."""
    return pd.cut(edges, bins=EDGE_BUCKETS, right=True, include_lowest=False)


def compute_edge_bucket_table(
    df: pd.DataFrame, edge_col: str, won_col: str,
) -> pd.DataFrame:
    """Per-bucket n, win rate, avg edge. Positive-edge bets only."""
    scope = df[df[edge_col] > 0].copy()
    scope["bucket"] = assign_edge_buckets(scope[edge_col])
    out = (
        scope.groupby("bucket", observed=True)
        .agg(n=(won_col, "size"), win_rate=(won_col, "mean"),
             avg_edge=(edge_col, "mean"))
        .reset_index()
    )
    return out.round(3)


def plot_edge_bucket_performance(
    edge_bucket_df: pd.DataFrame,
    title: str = "Edge bucket — bet-side win rate",
    output_path: Path | None = None,
) -> plt.Figure:
    """Win rate per edge bucket, n annotated on each bar. 50% dashed line
    is not break-even (moneylines vary) but is a visual anchor."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(edge_bucket_df))
    bars = ax.bar(x, edge_bucket_df["win_rate"], color="steelblue", edgecolor="white")
    ax.axhline(0.5, color="red", ls="--", lw=1.5, alpha=0.6, label="50 % reference")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in edge_bucket_df["bucket"]], rotation=15)
    for xi, row in zip(x, edge_bucket_df.itertuples()):
        ax.text(xi, row.win_rate + 0.01,
                f"n={int(row.n)}\n{row.win_rate:.1%}",
                ha="center", fontsize=9)
    ax.set_xlabel("Model edge bucket (positive disagreement vs. market)")
    ax.set_ylabel("Bet-side win rate")
    ax.set_title(f"{title}\nBreak-even varies with moneyline; ROI (not win rate) is the alpha test",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(0.7, edge_bucket_df["win_rate"].max() + 0.08))
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — equity curve overlay
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_dict: Mapping[str, pd.Series],
    starting_bankroll: float = 10_000.0,
    output_path: Path | None = None,
) -> plt.Figure:
    """Overlay equity curves. Each series is indexed by bet day only
    (sparse — no off-season fill). Starting bankroll annotated."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axhline(starting_bankroll, ls="--", color="gray", lw=1,
               label=f"Start = ${starting_bankroll:,.0f}")
    for name, eq in equity_dict.items():
        if eq is None or eq.empty:
            continue
        color = VARIANT_COLORS.get(name, None)
        final = float(eq.iloc[-1])
        peak = float(eq.cummax().iloc[-1])
        mdd = (eq / eq.cummax() - 1).min()
        pretty = VARIANT_PRETTY.get(name, name)
        ax.plot(pd.to_datetime(eq.index), eq.values, color=color, lw=1.8,
                label=f"{pretty}: ${final:,.0f}  (MDD {mdd:.1%})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Equity curves — 0.25× Kelly, 2 % min-edge\n(bankroll on bet days only)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — drawdown curve
# ---------------------------------------------------------------------------

def drawdown_series(equity: pd.Series) -> pd.Series:
    """Peak-to-trough drawdown: equity / running peak − 1."""
    if equity.empty:
        return pd.Series(dtype=float)
    return equity / equity.cummax() - 1.0


def plot_drawdown_curve(
    equity_dict: Mapping[str, pd.Series],
    output_path: Path | None = None,
) -> plt.Figure:
    """Drawdown vs. time per variant. Uses running peak (cummax) so flat
    off-season periods don't reset."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, eq in equity_dict.items():
        if eq is None or eq.empty:
            continue
        dd = drawdown_series(eq)
        color = VARIANT_COLORS.get(name, None)
        pretty = VARIANT_PRETTY.get(name, name)
        ax.plot(pd.to_datetime(dd.index), dd.values, color=color, lw=1.4,
                label=f"{pretty}: MDD {dd.min():.1%}")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Drawdown curves — running peak (cummax)\n"
                 "quant readers care about risk, not just final return",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 6 — meta-LR coefficients
# ---------------------------------------------------------------------------

def plot_lr_coefficients(
    coef: np.ndarray,
    feature_names: Iterable[str],
    output_path: Path | None = None,
    title: str = "Meta-LR standardized coefficients",
) -> plt.Figure:
    """Horizontal bar chart, sorted by |coef|. The caller is responsible
    for the collinearity caveat in the surrounding writeup — but we
    include it in the title too so no chart-viewer misses it."""
    coef = np.asarray(coef).ravel()
    feats = list(feature_names)
    assert len(coef) == len(feats), "coef and feature_names length mismatch"

    order = np.argsort(-np.abs(coef))
    coef = coef[order]
    feats = [feats[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(feats))))
    colors = ["#C44E52" if c < 0 else "#4C72B0" for c in coef]
    ax.barh(range(len(coef))[::-1], coef, color=colors, edgecolor="white")
    ax.set_yticks(range(len(coef))[::-1])
    ax.set_yticklabels(feats)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Standardized coefficient (after scaler)")
    ax.set_title(f"{title}\n"
                 "Coefficients are regularized and collinear — do NOT interpret as causal importance",
                 fontsize=11, fontweight="bold")
    for i, c in enumerate(coef):
        y = len(coef) - 1 - i
        ax.text(c, y, f"  {c:+.3f}", va="center",
                ha="left" if c >= 0 else "right", fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 7 — meta-XGB feature importance
# ---------------------------------------------------------------------------

def plot_xgb_feature_importance(
    model,
    X,
    y,
    feature_names: Iterable[str],
    output_path: Path | None = None,
    n_repeats: int = 10,
    random_state: int = 0,
) -> plt.Figure:
    """Permutation importance with a gain fallback.

    Permutation importance is more defensible than the built-in gain —
    it doesn't credit a split for pushing down a feature's own importance.
    Falls back to `model.feature_importances_` if permutation fails
    (e.g., tiny sample, early-stop artifact).
    """
    feats = list(feature_names)
    try:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        vals = result.importances_mean
        err = result.importances_std
        method_label = f"permutation importance (n_repeats={n_repeats})"
    except Exception:
        vals = np.asarray(model.feature_importances_, dtype=float)
        err = np.zeros_like(vals)
        method_label = "XGBoost gain importance (fallback)"

    order = np.argsort(vals)
    vals = vals[order]
    err = err[order]
    feats = [feats[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(feats))))
    ax.barh(range(len(vals)), vals, xerr=err, color="#8172B2", edgecolor="white", alpha=0.9)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(feats)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Importance score")
    ax.set_title(f"Meta-XGB feature importance\n({method_label})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Chart 8 — fold-level performance
# ---------------------------------------------------------------------------

def plot_fold_performance(
    fold_metrics_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """One panel per metric (log-loss, ROI, hit rate, n bets). Rows are
    folds, one color per variant.

    Expects columns: `variant`, `fold`, `log_loss`, `roi`, `hit_rate`, `n_bets`.
    """
    variants = list(fold_metrics_df["variant"].unique())
    folds = list(fold_metrics_df["fold"].unique())
    metrics = [
        ("log_loss", "Log-loss (lower is better)"),
        ("roi", "ROI @ 2% edge"),
        ("hit_rate", "Bet-side hit rate"),
        ("n_bets", "# bets"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    width = 0.8 / max(len(variants), 1)

    for ax, (mcol, mname) in zip(axes, metrics):
        for i, v in enumerate(variants):
            sub = fold_metrics_df[fold_metrics_df["variant"] == v].set_index("fold").reindex(folds)
            xs = np.arange(len(folds)) + (i - (len(variants) - 1) / 2) * width
            vals = sub[mcol].to_numpy() if mcol in sub.columns else np.full(len(folds), np.nan)
            color = VARIANT_COLORS.get(v, None)
            ax.bar(xs, np.nan_to_num(vals, nan=0.0), width=width, color=color,
                   label=VARIANT_PRETTY.get(v, v), edgecolor="white")
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels(folds)
        ax.set_title(mname, fontsize=11)
        if mcol == "roi":
            ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1%}"))
        elif mcol == "hit_rate":
            ax.axhline(0.5, color="red", lw=0.8, ls="--", alpha=0.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Per-fold performance — consistency check",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    _save(fig, output_path)
    return fig
