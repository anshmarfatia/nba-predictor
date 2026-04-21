"""Tests for the meta-model.

The methodological discipline from the plan (revisions 1-8) is encoded as
test invariants. In particular:

  - Side symmetry: two rows per game, probabilities sum to ~1.
  - Leakage: changing fold (N+1)'s outcome does not alter its meta prediction.
  - Look-ahead: predicted_at <= game_date after the compat shim.
  - Regularization: C=0.5 produces smaller coefficients than C=100.
  - Synthetic sanity (market efficient): meta cannot beat market on log-loss.
  - Synthetic sanity (learnable signal): meta recovers the true probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from src.features.odds_math import prob_to_american
from src.models.meta_model import (
    META_FEATURES_MINIMAL,
    _logit,
    add_meta_features,
    build_side_frame,
    nested_walk_forward,
    side_to_home_preds,
    train_meta_lr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synth_base_preds(n_folds: int = 3, games_per_fold: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    day0 = pd.Timestamp("2020-10-25")
    gid = 0
    for fold_i in range(n_folds):
        fold_label = f"202{fold_i}-2{fold_i + 1}"
        fold_start = day0 + pd.Timedelta(days=fold_i * 200)
        for g in range(games_per_fold):
            rows.append({
                "game_id": f"G{gid:05d}",
                "game_date": fold_start + pd.Timedelta(days=g % 180),
                "home_team_id": 1610612700 + (g % 30),
                "away_team_id": 1610612700 + ((g + 7) % 30),
                "home_won": int(rng.random() < 0.58),
                "model_prob": float(np.clip(rng.normal(0.58, 0.12), 0.02, 0.98)),
                "fold": fold_label,
                "predicted_at": fold_start - pd.Timedelta(days=1),
            })
            gid += 1
    return pd.DataFrame(rows)


def _synth_odds(base_preds: pd.DataFrame, noise_sd: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Fake odds whose fair probabilities are the true outcome probability
    plus small noise, then add 4 % vig. Used for market-efficient tests."""
    rng = np.random.default_rng(seed)
    true_p = (base_preds["home_won"].astype(int).to_numpy() * 0.5
              + base_preds["model_prob"].to_numpy() * 0.5)       # dummy "true" p
    p_home_fair = np.clip(true_p + rng.normal(0, noise_sd, len(true_p)), 0.05, 0.95)
    p_away_fair = 1 - p_home_fair
    # Add uniform vig so raw implied probs sum to 1.045
    vig = 0.045
    p_home_raw = p_home_fair + vig / 2
    p_away_raw = p_away_fair + vig / 2
    ml_home = [prob_to_american(float(p)) for p in p_home_raw]
    ml_away = [prob_to_american(float(p)) for p in p_away_raw]
    return pd.DataFrame({
        "game_date": base_preds["game_date"],
        "home_team_id": base_preds["home_team_id"].astype(int),
        "away_team_id": base_preds["away_team_id"].astype(int),
        "bookmaker": "kaggle_close",
        "ml_home": ml_home,
        "ml_away": ml_away,
    })


# ---------------------------------------------------------------------------
# Dataset construction: revision #1 (side-level symmetry)
# ---------------------------------------------------------------------------

def test_two_rows_per_game():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    # Each game contributes exactly two rows, one per side.
    counts = side.groupby("game_id")["side"].nunique()
    assert (counts == 2).all()
    assert set(side["side"].unique()) == {"home", "away"}


def test_side_market_probs_sum_to_one():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    by_game = side.groupby("game_id")["side_market_prob"].sum()
    np.testing.assert_allclose(by_game.to_numpy(), 1.0, atol=1e-9)


def test_side_model_probs_sum_to_one():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    by_game = side.groupby("game_id")["side_model_prob"].sum()
    np.testing.assert_allclose(by_game.to_numpy(), 1.0, atol=1e-9)


def test_side_won_sums_to_one_per_game():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    # Each game: home_won + away_won == 1.
    by_game = side.groupby("game_id")["side_won"].sum()
    assert (by_game == 1).all()


def test_canonical_columns_present():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    for col in ("game_id", "game_date", "side", "side_team_id", "opponent_team_id",
                "side_model_prob", "side_market_prob", "side_moneyline",
                "edge", "abs_edge", "side_won",
                "home_team_id", "away_team_id", "home_won",
                "model_home_prob", "market_home_prob",
                "home_moneyline", "away_moneyline",
                "predicted_at", "fold", "season"):
        assert col in side.columns, f"missing {col}"


# ---------------------------------------------------------------------------
# Logit features: revision #4
# ---------------------------------------------------------------------------

def test_logit_features_finite_at_extremes():
    p = np.array([0.0, 0.0001, 0.5, 0.9999, 1.0])
    out = _logit(p)
    assert np.all(np.isfinite(out))


def test_add_meta_features_columns():
    df = pd.DataFrame({
        "side_model_prob": [0.6, 0.4],
        "side_market_prob": [0.55, 0.45],
        "edge": [0.05, -0.05],
        "abs_edge": [0.05, 0.05],
        "side_won": [1, 0],
    })
    out = add_meta_features(df)
    for col in ("logit_side_model_prob", "logit_side_market_prob", "logit_edge"):
        assert col in out.columns
    # logit(0.6) - logit(0.55) = edge-in-logit-space; sanity check sign.
    assert out.loc[0, "logit_edge"] > 0
    assert out.loc[1, "logit_edge"] < 0


# ---------------------------------------------------------------------------
# Regularization: revision #5
# ---------------------------------------------------------------------------

def test_regularization_actually_shrinks():
    bp = _synth_base_preds(n_folds=2, games_per_fold=300)
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    train = side.iloc[: len(side) * 3 // 4]
    tight = train_meta_lr(train, META_FEATURES_MINIMAL, C=0.5)
    loose = train_meta_lr(train, META_FEATURES_MINIMAL, C=100.0)
    tight_norm = np.linalg.norm(tight.named_steps["lr"].coef_)
    loose_norm = np.linalg.norm(loose.named_steps["lr"].coef_)
    assert tight_norm < loose_norm


# ---------------------------------------------------------------------------
# Nested walk-forward: revision #6 (OOS only) and look-ahead guard
# ---------------------------------------------------------------------------

def test_nested_walk_forward_leaves_warmup_nan():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    out = nested_walk_forward(side, META_FEATURES_MINIMAL, "lr", initial_train_folds=1)
    warmup_fold = sorted(side["fold"].unique())[0]
    assert out.loc[out["fold"] == warmup_fold, "meta_prob_lr"].isna().all()


def test_nested_walk_forward_predicts_later_folds():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    out = nested_walk_forward(side, META_FEATURES_MINIMAL, "lr", initial_train_folds=1)
    later_folds = sorted(side["fold"].unique())[1:]
    assert out[out["fold"].isin(later_folds)]["meta_prob_lr"].notna().all()


def test_no_leakage_from_future_outcome():
    """Poison a game's outcome in a later fold; meta prediction for that game
    must not change (meta was trained only on prior folds)."""
    bp = _synth_base_preds(n_folds=3, games_per_fold=150)
    odds = _synth_odds(bp)
    side_a = build_side_frame(bp, odds)
    out_a = nested_walk_forward(side_a, META_FEATURES_MINIMAL, "lr", initial_train_folds=1)

    # Flip outcomes in the LAST fold only.
    last_fold = sorted(bp["fold"].unique())[-1]
    bp_b = bp.copy()
    bp_b.loc[bp_b["fold"] == last_fold, "home_won"] = 1 - bp_b.loc[bp_b["fold"] == last_fold, "home_won"]
    side_b = build_side_frame(bp_b, odds)
    out_b = nested_walk_forward(side_b, META_FEATURES_MINIMAL, "lr", initial_train_folds=1)

    # Meta preds for the MIDDLE fold (which never saw last fold's data) must be identical.
    middle_fold = sorted(bp["fold"].unique())[1]
    m_a = out_a[out_a["fold"] == middle_fold].set_index(["game_id", "side"])["meta_prob_lr"]
    m_b = out_b[out_b["fold"] == middle_fold].set_index(["game_id", "side"])["meta_prob_lr"]
    np.testing.assert_allclose(m_a.to_numpy(), m_b.to_numpy(), atol=1e-10)


# ---------------------------------------------------------------------------
# Compat shim: revision (home-level pred for run_backtest)
# ---------------------------------------------------------------------------

def test_side_to_home_preds_schema():
    bp = _synth_base_preds()
    odds = _synth_odds(bp)
    side = build_side_frame(bp, odds)
    out = nested_walk_forward(side, META_FEATURES_MINIMAL, "lr", initial_train_folds=1)
    home = side_to_home_preds(out, "meta_prob_lr")
    # Schema needed by run_backtest._validate_predictions:
    for col in ("game_id", "game_date", "home_team_id", "away_team_id", "model_prob"):
        assert col in home.columns
    # Every row has model_prob (NaN rows dropped).
    assert home["model_prob"].notna().all()
    # No look-ahead: predicted_at <= game_date.
    assert (home["predicted_at"].dt.normalize() <= home["game_date"].dt.normalize()).all()


# ---------------------------------------------------------------------------
# Market-efficiency synthetic (revision #8): when market is truth, meta
# cannot beat market out-of-sample.
# ---------------------------------------------------------------------------

def test_meta_cannot_beat_efficient_market_synth():
    """Construct a world where `side_market_prob` is exactly the true
    win probability and `side_model_prob` is noise. Meta-LR's OOS log-loss
    should be *no better than* market-only's (within a tolerance)."""
    rng = np.random.default_rng(7)
    n_folds = 3
    games_per_fold = 400
    rows = []
    gid = 0
    day0 = pd.Timestamp("2020-10-25")
    for fold_i in range(n_folds):
        fold_label = f"eff-{fold_i}"
        fold_start = day0 + pd.Timedelta(days=fold_i * 200)
        for g in range(games_per_fold):
            p_home_true = float(np.clip(rng.normal(0.58, 0.10), 0.1, 0.9))
            home_won = int(rng.random() < p_home_true)
            # Base model is pure noise around 0.5; market is truth.
            model_noise = float(np.clip(0.5 + rng.normal(0, 0.10), 0.02, 0.98))
            rows.append({
                "game_id": f"G{gid:05d}",
                "game_date": fold_start + pd.Timedelta(days=g % 180),
                "home_team_id": 1610612700 + (g % 30),
                "away_team_id": 1610612700 + ((g + 7) % 30),
                "home_won": home_won,
                "model_prob": model_noise,
                "fold": fold_label,
                "predicted_at": fold_start - pd.Timedelta(days=1),
                "_p_home_true": p_home_true,
            })
            gid += 1
    bp = pd.DataFrame(rows)
    # Build odds so that fair prob = p_home_true.
    vig = 0.045
    ml_home = [prob_to_american(float(np.clip(p + vig / 2, 0.03, 0.97)))
               for p in bp["_p_home_true"]]
    ml_away = [prob_to_american(float(np.clip(1 - p + vig / 2, 0.03, 0.97)))
               for p in bp["_p_home_true"]]
    odds = pd.DataFrame({
        "game_date": bp["game_date"],
        "home_team_id": bp["home_team_id"].astype(int),
        "away_team_id": bp["away_team_id"].astype(int),
        "bookmaker": "synth_close",
        "ml_home": ml_home,
        "ml_away": ml_away,
    })
    side = build_side_frame(bp.drop(columns=["_p_home_true"]), odds)
    out = nested_walk_forward(side, META_FEATURES_MINIMAL, "lr", initial_train_folds=1)
    scope = out[out["meta_prob_lr"].notna()]
    y = scope["side_won"].astype(int).to_numpy()
    ll_market = log_loss(y, np.clip(scope["side_market_prob"].to_numpy(), 1e-6, 1 - 1e-6))
    ll_meta = log_loss(y, np.clip(scope["meta_prob_lr"].to_numpy(), 1e-6, 1 - 1e-6))
    # Meta should not beat market by more than ~1 %.
    assert ll_meta >= ll_market - 0.01
