"""Streamlit dashboard for the NBA prediction engine.

Launch with:
    streamlit run dashboard/app.py

Five tabs:
  1. Today's Predictions  — model-implied win probabilities for upcoming games
  2. Historical Accuracy   — rolling accuracy + reliability curve on past predictions
  3. Edge Finder           — model probs vs. market implied probs (from The Odds API)
  4. Elo Leaderboard       — current team ratings + recent form
  5. Backtest              — bankroll simulation over walk-forward predictions + historical odds
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from db import ...` when Streamlit launches the script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import teams as nba_static_teams
from sqlalchemy import text

from db import engine
from src.features.elo import add_elo, final_elos
from src.features.matchup import load_team_games, to_matchup
from src.features.odds_math import moneyline_to_fair_prob
from src.finance.backtest import BacktestConfig, run_backtest
from src.finance.staking import build as build_strategy
from src.models.calibration import expected_calibration_error, reliability_table
from src.models.walkforward import make_xgb_predict_fn, walk_forward_predictions
from src.models.xgboost_model import select_features

st.set_page_config(page_title="CourtIQ", page_icon="🏀", layout="wide")


# ---------- cached loaders ----------

@st.cache_resource
def team_lookup() -> dict[int, dict]:
    return {t["id"]: t for t in nba_static_teams.get_teams()}


@st.cache_data(ttl=120)
def load_predictions() -> pd.DataFrame:
    try:
        df = pd.read_sql_query(text("SELECT * FROM predictions"), engine)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@st.cache_data(ttl=300)
def load_matchups_with_outcomes() -> pd.DataFrame:
    df = load_team_games(engine)
    m = to_matchup(df)
    return m[["game_id", "game_date", "season", "home_team_id", "away_team_id", "home_won"]].copy()


@st.cache_data(ttl=300)
def load_latest_odds() -> pd.DataFrame:
    try:
        df = pd.read_sql_query(text("SELECT * FROM odds_snapshots"), engine)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    # Keep only the latest snapshot per (event, bookmaker).
    return (
        df.sort_values("fetched_at")
        .groupby(["event_id", "bookmaker"], as_index=False)
        .tail(1)
    )


@st.cache_data(ttl=600)
def current_elos() -> pd.DataFrame:
    df = load_team_games(engine)
    m = to_matchup(df)
    m = add_elo(m)
    ratings, last_season = final_elos(m)
    lookup = team_lookup()
    rows = []
    for tid, r in ratings.items():
        t = lookup.get(tid, {})
        rows.append({
            "team_id": tid,
            "team": t.get("full_name", str(tid)),
            "abbrev": t.get("abbreviation", "???"),
            "elo": round(r, 1),
            "last_season": last_season.get(tid),
        })
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def abbrev(team_id: int) -> str:
    return team_lookup().get(int(team_id), {}).get("abbreviation", str(team_id))


# ---------- sidebar ----------

st.sidebar.title("🏀 CourtIQ")
st.sidebar.caption("NBA win-probability engine")
predictions = load_predictions()
odds = load_latest_odds()
st.sidebar.metric("Predictions in DB", len(predictions))
st.sidebar.metric("Odds snapshots", len(odds))
st.sidebar.markdown(
    "**Run the daily job**\n\n"
    "`python -m src.pipeline.daily_update --model-version v1`"
)

tab_today, tab_history, tab_edge, tab_elo, tab_backtest = st.tabs(
    ["Today's Predictions", "Historical Accuracy", "Edge Finder", "Elo Leaderboard", "Backtest"]
)


# ---------- Tab 1: today's predictions ----------

with tab_today:
    st.header("Model-implied win probabilities")
    if predictions.empty:
        st.info("No predictions yet. Run `python -m src.pipeline.daily_update --model-version v1`.")
    else:
        dates = sorted(predictions["game_date"].dt.date.unique(), reverse=True)
        selected = st.selectbox("Date", dates, index=0)
        day = predictions[predictions["game_date"].dt.date == selected].copy()
        day["home"] = day["home_team_id"].apply(abbrev)
        day["away"] = day["away_team_id"].apply(abbrev)
        day["matchup"] = day["away"] + " @ " + day["home"]
        day["favorite"] = np.where(day["model_prob"] >= 0.5, day["home"], day["away"])
        day["fav_prob"] = np.where(day["model_prob"] >= 0.5, day["model_prob"], 1 - day["model_prob"])
        view = day[["matchup", "model_prob", "favorite", "fav_prob", "predicted_at"]].rename(
            columns={"model_prob": "P(home win)", "fav_prob": "P(favorite)"}
        ).sort_values("P(home win)", ascending=False)
        st.dataframe(
            view,
            column_config={
                "P(home win)": st.column_config.ProgressColumn(
                    "P(home win)", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "P(favorite)": st.column_config.NumberColumn(format="%.3f"),
            },
            hide_index=True,
            width="stretch",
        )


# ---------- Tab 2: historical accuracy + calibration ----------

with tab_history:
    st.header("Out-of-sample performance")
    outcomes = load_matchups_with_outcomes()
    joined = predictions.merge(
        outcomes[["game_id", "home_won"]], on="game_id", how="inner"
    )
    if joined.empty:
        st.info("No predictions matched to outcomes yet. After games finish and the next "
                "daily job runs, accuracy and calibration will populate here.")
    else:
        joined["correct"] = ((joined["model_prob"] >= 0.5).astype(int) == joined["home_won"]).astype(int)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Games graded", len(joined))
        col_b.metric("Accuracy", f"{joined['correct'].mean():.3f}")
        y, p = joined["home_won"].to_numpy(), np.clip(joined["model_prob"].to_numpy(), 1e-6, 1 - 1e-6)
        logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        col_c.metric("Log loss", f"{logloss:.3f}")

        st.subheader("Reliability diagram")
        tbl = reliability_table(joined["home_won"], joined["model_prob"], n_bins=10)
        valid = tbl[tbl["n"] > 0]
        chart = pd.DataFrame({
            "predicted": valid["mean_predicted"].tolist() + [0.0, 1.0],
            "actual": valid["actual_rate"].tolist() + [0.0, 1.0],
            "series": ["model"] * len(valid) + ["perfect", "perfect"],
        })
        st.vega_lite_chart({
            "data": {"values": chart.to_dict(orient="records")},
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": "predicted", "type": "quantitative", "scale": {"domain": [0, 1]}},
                "y": {"field": "actual", "type": "quantitative", "scale": {"domain": [0, 1]}},
                "color": {"field": "series", "type": "nominal"},
            },
            "height": 400,
        }, width="stretch")
        st.caption(f"Expected Calibration Error (ECE) = {expected_calibration_error(tbl):.3f}")

        st.subheader("Per-bin reliability table")
        st.dataframe(
            tbl[tbl["n"] > 0].round({"mean_predicted": 3, "actual_rate": 3}),
            hide_index=True, width="stretch",
        )


# ---------- Tab 3: edge finder ----------

with tab_edge:
    st.header("Model vs. Market")
    if odds.empty:
        st.info(
            "No odds ingested yet. Set `ODDS_API_KEY` in `.env` and run "
            "`python -m src.ingest.ingest_odds` (or the daily job) to populate this tab."
        )
    elif predictions.empty:
        st.info("No predictions to compare against.")
    else:
        o = odds.dropna(subset=["ml_home", "ml_away"]).copy()
        o[["market_prob_home", "market_prob_away"]] = o.apply(
            lambda r: pd.Series(moneyline_to_fair_prob(r["ml_home"], r["ml_away"])),
            axis=1,
        )
        merged = predictions.merge(
            o[["game_date", "home_team_id", "away_team_id", "bookmaker",
               "ml_home", "ml_away", "market_prob_home"]],
            on=["game_date", "home_team_id", "away_team_id"],
            how="inner",
        )
        if merged.empty:
            st.warning("Odds and predictions exist but couldn't be joined (team-name "
                       "mapping mismatch or date offset). Check team_map.py and the "
                       "game_date conversion.")
        else:
            merged["edge"] = merged["model_prob"] - merged["market_prob_home"]
            merged["home"] = merged["home_team_id"].apply(abbrev)
            merged["away"] = merged["away_team_id"].apply(abbrev)
            merged["matchup"] = merged["away"] + " @ " + merged["home"]
            view = merged.sort_values("edge", ascending=False)[
                ["matchup", "bookmaker", "model_prob", "market_prob_home", "edge", "ml_home", "ml_away"]
            ].rename(columns={
                "model_prob": "model P(home)",
                "market_prob_home": "market P(home)",
            })
            st.dataframe(view, hide_index=True, width="stretch")


# ---------- Tab 4: elo leaderboard ----------

with tab_elo:
    st.header("Current team Elo ratings")
    elo_df = current_elos()
    col_top, col_bot = st.columns(2)
    with col_top:
        st.subheader("Top 10")
        st.dataframe(elo_df.head(10)[["team", "abbrev", "elo"]],
                     hide_index=True, width="stretch")
    with col_bot:
        st.subheader("Bottom 10")
        st.dataframe(elo_df.tail(10)[["team", "abbrev", "elo"]][::-1],
                     hide_index=True, width="stretch")
    st.subheader("All 30 teams")
    st.bar_chart(elo_df.set_index("abbrev")["elo"])


# ---------- Tab 5: backtest ----------

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


@st.cache_data(ttl=3600)
def _walk_forward_preds(initial_train: int, mode: str, window: int) -> pd.DataFrame:
    """Cache the walk-forward run — it trains N XGBoost models and takes a while."""
    df = pd.read_parquet(FEATURES_PATH)
    features = select_features(df)
    return walk_forward_predictions(
        df, make_xgb_predict_fn(features),
        initial_train_seasons=initial_train, mode=mode, window=window,
    )


@st.cache_data(ttl=300)
def _historical_odds() -> pd.DataFrame:
    try:
        df = pd.read_sql_query(
            text("SELECT game_date, home_team_id, away_team_id, bookmaker, ml_home, ml_away "
                 "FROM odds_snapshots WHERE ml_home IS NOT NULL AND ml_away IS NOT NULL"),
            engine,
        )
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@st.cache_data(ttl=600)
def _outcomes() -> pd.DataFrame:
    raw = load_team_games(engine)
    return (
        raw[raw["is_home"]][["game_id", "won"]]
        .rename(columns={"won": "home_won"})
        .drop_duplicates(subset="game_id")
    )


with tab_backtest:
    st.header("Bankroll simulation")
    odds_hist = _historical_odds()
    if odds_hist.empty:
        st.info(
            "No historical odds in the database. Import a Kaggle NBA odds CSV:\n\n"
            "`python -m src.ingest.ingest_historical_odds --inspect path/to/file.csv`\n\n"
            "then build a column map and run with `--csv ... --column-map ...`."
        )
    else:
        col1, col2, col3 = st.columns(3)
        strategy_kind = col1.selectbox(
            "Strategy",
            options=["fractional_kelly", "threshold_kelly", "full_kelly",
                     "capped_kelly", "fixed_fractional", "flat"],
            index=0,
        )
        bankroll = col1.number_input("Starting bankroll ($)", value=10_000, step=1000, min_value=100)

        multiplier = col2.slider("Kelly multiplier", 0.05, 1.00, 0.25, step=0.05)
        min_edge = col2.slider("Min edge (backtest-level)", 0.00, 0.10, 0.02, step=0.005)

        max_conc = col3.slider("Max concurrent exposure", 0.10, 1.00, 0.50, step=0.05)
        side = col3.selectbox("Side", ["best", "home", "away"], index=0)

        cfg_dict: dict = {"type": strategy_kind}
        if strategy_kind == "flat":
            cfg_dict["unit"] = col1.number_input("Flat unit ($)", value=100.0, step=10.0)
        elif strategy_kind == "fixed_fractional":
            cfg_dict["pct"] = col1.slider("Fixed pct", 0.005, 0.10, 0.02, step=0.005)
        elif strategy_kind in ("fractional_kelly", "threshold_kelly", "capped_kelly"):
            cfg_dict["multiplier"] = multiplier
            if strategy_kind == "threshold_kelly":
                cfg_dict["min_edge"] = col2.slider("Strategy-level min edge", 0.00, 0.10, 0.02, step=0.005)
            if strategy_kind == "capped_kelly":
                cfg_dict["max_fraction"] = col2.slider("Per-bet cap", 0.01, 0.25, 0.05, step=0.01)

        if st.button("Run backtest", type="primary"):
            with st.spinner("Training walk-forward predictions and replaying games..."):
                preds = _walk_forward_preds(initial_train=3, mode="expanding", window=3)
                outcomes = _outcomes()
                strategy = build_strategy(cfg_dict)
                cfg = BacktestConfig(
                    strategy=strategy,
                    starting_bankroll=float(bankroll),
                    min_edge=float(min_edge),
                    side=side,
                    bookmaker=None,
                    max_concurrent_exposure=float(max_conc),
                )
                result = run_backtest(preds, odds_hist, outcomes, cfg)

            s = result.summary
            st.subheader("Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROI", f"{s['roi']:+.2%}")
            m1.metric("Hit rate", f"{s['hit_rate']:.2%}" if s["hit_rate"] == s["hit_rate"] else "—")
            m2.metric("Sharpe", f"{s['sharpe']:.2f}")
            m2.metric("Sortino", f"{s['sortino']:.2f}")
            m3.metric("Max drawdown", f"{s['max_drawdown']:.2%}")
            m3.metric("Calmar", f"{s['calmar']:.2f}" if np.isfinite(s["calmar"]) else "∞")
            m4.metric("# bets", f"{s['n_bets']}")
            m4.metric("Avg edge", f"{s['avg_edge']:+.3f}")

            st.subheader("Equity curve")
            eq_df = result.equity.reset_index()
            eq_df.columns = ["date", "bankroll"]
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            st.line_chart(eq_df.set_index("date")["bankroll"])

            if not result.bets.empty:
                st.subheader("Drawdown curve")
                eq = result.equity
                dd = eq / eq.cummax() - 1
                dd_df = dd.reset_index()
                dd_df.columns = ["date", "drawdown"]
                dd_df["date"] = pd.to_datetime(dd_df["date"])
                st.area_chart(dd_df.set_index("date")["drawdown"])

                st.subheader("Stake size distribution")
                st.bar_chart(result.bets["stake"].round(0).value_counts().sort_index())

                st.subheader("Bets")
                view_cols = [
                    "game_date", "game_id", "side", "entry_ml", "model_prob",
                    "edge", "stake", "kelly_fraction_used", "status", "payout",
                ]
                st.dataframe(result.bets[view_cols].round({
                    "model_prob": 3, "edge": 3, "stake": 2,
                    "kelly_fraction_used": 4, "payout": 2,
                }), hide_index=True, width="stretch")
            else:
                st.info("No bets placed — try lowering min edge or switching strategy.")
