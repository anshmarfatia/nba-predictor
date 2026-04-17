"""Score upcoming NBA games using a saved model.

Flow:
  1. Resolve the target date (default: today).
  2. Pull scheduled matchups for that date from nba_api's ScoreboardV2.
  3. Build per-team "state-as-of-date" features from the historical tables
     (rolling stats, rest, Elo) using only games strictly before the date —
     this is the critical leakage discipline for prediction at serving time.
  4. Load the saved XGBoost model + feature manifest.
  5. Score each game, upsert one row per (game, model_version) into
     the `predictions` table.

The same table is what `market_compare.py` joins against to compute edge.

Usage:
    python -m src.pipeline.predict --save v1 --create-tables     # first run
    python -m src.pipeline.predict --save v1                      # today
    python -m src.pipeline.predict --save v1 --date 2024-04-01    # backtest
    python -m src.pipeline.predict --save v1 --date 2024-04-01 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2
from sqlalchemy import BigInteger, Date, DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Mapped, mapped_column
from xgboost import XGBClassifier

from sqlalchemy import text

from db import Base, SessionLocal, engine
from src.features.elo import add_elo, final_elos, pre_game_elo
from src.features.matchup import load_team_games, to_matchup
from src.features.player_features import add_player_features
from src.features.rolling_stats import add_rolling
from src.features.rest_features import add_rest
from src.features.situational import COVID_SEASONS, add_situational

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict")

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
ROLLING_SUFFIXES = ("_r5", "_r10", "_r20")


class Prediction(Base):
    """One row per (game, model_version). Upsert on re-runs so the latest
    probability always overwrites stale ones."""

    __tablename__ = "predictions"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    home_team_id: Mapped[int] = mapped_column(BigInteger, index=True)
    away_team_id: Mapped[int] = mapped_column(BigInteger, index=True)
    model_prob: Mapped[float] = mapped_column(Float)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def season_from_date(d: date) -> str:
    """NBA seasons run Oct–Jun. A date in Oct/Nov/Dec belongs to that year's
    season ('2026-27'); a date in Jan–Jun belongs to the PRIOR year's."""
    year = d.year if d.month >= 10 else d.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def load_model(version: str) -> tuple[XGBClassifier, list[str]]:
    model_path = MODELS_DIR / f"{version}.json"
    manifest_path = MODELS_DIR / f"{version}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model = XGBClassifier()
    model.load_model(model_path)
    log.info("Loaded model %s (trained %s, %d features)",
             version, manifest.get("saved_at"), len(manifest["features"]))
    return model, manifest["features"]


def fetch_scoreboard(as_of: date) -> pd.DataFrame:
    """Today's scheduled games. ScoreboardV2's GameHeader frame has the IDs
    and matchup we need; we ignore teams that don't resolve (non-NBA entries
    shouldn't appear but guard anyway)."""
    sb = ScoreboardV2(game_date=as_of.strftime("%m/%d/%Y"), timeout=30)
    header = sb.get_data_frames()[0]
    if header.empty:
        return pd.DataFrame(columns=["game_id", "home_team_id", "away_team_id", "game_date"])
    out = header.rename(columns=str.lower)[["game_id", "home_team_id", "visitor_team_id"]]
    out = out.rename(columns={"visitor_team_id": "away_team_id"})
    out["game_date"] = as_of
    return out


PLAYER_FEAT_COLS = (
    "rotation_avg_minutes_recent",
    "top1_active_recent",
    "top3_active_recent",
    "rotation_minutes_share_recent",
)


def team_state_as_of(as_of: date) -> pd.DataFrame:
    """Per-team latest feature row, computed only from games strictly before
    `as_of`. This is what we stitch into upcoming-game matchup rows."""
    raw = load_team_games(engine)
    raw = raw[raw["game_date"] < pd.Timestamp(as_of)]
    raw = add_rolling(raw)
    raw = add_rest(raw)
    raw = add_situational(raw)
    raw = raw.sort_values(["team_id", "game_date", "game_id"])
    return raw.groupby("team_id", as_index=False).tail(1)


def player_state_as_of(team_ids: list[int], season: str, as_of: date) -> dict[int, dict]:
    """Compute player-availability features for each team as if they were about
    to play a game on `as_of`. Builds phantom team-game rows at as_of and reuses
    `add_player_features` so the leakage discipline is identical to training."""
    if not team_ids:
        return {}
    phantom = pd.DataFrame([
        {
            "team_id": int(tid),
            "season": season,
            "game_id": f"phantom-{tid}",
            "game_date": pd.Timestamp(as_of),
        }
        for tid in team_ids
    ])
    with engine.connect() as conn:
        pg = pd.read_sql_query(
            text("SELECT game_id, player_id, team_id, season, game_date, minutes "
                 "FROM player_games "
                 "WHERE team_id = ANY(:tids) AND game_date < :as_of"),
            conn,
            params={"tids": [int(t) for t in team_ids], "as_of": as_of},
        )
    if pg.empty:
        return {int(tid): dict.fromkeys(PLAYER_FEAT_COLS, 0.0) for tid in team_ids}

    feats = add_player_features(phantom, pg)
    return feats.set_index("team_id")[list(PLAYER_FEAT_COLS)].to_dict("index")


def build_upcoming_matchups(scoreboard: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Construct matchup rows with features computed as-of `as_of`. Feature
    layout exactly mirrors the training-time frame produced by build_training_set."""
    if scoreboard.empty:
        return scoreboard

    state = team_state_as_of(as_of).set_index("team_id")

    raw = load_team_games(engine)
    raw = raw[raw["game_date"] < pd.Timestamp(as_of)]
    matchup_hist = to_matchup(raw)
    matchup_hist = add_elo(matchup_hist)
    ratings, last_season = final_elos(matchup_hist)

    game_season = season_from_date(as_of)
    month = as_of.month
    is_covid = int(game_season in COVID_SEASONS)

    team_ids_today = sorted(set(scoreboard["home_team_id"].astype(int)) |
                            set(scoreboard["away_team_id"].astype(int)))
    player_state = player_state_as_of(team_ids_today, game_season, as_of)

    rows: list[dict] = []
    for _, g in scoreboard.iterrows():
        h, a = int(g["home_team_id"]), int(g["away_team_id"])
        if h not in state.index or a not in state.index:
            log.warning("Missing state for team_id %s or %s — skipping game %s",
                        h, a, g["game_id"])
            continue
        hs, asr = state.loc[h], state.loc[a]

        home_elo = pre_game_elo(h, game_season, ratings, last_season)
        away_elo = pre_game_elo(a, game_season, ratings, last_season)
        home_rest = (pd.Timestamp(as_of) - hs["game_date"]).days
        away_rest = (pd.Timestamp(as_of) - asr["game_date"]).days

        row = {
            "game_id": g["game_id"],
            "game_date": as_of,
            "season": game_season,
            "home_team_id": h,
            "away_team_id": a,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
            "rest_diff": home_rest - away_rest,
            "home_is_b2b": int(home_rest == 1),
            "away_is_b2b": int(away_rest == 1),
            "home_game_number": int(hs["game_number"]) + 1,
            "away_game_number": int(asr["game_number"]) + 1,
            "home_season_progress": (int(hs["game_number"]) + 1) / 82.0,
            "away_season_progress": (int(asr["game_number"]) + 1) / 82.0,
            "home_month": month,
            "home_is_covid_season": is_covid,
        }
        # Copy all rolling columns from each side's state row.
        for col in state.columns:
            if any(col.endswith(s) for s in ROLLING_SUFFIXES):
                row[f"home_{col}"] = hs[col]
                row[f"away_{col}"] = asr[col]

        # Player-availability features computed at as-of date for both teams.
        h_pf = player_state.get(h, dict.fromkeys(PLAYER_FEAT_COLS, 0.0))
        a_pf = player_state.get(a, dict.fromkeys(PLAYER_FEAT_COLS, 0.0))
        for col in PLAYER_FEAT_COLS:
            row[f"home_{col}"] = h_pf.get(col, 0.0)
            row[f"away_{col}"] = a_pf.get(col, 0.0)

        rows.append(row)
    return pd.DataFrame(rows)


def upsert(records: list[dict]) -> int:
    if not records:
        return 0
    stmt = pg_insert(Prediction).values(records)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in Prediction.__table__.columns
        if c.name not in ("game_id", "model_version", "predicted_at")
    }
    update_cols["predicted_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        index_elements=["game_id", "model_version"], set_=update_cols
    )
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()
    return len(records)


def predict_for_date(
    as_of: date,
    model_version: str,
    dry_run: bool = False,
    scoreboard: pd.DataFrame | None = None,
) -> int:
    """Score all NBA games on `as_of` with the given saved model.

    If `scoreboard` is None, fetches live from ScoreboardV2 (today's / recent
    games). Callers doing a backfill pass reconstruct the scoreboard from the
    already-ingested `team_games` table and inject it here — that avoids 170+
    live API calls over a full season.
    """
    model, features = load_model(model_version)
    if scoreboard is None:
        scoreboard = fetch_scoreboard(as_of)
    if scoreboard.empty:
        log.info("No games scheduled on %s.", as_of)
        return 0
    upcoming = build_upcoming_matchups(scoreboard, as_of)
    if upcoming.empty:
        log.warning("No upcoming matchups could be built.")
        return 0

    missing = [f for f in features if f not in upcoming.columns]
    if missing:
        log.warning("Missing %d feature columns at inference: %s", len(missing), missing[:5])
        for m in missing:
            upcoming[m] = np.nan

    probs = model.predict_proba(upcoming[features])[:, 1]
    records = [
        {
            "game_id": row["game_id"],
            "model_version": model_version,
            "game_date": row["game_date"],
            "home_team_id": int(row["home_team_id"]),
            "away_team_id": int(row["away_team_id"]),
            "model_prob": float(p),
        }
        for (_, row), p in zip(upcoming.iterrows(), probs)
    ]
    for r in records:
        log.info("  %s  home=%s away=%s  P(home)=%.3f",
                 r["game_id"], r["home_team_id"], r["away_team_id"], r["model_prob"])
    if dry_run:
        return len(records)
    return upsert(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score NBA games for a given date.")
    p.add_argument("--save", required=True, help="Model version to load (e.g. 'v1').")
    p.add_argument("--date", type=lambda s: datetime.fromisoformat(s).date(),
                   default=date.today(),
                   help="Target date (YYYY-MM-DD). Default: today.")
    p.add_argument("--create-tables", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print predictions; don't write to DB.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.create_tables:
        Base.metadata.create_all(engine)
    log.info("Scoring %s (model=%s)...", args.date, args.save)
    n = predict_for_date(args.date, args.save, dry_run=args.dry_run)
    verb = "Would write" if args.dry_run else "Upserted"
    log.info("%s %d predictions.", verb, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
