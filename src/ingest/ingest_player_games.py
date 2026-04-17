"""Ingest per-player-per-game logs from nba_api into PostgreSQL.

One row per (player, game). Stored grain is intentionally minimal — only the
columns the feature layer actually needs (player_id, team_id, minutes, pts,
plus_minus) — to keep the table compact across 270K+ rows.

Built on `LeagueGameLog(player_or_team_abbreviation='P')`, which mirrors the
team-grain endpoint already used in ingest_games.py, so the upsert flow,
season looping, and CSV-backup logic follow the same shape.

Usage:
    python -m src.ingest.ingest_player_games --create-tables --start-season 2016 --end-season 2024
    python -m src.ingest.ingest_player_games --start-season 2024 --end-season 2024
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Mapped, mapped_column

from db import Base, SessionLocal, engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_player_games")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_TYPES = ("Regular Season", "Playoffs", "Pre Season", "All Star")
REQUEST_PAUSE_SECONDS = 1.5


class PlayerGame(Base):
    """One row per player per game.

    Joins to `team_games` on (game_id, team_id) — same matchup grain we use
    for everything else. `minutes` is the primary feature signal (zero =
    didn't play / DNP, low values = injured/limited)."""

    __tablename__ = "player_games"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    player_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    team_id: Mapped[int] = mapped_column(BigInteger, index=True)
    season: Mapped[str] = mapped_column(String(10), index=True)
    season_type: Mapped[str] = mapped_column(String(20), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)

    player_name: Mapped[str] = mapped_column(String(64))

    minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    pts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ast: Mapped[int | None] = mapped_column(Integer, nullable=True)
    plus_minus: Mapped[float | None] = mapped_column(Float, nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    log.info("Fetching player-game logs for %s (%s)...", season, season_type)
    endpoint = LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
        timeout=120,
    )
    df = endpoint.get_data_frames()[0]
    log.info("  -> %d rows", len(df))
    return df


def _parse_minutes(val) -> float | None:
    """nba_api sometimes returns minutes as float, sometimes as 'MM:SS' string."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) + float(ss) / 60.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def transform(df: pd.DataFrame, season: str, season_type: str) -> list[dict]:
    if df.empty:
        return []
    df = df.rename(columns=str.lower).copy()
    df["season"] = season
    df["season_type"] = season_type
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["minutes"] = df["min"].map(_parse_minutes)

    keep_cols = [
        "game_id", "player_id", "team_id", "season", "season_type", "game_date",
        "player_name", "minutes", "pts", "reb", "ast", "plus_minus",
    ]
    df = df[keep_cols].where(pd.notna(df[keep_cols]), None)
    return df.to_dict(orient="records")


def upsert(records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0
    stmt = pg_insert(PlayerGame).values(records)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in PlayerGame.__table__.columns
        if c.name not in ("game_id", "player_id", "ingested_at")
    }
    update_cols["ingested_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        index_elements=["game_id", "player_id"], set_=update_cols
    )
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()
    return len(records)


def ingest_season(season: str, season_type: str = "Regular Season") -> int:
    """Fetch one season + season_type and upsert. Shared by the CLI and the
    daily orchestrator (when player ingestion is added there)."""
    raw = fetch_season(season, season_type)
    records = transform(raw, season, season_type)
    return upsert(records)


def parse_args() -> argparse.Namespace:
    current_year = date.today().year
    p = argparse.ArgumentParser(description="Ingest NBA player-game logs.")
    p.add_argument("--start-season", type=int, default=2016)
    p.add_argument("--end-season", type=int, default=current_year)
    p.add_argument("--season-type", choices=SEASON_TYPES, default="Regular Season")
    p.add_argument("--create-tables", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.create_tables:
        log.info("Creating tables (if not exists)...")
        Base.metadata.create_all(engine)

    total = 0
    for year in range(args.start_season, args.end_season + 1):
        season = season_label(year)
        try:
            n = ingest_season(season, args.season_type)
            log.info("  -> upserted %d rows", n)
            total += n
        except Exception as exc:
            log.error("Failed to fetch %s: %s", season, exc)
            continue
        time.sleep(REQUEST_PAUSE_SECONDS)

    log.info("Done. Upserted %d total rows across %d seasons.",
             total, args.end_season - args.start_season + 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
