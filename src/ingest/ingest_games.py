"""Ingest historical NBA game logs from nba_api into PostgreSQL.

Pulls one row per team per game (the natural grain returned by LeagueGameLog)
across a configurable range of seasons and upserts into the `team_games` table.
Safe to re-run daily — existing rows are updated rather than duplicated.

Usage:
    python -m src.ingest.ingest_games --start-season 2016 --end-season 2024
    python -m src.ingest.ingest_games --start-season 2023 --end-season 2024 --season-type Playoffs
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
    Boolean,
    Date,
    Float,
    Integer,
    String,
    DateTime,
    func,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Mapped, mapped_column

from db import Base, SessionLocal, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ingest_games")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_TYPES = ("Regular Season", "Playoffs", "Pre Season", "All Star")
REQUEST_PAUSE_SECONDS = 1.0


class TeamGame(Base):
    """One row per team per game (LeagueGameLog grain)."""

    __tablename__ = "team_games"

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    season: Mapped[str] = mapped_column(String(10), index=True)
    season_type: Mapped[str] = mapped_column(String(20), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)

    team_abbreviation: Mapped[str] = mapped_column(String(5))
    team_name: Mapped[str] = mapped_column(String(64))
    matchup: Mapped[str] = mapped_column(String(32))
    is_home: Mapped[bool] = mapped_column(Boolean)
    wl: Mapped[str | None] = mapped_column(String(1), nullable=True)

    min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fgm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fga: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fg3m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fg3a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fg3_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ftm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fta: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ft_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    oreb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dreb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ast: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stl: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blk: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tov: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    plus_minus: Mapped[float | None] = mapped_column(Float, nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def season_label(start_year: int) -> str:
    """Convert a starting calendar year (e.g. 2023) into nba_api format ('2023-24')."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season_label(today: date | None = None) -> str:
    """NBA seasons start in October. Oct-Dec belongs to that year's season,
    Jan-Sep to the prior year's."""
    d = today or date.today()
    y = d.year if d.month >= 10 else d.year - 1
    return season_label(y)


def ingest_season(season: str, season_type: str = "Regular Season") -> int:
    """Fetch one season + season_type and upsert. Returns # rows upserted.
    Shared between the CLI and the daily orchestrator."""
    raw = fetch_season(season, season_type)
    records = transform(raw, season, season_type)
    return upsert(records)


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    log.info("Fetching %s (%s)...", season, season_type)
    endpoint = LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
        timeout=60,
    )
    df = endpoint.get_data_frames()[0]
    log.info("  -> %d rows", len(df))
    return df


def transform(df: pd.DataFrame, season: str, season_type: str) -> list[dict]:
    if df.empty:
        return []
    df = df.rename(columns=str.lower).copy()
    df["season"] = season
    df["season_type"] = season_type
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["is_home"] = ~df["matchup"].str.contains("@")

    columns = {c.name for c in TeamGame.__table__.columns} - {"ingested_at"}
    keep = [c for c in df.columns if c in columns]
    return df[keep].where(pd.notna(df[keep]), None).to_dict(orient="records")


def upsert(records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0
    stmt = pg_insert(TeamGame).values(records)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in TeamGame.__table__.columns
        if c.name not in ("game_id", "team_id", "ingested_at")
    }
    update_cols["ingested_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        index_elements=["game_id", "team_id"], set_=update_cols
    )
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()
    return len(records)


def backup_csv(df: pd.DataFrame, season: str, season_type: str) -> None:
    if df.empty:
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    safe_type = season_type.lower().replace(" ", "_")
    path = RAW_DIR / f"team_games_{season}_{safe_type}_{stamp}.csv"
    df.to_csv(path, index=False)
    log.info("  -> CSV backup: %s", path)


def parse_args() -> argparse.Namespace:
    current_year = date.today().year
    p = argparse.ArgumentParser(description="Ingest NBA team-game logs into PostgreSQL.")
    p.add_argument("--start-season", type=int, default=2016,
                   help="Starting calendar year (e.g. 2016 = '2016-17').")
    p.add_argument("--end-season", type=int, default=current_year,
                   help="Ending calendar year, inclusive.")
    p.add_argument("--season-type", choices=SEASON_TYPES, default="Regular Season")
    p.add_argument("--csv-backup", action="store_true",
                   help="Also write a timestamped CSV to data/raw/ per season.")
    p.add_argument("--create-tables", action="store_true",
                   help="Create tables if they don't exist before ingesting.")
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
            raw = fetch_season(season, args.season_type)
        except Exception as exc:
            log.error("Failed to fetch %s: %s", season, exc)
            continue

        if args.csv_backup:
            backup_csv(raw, season, args.season_type)

        records = transform(raw, season, args.season_type)
        n = upsert(records)
        log.info("  -> upserted %d rows", n)
        total += n

        time.sleep(REQUEST_PAUSE_SECONDS)

    log.info("Done. Upserted %d total rows across %d seasons.",
             total, args.end_season - args.start_season + 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
