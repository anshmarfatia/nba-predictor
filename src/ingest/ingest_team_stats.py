"""Ingest per-game advanced team stats from nba_api into PostgreSQL.

Uses TeamGameLogs with MeasureType='Advanced' to get one row per team per game
of advanced metrics (ORtg, DRtg, Net Rating, Pace, TS%, eFG%, rebound rates,
turnover rate, assist ratio, PIE, etc.). Same grain as `team_games`, joined on
(game_id, team_id). Safe to re-run daily — uses Postgres UPSERT.

Usage:
    python -m src.ingest.ingest_team_stats --create-tables --start-season 2016 --end-season 2024
    python -m src.ingest.ingest_team_stats --start-season 2024 --end-season 2024 --season-type Playoffs
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
from nba_api.stats.endpoints import TeamGameLogs
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Mapped, mapped_column

from db import Base, SessionLocal, engine
from src.ingest.ingest_games import TeamGame  # noqa: F401  (register FK target on Base.metadata)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ingest_team_stats")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SEASON_TYPES = ("Regular Season", "Playoffs", "Pre Season", "All Star")
REQUEST_PAUSE_SECONDS = 1.0


class TeamGameAdvanced(Base):
    """One row per team per game (TeamGameLogs Advanced grain).

    Joins to `team_games` on (game_id, team_id).
    """

    __tablename__ = "team_games_advanced"
    __table_args__ = (
        ForeignKeyConstraint(
            ["game_id", "team_id"],
            ["team_games.game_id", "team_games.team_id"],
            ondelete="CASCADE",
        ),
    )

    game_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    team_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    season: Mapped[str] = mapped_column(String(10), index=True)
    season_type: Mapped[str] = mapped_column(String(20), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)

    min: Mapped[float | None] = mapped_column(Float, nullable=True)
    off_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    def_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    e_off_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    e_def_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    e_net_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    pace: Mapped[float | None] = mapped_column(Float, nullable=True)
    e_pace: Mapped[float | None] = mapped_column(Float, nullable=True)
    pace_per40: Mapped[float | None] = mapped_column(Float, nullable=True)
    poss: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ts_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    efg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_tov: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    oreb_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dreb_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    tm_tov_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    usg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    e_usg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    pie: Mapped[float | None] = mapped_column(Float, nullable=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def ingest_season(season: str, season_type: str = "Regular Season") -> int:
    """Fetch one season + season_type and upsert. Shared by the CLI and orchestrator."""
    raw = fetch_season(season, season_type)
    records = transform(raw, season, season_type)
    return upsert(records)


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    log.info("Fetching advanced stats for %s (%s)...", season, season_type)
    endpoint = TeamGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        measure_type_player_game_logs_nullable="Advanced",
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

    columns = {c.name for c in TeamGameAdvanced.__table__.columns} - {"ingested_at"}
    keep = [c for c in df.columns if c in columns]
    return df[keep].where(pd.notna(df[keep]), None).to_dict(orient="records")


def upsert(records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0
    stmt = pg_insert(TeamGameAdvanced).values(records)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in TeamGameAdvanced.__table__.columns
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
    path = RAW_DIR / f"team_games_advanced_{season}_{safe_type}_{stamp}.csv"
    df.to_csv(path, index=False)
    log.info("  -> CSV backup: %s", path)


def parse_args() -> argparse.Namespace:
    current_year = date.today().year
    p = argparse.ArgumentParser(description="Ingest NBA per-game advanced team stats.")
    p.add_argument("--start-season", type=int, default=2016)
    p.add_argument("--end-season", type=int, default=current_year)
    p.add_argument("--season-type", choices=SEASON_TYPES, default="Regular Season")
    p.add_argument("--csv-backup", action="store_true")
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
