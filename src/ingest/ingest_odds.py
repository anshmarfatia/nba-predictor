"""Ingest current NBA odds snapshots from The Odds API.

The free tier of the-odds-api.com covers upcoming NBA games with moneyline,
spread, and total markets across major US books. Historical odds are a paid
feature, so this script is designed for *forward* collection: run it daily
and the database accumulates a history matched to outcomes as games finish.

Requires ODDS_API_KEY in .env. Sign up at https://the-odds-api.com.

Usage:
    python -m src.ingest.ingest_odds --create-tables          # first run
    python -m src.ingest.ingest_odds                          # daily
    python -m src.ingest.ingest_odds --parse-fixture path.json  # test offline
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
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
from src.ingest.team_map import resolve_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_odds")

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_BOOKMAKERS = ("draftkings", "fanduel", "betmgm", "caesars")


class OddsSnapshot(Base):
    """One row per (event, bookmaker, fetch). We keep every snapshot so line
    movement is recoverable from history."""

    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    event_id: Mapped[str] = mapped_column(String(64), index=True)
    commence_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    game_date: Mapped[pd.Timestamp] = mapped_column(Date, index=True)

    home_team: Mapped[str] = mapped_column(String(64))
    away_team: Mapped[str] = mapped_column(String(64))
    home_team_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    away_team_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)

    bookmaker: Mapped[str] = mapped_column(String(32), index=True)

    ml_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_home_point: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_home_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_away_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_point: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_over_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_under_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def fetch_live(
    api_key: str,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> list[dict]:
    """Hit the live odds endpoint. Returns the raw JSON payload."""
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": ",".join(bookmakers),
    }
    resp = requests.get(ODDS_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    log.info("Remaining API requests: %s", resp.headers.get("x-requests-remaining", "?"))
    return resp.json()


def parse_event(event: dict, home_team: str, away_team: str) -> list[dict]:
    """Flatten one event's bookmakers into one row per bookmaker."""
    rows = []
    commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))

    for bm in event.get("bookmakers", []):
        row = {
            "event_id": event["id"],
            "commence_time": commence,
            "game_date": commence.astimezone(timezone.utc).date(),
            "home_team": home_team,
            "away_team": away_team,
            "home_team_id": resolve_id(home_team),
            "away_team_id": resolve_id(away_team),
            "bookmaker": bm["key"],
        }
        for market in bm.get("markets", []):
            outs = market.get("outcomes", [])
            if market["key"] == "h2h":
                for o in outs:
                    if o["name"] == home_team:
                        row["ml_home"] = float(o["price"])
                    elif o["name"] == away_team:
                        row["ml_away"] = float(o["price"])
            elif market["key"] == "spreads":
                for o in outs:
                    if o["name"] == home_team:
                        row["spread_home_point"] = float(o.get("point", 0.0))
                        row["spread_home_price"] = float(o["price"])
                    elif o["name"] == away_team:
                        row["spread_away_price"] = float(o["price"])
            elif market["key"] == "totals":
                for o in outs:
                    if o["name"] == "Over":
                        row["total_point"] = float(o.get("point", 0.0))
                        row["total_over_price"] = float(o["price"])
                    elif o["name"] == "Under":
                        row["total_under_price"] = float(o["price"])
        rows.append(row)
    return rows


def parse_events(payload: list[dict]) -> list[dict]:
    """Flatten the top-level events list into per-bookmaker rows."""
    rows: list[dict] = []
    for event in payload:
        rows.extend(parse_event(event, event["home_team"], event["away_team"]))
    return rows


def upsert(records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0
    stmt = pg_insert(OddsSnapshot).values(records)
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()
    return len(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch NBA odds snapshots from The Odds API.")
    p.add_argument("--create-tables", action="store_true")
    p.add_argument("--parse-fixture", type=Path,
                   help="Parse a saved JSON payload instead of hitting the live API.")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch + parse but don't write to the DB.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.create_tables:
        log.info("Creating odds_snapshots table (if not exists)...")
        Base.metadata.create_all(engine)

    if args.parse_fixture:
        log.info("Parsing fixture %s...", args.parse_fixture)
        payload = json.loads(args.parse_fixture.read_text())
    else:
        api_key = os.environ.get("ODDS_API_KEY")
        if not api_key:
            log.error("Set ODDS_API_KEY in .env (sign up at https://the-odds-api.com).")
            return 1
        log.info("Fetching live NBA odds...")
        payload = fetch_live(api_key)

    log.info("Payload: %d events", len(payload))
    records = parse_events(payload)
    log.info("Parsed %d (event, bookmaker) rows", len(records))

    if args.dry_run:
        for r in records[:3]:
            log.info("  sample: %s", r)
        return 0

    n = upsert(records)
    log.info("Inserted %d rows into odds_snapshots", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
