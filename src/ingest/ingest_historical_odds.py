"""One-shot ingest of historical NBA closing odds from a CSV.

The Odds API's free tier only ships forward-looking data, and even paid
historical feeds are expensive. Free Kaggle NBA odds CSVs fill the gap, but
they are almost always **single-line** per game (one closing moneyline,
often Pinnacle or an offshore average), not multi-bookmaker snapshots.

Workflow:

    # 1. See what's in the file.
    python -m src.ingest.ingest_historical_odds --inspect data/raw/nba_odds.csv

    # 2. Write a column map (JSON) and import.
    python -m src.ingest.ingest_historical_odds \\
        --csv data/raw/nba_odds.csv \\
        --column-map data/raw/odds_column_map.json

The column map maps CSV columns → OddsSnapshot fields. Required keys:
`date`, `home_team`, `away_team`, `home_ml`, `away_ml`. `bookmaker` may be a
literal (`"pinnacle_close"`) or a column name; if omitted, every row is
tagged `"kaggle_close"`. Spread/total columns are optional and ingested when
present.

Re-running over the same bookmaker is a no-op unless `--replace` is passed,
which deletes existing rows for that bookmaker label before inserting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path

import math

import pandas as pd
from sqlalchemy import delete, select

from db import Base, SessionLocal, engine
from src.ingest.ingest_odds import OddsSnapshot
from src.ingest.team_map import resolve_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_historical_odds")

REQUIRED_KEYS = ("date", "home_team", "away_team", "home_ml", "away_ml")
OPTIONAL_NUMERIC_KEYS = (
    "spread_home_point", "spread_home_price", "spread_away_price",
    "total_point", "total_over_price", "total_under_price",
)
DEFAULT_BOOKMAKER = "kaggle_close"


@dataclass(frozen=True)
class ColumnMap:
    """Declarative mapping from CSV columns → OddsSnapshot fields."""
    mapping: dict[str, str]
    bookmaker_literal: str | None

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMap":
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"Column map missing required keys: {missing}")
        bookmaker = d.get("bookmaker")
        # Heuristic: if it looks like a CSV column name we'd expect the CSV
        # to have, treat it as a column; otherwise a literal. Users can force
        # either by making the string a valid column name or not.
        return cls(mapping={k: v for k, v in d.items() if k != "bookmaker"},
                   bookmaker_literal=bookmaker)


def inspect_csv(path: Path, n: int = 5) -> None:
    """Print columns + first n rows so the user can build a column map."""
    df = pd.read_csv(path, nrows=n)
    log.info("File: %s  (%d rows in preview)", path, len(df))
    log.info("Columns (%d):", len(df.columns))
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(2).tolist()
        log.info("  %-30s  sample=%s", c, sample)
    log.info("\nFirst %d rows:\n%s", len(df), df.to_string(max_cols=12))


def _parse_date(s) -> datetime:
    """Accept ISO date, 'YYYY-MM-DD', 'MM/DD/YYYY', pandas Timestamp, etc."""
    return pd.to_datetime(s).to_pydatetime()


def _resolve_bookmaker(row: pd.Series, cmap: ColumnMap) -> str:
    """Bookmaker is either a CSV column lookup or a literal fallback."""
    label = cmap.bookmaker_literal or DEFAULT_BOOKMAKER
    if label in row.index:
        val = row[label]
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return label


def build_records(df: pd.DataFrame, cmap: ColumnMap) -> tuple[list[dict], int, int]:
    """Map CSV rows → OddsSnapshot dicts. Returns (records, n_resolved, n_skipped)."""
    records: list[dict] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            d = _parse_date(row[cmap.mapping["date"]])
        except Exception as exc:
            log.warning("Bad date %r: %s", row.get(cmap.mapping["date"]), exc)
            skipped += 1
            continue

        home_name = str(row[cmap.mapping["home_team"]]).strip()
        away_name = str(row[cmap.mapping["away_team"]]).strip()
        home_id = resolve_id(home_name)
        away_id = resolve_id(away_name)
        if home_id is None or away_id is None:
            log.warning("Unresolved team(s): home=%r away=%r", home_name, away_name)
            skipped += 1
            continue

        commence_time = datetime.combine(d.date(), time.min, tzinfo=timezone.utc)
        # Proxy "closing" timestamp: end-of-day on game date. It's the only
        # anchor we have; never tag an odds row with a real live-fetch time.
        fetched_at = datetime.combine(d.date(), time(23, 59, 59), tzinfo=timezone.utc)
        event_id = f"hist-{d.date().isoformat()}-{home_id}-{away_id}"

        try:
            ml_home = float(row[cmap.mapping["home_ml"]])
            ml_away = float(row[cmap.mapping["away_ml"]])
        except (ValueError, TypeError):
            log.warning("Non-numeric ML for %s vs %s on %s", home_name, away_name, d.date())
            skipped += 1
            continue
        # float(nan) doesn't raise — must check explicitly or NaN moneylines
        # slip into the DB and corrupt every downstream kelly/edge calc.
        if not (math.isfinite(ml_home) and math.isfinite(ml_away)):
            skipped += 1
            continue

        rec: dict = {
            "event_id": event_id,
            "commence_time": commence_time,
            "game_date": d.date(),
            "home_team": home_name,
            "away_team": away_name,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "bookmaker": _resolve_bookmaker(row, cmap),
            "ml_home": ml_home,
            "ml_away": ml_away,
            "fetched_at": fetched_at,
        }
        for key in OPTIONAL_NUMERIC_KEYS:
            col = cmap.mapping.get(key)
            if col and col in row.index:
                v = row[col]
                if pd.notna(v):
                    try:
                        rec[key] = float(v)
                    except (ValueError, TypeError):
                        pass
        records.append(rec)
    return records, len(records), skipped


def _unique_bookmakers(records: list[dict]) -> set[str]:
    return {r["bookmaker"] for r in records}


def write(records: list[dict], replace: bool = False) -> int:
    """Insert records. With `replace=True`, wipe existing rows for each
    bookmaker in `records` first. Without it, refuse to run if any of those
    bookmakers already have rows — safer for a one-shot import."""
    if not records:
        return 0
    books = _unique_bookmakers(records)
    with SessionLocal() as session:
        existing = session.scalars(
            select(OddsSnapshot.bookmaker)
            .where(OddsSnapshot.bookmaker.in_(books))
            .limit(1)
        ).first()
        if existing and not replace:
            raise RuntimeError(
                f"Bookmaker(s) {sorted(books)} already present in odds_snapshots. "
                "Pass --replace to delete and re-import."
            )
        if replace and existing:
            n_del = session.execute(
                delete(OddsSnapshot).where(OddsSnapshot.bookmaker.in_(books))
            ).rowcount
            log.info("Deleted %d existing rows for bookmaker(s) %s", n_del, sorted(books))

        session.bulk_insert_mappings(OddsSnapshot, records)
        session.commit()
    return len(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest a historical NBA odds CSV.")
    p.add_argument("--inspect", type=Path,
                   help="Print columns + sample rows and exit; use to build the column map.")
    p.add_argument("--csv", type=Path, help="CSV file to ingest.")
    p.add_argument("--column-map", type=Path,
                   help="JSON file mapping CSV columns → OddsSnapshot fields.")
    p.add_argument("--replace", action="store_true",
                   help="Delete existing rows for any bookmaker in this file before inserting.")
    p.add_argument("--create-tables", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Parse but don't write.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.create_tables:
        Base.metadata.create_all(engine)

    if args.inspect:
        inspect_csv(args.inspect)
        return 0

    if not args.csv or not args.column_map:
        log.error("Provide --csv and --column-map (or --inspect). See --help.")
        return 2

    cmap = ColumnMap.from_dict(json.loads(args.column_map.read_text()))
    df = pd.read_csv(args.csv)
    log.info("Loaded %d rows from %s", len(df), args.csv)

    records, n_ok, n_skip = build_records(df, cmap)
    log.info("Built %d records  (resolved=%d skipped=%d)", len(records), n_ok, n_skip)
    skip_rate = n_skip / max(1, n_ok + n_skip)
    if skip_rate > 0.05:
        log.warning("Skip rate %.1f%% is high — re-check team-name mapping.", skip_rate * 100)

    log.info("Bookmakers in file: %s", sorted(_unique_bookmakers(records)))

    if args.dry_run:
        for r in records[:3]:
            log.info("  sample: %s", r)
        return 0

    n = write(records, replace=args.replace)
    log.info("Inserted %d rows into odds_snapshots", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
