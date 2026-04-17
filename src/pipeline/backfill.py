"""Backfill predictions over a date range.

Populates the `predictions` table with out-of-sample scores for every NBA
game between `--start-date` and `--end-date`. The dashboard's Historical
Accuracy tab needs this to show a meaningful reliability curve — 15 graded
games isn't enough to see calibration clearly; 1,000+ is.

We reconstruct each day's scoreboard directly from the `team_games` table
(already ingested) rather than hitting ScoreboardV2 per day. That turns
~170 network calls into zero and is strictly idempotent.

Usage:
    python -m src.pipeline.backfill --model-version v1 \\
        --start-date 2024-10-22 --end-date 2025-04-13
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Iterator

import pandas as pd

from db import engine
from src.features.matchup import load_team_games, to_matchup
from src.pipeline.predict import predict_for_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill")


def daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def scoreboard_for(
    all_team_games: pd.DataFrame,
    as_of: date,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Reconstruct the scoreboard for a past date from already-ingested rows.

    Uses `to_matchup` to collapse team-game pairs → one row per game and to
    drop neutral-site games (same filter training uses)."""
    day = all_team_games[
        (all_team_games["game_date"] == pd.Timestamp(as_of))
        & (all_team_games["season_type"] == season_type)
    ]
    m = to_matchup(day)
    if m.empty:
        return pd.DataFrame(columns=["game_id", "home_team_id", "away_team_id", "game_date"])
    return m[["game_id", "home_team_id", "away_team_id"]].assign(game_date=as_of)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill model predictions over a date range.")
    p.add_argument("--model-version", required=True,
                   help="Saved model to score with (e.g. 'v1').")
    p.add_argument("--start-date", type=lambda s: datetime.fromisoformat(s).date(),
                   required=True)
    p.add_argument("--end-date", type=lambda s: datetime.fromisoformat(s).date(),
                   required=True)
    p.add_argument("--season-type", default="Regular Season")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        log.error("--start-date (%s) > --end-date (%s)", args.start_date, args.end_date)
        return 1

    log.info("Loading team_games once for the whole backfill...")
    all_games = load_team_games(engine, season_type=args.season_type)

    total = 0
    days_with_games = 0
    days_empty = 0
    for d in daterange(args.start_date, args.end_date):
        sb = scoreboard_for(all_games, d, season_type=args.season_type)
        if sb.empty:
            days_empty += 1
            continue
        n = predict_for_date(d, args.model_version, dry_run=args.dry_run, scoreboard=sb)
        days_with_games += 1
        total += n
        log.info("%s: %d games scored (running total: %d)", d, n, total)

    verb = "Would write" if args.dry_run else "Upserted"
    log.info("Done. %s %d predictions across %d game-days (%d empty days skipped).",
             verb, total, days_with_games, days_empty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
