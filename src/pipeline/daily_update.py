"""Daily orchestrator. Run this once a day from cron / APScheduler / GitHub Actions.

Chain:
  1. Ingest yesterday's completed games (box score + advanced stats).
  2. Snapshot current odds for upcoming games (skipped if ODDS_API_KEY is unset).
  3. Score today's scheduled games with the pinned model version.

Each step is wrapped in try/except so one failure does not kill the rest —
we want partial progress over all-or-nothing. The exit code is 0 only if
every *required* step succeeded (odds is optional).

Usage:
    python -m src.pipeline.daily_update --model-version v1
    python -m src.pipeline.daily_update --model-version v1 --date 2025-04-11
    python -m src.pipeline.daily_update --model-version v1 --skip-odds
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from db import Base, engine
from src.ingest import ingest_games, ingest_team_stats  # noqa: F401 (register tables)
from src.ingest.ingest_odds import fetch_live, parse_events, upsert as upsert_odds  # noqa: F401
from src.pipeline.predict import predict_for_date  # noqa: F401 (registers Prediction table)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("daily_update")


@dataclass
class StepResult:
    name: str
    status: str          # 'ok' | 'skipped' | 'failed'
    detail: str = ""
    n: int = 0

    @property
    def required_ok(self) -> bool:
        return self.status in ("ok", "skipped")


def run_step(name: str, fn: Callable[[], int], required: bool = True) -> StepResult:
    try:
        n = fn()
        log.info("[%s] ok (n=%d)", name, n)
        return StepResult(name, "ok", n=n)
    except Exception as e:
        log.exception("[%s] FAILED: %s", name, e)
        return StepResult(name, "failed", detail=f"{type(e).__name__}: {e}")


def ingest_games_step() -> int:
    season = ingest_games.current_season_label()
    log.info("Ingesting team_games for %s...", season)
    return ingest_games.ingest_season(season)


def ingest_advanced_step() -> int:
    season = ingest_games.current_season_label()
    log.info("Ingesting team_games_advanced for %s...", season)
    return ingest_team_stats.ingest_season(season)


def ingest_odds_step() -> int:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY not set")
    payload = fetch_live(api_key)
    records = parse_events(payload)
    log.info("Fetched %d (event, bookmaker) rows.", len(records))
    return upsert_odds(records)


def predict_step(as_of: date, model_version: str) -> int:
    log.info("Scoring games for %s with model=%s...", as_of, model_version)
    return predict_for_date(as_of, model_version)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily NBA data + prediction update.")
    p.add_argument("--model-version", required=True,
                   help="Saved model to use for today's predictions (e.g. 'v1').")
    p.add_argument("--date", type=lambda s: datetime.fromisoformat(s).date(),
                   default=date.today(),
                   help="Target date for predictions. Default: today.")
    p.add_argument("--skip-odds", action="store_true",
                   help="Skip the odds snapshot step.")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip game + advanced-stat ingestion (useful for quick re-scoring).")
    p.add_argument("--create-tables", action="store_true",
                   help="Create any missing tables before running. Safe on every run.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.create_tables:
        log.info("Creating any missing tables...")
        Base.metadata.create_all(engine)
    steps: list[StepResult] = []

    if args.skip_ingest:
        steps.append(StepResult("ingest_games", "skipped", detail="--skip-ingest"))
        steps.append(StepResult("ingest_advanced", "skipped", detail="--skip-ingest"))
    else:
        steps.append(run_step("ingest_games", ingest_games_step))
        steps.append(run_step("ingest_advanced", ingest_advanced_step))

    if args.skip_odds:
        steps.append(StepResult("ingest_odds", "skipped", detail="--skip-odds"))
    elif not os.environ.get("ODDS_API_KEY"):
        steps.append(StepResult("ingest_odds", "skipped", detail="ODDS_API_KEY unset"))
    else:
        steps.append(run_step("ingest_odds", ingest_odds_step, required=False))

    steps.append(run_step("predict", lambda: predict_step(args.date, args.model_version)))

    log.info("=" * 60)
    log.info("Daily update summary for %s:", args.date)
    for s in steps:
        tag = {"ok": " OK ", "skipped": "SKIP", "failed": "FAIL"}[s.status]
        tail = f"  ({s.detail})" if s.detail else f"  n={s.n}"
        log.info("  [%s] %-18s%s", tag, s.name, tail)
    log.info("=" * 60)

    required = ["ingest_games", "ingest_advanced", "predict"]
    failed_required = [s for s in steps if s.name in required and s.status == "failed"]
    return 0 if not failed_required else 1


if __name__ == "__main__":
    sys.exit(main())
