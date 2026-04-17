"""Assemble the model-ready training matrix from ingested raw tables.

Pulls `team_games` ⨝ `team_games_advanced`, applies the full feature stack
(rolling stats, rest, situational, Elo), pivots to the matchup grain, and
writes a parquet file ready for training.

Usage:
    python -m src.pipeline.build_training_set
    python -m src.pipeline.build_training_set --min-game-number 10 --exclude-covid
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import engine
from src.features.elo import add_elo
from src.features.matchup import load_team_games, to_matchup
from src.features.player_features import add_player_features
from src.features.rest_features import add_rest, add_rest_diff
from src.features.rolling_stats import add_rolling
from src.features.situational import COVID_SEASONS, add_situational

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_training_set")

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"


def build(
    season_type: str = "Regular Season",
    min_game_number: int = 10,
    exclude_covid: bool = False,
    include_player_features: bool = True,
) -> pd.DataFrame:
    """Load raw, compute features, pivot, filter. Returns the matchup frame."""
    log.info("Loading team-game rows (%s)...", season_type)
    df = load_team_games(engine, season_type=season_type)
    log.info("  %d rows across %d seasons", len(df), df["season"].nunique())

    log.info("Applying per-team-game features: rolling, rest, situational...")
    df = add_rolling(df)
    df = add_rest(df)
    df = add_situational(df)

    if include_player_features:
        log.info("Loading player_games and computing rotation-availability features...")
        with engine.connect() as conn:
            pg = pd.read_sql_query(
                text("SELECT game_id, player_id, team_id, season, game_date, minutes "
                     "FROM player_games WHERE season_type = :st"),
                conn,
                params={"st": season_type},
            )
        log.info("  %d player-game rows", len(pg))
        df = add_player_features(df, pg)

    log.info("Pivoting to matchup grain...")
    m = to_matchup(df)
    log.info("  %d matchup rows", len(m))

    log.info("Applying matchup-level features: rest_diff, elo...")
    m = add_rest_diff(m)
    m = add_elo(m)

    before = len(m)
    if exclude_covid:
        m = m[~m["season"].isin(COVID_SEASONS)].copy()
        log.info("  Excluded COVID seasons: %d → %d rows", before, len(m))

    before = len(m)
    m = m[
        (m["home_game_number"] >= min_game_number)
        & (m["away_game_number"] >= min_game_number)
    ].copy()
    log.info("  Filtered to game_number >= %d: %d → %d rows", min_game_number, before, len(m))

    return m.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the NBA model training matrix.")
    p.add_argument("--season-type", default="Regular Season")
    p.add_argument("--min-game-number", type=int, default=10,
                   help="Require both teams to have played at least N games this season.")
    p.add_argument("--exclude-covid", action="store_true",
                   help="Drop the 2019-20 and 2020-21 seasons.")
    p.add_argument("--no-player-features", action="store_true",
                   help="Skip player-availability features (faster; used for ablation).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    m = build(
        season_type=args.season_type,
        min_game_number=args.min_game_number,
        exclude_covid=args.exclude_covid,
        include_player_features=not args.no_player_features,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(args.output, index=False)
    log.info("Wrote %d rows × %d cols to %s", len(m), len(m.columns), args.output)
    log.info("Season breakdown:")
    for s, n in m.groupby("season").size().items():
        log.info("  %s: %d games", s, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
