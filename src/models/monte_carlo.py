"""Monte Carlo playoff probability simulator.

Given a mid-season cutoff date, simulates the remaining NBA schedule thousands
of times using Elo-derived win probabilities. Each simulation produces a
complete season record per team; aggregating across simulations yields:
  - P(playoff) — top-6 seed in conference (auto-qualify)
  - P(play-in) — seeds 7-10 (the play-in tournament)
  - P(lottery) — seeds 11-15 (eliminated)
  - Expected final wins distribution

Usage:
    python -m src.models.monte_carlo --date 2025-02-01 --n 10000
    python -m src.models.monte_carlo --date 2025-03-01 --n 5000
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from typing import NamedTuple

import numpy as np
import pandas as pd

from db import engine
from src.features.elo import final_elos, pre_game_elo
from src.features.matchup import load_team_games, to_matchup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("monte_carlo")

EAST = frozenset({
    1610612737,  # ATL
    1610612738,  # BOS
    1610612751,  # BKN
    1610612766,  # CHA
    1610612741,  # CHI
    1610612739,  # CLE
    1610612765,  # DET
    1610612754,  # IND
    1610612748,  # MIA
    1610612749,  # MIL
    1610612752,  # NYK
    1610612753,  # ORL
    1610612755,  # PHI
    1610612761,  # TOR
    1610612764,  # WAS
})
WEST = frozenset({
    1610612742,  # DAL
    1610612743,  # DEN
    1610612744,  # GSW
    1610612745,  # HOU
    1610612746,  # LAC
    1610612747,  # LAL
    1610612763,  # MEM
    1610612750,  # MIN
    1610612740,  # NOP
    1610612760,  # OKC
    1610612756,  # PHX
    1610612757,  # POR
    1610612758,  # SAC
    1610612759,  # SAS
    1610612762,  # UTA
})


def conference_of(team_id: int) -> str:
    if team_id in EAST:
        return "East"
    if team_id in WEST:
        return "West"
    return "Unknown"


class SeasonState(NamedTuple):
    standings: pd.DataFrame
    remaining: pd.DataFrame
    ratings: dict[int, float]
    last_season: dict[int, str]
    cutoff_season: str


def season_state_at(
    cutoff: date,
    season: str | None = None,
    home_advantage: float = 100.0,
) -> SeasonState:
    """Build standings + remaining schedule + Elo ratings as of `cutoff`."""
    raw = load_team_games(engine)
    if season is None:
        year = cutoff.year if cutoff.month >= 10 else cutoff.year - 1
        season = f"{year}-{str(year + 1)[-2:]}"

    season_games = raw[
        (raw["season"] == season) & (raw["season_type"] == "Regular Season")
    ].copy()

    played = season_games[season_games["game_date"] <= pd.Timestamp(cutoff)]
    standings = played.groupby("team_id").agg(
        wins=("wl", lambda s: (s == "W").sum()),
        losses=("wl", lambda s: (s == "L").sum()),
        games=("game_id", "size"),
    ).reset_index()
    standings["conference"] = standings["team_id"].map(conference_of)

    remaining_matchup = to_matchup(
        season_games[season_games["game_date"] > pd.Timestamp(cutoff)]
    )
    remaining = remaining_matchup[["game_id", "game_date", "home_team_id", "away_team_id"]].copy()

    all_prior = raw[raw["game_date"] <= pd.Timestamp(cutoff)]
    m = to_matchup(all_prior)
    from src.features.elo import add_elo
    m = add_elo(m)
    ratings, last_season_map = final_elos(m)

    log.info("As of %s (%s): %d teams, %d played, %d remaining",
             cutoff, season, len(standings), len(played) // 2, len(remaining))

    return SeasonState(standings, remaining, ratings, last_season_map, season)


def win_probability(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 100.0,
) -> float:
    return 1.0 / (1.0 + 10 ** (-((home_elo + home_advantage) - away_elo) / 400.0))


def simulate_once(
    standings: pd.DataFrame,
    remaining: pd.DataFrame,
    ratings: dict[int, float],
    last_season: dict[int, str],
    season: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run one simulation: flip each remaining game, return final standings."""
    sim_wins = standings.set_index("team_id")["wins"].to_dict()
    sim_losses = standings.set_index("team_id")["losses"].to_dict()

    for _, game in remaining.iterrows():
        h, a = int(game["home_team_id"]), int(game["away_team_id"])
        h_elo = pre_game_elo(h, season, ratings, last_season)
        a_elo = pre_game_elo(a, season, ratings, last_season)
        p_home = win_probability(h_elo, a_elo)
        if rng.random() < p_home:
            sim_wins[h] = sim_wins.get(h, 0) + 1
            sim_losses[a] = sim_losses.get(a, 0) + 1
        else:
            sim_wins[a] = sim_wins.get(a, 0) + 1
            sim_losses[h] = sim_losses.get(h, 0) + 1

    final = pd.DataFrame([
        {"team_id": tid, "wins": sim_wins[tid], "losses": sim_losses.get(tid, 0)}
        for tid in sim_wins
    ])
    final["conference"] = final["team_id"].map(conference_of)
    return final


def seed_teams(final: pd.DataFrame) -> pd.DataFrame:
    """Assign conference seeds (1-15) by win %, with random tiebreaker."""
    out = final.copy()
    out["win_pct"] = out["wins"] / (out["wins"] + out["losses"])
    out["tiebreak"] = np.random.random(len(out))
    result = []
    for conf in ("East", "West"):
        conf_df = out[out["conference"] == conf].sort_values(
            ["win_pct", "tiebreak"], ascending=False
        ).reset_index(drop=True)
        conf_df["seed"] = range(1, len(conf_df) + 1)
        result.append(conf_df)
    return pd.concat(result, ignore_index=True)


def run_simulation(
    cutoff: date,
    n_simulations: int = 10_000,
    season: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run N simulations and aggregate probabilities.

    Returns one row per team with:
      playoff_prob, playin_prob, lottery_prob, wins_mean, wins_std
    """
    state = season_state_at(cutoff, season=season)
    rng = np.random.default_rng(seed)

    from nba_api.stats.static import teams as nba_teams
    lookup = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}

    playoff_counts: dict[int, int] = {}
    playin_counts: dict[int, int] = {}
    wins_accum: dict[int, list[int]] = {}

    for i in range(n_simulations):
        final = simulate_once(
            state.standings, state.remaining,
            state.ratings, state.last_season,
            state.cutoff_season, rng,
        )
        seeded = seed_teams(final)
        for _, row in seeded.iterrows():
            tid = int(row["team_id"])
            wins_accum.setdefault(tid, []).append(int(row["wins"]))
            if row["seed"] <= 6:
                playoff_counts[tid] = playoff_counts.get(tid, 0) + 1
            elif row["seed"] <= 10:
                playin_counts[tid] = playin_counts.get(tid, 0) + 1

    rows = []
    for tid in wins_accum:
        w = np.array(wins_accum[tid])
        rows.append({
            "team_id": tid,
            "team": lookup.get(tid, str(tid)),
            "conference": conference_of(tid),
            "current_wins": int(state.standings.loc[
                state.standings["team_id"] == tid, "wins"
            ].iloc[0]) if tid in state.standings["team_id"].values else 0,
            "current_losses": int(state.standings.loc[
                state.standings["team_id"] == tid, "losses"
            ].iloc[0]) if tid in state.standings["team_id"].values else 0,
            "wins_mean": round(w.mean(), 1),
            "wins_std": round(w.std(), 1),
            "wins_p5": int(np.percentile(w, 5)),
            "wins_p95": int(np.percentile(w, 95)),
            "playoff_prob": round(playoff_counts.get(tid, 0) / n_simulations, 4),
            "playin_prob": round(playin_counts.get(tid, 0) / n_simulations, 4),
            "lottery_prob": round(
                1 - (playoff_counts.get(tid, 0) + playin_counts.get(tid, 0)) / n_simulations,
                4,
            ),
        })

    result = pd.DataFrame(rows).sort_values(
        ["conference", "playoff_prob"], ascending=[True, False]
    ).reset_index(drop=True)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo NBA playoff probability simulator.")
    p.add_argument("--date", type=lambda s: datetime.fromisoformat(s).date(), required=True,
                   help="Cutoff date (standings + schedule as of this date).")
    p.add_argument("--season", type=str, default=None,
                   help="NBA season label (e.g. '2024-25'). Auto-detected from --date if omitted.")
    p.add_argument("-n", type=int, default=10_000, help="Number of simulations (default: 10000).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Running %d simulations as of %s...", args.n, args.date)
    result = run_simulation(args.date, n_simulations=args.n, season=args.season, seed=args.seed)

    for conf in ("East", "West"):
        log.info("\n=== %s ===", conf)
        c = result[result["conference"] == conf].reset_index(drop=True)
        log.info("\n%s", c[["team", "current_wins", "current_losses",
                            "wins_mean", "wins_p5", "wins_p95",
                            "playoff_prob", "playin_prob", "lottery_prob"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
