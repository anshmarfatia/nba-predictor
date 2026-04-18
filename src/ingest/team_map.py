"""Mapping between external team names and nba_api team_ids.

The Odds API returns full team names ('Boston Celtics'). Kaggle betting
CSVs often use lowercased short codes ('bos', 'gs', 'sa'). `nba_api`
itself uses canonical abbreviations ('BOS', 'GSW', 'SAS'). This module
bridges all three via `nba_api.stats.static.teams`.

Resolution order:
  1. Full name (case-insensitive), aliased if needed.
  2. Abbreviation (case-insensitive), aliased if needed.
"""
from __future__ import annotations

from functools import lru_cache

from nba_api.stats.static import teams as nba_teams

# Full-name aliases (Odds-API-style names that don't match nba_api exactly).
ODDS_ALIASES: dict[str, str] = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "phoenix suns ": "phoenix suns",
}

# Abbreviation aliases (Kaggle-style short codes → canonical nba_api abbrev).
# Known mismatches: Kaggle's nba_2008-2025.csv uses shorter forms than nba_api.
ABBREVIATION_ALIASES: dict[str, str] = {
    "gs": "GSW",
    "no": "NOP",
    "ny": "NYK",
    "sa": "SAS",
    "utah": "UTA",
    "wsh": "WAS",
    "nj": "BKN",         # New Jersey Nets → Brooklyn (pre-2012 games)
    "sea": "OKC",        # Seattle SuperSonics → OKC (pre-2008 games)
    "cha": "CHA",        # explicit: Charlotte Bobcats era
    "bkn": "BKN",
}


@lru_cache(maxsize=1)
def _by_fullname() -> dict[str, dict]:
    return {t["full_name"].lower(): t for t in nba_teams.get_teams()}


@lru_cache(maxsize=1)
def _by_abbreviation() -> dict[str, dict]:
    return {t["abbreviation"].upper(): t for t in nba_teams.get_teams()}


def resolve(name: str) -> dict | None:
    """Return the nba_api team record for an external team name, or None.

    Tries full-name resolution first, then abbreviation. Either form is
    normalized (strip + case-fold) and passed through its alias map.
    """
    if not name:
        return None
    key = name.strip().lower()
    key_full = ODDS_ALIASES.get(key, key)
    rec = _by_fullname().get(key_full)
    if rec is not None:
        return rec
    abbr = ABBREVIATION_ALIASES.get(key, key.upper())
    return _by_abbreviation().get(abbr)


def resolve_id(name: str) -> int | None:
    rec = resolve(name)
    return int(rec["id"]) if rec else None
