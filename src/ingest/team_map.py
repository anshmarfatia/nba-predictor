"""Mapping between The Odds API team names and nba_api team_ids.

The Odds API returns full team names like 'Boston Celtics'. `nba_api` uses
integer team IDs everywhere else in this project. This module bridges them
via `nba_api.stats.static.teams`, which ships a curated list of 30 teams.

We normalize by lowercased full name so 'LA Clippers' and 'Los Angeles
Clippers' collide into one entry — both spellings appear in the wild.
"""
from __future__ import annotations

from functools import lru_cache

from nba_api.stats.static import teams as nba_teams

# Known aliases that appear in odds feeds but don't match nba_api's `full_name`.
ODDS_ALIASES: dict[str, str] = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "phoenix suns ": "phoenix suns",
}


@lru_cache(maxsize=1)
def _lookup() -> dict[str, dict]:
    """Dict keyed by lowercased full name → team record."""
    return {t["full_name"].lower(): t for t in nba_teams.get_teams()}


def resolve(name: str) -> dict | None:
    """Return the nba_api team record for an odds-feed team name, or None."""
    if not name:
        return None
    key = name.strip().lower()
    key = ODDS_ALIASES.get(key, key)
    return _lookup().get(key)


def resolve_id(name: str) -> int | None:
    rec = resolve(name)
    return int(rec["id"]) if rec else None
