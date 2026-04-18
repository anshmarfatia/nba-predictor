"""`bet_log` table: one row per bet placed (live or backtest).

A bet's lifecycle: created with `status='open'`, then settled to `won`,
`lost`, `push`, or `void` once the game outcome is known. `payout` is the
net P&L (signed), not the gross return.

Backtest runs share a `backtest_run_id` (UUID per replay) so the whole run
can be queried / deleted atomically. Live bets leave it NULL.

The ORM model reuses the project's existing `Base` so `Base.metadata.create_all(engine)`
creates this table alongside the others in `db.py`.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from db import Base, SessionLocal, engine

log = logging.getLogger("bet_log")


STATUS_OPEN = "open"
STATUS_WON = "won"
STATUS_LOST = "lost"
STATUS_PUSH = "push"
STATUS_VOID = "void"


class Bet(Base):
    """One row per placed bet.

    The schema is wide on purpose — every column here is something a reader
    of a backtest needs for post-hoc audits (strategy comparison, Kelly
    leverage checks, edge-bucket P&L). Cheap to store, expensive to
    reconstruct later.
    """

    __tablename__ = "bet_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    game_id: Mapped[str] = mapped_column(String(20), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    bookmaker: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(4))            # 'home' | 'away'

    entry_ml: Mapped[float] = mapped_column(Float)
    entry_market_prob: Mapped[float] = mapped_column(Float)  # de-vigged
    model_prob: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)

    stake: Mapped[float] = mapped_column(Float)
    kelly_full: Mapped[float] = mapped_column(Float)
    kelly_fraction_used: Mapped[float] = mapped_column(Float)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    bankroll_before: Mapped[float] = mapped_column(Float)

    backtest_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_OPEN, index=True)
    payout: Mapped[float | None] = mapped_column(Float, nullable=True)  # signed net P&L
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_bet_log_run_placed", "backtest_run_id", "placed_at"),
        Index("ix_bet_log_strategy_status", "strategy", "status"),
        Index("ix_bet_log_game_model", "game_id", "model_version"),
    )


# ---------------------------------------------------------------------------
# Settlement math
# ---------------------------------------------------------------------------

def realize_pnl(stake: float, entry_ml: float, won: bool) -> float:
    """Signed net P&L for a moneyline bet at American odds `entry_ml`.

    `won=True`  →  stake * (ml/100) if ml>0, else stake * (100/|ml|)
    `won=False` →  -stake
    """
    if not won:
        return -float(stake)
    if entry_ml > 0:
        return float(stake * entry_ml / 100.0)
    return float(stake * 100.0 / abs(entry_ml))


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def new_run_id() -> str:
    return uuid.uuid4().hex


def ensure_table() -> None:
    """Create bet_log (and any other pending tables) if not present."""
    Base.metadata.create_all(engine)


def bulk_insert_bets(records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0
    with SessionLocal() as session:
        session.bulk_insert_mappings(Bet, records)
        session.commit()
    return len(records)


def delete_run(run_id: str) -> int:
    """Delete all bets from a given backtest_run_id. Useful for re-running."""
    from sqlalchemy import delete
    with SessionLocal() as session:
        n = session.execute(delete(Bet).where(Bet.backtest_run_id == run_id)).rowcount
        session.commit()
    log.info("Deleted %d bets for run %s", n, run_id)
    return n


def fetch_bets(
    run_id: str | None = None,
    strategy: str | None = None,
    since: date | None = None,
) -> pd.DataFrame:
    """Read bets back as a DataFrame for analysis / dashboards."""
    stmt = select(Bet)
    if run_id is not None:
        stmt = stmt.where(Bet.backtest_run_id == run_id)
    if strategy is not None:
        stmt = stmt.where(Bet.strategy == strategy)
    if since is not None:
        stmt = stmt.where(Bet.game_date >= since)
    with SessionLocal() as session:
        rows = session.scalars(stmt.order_by(Bet.placed_at, Bet.id)).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {c.name: getattr(r, c.name) for c in Bet.__table__.columns} for r in rows
    ])


@dataclass(frozen=True)
class Settlement:
    won: bool       # False means loss; pushes/voids handled separately
    status: str     # one of STATUS_*


def settle_bet(session, bet_id: int, settlement: Settlement) -> None:
    """Apply a settlement to a stored bet. Idempotent for same outcome."""
    bet = session.get(Bet, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet.status != STATUS_OPEN:
        return
    if settlement.status == STATUS_PUSH:
        bet.payout = 0.0
    elif settlement.status == STATUS_VOID:
        bet.payout = 0.0
    else:
        bet.payout = realize_pnl(bet.stake, bet.entry_ml, settlement.won)
    bet.status = settlement.status
    bet.settled_at = datetime.now(timezone.utc)
