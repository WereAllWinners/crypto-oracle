"""
Trade Logger
SQLite-backed log of every trade decision and its eventual outcome.
This is the foundation for the continual learning feedback loop.
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trades.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id        TEXT UNIQUE NOT NULL,
                pair            TEXT NOT NULL,
                direction       TEXT NOT NULL,   -- BUY / SELL
                status          TEXT NOT NULL,   -- open / closed / cancelled
                opened_at       TEXT NOT NULL,
                closed_at       TEXT,

                -- Prices
                entry_price     REAL NOT NULL,
                stop_loss       REAL NOT NULL,
                take_profit     REAL NOT NULL,
                exit_price      REAL,

                -- Sizing
                position_size_usd  REAL NOT NULL,
                position_size_asset REAL NOT NULL,
                risk_usd        REAL NOT NULL,
                risk_pct        REAL NOT NULL,
                reward_risk_ratio REAL NOT NULL,

                -- LLM signal
                confidence      INTEGER,
                model_response  TEXT,   -- full raw LLM text

                -- Outcome
                pnl_usd         REAL,
                pnl_pct         REAL,
                outcome         TEXT,   -- win / loss / breakeven / stopped_out / took_profit

                -- Full snapshots (JSON blobs)
                market_data_json    TEXT,
                trade_decision_json TEXT
            );

            CREATE TABLE IF NOT EXISTS rule_rejections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                rejected_at TEXT NOT NULL,
                pair        TEXT NOT NULL,
                direction   TEXT NOT NULL,
                confidence  INTEGER,
                reasons     TEXT NOT NULL,   -- JSON list
                market_data_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trades_pair    ON trades(pair);
            CREATE INDEX IF NOT EXISTS idx_trades_status  ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
        """)
    logger.info(f"Trade DB initialised at {DB_PATH}")


def log_approved_trade(
    trade_id: str,
    decision: dict,
    recommendation: dict,
    market_data: dict,
    model_response: str = "",
) -> None:
    """
    Called immediately when RiskManager approves a trade (before order placed).
    decision = TradeDecision.to_dict()
    """
    with _connect() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO trades (
                trade_id, pair, direction, status, opened_at,
                entry_price, stop_loss, take_profit,
                position_size_usd, position_size_asset,
                risk_usd, risk_pct, reward_risk_ratio,
                confidence, model_response,
                market_data_json, trade_decision_json
            ) VALUES (
                :trade_id, :pair, :direction, 'open', :opened_at,
                :entry_price, :stop_loss, :take_profit,
                :position_size_usd, :position_size_asset,
                :risk_usd, :risk_pct, :reward_risk_ratio,
                :confidence, :model_response,
                :market_data_json, :trade_decision_json
            )
        """, {
            "trade_id": trade_id,
            "pair": decision["pair"],
            "direction": decision["direction"],
            "opened_at": datetime.utcnow().isoformat(),
            "entry_price": decision["entry_price"],
            "stop_loss": decision["stop_loss"],
            "take_profit": decision["take_profit"],
            "position_size_usd": decision["position_size_usd"],
            "position_size_asset": decision["position_size_asset"],
            "risk_usd": decision["risk_usd"],
            "risk_pct": decision["risk_pct"],
            "reward_risk_ratio": decision["reward_risk_ratio"],
            "confidence": recommendation.get("confidence"),
            "model_response": model_response,
            "market_data_json": json.dumps(market_data),
            "trade_decision_json": json.dumps(decision),
        })


def log_rejection(
    pair: str,
    direction: str,
    confidence: Optional[int],
    reasons: list,
    market_data: dict,
) -> None:
    """Log every rejected signal for analysis."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO rule_rejections (rejected_at, pair, direction, confidence, reasons, market_data_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            pair,
            direction,
            confidence,
            json.dumps(reasons),
            json.dumps(market_data),
        ))


def close_trade(
    trade_id: str,
    exit_price: float,
    outcome: str,  # "win" | "loss" | "breakeven" | "stopped_out" | "took_profit"
) -> None:
    """
    Record trade outcome. Call when position is actually closed.
    outcome: use "stopped_out" if SL hit, "took_profit" if TP hit, "win"/"loss" for manual close.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT direction, entry_price, position_size_usd FROM trades WHERE trade_id = ?",
            (trade_id,)
        ).fetchone()

        if not row:
            logger.warning(f"close_trade: trade_id {trade_id} not found")
            return

        direction = row["direction"]
        entry = row["entry_price"]
        size_usd = row["position_size_usd"]

        if direction == "BUY":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        pnl_usd = size_usd * pnl_pct

        conn.execute("""
            UPDATE trades
            SET status='closed', closed_at=?, exit_price=?, pnl_usd=?, pnl_pct=?, outcome=?
            WHERE trade_id=?
        """, (
            datetime.utcnow().isoformat(),
            exit_price,
            pnl_usd,
            pnl_pct * 100,
            outcome,
            trade_id,
        ))

    logger.info(f"Trade {trade_id} closed: {outcome}  PnL=${pnl_usd:.2f} ({pnl_pct*100:.2f}%)")


def get_open_trades() -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='open' ORDER BY opened_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_closed_trades(limit: int = 500) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='closed' ORDER BY closed_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_daily_pnl() -> float:
    """Sum of today's closed PnL."""
    today = date.today().isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) as total FROM trades WHERE status='closed' AND closed_at >= ?",
            (today,)
        ).fetchone()
    return row["total"]


def get_performance_summary() -> dict:
    """Aggregate stats for the continual learner."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome IN ('win','took_profit') THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome IN ('loss','stopped_out') THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(pnl_usd) as total_pnl_usd,
                AVG(CASE WHEN outcome IN ('win','took_profit') THEN pnl_pct END) as avg_win_pct,
                AVG(CASE WHEN outcome IN ('loss','stopped_out') THEN pnl_pct END) as avg_loss_pct
            FROM trades WHERE status='closed'
        """).fetchone()

    total = rows["total"] or 0
    wins = rows["wins"] or 0
    return {
        "total_trades": total,
        "wins": wins,
        "losses": rows["losses"] or 0,
        "win_rate": wins / total if total > 0 else 0,
        "avg_pnl_pct": rows["avg_pnl_pct"] or 0,
        "total_pnl_usd": rows["total_pnl_usd"] or 0,
        "avg_win_pct": rows["avg_win_pct"] or 0,
        "avg_loss_pct": rows["avg_loss_pct"] or 0,
    }


# Auto-init on import
init_db()
