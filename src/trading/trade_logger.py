"""
Trade Logger
SQLite-backed log of every trade decision and its eventual outcome.
This is the foundation for the continual learning feedback loop.
"""

import hashlib
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

            CREATE TABLE IF NOT EXISTS training_examples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT UNIQUE NOT NULL,
                label       TEXT NOT NULL,
                pnl_pct     REAL,
                pair        TEXT,
                trade_id    TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hold_signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                pair             TEXT NOT NULL,
                confidence       INTEGER,
                price            REAL NOT NULL,
                market_data_json TEXT,
                logged_at        TEXT NOT NULL,
                validated_at     TEXT,
                forward_pnl_pct  REAL,
                hold_label       TEXT
            );
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


def get_live_readiness(min_trades: int = 30) -> dict:
    """
    Evaluate whether paper trading results justify moving to live capital.
    Uses the same R/R-aware gates as the backtester.

    Returns a dict with per-gate pass/fail and an overall verdict.
    Requires at least min_trades closed trades before any verdict is issued.
    """
    trades = get_closed_trades(limit=1000)
    closed = [t for t in trades if t["outcome"] is not None]
    n = len(closed)

    if n < min_trades:
        return {
            "ready": False,
            "verdict": f"INSUFFICIENT DATA — need {min_trades} closed trades, have {n}",
            "trades_needed": min_trades - n,
            "gates": [],
        }

    wins   = [t for t in closed if t["outcome"] in ("win", "took_profit")]
    losses = [t for t in closed if t["outcome"] in ("loss", "stopped_out")]

    win_rate     = len(wins) / n * 100
    avg_win_usd  = sum(t["pnl_usd"] for t in wins)  / len(wins)  if wins   else 0
    avg_loss_usd = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0  # negative
    total_pnl    = sum(t["pnl_usd"] for t in closed)

    gross_wins  = sum(t["pnl_usd"] for t in wins)
    gross_losses = abs(sum(t["pnl_usd"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # R/R-aware breakeven win rate
    avg_win  = abs(avg_win_usd)
    avg_loss = abs(avg_loss_usd)
    if avg_win > 0 and avg_loss > 0:
        rr_breakeven = avg_loss / (avg_win + avg_loss) * 100
    else:
        rr_breakeven = 50.0
    wr_threshold = round(rr_breakeven + 5, 1)

    gates = [
        (f"Min {min_trades} closed trades", n >= min_trades,       str(n)),
        (f"Win rate >= {wr_threshold}%",    win_rate >= wr_threshold, f"{win_rate:.1f}%"),
        ("Profit factor > 1.2",             profit_factor > 1.2,   f"{profit_factor:.2f}"),
        ("Positive total PnL",              total_pnl > 0,         f"${total_pnl:,.2f}"),
    ]

    all_pass = all(passed for _, passed, _ in gates)
    verdict = "READY FOR LIVE TRADING" if all_pass else "NOT READY — review failures below"

    return {
        "ready": all_pass,
        "verdict": verdict,
        "total_trades": n,
        "win_rate": round(win_rate, 1),
        "avg_win_usd": round(avg_win_usd, 2),
        "avg_loss_usd": round(avg_loss_usd, 2),
        "profit_factor": round(profit_factor, 2),
        "total_pnl_usd": round(total_pnl, 2),
        "rr_breakeven_pct": round(rr_breakeven, 1),
        "wr_threshold_pct": wr_threshold,
        "gates": gates,
    }


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


def log_hold_signal(pair: str, confidence: Optional[int], price: float, market_data: dict) -> None:
    """Log a HOLD decision for later forward-price validation."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO hold_signals (pair, confidence, price, market_data_json, logged_at)
            VALUES (?, ?, ?, ?, ?)
        """, (pair, confidence, price, json.dumps(market_data), datetime.utcnow().isoformat()))


def get_unvalidated_hold_signals(before: Optional[str] = None) -> list:
    """Return HOLD signals not yet validated, optionally only those logged before a cutoff."""
    with _connect() as conn:
        if before:
            rows = conn.execute(
                "SELECT * FROM hold_signals WHERE validated_at IS NULL AND logged_at < ? ORDER BY logged_at",
                (before,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hold_signals WHERE validated_at IS NULL ORDER BY logged_at"
            ).fetchall()
    return [dict(r) for r in rows]


def update_hold_signal(signal_id: int, forward_pnl_pct: float,
                       hold_label: str, validated_at: str) -> None:
    """Record the outcome of a validated HOLD signal."""
    with _connect() as conn:
        conn.execute("""
            UPDATE hold_signals
            SET forward_pnl_pct=?, hold_label=?, validated_at=?
            WHERE id=?
        """, (forward_pnl_pct, hold_label, validated_at, signal_id))


def get_existing_prompt_hashes() -> set:
    """Return the set of all prompt hashes already stored as training examples."""
    with _connect() as conn:
        rows = conn.execute("SELECT prompt_hash FROM training_examples").fetchall()
    return {r["prompt_hash"] for r in rows}


def log_training_example_hash(prompt_hash: str, label: str, pnl_pct: float,
                               pair: Optional[str], trade_id: Optional[str]) -> None:
    """Record that a prompt hash has been processed (prevents future duplicates)."""
    with _connect() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO training_examples
                (prompt_hash, label, pnl_pct, pair, trade_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (prompt_hash, label, pnl_pct, pair, trade_id, datetime.utcnow().isoformat()))


# Auto-init on import
init_db()
