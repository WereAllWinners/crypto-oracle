"""
Live Trader
Executes approved TradeDecision objects against Coinbase Advanced Trade
via CCXT. Mirrors paper_trader.py structure with a --mode flag.

Modes:
  paper  -- Simulated prices, no exchange connection, positions tracked in DB.
             Useful for full end-to-end simulation before touching real money.
  dry    -- Real prices fetched from exchange, orders LOGGED but not placed.
             Exchange credentials optional (public market data only).
  live   -- Real limit orders on Coinbase Advanced Trade.
             Requires LIVE_TRADING_ENABLED=true in .env.

Safety gates:
  - pause.flag file: create this file in the project root to pause new entries
    instantly (delete the file to resume). Simple and reliable.
  - LIVE_TRADING_ENABLED=true: env var gate for live mode.
  - Limit orders with postOnly=True: maker-only, lower fees, no surprise fills.
  - fill_timeout_sec: if limit order not filled within timeout, cancel + skip.
  - Partial fill handling: if partially filled on cancel, track actual qty.
  - Max slippage check: abort tracking if fill deviates > MAX_SLIPPAGE_PCT.
  - emergency_stop(): cancel all open orders + market-close all positions.

Usage:
  python src/trading/live_trader.py --mode paper
  python src/trading/live_trader.py --mode dry
  python src/trading/live_trader.py --mode live --monitor-only
  python src/trading/live_trader.py --mode live --emergency-stop
  python src/trading/live_trader.py --pause      (create pause.flag)
  python src/trading/live_trader.py --resume     (remove pause.flag)
"""

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import ccxt
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.risk_manager import TradeDecision
from trading.trade_logger import (
    log_approved_trade,
    close_trade,
    get_open_trades,
)
from trading.notifier import Notifier

# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

COINBASE_FEE_MAKER   = 0.002   # 0.2% maker fee (limit postOnly)
COINBASE_FEE_TAKER   = 0.004   # 0.4% taker fee (market / fallback)
MAX_SLIPPAGE_PCT     = float(os.getenv("MAX_SLIPPAGE_PCT", "0.01"))      # 1%
POLL_INTERVAL_SEC    = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
FILL_TIMEOUT_SEC     = int(os.getenv("FILL_TIMEOUT_SECONDS", "120"))     # 2 min
FILL_POLL_SEC        = 3                                                  # fetch_order cadence
LIVE_ENABLED_FLAG    = os.getenv("LIVE_TRADING_ENABLED", "false").lower()
USE_SANDBOX          = os.getenv("COINBASE_SANDBOX", "false").lower() == "true"

# Pause-flag: simplest possible kill-switch. Touch this file to pause.
PAUSE_FLAG = ROOT / "pause.flag"

# Legacy kill-switch path (used by API server; kept for backward compat)
KILL_SWITCH_FILE = ROOT / "data" / "TRADING_PAUSED"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pause-flag helpers (importable by api_server.py)
# ---------------------------------------------------------------------------

def is_paused() -> bool:
    """True if either pause.flag or data/TRADING_PAUSED exists."""
    return PAUSE_FLAG.exists() or KILL_SWITCH_FILE.exists()


def pause_trading(reason: str = "") -> None:
    """Create pause.flag to halt new entry orders."""
    PAUSE_FLAG.write_text(reason or "paused")
    logger.warning(f"[KillSwitch] Trading PAUSED -- {reason or 'pause.flag created'}")


def resume_trading() -> None:
    """Remove pause.flag (and legacy TRADING_PAUSED) to allow new entries."""
    for f in (PAUSE_FLAG, KILL_SWITCH_FILE):
        if f.exists():
            f.unlink()
    logger.info("[KillSwitch] Trading RESUMED -- pause.flag cleared")


# ---------------------------------------------------------------------------
# Exchange factory
# ---------------------------------------------------------------------------

def _build_exchange(mode: str) -> "ccxt.coinbaseadvanced | None":
    """
    Return an authenticated ccxt.coinbaseadvanced instance.

    paper mode: returns None (no exchange needed).
    dry mode:   builds instance without credentials (public data only).
    live mode:  builds with credentials; raises if they are missing.
    """
    if mode == "paper":
        return None

    api_key    = os.getenv("COINBASE_API_KEY", "")
    api_secret = os.getenv("COINBASE_SECRET_KEY", "")

    if mode == "live" and (not api_key or not api_secret):
        raise EnvironmentError(
            "COINBASE_API_KEY and COINBASE_SECRET_KEY must be set in .env "
            "to use live mode."
        )

    params = {
        "apiKey":          api_key,
        "secret":          api_secret,
        "enableRateLimit": True,
    }

    if USE_SANDBOX:
        # Coinbase Advanced Trade sandbox
        params["urls"] = {
            "api": {
                "public":  "https://api-public.sandbox.advanced.coinbase.com",
                "private": "https://api-public.sandbox.advanced.coinbase.com",
            }
        }
        logger.info("[LiveTrader] Using Coinbase Advanced Trade SANDBOX")

    try:
        exchange = ccxt.coinbaseadvanced(params)
    except AttributeError:
        # Older ccxt versions expose it as 'coinbase' with advanced option
        logger.warning(
            "[LiveTrader] ccxt.coinbaseadvanced not found; "
            "falling back to ccxt.coinbase with advanced=True"
        )
        exchange = ccxt.coinbase(params)
        exchange.options["advanced"] = True

    return exchange


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Multi-mode execution layer.

    Parameters
    ----------
    mode : str
        "paper"  -- fully simulated, no exchange
        "dry"    -- real prices, no orders placed
        "live"   -- real orders on Coinbase Advanced Trade
    fill_timeout_sec : int
        Seconds to wait for a limit order to fill before cancelling.
    """

    def __init__(self, mode: str = "dry", fill_timeout_sec: int = FILL_TIMEOUT_SEC):
        if mode not in ("paper", "dry", "live"):
            raise ValueError(f"mode must be paper|dry|live, got {mode!r}")

        self.mode             = mode
        self.fill_timeout_sec = fill_timeout_sec
        self.notifier         = Notifier()

        if mode == "live" and LIVE_ENABLED_FLAG != "true":
            raise RuntimeError(
                "Live trading is DISABLED. "
                "Set LIVE_TRADING_ENABLED=true in your .env to enable."
            )

        self.exchange = _build_exchange(mode)

        # In-memory position cache: trade_id -> position dict
        self._positions: dict = {}

        logger.info(f"[LiveTrader] Mode: {mode.upper()}")
        if mode == "live" and USE_SANDBOX:
            logger.info("[LiveTrader] Targeting Coinbase SANDBOX (not real money)")

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def startup(self):
        """Load open trades from SQLite and reconcile with exchange."""
        self.reconcile_with_exchange()

    def place_entry_order(self, decision: TradeDecision) -> dict:
        """
        Place a limit entry order (postOnly) for an approved TradeDecision.

        Flow:
          1. Check pause.flag
          2. Place limit order with postOnly=True
          3. Poll fetch_order until filled, timeout, or cancelled
          4. On timeout: cancel the order; if partially filled track that qty
          5. Slippage check on actual fill price
          6. Log to SQLite + notify

        Returns a fill_info dict. 'aborted' key is True if nothing was tracked.
        """
        if not decision.approved:
            raise ValueError("Cannot place order for a non-approved TradeDecision")

        # --- Pause check ---
        if is_paused():
            reason = ""
            for f in (PAUSE_FLAG, KILL_SWITCH_FILE):
                if f.exists():
                    try:
                        reason = f.read_text().strip()
                    except Exception:
                        pass
                    break
            logger.warning(
                f"[LiveTrader] PAUSED -- skipping {decision.direction} "
                f"{decision.pair}  reason: {reason or 'pause.flag exists'}"
            )
            return {"aborted": True, "reason": "paused",
                    "pair": decision.pair, "direction": decision.direction}

        pair        = decision.pair
        direction   = decision.direction          # "BUY" or "SELL"
        side        = direction.lower()           # "buy" or "sell"
        expected_px = decision.entry_price
        qty         = decision.position_size_asset
        size_usd    = decision.position_size_usd
        trade_id    = "LIVE_" + str(uuid.uuid4())

        # --- Place or simulate ---
        if self.mode == "paper":
            fill_price, fill_qty, order_id = self._simulate_fill(
                side, expected_px, qty
            )
            fee_rate = COINBASE_FEE_MAKER

        elif self.mode == "dry":
            current_px = self._fetch_current_price(pair)
            logger.info(
                f"[DRY] Would place {side.upper()} limit postOnly: "
                f"{pair}  limit={expected_px:.4f}  qty={qty:.6f} "
                f"(~${size_usd:.2f})  current={current_px:.4f}"
            )
            fill_price = expected_px
            fill_qty   = qty
            order_id   = "DRY_" + str(uuid.uuid4())
            fee_rate   = COINBASE_FEE_MAKER

        else:  # live
            result = self._place_limit_order(pair, side, qty, expected_px)
            if result is None:
                # Order placed but zero fill (timed out + cancelled, no partial)
                logger.warning(
                    f"[LiveTrader] Limit order for {pair} {side} expired with "
                    f"no fill. Signal skipped."
                )
                return {"aborted": True, "reason": "no_fill",
                        "pair": pair, "direction": direction}

            order_id, fill_price, fill_qty, fee_rate = result

        # --- Slippage check ---
        slippage    = abs(fill_price - expected_px) / expected_px if expected_px else 0
        slippage_ok = slippage <= MAX_SLIPPAGE_PCT

        if not slippage_ok:
            logger.error(
                f"[LiveTrader] Slippage {slippage*100:.3f}% exceeds limit "
                f"{MAX_SLIPPAGE_PCT*100:.1f}% (expected={expected_px:.4f}, "
                f"fill={fill_price:.4f}). Trade NOT tracked."
            )
            return {
                "trade_id":     trade_id,
                "order_id":     order_id,
                "pair":         pair,
                "direction":    direction,
                "fill_price":   fill_price,
                "fill_qty":     fill_qty,
                "slippage_pct": round(slippage * 100, 4),
                "slippage_ok":  False,
                "aborted":      True,
            }

        actual_usd  = fill_price * fill_qty
        fee_usd     = actual_usd * fee_rate

        logger.info(
            f"[LiveTrader] FILLED {direction} {pair}  order={order_id}  "
            f"fill_px={fill_price:.4f}  qty={fill_qty:.6f}  "
            f"size=${actual_usd:.2f}  fee=${fee_usd:.2f}  "
            f"slippage={slippage*100:.3f}%  mode={self.mode}"
        )
        self.notifier.trade_opened(
            pair=pair,
            direction=direction,
            size_usd=actual_usd,
            entry=fill_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            confidence=0,
            trade_id=trade_id,
        )

        # --- Persist to SQLite ---
        decision_dict = decision.to_dict()
        decision_dict["entry_price"]         = fill_price
        decision_dict["position_size_usd"]   = round(actual_usd, 2)
        decision_dict["position_size_asset"] = fill_qty

        log_approved_trade(
            trade_id=trade_id,
            decision=decision_dict,
            recommendation={"confidence": None, "decision": direction},
            market_data={"pair": pair, "price": fill_price, "mode": self.mode},
        )

        # --- Memory cache ---
        self._positions[trade_id] = {
            "trade_id":      trade_id,
            "order_id":      order_id,
            "pair":          pair,
            "direction":     direction,
            "entry_price":   fill_price,
            "stop_loss":     decision.stop_loss,
            "take_profit":   decision.take_profit,
            "size_usd":      actual_usd,
            "qty":           fill_qty,
            "fee_entry_usd": fee_usd,
            "opened_at":     datetime.utcnow().isoformat(),
            "mode":          self.mode,
        }

        return {
            "trade_id":     trade_id,
            "order_id":     order_id,
            "pair":         pair,
            "direction":    direction,
            "fill_price":   fill_price,
            "fill_qty":     fill_qty,
            "size_usd":     actual_usd,
            "fee_usd":      fee_usd,
            "slippage_pct": round(slippage * 100, 4),
            "slippage_ok":  True,
            "aborted":      False,
            "mode":         self.mode,
        }

    def monitor_positions(self, interval_seconds: int = POLL_INTERVAL_SEC):
        """
        Blocking SL/TP monitor loop. Polls every interval_seconds and
        closes any position that has hit SL or TP.
        """
        logger.info(
            f"[LiveTrader] SL/TP monitor started (poll every {interval_seconds}s, "
            f"mode={self.mode})"
        )
        while True:
            try:
                # Re-check pause flag each cycle — allow mid-run pausing
                if is_paused():
                    logger.info("[LiveTrader] Paused -- skipping SL/TP check")
                else:
                    self._check_sl_tp()
            except KeyboardInterrupt:
                logger.info("[LiveTrader] Monitor loop stopped.")
                break
            except Exception as exc:
                logger.error(f"[LiveTrader] Monitor error: {exc}", exc_info=True)
            time.sleep(interval_seconds)

    def close_position(self, trade_id: str, reason: str = "manual") -> dict:
        """
        Exit a position. Uses market order in live mode; simulated in paper/dry.

        reason: "stopped_out" | "took_profit" | "manual" | "emergency"
        """
        pos = self._positions.get(trade_id)
        if pos is None:
            for row in get_open_trades():
                if row["trade_id"] == trade_id:
                    pos = {
                        "trade_id":      row["trade_id"],
                        "pair":          row["pair"],
                        "direction":     row["direction"],
                        "entry_price":   row["entry_price"],
                        "stop_loss":     row["stop_loss"],
                        "take_profit":   row["take_profit"],
                        "size_usd":      row["position_size_usd"],
                        "qty":           row["position_size_asset"],
                        "fee_entry_usd": 0.0,
                        "opened_at":     row["opened_at"],
                        "mode":          self.mode,
                    }
                    break
            if pos is None:
                raise KeyError(f"trade_id {trade_id!r} not found in memory or DB")

        pair      = pos["pair"]
        qty       = pos["qty"]
        direction = pos["direction"]
        exit_side = "sell" if direction == "BUY" else "buy"

        if self.mode == "paper":
            exit_price = self._simulate_price(pair, pos["entry_price"])
            order_id   = "PAPER_EXIT_" + str(uuid.uuid4())
            fee_rate   = COINBASE_FEE_TAKER
            logger.info(
                f"[PAPER] Simulated {exit_side.upper()} close: "
                f"{direction} {pair}  qty={qty:.6f}  "
                f"sim_px={exit_price:.4f}  reason={reason}"
            )

        elif self.mode == "dry":
            exit_price = self._fetch_current_price(pair)
            order_id   = "DRY_EXIT_" + str(uuid.uuid4())
            fee_rate   = COINBASE_FEE_TAKER
            logger.info(
                f"[DRY] Would place {exit_side.upper()} market close: "
                f"{direction} {pair}  qty={qty:.6f}  "
                f"approx_px={exit_price:.4f}  reason={reason}"
            )

        else:  # live — market order for exits (speed over fees)
            order_id, exit_price, _ = self._place_market_order(
                pair=pair, side=exit_side, qty=qty,
            )
            fee_rate = COINBASE_FEE_TAKER

        fee_exit_usd = exit_price * qty * fee_rate

        if reason in ("stopped_out", "took_profit"):
            outcome = reason
        else:
            if direction == "BUY":
                outcome = "win" if exit_price > pos["entry_price"] else "loss"
            else:
                outcome = "win" if exit_price < pos["entry_price"] else "loss"

        if direction == "BUY":
            gross_pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            gross_pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

        gross_pnl_usd = pos["size_usd"] * gross_pnl_pct
        total_fees    = pos.get("fee_entry_usd", 0.0) + fee_exit_usd
        net_pnl_usd   = gross_pnl_usd - total_fees

        close_trade(trade_id=trade_id, exit_price=exit_price, outcome=outcome)

        logger.info(
            f"[LiveTrader] CLOSED {direction} {pair}  "
            f"reason={reason}  entry={pos['entry_price']:.4f}  "
            f"exit={exit_price:.4f}  gross=${gross_pnl_usd:+.2f}  "
            f"fees=${total_fees:.2f}  net=${net_pnl_usd:+.2f}  "
            f"outcome={outcome}  mode={self.mode}"
        )
        self.notifier.trade_closed(
            pair=pair,
            direction=direction,
            entry=pos["entry_price"],
            exit_price=exit_price,
            net_pnl_usd=net_pnl_usd,
            outcome=outcome,
            reason=reason,
            trade_id=trade_id,
        )

        self._positions.pop(trade_id, None)

        return {
            "trade_id":      trade_id,
            "order_id":      order_id,
            "pair":          pair,
            "direction":     direction,
            "exit_price":    exit_price,
            "gross_pnl_usd": round(gross_pnl_usd, 2),
            "fee_total_usd": round(total_fees, 2),
            "net_pnl_usd":   round(net_pnl_usd, 2),
            "outcome":       outcome,
            "reason":        reason,
            "mode":          self.mode,
        }

    def emergency_stop(self) -> list:
        """Cancel all open orders + market-close all positions."""
        logger.warning(
            f"[LiveTrader] EMERGENCY STOP -- closing all positions "
            f"(mode={self.mode})"
        )
        self.notifier.error("EMERGENCY STOP triggered -- closing all positions")

        results = []

        if self.mode == "live":
            try:
                open_orders = self.exchange.fetch_open_orders()
                for o in open_orders:
                    try:
                        self.exchange.cancel_order(o["id"], o["symbol"])
                        logger.info(
                            f"[LiveTrader] Cancelled order {o['id']} ({o['symbol']})"
                        )
                    except Exception as e:
                        logger.error(
                            f"[LiveTrader] Could not cancel order {o['id']}: {e}"
                        )
            except Exception as e:
                logger.error(f"[LiveTrader] Could not fetch open orders: {e}")

        for trade_id in list(self._positions.keys()):
            try:
                results.append(self.close_position(trade_id, reason="emergency"))
            except Exception as e:
                logger.error(f"[LiveTrader] Emergency close failed for {trade_id}: {e}")

        closed_ids = {r["trade_id"] for r in results}
        for row in get_open_trades():
            tid = row["trade_id"]
            if tid in closed_ids:
                continue
            try:
                self._positions[tid] = {
                    "trade_id":      tid,
                    "pair":          row["pair"],
                    "direction":     row["direction"],
                    "entry_price":   row["entry_price"],
                    "stop_loss":     row["stop_loss"],
                    "take_profit":   row["take_profit"],
                    "size_usd":      row["position_size_usd"],
                    "qty":           row["position_size_asset"],
                    "fee_entry_usd": 0.0,
                    "opened_at":     row["opened_at"],
                    "mode":          self.mode,
                }
                results.append(self.close_position(tid, reason="emergency"))
            except Exception as e:
                logger.error(
                    f"[LiveTrader] Emergency close (DB) failed for {tid}: {e}"
                )

        logger.warning(
            f"[LiveTrader] Emergency stop complete -- closed {len(results)} positions"
        )
        return results

    def reconcile_with_exchange(self) -> dict:
        """
        Startup: sync DB open trades with exchange balances.
        In paper/dry mode, just load DB trades into memory.
        """
        logger.info(f"[LiveTrader] Reconciling (mode={self.mode}) ...")
        db_open = get_open_trades()

        if self.mode in ("paper", "dry"):
            for row in db_open:
                self._positions[row["trade_id"]] = {
                    "trade_id":      row["trade_id"],
                    "pair":          row["pair"],
                    "direction":     row["direction"],
                    "entry_price":   row["entry_price"],
                    "stop_loss":     row["stop_loss"],
                    "take_profit":   row["take_profit"],
                    "size_usd":      row["position_size_usd"],
                    "qty":           row["position_size_asset"],
                    "fee_entry_usd": 0.0,
                    "opened_at":     row["opened_at"],
                    "mode":          self.mode,
                }
            logger.info(
                f"[LiveTrader] Loaded {len(db_open)} open trades from DB"
            )
            return {
                "db_open": len(db_open),
                "loaded": len(db_open),
                "discrepancies": [],
            }

        # live: verify balances against DB
        try:
            balance = self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"[LiveTrader] Could not fetch balance: {e}")
            balance = {"total": {}}

        discrepancies = []
        loaded = 0

        for row in db_open:
            tid          = row["trade_id"]
            pair         = row["pair"]
            base         = pair.split("/")[0]
            exchange_qty = balance.get("total", {}).get(base, 0.0)
            db_qty       = row["position_size_asset"]

            if exchange_qty >= db_qty * 0.80:
                self._positions[tid] = {
                    "trade_id":      tid,
                    "pair":          pair,
                    "direction":     row["direction"],
                    "entry_price":   row["entry_price"],
                    "stop_loss":     row["stop_loss"],
                    "take_profit":   row["take_profit"],
                    "size_usd":      row["position_size_usd"],
                    "qty":           db_qty,
                    "fee_entry_usd": 0.0,
                    "opened_at":     row["opened_at"],
                    "mode":          self.mode,
                }
                loaded += 1
                logger.info(
                    f"[LiveTrader] Reconciled {pair} {row['direction']}  "
                    f"db_qty={db_qty:.6f}  exchange_qty={exchange_qty:.6f}"
                )
            else:
                logger.warning(
                    f"[LiveTrader] Discrepancy for {tid}: DB expects {db_qty:.6f} "
                    f"{base} but exchange shows {exchange_qty:.6f}"
                )
                discrepancies.append({
                    "trade_id":    tid,
                    "pair":        pair,
                    "db_qty":      db_qty,
                    "exchange_qty": exchange_qty,
                })

        logger.info(
            f"[LiveTrader] Reconcile complete: {loaded}/{len(db_open)} loaded, "
            f"{len(discrepancies)} discrepancies"
        )
        return {
            "db_open":       len(db_open),
            "loaded":        loaded,
            "discrepancies": discrepancies,
        }

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _place_limit_order(
        self,
        pair:        str,
        side:        str,
        qty:         float,
        limit_price: float,
    ) -> "tuple | None":
        """
        Place a postOnly limit order and poll until filled, timed out, or
        cancelled externally.

        Returns (order_id, fill_price, fill_qty, fee_rate) on (partial) fill,
        or None if the order expired with zero fill.
        """
        logger.info(
            f"[LiveTrader] Placing {side.upper()} limit postOnly: "
            f"{pair}  price={limit_price:.4f}  qty={qty:.6f}"
        )

        try:
            order = self.exchange.create_order(
                symbol=pair,
                type="limit",
                side=side,
                amount=qty,
                price=limit_price,
                params={"postOnly": True},
            )
        except ccxt.OrderImmediatelyFillable:
            # postOnly rejected because it would cross the book — skip
            logger.warning(
                f"[LiveTrader] postOnly limit would have been a taker fill "
                f"({pair} {side} @ {limit_price:.4f}). Skipping."
            )
            return None
        except Exception as e:
            logger.error(f"[LiveTrader] Limit order placement failed: {e}")
            raise

        order_id  = order["id"]
        deadline  = time.monotonic() + self.fill_timeout_sec
        fill_qty  = 0.0
        fill_price = limit_price  # will be updated from fetch_order

        logger.info(
            f"[LiveTrader] Limit order {order_id} placed. "
            f"Waiting up to {self.fill_timeout_sec}s for fill ..."
        )

        while time.monotonic() < deadline:
            time.sleep(FILL_POLL_SEC)
            try:
                fetched = self.exchange.fetch_order(order_id, pair)
            except Exception as e:
                logger.warning(f"[LiveTrader] fetch_order error: {e}")
                continue

            status      = fetched.get("status", "")
            fill_qty    = float(fetched.get("filled") or 0)
            avg_px      = fetched.get("average") or fetched.get("price") or limit_price
            fill_price  = float(avg_px)

            if status == "closed" and fill_qty > 0:
                logger.info(
                    f"[LiveTrader] Order {order_id} FILLED: "
                    f"qty={fill_qty:.6f}  avg_px={fill_price:.4f}"
                )
                return order_id, fill_price, fill_qty, COINBASE_FEE_MAKER

            if status == "canceled":
                if fill_qty > 0:
                    logger.warning(
                        f"[LiveTrader] Order {order_id} cancelled externally "
                        f"with partial fill: qty={fill_qty:.6f}  px={fill_price:.4f}"
                    )
                    return order_id, fill_price, fill_qty, COINBASE_FEE_MAKER
                logger.warning(
                    f"[LiveTrader] Order {order_id} was cancelled externally "
                    f"with no fill."
                )
                return None

        # Timeout: cancel the order
        logger.warning(
            f"[LiveTrader] Limit order {order_id} not filled within "
            f"{self.fill_timeout_sec}s. Cancelling."
        )
        try:
            cancelled = self.exchange.cancel_order(order_id, pair)
            fill_qty  = float(cancelled.get("filled") or 0)
            avg_px    = cancelled.get("average") or cancelled.get("price") or limit_price
            fill_price = float(avg_px)
        except Exception as e:
            logger.error(f"[LiveTrader] Could not cancel order {order_id}: {e}")
            # Try one last fetch to see partial fill
            try:
                fetched   = self.exchange.fetch_order(order_id, pair)
                fill_qty  = float(fetched.get("filled") or 0)
                avg_px    = fetched.get("average") or fetched.get("price") or limit_price
                fill_price = float(avg_px)
            except Exception:
                fill_qty = 0.0

        if fill_qty > 0:
            logger.info(
                f"[LiveTrader] Partial fill on cancel: "
                f"qty={fill_qty:.6f}  px={fill_price:.4f}"
            )
            return order_id, fill_price, fill_qty, COINBASE_FEE_MAKER

        return None

    def _place_market_order(self, pair: str, side: str, qty: float) -> tuple:
        """
        Place a market order (used for emergency exits in live mode).
        Returns (order_id, fill_price, fill_qty).
        """
        logger.info(
            f"[LiveTrader] Placing {side.upper()} MARKET order: "
            f"{pair}  qty={qty:.6f}"
        )
        order = self.exchange.create_order(
            symbol=pair, type="market", side=side, amount=qty,
        )
        order_id  = order["id"]
        fill_price = None
        fill_qty   = qty

        for attempt in range(5):
            try:
                fetched   = self.exchange.fetch_order(order_id, pair)
                avg_px    = fetched.get("average")
                filled_q  = fetched.get("filled", qty)
                if avg_px and float(avg_px) > 0:
                    fill_price = float(avg_px)
                    fill_qty   = float(filled_q)
                    break
            except Exception as e:
                logger.warning(f"[LiveTrader] fetch_order attempt {attempt+1}: {e}")
            time.sleep(1)

        if fill_price is None:
            fill_price = float(
                order.get("average") or order.get("price")
                or self._fetch_current_price(pair)
            )
            logger.warning(f"[LiveTrader] Using fallback fill price: {fill_price}")

        return order_id, fill_price, fill_qty

    def _fetch_current_price(self, pair: str) -> float:
        """Return last traded price for a pair."""
        if self.exchange is None:
            return 0.0
        try:
            return float(self.exchange.fetch_ticker(pair)["last"])
        except Exception as e:
            logger.error(f"[LiveTrader] Could not fetch price for {pair}: {e}")
            return 0.0

    def _simulate_fill(
        self, side: str, limit_price: float, qty: float
    ) -> tuple:
        """Paper mode: instant simulated fill at limit price."""
        order_id = "PAPER_" + str(uuid.uuid4())
        logger.info(
            f"[PAPER] Simulated {side.upper()} fill: "
            f"limit={limit_price:.4f}  qty={qty:.6f}"
        )
        return limit_price, qty, order_id

    def _simulate_price(self, pair: str, entry_price: float) -> float:
        """Paper mode: return entry price as exit (caller adds noise if desired)."""
        return entry_price

    def _check_sl_tp(self):
        """One SL/TP polling pass: close positions that hit their levels."""
        if not self._positions:
            return

        pairs  = list({pos["pair"] for pos in self._positions.values()})
        prices = {}

        if self.mode == "paper":
            # Paper: reuse cached entry prices (no market data needed)
            for pos in self._positions.values():
                prices[pos["pair"]] = pos["entry_price"]
        else:
            prices = {p: self._fetch_current_price(p) for p in pairs}

        for trade_id, pos in list(self._positions.items()):
            price = prices.get(pos["pair"], 0)
            if price <= 0:
                continue

            direction = pos["direction"]
            sl        = pos["stop_loss"]
            tp        = pos["take_profit"]

            if direction == "BUY":
                if price <= sl:
                    logger.warning(
                        f"[LiveTrader] STOP-LOSS HIT  {pos['pair']}  "
                        f"price={price:.4f}  sl={sl:.4f}"
                    )
                    self.close_position(trade_id, reason="stopped_out")
                elif price >= tp:
                    logger.info(
                        f"[LiveTrader] TAKE-PROFIT HIT  {pos['pair']}  "
                        f"price={price:.4f}  tp={tp:.4f}"
                    )
                    self.close_position(trade_id, reason="took_profit")
            else:
                if price >= sl:
                    logger.warning(
                        f"[LiveTrader] STOP-LOSS HIT  {pos['pair']}  "
                        f"price={price:.4f}  sl={sl:.4f}"
                    )
                    self.close_position(trade_id, reason="stopped_out")
                elif price <= tp:
                    logger.info(
                        f"[LiveTrader] TAKE-PROFIT HIT  {pos['pair']}  "
                        f"price={price:.4f}  tp={tp:.4f}"
                    )
                    self.close_position(trade_id, reason="took_profit")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Oracle Live Trader")
    parser.add_argument(
        "--mode", choices=["paper", "dry", "live"], default="dry",
        help=(
            "paper = fully simulated, no exchange; "
            "dry = real prices, no orders; "
            "live = real orders on Coinbase Advanced Trade"
        ),
    )
    parser.add_argument(
        "--monitor-only", action="store_true",
        help="Start the SL/TP monitor loop only (no new entry orders from CLI)"
    )
    parser.add_argument(
        "--emergency-stop", action="store_true",
        help="Close ALL open positions immediately and exit"
    )
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL_SEC,
        help=f"SL/TP poll interval in seconds (default {POLL_INTERVAL_SEC})"
    )
    parser.add_argument(
        "--fill-timeout", type=int, default=FILL_TIMEOUT_SEC,
        help=f"Seconds to wait for limit order fill (default {FILL_TIMEOUT_SEC})"
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Create pause.flag to halt all new entries and exit"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Remove pause.flag (and legacy TRADING_PAUSED) and exit"
    )
    args = parser.parse_args()

    # Pause management (no trader instance needed)
    if args.pause:
        pause_trading("paused via CLI")
        print(f"pause.flag created at: {PAUSE_FLAG}")
        sys.exit(0)

    if args.resume:
        resume_trading()
        print("pause.flag cleared. Trading can resume.")
        sys.exit(0)

    trader = LiveTrader(mode=args.mode, fill_timeout_sec=args.fill_timeout)
    trader.startup()

    if args.emergency_stop:
        results = trader.emergency_stop()
        print(f"Emergency stop complete. Closed {len(results)} positions.")
        sys.exit(0)

    if args.monitor_only:
        trader.monitor_positions(interval_seconds=args.interval)
