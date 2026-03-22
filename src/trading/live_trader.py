"""
Live Trader
Executes approved TradeDecision objects against Coinbase Advanced Trade
via CCXT. Manages SL/TP monitoring via polling loop (Coinbase spot does
not support native conditional orders for all pairs).

Safety gates:
  - LIVE_TRADING_ENABLED must be 'true' in .env to place real orders
  - Kill-switch file: create data/TRADING_PAUSED to halt new entries instantly
    (delete the file or call POST /admin/resume to resume)
  - --dry-run flag logs "would place" without touching the exchange
  - Max slippage check: abort tracking if fill deviates > MAX_SLIPPAGE_PCT
  - emergency_stop(): cancel all open orders + market-close all positions

Usage:
  python src/trading/live_trader.py --dry-run
  python src/trading/live_trader.py --monitor-only
  python src/trading/live_trader.py --emergency-stop
  python src/trading/live_trader.py --pause     (create kill-switch file)
  python src/trading/live_trader.py --resume    (remove kill-switch file)
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

load_dotenv()

COINBASE_FEE_TAKER = 0.004   # 0.4% taker fee per side
MAX_SLIPPAGE_PCT   = float(os.getenv("MAX_SLIPPAGE_PCT", "0.01"))     # 1%
POLL_INTERVAL_SEC  = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
LIVE_ENABLED_FLAG  = os.getenv("LIVE_TRADING_ENABLED", "false").lower()

# Kill-switch: creating this file pauses all new entries immediately
KILL_SWITCH_FILE = Path(__file__).parent.parent.parent / "data" / "TRADING_PAUSED"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kill-switch helpers (usable from outside this module too)
# ---------------------------------------------------------------------------

def is_paused() -> bool:
    """Return True if the kill-switch file exists."""
    return KILL_SWITCH_FILE.exists()


def pause_trading(reason: str = "") -> None:
    """Create the kill-switch file to halt new entry orders."""
    KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_FILE.write_text(reason or "paused")
    logger.warning(f"[KillSwitch] Trading PAUSED -- {reason or 'kill-switch file created'}")


def resume_trading() -> None:
    """Remove the kill-switch file to allow new entry orders."""
    if KILL_SWITCH_FILE.exists():
        KILL_SWITCH_FILE.unlink()
    logger.info("[KillSwitch] Trading RESUMED")


# ---------------------------------------------------------------------------
# Exchange factory
# ---------------------------------------------------------------------------

def _build_exchange(dry_run: bool = False) -> ccxt.coinbase:
    """
    Build an authenticated ccxt.coinbase instance.
    In dry-run mode the instance is still built so prices/balances can be
    fetched, but order-placement calls are intercepted before reaching it.
    """
    api_key    = os.getenv("COINBASE_API_KEY", "")
    api_secret = os.getenv("COINBASE_SECRET_KEY", "")

    if not dry_run and (not api_key or not api_secret):
        raise EnvironmentError(
            "COINBASE_API_KEY and COINBASE_SECRET_KEY must be set in .env "
            "for live trading."
        )

    exchange = ccxt.coinbase({
        "apiKey":          api_key,
        "secret":          api_secret,
        "enableRateLimit": True,
        "options": {
            "advanced": True,   # use Advanced Trade API endpoints
        },
    })
    return exchange


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Live execution layer.

    Parameters
    ----------
    dry_run : bool
        If True, goes through full logic but never calls exchange order
        methods. Prices, balances, and positions are still fetched.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run  = dry_run
        self.notifier = Notifier()

        # Kill-switch: must be explicitly enabled for real orders
        if not dry_run and LIVE_ENABLED_FLAG != "true":
            raise RuntimeError(
                "Live trading is DISABLED. "
                "Set LIVE_TRADING_ENABLED=true in your .env file to enable."
            )

        self.exchange = _build_exchange(dry_run=dry_run)

        # In-memory position cache: trade_id -> position dict
        self._positions: dict = {}

        if dry_run:
            logger.info("[LiveTrader] DRY-RUN mode -- no real orders will be placed")
        else:
            logger.info("[LiveTrader] LIVE mode -- real orders WILL be placed on Coinbase")

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def startup(self):
        """
        Load open trades from SQLite and reconcile with exchange.
        Call once before the monitor loop starts.
        """
        self.reconcile_with_exchange()

    def place_entry_order(self, decision: TradeDecision) -> dict:
        """
        Place a market entry order for an approved TradeDecision.

        Returns a fill_info dict:
          {
            "trade_id":    str,
            "order_id":    str,
            "pair":        str,
            "direction":   str,
            "fill_price":  float,
            "fill_qty":    float,
            "size_usd":    float,
            "fee_usd":     float,
            "slippage_pct": float,
            "slippage_ok": bool,
            "aborted":     bool,
          }
        """
        if not decision.approved:
            raise ValueError("Cannot place order for a non-approved TradeDecision")

        # Respect kill-switch: refuse new entries while paused
        if is_paused():
            reason = KILL_SWITCH_FILE.read_text().strip() if KILL_SWITCH_FILE.exists() else ""
            logger.warning(
                f"[LiveTrader] Kill-switch is ACTIVE -- skipping {decision.direction} "
                f"{decision.pair}  reason: {reason or 'TRADING_PAUSED file exists'}"
            )
            return {
                "aborted": True,
                "reason": "kill_switch",
                "pair": decision.pair,
                "direction": decision.direction,
            }

        pair        = decision.pair
        direction   = decision.direction.lower()   # "buy" or "sell"
        size_usd    = decision.position_size_usd
        expected_px = decision.entry_price
        qty         = decision.position_size_asset

        trade_id = "LIVE_" + str(uuid.uuid4())

        # --- Place or simulate ---
        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would place {direction.upper()} market order: "
                f"{pair}  qty={qty:.6f}  (~${size_usd:.2f})  "
                f"expected_px={expected_px:.4f}"
            )
            fill_price = expected_px
            fill_qty   = qty
            order_id   = "DRY_" + str(uuid.uuid4())
        else:
            order_id, fill_price, fill_qty = self._place_market_order(
                pair=pair, side=direction, qty=qty,
            )

        # --- Slippage check ---
        slippage    = abs(fill_price - expected_px) / expected_px if expected_px else 0
        slippage_ok = slippage <= MAX_SLIPPAGE_PCT

        if not slippage_ok:
            logger.error(
                f"[LiveTrader] Slippage {slippage*100:.3f}% exceeds limit "
                f"{MAX_SLIPPAGE_PCT*100:.1f}% (expected={expected_px:.4f}, "
                f"fill={fill_price:.4f}). Trade {trade_id} NOT tracked."
            )
            return {
                "trade_id":    trade_id,
                "order_id":    order_id,
                "pair":        pair,
                "direction":   decision.direction,
                "fill_price":  fill_price,
                "fill_qty":    fill_qty,
                "size_usd":    fill_price * fill_qty,
                "fee_usd":     fill_price * fill_qty * COINBASE_FEE_TAKER,
                "slippage_pct": round(slippage * 100, 4),
                "slippage_ok": False,
                "aborted":     True,
            }

        actual_usd = fill_price * fill_qty
        fee_usd    = actual_usd * COINBASE_FEE_TAKER

        logger.info(
            f"[LiveTrader] FILLED {decision.direction} {pair}  "
            f"order={order_id}  fill_px={fill_price:.4f}  "
            f"qty={fill_qty:.6f}  size=${actual_usd:.2f}  "
            f"fee=${fee_usd:.2f}  slippage={slippage*100:.3f}%"
        )
        self.notifier.trade_opened(
            pair=pair,
            direction=decision.direction,
            size_usd=actual_usd,
            entry=fill_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            confidence=0,
            trade_id=trade_id,
        )

        # --- Persist to SQLite (use actual fill price as entry) ---
        decision_dict = decision.to_dict()
        decision_dict["entry_price"]        = fill_price
        decision_dict["position_size_usd"]  = round(actual_usd, 2)
        decision_dict["position_size_asset"]= fill_qty

        log_approved_trade(
            trade_id=trade_id,
            decision=decision_dict,
            recommendation={"confidence": None, "decision": decision.direction},
            market_data={"pair": pair, "price": fill_price},
        )

        # --- Cache in memory ---
        self._positions[trade_id] = {
            "trade_id":      trade_id,
            "order_id":      order_id,
            "pair":          pair,
            "direction":     decision.direction,
            "entry_price":   fill_price,
            "stop_loss":     decision.stop_loss,
            "take_profit":   decision.take_profit,
            "size_usd":      actual_usd,
            "qty":           fill_qty,
            "fee_entry_usd": fee_usd,
            "opened_at":     datetime.utcnow().isoformat(),
        }

        return {
            "trade_id":    trade_id,
            "order_id":    order_id,
            "pair":        pair,
            "direction":   decision.direction,
            "fill_price":  fill_price,
            "fill_qty":    fill_qty,
            "size_usd":    actual_usd,
            "fee_usd":     fee_usd,
            "slippage_pct": round(slippage * 100, 4),
            "slippage_ok": True,
            "aborted":     False,
        }

    def monitor_positions(self, interval_seconds: int = POLL_INTERVAL_SEC):
        """
        Blocking SL/TP monitor loop. Polls current prices every
        interval_seconds and closes any position that has hit SL or TP.

        Run this in a background thread after startup().
        """
        logger.info(
            f"[LiveTrader] SL/TP monitor started (poll every {interval_seconds}s)"
        )
        while True:
            try:
                self._check_sl_tp()
            except KeyboardInterrupt:
                logger.info("[LiveTrader] Monitor loop stopped.")
                break
            except Exception as exc:
                logger.error(f"[LiveTrader] Monitor error: {exc}", exc_info=True)
            time.sleep(interval_seconds)

    def close_position(self, trade_id: str, reason: str = "manual") -> dict:
        """
        Exit a position at market price, log the outcome, and remove from cache.

        Parameters
        ----------
        trade_id : str
        reason   : "stopped_out" | "took_profit" | "manual" | "emergency"
        """
        pos = self._positions.get(trade_id)
        if pos is None:
            # Try loading from DB
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
                    }
                    break
            if pos is None:
                raise KeyError(f"trade_id {trade_id!r} not found in memory or DB")

        pair      = pos["pair"]
        qty       = pos["qty"]
        direction = pos["direction"]
        exit_side = "sell" if direction == "BUY" else "buy"

        if self.dry_run:
            exit_price = self._fetch_current_price(pair)
            order_id   = "DRY_EXIT_" + str(uuid.uuid4())
            logger.info(
                f"[DRY-RUN] Would place {exit_side.upper()} to close "
                f"{direction} {pair}  qty={qty:.6f}  reason={reason}  "
                f"approx_px={exit_price:.4f}"
            )
        else:
            order_id, exit_price, _ = self._place_market_order(
                pair=pair, side=exit_side, qty=qty,
            )

        fee_exit_usd = exit_price * qty * COINBASE_FEE_TAKER

        # Determine outcome
        if reason in ("stopped_out", "took_profit"):
            outcome = reason
        else:
            if direction == "BUY":
                outcome = "win" if exit_price > pos["entry_price"] else "loss"
            else:
                outcome = "win" if exit_price < pos["entry_price"] else "loss"

        # PnL (effective prices already account for fees at entry; we subtract
        # entry fee + exit fee from gross here for the log line only)
        if direction == "BUY":
            gross_pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            gross_pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

        gross_pnl_usd = pos["size_usd"] * gross_pnl_pct
        total_fees    = pos.get("fee_entry_usd", 0.0) + fee_exit_usd
        net_pnl_usd   = gross_pnl_usd - total_fees

        # Persist to SQLite
        close_trade(trade_id=trade_id, exit_price=exit_price, outcome=outcome)

        logger.info(
            f"[LiveTrader] CLOSED {direction} {pair}  "
            f"reason={reason}  entry={pos['entry_price']:.4f}  "
            f"exit={exit_price:.4f}  gross=${gross_pnl_usd:+.2f}  "
            f"fees=${total_fees:.2f}  net=${net_pnl_usd:+.2f}  "
            f"outcome={outcome}"
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
        }

    def emergency_stop(self) -> list:
        """
        Close ALL open positions at market and cancel all open orders.
        Nuclear option — use when something is critically wrong.
        """
        logger.warning("[LiveTrader] EMERGENCY STOP -- closing all positions at market")
        self.notifier.error("EMERGENCY STOP triggered -- closing all positions at market")

        results = []

        # 1. Cancel all open limit/stop orders
        if not self.dry_run:
            try:
                open_orders = self.exchange.fetch_open_orders()
                for o in open_orders:
                    try:
                        self.exchange.cancel_order(o["id"], o["symbol"])
                        logger.info(f"[LiveTrader] Cancelled order {o['id']} ({o['symbol']})")
                    except Exception as e:
                        logger.error(f"[LiveTrader] Could not cancel order {o['id']}: {e}")
            except Exception as e:
                logger.error(f"[LiveTrader] Could not fetch open orders: {e}")

        # 2. Close all in-memory positions
        for trade_id in list(self._positions.keys()):
            try:
                results.append(self.close_position(trade_id, reason="emergency"))
            except Exception as e:
                logger.error(f"[LiveTrader] Emergency close failed for {trade_id}: {e}")

        # 3. Close any DB-open trades not in memory
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
                }
                results.append(self.close_position(tid, reason="emergency"))
            except Exception as e:
                logger.error(f"[LiveTrader] Emergency close (DB) failed for {tid}: {e}")

        logger.warning(f"[LiveTrader] Emergency stop complete -- closed {len(results)} positions")
        return results

    def reconcile_with_exchange(self) -> dict:
        """
        On startup: sync local DB open trades with actual exchange balances.

        For each DB-open trade:
          - If the exchange still holds >= 80% of the expected qty: load into
            memory cache (assume position is still open).
          - If not: log a warning (may have been closed externally or manually).

        Returns a summary dict.
        """
        logger.info("[LiveTrader] Reconciling local DB with exchange...")

        db_open = get_open_trades()

        if self.dry_run:
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
                }
            logger.info(f"[DRY-RUN] Reconcile: loaded {len(db_open)} trades from DB")
            return {"db_open": len(db_open), "loaded": len(db_open), "discrepancies": []}

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
                }
                loaded += 1
                logger.info(
                    f"[LiveTrader] Reconciled {pair} {row['direction']}  "
                    f"db_qty={db_qty:.6f}  exchange_qty={exchange_qty:.6f}"
                )
            else:
                logger.warning(
                    f"[LiveTrader] Discrepancy for {tid}: DB expects {db_qty:.6f} {base} "
                    f"but exchange shows {exchange_qty:.6f} -- may be closed externally"
                )
                discrepancies.append(
                    {"trade_id": tid, "pair": pair, "db_qty": db_qty,
                     "exchange_qty": exchange_qty}
                )

        logger.info(
            f"[LiveTrader] Reconcile complete: {loaded}/{len(db_open)} loaded, "
            f"{len(discrepancies)} discrepancies"
        )
        return {"db_open": len(db_open), "loaded": loaded, "discrepancies": discrepancies}

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _place_market_order(self, pair: str, side: str, qty: float) -> tuple:
        """
        Place a market order on Coinbase Advanced Trade via CCXT.
        Returns (order_id, fill_price, fill_qty).

        Market orders on Coinbase fill synchronously; we use fetch_order
        to confirm the actual average fill price.
        """
        logger.info(f"[LiveTrader] Placing {side.upper()} market order: {pair} qty={qty:.6f}")
        try:
            order = self.exchange.create_order(
                symbol=pair,
                type="market",
                side=side,
                amount=qty,
            )
        except Exception as e:
            logger.error(f"[LiveTrader] Order placement failed: {e}")
            raise

        order_id   = order["id"]
        fill_price = None
        fill_qty   = qty

        # Retry briefly: the REST response may not include 'average' immediately
        for attempt in range(5):
            try:
                fetched   = self.exchange.fetch_order(order_id, pair)
                avg_px    = fetched.get("average")
                filled_qty= fetched.get("filled", qty)
                if avg_px and avg_px > 0:
                    fill_price = avg_px
                    fill_qty   = filled_qty
                    break
            except Exception as e:
                logger.warning(f"[LiveTrader] fetch_order attempt {attempt+1} failed: {e}")
            time.sleep(1)

        if fill_price is None:
            fill_price = (order.get("average") or order.get("price")
                          or self._fetch_current_price(pair))
            logger.warning(f"[LiveTrader] Using fallback fill price: {fill_price}")

        return order_id, fill_price, fill_qty

    def _fetch_current_price(self, pair: str) -> float:
        """Return last traded price for a pair."""
        try:
            return self.exchange.fetch_ticker(pair)["last"]
        except Exception as e:
            logger.error(f"[LiveTrader] Could not fetch price for {pair}: {e}")
            return 0.0

    def _check_sl_tp(self):
        """
        One polling pass: fetch prices for all open positions and close any
        that have breached their stop-loss or take-profit level.
        """
        if not self._positions:
            return

        pairs  = list({pos["pair"] for pos in self._positions.values()})
        prices = {p: self._fetch_current_price(p) for p in pairs}

        for trade_id, pos in list(self._positions.items()):
            price = prices.get(pos["pair"], 0)
            if price <= 0:
                continue

            direction = pos["direction"]
            sl = pos["stop_loss"]
            tp = pos["take_profit"]

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
            else:  # SELL / short
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
        "--dry-run", action="store_true",
        help="Simulate all order placement -- no real orders placed"
    )
    parser.add_argument(
        "--monitor-only", action="store_true",
        help="Start the SL/TP monitor loop only (no new entry orders)"
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
        "--pause", action="store_true",
        help="Activate kill-switch (create TRADING_PAUSED file) and exit"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Deactivate kill-switch (remove TRADING_PAUSED file) and exit"
    )
    args = parser.parse_args()

    # Kill-switch management (no trader instance needed)
    if args.pause:
        pause_trading("paused via CLI")
        print(f"Kill-switch activated: {KILL_SWITCH_FILE}")
        sys.exit(0)

    if args.resume:
        resume_trading()
        print("Kill-switch cleared. Trading can resume.")
        sys.exit(0)

    trader = LiveTrader(dry_run=args.dry_run)
    trader.startup()

    if args.emergency_stop:
        results = trader.emergency_stop()
        print(f"Emergency stop complete. Closed {len(results)} positions.")
        sys.exit(0)

    if args.monitor_only or args.dry_run:
        trader.monitor_positions(interval_seconds=args.interval)
