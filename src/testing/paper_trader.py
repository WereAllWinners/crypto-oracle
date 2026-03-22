"""
Paper Trader
Runs the full live pipeline (real market data → fine-tuned LLM → rule layer)
but executes against a simulated portfolio instead of a real exchange.

Tracks:
  - Paper portfolio equity, cash, positions
  - Live SL/TP monitoring via price polling
  - Daily/weekly performance reports
  - All decisions fed into trade_logger for the continual learning loop

Usage:
  # One-shot: evaluate current market and update paper positions
  python src/testing/paper_trader.py --pairs BTC/USD ETH/USD SOL/USD

  # Daemon: run every N minutes continuously
  python src/testing/paper_trader.py --pairs BTC/USD ETH/USD --loop --interval 60

  # View paper portfolio status
  python src/testing/paper_trader.py --status
"""

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.risk_manager import RiskManager, Portfolio
from trading.trade_logger import (
    log_approved_trade, log_rejection, close_trade,
    get_open_trades, get_performance_summary, get_daily_pnl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PAPER_STATE_PATH = Path(__file__).parent.parent.parent / "data" / "paper_portfolio.json"
PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

PAPER_TRADE_PREFIX = "PAPER_"  # all paper trade IDs start with this


# ============================================================================
# PAPER PORTFOLIO STATE
# ============================================================================

def _default_state(initial_equity: float) -> dict:
    return {
        "initial_equity":        initial_equity,
        "cash":                  initial_equity,
        "high_water_mark":       initial_equity,
        "open_positions":        [],   # {trade_id, pair, direction, size_usd, entry_price, stop_loss, take_profit, opened_at}
        "total_trades":          0,
        "created_at":            datetime.utcnow().isoformat(),
        "last_updated":          datetime.utcnow().isoformat(),
    }


def load_state(initial_equity: float = 100_000) -> dict:
    if PAPER_STATE_PATH.exists():
        return json.loads(PAPER_STATE_PATH.read_text())
    state = _default_state(initial_equity)
    save_state(state)
    return state


def save_state(state: dict):
    state["last_updated"] = datetime.utcnow().isoformat()
    PAPER_STATE_PATH.write_text(json.dumps(state, indent=2))


def portfolio_equity(state: dict, current_prices: dict) -> float:
    """Mark-to-market equity: cash + unrealised value of open positions."""
    equity = state["cash"]
    for pos in state["open_positions"]:
        price = current_prices.get(pos["pair"], pos["entry_price"])
        if pos["direction"] == "BUY":
            pnl = (price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
        else:
            pnl = (pos["entry_price"] - price) / pos["entry_price"] * pos["size_usd"]
        equity += pos["size_usd"] + pnl
    return equity


# ============================================================================
# PRICE FETCHING
# ============================================================================

def fetch_prices(pairs: list, exchange_name: str = "coinbase") -> dict:
    """Fetch current prices for a list of pairs."""
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    prices = {}
    for pair in pairs:
        try:
            ticker = exchange.fetch_ticker(pair)
            prices[pair] = ticker["last"]
        except Exception as e:
            logger.warning(f"Could not fetch price for {pair}: {e}")
    return prices


# ============================================================================
# SL/TP MONITOR
# ============================================================================

def check_exits(state: dict, current_prices: dict) -> list:
    """
    Check all open paper positions for SL/TP hits.
    Returns list of trade_ids that were closed.
    """
    closed = []
    remaining = []

    for pos in state["open_positions"]:
        pair   = pos["pair"]
        price  = current_prices.get(pair)
        if price is None:
            remaining.append(pos)
            continue

        direction = pos["direction"]
        sl = pos["stop_loss"]
        tp = pos["take_profit"]
        outcome = None
        exit_price = None

        if direction == "BUY":
            if price <= sl:
                outcome, exit_price = "stopped_out", sl
            elif price >= tp:
                outcome, exit_price = "took_profit", tp
        else:  # SELL
            if price >= sl:
                outcome, exit_price = "stopped_out", sl
            elif price <= tp:
                outcome, exit_price = "took_profit", tp

        if outcome:
            # Realise PnL back to cash
            if direction == "BUY":
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
            else:
                pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

            pnl_usd = pos["size_usd"] * pnl_pct
            state["cash"] += pos["size_usd"] + pnl_usd

            close_trade(
                trade_id=pos["trade_id"],
                exit_price=exit_price,
                outcome=outcome,
            )

            logger.info(
                f"[Paper] {outcome.upper()}  {pair} {direction}  "
                f"entry=${pos['entry_price']:,.2f}  exit=${exit_price:,.2f}  "
                f"PnL=${pnl_usd:+,.2f} ({pnl_pct*100:+.2f}%)"
            )
            closed.append(pos["trade_id"])
        else:
            remaining.append(pos)

    state["open_positions"] = remaining
    return closed


# ============================================================================
# SIGNAL → PAPER TRADE
# ============================================================================

def run_paper_cycle(
    pairs: list,
    state: dict,
    risk_manager: RiskManager,
    oracle=None,
    market_analyzer=None,
) -> dict:
    """
    One full cycle:
      1. Fetch live prices
      2. Monitor existing positions for SL/TP
      3. For each pair, get LLM signal + apply rule layer
      4. Open new paper positions for approved trades
    """
    cycle_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "exits": [],
        "new_trades": [],
        "rejected": [],
    }

    # 1. Prices for all pairs + open positions
    all_pairs = list(set(pairs + [p["pair"] for p in state["open_positions"]]))
    prices = fetch_prices(all_pairs)

    # 2. Check exits
    closed = check_exits(state, prices)
    cycle_log["exits"] = closed

    # 3. Mark-to-market equity
    equity = portfolio_equity(state, prices)
    state["high_water_mark"] = max(state["high_water_mark"], equity)

    daily_pnl = get_daily_pnl()

    # 4. Signal + rule layer for each pair
    for pair in pairs:
        # Skip if already in this pair
        if any(p["pair"] == pair for p in state["open_positions"]):
            logger.info(f"[Paper] {pair}: already in position, skipping")
            continue

        current_price = prices.get(pair)
        if not current_price:
            logger.warning(f"[Paper] {pair}: no price available, skipping")
            continue

        # Get market data + LLM signal
        try:
            if market_analyzer and oracle:
                market_data = market_analyzer.get_current_market_data(pair)
                prediction  = oracle.predict(market_data, temperature=0.5)
                rec         = prediction["recommendation"]
                full_response = prediction.get("full_response", "")
            else:
                # Fallback: no model loaded — use current price only
                logger.warning(f"[Paper] {pair}: no model/analyzer, using placeholder signal")
                rec = {"decision": "HOLD", "confidence": 0,
                       "entry_price": current_price, "stop_loss": None, "take_profit": None}
                market_data   = {"pair": pair, "price": current_price}
                full_response = ""
        except Exception as e:
            logger.error(f"[Paper] {pair}: error getting signal: {e}")
            continue

        # Rule layer
        portfolio = Portfolio(
            total_equity=equity,
            available_cash=state["cash"],
            open_positions=[
                {"pair": p["pair"], "direction": p["direction"],
                 "size_usd": p["size_usd"], "entry_price": p["entry_price"]}
                for p in state["open_positions"]
            ],
            daily_pnl_usd=daily_pnl,
            high_water_mark_equity=state["high_water_mark"],
        )

        decision = risk_manager.evaluate(rec, market_data, portfolio)

        if not decision.approved:
            logger.info(
                f"[Paper] {pair}: {rec.get('decision', 'HOLD')} rejected — "
                f"{', '.join(decision.rejection_reasons)}"
            )
            log_rejection(
                pair=pair,
                direction=rec.get("decision", "UNKNOWN"),
                confidence=rec.get("confidence"),
                reasons=decision.rejection_reasons,
                market_data=market_data,
            )
            cycle_log["rejected"].append({
                "pair": pair, "reasons": decision.rejection_reasons
            })
            continue

        # Open paper position
        trade_id = PAPER_TRADE_PREFIX + str(uuid.uuid4())
        decision_dict = decision.to_dict()

        log_approved_trade(
            trade_id=trade_id,
            decision=decision_dict,
            recommendation=rec,
            market_data=market_data,
            model_response=full_response,
        )

        # Deduct from cash
        state["cash"] = max(0, state["cash"] - decision.position_size_usd)
        state["total_trades"] += 1

        position = {
            "trade_id":    trade_id,
            "pair":        pair,
            "direction":   decision.direction,
            "size_usd":    decision.position_size_usd,
            "entry_price": decision.entry_price,
            "stop_loss":   decision.stop_loss,
            "take_profit": decision.take_profit,
            "opened_at":   datetime.utcnow().isoformat(),
        }
        state["open_positions"].append(position)

        logger.info(
            f"[Paper] OPENED {decision.direction} {pair}  "
            f"size=${decision.position_size_usd:,.0f}  "
            f"entry=${decision.entry_price:,.2f}  "
            f"SL=${decision.stop_loss:,.2f}  TP=${decision.take_profit:,.2f}  "
            f"conf={rec.get('confidence')}%"
        )
        cycle_log["new_trades"].append({
            "trade_id": trade_id,
            "pair": pair,
            "direction": decision.direction,
            "size_usd": decision.position_size_usd,
        })

    save_state(state)
    return cycle_log


# ============================================================================
# STATUS REPORT
# ============================================================================

def print_status(state: dict):
    prices = fetch_prices([p["pair"] for p in state["open_positions"]])
    equity = portfolio_equity(state, prices)
    initial = state["initial_equity"]
    total_return = (equity - initial) / initial * 100

    perf = get_performance_summary()

    print(f"\n{'='*60}")
    print(f"PAPER PORTFOLIO STATUS  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")
    print(f"  Initial equity:  ${initial:>12,.2f}")
    print(f"  Current equity:  ${equity:>12,.2f}  ({total_return:+.2f}%)")
    print(f"  Cash:            ${state['cash']:>12,.2f}")
    print(f"  High-water mark: ${state['high_water_mark']:>12,.2f}")
    print(f"  Daily PnL:       ${get_daily_pnl():>12,.2f}")
    print(f"\n  Open positions: {len(state['open_positions'])}")
    for pos in state["open_positions"]:
        price = prices.get(pos["pair"], pos["entry_price"])
        if pos["direction"] == "BUY":
            unreal = (price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
        else:
            unreal = (pos["entry_price"] - price) / pos["entry_price"] * pos["size_usd"]
        print(f"    {pos['direction']:4} {pos['pair']:10}  "
              f"entry=${pos['entry_price']:,.2f}  now=${price:,.2f}  "
              f"unreal={unreal:+,.2f}")

    print(f"\n  Closed trades:")
    print(f"    Total:       {perf['total_trades']}")
    print(f"    Win rate:    {perf['win_rate']*100:.1f}%")
    print(f"    Avg PnL:     {perf['avg_pnl_pct']:.2f}%")
    print(f"    Total PnL:   ${perf['total_pnl_usd']:,.2f}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Oracle Paper Trader")
    parser.add_argument("--pairs",    nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD"])
    parser.add_argument("--equity",   type=float, default=100_000, help="Starting paper equity")
    parser.add_argument("--risk",     type=float, default=0.01)
    parser.add_argument("--model",    default="models/crypto-oracle-qwen-32b/final_model")
    parser.add_argument("--loop",     action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between cycles (loop mode)")
    parser.add_argument("--status",   action="store_true", help="Print portfolio status and exit")
    parser.add_argument("--reset",    action="store_true", help="Reset paper portfolio to initial state")
    args = parser.parse_args()

    if args.reset:
        if PAPER_STATE_PATH.exists():
            PAPER_STATE_PATH.unlink()
        logger.info(f"Paper portfolio reset (initial equity: ${args.equity:,.0f})")
        sys.exit(0)

    state = load_state(args.equity)

    if args.status:
        print_status(state)
        sys.exit(0)

    # Load model + market analyzer
    logger.info(f"Loading fine-tuned model: {args.model}")
    try:
        from inference.crypto_oracle import CryptoOracle
        from inference.market_analyzer import MarketAnalyzer
        oracle          = CryptoOracle(model_path=args.model)
        market_analyzer = MarketAnalyzer()
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        sys.exit(1)

    rm = RiskManager(risk_pct_per_trade=args.risk)

    if args.loop:
        logger.info(f"Starting paper trading loop — cycle every {args.interval} min")
        logger.info(f"Pairs: {args.pairs}")
        while True:
            try:
                run_paper_cycle(args.pairs, state, rm, oracle, market_analyzer)
                print_status(state)
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            logger.info(f"Sleeping {args.interval} min…")
            time.sleep(args.interval * 60)
    else:
        run_paper_cycle(args.pairs, state, rm, oracle, market_analyzer)
        print_status(state)
