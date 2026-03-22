"""
Historical Backtester
Simulates the full pipeline (technical indicators -> LLM -> rule layer -> trade)
on historical OHLCV data to validate strategy performance before going live.

Two modes:
  --fast     Uses rule-based signals (no LLM). Good for rapid iteration.
  --model    Actually calls the fine-tuned LLM on sampled bars. Slow but accurate.

Metrics reported:
  Total return, Sharpe ratio, Max drawdown, Win rate, Profit factor,
  Avg win/loss, Trade count, Calmar ratio

Usage:
  python src/testing/backtester.py --pair BTC/USD --fast
  python src/testing/backtester.py --pair BTC/USD --model --samples 150
  python src/testing/backtester.py --pair ETH/USD --fast --start 2023-01-01 --end 2024-01-01
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe on Windows/headless
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.risk_manager import RiskManager, Portfolio

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ohlcv"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "backtest_results"

# Realistic cost model
COINBASE_FEE   = 0.004   # 0.4% taker fee per side (entry + exit)
_MAJOR_PAIRS   = {"BTC/USD", "ETH/USD"}
_SLIPPAGE_MAJOR= 0.0005  # 0.05% for BTC/ETH
_SLIPPAGE_ALT  = 0.0015  # 0.15% for alts


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(pair: str, timeframe: str = "1h") -> pd.DataFrame:
    filename = pair.replace("/", "_") + f"_{timeframe}.csv"
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No OHLCV data for {pair} at {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to the dataframe."""
    d = df.copy()
    d["sma_20"]  = ta.trend.sma_indicator(d["close"], window=20)
    d["sma_50"]  = ta.trend.sma_indicator(d["close"], window=50)
    d["sma_200"] = ta.trend.sma_indicator(d["close"], window=200)
    d["rsi"]     = ta.momentum.rsi(d["close"], window=14)
    macd         = ta.trend.MACD(d["close"])
    d["macd"]    = macd.macd()
    d["macd_sig"]= macd.macd_signal()
    bb           = ta.volatility.BollingerBands(d["close"], window=20)
    d["bb_upper"]= bb.bollinger_hband()
    d["bb_lower"]= bb.bollinger_lband()
    d["vol_sma"] = d["volume"].rolling(20).mean()
    d["atr"]     = ta.volatility.average_true_range(d["high"], d["low"], d["close"], window=14)
    return d.dropna()


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def fast_signal(row: pd.Series) -> dict:
    """
    Rule-based proxy signal — no LLM. Approximates what a well-trained model
    should output given this market snapshot. Used for rapid backtesting.
    """
    price      = row["close"]
    rsi        = row["rsi"]
    macd_cross = row["macd"] - row["macd_sig"]
    above_sma20 = price > row["sma_20"]
    above_sma50 = price > row["sma_50"]
    atr        = row["atr"]

    # BUY: oversold RSI + MACD turning up, regardless of trend
    if rsi < 42 and macd_cross > 0:
        direction  = "BUY"
        confidence = min(90, 65 + int((42 - rsi)))
    # SELL: overbought RSI + MACD turning down, or strong downtrend
    elif rsi > 58 and macd_cross < 0 and not above_sma20:
        direction  = "SELL"
        confidence = min(90, 65 + int((rsi - 58)))
    else:
        direction  = "HOLD"
        confidence = 50

    # Prices based on ATR
    if direction == "BUY":
        stop  = price - 1.5 * atr
        tp    = price + 3.0 * atr
    elif direction == "SELL":
        stop  = price + 1.5 * atr
        tp    = price - 3.0 * atr
    else:
        stop = tp = None

    return {
        "decision":    direction,
        "confidence":  confidence,
        "entry_price": price,
        "stop_loss":   round(stop, 8) if stop else None,
        "take_profit": round(tp,   8) if tp   else None,
    }


def _get_slippage(pair: str) -> float:
    """One-way slippage estimate for the given pair."""
    return _SLIPPAGE_MAJOR if pair in _MAJOR_PAIRS else _SLIPPAGE_ALT


def _effective_prices(pair: str, direction: str,
                      entry: float, exit_px: float) -> tuple:
    """
    Apply taker fee + slippage to entry and exit prices.
    Returns (effective_entry, effective_exit) for PnL calculation.

    BUY:  we pay more on entry, receive less on exit.
    SELL: we receive less on entry, pay more on exit.
    """
    cost = COINBASE_FEE + _get_slippage(pair)   # per-side total cost
    if direction == "BUY":
        return entry * (1 + cost), exit_px * (1 - cost)
    else:
        return entry * (1 - cost), exit_px * (1 + cost)


def llm_signal(oracle, row: pd.Series, pair: str) -> dict:
    """Call the fine-tuned LLM and return its parsed recommendation."""
    price = row["close"]
    rsi   = row["rsi"]

    if rsi > 70:
        rsi_state = f"overbought (RSI {rsi:.1f})"
    elif rsi < 30:
        rsi_state = f"oversold (RSI {rsi:.1f})"
    else:
        rsi_state = f"neutral (RSI {rsi:.1f})"

    if price > row["sma_20"] > row["sma_50"]:
        trend = "strong uptrend"
    elif price > row["sma_20"]:
        trend = "mild uptrend"
    elif price < row["sma_20"] < row["sma_50"]:
        trend = "strong downtrend"
    else:
        trend = "sideways"

    bb_pos = (price - row["bb_lower"]) / max(row["bb_upper"] - row["bb_lower"], 1e-9)
    if bb_pos > 0.8:
        bb_state = "near upper band (potential reversal)"
    elif bb_pos < 0.2:
        bb_state = "near lower band (potential bounce)"
    else:
        bb_state = "mid-range"

    vol_ratio = row["volume"] / max(row["vol_sma"], 1e-9)
    vol_state = "high" if vol_ratio > 1.5 else "normal" if vol_ratio > 0.7 else "low"

    market_data = {
        "pair": pair,
        "price": price,
        "change_1h": 0,
        "change_24h": 0,
        "technical": {
            "trend": trend,
            "rsi_state": rsi_state,
            "rsi": rsi,
            "macd_signal": "bullish" if row["macd"] > row["macd_sig"] else "bearish",
            "bb_state": bb_state,
            "volume_state": vol_state,
            "sma_20": row["sma_20"],
            "sma_50": row["sma_50"],
            "sma_200": row["sma_200"],
            "volume_ratio": vol_ratio,
        },
    }
    pred = oracle.predict(market_data, temperature=0.3)
    return pred["recommendation"]


# ============================================================================
# TRADE SIMULATION
# ============================================================================

def find_exit(df: pd.DataFrame, entry_idx: int, direction: str,
              stop_loss: float, take_profit: float,
              max_bars: int = 240) -> tuple:
    """
    Scan forward bar-by-bar to find when SL or TP is hit.
    Returns (exit_price, bars_held, outcome).
    Uses bar's high/low to detect SL/TP touches within the candle.
    """
    n = len(df)
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, n)):
        high  = df.iloc[i]["high"]
        low   = df.iloc[i]["low"]
        close = df.iloc[i]["close"]

        if direction == "BUY":
            if low <= stop_loss:
                return stop_loss, i - entry_idx, "stopped_out"
            if high >= take_profit:
                return take_profit, i - entry_idx, "took_profit"
        else:  # SELL
            if high >= stop_loss:
                return stop_loss, i - entry_idx, "stopped_out"
            if low <= take_profit:
                return take_profit, i - entry_idx, "took_profit"

    # Max hold period reached — exit at close
    exit_price = df.iloc[min(entry_idx + max_bars, n - 1)]["close"]
    outcome = "win" if (
        (direction == "BUY"  and exit_price > df.iloc[entry_idx]["close"]) or
        (direction == "SELL" and exit_price < df.iloc[entry_idx]["close"])
    ) else "loss"
    return exit_price, min(max_bars, n - 1 - entry_idx), outcome


# ============================================================================
# BACKTESTER
# ============================================================================

def run_backtest(
    pair: str,
    timeframe: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_model: bool = False,
    model_path: str = "models/crypto-oracle-qwen-32b/final_model",
    samples: int = 200,          # bars to evaluate when use_model=True
    initial_equity: float = 100_000,
    risk_pct: float = 0.01,
    eval_every: int = 24,        # fast mode: evaluate every N bars
) -> dict:

    print(f"\n{'='*60}")
    print(f"BACKTEST  {pair}  [{timeframe}]  mode={'LLM' if use_model else 'fast'}")
    print(f"{'='*60}")

    # ---- Load & prepare data ----
    df_raw = load_ohlcv(pair, timeframe)
    if start:
        df_raw = df_raw[df_raw.index >= start]
    if end:
        df_raw = df_raw[df_raw.index <= end]

    df = compute_indicators(df_raw)
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}  ({len(df):,} bars)")

    # Require at least 200 bars of warmup
    df = df.iloc[200:]
    if len(df) < 50:
        raise ValueError("Not enough data after warmup. Try a longer date range.")

    # ---- Select evaluation points ----
    if use_model:
        # Sample evenly across the dataset
        indices = np.linspace(0, len(df) - 1, num=min(samples, len(df)), dtype=int)
        indices = sorted(set(indices))
        from inference.crypto_oracle import CryptoOracle
        oracle = CryptoOracle(model_path=model_path)
        print(f"LLM mode: evaluating {len(indices)} sampled bars (this will take a while…)")
    else:
        # Every N bars
        indices = list(range(0, len(df), eval_every))
        print(f"Fast mode: evaluating every {eval_every} bars -> {len(indices)} evaluation points")

    # ---- Risk manager ----
    rm = RiskManager(risk_pct_per_trade=risk_pct)

    # ---- Simulation state ----
    equity = initial_equity
    hwm    = initial_equity
    trades = []
    equity_curve = [{"timestamp": str(df.index[0]), "equity": equity}]
    next_entry_bar = 0  # bar index after which we can enter the next trade

    for idx in indices:
        if idx >= len(df) - 1:
            break
        if idx < next_entry_bar:
            continue  # still in a position, wait for exit bar

        row = df.iloc[idx]

        # Generate signal
        if use_model:
            rec = llm_signal(oracle, row, pair)
            print(f"  [{df.index[idx].date()}] model -> {rec.get('decision')} "
                  f"conf={rec.get('confidence')}%", end="")
        else:
            rec = fast_signal(row)

        if rec["decision"] == "HOLD":
            continue

        # Rule layer
        portfolio = Portfolio(
            total_equity=equity,
            available_cash=equity,  # simplified: all cash available
            open_positions=[],
            daily_pnl_usd=0,
            high_water_mark_equity=hwm,
        )
        decision = rm.evaluate(rec, {"pair": pair, "price": row["close"]}, portfolio)

        if not decision.approved:
            if use_model:
                print(f" -> rejected: {decision.rejection_reasons[0]}")
            continue

        if use_model:
            print(f" -> APPROVED  size=${decision.position_size_usd:.0f}")

        # Simulate outcome
        exit_price, bars_held, outcome = find_exit(
            df, idx,
            direction=decision.direction,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )

        entry_price = decision.entry_price
        size_usd    = decision.position_size_usd

        # Apply realistic fee + slippage to get net PnL
        eff_entry, eff_exit = _effective_prices(
            pair, decision.direction, entry_price, exit_price
        )
        if decision.direction == "BUY":
            pnl_pct = (eff_exit - eff_entry) / eff_entry
        else:
            pnl_pct = (eff_entry - eff_exit) / eff_entry

        fee_usd = size_usd * COINBASE_FEE * 2   # entry + exit fee (display only)
        pnl_usd = size_usd * pnl_pct            # already fee+slippage adjusted
        equity  = max(0, equity + pnl_usd)
        hwm     = max(hwm, equity)

        trade = {
            "entry_time":   str(df.index[idx]),
            "exit_time":    str(df.index[min(idx + bars_held, len(df)-1)]),
            "pair":         pair,
            "direction":    decision.direction,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "eff_entry":    round(eff_entry, 6),
            "eff_exit":     round(eff_exit, 6),
            "stop_loss":    decision.stop_loss,
            "take_profit":  decision.take_profit,
            "size_usd":     size_usd,
            "fee_usd":      round(fee_usd, 2),
            "pnl_usd":      round(pnl_usd, 2),
            "pnl_pct":      round(pnl_pct * 100, 3),
            "bars_held":    bars_held,
            "outcome":      outcome,
            "equity_after": round(equity, 2),
        }
        trades.append(trade)
        equity_curve.append({"timestamp": trade["exit_time"], "equity": equity})
        next_entry_bar = idx + bars_held + 1  # don't enter again until this position closes

    # ---- Metrics ----
    metrics = _compute_metrics(trades, initial_equity, equity, equity_curve)

    result = {
        "pair":           pair,
        "timeframe":      timeframe,
        "start":          str(df.index[0].date()),
        "end":            str(df.index[-1].date()),
        "mode":           "llm" if use_model else "fast",
        "initial_equity": initial_equity,
        "final_equity":   round(equity, 2),
        "metrics":        metrics,
        "trades":         trades,
        "equity_curve":   equity_curve,
    }

    _print_metrics(result)

    # Save equity curve PNG
    try:
        plot_path = _save_equity_curve(result)
        result["equity_curve_png"] = plot_path
    except Exception as exc:
        logger.warning(f"Could not save equity curve: {exc}")

    return result


# ============================================================================
# METRICS
# ============================================================================

def _compute_metrics(trades: list, initial_equity: float,
                     final_equity: float, equity_curve: list) -> dict:
    if not trades:
        return {"error": "No trades executed"}

    pnls   = [t["pnl_usd"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate      = len(wins) / len(pnls)
    avg_win       = np.mean(wins)   if wins   else 0.0
    avg_loss      = np.mean(losses) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")
    total_fees    = sum(t.get("fee_usd", 0.0) for t in trades)
    avg_hold      = np.mean([t["bars_held"] for t in trades])

    # Net return (pnl_usd already includes fee+slippage via effective prices)
    total_return = (final_equity - initial_equity) / initial_equity

    # Gross return for comparison (without fee adjustment)
    gross_pnl    = sum(
        (t["exit_price"] - t["entry_price"]) / t["entry_price"] * t["size_usd"]
        if t["direction"] == "BUY"
        else (t["entry_price"] - t["exit_price"]) / t["entry_price"] * t["size_usd"]
        for t in trades
    )
    gross_return = gross_pnl / initial_equity

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    peak   = equities[0]
    max_dd = 0.0
    for e in equities:
        peak  = max(peak, e)
        dd    = (peak - e) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe (annualised, 8760 hourly bars per year)
    if len(pnls) > 1:
        returns = np.array(pnls) / initial_equity
        std_r   = np.std(returns)
        sharpe  = (np.mean(returns) / std_r) * np.sqrt(8760) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    calmar = total_return / max_dd if max_dd > 0 else float("inf")

    return {
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         round(win_rate * 100, 1),
        "avg_win_usd":      round(avg_win, 2),
        "avg_loss_usd":     round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 2),
        "gross_return_pct": round(gross_return * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "total_fees_usd":   round(total_fees, 2),
        "avg_hold_bars":    round(avg_hold, 1),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio":     round(sharpe, 2),
        "calmar_ratio":     round(calmar, 2),
    }


def _print_metrics(result: dict):
    m = result["metrics"]
    if "error" in m:
        print(f"\n  No trades executed in this period.")
        return

    print(f"\n  Period:                {result['start']} -> {result['end']}")
    print(f"  Trades:                {m['total_trades']}  (W:{m['wins']} / L:{m['losses']})")
    print(f"  Win rate:              {m['win_rate']}%")
    print(f"  Avg win:               ${m['avg_win_usd']:,.2f}")
    print(f"  Avg loss:              ${m['avg_loss_usd']:,.2f}")
    print(f"  Profit factor:         {m['profit_factor']}")
    print(f"  Gross return:          {m['gross_return_pct']}%  (before fees/slippage)")
    print(f"  Net return:            {m['total_return_pct']}%  (after fees+slippage)")
    print(f"  Total fees paid:       ${m['total_fees_usd']:,.2f}")
    print(f"  Avg hold time:         {m['avg_hold_bars']} bars")
    print(f"  Max drawdown:          {m['max_drawdown_pct']}%")
    print(f"  Sharpe ratio:          {m['sharpe_ratio']}")
    print(f"  Calmar ratio:          {m['calmar_ratio']}")
    print(f"  Equity:                ${result['initial_equity']:,.0f} -> ${result['final_equity']:,.0f}")

    # Stricter go/no-go gates for live readiness
    print(f"\n  GO / NO-GO GATES  (fees+slippage included):")
    gates = [
        ("Win rate >= 55%",         m["win_rate"] >= 55,          f"{m['win_rate']}%"),
        ("Profit factor > 1.5",     m["profit_factor"] > 1.5,     str(m["profit_factor"])),
        ("Max drawdown < 15%",      m["max_drawdown_pct"] < 15,   f"{m['max_drawdown_pct']}%"),
        ("Sharpe ratio > 0.8",      m["sharpe_ratio"] > 0.8,      str(m["sharpe_ratio"])),
        ("Positive net return",     m["total_return_pct"] > 0,    f"{m['total_return_pct']}%"),
        ("Min 20 trades",           m["total_trades"] >= 20,      str(m["total_trades"])),
    ]
    all_pass = True
    for label, passed, value in gates:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{mark}] {label:<32} ({value})")

    verdict = "READY FOR PAPER TRADING" if all_pass else "NOT READY -- review failures above"
    print(f"\n  Verdict: {verdict}")
    print(f"{'='*60}\n")


def _save_equity_curve(result: dict) -> str:
    """
    Save an equity curve PNG to data/backtest_results/.
    Shows: equity line, drawdown shading, BUY/SELL entry markers.
    Returns the saved file path.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_safe = result["pair"].replace("/", "_")
    out_path  = RESULTS_DIR / f"{pair_safe}_{result['timeframe']}_{ts}_equity.png"

    eq_curve = result["equity_curve"]
    trades   = result["trades"]

    eq_values = [e["equity"]    for e in eq_curve]
    eq_times  = [e["timestamp"] for e in eq_curve]

    try:
        eq_dt = [datetime.fromisoformat(t) for t in eq_times]
    except Exception:
        eq_dt = list(range(len(eq_times)))

    # Running peak for drawdown shading
    peaks = []
    running_peak = eq_values[0] if eq_values else 0
    for v in eq_values:
        running_peak = max(running_peak, v)
        peaks.append(running_peak)

    # Entry marker positions
    eq_lookup  = {e["timestamp"]: e["equity"] for e in eq_curve}
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for t in trades:
        eq_val = eq_lookup.get(t["exit_time"], t["equity_after"])
        try:
            dt = datetime.fromisoformat(t["entry_time"])
        except Exception:
            dt = t["entry_time"]
        if t["direction"] == "BUY":
            buy_x.append(dt);  buy_y.append(eq_val)
        else:
            sell_x.append(dt); sell_y.append(eq_val)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(eq_dt, peaks, eq_values, alpha=0.25, color="crimson",
                    label="Drawdown")
    ax.plot(eq_dt, eq_values, color="steelblue", linewidth=1.5, label="Equity")
    if buy_x:
        ax.scatter(buy_x, buy_y, marker="^", color="limegreen", s=55,
                   zorder=5, label="BUY entry")
    if sell_x:
        ax.scatter(sell_x, sell_y, marker="v", color="tomato", s=55,
                   zorder=5, label="SELL entry")

    m = result["metrics"]
    ax.set_title(
        f"{result['pair']} [{result['timeframe']}]  "
        f"Net={m['total_return_pct']}%  Gross={m['gross_return_pct']}%  "
        f"Fees=${m['total_fees_usd']:,.0f}  Sharpe={m['sharpe_ratio']}  "
        f"MaxDD={m['max_drawdown_pct']}%  WR={m['win_rate']}%",
        fontsize=9,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

    print(f"  Equity curve: {out_path}")
    return str(out_path)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Oracle Historical Backtester")
    parser.add_argument("--pair",      default="BTC/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start",     default=None, help="e.g. 2023-01-01")
    parser.add_argument("--end",       default=None, help="e.g. 2024-01-01")
    parser.add_argument("--fast",      action="store_true", help="Rule-based signals (no LLM)")
    parser.add_argument("--model",     action="store_true", help="Use fine-tuned LLM (slow)")
    parser.add_argument("--model-path",default="models/crypto-oracle-qwen-32b/final_model")
    parser.add_argument("--samples",   type=int, default=200, help="LLM mode: bars to sample")
    parser.add_argument("--equity",    type=float, default=100_000)
    parser.add_argument("--risk",      type=float, default=0.01, help="Risk per trade (0.01=1%)")
    parser.add_argument("--eval-every",type=int, default=24, help="Fast mode: eval every N bars")
    parser.add_argument("--out",       default=None, help="Save results JSON to path")
    args = parser.parse_args()

    if not args.fast and not args.model:
        print("Specify --fast (rule-based) or --model (use LLM). Defaulting to --fast.")
        args.fast = True

    result = run_backtest(
        pair=args.pair,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        use_model=args.model,
        model_path=args.model_path,
        samples=args.samples,
        initial_equity=args.equity,
        risk_pct=args.risk,
        eval_every=args.eval_every,
    )

    if args.out:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.out)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Results saved to {out_path}")
    else:
        # Auto-save with timestamp
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_safe = args.pair.replace("/", "_")
        mode = "llm" if args.model else "fast"
        out_path = RESULTS_DIR / f"backtest_{pair_safe}_{mode}_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Results auto-saved to {out_path}")
