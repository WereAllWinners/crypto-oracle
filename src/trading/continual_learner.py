"""
Continual Learning Pipeline

Improvements over v1:
  1. Tiered labels: strong_winner / winner / weak_winner / weak_loser / loser / strong_loser
  2. Losers excluded from SFT (reserved for future DPO preference pairs)
  3. Prompt hash deduplication — identical market snapshots never train twice
  4. HOLD validation — checks forward price to label HOLDs as correct or missed
  5. Delegates versioning + promotion to model_promoter (eval gate before swap)

Run manually:
  python src/trading/continual_learner.py            # full cycle
  python src/trading/continual_learner.py --dry-run  # analyse only, no writes
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATASETS_DIR      = Path(__file__).parent.parent.parent / "datasets"
FEEDBACK_PATH     = DATASETS_DIR / "trade_feedback.jsonl"
RETRAIN_THRESHOLD = 50   # new SFT examples before triggering a retrain
TRAIN_SCRIPT      = Path(__file__).parent.parent / "fine_tuning" / "train_qwen_crypto_oracle.py"

# ---------------------------------------------------------------------------
# Priority 1 — Tiered labels
# Thresholds calibrated for 6h candles, 1.5×ATR stop, 3×ATR TP.
# pnl_pct here is a fraction (e.g. 0.04 = 4%), NOT the percentage stored in DB.
# ---------------------------------------------------------------------------
_STRONG_WINNER = 0.040    # ≥4.0% — clean TP hit
_WINNER        = 0.015    # ≥1.5% — solid win
_WEAK_WINNER   = 0.003    # ≥0.3% — marginal win
_WEAK_LOSER    = -0.015   # >-1.5% — small loss
_LOSER         = -0.035   # >-3.5% — clear loss
# strong_loser  anything ≤-3.5%

SFT_EXCLUDE_LABELS = {"weak_loser", "loser", "strong_loser"}

# ---------------------------------------------------------------------------
# Priority 5 — HOLD validation parameters
# ---------------------------------------------------------------------------
HOLD_FORWARD_BARS   = 4      # look 4×6h = 24h ahead
HOLD_CORRECT_THRESH = 0.010  # |move| < 1%  → correct_hold
HOLD_MISSED_THRESH  = 0.020  # |move| ≥ 2%  → missed_opportunity


# ============================================================================
# HELPERS
# ============================================================================

def _tiered_label(pnl_fraction: float) -> str:
    """Map a P&L fraction to a tiered label."""
    if pnl_fraction >= _STRONG_WINNER:  return "strong_winner"
    if pnl_fraction >= _WINNER:         return "winner"
    if pnl_fraction >= _WEAK_WINNER:    return "weak_winner"
    if pnl_fraction >= _WEAK_LOSER:     return "weak_loser"
    if pnl_fraction >= _LOSER:          return "loser"
    return "strong_loser"


def _prompt_hash(instruction: str) -> str:
    return hashlib.md5(instruction.encode("utf-8")).hexdigest()


def _fetch_forward_price(pair: str, from_timestamp: str,
                         bars: int = HOLD_FORWARD_BARS,
                         timeframe: str = "6h") -> Optional[float]:
    """Fetch the close price N bars after from_timestamp via ccxt."""
    try:
        import ccxt
        ts = datetime.fromisoformat(from_timestamp.replace("Z", "+00:00"))
        since_ms = int(ts.timestamp() * 1000)
        exchange = ccxt.coinbase({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=bars + 2)
        if len(ohlcv) > bars:
            return float(ohlcv[bars][4])  # close price at bar N
    except Exception as exc:
        logger.warning(f"Could not fetch forward price for {pair} @ {from_timestamp}: {exc}")
    return None


# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

def analyse_patterns(trades: list) -> dict:
    """Summarise win/loss patterns by pair, direction, confidence, and label tier."""
    if not trades:
        return {}

    wins   = [t for t in trades if t.get("outcome") in ("win", "took_profit")]
    losses = [t for t in trades if t.get("outcome") in ("loss", "stopped_out")]

    insights: dict = {
        "total":              len(trades),
        "win_rate":           len(wins) / len(trades),
        "avg_pnl_pct":        sum(t.get("pnl_pct", 0) for t in trades) / len(trades),
        "by_pair":            {},
        "by_direction":       {"BUY": {}, "SELL": {}},
        "confidence_buckets": {},
        "label_distribution": {},
    }

    for pair in set(t["pair"] for t in trades):
        pt = [t for t in trades if t["pair"] == pair]
        pw = [t for t in pt if t.get("outcome") in ("win", "took_profit")]
        insights["by_pair"][pair] = {
            "total":       len(pt),
            "win_rate":    len(pw) / len(pt),
            "avg_pnl_pct": sum(t.get("pnl_pct", 0) for t in pt) / len(pt),
        }

    for direction in ("BUY", "SELL"):
        dt = [t for t in trades if t["direction"] == direction]
        if dt:
            dw = [t for t in dt if t.get("outcome") in ("win", "took_profit")]
            insights["by_direction"][direction] = {
                "total":       len(dt),
                "win_rate":    len(dw) / len(dt),
                "avg_pnl_pct": sum(t.get("pnl_pct", 0) for t in dt) / len(dt),
            }

    for lo, hi in [(0, 60), (60, 70), (70, 80), (80, 90), (90, 101)]:
        bt = [t for t in trades if lo <= (t.get("confidence") or 0) < hi]
        if bt:
            bw = [t for t in bt if t.get("outcome") in ("win", "took_profit")]
            insights["confidence_buckets"][f"{lo}-{hi}"] = {
                "total": len(bt), "win_rate": len(bw) / len(bt)
            }

    for trade in trades:
        pnl = (trade.get("pnl_pct") or 0) / 100.0
        label = _tiered_label(pnl)
        insights["label_distribution"][label] = insights["label_distribution"].get(label, 0) + 1

    return insights


# ============================================================================
# TRAINING EXAMPLE CONSTRUCTION
# ============================================================================

def _build_output_for_label(trade: dict, label: str) -> str:
    """Build an ideal model output string tuned to the outcome label tier."""
    pair  = trade["pair"]
    direction = trade["direction"]
    entry = trade["entry_price"]
    stop  = trade["stop_loss"]
    tp    = trade["take_profit"]
    pnl   = trade.get("pnl_pct") or 0
    conf  = trade.get("confidence") or 80
    rr    = trade.get("reward_risk_ratio") or 2.0

    if label == "strong_winner":
        conf_text = f"{min(conf + 5, 95)}%"
        quality   = "exceptional confluence"
    elif label == "winner":
        conf_text = f"{conf}%"
        quality   = "strong confluence"
    else:  # weak_winner
        conf_text = f"{max(conf - 5, 65)}%"
        quality   = "moderate confluence"

    return (
        f"**Decision: {direction}**\n"
        f"Confidence: {conf_text}\n\n"
        f"Entry: ${entry:,.2f}\n"
        f"Stop Loss: ${stop:,.2f}\n"
        f"Take Profit: ${tp:,.2f}\n"
        f"Risk/Reward: {rr:.1f}:1\n\n"
        f"**Reasoning:** Technical and macro conditions on {pair} showed {quality} "
        f"for a {direction.lower()} setup. "
        f"This trade achieved a {pnl:.1f}% outcome ({label.replace('_', ' ')}), "
        f"confirming signal validity. Continue to look for similar setups.\n\n"
        f"**Risk Management:** Stop at ${stop:,.2f} caps downside. "
        f"Position sized at 1% equity risk per trade."
    )


def _build_hold_output(signal: dict, hold_label: str) -> str:
    """Build training output for a validated HOLD signal."""
    pair  = signal["pair"]
    price = signal["price"]
    fwd   = signal.get("forward_pnl_pct") or 0

    if hold_label == "correct_hold":
        return (
            f"**Decision: HOLD**\n"
            f"Confidence: 75%\n\n"
            f"**Reasoning:** Current conditions on {pair} at ${price:,.2f} did not present "
            f"a high-conviction entry. Price moved only {fwd*100:.1f}% over the next 24 hours, "
            f"confirming the HOLD was correct. Patience protected capital from a "
            f"low-probability setup.\n\n"
            f"**Risk Management:** N/A — no position opened."
        )
    else:  # missed_opportunity
        direction = "BUY" if fwd > 0 else "SELL"
        return (
            f"**Decision: HOLD**\n"
            f"Confidence: 55%\n\n"
            f"**Reasoning:** A {direction} signal was present on {pair} at ${price:,.2f} "
            f"but conviction was insufficient for entry. Price moved {fwd*100:.1f}% "
            f"over the next 24 hours — in hindsight, the setup was valid. "
            f"Consider tightening entry criteria for similar conditions.\n\n"
            f"**Risk Management:** N/A — no position opened."
        )


def _reconstruct_instruction(trade: dict, market_data: dict) -> str:
    pair       = trade["pair"]
    entry      = trade["entry_price"]
    price      = market_data.get("price", entry)
    change_1h  = market_data.get("change_1h", 0)
    change_24h = market_data.get("change_24h", 0)
    tech       = market_data.get("technical") or {}

    parts = [
        f"Analyse the current market conditions for {pair} and provide a "
        f"comprehensive trading recommendation.\n\n"
        f"Current price: ${price:,.2f}\n"
        f"1h change: {change_1h:+.2f}%\n"
        f"24h change: {change_24h:+.2f}%\n\n"
        f"Technical Analysis:\n"
        f"- Trend: {tech.get('trend', 'unknown')}\n"
        f"- RSI: {tech.get('rsi_state', 'unknown')}\n"
        f"- MACD: {tech.get('macd_signal', 'unknown')}\n"
        f"- Bollinger Bands: {tech.get('bb_state', 'unknown')}\n"
        f"- Volume: {tech.get('volume_state', 'unknown')}\n"
    ]

    macro = market_data.get("macro") or {}
    if macro:
        parts.append(
            f"\nMacro Environment:\n"
            f"- DXY: ${macro.get('dxy_current', 0):.2f} — {macro.get('dxy_signal', 'unknown')}\n"
            f"- VIX: {macro.get('vix_current', 0):.2f} — {macro.get('vix_signal', 'unknown')}\n"
            f"- BTC Dominance: {macro.get('btc_dominance', 0):.1f}%\n"
        )

    onchain = market_data.get("onchain") or {}
    if onchain:
        parts.append(
            f"\nOn-Chain:\n"
            f"- Fear & Greed: {onchain.get('fear_greed_value', 0)}/100 "
            f"({onchain.get('fear_greed_classification', 'unknown')})\n"
        )

    parts.append(
        "\nWhat is your trading recommendation? Provide:\n"
        "1. Decision (BUY/SELL/HOLD)\n"
        "2. Confidence level\n"
        "3. Entry/exit prices\n"
        "4. Risk management plan\n"
        "5. Detailed reasoning"
    )
    return "".join(parts)


def _reconstruct_hold_instruction(signal: dict) -> str:
    market_data_raw = signal.get("market_data_json", "{}")
    try:
        market_data = json.loads(market_data_raw) if isinstance(market_data_raw, str) else market_data_raw
    except (json.JSONDecodeError, TypeError):
        market_data = {}
    fake_trade = {"pair": signal["pair"], "entry_price": signal["price"]}
    return _reconstruct_instruction(fake_trade, market_data)


# ============================================================================
# Priority 1+4 — EXAMPLE GENERATION WITH TIERED LABELS + DEDUPLICATION
# ============================================================================

def generate_training_examples(trades: list, validated_holds: list = None) -> list:
    """
    Convert closed trades + validated HOLDs into SFT training examples.

    - Winners (strong/winner/weak) → reinforcing examples
    - Losers → excluded from SFT, hash still recorded to prevent reprocessing
    - Validated HOLDs → correct_hold or missed_opportunity examples
    - Prompt hash deduplication against previously processed examples
    """
    from trading.trade_logger import get_existing_prompt_hashes, log_training_example_hash

    existing = get_existing_prompt_hashes()
    examples = []
    skipped_dup    = 0
    skipped_loser  = 0

    for trade in trades:
        try:
            market_data = json.loads(trade.get("market_data_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            market_data = {}

        instruction = _reconstruct_instruction(trade, market_data)
        ph = _prompt_hash(instruction)

        if ph in existing:
            skipped_dup += 1
            continue

        outcome = trade.get("outcome", "")
        if outcome not in ("win", "took_profit", "loss", "stopped_out"):
            continue

        # DB stores pnl_pct as a percentage value (e.g. 4.2), convert to fraction
        pnl_fraction = (trade.get("pnl_pct") or 0) / 100.0
        label = _tiered_label(pnl_fraction)

        # Always record the hash so we don't reprocess this trade in future cycles
        existing.add(ph)
        log_training_example_hash(ph, label, pnl_fraction,
                                  trade.get("pair"), trade.get("trade_id"))

        if label in SFT_EXCLUDE_LABELS:
            skipped_loser += 1
            continue  # reserved for future DPO; not added to SFT dataset

        examples.append({
            "instruction": instruction,
            "output":      _build_output_for_label(trade, label),
            "label":       label,
            "pnl_pct":     pnl_fraction,
            "pair":        trade.get("pair"),
            "prompt_hash": ph,
        })

    # Priority 5 — include validated HOLDs
    for signal in (validated_holds or []):
        hold_label = signal.get("hold_label")
        if hold_label not in ("correct_hold", "missed_opportunity"):
            continue

        instruction = _reconstruct_hold_instruction(signal)
        ph = _prompt_hash(instruction)

        if ph in existing:
            skipped_dup += 1
            continue

        existing.add(ph)
        log_training_example_hash(ph, hold_label, signal.get("forward_pnl_pct", 0),
                                  signal.get("pair"), None)

        examples.append({
            "instruction": instruction,
            "output":      _build_hold_output(signal, hold_label),
            "label":       hold_label,
            "pnl_pct":     signal.get("forward_pnl_pct", 0),
            "pair":        signal.get("pair"),
            "prompt_hash": ph,
        })

    logger.info(
        f"Generated {len(examples)} SFT examples "
        f"(skipped {skipped_loser} losers, {skipped_dup} duplicates)"
    )
    return examples


# ============================================================================
# Priority 5 — HOLD VALIDATION
# ============================================================================

def validate_hold_signals(timeframe: str = "6h") -> list:
    """
    For each unvalidated HOLD signal older than HOLD_FORWARD_BARS bars,
    fetch the forward price and label it correct_hold or missed_opportunity.
    Returns the list of newly validated signals.
    """
    from trading.trade_logger import get_unvalidated_hold_signals, update_hold_signal

    cutoff  = datetime.now(timezone.utc) - timedelta(hours=HOLD_FORWARD_BARS * 6)
    pending = get_unvalidated_hold_signals(before=cutoff.isoformat())

    validated = []
    for signal in pending:
        pair      = signal["pair"]
        logged_at = signal["logged_at"]
        entry_px  = signal["price"]

        fwd_price = _fetch_forward_price(pair, logged_at, bars=HOLD_FORWARD_BARS,
                                         timeframe=timeframe)
        if fwd_price is None:
            continue

        fwd_pnl  = (fwd_price - entry_px) / entry_px
        abs_move = abs(fwd_pnl)

        if abs_move < HOLD_CORRECT_THRESH:
            hold_label = "correct_hold"
        elif abs_move >= HOLD_MISSED_THRESH:
            hold_label = "missed_opportunity"
        else:
            hold_label = "correct_hold"  # ambiguous — treat conservatively

        update_hold_signal(
            signal_id=signal["id"],
            forward_pnl_pct=fwd_pnl,
            hold_label=hold_label,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )

        result = {**signal, "forward_pnl_pct": fwd_pnl, "hold_label": hold_label}
        validated.append(result)
        logger.info(f"HOLD validated: {pair}  fwd={fwd_pnl*100:+.2f}%  label={hold_label}")

    return validated


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

def append_to_dataset(examples: list, path: Path = FEEDBACK_PATH) -> int:
    """Write SFT examples (instruction + output only) to the feedback JSONL."""
    if not examples:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"instruction": ex["instruction"], "output": ex["output"]}) + "\n")
    logger.info(f"Appended {len(examples)} examples to {path}")
    return len(examples)


def count_new_examples(path: Path = FEEDBACK_PATH) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ============================================================================
# Priority 2+3 — RETRAINING WITH EVAL GATE
# ============================================================================

def maybe_trigger_retraining(
    threshold: int = RETRAIN_THRESHOLD,
    feedback_path: Path = FEEDBACK_PATH,
    dry_run: bool = False,
) -> bool:
    """
    Trigger incremental fine-tuning when enough new examples exist.
    Delegates versioning and the eval gate to model_promoter.
    """
    from trading.model_promoter import run_training_and_promote

    n = count_new_examples(feedback_path)
    logger.info(f"Feedback examples: {n} / {threshold} needed to retrain")

    if n < threshold:
        return False

    if dry_run:
        logger.info("[DRY RUN] Would trigger retraining now")
        return True

    _merge_feedback_into_train(feedback_path)
    return run_training_and_promote(feedback_path=feedback_path)


def _merge_feedback_into_train(feedback_path: Path) -> None:
    train_path = DATASETS_DIR / "enhanced_sft_train.jsonl"
    if not feedback_path.exists():
        return
    with open(feedback_path, encoding="utf-8") as src, \
         open(train_path, "a", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line)
    logger.info(f"Merged feedback examples into {train_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_learning_cycle(dry_run: bool = False, timeframe: str = "6h") -> dict:
    """
    Full cycle:
      1. Validate pending HOLD signals via forward price
      2. Load all closed trades
      3. Generate tiered-label SFT examples (deduped, losers excluded)
      4. Append winners to feedback dataset
      5. Trigger retraining + eval gate if threshold met
    """
    from trading.trade_logger import get_closed_trades, get_performance_summary

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    logger.info("=== Learning cycle started ===")

    validated_holds = validate_hold_signals(timeframe=timeframe)
    logger.info(f"Validated {len(validated_holds)} HOLD signals")

    trades = get_closed_trades()
    if not trades:
        logger.info("No closed trades yet — nothing to learn from")
        return {"status": "no_data"}

    examples = generate_training_examples(trades, validated_holds=validated_holds)
    added    = append_to_dataset(examples)
    triggered = maybe_trigger_retraining(dry_run=dry_run)

    summary = {
        "status":               "ok",
        "trades_analysed":      len(trades),
        "holds_validated":      len(validated_holds),
        "examples_generated":   len(examples),
        "examples_appended":    added,
        "retraining_triggered": triggered,
        "performance":          get_performance_summary(),
        "insights":             analyse_patterns(trades),
    }

    logger.info(
        f"Learning cycle complete — {len(trades)} trades → "
        f"{len(examples)} examples → retrain={triggered}"
    )
    return summary


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    result = run_learning_cycle(dry_run=dry)
    print(json.dumps(result, indent=2, default=str))
