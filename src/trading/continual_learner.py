"""
Continual Learning Pipeline
Analyses completed trades and generates new fine-tuning examples so the
model gets better over time based on real trading outcomes.

Loop:
  1. Collect trade outcomes via trade_logger
  2. Analyse patterns (win/loss by market condition, confidence, pair, etc.)
  3. Generate JSONL training examples from outcomes
  4. Append to dataset and optionally trigger a new fine-tuning run
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"
TRADE_EXAMPLES_PATH = DATASETS_DIR / "trade_feedback.jsonl"
RETRAIN_THRESHOLD = 50  # new examples before triggering retraining
TRAIN_SCRIPT = Path(__file__).parent.parent / "fine_tuning" / "train_qwen_crypto_oracle.py"


# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

def analyse_patterns(trades: list) -> dict:
    """
    Summarise what market conditions correlate with wins/losses.
    Returns an insights dict (also useful for prompt injection).
    """
    if not trades:
        return {}

    wins = [t for t in trades if t.get("outcome") in ("win", "took_profit")]
    losses = [t for t in trades if t.get("outcome") in ("loss", "stopped_out")]

    insights = {
        "total": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_pnl_pct": sum(t.get("pnl_pct", 0) for t in trades) / len(trades),
        "by_pair": {},
        "by_direction": {"BUY": {}, "SELL": {}},
        "confidence_buckets": {},
    }

    # Per-pair stats
    pairs = set(t["pair"] for t in trades)
    for pair in pairs:
        pair_trades = [t for t in trades if t["pair"] == pair]
        pair_wins = [t for t in pair_trades if t.get("outcome") in ("win", "took_profit")]
        insights["by_pair"][pair] = {
            "total": len(pair_trades),
            "win_rate": len(pair_wins) / len(pair_trades),
            "avg_pnl_pct": sum(t.get("pnl_pct", 0) for t in pair_trades) / len(pair_trades),
        }

    # Per-direction stats
    for direction in ("BUY", "SELL"):
        dir_trades = [t for t in trades if t["direction"] == direction]
        if dir_trades:
            dir_wins = [t for t in dir_trades if t.get("outcome") in ("win", "took_profit")]
            insights["by_direction"][direction] = {
                "total": len(dir_trades),
                "win_rate": len(dir_wins) / len(dir_trades),
                "avg_pnl_pct": sum(t.get("pnl_pct", 0) for t in dir_trades) / len(dir_trades),
            }

    # Confidence buckets: <60, 60-70, 70-80, 80-90, 90+
    buckets = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
    for lo, hi in buckets:
        bucket_trades = [
            t for t in trades
            if t.get("confidence") is not None and lo <= (t["confidence"] or 0) < hi
        ]
        if bucket_trades:
            bucket_wins = [t for t in bucket_trades if t.get("outcome") in ("win", "took_profit")]
            insights["confidence_buckets"][f"{lo}-{hi}"] = {
                "total": len(bucket_trades),
                "win_rate": len(bucket_wins) / len(bucket_trades),
            }

    return insights


# ============================================================================
# TRAINING EXAMPLE GENERATION
# ============================================================================

def _build_corrective_output(trade: dict) -> str:
    """
    For a losing trade, generate a corrective model output that teaches
    the model to say HOLD (or give a tighter recommendation) in similar conditions.
    """
    pair = trade["pair"]
    entry = trade["entry_price"]
    stop = trade["stop_loss"]
    tp = trade["take_profit"]
    pnl = trade.get("pnl_pct", 0)

    return (
        f"**Decision: HOLD**\n"
        f"Confidence: 45%\n\n"
        f"**Reasoning:** Based on retrospective analysis of similar setups on {pair}, "
        f"this entry at ${entry:,.2f} with stop at ${stop:,.2f} and target ${tp:,.2f} "
        f"resulted in a {pnl:.1f}% loss. The risk/reward appeared acceptable but "
        f"market conditions did not support follow-through. Recommend waiting for "
        f"stronger confirmation before entering. No trade at this time.\n\n"
        f"**Risk Management:** N/A — HOLD"
    )


def _build_reinforcing_output(trade: dict) -> str:
    """
    For a winning trade, reinforce the model's reasoning by reconstructing
    what the winning recommendation should have looked like.
    """
    pair = trade["pair"]
    direction = trade["direction"]
    entry = trade["entry_price"]
    stop = trade["stop_loss"]
    tp = trade["take_profit"]
    pnl = trade.get("pnl_pct", 0)
    conf = trade.get("confidence", 80)
    rr = trade.get("reward_risk_ratio", 2.0)

    return (
        f"**Decision: {direction}**\n"
        f"Confidence: {conf}%\n\n"
        f"Entry: ${entry:,.2f}\n"
        f"Stop Loss: ${stop:,.2f}\n"
        f"Take Profit: ${tp:,.2f}\n"
        f"Risk/Reward: {rr:.1f}:1\n\n"
        f"**Reasoning:** Technical and macro conditions on {pair} aligned for a "
        f"{direction.lower()} opportunity. This trade achieved a {pnl:.1f}% gain, "
        f"confirming the setup was valid. Continue to look for similar confluence "
        f"of signals before entering.\n\n"
        f"**Risk Management:** Stop at ${stop:,.2f} limits downside to defined risk. "
        f"Position sized at 1% equity risk per trade."
    )


def generate_training_examples(trades: list) -> list:
    """
    Convert closed trade records to JSONL training examples.
    Winners → reinforcing examples.
    Losers → corrective examples (teach HOLD).
    Returns list of {instruction, output} dicts.
    """
    examples = []

    for trade in trades:
        market_data_raw = trade.get("market_data_json", "{}")
        try:
            market_data = json.loads(market_data_raw) if isinstance(market_data_raw, str) else market_data_raw
        except (json.JSONDecodeError, TypeError):
            market_data = {}

        # Reconstruct the instruction from stored market data
        # Re-use the same format as crypto_oracle._format_prompt if possible
        instruction = _reconstruct_instruction(trade, market_data)

        outcome = trade.get("outcome", "")
        if outcome in ("win", "took_profit"):
            output = _build_reinforcing_output(trade)
        elif outcome in ("loss", "stopped_out"):
            output = _build_corrective_output(trade)
        else:
            continue  # skip ambiguous outcomes

        examples.append({"instruction": instruction, "output": output})

    return examples


def _reconstruct_instruction(trade: dict, market_data: dict) -> str:
    """Reconstruct a training instruction from a logged trade."""
    pair = trade["pair"]
    entry = trade["entry_price"]
    price = market_data.get("price", entry)
    change_1h = market_data.get("change_1h", 0)
    change_24h = market_data.get("change_24h", 0)
    tech = market_data.get("technical", {})

    instruction = (
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
    )

    macro = market_data.get("macro", {})
    if macro:
        instruction += (
            f"\nMacro Environment:\n"
            f"- DXY: ${macro.get('dxy_current', 0):.2f} — {macro.get('dxy_signal', 'unknown')}\n"
            f"- VIX: {macro.get('vix_current', 0):.2f} — {macro.get('vix_signal', 'unknown')}\n"
            f"- BTC Dominance: {macro.get('btc_dominance', 0):.1f}%\n"
        )

    onchain = market_data.get("onchain", {})
    if onchain:
        instruction += (
            f"\nOn-Chain:\n"
            f"- Fear & Greed: {onchain.get('fear_greed_value', 0)}/100 "
            f"({onchain.get('fear_greed_classification', 'unknown')})\n"
        )

    instruction += (
        "\nWhat is your trading recommendation? Provide:\n"
        "1. Decision (BUY/SELL/HOLD)\n"
        "2. Confidence level\n"
        "3. Entry/exit prices\n"
        "4. Risk management plan\n"
        "5. Detailed reasoning"
    )
    return instruction


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

def append_to_dataset(examples: list, path: Path = TRADE_EXAMPLES_PATH) -> int:
    """Append new examples to the trade feedback JSONL file. Returns count added."""
    if not examples:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Appended {len(examples)} examples to {path}")
    return len(examples)


def count_new_examples(path: Path = TRADE_EXAMPLES_PATH) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ============================================================================
# RETRAINING TRIGGER
# ============================================================================

def maybe_trigger_retraining(
    threshold: int = RETRAIN_THRESHOLD,
    feedback_path: Path = TRADE_EXAMPLES_PATH,
    dry_run: bool = False,
) -> bool:
    """
    Kick off incremental fine-tuning if enough new feedback examples exist.

    The train script is re-run with `--train`, which will resume from the
    last stage adapter and incorporate the new feedback data.

    Returns True if retraining was triggered.
    """
    n = count_new_examples(feedback_path)
    logger.info(f"Feedback examples: {n} / {threshold} needed to retrain")

    if n < threshold:
        return False

    if dry_run:
        logger.info("[DRY RUN] Would trigger retraining now")
        return True

    logger.info(f"Triggering incremental fine-tuning with {n} feedback examples...")

    # Merge feedback into the training dataset before running
    _merge_feedback_into_train(feedback_path)

    cmd = [sys.executable, str(TRAIN_SCRIPT), "--train"]
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.Popen(
            cmd,
            cwd=str(TRAIN_SCRIPT.parent.parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(f"Retraining started (PID {result.pid})")
        # Archive the feedback file so we don't re-use these examples next run
        archived = feedback_path.with_suffix(f".{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl")
        feedback_path.rename(archived)
        logger.info(f"Feedback archived to {archived}")
        return True
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        return False


def _merge_feedback_into_train(feedback_path: Path) -> None:
    """Append feedback examples to the main training dataset."""
    train_path = DATASETS_DIR / "enhanced_sft_train.jsonl"
    if not feedback_path.exists():
        return
    with open(feedback_path, encoding="utf-8") as src, \
         open(train_path, "a", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line)
    logger.info(f"Merged {count_new_examples(feedback_path)} feedback examples into {train_path}")


# ============================================================================
# MAIN ENTRY POINT (run analysis manually)
# ============================================================================

def run_learning_cycle(dry_run: bool = False) -> dict:
    """
    Full cycle: load closed trades → analyse → generate examples → maybe retrain.
    Returns a summary dict.
    """
    from trading.trade_logger import get_closed_trades, get_performance_summary

    trades = get_closed_trades()
    if not trades:
        logger.info("No closed trades yet — nothing to learn from")
        return {"status": "no_data"}

    insights = analyse_patterns(trades)
    examples = generate_training_examples(trades)
    added = append_to_dataset(examples)
    triggered = maybe_trigger_retraining(dry_run=dry_run)

    summary = {
        "status": "ok",
        "trades_analysed": len(trades),
        "examples_generated": len(examples),
        "examples_appended": added,
        "retraining_triggered": triggered,
        "performance": get_performance_summary(),
        "insights": insights,
    }

    logger.info(
        f"Learning cycle complete: {len(trades)} trades → "
        f"{len(examples)} examples → retrain={triggered}"
    )
    return summary


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    result = run_learning_cycle(dry_run=dry)
    print(json.dumps(result, indent=2))
