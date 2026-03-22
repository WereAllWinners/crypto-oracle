"""
Model Evaluator
Tests fine-tuned model output quality BEFORE backtesting on live data.

Checks:
  1. Parse rate     — can we extract a decision from the response?
  2. Price sanity   — is stop on the right side of entry?
  3. Confidence     — does the model emit a confidence value?
  4. Consistency    — same conditions → same direction most of the time?
  5. Latency        — how long does inference take?

Run: python src/testing/model_evaluator.py --model models/crypto-oracle-qwen-32b/final_model
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# STANDARDISED TEST SCENARIOS
# Each represents a distinct market regime the model must handle correctly.
# ============================================================================

TEST_SCENARIOS = [
    {
        "name": "BTC strong uptrend overbought",
        "expected_direction": "SELL_OR_HOLD",  # overbought = caution
        "market_data": {
            "pair": "BTC/USD", "price": 95000, "change_1h": 1.2, "change_24h": 8.5,
            "technical": {
                "trend": "strong uptrend", "rsi_state": "overbought (RSI 78.3)",
                "macd_signal": "bullish", "bb_state": "near upper band (potential reversal)",
                "volume_state": "high", "sma_20": 88000, "sma_50": 82000, "sma_200": 70000,
                "rsi": 78.3, "volume_ratio": 1.8,
            },
            "macro": {"vix_current": 18.0, "vix_signal": "normal", "dxy_current": 103.0,
                      "dxy_change": -0.2, "dxy_signal": "neutral", "spy_current": 510.0,
                      "spy_change": 0.5, "spy_signal": "neutral", "btc_dominance": 54.0,
                      "btc_dom_phase": "btc_season"},
            "onchain": {"fear_greed_value": 78, "fear_greed_classification": "Greed",
                        "fear_greed_signal": "slightly_bearish"},
        },
    },
    {
        "name": "BTC strong downtrend oversold",
        "expected_direction": "BUY_OR_HOLD",  # oversold = potential bounce
        "market_data": {
            "pair": "BTC/USD", "price": 58000, "change_1h": -1.5, "change_24h": -12.0,
            "technical": {
                "trend": "strong downtrend", "rsi_state": "oversold (RSI 24.1)",
                "macd_signal": "bearish", "bb_state": "near lower band (potential bounce)",
                "volume_state": "high", "sma_20": 68000, "sma_50": 72000, "sma_200": 65000,
                "rsi": 24.1, "volume_ratio": 2.1,
            },
            "macro": {"vix_current": 32.0, "vix_signal": "elevated_fear", "dxy_current": 106.0,
                      "dxy_change": 1.2, "dxy_signal": "bearish_crypto", "spy_current": 480.0,
                      "spy_change": -2.1, "spy_signal": "bearish", "btc_dominance": 51.0,
                      "btc_dom_phase": "btc_season"},
            "onchain": {"fear_greed_value": 22, "fear_greed_classification": "Extreme Fear",
                        "fear_greed_signal": "bullish"},
        },
    },
    {
        "name": "ETH sideways consolidation",
        "expected_direction": "HOLD",
        "market_data": {
            "pair": "ETH/USD", "price": 3200, "change_1h": 0.1, "change_24h": 0.8,
            "technical": {
                "trend": "sideways", "rsi_state": "neutral (RSI 51.2)",
                "macd_signal": "bearish", "bb_state": "mid-range",
                "volume_state": "low", "sma_20": 3180, "sma_50": 3150, "sma_200": 2900,
                "rsi": 51.2, "volume_ratio": 0.6,
            },
            "macro": {"vix_current": 20.0, "vix_signal": "normal", "dxy_current": 104.0,
                      "dxy_change": 0.0, "dxy_signal": "neutral", "spy_current": 505.0,
                      "spy_change": 0.1, "spy_signal": "neutral", "btc_dominance": 52.0,
                      "btc_dom_phase": "btc_season"},
            "onchain": {"fear_greed_value": 50, "fear_greed_classification": "Neutral",
                        "fear_greed_signal": "neutral"},
        },
    },
    {
        "name": "SOL breakout bullish momentum",
        "expected_direction": "BUY_OR_HOLD",
        "market_data": {
            "pair": "SOL/USD", "price": 185, "change_1h": 2.3, "change_24h": 14.0,
            "technical": {
                "trend": "strong uptrend", "rsi_state": "neutral (RSI 62.5)",
                "macd_signal": "bullish", "bb_state": "near upper band (potential reversal)",
                "volume_state": "high", "sma_20": 170, "sma_50": 155, "sma_200": 120,
                "rsi": 62.5, "volume_ratio": 2.5,
            },
            "macro": {"vix_current": 16.0, "vix_signal": "complacency", "dxy_current": 102.0,
                      "dxy_change": -0.5, "dxy_signal": "bullish_crypto", "spy_current": 515.0,
                      "spy_change": 1.2, "spy_signal": "bullish", "btc_dominance": 48.0,
                      "btc_dom_phase": "alt_season"},
            "onchain": {"fear_greed_value": 68, "fear_greed_classification": "Greed",
                        "fear_greed_signal": "slightly_bearish"},
        },
    },
    {
        "name": "BTC extreme market crash VIX spike",
        "expected_direction": "HOLD_OR_SELL",  # panic selling environment
        "market_data": {
            "pair": "BTC/USD", "price": 42000, "change_1h": -5.2, "change_24h": -22.0,
            "technical": {
                "trend": "strong downtrend", "rsi_state": "oversold (RSI 18.0)",
                "macd_signal": "bearish", "bb_state": "near lower band (potential bounce)",
                "volume_state": "high", "sma_20": 58000, "sma_50": 63000, "sma_200": 55000,
                "rsi": 18.0, "volume_ratio": 4.2,
            },
            "macro": {"vix_current": 48.0, "vix_signal": "extreme_fear", "dxy_current": 108.0,
                      "dxy_change": 2.5, "dxy_signal": "bearish_crypto", "spy_current": 450.0,
                      "spy_change": -5.0, "spy_signal": "bearish", "btc_dominance": 45.0,
                      "btc_dom_phase": "btc_season"},
            "onchain": {"fear_greed_value": 8, "fear_greed_classification": "Extreme Fear",
                        "fear_greed_signal": "bullish"},
        },
    },
]

# Consistency check: run same scenario twice, direction should match
CONSISTENCY_SCENARIO = TEST_SCENARIOS[0]


# ============================================================================
# EVALUATOR
# ============================================================================

def evaluate_model(model_path: str, temperature: float = 0.3) -> dict:
    """Run all evaluation checks and return a results report."""
    from inference.crypto_oracle import CryptoOracle

    print(f"\n{'='*60}")
    print(f"MODEL EVALUATOR")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    oracle = CryptoOracle(model_path=model_path)

    results = {
        "model_path": model_path,
        "total_scenarios": len(TEST_SCENARIOS),
        "parsed": 0,
        "has_confidence": 0,
        "has_stop_loss": 0,
        "has_take_profit": 0,
        "stop_sanity_pass": 0,  # stop on correct side of entry
        "tp_sanity_pass": 0,
        "direction_alignment": 0,  # matches expected_direction hint
        "latencies_s": [],
        "scenarios": [],
    }

    for i, scenario in enumerate(TEST_SCENARIOS):
        name = scenario["name"]
        market_data = scenario["market_data"]
        expected = scenario["expected_direction"]

        print(f"[{i+1}/{len(TEST_SCENARIOS)}] {name}")

        t0 = time.time()
        prediction = oracle.predict(market_data, temperature=temperature)
        latency = time.time() - t0

        rec = prediction["recommendation"]
        direction = rec.get("decision", "UNKNOWN")
        confidence = rec.get("confidence")
        entry = rec.get("entry_price") or market_data["price"]
        stop = rec.get("stop_loss")
        tp = rec.get("take_profit")

        # Parse check
        parsed = direction in ("BUY", "SELL", "HOLD")
        if parsed:
            results["parsed"] += 1

        # Confidence check
        if confidence is not None:
            results["has_confidence"] += 1

        # Stop loss sanity — HOLD never requires a stop
        if direction == "HOLD":
            stop_ok = True
            results["stop_sanity_pass"] += 1
            if stop:
                results["has_stop_loss"] += 1
        else:
            stop_ok = False
            if stop:
                results["has_stop_loss"] += 1
                if direction == "BUY" and stop < entry:
                    stop_ok = True
                    results["stop_sanity_pass"] += 1
                elif direction == "SELL" and stop > entry:
                    stop_ok = True
                    results["stop_sanity_pass"] += 1

        # Take profit sanity — HOLD never requires a TP
        if direction == "HOLD":
            tp_ok = True
            results["tp_sanity_pass"] += 1
            if tp:
                results["has_take_profit"] += 1
        else:
            tp_ok = False
            if tp:
                results["has_take_profit"] += 1
                if direction == "BUY" and tp > entry:
                    tp_ok = True
                    results["tp_sanity_pass"] += 1
                elif direction == "SELL" and tp < entry:
                    tp_ok = True
                    results["tp_sanity_pass"] += 1

        # Direction alignment (soft check against expected)
        aligned = _check_alignment(direction, expected)
        if aligned:
            results["direction_alignment"] += 1

        results["latencies_s"].append(latency)

        scenario_result = {
            "name": name,
            "expected": expected,
            "decision": direction,
            "confidence": confidence,
            "entry": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "parsed": parsed,
            "stop_sanity": stop_ok,
            "tp_sanity": tp_ok,
            "aligned": aligned,
            "latency_s": round(latency, 1),
        }
        results["scenarios"].append(scenario_result)

        status = "PASS" if (parsed and stop_ok and tp_ok) else "WARN"
        print(f"    {status}  direction={direction}  conf={confidence}%  "
              f"stop={'ok' if stop_ok else 'FAIL'}  tp={'ok' if tp_ok else 'FAIL'}  "
              f"aligned={aligned}  latency={latency:.1f}s")

    # Consistency test (run scenario 0 again, direction should match)
    print(f"\n[Consistency] Re-running '{CONSISTENCY_SCENARIO['name']}'...")
    t0 = time.time()
    pred2 = oracle.predict(CONSISTENCY_SCENARIO["market_data"], temperature=temperature)
    latency2 = time.time() - t0
    dir1 = results["scenarios"][0]["decision"]
    dir2 = pred2["recommendation"].get("decision", "UNKNOWN")
    consistent = dir1 == dir2
    results["consistency"] = consistent
    print(f"    Run 1: {dir1}  Run 2: {dir2}  Consistent: {consistent}  ({latency2:.1f}s)")

    # Summary
    n = len(TEST_SCENARIOS)
    avg_latency = sum(results["latencies_s"]) / len(results["latencies_s"])
    results["avg_latency_s"] = round(avg_latency, 1)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Parse rate:           {results['parsed']}/{n} ({results['parsed']/n*100:.0f}%)")
    print(f"Has confidence:       {results['has_confidence']}/{n}")
    print(f"Has stop loss:        {results['has_stop_loss']}/{n}")
    print(f"Has take profit:      {results['has_take_profit']}/{n}")
    print(f"Stop sanity:          {results['stop_sanity_pass']}/{n}")
    print(f"TP sanity:            {results['tp_sanity_pass']}/{n}")
    print(f"Direction alignment:  {results['direction_alignment']}/{n}")
    print(f"Consistency:          {results['consistency']}")
    print(f"Avg latency:          {avg_latency:.1f}s/prediction")

    # Gate: model is production-ready only if all hard checks pass
    hard_pass = (
        results["parsed"] == n                        # 100% parse rate required
        and results["stop_sanity_pass"] >= n - 1      # ≥80% stop sanity
        and results["tp_sanity_pass"] >= n - 1        # ≥80% TP sanity
        and results["has_confidence"] >= n - 1        # ≥80% emit confidence
    )
    results["production_ready"] = hard_pass
    verdict = "READY FOR BACKTESTING" if hard_pass else "NOT READY - fix issues above"
    print(f"\nVerdict: {verdict}")
    print(f"{'='*60}\n")

    return results


def _check_alignment(direction: str, expected: str) -> bool:
    """Soft check: does model direction match expected regime hint?"""
    if "BUY" in expected and direction == "BUY":
        return True
    if "SELL" in expected and direction == "SELL":
        return True
    if "HOLD" in expected and direction == "HOLD":
        return True
    if expected == "HOLD_OR_SELL" and direction in ("HOLD", "SELL"):
        return True
    if expected == "BUY_OR_HOLD" and direction in ("BUY", "HOLD"):
        return True
    if expected == "SELL_OR_HOLD" and direction in ("SELL", "HOLD"):
        return True
    return False


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/crypto-oracle-qwen-32b/final_model",
                        help="Path to fine-tuned model")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Lower temp = more deterministic (better for evaluation)")
    parser.add_argument("--out", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    results = evaluate_model(args.model, args.temperature)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.out}")
