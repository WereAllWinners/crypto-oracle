"""
Quick diagnostic: print the raw model output for one strong BUY and one strong SELL
scenario to see what the model is actually generating before the parser touches it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.crypto_oracle import CryptoOracle

oracle = CryptoOracle(model_path="models/crypto-oracle-qwen-32b-v3/final_model")

scenarios = [
    {
        "label": "STRONG BULL — price above all SMAs, low RSI, MACD bullish, breakout",
        "data": {
            "pair": "BTC/USD",
            "price": 65000,
            "change_1h": 2.5,
            "change_24h": 8.2,
            "technical": {
                "trend": "strong uptrend (strong)",
                "rsi_state": "neutral (RSI 48.2)",
                "macd_signal": "bullish crossover",
                "bb_state": "mid-range",
                "volume_state": "high (2.1x average)",
                "sma_20": 62000,
                "sma_50": 58000,
                "sma_200": 45000,
            },
        },
    },
    {
        "label": "STRONG BEAR — price below all SMAs, RSI overbought, MACD bearish, breakdown",
        "data": {
            "pair": "BTC/USD",
            "price": 28000,
            "change_1h": -3.1,
            "change_24h": -9.4,
            "technical": {
                "trend": "strong downtrend (strong)",
                "rsi_state": "overbought (RSI 72.4)",
                "macd_signal": "bearish crossover",
                "bb_state": "near upper band (potential reversal)",
                "volume_state": "high (2.8x average)",
                "sma_20": 31000,
                "sma_50": 35000,
                "sma_200": 42000,
            },
        },
    },
]

for s in scenarios:
    print("\n" + "=" * 70)
    print(f"SCENARIO: {s['label']}")
    print("=" * 70)

    result = oracle.predict(s["data"], temperature=0.7)
    rec = result["recommendation"]

    print(f"\nPARSED: {rec['decision']}  conf={rec['confidence']}%")
    print(f"Entry={rec['entry_price']}  SL={rec['stop_loss']}  TP={rec['take_profit']}")
    print("\n--- RAW MODEL OUTPUT ---")
    print(result["full_response"])
    print("--- END RAW OUTPUT ---")
