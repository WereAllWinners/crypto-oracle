#!/usr/bin/env python3
"""
Download historical auxiliary data needed for dataset building:
  1. Macro data (DXY, SPY, VIX) via yfinance → data/macro/historical_macro.csv
  2. Fear & Greed Index via alternative.me API → data/onchain/historical_fear_greed.csv

Run once before rebuilding the dataset:
  conda run -n unsloth python scripts/download_historical_aux_data.py
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
MACRO_OUT  = ROOT / "data" / "macro"  / "historical_macro.csv"
FG_OUT     = ROOT / "data" / "onchain" / "historical_fear_greed.csv"
START_DATE = "2018-01-01"
END_DATE   = datetime.now().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────

def _dxy_signal(change_5d: float) -> str:
    """Rising DXY = stronger dollar = headwind for crypto."""
    if change_5d < -0.5:
        return "bullish_crypto"
    if change_5d > 0.5:
        return "bearish_crypto"
    return "neutral"


def _spy_signal(change_5d: float) -> str:
    """Rising equities = risk-on = tailwind for crypto."""
    if change_5d > 1.0:
        return "bullish_crypto"
    if change_5d < -2.0:
        return "bearish"
    return "neutral"


def _vix_signal(vix: float) -> str:
    if vix > 30:
        return "high_fear"
    if vix > 20:
        return "elevated"
    return "normal"


def _fg_signal(value: int) -> str:
    """Contrarian: extreme fear → buy signal, extreme greed → sell signal."""
    if value <= 25:
        return "bullish"
    if value >= 75:
        return "bearish"
    return "neutral"


# ─────────────────────────────────────────────
# 1. Macro data
# ─────────────────────────────────────────────

def download_macro() -> None:
    print(f"\n[1/2] Downloading macro data (DXY, SPY, VIX) {START_DATE} → {END_DATE} ...")

    tickers = {"dxy": "DX-Y.NYB", "spy": "SPY", "vix": "^VIX"}
    raw = {}
    for name, symbol in tickers.items():
        print(f"      {symbol} ...", end=" ", flush=True)
        hist = yf.download(symbol, start=START_DATE, end=END_DATE,
                           progress=False, auto_adjust=True)
        raw[name] = hist["Close"].squeeze()
        print(f"{len(raw[name])} rows")

    df = pd.DataFrame(raw)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()

    # Forward-fill weekends/holidays so every calendar day has a value
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="D")
    df = df.reindex(full_idx).ffill().dropna()

    # 5-day change for trend-direction signals; 1-day change for context display
    df["dxy_change_5d"] = df["dxy"].pct_change(5) * 100
    df["spy_change_5d"] = df["spy"].pct_change(5) * 100
    df["dxy_change_1d"] = df["dxy"].pct_change(1) * 100
    df["spy_change_1d"] = df["spy"].pct_change(1) * 100

    df["dxy_signal"] = df["dxy_change_5d"].apply(_dxy_signal)
    df["spy_signal"] = df["spy_change_5d"].apply(_spy_signal)
    df["vix_signal"] = df["vix"].apply(_vix_signal)

    df = df.drop(columns=["dxy_change_5d", "spy_change_5d"]).dropna()

    MACRO_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MACRO_OUT)
    print(f"      ✅ Saved {len(df):,} rows → {MACRO_OUT}")
    print(f"         Range: {df.index[0].date()} to {df.index[-1].date()}")


# ─────────────────────────────────────────────
# 2. Fear & Greed Index
# ─────────────────────────────────────────────

def download_fear_greed() -> None:
    print("\n[2/2] Downloading Fear & Greed Index from alternative.me ...")
    url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=us"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    payload = resp.json()
    entries = payload.get("data", [])
    if not entries:
        print("      ⚠️  No data returned — check API availability")
        return

    records = []
    for item in entries:
        try:
            # date_format=us → MM/DD/YYYY
            date = pd.to_datetime(item["timestamp"], format="%m/%d/%Y")
        except Exception:
            date = pd.to_datetime(item["timestamp"])
        records.append({
            "date":           date.normalize(),
            "value":          int(item["value"]),
            "classification": item["value_classification"],
            "signal":         _fg_signal(int(item["value"])),
        })

    df = pd.DataFrame(records).set_index("date").sort_index()

    FG_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FG_OUT)
    print(f"      ✅ Saved {len(df):,} rows → {FG_OUT}")
    print(f"         Range: {df.index[0].date()} to {df.index[-1].date()}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HISTORICAL AUX DATA DOWNLOADER")
    print("=" * 60)

    download_macro()
    download_fear_greed()

    print("\n✅ All done. Re-run dataset_builder to rebuild training data.")
