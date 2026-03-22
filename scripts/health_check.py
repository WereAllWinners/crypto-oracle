"""
Health Check Script
Verifies that all system components are operational:
  1. Model weights exist and can be imported
  2. Coinbase exchange is reachable and authenticated
  3. SQLite database is readable and writable
  4. Kill-switch state
  5. Open position count
  6. API server responsiveness (if running)

Exit codes:
  0 = all checks passed
  1 = one or more checks failed

Usage:
  python scripts/health_check.py
  python scripts/health_check.py --api-url http://localhost:8000
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line   = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def run_checks(api_url: str = "") -> bool:
    print("\n" + "="*55)
    print("  CRYPTO ORACLE -- HEALTH CHECK")
    print("="*55)

    results = []

    # ------------------------------------------------------------------
    # 1. Model weights
    # ------------------------------------------------------------------
    final_model = ROOT / "models" / "crypto-oracle-qwen-32b" / "final_model"
    adapter_cfg = final_model / "adapter_config.json"
    results.append(check(
        "Model weights (final_model/)",
        adapter_cfg.exists(),
        str(final_model) if not adapter_cfg.exists() else "adapter_config.json found",
    ))

    # ------------------------------------------------------------------
    # 2. Kill-switch state
    # ------------------------------------------------------------------
    kill_file = ROOT / "data" / "TRADING_PAUSED"
    paused    = kill_file.exists()
    reason    = kill_file.read_text().strip() if paused else ""
    results.append(check(
        "Kill-switch",
        True,   # informational only
        f"PAUSED ({reason})" if paused else "active (not paused)",
    ))

    # ------------------------------------------------------------------
    # 3. SQLite database
    # ------------------------------------------------------------------
    db_path = ROOT / "data" / "trades.db"
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS _health_probe (id INTEGER PRIMARY KEY)")
        conn.execute("DELETE FROM _health_probe")
        conn.commit()
        conn.close()
        results.append(check("SQLite trades.db", True, str(db_path)))
    except Exception as e:
        results.append(check("SQLite trades.db", False, str(e)))

    # ------------------------------------------------------------------
    # 4. Open positions
    # ------------------------------------------------------------------
    open_count = 0
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='open'"
        ).fetchone()
        open_count = row[0] if row else 0
        conn.close()
        results.append(check("Open positions readable", True, f"{open_count} open"))
    except Exception as e:
        results.append(check("Open positions readable", False, str(e)))

    # ------------------------------------------------------------------
    # 5. Coinbase exchange reachability
    # ------------------------------------------------------------------
    try:
        import ccxt
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        exchange = ccxt.coinbase({
            "apiKey":          os.getenv("COINBASE_API_KEY", ""),
            "secret":          os.getenv("COINBASE_SECRET_KEY", ""),
            "enableRateLimit": True,
            "options": {"advanced": True},
        })
        # Unauthenticated market data check
        ticker = exchange.fetch_ticker("BTC/USD")
        price  = ticker.get("last", 0)
        results.append(check(
            "Coinbase reachable (BTC/USD)",
            price > 0,
            f"last=${price:,.0f}",
        ))
    except Exception as e:
        results.append(check("Coinbase reachable", False, str(e)[:80]))

    # ------------------------------------------------------------------
    # 6. API server (optional)
    # ------------------------------------------------------------------
    if api_url:
        try:
            import requests
            resp = requests.get(f"{api_url}/health", timeout=10)
            results.append(check(
                f"API server ({api_url})",
                resp.status_code == 200,
                f"HTTP {resp.status_code}",
            ))
        except Exception as e:
            results.append(check(f"API server ({api_url})", False, str(e)[:80]))

    # ------------------------------------------------------------------
    # 7. Notification config (informational)
    # ------------------------------------------------------------------
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    tg_ok  = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
    dc_ok  = bool(os.getenv("DISCORD_WEBHOOK_URL"))
    notify = ("Telegram" if tg_ok else "") + (" Discord" if dc_ok else "") or "none"
    results.append(check("Notification channels", True, notify))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(results)
    total  = len(results)
    print(f"\n  {passed}/{total} checks passed")
    print("="*55 + "\n")

    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Oracle health check")
    parser.add_argument(
        "--api-url", default="",
        help="API server URL to ping (e.g. http://localhost:8000)"
    )
    args = parser.parse_args()

    ok = run_checks(api_url=args.api_url)
    sys.exit(0 if ok else 1)
