"""
Health/status HTTP server for the crypto oracle paper trader.
Runs as a background daemon thread — call start() once from paper_trader.py.

Endpoints:
  GET /health    — Liveness probe: {"status": "ok", "uptime_seconds": N}
  GET /status    — Full portfolio + readiness JSON
  GET /trades    — Last 50 closed trades
  GET /positions — Current open positions
"""

import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Injected by start()
_state_path: Path = None
_started_at: datetime = None


def _get_full_status() -> dict:
    from trading.trade_logger import (
        get_performance_summary, get_live_readiness,
        get_open_trades, get_daily_pnl,
    )

    state = {}
    if _state_path and _state_path.exists():
        try:
            state = json.loads(_state_path.read_text())
        except Exception:
            pass

    perf      = get_performance_summary()
    readiness = get_live_readiness()
    open_pos  = get_open_trades()
    daily_pnl = get_daily_pnl()
    uptime    = int((datetime.now(timezone.utc) - _started_at).total_seconds()) if _started_at else 0

    gates_list = [
        {"label": label, "pass": passed, "value": value}
        for label, passed, value in readiness.get("gates", [])
    ]

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": uptime,
        "portfolio": {
            "initial_equity":      state.get("initial_equity", 0),
            "cash":                state.get("cash", 0),
            "high_water_mark":     state.get("high_water_mark", 0),
            "total_trades_opened": state.get("total_trades", 0),
            "open_positions":      len(state.get("open_positions", [])),
            "daily_pnl_usd":       daily_pnl,
        },
        "performance": perf,
        "readiness": {
            "verdict": readiness["verdict"],
            "ready":   readiness["ready"],
            "trades_needed": readiness.get("trades_needed"),
            "win_rate":      readiness.get("win_rate"),
            "profit_factor": readiness.get("profit_factor"),
            "total_pnl_usd": readiness.get("total_pnl_usd"),
            "gates": gates_list,
        },
        "open_positions": open_pos,
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request access logs

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path in ("/", "/status"):
                self._json(_get_full_status())

            elif self.path == "/health":
                uptime = int((datetime.now(timezone.utc) - _started_at).total_seconds()) if _started_at else 0
                self._json({"status": "ok", "uptime_seconds": uptime})

            elif self.path == "/trades":
                from trading.trade_logger import get_closed_trades
                self._json({"trades": get_closed_trades(limit=50)})

            elif self.path == "/positions":
                from trading.trade_logger import get_open_trades
                self._json({"positions": get_open_trades()})

            else:
                self._json({"error": "not found"}, 404)

        except Exception as exc:
            self._json({"error": str(exc)}, 500)


def start(port: int = 8080, state_path: Path = None) -> HTTPServer:
    """
    Launch the health server in a background daemon thread.
    Returns the HTTPServer instance (can be used to shut it down).
    """
    global _state_path, _started_at
    _state_path  = state_path
    _started_at  = datetime.now(timezone.utc)

    server = HTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-server")
    thread.start()
    return server
