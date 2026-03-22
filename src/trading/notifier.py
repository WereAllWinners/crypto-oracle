"""
Notifier
Sends alerts to Telegram and/or Discord on key trading events:
  - Trade opened / closed
  - Trade rejected by risk manager
  - Drawdown alert (>X%)
  - Daily P&L summary
  - System errors
  - System start / stop

All sends are fire-and-forget in a background thread so a slow or
failing webhook never blocks the trading loop.

Config (.env):
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...        (your personal or group chat ID)
  DISCORD_WEBHOOK_URL=...
  NOTIFY_DD_THRESHOLD=10      (alert when drawdown exceeds this %)

Usage:
  from trading.notifier import Notifier
  n = Notifier()
  n.trade_opened(pair="BTC/USD", direction="BUY", size_usd=1000,
                 entry=67000, sl=65000, tp=71000, confidence=78)
"""

import logging
import os
import threading
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID",   "")
DISCORD_URL    = os.getenv("DISCORD_WEBHOOK_URL", "")
DD_THRESHOLD   = float(os.getenv("NOTIFY_DD_THRESHOLD", "10"))


# ---------------------------------------------------------------------------
# Internal send helpers
# ---------------------------------------------------------------------------

def _send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(f"[Notifier] Telegram error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[Notifier] Telegram send failed: {e}")


def _send_discord(text: str) -> None:
    if not DISCORD_URL:
        return
    try:
        resp = requests.post(
            DISCORD_URL,
            json={"content": text},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(f"[Notifier] Discord error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[Notifier] Discord send failed: {e}")


def _fire(text: str) -> None:
    """Send to all configured channels in a daemon thread (non-blocking)."""
    if not TELEGRAM_TOKEN and not DISCORD_URL:
        return
    t = threading.Thread(target=_dispatch, args=(text,), daemon=True)
    t.start()


def _dispatch(text: str) -> None:
    _send_telegram(text)
    _send_discord(text)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Notifier:
    """
    Drop-in alert sender. All methods return immediately; actual HTTP
    requests are made in background daemon threads.
    """

    def __init__(self):
        channels = []
        if TELEGRAM_TOKEN and TELEGRAM_CHAT:
            channels.append("Telegram")
        if DISCORD_URL:
            channels.append("Discord")
        if channels:
            logger.info(f"[Notifier] Active channels: {', '.join(channels)}")
        else:
            logger.info("[Notifier] No channels configured -- alerts disabled")

    def trade_opened(
        self,
        pair: str,
        direction: str,
        size_usd: float,
        entry: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        trade_id: str = "",
    ) -> None:
        arrow = "BUY" if direction == "BUY" else "SELL"
        rr = abs(take_profit - entry) / abs(stop_loss - entry) if abs(stop_loss - entry) > 0 else 0
        text = (
            f"[TRADE OPENED]  {arrow} {pair}\n"
            f"Entry:      ${entry:,.4f}\n"
            f"Stop-loss:  ${stop_loss:,.4f}\n"
            f"Take-profit:${take_profit:,.4f}\n"
            f"Size:       ${size_usd:,.2f}\n"
            f"R/R:        1:{rr:.2f}\n"
            f"Confidence: {confidence}%\n"
            f"ID: {trade_id}\n"
            f"{_ts()}"
        )
        _fire(text)

    def trade_closed(
        self,
        pair: str,
        direction: str,
        entry: float,
        exit_price: float,
        net_pnl_usd: float,
        outcome: str,
        reason: str,
        trade_id: str = "",
    ) -> None:
        sign  = "+" if net_pnl_usd >= 0 else ""
        emoji = "WIN" if net_pnl_usd >= 0 else "LOSS"
        text = (
            f"[TRADE CLOSED]  {emoji} {direction} {pair}\n"
            f"Entry:    ${entry:,.4f}\n"
            f"Exit:     ${exit_price:,.4f}\n"
            f"Net P&L:  {sign}${net_pnl_usd:,.2f}\n"
            f"Outcome:  {outcome}  ({reason})\n"
            f"ID: {trade_id}\n"
            f"{_ts()}"
        )
        _fire(text)

    def trade_rejected(
        self,
        pair: str,
        direction: str,
        confidence: Optional[int],
        reasons: list,
    ) -> None:
        reasons_str = "; ".join(reasons[:3])
        text = (
            f"[REJECTED]  {direction} {pair}\n"
            f"Confidence: {confidence}%\n"
            f"Reasons: {reasons_str}\n"
            f"{_ts()}"
        )
        _fire(text)

    def drawdown_alert(
        self,
        current_equity: float,
        high_water_mark: float,
        drawdown_pct: float,
    ) -> None:
        if drawdown_pct < DD_THRESHOLD:
            return
        text = (
            f"[DRAWDOWN ALERT]  {drawdown_pct:.1f}% drawdown\n"
            f"Peak equity:    ${high_water_mark:,.2f}\n"
            f"Current equity: ${current_equity:,.2f}\n"
            f"Loss from peak: ${high_water_mark - current_equity:,.2f}\n"
            f"{_ts()}"
        )
        _fire(text)

    def daily_summary(
        self,
        date: str,
        trades_taken: int,
        trades_won: int,
        daily_pnl_usd: float,
        total_equity: float,
    ) -> None:
        sign = "+" if daily_pnl_usd >= 0 else ""
        wr   = (trades_won / trades_taken * 100) if trades_taken > 0 else 0
        text = (
            f"[DAILY SUMMARY]  {date}\n"
            f"Trades taken: {trades_taken}  (W:{trades_won} / L:{trades_taken - trades_won})\n"
            f"Win rate:     {wr:.0f}%\n"
            f"Daily P&L:   {sign}${daily_pnl_usd:,.2f}\n"
            f"Equity:       ${total_equity:,.2f}\n"
            f"{_ts()}"
        )
        _fire(text)

    def system_started(self, pairs: list, equity: float) -> None:
        text = (
            f"[SYSTEM STARTED]  Crypto Oracle is live\n"
            f"Pairs:   {', '.join(pairs)}\n"
            f"Equity:  ${equity:,.2f}\n"
            f"{_ts()}"
        )
        _fire(text)

    def system_stopped(self, reason: str = "manual shutdown") -> None:
        text = (
            f"[SYSTEM STOPPED]  Crypto Oracle stopped\n"
            f"Reason: {reason}\n"
            f"{_ts()}"
        )
        _fire(text)

    def error(self, message: str, context: str = "") -> None:
        text = (
            f"[ERROR]  Crypto Oracle\n"
            f"{message}\n"
            f"{context}\n"
            f"{_ts()}"
        )
        _fire(text)

    def paused(self, reason: str = "") -> None:
        text = (
            f"[TRADING PAUSED]\n"
            f"Reason: {reason or 'kill-switch activated'}\n"
            f"{_ts()}"
        )
        _fire(text)

    def resumed(self) -> None:
        _fire(f"[TRADING RESUMED]  Kill-switch cleared\n{_ts()}")
