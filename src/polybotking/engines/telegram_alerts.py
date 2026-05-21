"""
Telegram Live Dashboard
========================
Real-time trade notifications and bot status via Telegram.

Sends alerts for:
- Trade executed (entry)
- Trade closed (exit + PnL)
- Daily performance summary
- Circuit breaker activated
- Geo-block warnings
"""

import asyncio
from datetime import datetime
from typing import Optional

import httpx

from polybotking.config import settings
from polybotking.logger import get_logger

logger = get_logger("telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerts:
    """Send real-time trading alerts via Telegram bot."""

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self._enabled: bool = False

    async def start(self):
        """Initialize Telegram alerts."""
        self._enabled = bool(settings.alerts.enable_telegram and
                            settings.alerts.telegram_bot_token and
                            settings.alerts.telegram_chat_id)
        if self._enabled:
            self.http_client = httpx.AsyncClient(timeout=10.0)
            logger.info("telegram_alerts_started")
        else:
            logger.info("telegram_alerts_disabled")

    async def stop(self):
        """Shutdown."""
        if self.http_client:
            await self.http_client.aclose()

    async def _send(self, text: str):
        """Send message to Telegram."""
        if not self._enabled or not self.http_client:
            return
        try:
            url = TELEGRAM_API.format(token=settings.alerts.telegram_bot_token)
            await self.http_client.post(url, json={
                "chat_id": settings.alerts.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })
        except Exception as e:
            logger.warning("telegram_send_error", error=str(e))

    async def trade_executed(self, market_id: str, direction: str, price: float,
                            size: float, ev: float, edge: float):
        """Alert: new trade placed."""
        msg = (
            f"📊 *TRADE EXECUTED*\n"
            f"Market: `{market_id[:16]}`\n"
            f"Direction: *{direction}* @ ${price:.3f}\n"
            f"Size: ${size:.2f}\n"
            f"Edge: {edge:.1%} | EV: {ev:.1%}"
        )
        await self._send(msg)

    async def trade_closed(self, market_id: str, pnl: float, bankroll: float,
                          win_rate: float, reason: str):
        """Alert: trade closed with PnL."""
        emoji = "💰" if pnl > 0 else "📉"
        msg = (
            f"{emoji} *TRADE CLOSED*\n"
            f"Market: `{market_id[:16]}`\n"
            f"PnL: *${pnl:+.2f}*\n"
            f"Bankroll: ${bankroll:.2f}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Reason: {reason}"
        )
        await self._send(msg)

    async def daily_summary(self, bankroll: float, pnl_today: float, trades_today: int,
                           win_rate: float, drawdown: float):
        """Alert: daily performance summary."""
        msg = (
            f"📈 *DAILY SUMMARY*\n"
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n"
            f"Bankroll: ${bankroll:.2f}\n"
            f"Today PnL: ${pnl_today:+.2f}\n"
            f"Trades: {trades_today}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Drawdown: {drawdown:.1%}"
        )
        await self._send(msg)

    async def circuit_breaker(self, reason: str, bankroll: float):
        """Alert: circuit breaker activated."""
        msg = (
            f"🚨 *CIRCUIT BREAKER ACTIVATED*\n"
            f"Reason: {reason}\n"
            f"Bankroll: ${bankroll:.2f}\n"
            f"Trading PAUSED until conditions improve."
        )
        await self._send(msg)

    async def bot_started(self, bankroll: float, ip: str, country: str):
        """Alert: bot started successfully."""
        msg = (
            f"🤖 *PolyBotKing STARTED*\n"
            f"Bankroll: ${bankroll:.2f}\n"
            f"VPS: {ip} ({country})\n"
            f"Status: Running 24/7"
        )
        await self._send(msg)


# Singleton
telegram_alerts = TelegramAlerts()
