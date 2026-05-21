"""
WebSocket Real-Time Feed Engine
================================
Connects to Polymarket CLOB WebSocket for INSTANT price updates.
No more 30-second polling — reacts to price changes in milliseconds.

Core capabilities:
- Real-time orderbook updates via WebSocket
- Instant price change detection
- Live trade stream monitoring
- Auto-reconnect on disconnect
- Event-driven architecture (callbacks on price change)
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import websockets

from polybotking.config import settings
from polybotking.logger import get_logger

logger = get_logger("websocket_feed")

# Polymarket WebSocket endpoints
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class PriceUpdate:
    """Real-time price update from WebSocket."""
    market_id: str
    token_id: str
    timestamp: datetime
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.5
    spread: float = 0.0
    last_trade_price: float = 0.0
    last_trade_size: float = 0.0
    last_trade_side: str = ""  # "BUY" or "SELL"


@dataclass
class TradeEvent:
    """Real-time trade event from WebSocket."""
    market_id: str
    token_id: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    timestamp: datetime
    maker: str = ""
    taker: str = ""


class WebSocketFeed:
    """
    Real-time WebSocket connection to Polymarket CLOB.
    Provides instant price updates and trade events.
    """

    def __init__(self):
        self._running: bool = False
        self._ws = None
        self._subscribed_markets: set[str] = set()
        self._price_callbacks: list[Callable] = []
        self._trade_callbacks: list[Callable] = []
        self._latest_prices: dict[str, PriceUpdate] = {}
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 60.0

    async def start(self):
        """Start WebSocket feed."""
        self._running = True
        logger.info("websocket_feed_started")

    async def stop(self):
        """Stop WebSocket feed."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("websocket_feed_stopped")

    # =========================================================================
    # SUBSCRIPTION MANAGEMENT
    # =========================================================================

    def on_price_update(self, callback: Callable):
        """Register callback for price updates."""
        self._price_callbacks.append(callback)

    def on_trade(self, callback: Callable):
        """Register callback for trade events."""
        self._trade_callbacks.append(callback)

    async def subscribe_markets(self, token_ids: list[str]):
        """Subscribe to real-time updates for given token IDs."""
        self._subscribed_markets.update(token_ids)

        if self._ws:
            for token_id in token_ids:
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "channel": "market",
                    "assets_id": token_id,
                })
                try:
                    await self._ws.send(subscribe_msg)
                except Exception:
                    pass

        logger.info("markets_subscribed", count=len(token_ids))

    async def unsubscribe_markets(self, token_ids: list[str]):
        """Unsubscribe from market updates."""
        self._subscribed_markets -= set(token_ids)

    # =========================================================================
    # WEBSOCKET CONNECTION
    # =========================================================================

    async def run_forever(self):
        """
        Run WebSocket connection with auto-reconnect.
        Call this as an asyncio task.
        """
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                if not self._running:
                    break
                logger.warning("websocket_disconnected", error=str(e),
                             reconnect_in=f"{self._reconnect_delay:.1f}s")
                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_and_listen(self):
        """Connect to WebSocket and process messages."""
        async with websockets.connect(
            WS_URL,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset backoff on successful connect
            logger.info("websocket_connected", url=WS_URL)

            # Re-subscribe to all markets
            for token_id in self._subscribed_markets:
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "channel": "market",
                    "assets_id": token_id,
                })
                await ws.send(subscribe_msg)

            # Listen for messages
            async for message in ws:
                if not self._running:
                    break
                await self._process_message(message)

    async def _process_message(self, raw_message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type", "")

            if msg_type == "book":
                await self._handle_book_update(data)
            elif msg_type == "trade":
                await self._handle_trade(data)
            elif msg_type == "price_change":
                await self._handle_price_change(data)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("ws_message_parse_error", error=str(e))

    # =========================================================================
    # MESSAGE HANDLERS
    # =========================================================================

    async def _handle_book_update(self, data: dict):
        """Handle orderbook update message."""
        token_id = data.get("asset_id", "")
        market_id = data.get("market", token_id)

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        update = PriceUpdate(
            market_id=market_id,
            token_id=token_id,
            timestamp=datetime.utcnow(),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
        )

        self._latest_prices[token_id] = update

        # Notify callbacks
        for callback in self._price_callbacks:
            try:
                await callback(update)
            except Exception:
                pass

    async def _handle_trade(self, data: dict):
        """Handle trade event message."""
        token_id = data.get("asset_id", "")
        market_id = data.get("market", token_id)

        event = TradeEvent(
            market_id=market_id,
            token_id=token_id,
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            side=data.get("side", "").upper(),
            timestamp=datetime.utcnow(),
            maker=data.get("maker", ""),
            taker=data.get("taker", ""),
        )

        # Update latest price from trade
        if token_id in self._latest_prices:
            self._latest_prices[token_id].last_trade_price = event.price
            self._latest_prices[token_id].last_trade_size = event.size
            self._latest_prices[token_id].last_trade_side = event.side

        # Notify callbacks
        for callback in self._trade_callbacks:
            try:
                await callback(event)
            except Exception:
                pass

    async def _handle_price_change(self, data: dict):
        """Handle price change notification."""
        token_id = data.get("asset_id", "")
        new_price = float(data.get("price", 0))

        if token_id in self._latest_prices:
            self._latest_prices[token_id].mid_price = new_price
            self._latest_prices[token_id].timestamp = datetime.utcnow()

    # =========================================================================
    # PRICE ACCESS
    # =========================================================================

    def get_latest_price(self, token_id: str) -> Optional[PriceUpdate]:
        """Get the latest price for a token."""
        return self._latest_prices.get(token_id)

    def get_all_prices(self) -> dict[str, PriceUpdate]:
        """Get all latest prices."""
        return self._latest_prices.copy()

    # =========================================================================
    # SPIKE DETECTION (Real-time)
    # =========================================================================

    async def detect_price_spike(self, token_id: str, threshold: float = 0.05) -> bool:
        """
        Detect if a price spike occurred (>5% move in short time).
        Useful for immediate entry/exit decisions.
        """
        current = self._latest_prices.get(token_id)
        if not current:
            return False

        # Compare mid_price to last_trade_price
        if current.last_trade_price > 0:
            move = abs(current.mid_price - current.last_trade_price)
            if move >= threshold:
                logger.info("price_spike_detected", token_id=token_id[:12],
                          move=f"{move:.3f}", threshold=threshold)
                return True

        return False


# Singleton
websocket_feed = WebSocketFeed()
