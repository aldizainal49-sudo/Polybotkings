"""
Execution Engine
================
Handles order placement, fills tracking, and position management
on Polymarket CLOB (Central Limit Order Book).

Core capabilities:
- Place limit/market orders via py-clob-client
- Track order fills and partial fills
- Manage open positions
- Exit positions on signal or timing
- Fee-aware execution with slippage control
"""

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import httpx
from sqlalchemy import select, update

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Trade, TradeStatus, SignalType, Market, async_session

logger = get_logger("execution")

CLOB_BASE_URL = "https://clob.polymarket.com"


@dataclass
class OrderResult:
    """Result of an order placement."""
    success: bool
    order_id: str = ""
    fill_price: float = 0.0
    filled_size: float = 0.0
    status: str = ""
    error: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class Position:
    """Active position tracking."""
    trade_id: int
    market_id: str
    side: str  # YES/NO
    entry_price: float
    size: float
    cost: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: datetime = None

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.utcnow()


class ExecutionEngine:
    """
    Order execution and position management.
    Interfaces with Polymarket CLOB for trade placement.
    """

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.clob_client = None  # py-clob-client instance
        self.active_positions: dict[str, Position] = {}  # market_id -> position
        self._running: bool = False

    async def start(self):
        """Initialize execution engine with CLOB client."""
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

        # Initialize py-clob-client-v2 (CLOB V2 with pUSD collateral)
        try:
            from py_clob_client_v2.client import ClobClient
            from py_clob_client_v2.clob_types import ApiCreds

            if settings.polymarket.api_key and settings.polymarket.private_key:
                creds = ApiCreds(
                    api_key=settings.polymarket.api_key,
                    api_secret=settings.polymarket.api_secret,
                    api_passphrase=settings.polymarket.api_passphrase,
                )
                self.clob_client = ClobClient(
                    host=CLOB_BASE_URL,
                    chain_id=settings.polymarket.chain_id,
                    key=settings.polymarket.private_key,
                    creds=creds,
                    # V2: pUSD collateral, deposit wallets
                    funder=settings.polymarket.wallet_type,
                )
                logger.info("clob_client_v2_initialized", collateral="pUSD", wallet_type=settings.polymarket.wallet_type)
            else:
                logger.warning("clob_client_no_credentials", msg="Running in paper-trade mode")

        except ImportError:
            logger.warning("py_clob_client_v2_not_installed", msg="Running in simulation mode")

        # Load active positions from DB
        await self._load_active_positions()

        self._running = True
        logger.info("execution_engine_started", active_positions=len(self.active_positions))

    async def stop(self):
        """Shutdown execution engine."""
        self._running = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info("execution_engine_stopped")

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    async def execute_trade(self, decision) -> Optional[OrderResult]:
        """
        Execute a trading decision from the orchestrator.
        
        Args:
            decision: TradingDecision from orchestrator
            
        Returns:
            OrderResult with fill details
        """
        from polybotking.orchestrator import TradingDecision

        if not isinstance(decision, TradingDecision):
            return None

        if not decision.position_size:
            return OrderResult(success=False, error="No position size")

        market_id = decision.market_id
        direction = decision.direction
        size = decision.position_size.size_usd
        price = decision.market_signal.market_price if decision.market_signal else 0.5

        logger.info(
            "executing_trade",
            market=market_id[:12],
            direction=direction,
            size=f"${size:.2f}",
            price=f"{price:.3f}",
            ev=f"{decision.final_ev:.3f}",
        )

        # Place order
        result = await self._place_order(
            market_id=market_id,
            side=direction,
            price=price,
            size=size,
        )

        if result.success:
            # Record trade in database
            await self._record_trade(decision, result)

            # Track active position
            self.active_positions[market_id] = Position(
                trade_id=0,  # Updated after DB insert
                market_id=market_id,
                side=direction,
                entry_price=result.fill_price or price,
                size=size,
                cost=size * (result.fill_price or price),
            )

            logger.info(
                "trade_executed",
                market=market_id[:12],
                direction=direction,
                fill_price=f"{result.fill_price:.3f}",
                order_id=result.order_id[:12],
            )
        else:
            logger.error(
                "trade_execution_failed",
                market=market_id[:12],
                error=result.error,
            )

        return result

    async def _place_order(
        self,
        market_id: str,
        side: str,
        price: float,
        size: float,
    ) -> OrderResult:
        """
        Place order on Polymarket CLOB.
        Uses limit order at slightly aggressive price for faster fills.
        """
        if self.clob_client is None:
            # Paper trading / simulation mode
            return await self._simulate_order(market_id, side, price, size)

        try:
            from py_clob_client_v2.clob_types import OrderArgs, OrderType
            from py_clob_client_v2.order_builder.constants import BUY, SELL

            # Determine token_id (YES token for BUY YES, NO token for BUY NO)
            token_id = await self._get_token_id(market_id, side)
            if not token_id:
                return OrderResult(success=False, error="Could not resolve token_id")

            # Build order - use limit order with slight price improvement
            # For BUY: set price slightly above best ask for faster fill
            aggressive_price = price * 1.005 if price < 0.95 else price

            order_args = OrderArgs(
                token_id=token_id,
                price=round(aggressive_price, 2),
                size=size,
                side=BUY,
                order_type=OrderType.GTC,  # Good till cancelled
            )

            # Sign and place order
            signed_order = self.clob_client.create_order(order_args)
            response = self.clob_client.post_order(signed_order)

            order_id = response.get("orderID", "")
            if order_id:
                # Wait briefly for fill
                await asyncio.sleep(2.0)

                # Check if filled
                order_status = await self._check_order_status(order_id)

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    fill_price=order_status.get("avg_fill_price", aggressive_price),
                    filled_size=order_status.get("filled_size", size),
                    status=order_status.get("status", "open"),
                )
            else:
                return OrderResult(
                    success=False,
                    error=response.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error("order_placement_error", error=str(e))
            return OrderResult(success=False, error=str(e))

    async def _simulate_order(
        self, market_id: str, side: str, price: float, size: float
    ) -> OrderResult:
        """Simulate order execution for paper trading."""
        # Simulate slight slippage
        import random
        slippage = random.uniform(0.001, 0.005)
        fill_price = price + slippage if side == "YES" else price - slippage

        return OrderResult(
            success=True,
            order_id=f"sim_{market_id[:8]}_{datetime.utcnow().timestamp():.0f}",
            fill_price=fill_price,
            filled_size=size,
            status="filled",
        )

    async def _get_token_id(self, market_id: str, side: str) -> Optional[str]:
        """Get the CLOB token ID for a market/side combination."""
        try:
            resp = await self.http_client.get(
                f"{CLOB_BASE_URL}/markets/{market_id}"
            )
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get("tokens", [])
                for token in tokens:
                    if token.get("outcome", "").upper() == side.upper():
                        return token.get("token_id")
            return None
        except httpx.HTTPError:
            return None

    async def _check_order_status(self, order_id: str) -> dict:
        """Check status of a placed order."""
        if not self.clob_client:
            return {"status": "filled", "avg_fill_price": 0, "filled_size": 0}

        try:
            order = self.clob_client.get_order(order_id)
            return {
                "status": order.get("status", "unknown"),
                "avg_fill_price": float(order.get("avg_fill_price", 0) or 0),
                "filled_size": float(order.get("size_matched", 0) or 0),
            }
        except Exception:
            return {"status": "unknown"}

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    async def close_position(self, market_id: str, reason: str = "signal") -> Optional[OrderResult]:
        """Close an active position."""
        position = self.active_positions.get(market_id)
        if not position:
            return None

        # Sell the position (opposite side)
        exit_side = "NO" if position.side == "YES" else "YES"
        exit_price = position.current_price or position.entry_price

        result = await self._place_order(
            market_id=market_id,
            side=exit_side,
            price=1 - exit_price,  # Selling YES = buying NO equivalent
            size=position.size,
        )

        if result.success:
            # Calculate PnL
            if position.side == "YES":
                pnl = (result.fill_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - result.fill_price) * position.size

            # Update trade in DB
            await self._close_trade_in_db(market_id, result.fill_price, pnl)

            # Remove from active positions
            del self.active_positions[market_id]

            logger.info(
                "position_closed",
                market=market_id[:12],
                pnl=f"${pnl:.2f}",
                reason=reason,
            )

            # Feed outcome to risk engine
            from polybotking.engines.risk_engine import risk_engine, TradeOutcome
            outcome = TradeOutcome(
                market_id=market_id,
                edge_at_entry=0.0,  # Retrieved from trade record
                predicted_prob=0.0,
                actual_outcome=pnl > 0,
                pnl=pnl,
                hold_time_hours=(datetime.utcnow() - position.entry_time).total_seconds() / 3600,
            )
            risk_engine.record_outcome(outcome)

        return result

    async def check_resolved_markets(self):
        """Check if any active positions are in resolved markets."""
        for market_id in list(self.active_positions.keys()):
            try:
                resp = await self.http_client.get(f"{CLOB_BASE_URL}/markets/{market_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("closed") or data.get("resolved"):
                        # Market resolved - determine outcome
                        winning_outcome = data.get("winning_outcome", "")
                        position = self.active_positions[market_id]

                        if winning_outcome.upper() == position.side:
                            # Won: payout = size * (1 - entry_price) / entry_price
                            pnl = position.size * (1 - position.entry_price)
                        else:
                            # Lost: lose entire cost
                            pnl = -position.cost

                        await self._close_trade_in_db(market_id, 1.0 if pnl > 0 else 0.0, pnl)
                        del self.active_positions[market_id]

                        # Record outcome
                        from polybotking.engines.risk_engine import risk_engine, TradeOutcome
                        outcome = TradeOutcome(
                            market_id=market_id,
                            edge_at_entry=0.0,
                            predicted_prob=0.0,
                            actual_outcome=pnl > 0,
                            pnl=pnl,
                            hold_time_hours=(datetime.utcnow() - position.entry_time).total_seconds() / 3600,
                        )
                        risk_engine.record_outcome(outcome)

                        logger.info("market_resolved", market=market_id[:12], pnl=f"${pnl:.2f}", won=pnl > 0)

            except Exception as e:
                logger.warning("resolution_check_error", market=market_id[:12], error=str(e))

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    async def _record_trade(self, decision, result: OrderResult):
        """Record a new trade in database."""
        async with async_session() as session:
            trade = Trade(
                market_id=decision.market_id,
                order_id=result.order_id,
                side=decision.direction,
                entry_price=result.fill_price or decision.market_signal.market_price,
                size=decision.position_size.size_usd,
                cost=decision.position_size.size_usd * (result.fill_price or decision.market_signal.market_price),
                signal_type=decision.market_signal.signal_type if decision.market_signal else SignalType.COMBINED,
                confidence=decision.combined_confidence,
                kelly_size=decision.position_size.kelly_adjusted,
                edge_at_entry=decision.combined_edge,
                status=TradeStatus.OPEN,
            )
            session.add(trade)
            await session.commit()

    async def _close_trade_in_db(self, market_id: str, exit_price: float, pnl: float):
        """Update trade record with exit data."""
        async with async_session() as session:
            result = await session.execute(
                select(Trade).where(
                    Trade.market_id == market_id,
                    Trade.status == TradeStatus.OPEN,
                ).order_by(Trade.entry_time.desc()).limit(1)
            )
            trade = result.scalar_one_or_none()
            if trade:
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.status = TradeStatus.CLOSED
                trade.exit_time = datetime.utcnow()
                await session.commit()

    async def _load_active_positions(self):
        """Load active positions from database."""
        async with async_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.status == TradeStatus.OPEN)
            )
            trades = result.scalars().all()
            for t in trades:
                self.active_positions[t.market_id] = Position(
                    trade_id=t.id,
                    market_id=t.market_id,
                    side=t.side,
                    entry_price=t.entry_price,
                    size=t.size,
                    cost=t.cost,
                    entry_time=t.entry_time,
                )

    # =========================================================================
    # POSITION MONITORING
    # =========================================================================

    async def update_positions(self):
        """Update current prices for all active positions."""
        for market_id, position in self.active_positions.items():
            try:
                resp = await self.http_client.get(
                    f"{CLOB_BASE_URL}/book",
                    params={"token_id": market_id}
                )
                if resp.status_code == 200:
                    book = resp.json()
                    bids = book.get("bids", [])
                    if bids:
                        current_price = float(bids[0]["price"])
                        position.current_price = current_price

                        # Calculate unrealized PnL
                        if position.side == "YES":
                            position.unrealized_pnl = (current_price - position.entry_price) * position.size
                        else:
                            position.unrealized_pnl = (position.entry_price - current_price) * position.size

            except Exception:
                pass


# Singleton
execution_engine = ExecutionEngine()
