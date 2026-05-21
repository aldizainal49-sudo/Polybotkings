"""
Exit Optimization Engine
=========================
Takes profit BEFORE market resolution. Sells positions at peak instead
of waiting days for resolution.

Core capabilities:
- Take profit at configurable thresholds (e.g., +20% unrealized)
- Trailing stop-loss (lock in gains, limit downside)
- Time-based exits (exit before resolution if profit target met)
- Momentum-based exits (sell when momentum fades)
- Capital recycling (free up capital for new trades faster)

Result: Capital rotates 3-5x faster → compound growth accelerated
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from polybotking.config import settings
from polybotking.logger import get_logger

logger = get_logger("exit_optimizer")


@dataclass
class ExitDecision:
    """Decision to exit a position."""
    market_id: str
    action: str  # "TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP", "TIME_EXIT"
    reason: str
    current_price: float
    entry_price: float
    unrealized_pnl_pct: float
    urgency: str  # "immediate", "soon", "monitor"


class ExitOptimizer:
    """
    Optimizes exit timing for active positions.
    Sells before resolution to recycle capital faster.
    """

    def __init__(self):
        self._running: bool = False
        # Exit thresholds
        self.take_profit_pct: float = 0.25  # Take profit at +25% unrealized
        self.stop_loss_pct: float = -0.15  # Stop loss at -15%
        self.trailing_stop_pct: float = 0.10  # Trail 10% from peak
        self.time_exit_hours: float = 48.0  # Consider exit after 48h if profitable
        # Track peak prices for trailing stop
        self.peak_prices: dict[str, float] = {}  # market_id -> highest price seen

    async def start(self):
        """Start exit optimizer."""
        self._running = True
        logger.info("exit_optimizer_started",
                   take_profit=f"{self.take_profit_pct:.0%}",
                   stop_loss=f"{self.stop_loss_pct:.0%}",
                   trailing_stop=f"{self.trailing_stop_pct:.0%}")

    async def stop(self):
        """Stop exit optimizer."""
        self._running = False
        logger.info("exit_optimizer_stopped")

    # =========================================================================
    # EXIT ANALYSIS
    # =========================================================================

    def analyze_position(
        self,
        market_id: str,
        side: str,
        entry_price: float,
        current_price: float,
        entry_time: datetime,
        size: float,
    ) -> Optional[ExitDecision]:
        """
        Analyze a position and decide whether to exit.
        
        Checks in order:
        1. Stop loss (protect capital)
        2. Take profit (lock gains)
        3. Trailing stop (ride momentum, exit on reversal)
        4. Time-based exit (capital efficiency)
        """
        # Calculate unrealized PnL percentage
        if side == "YES":
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pnl_pct = (entry_price - current_price) / entry_price

        # Update peak price for trailing stop
        if market_id not in self.peak_prices:
            self.peak_prices[market_id] = current_price
        else:
            if side == "YES" and current_price > self.peak_prices[market_id]:
                self.peak_prices[market_id] = current_price
            elif side == "NO" and current_price < self.peak_prices[market_id]:
                self.peak_prices[market_id] = current_price

        # --- CHECK 1: Stop Loss ---
        if unrealized_pnl_pct <= self.stop_loss_pct:
            return ExitDecision(
                market_id=market_id,
                action="STOP_LOSS",
                reason=f"Unrealized loss {unrealized_pnl_pct:.1%} hit stop loss {self.stop_loss_pct:.1%}",
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
                urgency="immediate",
            )

        # --- CHECK 2: Take Profit ---
        if unrealized_pnl_pct >= self.take_profit_pct:
            return ExitDecision(
                market_id=market_id,
                action="TAKE_PROFIT",
                reason=f"Unrealized profit {unrealized_pnl_pct:.1%} hit target {self.take_profit_pct:.1%}",
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
                urgency="soon",
            )

        # --- CHECK 3: Trailing Stop ---
        peak = self.peak_prices.get(market_id, current_price)
        if side == "YES":
            drop_from_peak = (peak - current_price) / peak if peak > 0 else 0
        else:
            drop_from_peak = (current_price - peak) / peak if peak > 0 else 0

        # Only activate trailing stop if we're already in profit
        if unrealized_pnl_pct > 0.10 and drop_from_peak >= self.trailing_stop_pct:
            return ExitDecision(
                market_id=market_id,
                action="TRAILING_STOP",
                reason=f"Price dropped {drop_from_peak:.1%} from peak. Locking {unrealized_pnl_pct:.1%} profit",
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
                urgency="immediate",
            )

        # --- CHECK 4: Time-based Exit ---
        hold_time = (datetime.utcnow() - entry_time).total_seconds() / 3600
        if hold_time >= self.time_exit_hours and unrealized_pnl_pct > 0.05:
            return ExitDecision(
                market_id=market_id,
                action="TIME_EXIT",
                reason=f"Held {hold_time:.0f}h with {unrealized_pnl_pct:.1%} profit. Recycle capital",
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
                urgency="soon",
            )

        return None  # Hold position

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    def analyze_all_positions(self, positions: dict) -> list[ExitDecision]:
        """
        Analyze all active positions for exit signals.
        Returns list of positions that should be exited.
        """
        exits = []

        for market_id, position in positions.items():
            decision = self.analyze_position(
                market_id=market_id,
                side=position.side,
                entry_price=position.entry_price,
                current_price=position.current_price or position.entry_price,
                entry_time=position.entry_time or datetime.utcnow(),
                size=position.size,
            )

            if decision:
                exits.append(decision)
                logger.info(
                    "exit_signal",
                    market=market_id[:12],
                    action=decision.action,
                    pnl_pct=f"{decision.unrealized_pnl_pct:.1%}",
                    reason=decision.reason,
                )

        return exits

    def clear_peak(self, market_id: str):
        """Clear peak price tracking when position is closed."""
        self.peak_prices.pop(market_id, None)


# Singleton
exit_optimizer = ExitOptimizer()
