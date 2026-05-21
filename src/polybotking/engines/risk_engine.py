"""
Risk & Position Sizing Engine
==============================
Kelly Criterion-based position sizing with dynamic risk management.
Gets SMARTER the more it trades (Bayesian probability updates).

Core capabilities:
- Kelly Criterion optimal sizing (fractional Kelly for safety)
- Bankroll management with compound growth
- Dynamic probability recalibration from trade outcomes
- Drawdown protection (circuit breakers)
- Auto risk-size adjustment based on market conditions
- Win-streak/loss-streak detection and adaptation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from sqlalchemy import select, func

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Trade, TradeStatus, BotState, async_session

logger = get_logger("risk_engine")


@dataclass
class PositionSize:
    """Calculated position size for a trade."""
    market_id: str
    size_usd: float
    size_pct_bankroll: float
    kelly_full: float  # Full Kelly fraction
    kelly_adjusted: float  # Fractional Kelly used
    edge: float
    probability: float
    max_loss: float
    risk_reward_ratio: float
    reasoning: str = ""


@dataclass
class RiskState:
    """Current risk management state."""
    current_bankroll: float
    initial_bankroll: float
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    current_drawdown: float
    max_drawdown: float
    peak_bankroll: float
    consecutive_wins: int
    consecutive_losses: int
    open_positions: int
    total_exposure: float
    kelly_multiplier: float  # Dynamic Kelly fraction
    is_circuit_breaker_active: bool


@dataclass
class TradeOutcome:
    """Outcome of a completed trade for learning."""
    market_id: str
    edge_at_entry: float
    predicted_prob: float
    actual_outcome: bool  # True = won
    pnl: float
    hold_time_hours: float


class RiskEngine:
    """
    Intelligent risk management and position sizing.
    Uses Kelly Criterion with Bayesian learning to improve over time.
    """

    def __init__(self):
        self.state: Optional[RiskState] = None
        self.trade_outcomes: deque = deque(maxlen=1000)
        self.edge_calibration: dict[str, list[float]] = {}  # signal_type -> historical accuracy
        self._running: bool = False

    async def start(self):
        """Initialize risk engine with state from database."""
        await self._load_state()
        self._running = True
        logger.info(
            "risk_engine_started",
            bankroll=self.state.current_bankroll,
            win_rate=f"{self.state.win_rate:.2%}",
        )

    async def stop(self):
        """Persist state and shutdown."""
        await self._save_state()
        self._running = False
        logger.info("risk_engine_stopped")

    # =========================================================================
    # KELLY CRITERION
    # =========================================================================

    def kelly_criterion(self, probability: float, odds: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Kelly Formula: f* = (bp - q) / b
        Where:
            f* = fraction of bankroll to bet
            b = net odds (payout ratio - 1)
            p = probability of winning
            q = probability of losing (1 - p)
            
        For binary markets at price P:
            b = (1 - P) / P  (buying YES at price P, win 1-P, lose P)
            f* = (p * b - q) / b = p - q/b = p - (1-p)*P/(1-P)
            
        Simplified: f* = (p - market_price) / (1 - market_price)
        """
        if probability <= 0 or probability >= 1:
            return 0.0
        if odds <= 0:
            return 0.0

        q = 1 - probability
        kelly = (probability * odds - q) / odds

        # Never bet negative (no edge)
        return max(0.0, kelly)

    def fractional_kelly(self, full_kelly: float, fraction: Optional[float] = None) -> float:
        """
        Apply fractional Kelly for safety.
        Full Kelly is optimal but volatile; fractional Kelly reduces variance.
        
        Default fraction from config, adjusted dynamically based on:
        - Recent performance (increase after wins, decrease after losses)
        - Drawdown level (reduce as drawdown increases)
        - Confidence in edge estimate
        """
        if fraction is None:
            fraction = self.state.kelly_multiplier if self.state else settings.risk.kelly_fraction

        adjusted = full_kelly * fraction

        # Cap at max position size
        max_size = settings.risk.max_position_size_pct
        return min(adjusted, max_size)

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        market_id: str,
        market_price: float,
        true_probability: float,
        confidence: float = 0.7,
        signal_type: str = "combined"
    ) -> Optional[PositionSize]:
        """
        Calculate optimal position size for a trade.
        
        Combines Kelly Criterion with:
        - Confidence-adjusted probability
        - Dynamic Kelly fraction
        - Bankroll and drawdown constraints
        - Correlation-adjusted exposure
        """
        if not self.state:
            return None

        # Check circuit breaker
        if self.state.is_circuit_breaker_active:
            logger.warning("circuit_breaker_active", market_id=market_id)
            return None

        # Check max positions
        if self.state.open_positions >= settings.trading.max_concurrent_positions:
            logger.info("max_positions_reached", market_id=market_id)
            return None

        # Confidence-adjusted probability
        # Shrink probability toward 0.5 based on confidence
        adj_probability = 0.5 + (true_probability - 0.5) * confidence

        # Calculate edge
        edge = adj_probability - market_price
        if edge < settings.risk.min_edge_threshold:
            return None

        # Kelly calculation
        odds = (1 - market_price) / market_price if market_price > 0 else 0
        if odds <= 0:
            return None

        full_kelly = self.kelly_criterion(adj_probability, odds)
        if full_kelly <= 0:
            return None

        # Dynamic Kelly fraction
        kelly_fraction = self._dynamic_kelly_fraction(confidence, signal_type)
        adjusted_kelly = self.fractional_kelly(full_kelly, kelly_fraction)

        # Calculate USD size
        size_usd = self.state.current_bankroll * adjusted_kelly

        # Minimum trade size ($0.50)
        if size_usd < 0.50:
            return None

        # Maximum loss for this trade
        max_loss = size_usd * market_price  # Lose the cost if market goes to 0

        # Risk/reward ratio
        potential_win = size_usd * (1 - market_price)
        rr_ratio = potential_win / max_loss if max_loss > 0 else 0

        return PositionSize(
            market_id=market_id,
            size_usd=round(size_usd, 2),
            size_pct_bankroll=adjusted_kelly,
            kelly_full=full_kelly,
            kelly_adjusted=adjusted_kelly,
            edge=edge,
            probability=adj_probability,
            max_loss=round(max_loss, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            reasoning=(
                f"Kelly={full_kelly:.3f} Adj={adjusted_kelly:.3f} "
                f"Edge={edge:.3f} Conf={confidence:.2f} "
                f"Bankroll=${self.state.current_bankroll:.2f}"
            ),
        )

    def _dynamic_kelly_fraction(self, confidence: float, signal_type: str) -> float:
        """
        Dynamically adjust Kelly fraction based on:
        1. Recent performance (win/loss streaks)
        2. Current drawdown
        3. Signal confidence
        4. Historical accuracy of this signal type
        """
        base_fraction = settings.risk.kelly_fraction  # 0.25 default

        # Adjust for win/loss streaks
        if self.state.consecutive_wins >= 3:
            # Increase slightly on hot streak (compound faster)
            streak_adj = min(self.state.consecutive_wins * 0.02, 0.10)
            base_fraction += streak_adj
        elif self.state.consecutive_losses >= 2:
            # Decrease on cold streak (protect capital)
            streak_adj = min(self.state.consecutive_losses * 0.05, 0.15)
            base_fraction -= streak_adj

        # Adjust for drawdown
        if self.state.current_drawdown > 0.15:
            # Significant drawdown → reduce size aggressively
            dd_adj = (self.state.current_drawdown - 0.15) * 2
            base_fraction -= dd_adj
        elif self.state.current_drawdown < 0.05:
            # Low drawdown → slightly more aggressive
            base_fraction += 0.03

        # Adjust for confidence
        conf_multiplier = 0.5 + confidence * 0.5  # Range: 0.5x to 1.0x
        base_fraction *= conf_multiplier

        # Historical accuracy of signal type
        if signal_type in self.edge_calibration:
            history = self.edge_calibration[signal_type]
            if len(history) >= 10:
                accuracy = np.mean(history[-50:])
                if accuracy > 0.7:
                    base_fraction *= 1.1  # Reward accurate signals
                elif accuracy < 0.5:
                    base_fraction *= 0.7  # Penalize poor signals

        # Bounds
        return max(0.05, min(base_fraction, 0.40))

    # =========================================================================
    # DRAWDOWN PROTECTION
    # =========================================================================

    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should be activated.
        Triggers:
        - Drawdown exceeds max threshold
        - 5+ consecutive losses
        - Bankroll < 20% of peak
        """
        if not self.state:
            return False

        triggers = []

        # Max drawdown trigger
        if self.state.current_drawdown >= settings.risk.max_drawdown_pct:
            triggers.append(f"drawdown={self.state.current_drawdown:.1%}")

        # Consecutive losses
        if self.state.consecutive_losses >= 5:
            triggers.append(f"consec_losses={self.state.consecutive_losses}")

        # Bankroll collapse
        if self.state.current_bankroll < self.state.peak_bankroll * 0.2:
            triggers.append("bankroll_collapse")

        if triggers:
            self.state.is_circuit_breaker_active = True
            logger.warning("circuit_breaker_triggered", triggers=triggers)
            return True

        # Deactivate if conditions improve
        if self.state.is_circuit_breaker_active:
            if (self.state.current_drawdown < settings.risk.max_drawdown_pct * 0.5
                    and self.state.consecutive_losses < 3):
                self.state.is_circuit_breaker_active = False
                logger.info("circuit_breaker_deactivated")

        return False

    # =========================================================================
    # LEARNING & CALIBRATION
    # =========================================================================

    def record_outcome(self, outcome: TradeOutcome):
        """
        Record trade outcome for Bayesian learning.
        Updates edge calibration and performance stats.
        """
        self.trade_outcomes.append(outcome)

        # Update win/loss state
        if outcome.actual_outcome:
            self.state.win_count += 1
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.loss_count += 1
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Update bankroll
        self.state.current_bankroll += outcome.pnl
        self.state.total_pnl += outcome.pnl

        # Update peak and drawdown
        if self.state.current_bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = self.state.current_bankroll
        self.state.current_drawdown = (
            (self.state.peak_bankroll - self.state.current_bankroll) / self.state.peak_bankroll
            if self.state.peak_bankroll > 0 else 0
        )

        # Update win rate
        total = self.state.win_count + self.state.loss_count
        self.state.win_rate = self.state.win_count / total if total > 0 else 0

        # Check circuit breaker
        self.check_circuit_breaker()

        # Update Kelly multiplier based on recent performance
        self._update_kelly_multiplier()

        logger.info(
            "outcome_recorded",
            won=outcome.actual_outcome,
            pnl=f"${outcome.pnl:.2f}",
            bankroll=f"${self.state.current_bankroll:.2f}",
            winrate=f"{self.state.win_rate:.1%}",
            drawdown=f"{self.state.current_drawdown:.1%}",
        )

    def _update_kelly_multiplier(self):
        """
        Bayesian update of Kelly multiplier.
        More data → more confidence → larger Kelly fraction.
        """
        total_trades = self.state.win_count + self.state.loss_count

        if total_trades < 10:
            # Conservative early on
            self.state.kelly_multiplier = 0.15
        elif total_trades < 30:
            self.state.kelly_multiplier = 0.20
        elif total_trades < 100:
            # Growing confidence
            if self.state.win_rate > 0.70:
                self.state.kelly_multiplier = 0.30
            elif self.state.win_rate > 0.60:
                self.state.kelly_multiplier = 0.25
            else:
                self.state.kelly_multiplier = 0.15
        else:
            # Established track record
            if self.state.win_rate > 0.75:
                self.state.kelly_multiplier = 0.35
            elif self.state.win_rate > 0.65:
                self.state.kelly_multiplier = 0.30
            elif self.state.win_rate > 0.55:
                self.state.kelly_multiplier = 0.20
            else:
                self.state.kelly_multiplier = 0.10  # Poor performance, reduce

    def recalibrate_edge(self, signal_type: str, predicted_edge: float, actual_won: bool):
        """
        Recalibrate edge estimates for each signal type.
        Tracks accuracy to weight future signals appropriately.
        """
        if signal_type not in self.edge_calibration:
            self.edge_calibration[signal_type] = []

        # 1.0 if prediction was correct, 0.0 if not
        self.edge_calibration[signal_type].append(1.0 if actual_won else 0.0)

        # Keep last 200 outcomes per signal type
        if len(self.edge_calibration[signal_type]) > 200:
            self.edge_calibration[signal_type] = self.edge_calibration[signal_type][-200:]

    def get_calibration_stats(self) -> dict:
        """Get calibration statistics for all signal types."""
        stats = {}
        for sig_type, outcomes in self.edge_calibration.items():
            if len(outcomes) >= 5:
                stats[sig_type] = {
                    "accuracy": np.mean(outcomes),
                    "sample_size": len(outcomes),
                    "recent_accuracy": np.mean(outcomes[-20:]) if len(outcomes) >= 20 else np.mean(outcomes),
                }
        return stats

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    async def _load_state(self):
        """Load risk state from database."""
        async with async_session() as session:
            result = await session.get(BotState, "risk_state")

            if result and result.value:
                data = result.value
                self.state = RiskState(
                    current_bankroll=data.get("current_bankroll", settings.risk.initial_bankroll),
                    initial_bankroll=settings.risk.initial_bankroll,
                    total_pnl=data.get("total_pnl", 0.0),
                    win_count=data.get("win_count", 0),
                    loss_count=data.get("loss_count", 0),
                    win_rate=data.get("win_rate", 0.0),
                    current_drawdown=data.get("current_drawdown", 0.0),
                    max_drawdown=data.get("max_drawdown", 0.0),
                    peak_bankroll=data.get("peak_bankroll", settings.risk.initial_bankroll),
                    consecutive_wins=data.get("consecutive_wins", 0),
                    consecutive_losses=data.get("consecutive_losses", 0),
                    open_positions=data.get("open_positions", 0),
                    total_exposure=data.get("total_exposure", 0.0),
                    kelly_multiplier=data.get("kelly_multiplier", settings.risk.kelly_fraction),
                    is_circuit_breaker_active=data.get("is_circuit_breaker_active", False),
                )

                # Load edge calibration
                self.edge_calibration = data.get("edge_calibration", {})
            else:
                # Fresh start
                self.state = RiskState(
                    current_bankroll=settings.risk.initial_bankroll,
                    initial_bankroll=settings.risk.initial_bankroll,
                    total_pnl=0.0,
                    win_count=0,
                    loss_count=0,
                    win_rate=0.0,
                    current_drawdown=0.0,
                    max_drawdown=0.0,
                    peak_bankroll=settings.risk.initial_bankroll,
                    consecutive_wins=0,
                    consecutive_losses=0,
                    open_positions=0,
                    total_exposure=0.0,
                    kelly_multiplier=settings.risk.kelly_fraction,
                    is_circuit_breaker_active=False,
                )

    async def _save_state(self):
        """Persist risk state to database."""
        if not self.state:
            return

        async with async_session() as session:
            data = {
                "current_bankroll": self.state.current_bankroll,
                "total_pnl": self.state.total_pnl,
                "win_count": self.state.win_count,
                "loss_count": self.state.loss_count,
                "win_rate": self.state.win_rate,
                "current_drawdown": self.state.current_drawdown,
                "max_drawdown": self.state.max_drawdown,
                "peak_bankroll": self.state.peak_bankroll,
                "consecutive_wins": self.state.consecutive_wins,
                "consecutive_losses": self.state.consecutive_losses,
                "open_positions": self.state.open_positions,
                "total_exposure": self.state.total_exposure,
                "kelly_multiplier": self.state.kelly_multiplier,
                "is_circuit_breaker_active": self.state.is_circuit_breaker_active,
                "edge_calibration": self.edge_calibration,
                "last_updated": datetime.utcnow().isoformat(),
            }

            existing = await session.get(BotState, "risk_state")
            if existing:
                existing.value = data
            else:
                session.add(BotState(key="risk_state", value=data))

            await session.commit()


# Singleton
risk_engine = RiskEngine()
