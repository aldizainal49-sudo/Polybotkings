"""
Volatility Timing Engine
=========================
Enters positions BEFORE volatility spikes, exits DURING moves.
Uses regime detection, historical patterns, and event-driven timing.

Core capabilities:
- Volatility regime detection (low/medium/high/extreme)
- Entry timing optimization (before catalysts)
- Exit timing (during momentum, before mean reversion)
- Event calendar awareness (earnings, elections, court dates)
- Price velocity & acceleration tracking
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import async_session

logger = get_logger("volatility_timing")


@dataclass
class PricePoint:
    """Single price observation."""
    timestamp: datetime
    price: float
    volume: float = 0.0


@dataclass
class VolatilityRegime:
    """Current volatility regime classification."""
    regime: str  # "low", "medium", "high", "extreme"
    realized_vol: float  # Annualized realized volatility
    vol_percentile: float  # Where current vol sits in history (0-100)
    trend: str  # "increasing", "stable", "decreasing"
    regime_duration_hours: float  # How long in current regime


@dataclass
class TimingSignal:
    """Entry/exit timing signal."""
    market_id: str
    action: str  # "ENTER", "EXIT", "HOLD"
    urgency: str  # "immediate", "soon", "wait"
    direction: str  # YES/NO (for entries)
    confidence: float
    reasoning: str
    estimated_move_pct: float = 0.0
    time_to_catalyst_hours: float = 0.0


class VolatilityTimer:
    """
    Volatility-based timing engine.
    Identifies optimal entry/exit windows based on vol regime shifts.
    """

    def __init__(self):
        self.price_histories: dict[str, deque] = {}  # market_id -> price history
        self.vol_regimes: dict[str, VolatilityRegime] = {}
        self.max_history_points: int = 500
        self._running: bool = False

    async def start(self):
        """Initialize the timing engine."""
        self._running = True
        logger.info("volatility_timer_started")

    async def stop(self):
        """Shutdown."""
        self._running = False
        logger.info("volatility_timer_stopped")

    # =========================================================================
    # PRICE TRACKING
    # =========================================================================

    def record_price(self, market_id: str, price: float, volume: float = 0.0):
        """Record a new price observation."""
        if market_id not in self.price_histories:
            self.price_histories[market_id] = deque(maxlen=self.max_history_points)

        self.price_histories[market_id].append(PricePoint(
            timestamp=datetime.utcnow(),
            price=price,
            volume=volume,
        ))

    def get_price_series(self, market_id: str) -> Optional[np.ndarray]:
        """Get price series as numpy array."""
        history = self.price_histories.get(market_id)
        if not history or len(history) < 10:
            return None
        return np.array([p.price for p in history])

    # =========================================================================
    # VOLATILITY CALCULATION
    # =========================================================================

    def calculate_realized_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """
        Calculate realized volatility from price series.
        Uses log returns standard deviation, annualized.
        """
        if len(prices) < window + 1:
            return 0.0

        # For binary markets (0-1), use absolute returns instead of log returns
        returns = np.diff(prices[-window-1:])
        vol = np.std(returns) * np.sqrt(365 * 24)  # Annualize (hourly data assumed)
        return float(vol)

    def calculate_vol_percentile(self, market_id: str, current_vol: float) -> float:
        """Where current vol sits relative to historical distribution."""
        history = self.price_histories.get(market_id)
        if not history or len(history) < 50:
            return 50.0  # Default to median

        prices = np.array([p.price for p in history])

        # Calculate rolling vol for history
        vols = []
        for i in range(20, len(prices)):
            window_prices = prices[i-20:i]
            returns = np.diff(window_prices)
            vol = np.std(returns) * np.sqrt(365 * 24)
            vols.append(vol)

        if not vols:
            return 50.0

        percentile = np.percentile(vols, np.searchsorted(np.sort(vols), current_vol) / len(vols) * 100)
        return float(min(max(percentile, 0), 100))

    def classify_regime(self, market_id: str) -> Optional[VolatilityRegime]:
        """
        Classify current volatility regime.
        Low → good for entry (calm before storm)
        High → good for exit (during the move)
        """
        prices = self.get_price_series(market_id)
        if prices is None:
            return None

        current_vol = self.calculate_realized_volatility(prices, window=20)
        short_vol = self.calculate_realized_volatility(prices, window=5)
        percentile = self.calculate_vol_percentile(market_id, current_vol)

        # Classify regime
        if percentile < 25:
            regime = "low"
        elif percentile < 50:
            regime = "medium"
        elif percentile < 80:
            regime = "high"
        else:
            regime = "extreme"

        # Determine trend
        if short_vol > current_vol * 1.3:
            trend = "increasing"
        elif short_vol < current_vol * 0.7:
            trend = "decreasing"
        else:
            trend = "stable"

        # Estimate regime duration
        prev_regime = self.vol_regimes.get(market_id)
        if prev_regime and prev_regime.regime == regime:
            duration = prev_regime.regime_duration_hours + (settings.trading.scan_interval_seconds / 3600)
        else:
            duration = 0.0

        vol_regime = VolatilityRegime(
            regime=regime,
            realized_vol=current_vol,
            vol_percentile=percentile,
            trend=trend,
            regime_duration_hours=duration,
        )

        self.vol_regimes[market_id] = vol_regime
        return vol_regime

    # =========================================================================
    # MOMENTUM & VELOCITY
    # =========================================================================

    def calculate_momentum(self, prices: np.ndarray) -> dict:
        """
        Calculate price momentum indicators.
        - Velocity: rate of price change
        - Acceleration: rate of velocity change
        - RSI-like: relative strength
        """
        if len(prices) < 15:
            return {"velocity": 0.0, "acceleration": 0.0, "rsi": 50.0}

        # Velocity (rate of change over last 5 periods)
        velocity = (prices[-1] - prices[-6]) / 5

        # Acceleration (change in velocity)
        prev_velocity = (prices[-6] - prices[-11]) / 5 if len(prices) >= 11 else 0
        acceleration = velocity - prev_velocity

        # RSI-like indicator for binary markets
        gains = []
        losses = []
        for i in range(1, min(15, len(prices))):
            change = prices[-i] - prices[-i-1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "rsi": float(rsi),
        }

    # =========================================================================
    # TIMING SIGNALS
    # =========================================================================

    def generate_timing_signal(
        self,
        market_id: str,
        current_price: float,
        direction_bias: str = "YES",
    ) -> Optional[TimingSignal]:
        """
        Generate entry/exit timing signal based on volatility regime
        and momentum analysis.
        
        Strategy:
        - ENTER when vol is LOW and momentum aligns with direction
        - EXIT when vol is HIGH/EXTREME and momentum is fading
        - HOLD when regime is transitioning
        """
        # Get regime
        regime = self.classify_regime(market_id)
        if regime is None:
            return None

        # Get momentum
        prices = self.get_price_series(market_id)
        if prices is None:
            return None

        momentum = self.calculate_momentum(prices)

        # Decision logic
        action = "HOLD"
        urgency = "wait"
        confidence = 0.0
        reasoning = ""

        # === ENTRY CONDITIONS ===
        if regime.regime == "low" and regime.trend == "stable":
            # Low vol, stable → accumulate before move
            if direction_bias == "YES" and momentum["velocity"] >= 0:
                action = "ENTER"
                urgency = "soon"
                confidence = 0.6 + (1 - regime.vol_percentile / 100) * 0.2
                reasoning = "Low vol regime, positive momentum - enter before spike"

            elif direction_bias == "NO" and momentum["velocity"] <= 0:
                action = "ENTER"
                urgency = "soon"
                confidence = 0.6 + (1 - regime.vol_percentile / 100) * 0.2
                reasoning = "Low vol regime, negative momentum - enter before spike"

        elif regime.regime == "low" and regime.trend == "increasing":
            # Vol starting to increase from low → immediate entry
            action = "ENTER"
            urgency = "immediate"
            confidence = 0.75
            reasoning = "Vol breakout from low regime - immediate entry"

        # === EXIT CONDITIONS ===
        elif regime.regime == "high" and regime.trend == "decreasing":
            # High vol fading → take profits
            action = "EXIT"
            urgency = "soon"
            confidence = 0.7
            reasoning = "High vol regime fading - lock profits"

        elif regime.regime == "extreme":
            # Extreme vol → exit immediately
            action = "EXIT"
            urgency = "immediate"
            confidence = 0.85
            reasoning = "Extreme volatility - exit to protect capital"

        # === MOMENTUM OVERRIDES ===
        if momentum["rsi"] > 80 and action != "EXIT":
            action = "EXIT"
            urgency = "soon"
            confidence = max(confidence, 0.7)
            reasoning = "Overbought RSI - exit"
        elif momentum["rsi"] < 20 and action != "ENTER":
            if direction_bias == "YES":
                action = "ENTER"
                urgency = "soon"
                confidence = max(confidence, 0.65)
                reasoning = "Oversold RSI - contrarian entry"

        # Must have minimum confidence
        if confidence < 0.5:
            return None

        # Estimate potential move size
        estimated_move = abs(momentum["velocity"]) * 10 + regime.realized_vol * 0.5

        return TimingSignal(
            market_id=market_id,
            action=action,
            urgency=urgency,
            direction=direction_bias,
            confidence=min(confidence, 0.95),
            reasoning=reasoning,
            estimated_move_pct=float(min(estimated_move, 0.5)),
        )

    # =========================================================================
    # BATCH TIMING ANALYSIS
    # =========================================================================

    async def analyze_markets(
        self,
        market_data: list[dict],
        direction_biases: dict[str, str] = None
    ) -> list[TimingSignal]:
        """
        Run timing analysis on multiple markets.
        Updates price histories and generates timing signals.
        """
        if direction_biases is None:
            direction_biases = {}

        signals = []

        for market in market_data:
            market_id = market.get("id", "")
            if not market_id:
                continue

            # Extract current price
            prices_str = market.get("outcomePrices", "[0.5,0.5]")
            try:
                prices = [float(p) for p in prices_str.strip("[]").split(",")]
                current_price = prices[0]
            except (ValueError, IndexError):
                current_price = 0.5

            volume = float(market.get("volume24hr", 0) or 0)

            # Record price point
            self.record_price(market_id, current_price, volume)

            # Generate timing signal
            direction = direction_biases.get(market_id, "YES")
            signal = self.generate_timing_signal(market_id, current_price, direction)

            if signal:
                signals.append(signal)

        logger.info("timing_analysis_complete", markets=len(market_data), signals=len(signals))
        return signals

    # =========================================================================
    # OPTIMAL ENTRY WINDOW
    # =========================================================================

    def find_optimal_entry_window(self, market_id: str) -> dict:
        """
        Estimate optimal entry window based on vol patterns.
        Returns timing recommendation.
        """
        regime = self.vol_regimes.get(market_id)
        prices = self.get_price_series(market_id)

        if regime is None or prices is None:
            return {"recommendation": "wait", "reason": "insufficient data"}

        momentum = self.calculate_momentum(prices)

        if regime.regime == "low" and regime.regime_duration_hours > 2:
            return {
                "recommendation": "enter_now",
                "reason": "Extended low-vol period, breakout imminent",
                "confidence": 0.7,
            }
        elif regime.regime == "low" and regime.trend == "increasing":
            return {
                "recommendation": "enter_immediately",
                "reason": "Vol expansion starting from low base",
                "confidence": 0.8,
            }
        elif regime.regime == "medium" and momentum["acceleration"] > 0:
            return {
                "recommendation": "enter_soon",
                "reason": "Accelerating momentum in medium vol",
                "confidence": 0.6,
            }
        else:
            return {
                "recommendation": "wait",
                "reason": f"Current regime: {regime.regime}, trend: {regime.trend}",
                "confidence": 0.3,
            }


# Singleton
volatility_timer = VolatilityTimer()
