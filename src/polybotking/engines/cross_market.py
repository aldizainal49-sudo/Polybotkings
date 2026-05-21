"""
Cross-Market Correlation Engine
================================
Detects related markets and trades correlated opportunities.
When one market moves, related markets often follow.

Core capabilities:
- Detect market pairs with shared keywords/topics
- Track correlation between related market prices
- Generate signals when correlated market hasn't adjusted yet
- Category-based grouping (crypto, politics, sports, etc.)
- Cascade detection (one event triggers multiple market moves)
"""

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from polybotking.config import settings
from polybotking.logger import get_logger

logger = get_logger("cross_market")


@dataclass
class MarketCorrelation:
    """Correlation between two markets."""
    market_a_id: str
    market_b_id: str
    correlation: float  # -1 to 1
    shared_keywords: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class CrossMarketSignal:
    """Signal from cross-market analysis."""
    market_id: str
    direction: str  # YES/NO
    trigger_market_id: str  # Market that moved first
    trigger_move: float  # How much trigger market moved
    expected_move: float  # Expected move in target market
    confidence: float
    reasoning: str


class CrossMarketEngine:
    """
    Detects correlated markets and generates signals
    when one market moves but related ones haven't adjusted.
    """

    def __init__(self):
        self._running: bool = False
        self.market_groups: dict[str, list[str]] = {}  # category -> market_ids
        self.keyword_index: dict[str, list[str]] = {}  # keyword -> market_ids
        self.price_history: dict[str, list[float]] = {}  # market_id -> prices
        self.correlations: list[MarketCorrelation] = []

    async def start(self):
        """Start cross-market engine."""
        self._running = True
        logger.info("cross_market_started")

    async def stop(self):
        """Stop."""
        self._running = False
        logger.info("cross_market_stopped")

    # =========================================================================
    # MARKET GROUPING
    # =========================================================================

    def index_market(self, market_id: str, question: str, category: str = ""):
        """Index a market by keywords and category for correlation detection."""
        import re

        # Extract keywords
        stop_words = {"will", "the", "be", "is", "are", "by", "in", "on", "at",
                      "to", "for", "of", "a", "an", "this", "that", "before", "after"}
        words = re.findall(r'[\w]+', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Index by keyword
        for kw in keywords:
            if kw not in self.keyword_index:
                self.keyword_index[kw] = []
            if market_id not in self.keyword_index[kw]:
                self.keyword_index[kw].append(market_id)

        # Index by category
        if category:
            if category not in self.market_groups:
                self.market_groups[category] = []
            if market_id not in self.market_groups[category]:
                self.market_groups[category].append(market_id)

    def find_related_markets(self, market_id: str, question: str) -> list[str]:
        """Find markets related to a given market by shared keywords."""
        import re

        words = re.findall(r'[\w]+', question.lower())
        keywords = [w for w in words if len(w) > 3]

        related = defaultdict(int)
        for kw in keywords:
            for mid in self.keyword_index.get(kw, []):
                if mid != market_id:
                    related[mid] += 1

        # Sort by number of shared keywords
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return [mid for mid, count in sorted_related[:10] if count >= 2]

    # =========================================================================
    # PRICE TRACKING
    # =========================================================================

    def record_price(self, market_id: str, price: float):
        """Record price for correlation calculation."""
        if market_id not in self.price_history:
            self.price_history[market_id] = []
        self.price_history[market_id].append(price)
        # Keep last 100 prices
        if len(self.price_history[market_id]) > 100:
            self.price_history[market_id] = self.price_history[market_id][-100:]

    # =========================================================================
    # CORRELATION DETECTION
    # =========================================================================

    def calculate_correlation(self, market_a_id: str, market_b_id: str) -> float:
        """Calculate price correlation between two markets."""
        prices_a = self.price_history.get(market_a_id, [])
        prices_b = self.price_history.get(market_b_id, [])

        if len(prices_a) < 10 or len(prices_b) < 10:
            return 0.0

        # Align lengths
        min_len = min(len(prices_a), len(prices_b))
        a = np.array(prices_a[-min_len:])
        b = np.array(prices_b[-min_len:])

        # Pearson correlation
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        correlation = np.corrcoef(a, b)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    # =========================================================================
    # CROSS-MARKET SIGNALS
    # =========================================================================

    def detect_cascade_opportunity(
        self,
        trigger_market_id: str,
        trigger_price_change: float,
        trigger_direction: str,
    ) -> list[CrossMarketSignal]:
        """
        Detect cascade opportunities when one market moves.
        Related markets that haven't adjusted yet = opportunity.
        """
        signals = []

        # Find related markets
        related_markets = []
        for kw, markets in self.keyword_index.items():
            if trigger_market_id in markets:
                for mid in markets:
                    if mid != trigger_market_id and mid not in [s.market_id for s in signals]:
                        related_markets.append(mid)

        for target_id in related_markets[:5]:  # Top 5 related
            # Check if correlated
            correlation = self.calculate_correlation(trigger_market_id, target_id)

            if abs(correlation) < 0.3:
                continue  # Not correlated enough

            # Check if target has moved
            target_prices = self.price_history.get(target_id, [])
            if len(target_prices) < 2:
                continue

            recent_move = target_prices[-1] - target_prices[-5] if len(target_prices) >= 5 else 0
            expected_move = trigger_price_change * correlation

            # If target hasn't moved as much as expected → opportunity
            gap = abs(expected_move) - abs(recent_move)
            if gap < 0.03:
                continue  # Already adjusted

            # Determine direction
            if correlation > 0:
                direction = trigger_direction  # Same direction as trigger
            else:
                direction = "NO" if trigger_direction == "YES" else "YES"  # Opposite

            confidence = min(abs(correlation) * abs(gap) * 5, 0.85)

            if confidence > 0.4:
                signals.append(CrossMarketSignal(
                    market_id=target_id,
                    direction=direction,
                    trigger_market_id=trigger_market_id,
                    trigger_move=trigger_price_change,
                    expected_move=expected_move,
                    confidence=confidence,
                    reasoning=f"Correlated market (r={correlation:.2f}) hasn't adjusted. Gap={gap:.3f}",
                ))

        if signals:
            logger.info("cascade_detected", trigger=trigger_market_id[:12],
                       signals=len(signals))

        return signals

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    async def analyze_opportunities(
        self, markets: list[dict], price_changes: dict[str, float]
    ) -> list[CrossMarketSignal]:
        """
        Analyze cross-market opportunities from recent price changes.
        """
        all_signals = []

        # Index all markets
        for market in markets:
            mid = market.get("id", "")
            question = market.get("question", "")
            category = market.get("category", "")
            if mid and question:
                self.index_market(mid, question, category)

            # Record price
            prices_str = market.get("outcomePrices", "[0.5,0.5]")
            try:
                prices = [float(p) for p in prices_str.strip("[]").split(",")]
                self.record_price(mid, prices[0])
            except (ValueError, IndexError):
                pass

        # Check each significant price change for cascade
        for market_id, change in price_changes.items():
            if abs(change) >= 0.05:  # >5% move = potential cascade trigger
                direction = "YES" if change > 0 else "NO"
                signals = self.detect_cascade_opportunity(market_id, change, direction)
                all_signals.extend(signals)

        return all_signals


# Singleton
cross_market = CrossMarketEngine()
