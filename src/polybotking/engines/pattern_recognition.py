"""
Pattern Recognition Engine
===========================
Learns from every trade outcome. Identifies winning patterns and avoids losing ones.
Gets smarter over time — the more it trades, the better it predicts.

Core capabilities:
- Track features of every trade (category, time, sentiment, direction, edge)
- Cluster winning trades vs losing trades
- Score new opportunities based on similarity to past winners
- Auto-adjust strategy based on what ACTUALLY works
- Remember: "crypto + bullish sentiment + whale entry + low vol = 85% win"
"""

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from sqlalchemy import select

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Trade, TradeStatus, BotState, async_session

logger = get_logger("pattern_recognition")


@dataclass
class TradePattern:
    """Feature set of a trade for pattern learning."""
    # Market features
    category: str = ""
    market_price: float = 0.5
    volume_level: str = ""  # "low", "medium", "high"
    spread_level: str = ""  # "tight", "medium", "wide"

    # Signal features
    direction: str = "YES"
    edge: float = 0.0
    confidence: float = 0.0
    num_engines_agree: int = 0

    # Context features
    sentiment_score: float = 0.0
    has_whale_activity: bool = False
    volatility_regime: str = ""  # "low", "medium", "high"
    hour_of_day: int = 0
    day_of_week: int = 0

    # Outcome
    won: bool = False
    pnl_pct: float = 0.0


@dataclass
class PatternScore:
    """Score for a new opportunity based on pattern matching."""
    market_id: str
    pattern_confidence: float  # 0-1, how confident based on past patterns
    similar_wins: int  # number of similar past trades that won
    similar_losses: int  # number of similar past trades that lost
    estimated_win_rate: float  # predicted win rate for this pattern
    top_features: list[str] = field(default_factory=list)  # why this pattern is good/bad
    recommendation: str = "neutral"  # "strong_buy", "buy", "neutral", "avoid"


class PatternRecognition:
    """
    Learns winning patterns from trade history.
    Gets smarter with every trade outcome.
    """

    def __init__(self):
        self._running: bool = False
        self.trade_patterns: list[TradePattern] = []
        self.pattern_stats: dict[str, dict] = {}  # pattern_key -> {wins, losses, avg_pnl}
        self.min_samples: int = 5  # Minimum trades before pattern is trusted

    async def start(self):
        """Initialize pattern recognition from historical trades."""
        await self._load_patterns_from_db()
        self._running = True
        logger.info("pattern_recognition_started", patterns=len(self.trade_patterns))

    async def stop(self):
        """Save patterns and shutdown."""
        await self._save_patterns_to_db()
        self._running = False
        logger.info("pattern_recognition_stopped")

    # =========================================================================
    # PATTERN EXTRACTION
    # =========================================================================

    def extract_pattern(
        self,
        category: str = "",
        market_price: float = 0.5,
        volume: float = 0,
        spread: float = 0,
        direction: str = "YES",
        edge: float = 0,
        confidence: float = 0,
        num_engines_agree: int = 0,
        sentiment_score: float = 0,
        has_whale: bool = False,
        vol_regime: str = "",
    ) -> TradePattern:
        """Extract a pattern feature set from trade context."""
        # Categorize volume
        if volume > 100000:
            vol_level = "high"
        elif volume > 10000:
            vol_level = "medium"
        else:
            vol_level = "low"

        # Categorize spread
        if spread < 0.03:
            spread_level = "tight"
        elif spread < 0.06:
            spread_level = "medium"
        else:
            spread_level = "wide"

        now = datetime.utcnow()

        return TradePattern(
            category=category,
            market_price=round(market_price, 1),
            volume_level=vol_level,
            spread_level=spread_level,
            direction=direction,
            edge=round(edge, 2),
            confidence=round(confidence, 1),
            num_engines_agree=num_engines_agree,
            sentiment_score=round(sentiment_score, 1),
            has_whale_activity=has_whale,
            volatility_regime=vol_regime,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
        )

    def _pattern_key(self, pattern: TradePattern) -> str:
        """Generate a unique key for pattern grouping."""
        return (
            f"{pattern.category}|{pattern.volume_level}|{pattern.spread_level}|"
            f"{pattern.direction}|{pattern.num_engines_agree}|"
            f"{pattern.has_whale_activity}|{pattern.volatility_regime}"
        )

    # =========================================================================
    # PATTERN LEARNING
    # =========================================================================

    def record_outcome(self, pattern: TradePattern, won: bool, pnl_pct: float):
        """Record trade outcome for pattern learning."""
        pattern.won = won
        pattern.pnl_pct = pnl_pct
        self.trade_patterns.append(pattern)

        # Update pattern stats
        key = self._pattern_key(pattern)
        if key not in self.pattern_stats:
            self.pattern_stats[key] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}

        stats = self.pattern_stats[key]
        stats["count"] += 1
        stats["total_pnl"] += pnl_pct
        if won:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        win_rate = stats["wins"] / stats["count"] if stats["count"] > 0 else 0

        logger.info(
            "pattern_recorded",
            key=key[:30],
            won=won,
            pattern_winrate=f"{win_rate:.1%}",
            sample_size=stats["count"],
        )

    # =========================================================================
    # PATTERN SCORING
    # =========================================================================

    def score_opportunity(self, pattern: TradePattern) -> PatternScore:
        """
        Score a new opportunity based on similarity to past winning patterns.
        Returns recommendation and confidence.
        """
        key = self._pattern_key(pattern)
        market_id = ""  # Set by caller

        # Exact pattern match
        if key in self.pattern_stats:
            stats = self.pattern_stats[key]
            if stats["count"] >= self.min_samples:
                win_rate = stats["wins"] / stats["count"]
                avg_pnl = stats["total_pnl"] / stats["count"]

                # Determine recommendation
                if win_rate >= 0.80:
                    recommendation = "strong_buy"
                    confidence = 0.90
                elif win_rate >= 0.70:
                    recommendation = "buy"
                    confidence = 0.75
                elif win_rate >= 0.55:
                    recommendation = "neutral"
                    confidence = 0.50
                else:
                    recommendation = "avoid"
                    confidence = 0.30

                return PatternScore(
                    market_id=market_id,
                    pattern_confidence=confidence,
                    similar_wins=stats["wins"],
                    similar_losses=stats["losses"],
                    estimated_win_rate=win_rate,
                    top_features=[f"Exact pattern match: {stats['count']} samples, {win_rate:.0%} WR"],
                    recommendation=recommendation,
                )

        # Fuzzy match - find similar patterns
        similar_wins = 0
        similar_losses = 0

        for past in self.trade_patterns[-200:]:  # Last 200 trades
            similarity = self._calculate_similarity(pattern, past)
            if similarity >= 0.7:  # 70%+ similar
                if past.won:
                    similar_wins += 1
                else:
                    similar_losses += 1

        total_similar = similar_wins + similar_losses
        if total_similar >= 3:
            win_rate = similar_wins / total_similar
            if win_rate >= 0.75:
                recommendation = "buy"
                confidence = min(win_rate, 0.85)
            elif win_rate < 0.45:
                recommendation = "avoid"
                confidence = 0.30
            else:
                recommendation = "neutral"
                confidence = 0.50
        else:
            # Not enough data
            win_rate = 0.5
            recommendation = "neutral"
            confidence = 0.40

        return PatternScore(
            market_id=market_id,
            pattern_confidence=confidence,
            similar_wins=similar_wins,
            similar_losses=similar_losses,
            estimated_win_rate=win_rate,
            top_features=self._get_top_features(pattern),
            recommendation=recommendation,
        )

    def _calculate_similarity(self, a: TradePattern, b: TradePattern) -> float:
        """Calculate similarity between two patterns (0-1)."""
        score = 0.0
        total_weight = 0.0

        # Category match (weight: 2)
        if a.category and b.category:
            if a.category == b.category:
                score += 2.0
            total_weight += 2.0

        # Volume level (weight: 1)
        if a.volume_level == b.volume_level:
            score += 1.0
        total_weight += 1.0

        # Direction (weight: 1.5)
        if a.direction == b.direction:
            score += 1.5
        total_weight += 1.5

        # Engines agree (weight: 2)
        if abs(a.num_engines_agree - b.num_engines_agree) <= 1:
            score += 2.0
        total_weight += 2.0

        # Whale activity (weight: 1.5)
        if a.has_whale_activity == b.has_whale_activity:
            score += 1.5
        total_weight += 1.5

        # Volatility regime (weight: 1)
        if a.volatility_regime == b.volatility_regime:
            score += 1.0
        total_weight += 1.0

        # Edge similarity (weight: 1)
        if abs(a.edge - b.edge) < 0.03:
            score += 1.0
        total_weight += 1.0

        return score / total_weight if total_weight > 0 else 0.0

    def _get_top_features(self, pattern: TradePattern) -> list[str]:
        """Get human-readable top features of a pattern."""
        features = []
        if pattern.has_whale_activity:
            features.append("whale_active")
        if pattern.num_engines_agree >= 4:
            features.append("strong_consensus")
        if pattern.volatility_regime == "low":
            features.append("low_vol_entry")
        if pattern.volume_level == "high":
            features.append("high_volume")
        if pattern.edge >= 0.08:
            features.append("high_edge")
        return features

    # =========================================================================
    # BEST PATTERNS (for reporting)
    # =========================================================================

    def get_best_patterns(self, top_n: int = 5) -> list[dict]:
        """Get the most profitable patterns discovered."""
        ranked = []
        for key, stats in self.pattern_stats.items():
            if stats["count"] >= self.min_samples:
                win_rate = stats["wins"] / stats["count"]
                ranked.append({
                    "pattern": key,
                    "win_rate": win_rate,
                    "sample_size": stats["count"],
                    "avg_pnl": stats["total_pnl"] / stats["count"],
                })

        ranked.sort(key=lambda x: x["win_rate"], reverse=True)
        return ranked[:top_n]

    def get_worst_patterns(self, top_n: int = 5) -> list[dict]:
        """Get patterns to AVOID."""
        ranked = []
        for key, stats in self.pattern_stats.items():
            if stats["count"] >= self.min_samples:
                win_rate = stats["wins"] / stats["count"]
                ranked.append({
                    "pattern": key,
                    "win_rate": win_rate,
                    "sample_size": stats["count"],
                    "avg_pnl": stats["total_pnl"] / stats["count"],
                })

        ranked.sort(key=lambda x: x["win_rate"])
        return ranked[:top_n]

    # =========================================================================
    # DATABASE PERSISTENCE
    # =========================================================================

    async def _load_patterns_from_db(self):
        """Load pattern stats from database."""
        async with async_session() as session:
            result = await session.get(BotState, "pattern_stats")
            if result and result.value:
                self.pattern_stats = result.value.get("stats", {})
                logger.info("patterns_loaded", patterns=len(self.pattern_stats))

    async def _save_patterns_to_db(self):
        """Save pattern stats to database."""
        async with async_session() as session:
            data = {
                "stats": self.pattern_stats,
                "last_updated": datetime.utcnow().isoformat(),
                "total_patterns": len(self.trade_patterns),
            }
            existing = await session.get(BotState, "pattern_stats")
            if existing:
                existing.value = data
            else:
                session.add(BotState(key="pattern_stats", value=data))
            await session.commit()


# Singleton
pattern_recognition = PatternRecognition()
