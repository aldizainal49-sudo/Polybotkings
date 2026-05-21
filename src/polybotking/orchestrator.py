"""
Orchestrator - Multi-Agent Coordination Pipeline
=================================================
The brain of PolyBotKing. Coordinates all engines into a unified
decision-making pipeline.

Pipeline:
Event Ingestion → CLOB Snapshot → News Fetch → AI Inference → EV Gate → Sizing → Execution

Orchestration flow:
1. Market Scanner discovers opportunities (CLOB snapshots, mispricing)
2. Wallet Intelligence provides smart money signals
3. Sentiment AI analyzes news/social before market adjusts
4. Volatility Timer determines optimal entry/exit windows
5. Risk Engine sizes positions via Kelly Criterion
6. Execution Engine places orders on Polymarket CLOB
7. Outcomes fed back for Bayesian learning

The more it runs, the smarter it gets:
- Probability recalibration from outcomes
- Kelly fraction adjustment from win rate
- Signal type accuracy tracking
- Adaptive strategy weighting
"""

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Signal, SignalType, async_session
from polybotking.engines.market_scanner import market_scanner, MarketOpportunity
from polybotking.engines.wallet_intelligence import wallet_intelligence, SmartMoneySignal
from polybotking.engines.sentiment_ai import sentiment_ai, SentimentSignal
from polybotking.engines.volatility_timing import volatility_timer, TimingSignal
from polybotking.engines.risk_engine import risk_engine, PositionSize
from polybotking.engines.ai_reasoning import ai_reasoning, DualAISignal

logger = get_logger("orchestrator")


@dataclass
class TradingDecision:
    """Final trading decision after all engines vote."""
    market_id: str
    question: str
    direction: str  # YES/NO
    action: str  # "EXECUTE", "SKIP", "MONITOR"

    # Composite scores
    combined_edge: float
    combined_confidence: float
    final_ev: float
    true_probability: float

    # Position sizing
    position_size: Optional[PositionSize] = None

    # Engine signals
    market_signal: Optional[MarketOpportunity] = None
    wallet_signal: Optional[SmartMoneySignal] = None
    sentiment_signal: Optional[SentimentSignal] = None
    timing_signal: Optional[TimingSignal] = None
    ai_signal: Optional[DualAISignal] = None

    # Decision metadata
    reasoning: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pipeline_latency_ms: float = 0.0


@dataclass
class OrchestratorState:
    """Runtime state of the orchestrator."""
    cycles_completed: int = 0
    total_decisions: int = 0
    total_executions: int = 0
    total_skipped: int = 0
    last_cycle_time: Optional[datetime] = None
    avg_cycle_duration_s: float = 0.0
    is_running: bool = False


class Orchestrator:
    """
    Multi-agent orchestrator. Coordinates all engines and makes
    final trading decisions using weighted signal combination.
    """

    def __init__(self):
        self.state = OrchestratorState()
        self.execution_callback = None  # Set by execution engine
        self._running: bool = False

        # Signal weights (adjusted dynamically based on calibration)
        self.weights = {
            "market": 0.25,    # Mispricing/arbitrage detection
            "wallet": 0.25,    # Smart money flow
            "sentiment": 0.25, # News/social sentiment
            "timing": 0.25,    # Volatility timing
        }

    async def start(self, execution_callback=None):
        """Initialize all engines and start orchestration."""
        self.execution_callback = execution_callback

        # Start all engines concurrently
        await asyncio.gather(
            market_scanner.start(),
            wallet_intelligence.start(),
            sentiment_ai.start(),
            volatility_timer.start(),
            risk_engine.start(),
            ai_reasoning.start(),
        )

        self._running = True
        self.state.is_running = True
        logger.info("orchestrator_started", weights=self.weights)

    async def stop(self):
        """Gracefully shutdown all engines."""
        self._running = False
        self.state.is_running = False

        await asyncio.gather(
            market_scanner.stop(),
            wallet_intelligence.stop(),
            sentiment_ai.stop(),
            volatility_timer.stop(),
            risk_engine.stop(),
            ai_reasoning.stop(),
        )

        logger.info(
            "orchestrator_stopped",
            cycles=self.state.cycles_completed,
            executions=self.state.total_executions,
        )

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    async def run_pipeline_cycle(self) -> list[TradingDecision]:
        """
        Execute one full pipeline cycle:
        
        1. Event Ingestion → Fetch markets from CLOB
        2. CLOB Snapshot → Orderbook analysis
        3. News Fetch → Sentiment analysis
        4. AI Inference → Combine all signals
        5. EV Gate → Filter by minimum EV
        6. Sizing → Kelly Criterion position sizing
        7. Execution → Place orders
        """
        cycle_start = datetime.utcnow()
        decisions = []

        try:
            # ===== STAGE 1: Market Discovery =====
            logger.info("pipeline_stage_1", stage="market_discovery")
            market_opportunities = await market_scanner.run_scan_cycle()

            if not market_opportunities:
                logger.info("no_opportunities_found")
                return []

            # Get market IDs for subsequent analysis
            market_ids = [opp.market_id for opp in market_opportunities[:20]]  # Top 20

            # ===== STAGE 2 & 3: Parallel Signal Gathering =====
            logger.info("pipeline_stage_2_3", stage="signal_gathering", markets=len(market_ids))

            # Get market data for sentiment and timing
            market_data = [
                {"id": opp.market_id, "question": opp.question,
                 "outcomePrices": f"[{opp.market_price},{1-opp.market_price}]"}
                for opp in market_opportunities[:20]
            ]

            # Run all signal engines in parallel
            wallet_signals_task = wallet_intelligence.run_analysis_cycle(market_ids)
            sentiment_signals_task = sentiment_ai.analyze_markets(market_data)
            timing_signals_task = volatility_timer.analyze_markets(market_data)

            # Prepare market data for AI reasoning
            ai_market_data = [
                {
                    "market_id": opp.market_id,
                    "question": opp.question,
                    "yes_price": opp.market_price,
                    "no_price": 1 - opp.market_price,
                    "volume": 0,
                    "end_date": "",
                    "sentiment_context": "",
                    "wallet_context": "",
                }
                for opp in market_opportunities[:10]  # AI only for top 10 (rate limit)
            ]
            ai_signals_task = ai_reasoning.analyze_markets(ai_market_data)

            wallet_signals, sentiment_signals, timing_signals, ai_signals = await asyncio.gather(
                wallet_signals_task,
                sentiment_signals_task,
                timing_signals_task,
                ai_signals_task,
                return_exceptions=True,
            )

            # Handle exceptions gracefully
            if isinstance(wallet_signals, Exception):
                logger.error("wallet_engine_error", error=str(wallet_signals))
                wallet_signals = []
            if isinstance(sentiment_signals, Exception):
                logger.error("sentiment_engine_error", error=str(sentiment_signals))
                sentiment_signals = []
            if isinstance(timing_signals, Exception):
                logger.error("timing_engine_error", error=str(timing_signals))
                timing_signals = []
            if isinstance(ai_signals, Exception):
                logger.error("ai_engine_error", error=str(ai_signals))
                ai_signals = []

            # Index signals by market_id
            wallet_by_market = {s.market_id: s for s in wallet_signals}
            sentiment_by_market = {s.market_id: s for s in sentiment_signals}
            timing_by_market = {s.market_id: s for s in timing_signals}
            ai_by_market = {s.market_id: s for s in ai_signals}

            # ===== STAGE 4: AI Inference - Signal Combination =====
            logger.info("pipeline_stage_4", stage="signal_combination")

            for opp in market_opportunities[:20]:
                decision = await self._combine_signals(
                    opportunity=opp,
                    wallet_signal=wallet_by_market.get(opp.market_id),
                    sentiment_signal=sentiment_by_market.get(opp.market_id),
                    timing_signal=timing_by_market.get(opp.market_id),
                    ai_signal=ai_by_market.get(opp.market_id),
                )

                if decision:
                    decisions.append(decision)

            # ===== STAGE 5: EV Gate =====
            logger.info("pipeline_stage_5", stage="ev_gate", candidates=len(decisions))
            decisions = self._apply_ev_gate(decisions)

            # ===== STAGE 6: Position Sizing =====
            logger.info("pipeline_stage_6", stage="sizing", passing_ev=len(decisions))
            decisions = self._apply_sizing(decisions)

            # ===== STAGE 7: Execution =====
            logger.info("pipeline_stage_7", stage="execution", sized=len(decisions))
            for decision in decisions:
                if decision.action == "EXECUTE" and self.execution_callback:
                    await self.execution_callback(decision)
                    self.state.total_executions += 1

        except Exception as e:
            logger.error("pipeline_cycle_error", error=str(e))

        # Update state
        cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
        self.state.cycles_completed += 1
        self.state.total_decisions += len(decisions)
        self.state.last_cycle_time = datetime.utcnow()
        self.state.avg_cycle_duration_s = (
            (self.state.avg_cycle_duration_s * (self.state.cycles_completed - 1) + cycle_duration)
            / self.state.cycles_completed
        )

        logger.info(
            "pipeline_cycle_complete",
            cycle=self.state.cycles_completed,
            decisions=len(decisions),
            executions=sum(1 for d in decisions if d.action == "EXECUTE"),
            duration_s=f"{cycle_duration:.1f}",
        )

        return decisions

    # =========================================================================
    # SIGNAL COMBINATION
    # =========================================================================

    async def _combine_signals(
        self,
        opportunity: MarketOpportunity,
        wallet_signal: Optional[SmartMoneySignal],
        sentiment_signal: Optional[SentimentSignal],
        timing_signal: Optional[TimingSignal],
        ai_signal: Optional[DualAISignal] = None,
    ) -> Optional[TradingDecision]:
        """
        Combine all engine signals into a unified trading decision.
        Uses weighted voting with dynamic weight adjustment.
        """
        reasoning = []
        direction_votes = {"YES": 0.0, "NO": 0.0}
        confidence_scores = []
        edge_estimates = []

        # --- Market Scanner Signal ---
        direction_votes[opportunity.direction] += self.weights["market"]
        confidence_scores.append(opportunity.confidence * self.weights["market"])
        edge_estimates.append(opportunity.edge)
        reasoning.append(f"Market: {opportunity.direction} edge={opportunity.edge:.3f}")

        # --- Wallet Intelligence Signal ---
        if wallet_signal:
            direction_votes[wallet_signal.direction] += self.weights["wallet"] * wallet_signal.confidence
            confidence_scores.append(wallet_signal.confidence * self.weights["wallet"])
            reasoning.append(
                f"Wallet: {wallet_signal.direction} consensus={wallet_signal.wallet_consensus:.1%} "
                f"avg_wr={wallet_signal.avg_wallet_winrate:.1%} wallets={wallet_signal.num_wallets}"
            )

        # --- Sentiment AI Signal ---
        if sentiment_signal:
            direction_votes[sentiment_signal.direction] += self.weights["sentiment"] * sentiment_signal.confidence
            confidence_scores.append(sentiment_signal.confidence * self.weights["sentiment"])
            reasoning.append(
                f"Sentiment: {sentiment_signal.direction} score={sentiment_signal.sentiment_score:.3f} "
                f"mentions={sentiment_signal.volume_of_mentions}"
            )

        # --- Volatility Timing Signal ---
        if timing_signal:
            if timing_signal.action == "ENTER":
                direction_votes[timing_signal.direction] += self.weights["timing"] * timing_signal.confidence
                confidence_scores.append(timing_signal.confidence * self.weights["timing"])
                reasoning.append(f"Timing: ENTER {timing_signal.urgency} - {timing_signal.reasoning}")
            elif timing_signal.action == "EXIT":
                # Timing says exit → reduce confidence
                confidence_scores.append(-0.2)
                reasoning.append(f"Timing: EXIT warning - {timing_signal.reasoning}")

        # --- Dual AI Reasoning Signal ---
        if ai_signal and ai_signal.consensus_confidence > 0.3:
            ai_weight = 0.20  # AI gets significant weight when available
            direction_votes[ai_signal.consensus_direction] += ai_weight * ai_signal.consensus_confidence
            confidence_scores.append(ai_signal.consensus_confidence * ai_weight)
            # Strong agreement boosts confidence significantly
            if ai_signal.agreement_level == "strong":
                confidence_scores.append(0.15)  # Bonus for strong AI consensus
            reasoning.append(
                f"AI({ai_signal.agreement_level}): {ai_signal.consensus_direction} "
                f"p={ai_signal.consensus_probability:.3f} conf={ai_signal.consensus_confidence:.2f}"
            )

        # --- Determine Final Direction ---
        final_direction = max(direction_votes, key=direction_votes.get)
        direction_strength = direction_votes[final_direction] - direction_votes[
            "NO" if final_direction == "YES" else "YES"
        ]

        # Need consensus
        if direction_strength < 0.15:
            return None

        # --- Calculate Combined Metrics ---
        combined_confidence = sum(max(c, 0) for c in confidence_scores)
        combined_edge = sum(edge_estimates) / len(edge_estimates) if edge_estimates else 0

        # Recalibrate probability using wallet intelligence
        true_probability = wallet_intelligence.recalibrate_probability(
            market_price=opportunity.market_price,
            smart_money_signal=wallet_signal,
            base_estimate=opportunity.true_probability if opportunity.true_probability else 0.5,
        )

        # Final EV
        if final_direction == "YES":
            final_ev = true_probability - opportunity.market_price
        else:
            final_ev = (1 - true_probability) - (1 - opportunity.market_price)

        return TradingDecision(
            market_id=opportunity.market_id,
            question=opportunity.question,
            direction=final_direction,
            action="PENDING",  # Set by EV gate
            combined_edge=combined_edge,
            combined_confidence=min(combined_confidence, 0.95),
            final_ev=final_ev,
            true_probability=true_probability,
            market_signal=opportunity,
            wallet_signal=wallet_signal,
            sentiment_signal=sentiment_signal,
            timing_signal=timing_signal,
            reasoning=reasoning,
        )

    # =========================================================================
    # EV GATE
    # =========================================================================

    def _apply_ev_gate(self, decisions: list[TradingDecision]) -> list[TradingDecision]:
        """
        Filter decisions by minimum Expected Value threshold.
        Only pass decisions with positive EV above threshold.
        """
        passed = []
        for decision in decisions:
            if decision.final_ev >= settings.risk.min_ev_threshold:
                decision.action = "EXECUTE"
                passed.append(decision)
            else:
                decision.action = "SKIP"
                self.state.total_skipped += 1

        # Sort by EV (best first)
        passed.sort(key=lambda d: d.final_ev, reverse=True)

        # Limit to max concurrent positions
        max_new = settings.trading.max_concurrent_positions - (risk_engine.state.open_positions if risk_engine.state else 0)
        return passed[:max(max_new, 0)]

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def _apply_sizing(self, decisions: list[TradingDecision]) -> list[TradingDecision]:
        """Apply Kelly Criterion position sizing to each decision."""
        sized = []

        for decision in decisions:
            position = risk_engine.calculate_position_size(
                market_id=decision.market_id,
                market_price=decision.market_signal.market_price if decision.market_signal else 0.5,
                true_probability=decision.true_probability,
                confidence=decision.combined_confidence,
                signal_type=decision.market_signal.signal_type.value if decision.market_signal else "combined",
            )

            if position:
                decision.position_size = position
                sized.append(decision)
                decision.reasoning.append(
                    f"Size: ${position.size_usd:.2f} ({position.size_pct_bankroll:.1%}) "
                    f"Kelly={position.kelly_adjusted:.3f} R:R={position.risk_reward_ratio:.1f}"
                )
            else:
                decision.action = "SKIP"
                decision.reasoning.append("Size: SKIP - below minimum or circuit breaker")

        return sized

    # =========================================================================
    # DYNAMIC WEIGHT ADJUSTMENT
    # =========================================================================

    def adjust_weights(self):
        """
        Adjust signal weights based on historical accuracy.
        Engines that produce more accurate signals get higher weight.
        """
        calibration = risk_engine.get_calibration_stats()

        if not calibration:
            return

        # Map signal types to engine weights
        type_to_engine = {
            "mispricing": "market",
            "arbitrage": "market",
            "wallet_follow": "wallet",
            "sentiment": "sentiment",
            "volatility": "timing",
        }

        engine_accuracies = {}
        for sig_type, stats in calibration.items():
            engine = type_to_engine.get(sig_type)
            if engine and stats["sample_size"] >= 10:
                if engine not in engine_accuracies:
                    engine_accuracies[engine] = []
                engine_accuracies[engine].append(stats["accuracy"])

        if not engine_accuracies:
            return

        # Calculate new weights proportional to accuracy
        avg_accuracies = {eng: np.mean(accs) for eng, accs in engine_accuracies.items()}
        total_accuracy = sum(avg_accuracies.values())

        if total_accuracy > 0:
            for engine, accuracy in avg_accuracies.items():
                self.weights[engine] = accuracy / total_accuracy

        # Ensure minimum weight for each engine
        for key in self.weights:
            self.weights[key] = max(self.weights[key], 0.10)

        # Normalize to sum = 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info("weights_adjusted", weights=self.weights)

    # =========================================================================
    # CONTINUOUS OPERATION
    # =========================================================================

    async def run_continuous(self):
        """
        Run continuous orchestration loop.
        Scans, analyzes, decides, and executes every cycle.
        """
        logger.info("continuous_orchestration_started",
                   interval=settings.trading.scan_interval_seconds)

        cycle_count = 0
        while self._running:
            try:
                await self.run_pipeline_cycle()

                # Adjust weights periodically (every 10 cycles)
                cycle_count += 1
                if cycle_count % 10 == 0:
                    self.adjust_weights()

                # Save risk state periodically
                if cycle_count % 5 == 0:
                    await risk_engine._save_state()

            except Exception as e:
                logger.error("orchestration_loop_error", error=str(e))

            await asyncio.sleep(settings.trading.scan_interval_seconds)


# Singleton
orchestrator = Orchestrator()
