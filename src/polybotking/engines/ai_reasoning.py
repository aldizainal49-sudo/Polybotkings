"""
Dual AI Reasoning Engine
=========================
Uses OpenAI (GPT-4o) + Anthropic (Claude) for advanced market analysis.
Both AIs analyze independently, then results are compared for higher confidence.

Strategy:
- Each AI gets the same market data + context
- Both give probability estimate + reasoning
- If both agree → HIGH confidence signal
- If they disagree → LOWER confidence or SKIP
- Consensus = stronger edge detection

This is OPTIONAL — bot works without AI keys (uses TextBlob fallback).
"""

import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import httpx

from polybotking.config import settings
from polybotking.logger import get_logger

logger = get_logger("ai_reasoning")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


@dataclass
class AIAnalysis:
    """Result from a single AI analysis."""
    provider: str  # "openai" or "anthropic"
    probability_estimate: float  # 0.0 to 1.0
    direction: str  # "YES" or "NO"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    model_used: str
    latency_ms: float = 0.0


@dataclass
class DualAISignal:
    """Combined signal from both AIs."""
    market_id: str
    openai_analysis: Optional[AIAnalysis] = None
    anthropic_analysis: Optional[AIAnalysis] = None
    consensus_probability: float = 0.5
    consensus_direction: str = "YES"
    consensus_confidence: float = 0.0
    agreement_level: str = "none"  # "strong", "moderate", "disagree", "none"
    combined_reasoning: str = ""


class AIReasoningEngine:
    """
    Dual AI engine that uses OpenAI + Anthropic for market analysis.
    Both AIs analyze independently → compare for consensus.
    """

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self._running: bool = False

    async def start(self):
        """Initialize AI reasoning engine."""
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self._running = True

        providers = []
        if settings.ai.openai_api_key:
            providers.append("openai")
        if settings.ai.anthropic_api_key:
            providers.append("anthropic")

        if not providers:
            logger.info("ai_reasoning_started", providers="none (TextBlob fallback)")
        else:
            logger.info("ai_reasoning_started", providers=providers)

    async def stop(self):
        """Shutdown."""
        self._running = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info("ai_reasoning_stopped")

    # =========================================================================
    # MARKET ANALYSIS PROMPT
    # =========================================================================

    def _build_analysis_prompt(self, market_data: dict) -> str:
        """Build the analysis prompt for AI models."""
        question = market_data.get("question", "")
        yes_price = market_data.get("yes_price", 0.5)
        no_price = market_data.get("no_price", 0.5)
        volume = market_data.get("volume", 0)
        end_date = market_data.get("end_date", "")
        sentiment_context = market_data.get("sentiment_context", "")
        wallet_context = market_data.get("wallet_context", "")

        return f"""You are an expert prediction market analyst. Analyze this Polymarket market and provide your probability estimate.

MARKET: "{question}"
CURRENT PRICE: YES=${yes_price:.3f} / NO=${no_price:.3f}
VOLUME: ${volume:,.0f}
RESOLVES: {end_date}

ADDITIONAL CONTEXT:
- Sentiment from news/reddit: {sentiment_context}
- Smart money wallet activity: {wallet_context}

INSTRUCTIONS:
1. Estimate the TRUE probability of YES outcome (0.00 to 1.00)
2. Compare your estimate to current market price
3. If your estimate > market price → direction is YES (underpriced)
4. If your estimate < market price → direction is NO (overpriced)
5. Rate your confidence in this analysis (0.00 to 1.00)

RESPOND IN EXACTLY THIS FORMAT (no other text):
PROBABILITY: 0.XX
DIRECTION: YES or NO
CONFIDENCE: 0.XX
REASONING: One sentence explaining why"""

    # =========================================================================
    # OPENAI ANALYSIS
    # =========================================================================

    async def analyze_with_openai(self, market_data: dict) -> Optional[AIAnalysis]:
        """Run analysis using OpenAI GPT."""
        if not settings.ai.openai_api_key:
            return None

        prompt = self._build_analysis_prompt(market_data)
        start_time = datetime.utcnow()

        try:
            resp = await self.http_client.post(
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.ai.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.ai.openai_model,
                    "messages": [
                        {"role": "system", "content": "You are a quantitative prediction market analyst. Be precise and data-driven."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,  # Low temperature for consistency
                    "max_tokens": 200,
                }
            )

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            if resp.status_code != 200:
                logger.warning("openai_api_error", status=resp.status_code)
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Parse response
            analysis = self._parse_ai_response(content, "openai", settings.ai.openai_model, latency)
            return analysis

        except Exception as e:
            logger.warning("openai_analysis_error", error=str(e))
            return None

    # =========================================================================
    # ANTHROPIC ANALYSIS
    # =========================================================================

    async def analyze_with_anthropic(self, market_data: dict) -> Optional[AIAnalysis]:
        """Run analysis using Anthropic Claude."""
        if not settings.ai.anthropic_api_key:
            return None

        prompt = self._build_analysis_prompt(market_data)
        start_time = datetime.utcnow()

        try:
            resp = await self.http_client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": settings.ai.anthropic_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": settings.ai.anthropic_model,
                    "max_tokens": 200,
                    "temperature": 0.3,
                    "system": "You are a quantitative prediction market analyst. Be precise and data-driven.",
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                }
            )

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            if resp.status_code != 200:
                logger.warning("anthropic_api_error", status=resp.status_code)
                return None

            data = resp.json()
            content = data["content"][0]["text"].strip()

            # Parse response
            analysis = self._parse_ai_response(content, "anthropic", settings.ai.anthropic_model, latency)
            return analysis

        except Exception as e:
            logger.warning("anthropic_analysis_error", error=str(e))
            return None

    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================

    def _parse_ai_response(
        self, content: str, provider: str, model: str, latency: float
    ) -> Optional[AIAnalysis]:
        """Parse structured AI response into AIAnalysis."""
        try:
            lines = content.strip().split("\n")
            probability = 0.5
            direction = "YES"
            confidence = 0.5
            reasoning = ""

            for line in lines:
                line = line.strip()
                if line.upper().startswith("PROBABILITY:"):
                    val = line.split(":", 1)[1].strip()
                    probability = float(val)
                elif line.upper().startswith("DIRECTION:"):
                    val = line.split(":", 1)[1].strip().upper()
                    direction = "YES" if "YES" in val else "NO"
                elif line.upper().startswith("CONFIDENCE:"):
                    val = line.split(":", 1)[1].strip()
                    confidence = float(val)
                elif line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            # Validate ranges
            probability = max(0.05, min(0.95, probability))
            confidence = max(0.1, min(0.95, confidence))

            return AIAnalysis(
                provider=provider,
                probability_estimate=probability,
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model,
                latency_ms=latency,
            )

        except (ValueError, IndexError) as e:
            logger.warning("ai_parse_error", provider=provider, error=str(e))
            return None

    # =========================================================================
    # DUAL AI CONSENSUS
    # =========================================================================

    async def analyze_market(self, market_data: dict) -> DualAISignal:
        """
        Run dual AI analysis and determine consensus.
        
        Consensus levels:
        - STRONG: Both AIs agree on direction AND probability within 10%
        - MODERATE: Both agree on direction, probability differs 10-20%
        - DISAGREE: AIs give different directions
        - NONE: Only one AI available or both failed
        """
        market_id = market_data.get("market_id", "")
        signal = DualAISignal(market_id=market_id)

        provider = settings.ai.ai_provider

        # Run analyses based on config
        tasks = []
        if provider in ("openai", "both") and settings.ai.openai_api_key:
            tasks.append(("openai", self.analyze_with_openai(market_data)))
        if provider in ("anthropic", "both") and settings.ai.anthropic_api_key:
            tasks.append(("anthropic", self.analyze_with_anthropic(market_data)))

        if not tasks:
            return signal  # No AI available

        # Run concurrently
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, AIAnalysis):
                if tasks[i][0] == "openai":
                    signal.openai_analysis = result
                else:
                    signal.anthropic_analysis = result

        # Determine consensus
        signal = self._calculate_consensus(signal)

        logger.info(
            "dual_ai_analysis_complete",
            market=market_id[:12],
            agreement=signal.agreement_level,
            consensus_prob=f"{signal.consensus_probability:.3f}",
            consensus_dir=signal.consensus_direction,
            confidence=f"{signal.consensus_confidence:.2f}",
        )

        return signal

    def _calculate_consensus(self, signal: DualAISignal) -> DualAISignal:
        """Calculate consensus between two AI analyses."""
        openai = signal.openai_analysis
        anthropic = signal.anthropic_analysis

        # Both available
        if openai and anthropic:
            prob_diff = abs(openai.probability_estimate - anthropic.probability_estimate)

            # Check direction agreement
            if openai.direction == anthropic.direction:
                if prob_diff <= 0.10:
                    signal.agreement_level = "strong"
                    # Average probability, boost confidence
                    signal.consensus_probability = (openai.probability_estimate + anthropic.probability_estimate) / 2
                    signal.consensus_confidence = min(
                        (openai.confidence + anthropic.confidence) / 2 + 0.15, 0.95
                    )
                elif prob_diff <= 0.20:
                    signal.agreement_level = "moderate"
                    signal.consensus_probability = (openai.probability_estimate + anthropic.probability_estimate) / 2
                    signal.consensus_confidence = (openai.confidence + anthropic.confidence) / 2
                else:
                    signal.agreement_level = "moderate"
                    # Weighted by individual confidence
                    total_conf = openai.confidence + anthropic.confidence
                    if total_conf > 0:
                        signal.consensus_probability = (
                            openai.probability_estimate * openai.confidence +
                            anthropic.probability_estimate * anthropic.confidence
                        ) / total_conf
                    signal.consensus_confidence = (openai.confidence + anthropic.confidence) / 2 - 0.1

                signal.consensus_direction = openai.direction

            else:
                # Disagree on direction → lower confidence significantly
                signal.agreement_level = "disagree"
                # Use the one with higher confidence
                if openai.confidence > anthropic.confidence:
                    signal.consensus_probability = openai.probability_estimate
                    signal.consensus_direction = openai.direction
                else:
                    signal.consensus_probability = anthropic.probability_estimate
                    signal.consensus_direction = anthropic.direction
                signal.consensus_confidence = abs(openai.confidence - anthropic.confidence) * 0.5

            signal.combined_reasoning = (
                f"OpenAI({openai.model_used}): {openai.direction} p={openai.probability_estimate:.2f} "
                f"conf={openai.confidence:.2f} | "
                f"Anthropic({anthropic.model_used}): {anthropic.direction} p={anthropic.probability_estimate:.2f} "
                f"conf={anthropic.confidence:.2f} | "
                f"Agreement: {signal.agreement_level}"
            )

        # Only one available
        elif openai:
            signal.agreement_level = "none"
            signal.consensus_probability = openai.probability_estimate
            signal.consensus_direction = openai.direction
            signal.consensus_confidence = openai.confidence * 0.8  # Slightly less confident with single AI
            signal.combined_reasoning = f"OpenAI only: {openai.reasoning}"

        elif anthropic:
            signal.agreement_level = "none"
            signal.consensus_probability = anthropic.probability_estimate
            signal.consensus_direction = anthropic.direction
            signal.consensus_confidence = anthropic.confidence * 0.8
            signal.combined_reasoning = f"Anthropic only: {anthropic.reasoning}"

        return signal

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    async def analyze_markets(self, markets: list[dict]) -> list[DualAISignal]:
        """
        Run dual AI analysis on multiple markets.
        Rate limited to avoid API overuse.
        """
        signals = []
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent AI calls

        async def _analyze(market):
            async with semaphore:
                signal = await self.analyze_market(market)
                await asyncio.sleep(2.0)  # Rate limit between calls
                return signal

        tasks = [_analyze(m) for m in markets[:10]]  # Max 10 markets per cycle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, DualAISignal) and result.consensus_confidence > 0.3:
                signals.append(result)

        logger.info("ai_batch_complete", markets=len(markets), signals=len(signals))
        return signals


# Singleton
ai_reasoning = AIReasoningEngine()
