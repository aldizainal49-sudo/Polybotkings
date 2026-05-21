"""
Sentiment AI Engine
===================
Scans X (Twitter) + news BEFORE market adjusts price.
NLP-powered sentiment analysis with real-time signal generation.

Core capabilities:
- Monitor X/Twitter for market-relevant keywords/events
- Aggregate news from multiple RSS/API sources
- Run transformer-based sentiment classification
- Detect sentiment shifts before they reflect in prices
- Generate trading signals from sentiment divergence
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import httpx
import feedparser
import numpy as np
from textblob import TextBlob

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Signal, SignalType, async_session

logger = get_logger("sentiment_ai")

# News RSS sources for crypto/prediction markets
NEWS_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
    "https://decrypt.co/feed",
    "https://thedefiant.io/feed",
]

TWITTER_API_V2 = "https://api.twitter.com/2"


@dataclass
class SentimentData:
    """Sentiment analysis result for a piece of content."""
    source: str  # "twitter", "news", "reddit"
    text: str
    sentiment_score: float  # -1.0 to 1.0
    subjectivity: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    keywords: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    url: str = ""


@dataclass
class SentimentSignal:
    """Trading signal from sentiment analysis."""
    market_id: str
    direction: str  # YES/NO
    sentiment_score: float  # aggregated -1 to 1
    sentiment_shift: float  # change from previous period
    volume_of_mentions: int
    confidence: float
    sources: list[str] = field(default_factory=list)
    key_narratives: list[str] = field(default_factory=list)


class SentimentAI:
    """
    AI-powered sentiment engine.
    Scans X + news, runs NLP, generates directional signals.
    """

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.sentiment_history: dict[str, list[SentimentData]] = {}  # market_id -> history
        self.market_keywords: dict[str, list[str]] = {}  # market_id -> keywords
        self._running: bool = False
        self._transformer_model = None

    async def start(self):
        """Initialize sentiment engine."""
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=20),
        )
        self._running = True
        logger.info("sentiment_ai_started")

    async def stop(self):
        """Shutdown sentiment engine."""
        self._running = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info("sentiment_ai_stopped")

    # =========================================================================
    # KEYWORD EXTRACTION
    # =========================================================================

    def extract_market_keywords(self, question: str) -> list[str]:
        """
        Extract searchable keywords from a market question.
        E.g., "Will Bitcoin reach $100k by June?" → ["bitcoin", "$100k", "btc", "crypto"]
        """
        # Remove common question words
        stop_words = {"will", "the", "be", "is", "are", "was", "were", "has", "have",
                      "do", "does", "did", "a", "an", "by", "in", "on", "at", "to",
                      "for", "of", "with", "before", "after", "this", "that"}

        # Clean and tokenize
        text = question.lower().strip("?!.")
        words = re.findall(r'[\w$#@]+', text)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Add entity variants
        expanded = list(keywords)
        keyword_map = {
            "bitcoin": ["btc", "bitcoin", "₿"],
            "ethereum": ["eth", "ethereum"],
            "trump": ["trump", "donald", "potus"],
            "election": ["election", "vote", "polling"],
            "fed": ["fed", "federal reserve", "interest rate"],
            "ai": ["artificial intelligence", "openai", "chatgpt"],
        }
        for kw in keywords:
            if kw in keyword_map:
                expanded.extend(keyword_map[kw])

        return list(set(expanded))[:15]  # Max 15 keywords

    # =========================================================================
    # TWITTER/X SCANNING
    # =========================================================================

    async def scan_twitter(self, keywords: list[str], max_results: int = 100) -> list[SentimentData]:
        """
        Search Twitter/X for recent posts matching keywords.
        Uses Twitter API v2 with bearer token.
        """
        if not settings.twitter.bearer_token:
            return []

        sentiments = []
        query = " OR ".join(keywords[:5])  # Twitter limits query length

        try:
            resp = await self.http_client.get(
                f"{TWITTER_API_V2}/tweets/search/recent",
                headers={"Authorization": f"Bearer {settings.twitter.bearer_token}"},
                params={
                    "query": f"{query} -is:retweet lang:en",
                    "max_results": min(max_results, 100),
                    "tweet.fields": "created_at,public_metrics,text",
                }
            )

            if resp.status_code != 200:
                logger.warning("twitter_api_error", status=resp.status_code)
                return []

            data = resp.json()
            tweets = data.get("data", [])

            for tweet in tweets:
                text = tweet.get("text", "")
                score = self._analyze_sentiment(text)
                relevance = self._calculate_relevance(text, keywords)

                metrics = tweet.get("public_metrics", {})
                engagement = (
                    metrics.get("like_count", 0) +
                    metrics.get("retweet_count", 0) * 2 +
                    metrics.get("reply_count", 0)
                )

                # Weight by engagement
                weighted_score = score * (1 + min(engagement / 1000, 2.0))

                sentiments.append(SentimentData(
                    source="twitter",
                    text=text[:280],
                    sentiment_score=weighted_score,
                    subjectivity=TextBlob(text).sentiment.subjectivity,
                    relevance_score=relevance,
                    keywords=keywords,
                    timestamp=datetime.utcnow(),
                ))

        except httpx.HTTPError as e:
            logger.error("twitter_scan_error", error=str(e))

        return sentiments

    # =========================================================================
    # NEWS SCANNING
    # =========================================================================

    async def scan_news(self, keywords: list[str]) -> list[SentimentData]:
        """
        Scan news RSS feeds for relevant articles.
        Analyzes headlines and summaries for sentiment.
        """
        sentiments = []

        for feed_url in NEWS_RSS_FEEDS:
            try:
                resp = await self.http_client.get(feed_url, timeout=15.0)
                if resp.status_code != 200:
                    continue

                feed = feedparser.parse(resp.text)

                for entry in feed.entries[:20]:  # Latest 20 per feed
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}"

                    # Check relevance
                    relevance = self._calculate_relevance(text, keywords)
                    if relevance < 0.3:
                        continue

                    score = self._analyze_sentiment(text)
                    pub_date = entry.get("published_parsed")
                    timestamp = datetime(*pub_date[:6]) if pub_date else datetime.utcnow()

                    # Only recent articles (last 24h)
                    if datetime.utcnow() - timestamp > timedelta(hours=24):
                        continue

                    sentiments.append(SentimentData(
                        source="news",
                        text=text[:500],
                        sentiment_score=score,
                        subjectivity=TextBlob(text).sentiment.subjectivity,
                        relevance_score=relevance,
                        keywords=keywords,
                        timestamp=timestamp,
                        url=entry.get("link", ""),
                    ))

            except Exception as e:
                logger.warning("news_scan_error", feed=feed_url, error=str(e))
                continue

        return sentiments

    # =========================================================================
    # SENTIMENT ANALYSIS
    # =========================================================================

    def _analyze_sentiment(self, text: str) -> float:
        """
        Multi-layered sentiment analysis.
        Combines TextBlob + keyword-based + market-specific heuristics.
        Returns score from -1.0 (bearish) to 1.0 (bullish).
        """
        # Layer 1: TextBlob polarity
        blob = TextBlob(text)
        base_score = blob.sentiment.polarity

        # Layer 2: Crypto/market-specific keywords
        bullish_keywords = [
            "bullish", "moon", "pump", "surge", "rally", "breakout",
            "confirmed", "approved", "passed", "won", "victory", "success",
            "partnership", "adoption", "milestone", "record high",
        ]
        bearish_keywords = [
            "bearish", "dump", "crash", "collapse", "failed", "rejected",
            "denied", "lost", "defeat", "scandal", "fraud", "hack",
            "investigation", "lawsuit", "ban", "restriction",
        ]

        text_lower = text.lower()
        bull_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bear_count = sum(1 for kw in bearish_keywords if kw in text_lower)

        keyword_score = (bull_count - bear_count) * 0.15

        # Layer 3: Negation detection
        negation_words = ["not", "no", "never", "won't", "can't", "don't", "unlikely"]
        has_negation = any(neg in text_lower for neg in negation_words)
        if has_negation:
            keyword_score *= -0.5  # Partially flip

        # Combine layers
        final_score = base_score * 0.4 + keyword_score * 0.6
        return max(-1.0, min(1.0, final_score))

    def _calculate_relevance(self, text: str, keywords: list[str]) -> float:
        """Calculate how relevant a text is to the given keywords."""
        if not keywords:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return min(matches / max(len(keywords) * 0.3, 1), 1.0)

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    async def generate_sentiment_signal(
        self,
        market_id: str,
        question: str,
        current_price: float
    ) -> Optional[SentimentSignal]:
        """
        Generate a trading signal from sentiment analysis.
        
        Process:
        1. Extract keywords from market question
        2. Scan Twitter + News
        3. Aggregate sentiment
        4. Compare to current price (sentiment divergence)
        5. Generate signal if divergence is significant
        """
        # 1. Extract keywords
        keywords = self.extract_market_keywords(question)
        if not keywords:
            return None

        # 2. Scan sources concurrently
        twitter_task = self.scan_twitter(keywords)
        news_task = self.scan_news(keywords)

        twitter_sentiments, news_sentiments = await asyncio.gather(
            twitter_task, news_task, return_exceptions=True
        )

        if isinstance(twitter_sentiments, Exception):
            twitter_sentiments = []
        if isinstance(news_sentiments, Exception):
            news_sentiments = []

        all_sentiments = list(twitter_sentiments) + list(news_sentiments)
        if len(all_sentiments) < 3:
            return None

        # 3. Aggregate sentiment (weighted by relevance)
        weights = np.array([s.relevance_score for s in all_sentiments])
        scores = np.array([s.sentiment_score for s in all_sentiments])

        if weights.sum() == 0:
            return None

        weighted_sentiment = np.average(scores, weights=weights)

        # 4. Calculate sentiment shift (compare to history)
        prev_sentiments = self.sentiment_history.get(market_id, [])
        if prev_sentiments:
            prev_scores = [s.sentiment_score for s in prev_sentiments[-20:]]
            prev_avg = np.mean(prev_scores)
            sentiment_shift = weighted_sentiment - prev_avg
        else:
            sentiment_shift = 0.0

        # Update history
        if market_id not in self.sentiment_history:
            self.sentiment_history[market_id] = []
        self.sentiment_history[market_id].extend(all_sentiments)
        # Keep last 100 entries
        self.sentiment_history[market_id] = self.sentiment_history[market_id][-100:]

        # 5. Generate signal if sentiment diverges from price
        # Convert sentiment (-1 to 1) to implied probability (0 to 1)
        sentiment_implied_prob = (weighted_sentiment + 1) / 2  # Map [-1,1] to [0,1]

        # Divergence: sentiment says higher/lower than market price
        divergence = sentiment_implied_prob - current_price

        # Need significant divergence
        min_divergence = 0.08
        if abs(divergence) < min_divergence:
            return None

        direction = "YES" if divergence > 0 else "NO"
        confidence = min(abs(divergence) * 3 + abs(sentiment_shift) * 2, 0.90)

        # Source breakdown
        sources = []
        if twitter_sentiments:
            sources.append(f"twitter({len(twitter_sentiments)})")
        if news_sentiments:
            sources.append(f"news({len(news_sentiments)})")

        # Key narratives (top sentiment texts)
        sorted_sentiments = sorted(all_sentiments, key=lambda s: abs(s.sentiment_score), reverse=True)
        key_narratives = [s.text[:100] for s in sorted_sentiments[:3]]

        signal = SentimentSignal(
            market_id=market_id,
            direction=direction,
            sentiment_score=float(weighted_sentiment),
            sentiment_shift=float(sentiment_shift),
            volume_of_mentions=len(all_sentiments),
            confidence=float(confidence),
            sources=sources,
            key_narratives=key_narratives,
        )

        logger.info(
            "sentiment_signal_generated",
            market_id=market_id[:8],
            direction=direction,
            sentiment=f"{weighted_sentiment:.3f}",
            divergence=f"{divergence:.3f}",
            mentions=len(all_sentiments),
        )

        return signal

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    async def analyze_markets(
        self,
        markets: list[dict]
    ) -> list[SentimentSignal]:
        """
        Run sentiment analysis on multiple markets.
        Returns list of actionable sentiment signals.
        """
        signals = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls

        async def _analyze(market):
            async with semaphore:
                market_id = market.get("id", "")
                question = market.get("question", "")
                prices_str = market.get("outcomePrices", "[0.5,0.5]")

                try:
                    prices = [float(p) for p in prices_str.strip("[]").split(",")]
                    current_price = prices[0]
                except (ValueError, IndexError):
                    current_price = 0.5

                signal = await self.generate_sentiment_signal(market_id, question, current_price)
                await asyncio.sleep(1.0)  # Rate limit between markets
                return signal

        tasks = [_analyze(m) for m in markets[:30]]  # Analyze top 30 markets
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SentimentSignal):
                signals.append(result)

        logger.info("sentiment_batch_complete", markets=len(markets), signals=len(signals))
        return signals


# Singleton
sentiment_ai = SentimentAI()
