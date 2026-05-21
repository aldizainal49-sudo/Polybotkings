"""
Sentiment AI Engine
===================
Scans Reddit + Google News RSS BEFORE market adjusts price.
NLP-powered sentiment analysis with real-time signal generation.

Core capabilities:
- Monitor Reddit crypto/prediction market subreddits (GRATIS)
- Scrape Google News RSS for market-relevant keywords (GRATIS)
- Aggregate news from multiple RSS/API sources (GRATIS)
- Run NLP-based sentiment classification
- Detect sentiment shifts before they reflect in prices
- Generate trading signals from sentiment divergence
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import httpx
import feedparser
import numpy as np
from textblob import TextBlob

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Signal, SignalType, async_session

logger = get_logger("sentiment_ai")

# ============================================================
# FREE DATA SOURCES
# ============================================================

# Reddit subreddits to monitor (crypto + prediction markets)
REDDIT_SUBREDDITS = [
    "cryptocurrency",
    "polymarket",
    "bitcoin",
    "ethereum",
    "CryptoMarkets",
    "defi",
    "altcoin",
    "predictit",
    "wallstreetbets",
]

# Google News RSS (GRATIS - tanpa API key)
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"

# Crypto news RSS feeds (GRATIS)
NEWS_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
    "https://decrypt.co/feed",
    "https://thedefiant.io/feed",
]

# Reddit OAuth endpoint
REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_URL = "https://oauth.reddit.com"


@dataclass
class SentimentData:
    """Sentiment analysis result for a piece of content."""
    source: str  # "reddit", "google_news", "news_rss"
    text: str
    sentiment_score: float  # -1.0 to 1.0
    subjectivity: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    keywords: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    url: str = ""
    upvotes: int = 0  # Reddit score


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
    Scans Reddit + Google News RSS, runs NLP, generates directional signals.
    ALL DATA SOURCES ARE FREE.
    """

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.sentiment_history: dict[str, list[SentimentData]] = {}  # market_id -> history
        self.market_keywords: dict[str, list[str]] = {}  # market_id -> keywords
        self._running: bool = False
        self._reddit_token: str = ""
        self._reddit_token_expires: datetime = datetime.min

    async def start(self):
        """Initialize sentiment engine."""
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=20),
        )
        # Authenticate with Reddit if credentials available
        await self._reddit_authenticate()
        self._running = True
        logger.info("sentiment_ai_started", sources=["reddit", "google_news_rss", "crypto_news_rss"])

    async def stop(self):
        """Shutdown sentiment engine."""
        self._running = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info("sentiment_ai_stopped")

    # =========================================================================
    # REDDIT AUTHENTICATION (GRATIS)
    # =========================================================================

    async def _reddit_authenticate(self):
        """Get Reddit OAuth token (free tier - 100 req/min)."""
        if not settings.reddit.client_id or not settings.reddit.client_secret:
            logger.info("reddit_no_credentials", msg="Reddit scanning disabled, using news RSS only")
            return

        try:
            auth = (settings.reddit.client_id, settings.reddit.client_secret)
            resp = await self.http_client.post(
                REDDIT_AUTH_URL,
                auth=auth,
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": settings.reddit.user_agent},
            )
            if resp.status_code == 200:
                data = resp.json()
                self._reddit_token = data.get("access_token", "")
                expires_in = data.get("expires_in", 3600)
                self._reddit_token_expires = datetime.utcnow() + timedelta(seconds=expires_in - 60)
                logger.info("reddit_authenticated", expires_in=expires_in)
            else:
                logger.warning("reddit_auth_failed", status=resp.status_code)
        except Exception as e:
            logger.warning("reddit_auth_error", error=str(e))

    async def _ensure_reddit_token(self):
        """Refresh Reddit token if expired."""
        if datetime.utcnow() >= self._reddit_token_expires:
            await self._reddit_authenticate()

    # =========================================================================
    # KEYWORD EXTRACTION
    # =========================================================================

    def extract_market_keywords(self, question: str) -> list[str]:
        """
        Extract searchable keywords from a market question.
        E.g., "Will Bitcoin reach $100k by June?" → ["bitcoin", "$100k", "btc", "crypto"]
        """
        stop_words = {"will", "the", "be", "is", "are", "was", "were", "has", "have",
                      "do", "does", "did", "a", "an", "by", "in", "on", "at", "to",
                      "for", "of", "with", "before", "after", "this", "that"}

        text = question.lower().strip("?!.")
        words = re.findall(r'[\w$#@]+', text)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Add entity variants
        expanded = list(keywords)
        keyword_map = {
            "bitcoin": ["btc", "bitcoin"],
            "ethereum": ["eth", "ethereum"],
            "trump": ["trump", "donald", "potus"],
            "election": ["election", "vote", "polling"],
            "fed": ["fed", "federal reserve", "interest rate"],
            "ai": ["artificial intelligence", "openai", "chatgpt"],
            "solana": ["sol", "solana"],
            "xrp": ["xrp", "ripple"],
        }
        for kw in keywords:
            if kw in keyword_map:
                expanded.extend(keyword_map[kw])

        return list(set(expanded))[:15]

    # =========================================================================
    # REDDIT SCANNING (GRATIS - 100 req/menit)
    # =========================================================================

    async def scan_reddit(self, keywords: list[str], max_posts: int = 50) -> list[SentimentData]:
        """
        Search Reddit for posts matching keywords.
        Uses Reddit API free tier (100 requests/minute).
        Scans: r/cryptocurrency, r/polymarket, r/bitcoin, etc.
        """
        if not self._reddit_token:
            return []

        await self._ensure_reddit_token()
        sentiments = []
        query = " OR ".join(keywords[:5])

        try:
            # Search across crypto subreddits
            headers = {
                "Authorization": f"Bearer {self._reddit_token}",
                "User-Agent": settings.reddit.user_agent,
            }

            # Search posts
            resp = await self.http_client.get(
                f"{REDDIT_API_URL}/search",
                headers=headers,
                params={
                    "q": query,
                    "sort": "hot",
                    "t": "day",  # Last 24 hours
                    "limit": max_posts,
                    "type": "link",
                }
            )

            if resp.status_code != 200:
                logger.warning("reddit_search_error", status=resp.status_code)
                return []

            data = resp.json()
            posts = data.get("data", {}).get("children", [])

            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")[:300]
                text = f"{title}. {selftext}".strip()
                score = post_data.get("score", 0)
                num_comments = post_data.get("num_comments", 0)

                if not text:
                    continue

                # Check relevance
                relevance = self._calculate_relevance(text, keywords)
                if relevance < 0.2:
                    continue

                # Analyze sentiment
                sentiment_score = self._analyze_sentiment(text)

                # Weight by engagement (upvotes + comments)
                engagement_weight = 1 + min((score + num_comments) / 500, 3.0)
                weighted_score = sentiment_score * engagement_weight

                sentiments.append(SentimentData(
                    source="reddit",
                    text=text[:400],
                    sentiment_score=weighted_score,
                    subjectivity=TextBlob(text).sentiment.subjectivity,
                    relevance_score=relevance,
                    keywords=keywords,
                    timestamp=datetime.utcfromtimestamp(post_data.get("created_utc", 0)),
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    upvotes=score,
                ))

            # Also scan comments from crypto subreddits
            for subreddit in REDDIT_SUBREDDITS[:3]:  # Top 3 subreddits
                await asyncio.sleep(0.1)  # Rate limit
                comments = await self._scan_subreddit_comments(subreddit, keywords, headers)
                sentiments.extend(comments)

        except httpx.HTTPError as e:
            logger.error("reddit_scan_error", error=str(e))

        logger.info("reddit_scan_complete", posts=len(sentiments), query=query[:30])
        return sentiments

    async def _scan_subreddit_comments(
        self, subreddit: str, keywords: list[str], headers: dict
    ) -> list[SentimentData]:
        """Scan hot posts' comments in a subreddit for sentiment."""
        sentiments = []
        try:
            resp = await self.http_client.get(
                f"{REDDIT_API_URL}/r/{subreddit}/hot",
                headers=headers,
                params={"limit": 10}
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            posts = data.get("data", {}).get("children", [])

            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "")

                # Only check posts relevant to keywords
                if not any(kw.lower() in title.lower() for kw in keywords[:3]):
                    continue

                text = f"{title}. {post_data.get('selftext', '')[:200]}"
                score = self._analyze_sentiment(text)
                relevance = self._calculate_relevance(text, keywords)

                if relevance >= 0.3:
                    sentiments.append(SentimentData(
                        source="reddit",
                        text=text[:300],
                        sentiment_score=score,
                        subjectivity=0.5,
                        relevance_score=relevance,
                        keywords=keywords,
                        upvotes=post_data.get("score", 0),
                    ))

        except Exception:
            pass

        return sentiments

    # =========================================================================
    # GOOGLE NEWS RSS (GRATIS - tanpa API key, unlimited)
    # =========================================================================

    async def scan_google_news(self, keywords: list[str]) -> list[SentimentData]:
        """
        Scrape Google News RSS for relevant articles.
        COMPLETELY FREE - no API key needed.
        """
        sentiments = []
        query = "+".join(keywords[:5])
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(" ".join(keywords[:5])))

        try:
            resp = await self.http_client.get(url, timeout=15.0)
            if resp.status_code != 200:
                return []

            feed = feedparser.parse(resp.text)

            for entry in feed.entries[:25]:  # Latest 25 articles
                title = entry.get("title", "")
                # Google News RSS includes source in title: "Title - Source"
                source_name = ""
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0]
                    source_name = parts[1] if len(parts) > 1 else ""

                summary = entry.get("summary", "")
                text = f"{title}. {summary}"

                # Check relevance
                relevance = self._calculate_relevance(text, keywords)
                if relevance < 0.25:
                    continue

                score = self._analyze_sentiment(text)

                # Parse date
                pub_date = entry.get("published_parsed")
                timestamp = datetime(*pub_date[:6]) if pub_date else datetime.utcnow()

                # Only recent (last 24h)
                if datetime.utcnow() - timestamp > timedelta(hours=24):
                    continue

                sentiments.append(SentimentData(
                    source="google_news",
                    text=text[:500],
                    sentiment_score=score,
                    subjectivity=TextBlob(text).sentiment.subjectivity,
                    relevance_score=relevance,
                    keywords=keywords,
                    timestamp=timestamp,
                    url=entry.get("link", ""),
                ))

        except Exception as e:
            logger.warning("google_news_scan_error", error=str(e))

        logger.info("google_news_scan_complete", articles=len(sentiments))
        return sentiments

    # =========================================================================
    # CRYPTO NEWS RSS (GRATIS - tanpa API key)
    # =========================================================================

    async def scan_crypto_news(self, keywords: list[str]) -> list[SentimentData]:
        """
        Scan crypto-specific news RSS feeds.
        CoinTelegraph, CoinDesk, Decrypt, etc. — all FREE.
        """
        sentiments = []

        for feed_url in NEWS_RSS_FEEDS:
            try:
                resp = await self.http_client.get(feed_url, timeout=15.0)
                if resp.status_code != 200:
                    continue

                feed = feedparser.parse(resp.text)

                for entry in feed.entries[:20]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}"

                    relevance = self._calculate_relevance(text, keywords)
                    if relevance < 0.3:
                        continue

                    score = self._analyze_sentiment(text)
                    pub_date = entry.get("published_parsed")
                    timestamp = datetime(*pub_date[:6]) if pub_date else datetime.utcnow()

                    # Only recent (last 24h)
                    if datetime.utcnow() - timestamp > timedelta(hours=24):
                        continue

                    sentiments.append(SentimentData(
                        source="news_rss",
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
    # SENTIMENT ANALYSIS (NLP)
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
            "partnership", "adoption", "milestone", "record high", "ath",
            "accumulate", "buy", "long", "green", "profit",
        ]
        bearish_keywords = [
            "bearish", "dump", "crash", "collapse", "failed", "rejected",
            "denied", "lost", "defeat", "scandal", "fraud", "hack",
            "investigation", "lawsuit", "ban", "restriction", "sell",
            "short", "red", "loss", "rug", "scam",
        ]

        text_lower = text.lower()
        bull_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bear_count = sum(1 for kw in bearish_keywords if kw in text_lower)

        keyword_score = (bull_count - bear_count) * 0.15

        # Layer 3: Negation detection
        negation_words = ["not", "no", "never", "won't", "can't", "don't", "unlikely"]
        has_negation = any(neg in text_lower for neg in negation_words)
        if has_negation:
            keyword_score *= -0.5

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
        2. Scan Reddit + Google News + Crypto News (ALL FREE)
        3. Aggregate sentiment
        4. Compare to current price (sentiment divergence)
        5. Generate signal if divergence is significant
        """
        # 1. Extract keywords
        keywords = self.extract_market_keywords(question)
        if not keywords:
            return None

        # 2. Scan ALL sources concurrently (all free)
        reddit_task = self.scan_reddit(keywords)
        google_news_task = self.scan_google_news(keywords)
        crypto_news_task = self.scan_crypto_news(keywords)

        reddit_sentiments, google_sentiments, crypto_sentiments = await asyncio.gather(
            reddit_task, google_news_task, crypto_news_task,
            return_exceptions=True
        )

        if isinstance(reddit_sentiments, Exception):
            reddit_sentiments = []
        if isinstance(google_sentiments, Exception):
            google_sentiments = []
        if isinstance(crypto_sentiments, Exception):
            crypto_sentiments = []

        all_sentiments = list(reddit_sentiments) + list(google_sentiments) + list(crypto_sentiments)
        if len(all_sentiments) < 3:
            return None

        # 3. Aggregate sentiment (weighted by relevance + engagement)
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
        self.sentiment_history[market_id] = self.sentiment_history[market_id][-100:]

        # 5. Generate signal if sentiment diverges from price
        sentiment_implied_prob = (weighted_sentiment + 1) / 2  # Map [-1,1] to [0,1]
        divergence = sentiment_implied_prob - current_price

        # Need significant divergence
        min_divergence = 0.08
        if abs(divergence) < min_divergence:
            return None

        direction = "YES" if divergence > 0 else "NO"
        confidence = min(abs(divergence) * 3 + abs(sentiment_shift) * 2, 0.90)

        # Source breakdown
        sources = []
        if reddit_sentiments:
            sources.append(f"reddit({len(reddit_sentiments)})")
        if google_sentiments:
            sources.append(f"google_news({len(google_sentiments)})")
        if crypto_sentiments:
            sources.append(f"crypto_rss({len(crypto_sentiments)})")

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
            sources=sources,
        )

        return signal

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    async def analyze_markets(self, markets: list[dict]) -> list[SentimentSignal]:
        """
        Run sentiment analysis on multiple markets.
        Returns list of actionable sentiment signals.
        """
        signals = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent scans

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

        tasks = [_analyze(m) for m in markets[:30]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SentimentSignal):
                signals.append(result)

        logger.info("sentiment_batch_complete", markets=len(markets), signals=len(signals))
        return signals


# Singleton
sentiment_ai = SentimentAI()
