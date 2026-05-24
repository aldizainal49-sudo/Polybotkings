"""
Market Scanner Engine
====================
Tracks thousands of Polymarket markets via CLOB API.
Detects mispricing, arbitrage pairs, and high-EV opportunities.

Pipeline:
1. Fetch all active markets from Polymarket CLOB
2. Filter by timeframe (1hr - 7 days)
3. Snapshot orderbook for each market
4. Detect mispricing (YES + NO != 1.0 adjusted for spread)
5. Find pair arbitrage opportunities
6. Score each market's Expected Value
7. Emit signals for opportunities above threshold
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import Market, Signal, MarketStatus, SignalType, async_session

logger = get_logger("market_scanner")

# Polymarket CLOB API endpoints
CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


@dataclass
class OrderbookSnapshot:
    """Point-in-time orderbook snapshot for a market."""
    market_id: str
    timestamp: datetime
    best_bid_yes: float = 0.0
    best_ask_yes: float = 0.0
    best_bid_no: float = 0.0
    best_ask_no: float = 0.0
    mid_yes: float = 0.5
    mid_no: float = 0.5
    spread_yes: float = 0.0
    spread_no: float = 0.0
    depth_yes: float = 0.0
    depth_no: float = 0.0
    volume_24h: float = 0.0


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage between YES/NO pair."""
    market_id: str
    yes_price: float
    no_price: float
    combined: float  # yes + no (should be ~1.0)
    edge: float  # deviation from 1.0
    direction: str  # which side is mispriced
    confidence: float = 0.0


@dataclass
class MarketOpportunity:
    """Scored market opportunity."""
    market_id: str
    question: str
    direction: str  # YES or NO
    market_price: float
    true_probability: float
    edge: float
    ev: float
    confidence: float
    signal_type: SignalType
    reasoning: str = ""
    orderbook: Optional[OrderbookSnapshot] = None


class MarketScanner:
    """
    Core market scanning engine.
    Continuously monitors Polymarket CLOB for opportunities.
    """

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.markets_cache: dict[str, dict] = {}
        self.orderbook_history: dict[str, list[OrderbookSnapshot]] = {}
        self.scan_count: int = 0
        self.opportunities_found: int = 0
        self._running: bool = False

    async def start(self):
        """Initialize the scanner."""
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            headers={"Accept": "application/json"}
        )
        self._running = True
        logger.info("market_scanner_started")

    async def stop(self):
        """Shutdown the scanner."""
        self._running = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info("market_scanner_stopped", scans=self.scan_count, opportunities=self.opportunities_found)

    # =========================================================================
    # MARKET DISCOVERY
    # =========================================================================

    async def fetch_all_markets(self) -> list[dict]:
        """
        Fetch all active markets from Polymarket Gamma API.
        Loops through every category in settings.trading.market_tags_list and
        deduplicates markets that appear under more than one tag.
        Filters: active, within timeframe window (1hr-7days).
        """
        seen_ids: set[str] = set()
        markets: list[dict] = []

        tags = settings.trading.market_tags_list
        # Empty list means "all categories" - we send a single request without a tag filter
        scan_tags: list[Optional[str]] = list(tags) if tags else [None]

        logger.info("scanner_tags_configured", tags=tags or ["<all>"])

        for tag in scan_tags:
            tag_markets = await self._fetch_markets_for_tag(tag)
            new_count = 0
            for m in tag_markets:
                mid = m.get("id") or m.get("conditionId") or ""
                if not mid:
                    continue
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                markets.append(m)
                new_count += 1
            logger.info(
                "tag_scan_done",
                tag=tag or "<all>",
                fetched=len(tag_markets),
                new_unique=new_count,
                running_total=len(markets),
            )

        # Filter by timeframe
        filtered = self._filter_by_timeframe(markets)
        logger.info(
            "markets_fetched",
            total=len(markets),
            filtered=len(filtered),
            tags=len(scan_tags),
        )
        return filtered

    async def _fetch_markets_for_tag(self, tag: Optional[str]) -> list[dict]:
        """
        Fetch every page of active markets for one tag (or no tag = all).
        Returns a flat list. Pagination stops on empty page, short page,
        or 4xx (which Polymarket returns when offset exceeds total).
        """
        markets: list[dict] = []
        offset = 0
        limit = 100

        while True:
            params: dict = {
                "limit": limit,
                "offset": offset,
                "active": True,
                "closed": False,
            }
            if tag:
                params["tag"] = tag

            try:
                resp = await self.http_client.get(
                    f"{GAMMA_BASE_URL}/markets",
                    params=params,
                )
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                markets.extend(batch)
                offset += limit

                if len(batch) < limit:
                    break

                # Rate limiting between pages
                await asyncio.sleep(0.1)

            except httpx.HTTPStatusError as e:
                # 4xx at high offset = "no more pages" - normal end of pagination.
                status = e.response.status_code if e.response is not None else 0
                if status in (400, 404, 422):
                    logger.info(
                        "pagination_complete",
                        tag=tag or "<all>",
                        offset=offset,
                        status=status,
                        total_fetched=len(markets),
                    )
                else:
                    logger.warning(
                        "fetch_markets_http_error",
                        tag=tag or "<all>",
                        status=status,
                        offset=offset,
                        error=str(e),
                    )
                break
            except httpx.HTTPError as e:
                logger.warning(
                    "fetch_markets_network_error",
                    tag=tag or "<all>",
                    error=str(e),
                    offset=offset,
                )
                break

        return markets

    def _filter_by_timeframe(self, markets: list[dict]) -> list[dict]:
        """Filter markets to those resolving within 1hr - 7 days."""
        now = datetime.utcnow()
        min_end = now + timedelta(hours=settings.trading.market_timeframe_min_hours)
        max_end = now + timedelta(days=settings.trading.market_timeframe_max_days)

        filtered = []
        for m in markets:
            end_date_str = m.get("end_date_iso") or m.get("endDate")
            if not end_date_str:
                continue
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                if min_end <= end_date <= max_end:
                    filtered.append(m)
            except (ValueError, TypeError):
                continue

        return filtered

    # =========================================================================
    # CLOB ORDERBOOK SNAPSHOTS
    # =========================================================================

    async def snapshot_orderbook(self, token_id: str) -> Optional[OrderbookSnapshot]:
        """
        Take a CLOB orderbook snapshot for a given token.
        Captures best bid/ask, spread, depth.
        """
        try:
            resp = await self.http_client.get(
                f"{CLOB_BASE_URL}/book",
                params={"token_id": token_id}
            )
            resp.raise_for_status()
            book = resp.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0

            # Calculate depth (sum of sizes at top 5 levels)
            bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
            ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])

            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.5
            spread = best_ask - best_bid

            return OrderbookSnapshot(
                market_id=token_id,
                timestamp=datetime.utcnow(),
                best_bid_yes=best_bid,
                best_ask_yes=best_ask,
                mid_yes=mid,
                spread_yes=spread,
                depth_yes=bid_depth + ask_depth,
            )

        except (httpx.HTTPError, KeyError, IndexError) as e:
            logger.warning("orderbook_snapshot_error", token_id=token_id, error=str(e))
            return None

    async def batch_snapshot(self, token_ids: list[str]) -> dict[str, OrderbookSnapshot]:
        """Snapshot multiple orderbooks concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

        async def _fetch(tid):
            async with semaphore:
                snap = await self.snapshot_orderbook(tid)
                await asyncio.sleep(0.05)  # Rate limit
                return tid, snap

        tasks = [_fetch(tid) for tid in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshots = {}
        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                snapshots[result[0]] = result[1]

        return snapshots

    # =========================================================================
    # MISPRICING DETECTION
    # =========================================================================

    def detect_mispricing(self, yes_price: float, no_price: float) -> Optional[ArbitrageOpportunity]:
        """
        Detect mispricing in YES/NO pair.
        
        In an efficient market: YES + NO = 1.0 (minus vig)
        If YES 0.62 + NO 0.41 = 1.03 → overpriced (sell opportunity)
        If YES 0.55 + NO 0.40 = 0.95 → underpriced (buy opportunity)
        
        Edge = |1.0 - combined| adjusted for typical vig (~2-3%)
        """
        combined = yes_price + no_price
        vig_adjusted = combined - 1.0  # Positive = overpriced, Negative = underpriced

        # Minimum edge threshold to account for spread/fees
        min_edge = settings.risk.min_edge_threshold

        if abs(vig_adjusted) < min_edge:
            return None

        if vig_adjusted > min_edge:
            # Combined > 1.0: one side is overpriced
            # Sell the more expensive side
            direction = "NO" if no_price > (1.0 - yes_price) else "YES"
            edge = vig_adjusted - 0.02  # Subtract typical fee
        elif vig_adjusted < -min_edge:
            # Combined < 1.0: underpriced, buy both or the cheaper one
            direction = "YES" if yes_price < no_price else "NO"
            edge = abs(vig_adjusted) - 0.02

        if edge <= 0:
            return None

        return ArbitrageOpportunity(
            market_id="",  # Set by caller
            yes_price=yes_price,
            no_price=no_price,
            combined=combined,
            edge=edge,
            direction=direction,
            confidence=min(edge * 10, 0.95),  # Scale edge to confidence
        )

    # =========================================================================
    # EXPECTED VALUE SCORING
    # =========================================================================

    def calculate_ev(
        self,
        market_price: float,
        true_probability: float,
        direction: str = "YES"
    ) -> float:
        """
        Calculate Expected Value of a trade.
        
        EV = (true_prob * potential_win) - ((1 - true_prob) * potential_loss)
        
        For YES at price P with true probability T:
        EV = T * (1 - P) - (1 - T) * P
        EV = T - P  (simplified for binary markets)
        """
        if direction == "YES":
            ev = true_probability - market_price
        else:
            ev = (1 - true_probability) - (1 - market_price)

        return ev

    def score_opportunity(
        self,
        market: dict,
        orderbook: Optional[OrderbookSnapshot],
        true_prob_estimate: float
    ) -> Optional[MarketOpportunity]:
        """
        Score a market opportunity combining all signals.
        Returns None if opportunity doesn't meet minimum thresholds.
        """
        yes_price = float(market.get("outcomePrices", "[0.5,0.5]").strip("[]").split(",")[0])
        no_price = 1.0 - yes_price

        # Determine direction (buy the underpriced side)
        if true_prob_estimate > yes_price:
            direction = "YES"
            market_price = yes_price
            edge = true_prob_estimate - yes_price
        else:
            direction = "NO"
            market_price = no_price
            edge = (1 - true_prob_estimate) - no_price

        # Calculate EV
        ev = self.calculate_ev(market_price, true_prob_estimate, direction)

        # Minimum thresholds
        if edge < settings.risk.min_edge_threshold:
            return None
        if ev < settings.risk.min_ev_threshold:
            return None

        # Confidence based on multiple factors
        confidence = self._calculate_confidence(edge, orderbook, market)

        return MarketOpportunity(
            market_id=market.get("id", ""),
            question=market.get("question", "Unknown"),
            direction=direction,
            market_price=market_price,
            true_probability=true_prob_estimate,
            edge=edge,
            ev=ev,
            confidence=confidence,
            signal_type=SignalType.MISPRICING,
            orderbook=orderbook,
            reasoning=f"Edge={edge:.3f} EV={ev:.3f} Dir={direction} Price={market_price:.3f}"
        )

    def _calculate_confidence(
        self,
        edge: float,
        orderbook: Optional[OrderbookSnapshot],
        market: dict
    ) -> float:
        """Calculate confidence score from multiple factors."""
        confidence = 0.0

        # Edge contribution (0-0.4)
        confidence += min(edge * 4, 0.4)

        # Liquidity/volume contribution (0-0.3)
        volume = float(market.get("volume", 0) or 0)
        if volume > 100000:
            confidence += 0.3
        elif volume > 10000:
            confidence += 0.2
        elif volume > 1000:
            confidence += 0.1

        # Orderbook depth contribution (0-0.3)
        if orderbook:
            if orderbook.spread_yes < 0.03:
                confidence += 0.2
            elif orderbook.spread_yes < 0.05:
                confidence += 0.1
            if orderbook.depth_yes > 1000:
                confidence += 0.1

        return min(confidence, 0.95)

    # =========================================================================
    # PAIR ARBITRAGE SCANNER
    # =========================================================================

    async def scan_pair_arbitrage(self, markets: list[dict]) -> list[ArbitrageOpportunity]:
        """
        Scan all markets for pair arbitrage.
        YES 0.62 + NO 0.41 = edge opportunity.
        """
        opportunities = []

        for market in markets:
            try:
                prices_str = market.get("outcomePrices", "")
                if not prices_str:
                    continue

                # Parse prices
                prices = [float(p) for p in prices_str.strip("[]").split(",")]
                if len(prices) < 2:
                    continue

                yes_price = prices[0]
                no_price = prices[1]

                arb = self.detect_mispricing(yes_price, no_price)
                if arb:
                    arb.market_id = market.get("id", "")
                    opportunities.append(arb)

            except (ValueError, IndexError):
                continue

        logger.info("pair_arbitrage_scan", markets_scanned=len(markets), opportunities=len(opportunities))
        return opportunities

    # =========================================================================
    # FULL SCAN CYCLE
    # =========================================================================

    async def run_scan_cycle(self) -> list[MarketOpportunity]:
        """
        Execute a full market scan cycle.
        
        Flow:
        1. Fetch all active markets
        2. Filter by timeframe
        3. Detect pair arbitrage
        4. Score opportunities
        5. Store to database
        6. Return actionable opportunities
        """
        self.scan_count += 1
        logger.info("scan_cycle_start", cycle=self.scan_count)

        # 1. Fetch markets
        markets = await self.fetch_all_markets()
        if not markets:
            logger.warning("no_markets_found")
            return []

        # 2. Scan for pair arbitrage
        arb_opportunities = await self.scan_pair_arbitrage(markets)

        # 3. Get orderbook snapshots for promising markets
        token_ids = []
        for m in markets[:100]:  # Top 100 by volume
            tokens = m.get("clobTokenIds", [])
            if tokens and isinstance(tokens[0], str) and len(tokens[0]) > 5:
                    token_ids.append(tokens[0])  # YES token

        snapshots = await self.batch_snapshot(token_ids[:50])  # Limit snapshots

        # 4. Score all opportunities
        opportunities = []

        # Arbitrage opportunities
        for arb in arb_opportunities:
            opp = MarketOpportunity(
                market_id=arb.market_id,
                question="",
                direction=arb.direction,
                market_price=arb.yes_price if arb.direction == "YES" else arb.no_price,
                true_probability=0.5,  # Updated by other engines
                edge=arb.edge,
                ev=arb.edge * 0.8,  # Conservative EV estimate
                confidence=arb.confidence,
                signal_type=SignalType.ARBITRAGE,
                reasoning=f"Pair arb: YES={arb.yes_price:.3f} NO={arb.no_price:.3f} Combined={arb.combined:.3f}"
            )
            opportunities.append(opp)

        # Mispricing opportunities (using volume-weighted estimate as initial true_prob)
        for market in markets:
            try:
                prices_str = market.get("outcomePrices", "")
                if not prices_str:
                    continue
                prices = [float(p) for p in prices_str.strip("[]").split(",")]
                yes_price = prices[0]

                # Initial probability estimate using volume-weighted mean reversion
                # This gets refined by Sentiment + Wallet Intelligence engines
                volume = float(market.get("volume", 0) or 0)
                true_prob_estimate = yes_price  # Baseline = market price

                # Look for volume anomalies suggesting mispricing
                volume_24h = float(market.get("volume24hr", 0) or 0)
                if volume > 0 and volume_24h > volume * 0.1:
                    # High recent activity might indicate price discovery
                    # Slight mean reversion bias
                    true_prob_estimate = yes_price * 0.95 + 0.5 * 0.05

                token_ids = market.get("clobTokenIds", [])
                snap = snapshots.get(token_ids[0]) if token_ids else None

                opp = self.score_opportunity(market, snap, true_prob_estimate)
                if opp:
                    opportunities.append(opp)

            except (ValueError, IndexError, TypeError):
                continue

        # 5. Store to database
        await self._store_scan_results(markets, opportunities)

        self.opportunities_found += len(opportunities)
        logger.info(
            "scan_cycle_complete",
            cycle=self.scan_count,
            markets=len(markets),
            opportunities=len(opportunities),
            arbitrage=len(arb_opportunities),
        )

        return opportunities

    async def _store_scan_results(self, markets: list[dict], opportunities: list[MarketOpportunity]):
        """Persist scan results to database."""
        async with async_session() as session:
            for m in markets[:500]:  # Store top 500
                market_id = m.get("id", "")
                if not market_id:
                    continue

                prices_str = m.get("outcomePrices", "[0.5,0.5]")
                try:
                    prices = [float(p) for p in prices_str.strip("[]").split(",")]
                except (ValueError, IndexError):
                    prices = [0.5, 0.5]

                # Upsert market
                existing = await session.get(Market, market_id)
                if existing:
                    existing.yes_price = prices[0] if prices else 0.5
                    existing.no_price = prices[1] if len(prices) > 1 else 0.5
                    existing.volume_24h = float(m.get("volume24hr", 0) or 0)
                    existing.last_scanned = datetime.utcnow()
                else:
                    end_date_str = m.get("end_date_iso") or m.get("endDate")
                    end_date = None
                    if end_date_str:
                        try:
                            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                        except ValueError:
                            pass

                    new_market = Market(
                        id=market_id,
                        condition_id=m.get("conditionId", ""),
                        question=m.get("question", "Unknown"),
                        category=m.get("category", ""),
                        end_date=end_date,
                        yes_price=prices[0] if prices else 0.5,
                        no_price=prices[1] if len(prices) > 1 else 0.5,
                        volume_24h=float(m.get("volume24hr", 0) or 0),
                        liquidity=float(m.get("liquidity", 0) or 0),
                        status=MarketStatus.ACTIVE,
                    )
                    session.add(new_market)

            # Store signals for opportunities
            for opp in opportunities:
                signal = Signal(
                    market_id=opp.market_id,
                    signal_type=opp.signal_type,
                    direction=opp.direction,
                    confidence=opp.confidence,
                    edge=opp.edge,
                    ev=opp.ev,
                    reasoning=opp.reasoning,
                )
                session.add(signal)

            await session.commit()

    # =========================================================================
    # CONTINUOUS MONITORING
    # =========================================================================

    async def run_continuous(self, callback=None):
        """
        Run continuous market scanning loop.
        Calls callback with opportunities each cycle.
        """
        logger.info("continuous_scanning_started", interval=settings.trading.scan_interval_seconds)

        while self._running:
            try:
                opportunities = await self.run_scan_cycle()

                if callback and opportunities:
                    await callback(opportunities)

            except Exception as e:
                logger.error("scan_cycle_error", error=str(e))

            await asyncio.sleep(settings.trading.scan_interval_seconds)

    # =========================================================================
    # v3: SMART ENTRY TIMING (Buy the Dip)
    # =========================================================================

    def should_wait_for_dip(self, market_id: str, current_price: float, direction: str) -> dict:
        """
        v3 Smart Entry: Instead of buying immediately at signal,
        wait for a micro-dip (1-3%) to get a better entry price.
        
        Returns:
            {"wait": True/False, "target_price": float, "reason": str}
        """
        history = self.orderbook_history.get(market_id, [])

        # Not enough data → don't wait, enter now
        if len(history) < 5:
            return {"wait": False, "target_price": current_price, "reason": "insufficient data"}

        # Calculate recent price range
        recent_prices = [snap.mid_yes for snap in history[-10:]]
        price_high = max(recent_prices)
        price_low = min(recent_prices)
        price_range = price_high - price_low

        # If price is at the HIGH of recent range → wait for dip
        if direction == "YES":
            position_in_range = (current_price - price_low) / price_range if price_range > 0 else 0.5

            if position_in_range > 0.75:
                # Price is near top of range → wait for pullback
                target = current_price * 0.97  # Wait for 3% dip
                return {
                    "wait": True,
                    "target_price": round(target, 3),
                    "reason": f"Price at {position_in_range:.0%} of range. Wait for dip to {target:.3f}",
                    "max_wait_seconds": 300,  # Wait max 5 minutes
                }
            elif position_in_range > 0.50:
                # Middle of range → small dip target
                target = current_price * 0.985  # 1.5% dip
                return {
                    "wait": True,
                    "target_price": round(target, 3),
                    "reason": f"Mid-range entry. Target {target:.3f} (-1.5%)",
                    "max_wait_seconds": 120,  # Wait max 2 minutes
                }
            else:
                # Price is at LOW of range → enter immediately (it's already dipped)
                return {"wait": False, "target_price": current_price, "reason": "already at dip"}

        else:  # direction == "NO" (want price to go down = NO is cheap)
            position_in_range = (price_high - current_price) / price_range if price_range > 0 else 0.5

            if position_in_range > 0.75:
                target = current_price * 1.03  # Wait for YES to spike (NO gets cheaper)
                return {
                    "wait": True,
                    "target_price": round(target, 3),
                    "reason": f"Wait for NO to get cheaper",
                    "max_wait_seconds": 300,
                }
            else:
                return {"wait": False, "target_price": current_price, "reason": "good NO entry"}

    async def wait_for_dip_entry(
        self, market_id: str, token_id: str, target_price: float, max_wait: int = 300
    ) -> Optional[float]:
        """
        Wait for price to hit target (dip entry).
        Returns actual entry price if dip occurs, None if timeout.
        """
        start_time = datetime.utcnow()
        check_interval = 5  # Check every 5 seconds

        while (datetime.utcnow() - start_time).total_seconds() < max_wait:
            snap = await self.snapshot_orderbook(token_id)
            if snap and snap.mid_yes <= target_price:
                logger.info("dip_entry_hit", token_id=token_id[:12],
                          target=f"{target_price:.3f}", actual=f"{snap.mid_yes:.3f}")
                return snap.mid_yes

            await asyncio.sleep(check_interval)

        # Timeout - dip didn't happen, enter at current price
        logger.info("dip_timeout", token_id=token_id[:12], target=f"{target_price:.3f}")
        return None  # Caller decides: enter at market or skip


# Singleton instance
market_scanner = MarketScanner()
