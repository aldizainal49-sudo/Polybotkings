"""
Wallet Intelligence Engine
==========================
Analyzes 14,000+ top trader wallets on Polymarket.
Cross-references win rates, size patterns, timing, and clusters.

Core capabilities:
- Track top performer wallets via on-chain data
- Cluster wallets by behavior (size, timing, category preference)
- Detect smart money flow before market moves
- Generate follow signals from high-conviction wallets
- Probability recalibration based on wallet consensus
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import httpx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from polybotking.config import settings
from polybotking.logger import get_logger
from polybotking.models import WalletProfile, Signal, SignalType, async_session

logger = get_logger("wallet_intelligence")

# Polymarket Gamma API for wallet activity
GAMMA_API = "https://gamma-api.polymarket.com"
POLYGON_RPC = "https://polygon-rpc.com"


@dataclass
class WalletActivity:
    """Recent activity from a tracked wallet."""
    address: str
    market_id: str
    side: str  # YES/NO
    size: float
    price: float
    timestamp: datetime
    is_entry: bool = True


@dataclass
class WalletCluster:
    """Group of wallets with similar behavior."""
    cluster_id: int
    wallet_count: int
    avg_winrate: float
    avg_pnl: float
    dominant_strategy: str  # "early_mover", "momentum", "contrarian", "arbitrage"
    avg_size_pattern: str  # "small", "medium", "large", "whale"
    active_markets: list[str] = field(default_factory=list)


@dataclass
class SmartMoneySignal:
    """Signal generated from smart money movement."""
    market_id: str
    direction: str
    wallet_consensus: float  # % of tracked wallets on same side
    avg_wallet_winrate: float
    total_volume: float
    num_wallets: int
    confidence: float
    cluster_ids: list[int] = field(default_factory=list)


class WalletIntelligence:
    """
    Analyzes top trader wallets to detect high-conviction moves.
    Builds profiles, clusters behavior, generates follow signals.
    """

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.wallet_profiles: dict[str, WalletProfile] = {}
        self.recent_activity: list[WalletActivity] = []
        self.clusters: list[WalletCluster] = []
        self._running: bool = False

    async def start(self):
        """Initialize wallet intelligence engine."""
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
        )
        await self._load_profiles_from_db()
        self._running = True
        logger.info("wallet_intelligence_started", tracked_wallets=len(self.wallet_profiles))

    async def stop(self):
        """Shutdown engine."""
        self._running = False
        if self.http_client:
            await self.http_client.aclose()
        logger.info("wallet_intelligence_stopped")

    # =========================================================================
    # WALLET DISCOVERY & PROFILING
    # =========================================================================

    async def discover_top_wallets(self, min_trades: int = 50, min_winrate: float = 0.60) -> list[dict]:
        """
        Discover top performing wallets from Polymarket leaderboard
        and on-chain activity analysis.
        """
        wallets = []

        try:
            # Fetch from Polymarket leaderboard/activity
            resp = await self.http_client.get(
                f"{GAMMA_API}/activity",
                params={"limit": 1000, "offset": 0}
            )
            if resp.status_code == 200:
                activities = resp.json()
                # Extract unique wallet addresses
                wallet_addresses = set()
                for act in activities:
                    addr = act.get("proxyWallet") or act.get("address", "")
                    if addr:
                        wallet_addresses.add(addr)

                logger.info("wallets_discovered", count=len(wallet_addresses))
                wallets = [{"address": addr} for addr in wallet_addresses]

        except httpx.HTTPError as e:
            logger.error("wallet_discovery_error", error=str(e))

        return wallets

    async def build_wallet_profile(self, address: str) -> Optional[WalletProfile]:
        """
        Build comprehensive profile for a single wallet.
        Analyzes trade history, win rate, patterns.
        """
        try:
            # Fetch wallet's trade history
            resp = await self.http_client.get(
                f"{GAMMA_API}/activity",
                params={"address": address, "limit": 500}
            )
            if resp.status_code != 200:
                return None

            trades = resp.json()
            if not trades or len(trades) < 10:
                return None

            # Calculate metrics
            total_trades = len(trades)
            wins = 0
            total_pnl = 0.0
            sizes = []
            hold_times = []
            categories = defaultdict(int)

            for trade in trades:
                size = float(trade.get("size", 0) or 0)
                pnl = float(trade.get("pnl", 0) or 0)

                sizes.append(size)
                total_pnl += pnl
                if pnl > 0:
                    wins += 1

                cat = trade.get("category", "unknown")
                categories[cat] += 1

            win_rate = wins / total_trades if total_trades > 0 else 0
            avg_size = np.mean(sizes) if sizes else 0

            # Determine size pattern
            if avg_size > 10000:
                size_pattern = "whale"
            elif avg_size > 1000:
                size_pattern = "large"
            elif avg_size > 100:
                size_pattern = "medium"
            else:
                size_pattern = "small"

            # Determine timing pattern based on entry relative to resolution
            timing_pattern = "mid"  # Default; refined with more data

            # Top categories
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            preferred_cats = [cat for cat, _ in top_categories]

            profile = WalletProfile(
                address=address,
                total_trades=total_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_position_size=avg_size,
                size_pattern=size_pattern,
                timing_pattern=timing_pattern,
                preferred_categories=preferred_cats,
                last_active=datetime.utcnow(),
                confidence_score=min(win_rate * (total_trades / 100), 0.95),
            )

            return profile

        except (httpx.HTTPError, ValueError) as e:
            logger.warning("profile_build_error", address=address[:10], error=str(e))
            return None

    async def bulk_profile_wallets(self, addresses: list[str], concurrency: int = 5):
        """Profile multiple wallets concurrently."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _profile(addr):
            async with semaphore:
                profile = await self.build_wallet_profile(addr)
                await asyncio.sleep(0.2)  # Rate limit
                return addr, profile

        tasks = [_profile(addr) for addr in addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_profiles = 0
        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                addr, profile = result
                self.wallet_profiles[addr] = profile
                new_profiles += 1

        logger.info("bulk_profiling_complete", total=len(addresses), profiled=new_profiles)

    # =========================================================================
    # CLUSTER ANALYSIS
    # =========================================================================

    def cluster_wallets(self, n_clusters: int = 8) -> list[WalletCluster]:
        """
        Cluster wallets by behavioral patterns using K-means.
        Features: win_rate, avg_size, trade_frequency, pnl, timing.
        """
        if len(self.wallet_profiles) < n_clusters * 2:
            logger.warning("insufficient_wallets_for_clustering", count=len(self.wallet_profiles))
            return []

        # Build feature matrix
        addresses = []
        features = []

        for addr, profile in self.wallet_profiles.items():
            if isinstance(profile, WalletProfile):
                addresses.append(addr)
                features.append([
                    profile.win_rate,
                    profile.avg_position_size,
                    profile.total_trades,
                    profile.total_pnl,
                    profile.avg_hold_time_hours,
                ])
            elif isinstance(profile, dict):
                addresses.append(addr)
                features.append([
                    profile.get("win_rate", 0),
                    profile.get("avg_position_size", 0),
                    profile.get("total_trades", 0),
                    profile.get("total_pnl", 0),
                    profile.get("avg_hold_time_hours", 0),
                ])

        if len(features) < n_clusters:
            return []

        # Normalize features
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Build cluster profiles
        clusters = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_features = X[cluster_mask]

            if len(cluster_features) == 0:
                continue

            avg_winrate = np.mean(cluster_features[:, 0])
            avg_size = np.mean(cluster_features[:, 1])
            avg_pnl = np.mean(cluster_features[:, 3])

            # Determine dominant strategy
            if avg_winrate > 0.75 and avg_size < 500:
                strategy = "early_mover"
            elif avg_winrate > 0.65 and avg_size > 5000:
                strategy = "whale_momentum"
            elif avg_winrate < 0.55 and avg_pnl > 0:
                strategy = "contrarian"
            else:
                strategy = "momentum"

            # Size pattern
            if avg_size > 10000:
                size_pat = "whale"
            elif avg_size > 1000:
                size_pat = "large"
            elif avg_size > 100:
                size_pat = "medium"
            else:
                size_pat = "small"

            cluster = WalletCluster(
                cluster_id=i,
                wallet_count=int(cluster_mask.sum()),
                avg_winrate=float(avg_winrate),
                avg_pnl=float(avg_pnl),
                dominant_strategy=strategy,
                avg_size_pattern=size_pat,
            )
            clusters.append(cluster)

            # Assign cluster_id to wallet profiles
            cluster_addresses = [addresses[j] for j in range(len(addresses)) if labels[j] == i]
            for addr in cluster_addresses:
                if addr in self.wallet_profiles and hasattr(self.wallet_profiles[addr], 'cluster_id'):
                    self.wallet_profiles[addr].cluster_id = i

        self.clusters = clusters
        logger.info("clustering_complete", n_clusters=len(clusters),
                   top_winrate=max(c.avg_winrate for c in clusters) if clusters else 0)
        return clusters

    # =========================================================================
    # SMART MONEY DETECTION
    # =========================================================================

    async def detect_smart_money_flow(self, market_id: str) -> Optional[SmartMoneySignal]:
        """
        Detect if smart money (high-winrate wallets) is flowing into a market.
        Returns signal if consensus among top wallets is strong.
        """
        try:
            # Fetch recent activity for this market
            resp = await self.http_client.get(
                f"{GAMMA_API}/activity",
                params={"market": market_id, "limit": 200}
            )
            if resp.status_code != 200:
                return None

            activities = resp.json()
            if not activities:
                return None

            # Cross-reference with tracked wallets
            yes_wallets = []
            no_wallets = []
            yes_volume = 0.0
            no_volume = 0.0

            for act in activities:
                addr = act.get("proxyWallet") or act.get("address", "")
                side = act.get("side", "").upper()
                size = float(act.get("size", 0) or 0)

                if addr in self.wallet_profiles:
                    profile = self.wallet_profiles[addr]
                    wr = profile.win_rate if hasattr(profile, 'win_rate') else 0

                    if wr >= 0.60:  # Only consider above-average wallets
                        if side == "BUY" or side == "YES":
                            yes_wallets.append((addr, wr, size))
                            yes_volume += size
                        elif side == "SELL" or side == "NO":
                            no_wallets.append((addr, wr, size))
                            no_volume += size

            total_tracked = len(yes_wallets) + len(no_wallets)
            if total_tracked < 3:
                return None

            # Determine consensus direction
            if len(yes_wallets) > len(no_wallets):
                direction = "YES"
                consensus_wallets = yes_wallets
                total_volume = yes_volume
            else:
                direction = "NO"
                consensus_wallets = no_wallets
                total_volume = no_volume

            consensus = len(consensus_wallets) / total_tracked
            avg_winrate = np.mean([wr for _, wr, _ in consensus_wallets])

            # Need strong consensus
            if consensus < 0.65 or avg_winrate < 0.65:
                return None

            # Calculate confidence
            confidence = (consensus * 0.4 + avg_winrate * 0.4 + min(total_tracked / 20, 1.0) * 0.2)

            return SmartMoneySignal(
                market_id=market_id,
                direction=direction,
                wallet_consensus=consensus,
                avg_wallet_winrate=float(avg_winrate),
                total_volume=total_volume,
                num_wallets=len(consensus_wallets),
                confidence=float(min(confidence, 0.95)),
            )

        except (httpx.HTTPError, ValueError) as e:
            logger.warning("smart_money_detection_error", market_id=market_id, error=str(e))
            return None

    # =========================================================================
    # PROBABILITY RECALIBRATION
    # =========================================================================

    def recalibrate_probability(
        self,
        market_price: float,
        smart_money_signal: Optional[SmartMoneySignal],
        base_estimate: float
    ) -> float:
        """
        Recalibrate true probability using wallet intelligence.
        
        Combines:
        - Market price (baseline)
        - Smart money consensus (strong signal)
        - Base estimate from other engines
        
        Weighted combination based on confidence levels.
        """
        if smart_money_signal is None:
            # No smart money data, use simple average
            return (market_price * 0.4 + base_estimate * 0.6)

        # Smart money weight based on confidence and consensus
        sm_weight = smart_money_signal.confidence * 0.5
        market_weight = 0.2
        base_weight = 1.0 - sm_weight - market_weight

        # Smart money directional adjustment
        if smart_money_signal.direction == "YES":
            sm_prob = market_price + (smart_money_signal.wallet_consensus - 0.5) * 0.3
        else:
            sm_prob = market_price - (smart_money_signal.wallet_consensus - 0.5) * 0.3

        sm_prob = max(0.05, min(0.95, sm_prob))

        # Weighted combination
        recalibrated = (
            sm_prob * sm_weight +
            market_price * market_weight +
            base_estimate * base_weight
        )

        return max(0.05, min(0.95, recalibrated))

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    async def _load_profiles_from_db(self):
        """Load existing wallet profiles from database."""
        async with async_session() as session:
            result = await session.execute(
                select(WalletProfile).where(WalletProfile.win_rate >= 0.55)
            )
            profiles = result.scalars().all()
            for p in profiles:
                self.wallet_profiles[p.address] = p
            logger.info("profiles_loaded", count=len(profiles))

    async def save_profiles_to_db(self):
        """Persist wallet profiles to database."""
        async with async_session() as session:
            for addr, profile in self.wallet_profiles.items():
                if isinstance(profile, WalletProfile):
                    existing = await session.get(WalletProfile, addr)
                    if existing:
                        existing.win_rate = profile.win_rate
                        existing.total_trades = profile.total_trades
                        existing.total_pnl = profile.total_pnl
                        existing.avg_position_size = profile.avg_position_size
                        existing.size_pattern = profile.size_pattern
                        existing.timing_pattern = profile.timing_pattern
                        existing.cluster_id = profile.cluster_id
                        existing.last_active = profile.last_active
                    else:
                        session.add(profile)
            await session.commit()

    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================

    async def run_analysis_cycle(self, market_ids: list[str]) -> list[SmartMoneySignal]:
        """
        Run full wallet intelligence cycle for given markets.
        Returns smart money signals for each market.
        """
        signals = []
        semaphore = asyncio.Semaphore(5)

        async def _analyze(mid):
            async with semaphore:
                signal = await self.detect_smart_money_flow(mid)
                await asyncio.sleep(0.1)
                return signal

        tasks = [_analyze(mid) for mid in market_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SmartMoneySignal):
                signals.append(result)

        logger.info("wallet_analysis_complete", markets=len(market_ids), signals=len(signals))
        return signals

    # =========================================================================
    # v3: TOP 50 WALLET REAL-TIME COPY
    # =========================================================================

    async def get_top_wallets(self, min_winrate: float = 0.75, min_trades: int = 30) -> list[str]:
        """Get top 50 most profitable wallet addresses."""
        top_wallets = []
        for addr, profile in self.wallet_profiles.items():
            if hasattr(profile, 'win_rate') and hasattr(profile, 'total_trades'):
                if profile.win_rate >= min_winrate and profile.total_trades >= min_trades:
                    top_wallets.append((addr, profile.win_rate, profile.total_pnl))

        # Sort by PnL then win rate
        top_wallets.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return [addr for addr, _, _ in top_wallets[:50]]

    async def scan_top_wallet_activity(self) -> list[SmartMoneySignal]:
        """
        v3: Scan TOP 50 wallets for recent activity.
        If a top wallet enters a position → generate copy signal.
        Much more targeted than scanning all 14k wallets.
        """
        top_addresses = await self.get_top_wallets()
        if not top_addresses:
            return []

        signals = []

        try:
            # Fetch recent activity from top wallets
            for addr in top_addresses[:20]:  # Check top 20 for rate limiting
                resp = await self.http_client.get(
                    f"{GAMMA_API}/activity",
                    params={"address": addr, "limit": 5}  # Last 5 trades
                )
                if resp.status_code != 200:
                    continue

                activities = resp.json()
                if not activities:
                    continue

                # Check if activity is recent (last 1 hour)
                for act in activities:
                    timestamp = act.get("timestamp", 0)
                    if isinstance(timestamp, (int, float)):
                        age_seconds = (datetime.utcnow() - datetime.utcfromtimestamp(timestamp)).total_seconds()
                    else:
                        age_seconds = 7200  # Default: too old

                    if age_seconds > 3600:  # Older than 1 hour → skip
                        continue

                    market_id = act.get("market", "") or act.get("conditionId", "")
                    side = act.get("side", "").upper()
                    size = float(act.get("size", 0) or 0)

                    if not market_id or not side or size < 10:
                        continue

                    # This is a FRESH trade from a top wallet → generate copy signal
                    profile = self.wallet_profiles.get(addr)
                    wr = profile.win_rate if hasattr(profile, 'win_rate') else 0.7

                    direction = "YES" if side in ("BUY", "YES") else "NO"

                    signal = SmartMoneySignal(
                        market_id=market_id,
                        direction=direction,
                        wallet_consensus=0.9,  # Single top wallet = high confidence
                        avg_wallet_winrate=wr,
                        total_volume=size,
                        num_wallets=1,
                        confidence=min(wr * 0.9, 0.85),  # Scale by win rate
                    )
                    signals.append(signal)

                    logger.info(
                        "top_wallet_copy_signal",
                        wallet=addr[:10],
                        market=market_id[:12],
                        direction=direction,
                        size=f"${size:.0f}",
                        winrate=f"{wr:.0%}",
                    )

                await asyncio.sleep(0.2)  # Rate limit

        except Exception as e:
            logger.warning("top_wallet_scan_error", error=str(e))

        logger.info("top_wallet_scan_complete", top_wallets=len(top_addresses), signals=len(signals))
        return signals


# Singleton
wallet_intelligence = WalletIntelligence()
