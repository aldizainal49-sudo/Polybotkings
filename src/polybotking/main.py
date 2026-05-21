"""
PolyBotKing Main Entry Point
==============================
Starts the full autonomous trading system.
Designed for 24/7 VPS operation with graceful shutdown.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from polybotking.config import settings
from polybotking.logger import setup_logging, get_logger
from polybotking.models import init_db
from polybotking.orchestrator import orchestrator
from polybotking.engines.execution import execution_engine

logger = get_logger("main")

# Banner
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ██████╗  ██████╗ ██╗  ██╗   ██╗██████╗  ██████╗ ████████║
║     ██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝██╔══██╗██╔═══██╗╚══██╔═║
║     ██████╔╝██║   ██║██║   ╚████╔╝ ██████╔╝██║   ██║   ██║  ║
║     ██╔═══╝ ██║   ██║██║    ╚██╔╝  ██╔══██╗██║   ██║   ██║  ║
║     ██║     ╚██████╔╝███████╗██║   ██████╔╝╚██████╔╝   ██║  ║
║     ╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═════╝  ╚═════╝    ╚═╝  ║
║                                                              ║
║           PolyBotKing v1.0 — Autonomous Trading              ║
║       Kelly Criterion • Sentiment AI • Smart Money           ║
║              $5 → $2,000 Target • 70-85% WR                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


class PolyBotKing:
    """Main bot controller for 24/7 VPS operation."""

    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._health_task = None
        self._monitor_task = None

    async def run(self):
        """Start the full trading system."""
        print(BANNER)
        setup_logging()

        logger.info("polybotking_starting", version="1.0.0", timestamp=datetime.utcnow().isoformat())
        logger.info("config_loaded",
                   initial_bankroll=settings.risk.initial_bankroll,
                   kelly_fraction=settings.risk.kelly_fraction,
                   scan_interval=settings.trading.scan_interval_seconds,
                   max_positions=settings.trading.max_concurrent_positions,
                   target_winrate=f"{settings.trading.target_winrate_min:.0%}-{settings.trading.target_winrate_max:.0%}")

        # Check geo-block status before starting
        logger.info("checking_geo_status")
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://clob.polymarket.com/geo")
                geo_data = resp.json()
                if geo_data.get("blocked"):
                    logger.error(
                        "GEO_BLOCKED",
                        ip=geo_data.get("ip"),
                        country=geo_data.get("country"),
                        msg="VPS IP is blocked by Polymarket! Change VPS location."
                    )
                    print(f"\n❌ BLOCKED! IP {geo_data.get('ip')} country={geo_data.get('country')}")
                    print("   Polymarket memblokir lokasi ini. Ganti VPS ke negara yang diizinkan.")
                    return
                else:
                    logger.info(
                        "geo_check_passed",
                        ip=geo_data.get("ip"),
                        country=geo_data.get("country"),
                        blocked=False,
                    )
                    print(f"   ✅ GEO OK: IP={geo_data.get('ip')} Country={geo_data.get('country')} Blocked=False")
        except Exception as e:
            logger.warning("geo_check_failed", error=str(e), msg="Continuing anyway...")

        # Initialize database
        logger.info("initializing_database")
        await init_db()

        # Start execution engine
        await execution_engine.start()

        # Start orchestrator with execution callback
        await orchestrator.start(execution_callback=execution_engine.execute_trade)

        # Start background tasks
        self._health_task = asyncio.create_task(self._health_check_loop())
        self._monitor_task = asyncio.create_task(self._position_monitor_loop())

        logger.info("polybotking_running", msg="All systems operational. Entering main loop.")

        # Main trading loop
        try:
            await orchestrator.run_continuous()
        except asyncio.CancelledError:
            logger.info("main_loop_cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("polybotking_shutting_down")

        # Cancel background tasks
        if self._health_task:
            self._health_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()

        # Stop orchestrator (stops all engines)
        await orchestrator.stop()
        await execution_engine.stop()

        logger.info("polybotking_shutdown_complete")

    async def _health_check_loop(self):
        """Periodic health check and status reporting."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                from polybotking.engines.risk_engine import risk_engine

                if risk_engine.state:
                    logger.info(
                        "health_check",
                        bankroll=f"${risk_engine.state.current_bankroll:.2f}",
                        pnl=f"${risk_engine.state.total_pnl:.2f}",
                        win_rate=f"{risk_engine.state.win_rate:.1%}",
                        drawdown=f"{risk_engine.state.current_drawdown:.1%}",
                        open_positions=len(execution_engine.active_positions),
                        cycles=orchestrator.state.cycles_completed,
                        executions=orchestrator.state.total_executions,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))

    async def _position_monitor_loop(self):
        """Monitor and update active positions."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                # Update position prices
                await execution_engine.update_positions()

                # Check for resolved markets
                await execution_engine.check_resolved_markets()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("position_monitor_error", error=str(e))


def main():
    """Entry point."""
    bot = PolyBotKing()

    # Handle signals for graceful shutdown
    loop = asyncio.new_event_loop()

    def signal_handler(sig, frame):
        logger.info("signal_received", signal=sig)
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
