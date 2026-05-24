"""
PolyBotKing Main Entry Point
==============================
Starts the full autonomous trading system.
Designed for 24/7 VPS operation with graceful shutdown.
"""

import asyncio
import signal
import threading
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables BEFORE importing anything that touches settings
load_dotenv()

from polybotking.config import settings  # noqa: E402
from polybotking.dashboard import run_dashboard  # noqa: E402
from polybotking.engines.execution import execution_engine  # noqa: E402
from polybotking.engines.exit_optimizer import exit_optimizer  # noqa: E402
from polybotking.engines.telegram_alerts import telegram_alerts  # noqa: E402
from polybotking.engines.websocket_feed import websocket_feed  # noqa: E402
from polybotking.logger import get_logger, setup_logging  # noqa: E402
from polybotking.models import init_db  # noqa: E402
from polybotking.orchestrator import orchestrator  # noqa: E402

logger = get_logger("main")

BANNER = r"""
+--------------------------------------------------------------+
|                                                              |
|              PolyBotKing v3.0 - Autonomous Trading           |
|        Kelly Criterion - Sentiment AI - Smart Money          |
|              $5 -> $2,000 Target - 70-85% WR                 |
|                                                              |
+--------------------------------------------------------------+
"""


def _start_dashboard_thread(port: int = 8080) -> threading.Thread:
    """Start the embedded health/status HTTP server in a background thread."""
    thread = threading.Thread(
        target=run_dashboard,
        kwargs={"port": port},
        name="polybotking-dashboard",
        daemon=True,
    )
    thread.start()
    return thread


class PolyBotKing:
    """Main bot controller for 24/7 VPS operation."""

    def __init__(self) -> None:
        self._shutdown_event: asyncio.Event | None = None
        self._health_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._ws_task: asyncio.Task | None = None
        self._dashboard_thread: threading.Thread | None = None

    async def run(self) -> None:
        """Start the full trading system."""
        print(BANNER)
        setup_logging()
        self._shutdown_event = asyncio.Event()

        logger.info(
            "polybotking_starting",
            version="3.0.0",
            timestamp=datetime.utcnow().isoformat(),
        )
        logger.info(
            "config_loaded",
            initial_bankroll=settings.risk.initial_bankroll,
            kelly_fraction=settings.risk.kelly_fraction,
            scan_interval=settings.trading.scan_interval_seconds,
            max_positions=settings.trading.max_concurrent_positions,
            target_winrate=(
                f"{settings.trading.target_winrate_min:.0%}"
                f"-{settings.trading.target_winrate_max:.0%}"
            ),
        )

        # Start the embedded health/status dashboard for docker healthchecks.
        try:
            self._dashboard_thread = _start_dashboard_thread(port=8080)
            logger.info("dashboard_thread_started", port=8080)
        except Exception as e:  # pragma: no cover - dashboard is best-effort
            logger.warning("dashboard_thread_failed", error=str(e))

        # Geo-block check (best-effort: do NOT abort if endpoint is unreachable
        # since some VPS providers block outbound to that endpoint until DNS is up).
        geo_ip = "unknown"
        geo_country = "??"
        geo_blocked = False

        logger.info("checking_geo_status")
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://clob.polymarket.com/geo")
                if resp.status_code == 200:
                    geo_data = resp.json()
                    geo_ip = str(geo_data.get("ip", "unknown"))
                    geo_country = str(geo_data.get("country", "??"))
                    geo_blocked = bool(geo_data.get("blocked", False))

                    if geo_blocked:
                        logger.error(
                            "GEO_BLOCKED",
                            ip=geo_ip,
                            country=geo_country,
                            msg="VPS IP blocked by Polymarket. Change VPS location.",
                        )
                        print(f"\nBLOCKED! IP {geo_ip} country={geo_country}")
                        print(
                            "   Polymarket memblokir lokasi ini. "
                            "Ganti VPS ke negara yang diizinkan."
                        )
                        return

                    logger.info(
                        "geo_check_passed",
                        ip=geo_ip,
                        country=geo_country,
                        blocked=False,
                    )
                    print(
                        f"   GEO OK: IP={geo_ip} Country={geo_country} "
                        "Blocked=False"
                    )
                else:
                    logger.warning(
                        "geo_check_non_200",
                        status=resp.status_code,
                        msg="Continuing anyway",
                    )
        except Exception as e:
            logger.warning(
                "geo_check_failed",
                error=str(e),
                msg="Continuing anyway",
            )

        # Initialize database
        logger.info("initializing_database")
        await init_db()

        # Start execution engine
        await execution_engine.start()

        # Start v2 engines
        await exit_optimizer.start()
        await telegram_alerts.start()

        # Start orchestrator with execution callback
        await orchestrator.start(
            execution_callback=execution_engine.execute_trade,
        )

        # Start background tasks
        self._health_task = asyncio.create_task(self._health_check_loop())
        self._monitor_task = asyncio.create_task(self._position_monitor_loop())

        # Start WebSocket feed (real-time price updates)
        if settings.trading.enable_websocket:
            await websocket_feed.start()
            self._ws_task = asyncio.create_task(websocket_feed.run_forever())

        # Send Telegram notification
        try:
            await telegram_alerts.bot_started(
                bankroll=settings.risk.initial_bankroll,
                ip=geo_ip,
                country=geo_country,
            )
        except Exception as e:
            logger.warning("telegram_start_alert_failed", error=str(e))

        logger.info(
            "polybotking_running",
            msg="All systems operational. Entering main loop.",
        )

        # Main trading loop
        try:
            await orchestrator.run_continuous()
        except asyncio.CancelledError:
            logger.info("main_loop_cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("polybotking_shutting_down")

        # Cancel background tasks
        for task in (self._health_task, self._monitor_task, self._ws_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Stop orchestrator (stops all engines)
        try:
            await orchestrator.stop()
        except Exception as e:
            logger.warning("orchestrator_stop_error", error=str(e))

        try:
            await execution_engine.stop()
        except Exception as e:
            logger.warning("execution_stop_error", error=str(e))

        try:
            await websocket_feed.stop()
        except Exception as e:
            logger.warning("websocket_stop_error", error=str(e))

        try:
            await telegram_alerts.stop()
        except Exception as e:
            logger.warning("telegram_stop_error", error=str(e))

        logger.info("polybotking_shutdown_complete")

    async def _health_check_loop(self) -> None:
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

    async def _position_monitor_loop(self) -> None:
        """Monitor and update active positions + exit optimization."""
        while True:
            try:
                await asyncio.sleep(60)

                # Update position prices
                await execution_engine.update_positions()

                # Check for resolved markets
                await execution_engine.check_resolved_markets()

                # Exit optimization (take profit / trailing stop)
                if execution_engine.active_positions:
                    exit_decisions = exit_optimizer.analyze_all_positions(
                        execution_engine.active_positions
                    )
                    for exit_decision in exit_decisions:
                        if exit_decision.urgency == "immediate":
                            result = await execution_engine.close_position(
                                exit_decision.market_id,
                                reason=exit_decision.action,
                            )
                            if result and result.success:
                                try:
                                    await telegram_alerts.trade_closed(
                                        market_id=exit_decision.market_id,
                                        pnl=exit_decision.unrealized_pnl_pct * 100,
                                        bankroll=0,
                                        win_rate=0,
                                        reason=exit_decision.action,
                                    )
                                except Exception as alert_err:
                                    logger.warning(
                                        "telegram_close_alert_failed",
                                        error=str(alert_err),
                                    )
                                exit_optimizer.clear_peak(exit_decision.market_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("position_monitor_error", error=str(e))


def main() -> None:
    """Entry point. Uses asyncio.run() for clean lifecycle management."""
    bot = PolyBotKing()

    # asyncio.run() owns the loop and handles cleanup
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):  # noqa: ARG001
        logger.info("signal_received", signal=sig)
        loop.call_soon_threadsafe(loop.stop)

    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal can only be installed in main thread; ignore otherwise
        pass

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
        loop.run_until_complete(bot.shutdown())
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            pass
        loop.close()


if __name__ == "__main__":
    main()
