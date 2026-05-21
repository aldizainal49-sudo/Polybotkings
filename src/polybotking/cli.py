"""
PolyBotKing CLI
===============
Command-line interface for managing the trading bot.
"""

import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="PolyBotKing")
def main():
    """PolyBotKing - Autonomous Polymarket Trading Bot"""
    pass


@main.command()
def run():
    """Start the bot in continuous trading mode (24/7)."""
    from polybotking.main import PolyBotKing
    bot = PolyBotKing()
    asyncio.run(bot.run())


@main.command()
def status():
    """Show current bot status and performance."""
    asyncio.run(_show_status())


async def _show_status():
    from polybotking.models import init_db, async_session, Trade, TradeStatus, BotState
    from sqlalchemy import select, func

    await init_db()

    async with async_session() as session:
        # Get risk state
        state = await session.get(BotState, "risk_state")
        risk_data = state.value if state else {}

        # Get trade stats
        total_trades = await session.scalar(select(func.count(Trade.id)))
        open_trades = await session.scalar(
            select(func.count(Trade.id)).where(Trade.status == TradeStatus.OPEN)
        )
        won_trades = await session.scalar(
            select(func.count(Trade.id)).where(Trade.pnl > 0)
        )
        total_pnl = await session.scalar(select(func.sum(Trade.pnl))) or 0

    # Display
    table = Table(title="PolyBotKing Status", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bankroll", f"${risk_data.get('current_bankroll', 5.0):.2f}")
    table.add_row("Total PnL", f"${total_pnl:.2f}")
    table.add_row("Win Rate", f"{risk_data.get('win_rate', 0):.1%}")
    table.add_row("Total Trades", str(total_trades or 0))
    table.add_row("Open Positions", str(open_trades or 0))
    table.add_row("Won Trades", str(won_trades or 0))
    table.add_row("Drawdown", f"{risk_data.get('current_drawdown', 0):.1%}")
    table.add_row("Kelly Multiplier", f"{risk_data.get('kelly_multiplier', 0.25):.2f}")
    table.add_row("Circuit Breaker", "ACTIVE" if risk_data.get('is_circuit_breaker_active') else "OFF")

    console.print(table)


@main.command()
def scan():
    """Run a single market scan cycle."""
    asyncio.run(_run_scan())


async def _run_scan():
    from polybotking.models import init_db
    from polybotking.engines.market_scanner import market_scanner

    await init_db()
    await market_scanner.start()

    console.print("[bold cyan]Running market scan...[/bold cyan]")
    opportunities = await market_scanner.run_scan_cycle()

    table = Table(title=f"Market Opportunities ({len(opportunities)} found)")
    table.add_column("Market", max_width=40)
    table.add_column("Dir", style="bold")
    table.add_column("Edge", style="green")
    table.add_column("EV", style="green")
    table.add_column("Confidence")
    table.add_column("Type")

    for opp in opportunities[:20]:
        table.add_row(
            opp.question[:40] if opp.question else opp.market_id[:12],
            opp.direction,
            f"{opp.edge:.3f}",
            f"{opp.ev:.3f}",
            f"{opp.confidence:.2f}",
            opp.signal_type.value,
        )

    console.print(table)
    await market_scanner.stop()


@main.command()
def backtest():
    """Run backtesting on historical data."""
    console.print("[yellow]Backtesting module - Coming soon[/yellow]")
    console.print("Use historical trade data to validate strategy performance.")


@main.command()
@click.option("--address", "-a", help="Wallet address to profile")
def wallet(address):
    """Analyze a specific wallet."""
    if not address:
        console.print("[red]Please provide a wallet address with --address[/red]")
        return
    asyncio.run(_analyze_wallet(address))


async def _analyze_wallet(address: str):
    from polybotking.models import init_db
    from polybotking.engines.wallet_intelligence import wallet_intelligence

    await init_db()
    await wallet_intelligence.start()

    console.print(f"[cyan]Analyzing wallet: {address[:12]}...[/cyan]")
    profile = await wallet_intelligence.build_wallet_profile(address)

    if profile:
        table = Table(title=f"Wallet Profile: {address[:16]}...")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Win Rate", f"{profile.win_rate:.1%}")
        table.add_row("Total Trades", str(profile.total_trades))
        table.add_row("Total PnL", f"${profile.total_pnl:.2f}")
        table.add_row("Avg Position Size", f"${profile.avg_position_size:.2f}")
        table.add_row("Size Pattern", profile.size_pattern)
        table.add_row("Timing Pattern", profile.timing_pattern)

        console.print(table)
    else:
        console.print("[red]Could not build profile for this wallet[/red]")

    await wallet_intelligence.stop()


@main.command()
def config():
    """Show current configuration."""
    from polybotking.config import settings

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Initial Bankroll", f"${settings.risk.initial_bankroll}")
    table.add_row("Kelly Fraction", f"{settings.risk.kelly_fraction}")
    table.add_row("Max Position %", f"{settings.risk.max_position_size_pct:.0%}")
    table.add_row("Max Drawdown %", f"{settings.risk.max_drawdown_pct:.0%}")
    table.add_row("Min Edge", f"{settings.risk.min_edge_threshold}")
    table.add_row("Min EV", f"{settings.risk.min_ev_threshold}")
    table.add_row("Scan Interval", f"{settings.trading.scan_interval_seconds}s")
    table.add_row("Max Positions", str(settings.trading.max_concurrent_positions))
    table.add_row("Market Window", f"{settings.trading.market_timeframe_min_hours}h - {settings.trading.market_timeframe_max_days}d")
    table.add_row("Target Winrate", f"{settings.trading.target_winrate_min:.0%} - {settings.trading.target_winrate_max:.0%}")
    table.add_row("Telegram Alerts", "ON" if settings.alerts.enable_telegram else "OFF")

    console.print(table)


if __name__ == "__main__":
    main()
