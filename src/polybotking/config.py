"""
Configuration management using pydantic-settings.
All parameters loaded from .env file or environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class PolymarketConfig(BaseSettings):
    """Polymarket CLOB API configuration."""
    api_key: str = Field(default="", alias="POLY_API_KEY")
    api_secret: str = Field(default="", alias="POLY_API_SECRET")
    api_passphrase: str = Field(default="", alias="POLY_API_PASSPHRASE")
    private_key: str = Field(default="", alias="POLY_PRIVATE_KEY")
    chain_id: int = Field(default=137, alias="POLY_CHAIN_ID")


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    url: str = Field(
        default="sqlite+aiosqlite:///data/polybotking.db",
        alias="DATABASE_URL"
    )


class TwitterConfig(BaseSettings):
    """Twitter/X API configuration."""
    bearer_token: str = Field(default="", alias="TWITTER_BEARER_TOKEN")
    api_key: str = Field(default="", alias="TWITTER_API_KEY")
    api_secret: str = Field(default="", alias="TWITTER_API_SECRET")


class NewsConfig(BaseSettings):
    """News API configuration."""
    api_key: str = Field(default="", alias="NEWS_API_KEY")


class AIConfig(BaseSettings):
    """AI/LLM configuration."""
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")


class RiskConfig(BaseSettings):
    """Risk and position sizing parameters."""
    initial_bankroll: float = Field(default=5.0, alias="INITIAL_BANKROLL")
    max_position_size_pct: float = Field(default=0.15, alias="MAX_POSITION_SIZE_PCT")
    max_drawdown_pct: float = Field(default=0.25, alias="MAX_DRAWDOWN_PCT")
    kelly_fraction: float = Field(default=0.25, alias="KELLY_FRACTION")
    min_edge_threshold: float = Field(default=0.05, alias="MIN_EDGE_THRESHOLD")
    min_ev_threshold: float = Field(default=0.02, alias="MIN_EV_THRESHOLD")


class TradingConfig(BaseSettings):
    """Trading parameters."""
    scan_interval_seconds: int = Field(default=30, alias="SCAN_INTERVAL_SECONDS")
    max_concurrent_positions: int = Field(default=10, alias="MAX_CONCURRENT_POSITIONS")
    market_timeframe_min_hours: int = Field(default=1, alias="MARKET_TIMEFRAME_MIN_HOURS")
    market_timeframe_max_days: int = Field(default=7, alias="MARKET_TIMEFRAME_MAX_DAYS")
    target_winrate_min: float = Field(default=0.70, alias="TARGET_WINRATE_MIN")
    target_winrate_max: float = Field(default=0.85, alias="TARGET_WINRATE_MAX")


class AlertsConfig(BaseSettings):
    """Alerting/notification configuration."""
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_telegram: bool = Field(default=False, alias="ENABLE_TELEGRAM_ALERTS")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")


class Settings:
    """Master settings container."""

    def __init__(self):
        self.polymarket = PolymarketConfig()
        self.database = DatabaseConfig()
        self.twitter = TwitterConfig()
        self.news = NewsConfig()
        self.ai = AIConfig()
        self.risk = RiskConfig()
        self.trading = TradingConfig()
        self.alerts = AlertsConfig()

    @property
    def data_dir(self) -> Path:
        path = Path("data")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        path = Path("logs")
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global singleton
settings = Settings()
