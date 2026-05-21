# PolyBotKing 👑

> Autonomous Polymarket Trading Bot | Multi-Agent Orchestration | $5 → $2,000 Target

**PolyBotKing** is a fully autonomous trading bot designed to run 24/7 on a VPS, targeting **70-85% win rate** on Polymarket prediction markets (1-hour to 7-day timeframes).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                                   │
│                                                                       │
│  Event Ingestion → CLOB Snapshot → News Fetch → AI Inference         │
│       → EV Gate → Position Sizing → Execution                        │
└──────────┬──────────────┬──────────────┬──────────────┬──────────────┘
           │              │              │              │
    ┌──────▼──────┐ ┌────▼─────┐ ┌─────▼─────┐ ┌─────▼──────┐
    │   MARKET    │ │  WALLET  │ │ SENTIMENT │ │ VOLATILITY │
    │  SCANNER    │ │  INTEL   │ │    AI     │ │   TIMING   │
    ├─────────────┤ ├──────────┤ ├───────────┤ ├────────────┤
    │ CLOB API    │ │14k+wallets│ │ X/Twitter │ │Vol regimes │
    │ Mispricing  │ │Clustering│ │ News RSS  │ │ Momentum   │
    │ Pair Arb    │ │WinRate   │ │ NLP Score │ │ RSI-like   │
    │ EV Scoring  │ │SmartMoney│ │ Divergence│ │Entry/Exit  │
    └─────────────┘ └──────────┘ └───────────┘ └────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    RISK ENGINE     │
                    ├────────────────────┤
                    │ Kelly Criterion    │
                    │ Fractional Kelly   │
                    │ Bayesian Learning  │
                    │ Circuit Breakers   │
                    │ Drawdown Protection│
                    └────────────────────┘
```

---

## Three Core Engines

### 1. Kelly Criterion + Bayesian Learning
- Optimal position sizing that **gets smarter over time**
- Dynamic Kelly fraction adjusted by win/loss streaks
- Probability recalibration from every trade outcome
- The more it trades, the more accurate its sizing becomes

### 2. Pair Arbitrage
- Detects mispricing: `YES 0.62 + NO 0.41 = 1.03` (overpriced edge)
- Scans all markets for `YES + NO ≠ 1.0` opportunities
- Fee-adjusted edge calculation with minimum thresholds

### 3. Sentiment AI + Volatility Timing
- Scans X/Twitter + crypto news **BEFORE** market adjusts
- NLP-powered sentiment scoring with market-specific keywords
- Enters positions before volatility spikes, exits during moves
- Regime detection (low → medium → high → extreme)

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Agent Orchestration | 5 specialized engines coordinated by central orchestrator |
| Smart Money Detection | Tracks 14,000+ wallet behaviors, clusters by strategy |
| Kelly Criterion Sizing | Mathematically optimal bet sizing, improves with data |
| Sentiment AI | Real-time NLP on Twitter/news before price adjusts |
| Volatility Timing | Enter during low-vol, exit during high-vol regimes |
| Pair Arbitrage | Detect YES+NO mispricing for risk-free edges |
| Drawdown Protection | Circuit breakers, max drawdown limits, streak detection |
| Bayesian Learning | Every outcome updates probability estimates |
| 24/7 VPS Operation | Docker deployment with health checks and auto-restart |
| Dynamic Weights | Engine weights auto-adjust based on historical accuracy |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for VPS deployment)
- Polymarket API credentials
- Twitter API bearer token (for sentiment)

### 1. Clone & Setup

```bash
git clone https://github.com/your-repo/polybotking.git
cd polybotking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
nano .env
```

**Required keys:**
- `POLY_API_KEY`, `POLY_API_SECRET`, `POLY_API_PASSPHRASE` — Polymarket CLOB API
- `POLY_PRIVATE_KEY` — Your Ethereum wallet private key (Polygon)
- `TWITTER_BEARER_TOKEN` — For sentiment scanning

### 3. Run

```bash
# Start trading bot
polybotking run

# Or run a single scan
polybotking scan

# Check status
polybotking status

# View config
polybotking config
```

---

## VPS Deployment (24/7)

### Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f polybotking

# Check health
curl http://localhost:8080/health

# Stop
docker-compose down
```

### Systemd Service (Alternative)

```bash
# Create service file
sudo nano /etc/systemd/system/polybotking.service
```

```ini
[Unit]
Description=PolyBotKing Trading Bot
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/opt/polybotking
ExecStart=/opt/polybotking/.venv/bin/polybotking run
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable polybotking
sudo systemctl start polybotking
sudo journalctl -u polybotking -f
```

---

## Pipeline Flow

```
Every 30 seconds:

1. EVENT INGESTION
   └── Fetch active markets from Polymarket CLOB API
   └── Filter: 1hr-7day resolution timeframe

2. CLOB SNAPSHOT
   └── Orderbook depth, spread, volume for top markets
   └── Detect pair arbitrage (YES + NO mispricing)

3. NEWS FETCH + SENTIMENT
   └── Scan Twitter/X for market keywords
   └── Scan crypto news RSS feeds
   └── NLP sentiment scoring + divergence detection

4. WALLET INTELLIGENCE
   └── Smart money flow detection
   └── Cross-reference with 14k+ profiled wallets
   └── Cluster consensus signals

5. AI INFERENCE
   └── Combine all engine signals (weighted voting)
   └── Recalibrate true probability
   └── Dynamic weight adjustment from calibration

6. EV GATE
   └── Filter: minimum edge > 5%, minimum EV > 2%
   └── Sort by EV descending
   └── Limit to max concurrent positions

7. KELLY SIZING
   └── Full Kelly calculation
   └── Fractional Kelly with dynamic multiplier
   └── Confidence-adjusted probability
   └── Drawdown-aware position limits

8. EXECUTION
   └── Place limit order on Polymarket CLOB
   └── Track fills and manage positions
   └── Monitor for market resolution
```

---

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_BANKROLL` | $5.00 | Starting capital |
| `KELLY_FRACTION` | 0.25 | Fraction of full Kelly (conservative) |
| `MAX_POSITION_SIZE_PCT` | 15% | Max single position as % of bankroll |
| `MAX_DRAWDOWN_PCT` | 25% | Circuit breaker trigger |
| `MIN_EDGE_THRESHOLD` | 5% | Minimum edge to consider a trade |
| `MIN_EV_THRESHOLD` | 2% | Minimum EV to execute |
| `MAX_CONCURRENT_POSITIONS` | 10 | Position diversification |

### Circuit Breakers
- **Max drawdown** (25%): All trading paused
- **5+ consecutive losses**: Trading paused
- **Bankroll < 20% of peak**: Emergency stop
- Auto-reactivation when conditions improve

---

## CLI Commands

```bash
polybotking run       # Start 24/7 trading
polybotking scan      # Single market scan
polybotking status    # Performance dashboard
polybotking config    # View configuration
polybotking wallet -a 0x...  # Analyze a wallet
polybotking backtest  # Historical validation (coming soon)
```

---

## Monitoring

- **Health endpoint**: `GET /health` → `{"status": "healthy"}`
- **Status endpoint**: `GET /status` → Full bot status JSON
- **Metrics endpoint**: `GET /metrics` → Prometheus format
- **Telegram alerts**: Real-time trade notifications (optional)
- **Structured logging**: JSON logs in `logs/` directory

---

## How It Gets Smarter

1. **Every trade outcome** → Updates Bayesian probability estimates
2. **Win rate tracking** → Dynamically adjusts Kelly multiplier
3. **Signal accuracy** → Weights engines that perform best
4. **Edge calibration** → Per-signal-type accuracy tracking
5. **Streak detection** → Increases/decreases aggression
6. **Wallet clustering** → Identifies new winning patterns

---

## Project Structure

```
polybotking/
├── src/polybotking/
│   ├── __init__.py          # Package info
│   ├── config.py            # Pydantic settings
│   ├── models.py            # SQLAlchemy models
│   ├── logger.py            # Structured logging
│   ├── orchestrator.py      # Multi-agent coordinator
│   ├── main.py              # 24/7 runner
│   ├── cli.py               # Click CLI
│   ├── dashboard.py         # Health HTTP server
│   └── engines/
│       ├── market_scanner.py      # CLOB + mispricing
│       ├── wallet_intelligence.py # Smart money tracking
│       ├── sentiment_ai.py        # NLP + news scanning
│       ├── volatility_timing.py   # Vol regime + timing
│       ├── risk_engine.py         # Kelly + risk mgmt
│       └── execution.py           # Order placement
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Disclaimer

This software is for educational and research purposes. Trading prediction markets involves significant risk. Past performance does not guarantee future results. Never trade with money you cannot afford to lose. The 70-85% win rate target and $5→$2,000 growth are goals, not guarantees.

---

## License

MIT
