# Crypto Oracle

A fine-tuned Qwen2.5-32B language model for cryptocurrency trading recommendations, with a deterministic risk management layer, historical backtesting, paper trading, and a continual learning pipeline.

---

## Overview

Crypto Oracle combines a locally-hosted fine-tuned LLM with a rule-based safety layer to generate structured trading signals (BUY / SELL / HOLD) for cryptocurrency pairs. It is designed for self-hosted deployment — your model, your data, no third-party AI calls in the trading loop.

**Architecture:**

```
Market Data (CCXT + yfinance + sentiment)
        │
        ▼
Fine-tuned Qwen2.5-32B (Unsloth LoRA)
        │
        ▼
Deterministic Rule Layer (RiskManager)
  - Signal quality gate (confidence, R/R ratio)
  - Portfolio limits (max positions, circuit breakers)
  - Macro overrides (VIX > 40 blocks longs)
  - Fixed fractional position sizing (1% risk/trade)
        │
        ▼
LiveTrader  (--mode paper | dry | live)
  paper: simulated fills, no exchange
  dry:   real prices, logs only
  live:  limit orders + postOnly on Coinbase Advanced Trade
         fill polling, partial fill handling, 120s timeout
        │
        ▼
SQLite Trade Journal  +  Telegram/Discord Alerts
        │
        ▼
Continual Learning (closed trades → retrain)
```

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24GB+ VRAM for inference (44GB+ for training Qwen2.5-32B)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~100GB for model weights + training data

> Tested on NVIDIA RTX A6000 (48GB VRAM), Windows 11 with CUDA 12.6.
> **vLLM is not supported on Windows.** Inference uses Unsloth directly.

### Software
- Python 3.10+
- CUDA 12.x + cuDNN
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/crypto-oracle.git
cd crypto-oracle
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

**Inference + API (no GPU training):**
```bash
pip install -r requirements.txt
```

**Training (requires CUDA, installs Unsloth + TRL):**
```bash
pip install -r requirements-train.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

See [Configuration](#configuration) for details on each key.

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Description |
|----------|----------|-------------|
| `COINBASE_API_KEY` | Yes | Coinbase Advanced Trade API key (org ID) |
| `COINBASE_SECRET_KEY` | Yes | Coinbase EC private key (multi-line PEM) |
| `CRYPTOPANIC_API_KEY` | No | CryptoPanic news API key (free tier works) |
| `REDDIT_CLIENT_ID` | No | Reddit app client ID (for sentiment) |
| `REDDIT_CLIENT_SECRET` | No | Reddit app client secret |
| `REDDIT_USER_AGENT` | No | Reddit user agent string |

**Getting a Coinbase API key:**
1. Go to [Coinbase Advanced Trade](https://advanced.coinbase.com/) → API → New API Key
2. Grant: View permissions for read-only data collection; Trade permissions only if executing live orders
3. Copy the key ID and download the private key PEM

**Getting a CryptoPanic key:**
- Register free at [cryptopanic.com/developers/api](https://cryptopanic.com/developers/api/)

---

## Data Collection

All data is stored locally in `data/` (git-ignored, must be collected before training or inference).

### Download historical OHLCV data

```bash
# Download 5+ years of candle data for all supported pairs
python src/data_collection/ohlcv_downloader.py \
  --pairs BTC/USD ETH/USD XRP/USD LTC/USD LINK/USD BCH/USD XLM/USD \
  --timeframes 1h 6h 1d \
  --since 2020-01-01

# Or run the full master collector (OHLCV + sentiment + macro + on-chain)
python src/data_collection/master_data_collector.py
```

### Build the training dataset

```bash
python src/data_collection/dataset_builder.py \
  --pairs BTC/USD ETH/USD XRP/USD LTC/USD LINK/USD BCH/USD XLM/USD \
  --timeframes 1h 6h 1d \
  --max-examples 5000

# Output: datasets/enhanced_sft_train.jsonl (~75K examples)
#         datasets/enhanced_sft_eval.jsonl  (~19K examples)
# Distribution: ~35% BUY / 33% HOLD / 32% SELL
```

---

## Training

> Training Qwen2.5-32B requires ~44GB VRAM and takes several days on a single A6000.
> The training script uses staged training with automatic checkpointing — safe to interrupt and resume.

```bash
# Start training (or resume from last checkpoint automatically)
python src/fine_tuning/train_qwen_crypto_oracle.py --train

# Recommended: run in tmux or screen to survive disconnects
# Windows CMD:
python src\fine_tuning\train_qwen_crypto_oracle.py --train > training.log 2>&1

# Linux / tmux:
python src/fine_tuning/train_qwen_crypto_oracle.py --train 2>&1 | tee training.log
```

**Resumption is automatic.** Stage checkpoints are saved to `models/crypto-oracle-qwen-32b/stage_N/adapter/`. If training is interrupted, re-running the same command resumes from the last completed stage. Mid-stage checkpoints (saved every 200 steps) are also used for intra-stage recovery.

**Training output:**
- `models/crypto-oracle-qwen-32b/stage_1/` through `stage_4/` — staged adapters
- `models/crypto-oracle-qwen-32b/final_model/` — final LoRA adapter + tokenizer

**Note:** GGUF conversion is skipped automatically on Windows (requires Linux + llama.cpp). To convert on Linux:
```python
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
```

---

## Testing

Run in this order before committing real capital:

### 1. Model quality evaluation

```bash
python src/testing/model_evaluator.py --model models/crypto-oracle-qwen-32b/final_model
```

Checks parse rate, confidence emission, stop/TP sanity, direction alignment across 5 standardized market scenarios. Hard gates: 100% parse rate, ≥80% stop/TP sanity.

### 2. Fast backtest (rule-based signals, no LLM)

```bash
python src/testing/backtester.py --pair BTC/USD --timeframe 1h --fast
```

Quick validation of the signal logic and exit mechanics on historical data. Runs in ~1 minute.

### 3. LLM backtest (actual model inference)

```bash
python src/testing/backtester.py --pair BTC/USD --timeframe 1h --model --bars 150
```

Samples 150 historical bars and runs the full LLM + rule layer pipeline on each. Takes ~30 minutes. Go/no-go gates: win rate ≥50%, profit factor >1.3, Sharpe >0.5, max drawdown <20%.

### 4. Paper trading (2–4 weeks minimum)

```bash
# Run every hour in a loop
python src/testing/paper_trader.py \
  --pairs BTC/USD ETH/USD XRP/USD \
  --loop --interval 3600

# Check paper portfolio status
python src/testing/paper_trader.py --status
```

Paper trading uses live market data against a simulated portfolio. All decisions are logged to `data/trades.db` and feed the continual learning pipeline.

---

## Docker Deployment

Docker is the recommended way to run the full system in production. It handles process management, log rotation, and GPU access automatically.

### Prerequisites

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host
- `.env` file configured (see [Configuration](#configuration))
- `models/` directory with trained LoRA adapter
- `data/` directory with downloaded OHLCV data

### Profiles

The compose file uses profiles to select what runs. **Pick one profile** — don't run all at once.

**Paper trading** (safe, no real money — start here):
```bash
docker-compose --profile paper up -d

# Monitor logs
docker-compose logs -f paper
docker-compose logs -f api

# Check paper portfolio
docker exec crypto-oracle-paper python src/testing/paper_trader.py --status
```

**Dry run** (real prices, logs "would place" — no orders placed):
```bash
docker-compose --profile dry up -d
docker-compose logs -f dry
```

**Live trading** (real money — read [Live Trading](#live-trading) first):
```bash
# .env must have LIVE_TRADING_ENABLED=true
docker-compose --profile live up -d
docker-compose logs -f live
docker-compose logs -f monitor
```

**Stop everything:**
```bash
docker-compose down
```

### Emergency kill-switch

```bash
# Pause new entries immediately (positions still monitored for SL/TP)
touch pause.flag

# Or via API:
curl -X POST http://localhost:8000/admin/pause

# Resume:
rm pause.flag
# Or via API:
curl -X POST http://localhost:8000/admin/resume
```

---

## Alerts & Notifications

Crypto Oracle sends alerts to Telegram and/or Discord when trades open, close, circuit breakers fire, or errors occur.

### Telegram setup

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → note the bot token
2. Start a chat with your new bot, then get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Add to `.env`:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Discord setup

1. Open your Discord server → **Server Settings** → **Integrations** → **Webhooks** → **New Webhook**
2. Copy the webhook URL
3. Add to `.env`:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

You can set up both — alerts fire to all configured channels simultaneously.

### Alert types

| Event | Message |
|-------|---------|
| Trade opened | Pair, direction, size, entry/SL/TP |
| Trade closed | Outcome, net P&L, reason (SL/TP/manual) |
| Trade rejected | Which rule blocked it |
| Drawdown alert | Current drawdown % vs threshold |
| System paused/resumed | Kill-switch state changes |
| Error | Unhandled exceptions in the trading loop |

---

## Live Trading

> **Read this section carefully before enabling live trading.**

The system supports three execution modes, controlled by the `--mode` flag:

| Mode | Exchange calls | Money at risk | Use for |
|------|---------------|---------------|---------|
| `paper` | None (simulated) | None | Initial validation (2–4 weeks min) |
| `dry` | Price fetches only | None | Integration testing, slippage estimates |
| `live` | Full order placement | Yes | Real trading |

### Recommended progression

**Step 1 — Paper trading (2–4 weeks minimum)**
```bash
docker-compose --profile paper up -d
```
Watch for: consistent trade generation, correct SL/TP monitoring, no crashes. Target: ≥50% win rate, profit factor >1.3 over 20+ trades.

**Step 2 — Dry run (a few days)**
```bash
docker-compose --profile dry up -d
```
Confirms: API key works, prices fetch correctly, logs look right. No orders placed.

**Step 3 — Sandbox (Coinbase test environment)**

Add to `.env`:
```
COINBASE_SANDBOX=true
LIVE_TRADING_ENABLED=true
```
```bash
docker-compose --profile live up -d
```
Tests real order placement flow on fake money. Verify fills, partial fill handling, and SL/TP closure.

**Step 4 — Tiny real test ($200–$1,000)**

Remove `COINBASE_SANDBOX=true`, keep `LIVE_TRADING_ENABLED=true`. Watch closely:
- Monitor logs in real time: `docker-compose logs -f live`
- Keep kill-switch ready: `touch pause.flag`
- Verify first fill notification arrives on Telegram/Discord
- Check open position in Coinbase UI matches DB: `python scripts/health_check.py`

**Only scale up after 2+ weeks of positive live results.**

### Safety features

- **Kill-switch** — `touch pause.flag` halts all new entries in seconds. Existing positions continue to be monitored.
- **Daily loss circuit breaker** — halts trading if daily loss exceeds 5% of equity (configurable)
- **Max drawdown circuit breaker** — halts trading if drawdown from peak exceeds 15%
- **VIX block** — no new long positions when VIX > 40
- **Slippage abort** — if fill deviates >1% from expected, trade is not tracked
- **Limit orders with postOnly** — maker-only fills, lower fees, no surprise market fills
- **Fill timeout** — if limit order doesn't fill within 120 seconds, it is cancelled

---

## Running the API

```bash
python src/api/api_server.py
# Server starts on http://localhost:8000
```

### Primary endpoint: POST /trade

Submit your current portfolio state and receive a trade decision:

```bash
curl -X POST http://localhost:8000/trade \
  -H "Content-Type: application/json" \
  -d '{
    "pair": "BTC/USD",
    "total_equity": 100000.0,
    "available_cash": 80000.0,
    "open_positions": [],
    "daily_pnl_usd": 0.0,
    "high_water_mark_equity": 100000.0
  }'
```

**Response:**
```json
{
  "approved": true,
  "direction": "BUY",
  "confidence": 78,
  "entry_price": 67420.50,
  "stop_loss": 65398.00,
  "take_profit": 70841.00,
  "position_size_usd": 3370.00,
  "risk_pct": 0.01,
  "reward_risk_ratio": 1.7,
  "applied_rules": ["fixed_fractional_sizing"],
  "rejection_reasons": []
}
```

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status + model loaded check |
| GET | `/pairs` | List supported pairs |
| POST | `/trade` | Full pipeline: data → LLM → rules → decision |
| POST | `/trade/close` | Log trade outcome (feeds learning) |
| GET | `/trade/open` | Current open positions |
| GET | `/trade/history` | Closed trade history |
| POST | `/learn` | Trigger continual learning cycle |
| POST | `/paper/trade` | Paper trade (simulated) |
| GET | `/paper/status` | Paper portfolio equity + positions |

---

## Continual Learning

Closed trades automatically improve the model over time:

1. Log a trade outcome via `POST /trade/close`
2. The system analyzes win/loss patterns by pair, direction, and confidence
3. When 50+ new examples accumulate, it generates JSONL training examples and triggers an incremental fine-tune

To trigger manually:
```bash
curl -X POST http://localhost:8000/learn
```

Or run the pipeline directly:
```python
from src.trading.continual_learner import run_learning_cycle
run_learning_cycle()
```

---

## Project Structure

```
crypto-oracle/
├── src/
│   ├── api/
│   │   └── api_server.py          # FastAPI server
│   ├── data_collection/
│   │   ├── master_data_collector.py
│   │   ├── ohlcv_downloader.py    # CCXT → CSV (Coinbase)
│   │   ├── sentiment_collector.py # CoinGecko, TextBlob, Reddit, YouTube
│   │   ├── macro_collector.py     # yfinance: DXY, VIX, SPY
│   │   ├── onchain_collector.py   # Fear & Greed Index
│   │   └── dataset_builder.py     # Combine all sources → training JSONL
│   ├── fine_tuning/
│   │   └── train_qwen_crypto_oracle.py  # Unsloth staged training
│   ├── inference/
│   │   ├── crypto_oracle.py       # CryptoOracle class (LLM inference)
│   │   └── market_analyzer.py     # Real-time data collection
│   ├── testing/
│   │   ├── model_evaluator.py     # Model output quality checks
│   │   ├── backtester.py          # Historical simulation
│   │   └── paper_trader.py        # Live simulation daemon
│   └── trading/
│       ├── risk_manager.py        # Deterministic rule layer (4-gate system)
│       ├── trade_logger.py        # SQLite trade journal
│       ├── live_trader.py         # CCXT order execution (paper/dry/live modes)
│       ├── notifier.py            # Telegram + Discord alerts
│       └── continual_learner.py   # Feedback → retrain pipeline
├── scripts/
│   └── health_check.py            # Verify model, exchange, DB, kill-switch
├── config/
│   ├── training_config.yaml
│   └── recommended_pairs.txt
├── Dockerfile                     # CUDA 12.6 + Python 3.11
├── docker-compose.yml             # paper / dry / live profiles
├── datasets/                      # git-ignored, generated locally
├── data/                          # git-ignored, collected locally
├── models/                        # git-ignored, trained locally
├── requirements.txt               # Runtime dependencies
├── requirements-train.txt         # Training dependencies (GPU)
└── .env.example                   # All environment variables with comments
```

---

## Supported Pairs

Default pairs (data available in `data/ohlcv/`):

| Pair | 1h | 6h | 1d |
|------|----|----|-----|
| BTC/USD | ✅ | ✅ | ✅ |
| ETH/USD | ✅ | ✅ | ✅ |
| XRP/USD | ✅ | ✅ | ✅ |
| LTC/USD | ✅ | ✅ | ✅ |
| LINK/USD | ✅ | ✅ | ✅ |
| BCH/USD | ✅ | ✅ | ✅ |
| XLM/USD | ✅ | ✅ | ✅ |

Any Coinbase Advanced Trade pair can be added by running the OHLCV downloader.

---

## Risk Warning

This software is for educational and research purposes. Cryptocurrency trading carries significant financial risk. Past performance of backtests does not guarantee future results.

**Before using real capital:**
- Complete at least 2–4 weeks of paper trading with positive results
- Run dry mode to confirm exchange connectivity and slippage estimates
- Test on the Coinbase sandbox (`COINBASE_SANDBOX=true`) to verify order placement
- Start with a small test account ($200–$1,000) under close supervision
- Keep the kill-switch ready (`touch pause.flag` or `POST /admin/pause`)
- Never risk more than you can afford to lose entirely

The default position size is 1% of equity per trade with a 15% maximum drawdown circuit breaker. These defaults are conservative by design — do not increase them until you have live trading results to justify it.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
