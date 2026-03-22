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
Trade Decision + SQLite Journal
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
│       ├── risk_manager.py        # Deterministic rule layer
│       ├── trade_logger.py        # SQLite trade journal
│       └── continual_learner.py   # Feedback → retrain pipeline
├── config/
│   ├── training_config.yaml
│   └── recommended_pairs.txt
├── datasets/                      # git-ignored, generated locally
├── data/                          # git-ignored, collected locally
├── models/                        # git-ignored, trained locally
├── requirements.txt               # Runtime dependencies
├── requirements-train.txt         # Training dependencies (GPU)
└── .env.example                   # Environment variable template
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

This software is for educational and research purposes. Cryptocurrency trading carries significant financial risk. Past performance of backtests does not guarantee future results. Always:

- Run paper trading for at minimum 2–4 weeks before using real capital
- Start with small position sizes (the default is 1% risk per trade)
- Never risk more than you can afford to lose
- Review all trade decisions before execution — this system is not a set-and-forget bot

---

## License

MIT License. See [LICENSE](LICENSE) for details.
