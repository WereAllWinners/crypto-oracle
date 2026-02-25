#!/bin/bash

# Comprehensive Data Download Script for Crypto Oracle
# Downloads 5+ years of historical data for all major cryptocurrencies on Coinbase

# Always run from the project root, regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=================================="
echo "Crypto Oracle - Full Data Download"
echo "=================================="
echo ""

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate   # Windows (Git Bash)
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate       # macOS / Linux
else
    echo "⚠️  Warning: no virtual environment found at venv/ — using system Python"
fi

# Read recommended pairs
PAIRS_FILE="config/recommended_pairs.txt"

if [ ! -f "$PAIRS_FILE" ]; then
    echo "❌ Error: $PAIRS_FILE not found"
    echo "Run the pair discovery script first"
    exit 1
fi

# Read pairs into array, stripping carriage returns (handles Windows CRLF line endings)
mapfile -t PAIRS < <(tr -d '\r' < "$PAIRS_FILE")

echo "📊 Downloading data for ${#PAIRS[@]} pairs"
echo "⏰ Estimated time: 2-4 hours"
echo ""
echo "Pairs to download:"
printf '%s\n' "${PAIRS[@]}"
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Cancelled"
    exit 1
fi

# Create log directory
mkdir -p logs

# Convert array to space-separated string
PAIRS_STR="${PAIRS[*]}"

# Run download
echo ""
echo "🚀 Starting download..."
echo "📝 Logging to: logs/download_full.log"
echo ""

nohup python src/data_collection/ohlcv_downloader.py \
  --pairs $PAIRS_STR \
  --timeframes 1h 6h 1d \
  --since 2019-01-01 \
  --exchange coinbase \
  > logs/download_full.log 2>&1 &

PID=$!
echo "✅ Download started in background (PID: $PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/download_full.log"
echo ""
echo "Check what's downloaded:"
echo "  ls -lh data/ohlcv/ | wc -l"
echo ""
echo "Stop download:"
echo "  kill $PID"
echo ""
