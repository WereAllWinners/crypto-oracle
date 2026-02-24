#!/bin/bash

echo "======================================"
echo "Data Collection Progress"
echo "======================================"
echo ""

# Count OHLCV files
ohlcv_count=$(ls -1 data/ohlcv/*.csv 2>/dev/null | wc -l)
echo "OHLCV files downloaded: $ohlcv_count"

# Show total size
total_size=$(du -sh data/ohlcv/ 2>/dev/null | cut -f1)
echo "Total OHLCV size: $total_size"

echo ""
echo "Latest downloads:"
ls -lht data/ohlcv/*.csv 2>/dev/null | head -10

echo ""
echo "======================================"
echo "Estimated Training Examples"
echo "======================================"

# Each file has roughly:
# - 1h timeframe: ~40,000 candles avg = ~35,000 examples
# - 4h timeframe: ~10,000 candles avg = ~8,000 examples  
# - 1d timeframe: ~2,000 candles avg = ~1,500 examples

# Count by timeframe
h1_count=$(ls -1 data/ohlcv/*_1h.csv 2>/dev/null | wc -l)
h4_count=$(ls -1 data/ohlcv/*_4h.csv 2>/dev/null | wc -l)
d1_count=$(ls -1 data/ohlcv/*_1d.csv 2>/dev/null | wc -l)

h1_examples=$((h1_count * 35000))
h4_examples=$((h4_count * 8000))
d1_examples=$((d1_count * 1500))

total_estimated=$((h1_examples + h4_examples + d1_examples))

echo "1h files: $h1_count (~$h1_examples examples)"
echo "4h files: $h4_count (~$h4_examples examples)"
echo "1d files: $d1_count (~$d1_examples examples)"
echo ""
echo "Total estimated examples: ~$total_estimated"

if [ $total_estimated -ge 50000 ]; then
    echo "✅ TARGET REACHED! Ready to build dataset."
else
    needed=$((50000 - total_estimated))
    echo "⏳ Need ~$needed more examples to reach 50K"
fi