"""
Enhanced Dataset Builder
Combines OHLCV, sentiment, on-chain, macro, and strategy research
Creates comprehensive training examples with full context
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDatasetBuilder:
    """Build comprehensive training datasets with all available signals"""

    def __init__(
        self,
        ohlcv_dir: str = 'data/ohlcv',
        sentiment_dir: str = 'data/sentiment',
        onchain_dir: str = 'data/onchain',
        macro_dir: str = 'data/macro',
        strategy_dir: str = 'data/strategies',
        output_dir: str = 'datasets'
    ):
        self.ohlcv_dir = Path(ohlcv_dir)
        self.sentiment_dir = Path(sentiment_dir)
        self.onchain_dir = Path(onchain_dir)
        self.macro_dir = Path(macro_dir)
        self.strategy_dir = Path(strategy_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Historical auxiliary data caches (loaded once on first use)
        self._hist_macro: Optional[pd.DataFrame] = None
        self._hist_fg: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal: load historical CSVs once and cache them
    # ------------------------------------------------------------------

    def _get_hist_macro(self) -> Optional[pd.DataFrame]:
        if self._hist_macro is not None:
            return self._hist_macro
        path = self.macro_dir / "historical_macro.csv"
        if not path.exists():
            logger.warning("historical_macro.csv not found — run scripts/download_historical_aux_data.py")
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()
        self._hist_macro = df
        logger.info(f"📈 Loaded historical macro: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
        return self._hist_macro

    def _get_hist_fg(self) -> Optional[pd.DataFrame]:
        if self._hist_fg is not None:
            return self._hist_fg
        path = self.onchain_dir / "historical_fear_greed.csv"
        if not path.exists():
            logger.warning("historical_fear_greed.csv not found — run scripts/download_historical_aux_data.py")
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()
        self._hist_fg = df
        logger.info(f"😱 Loaded historical F&G: {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
        return self._hist_fg

    def load_ohlcv(self, filename: str) -> pd.DataFrame:
        """Load OHLCV data"""
        filepath = self.ohlcv_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"OHLCV file not found: {filepath}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"📊 Loaded {filename}: {len(df)} candles")

        return df

    def load_sentiment(self, date: datetime) -> Optional[Dict]:
        """Sentiment is not available as historical data — always returns None.

        Returning None intentionally drops the sentiment section from training
        prompts, which is correct: the training data previously used a single
        static snapshot for all examples (data leakage / noisy label). Dropping
        it lets the model focus on signals that actually vary per timestep.
        """
        return None

    def _load_sentiment_UNUSED(self, date: datetime) -> Optional[Dict]:
        """Kept for reference — original implementation that loaded the latest snapshot."""
        # Find closest sentiment file
        sentiment_files = list(self.sentiment_dir.glob('sentiment_*.jsonl'))

        if not sentiment_files:
            return None

        # For now, load the most recent
        # In production, you'd match by date
        latest_file = sorted(sentiment_files)[-1]
        
        sentiments = []
        with open(latest_file, 'r') as f:
            for line in f:
                sentiments.append(json.loads(line))
        
        # Aggregate
        if not sentiments:
            return None
        
        avg_sentiment = sum(s['sentiment_score'] for s in sentiments) / len(sentiments)
        
        return {
            'avg_sentiment': avg_sentiment,
            'total_items': len(sentiments),
            'bullish_pct': len([s for s in sentiments if s['sentiment_score'] > 0.3]) / len(sentiments) * 100,
            'bearish_pct': len([s for s in sentiments if s['sentiment_score'] < -0.3]) / len(sentiments) * 100
        }
    
    def load_macro(self, date=None) -> Optional[Dict]:
        """Return macro data for the given date from the historical CSV.

        Falls back to the latest snapshot JSON only when historical CSV is
        absent AND no date is supplied (e.g. live inference).
        """
        hist = self._get_hist_macro()
        if hist is not None and date is not None:
            ts = pd.Timestamp(date).normalize()
            if ts < hist.index[0]:
                return None  # before our history starts
            row = hist.asof(ts)  # most-recent row ≤ ts
            if pd.isna(row["dxy"]):
                return None
            return {
                "dxy": {
                    "current":    float(row["dxy"]),
                    "change_pct": float(row["dxy_change_1d"]),
                    "signal":     str(row["dxy_signal"]),
                },
                "spy": {
                    "current":    float(row["spy"]),
                    "change_pct": float(row["spy_change_1d"]),
                    "signal":     str(row["spy_signal"]),
                },
                "vix": {
                    "current": float(row["vix"]),
                    "signal":  str(row["vix_signal"]),
                },
            }

        # Fallback: latest snapshot file (used during live inference)
        macro_files = list(self.macro_dir.glob('macro_*.json'))
        if not macro_files:
            return None
        with open(sorted(macro_files)[-1]) as f:
            return json.load(f)

    def load_onchain(self, date=None) -> Optional[Dict]:
        """Return on-chain data for the given date from the historical CSV.

        Falls back to the latest snapshot JSON for live inference.
        """
        hist = self._get_hist_fg()
        if hist is not None and date is not None:
            ts = pd.Timestamp(date).normalize()
            if ts < hist.index[0]:
                return None
            row = hist.asof(ts)
            if pd.isna(row["value"]):
                return None
            return {
                "fear_greed": {
                    "value":          int(row["value"]),
                    "classification": str(row["classification"]),
                    "signal":         str(row["signal"]),
                }
            }

        # Fallback: latest snapshot file
        onchain_files = list(self.onchain_dir.glob('onchain_*.json'))
        if not onchain_files:
            return None
        with open(sorted(onchain_files)[-1]) as f:
            return json.load(f)
    
    def load_strategy_research(self, pair: str, timeframe: str) -> Optional[Dict]:
        """Load strategy research for a pair"""
        filename = f"{pair.replace('/', '_')}_{timeframe}_research.json"
        filepath = self.strategy_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Trend
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Momentum
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch_rsi'] = ta.momentum.stochrsi(df['close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price changes
        df['price_change_1h'] = df['close'].pct_change(1) * 100
        df['price_change_4h'] = df['close'].pct_change(4) * 100
        df['price_change_24h'] = df['close'].pct_change(24) * 100
        
        # Volatility
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )
        
        return df
    
    def generate_enhanced_context(
        self,
        df: pd.DataFrame,
        idx: int,
        pair: str,
        sentiment: Optional[Dict] = None,
        macro: Optional[Dict] = None,
        onchain: Optional[Dict] = None,
        strategy: Optional[Dict] = None
    ) -> str:
        """Generate comprehensive market context"""
        
        row = df.iloc[idx]
        
        # Basic price info
        price = row['close']
        change_1h = row['price_change_1h']
        change_24h = row['price_change_24h']
        
        # Technical analysis
        if row['close'] > row['sma_20'] > row['sma_50']:
            trend = "strong uptrend"
            trend_strength = "strong"
        elif row['close'] > row['sma_20']:
            trend = "mild uptrend"
            trend_strength = "moderate"
        elif row['close'] < row['sma_20'] < row['sma_50']:
            trend = "strong downtrend"
            trend_strength = "strong"
        elif row['close'] < row['sma_20']:
            trend = "mild downtrend"
            trend_strength = "moderate"
        else:
            trend = "sideways"
            trend_strength = "weak"
        
        # RSI
        rsi = row['rsi']
        if rsi > 70:
            rsi_state = f"overbought (RSI {rsi:.1f})"
        elif rsi < 30:
            rsi_state = f"oversold (RSI {rsi:.1f})"
        else:
            rsi_state = f"neutral (RSI {rsi:.1f})"
        
        # MACD
        macd_signal = "bullish" if row['macd'] > row['macd_signal'] else "bearish"
        
        # Bollinger Bands
        bb_position = (price - row['bb_lower']) / (row['bb_upper'] - row['bb_lower']) * 100
        if bb_position > 80:
            bb_state = "near upper band (potential reversal)"
        elif bb_position < 20:
            bb_state = "near lower band (potential bounce)"
        else:
            bb_state = "mid-range"
        
        # Volume
        vol_state = "high" if row['volume_ratio'] > 1.5 else "normal" if row['volume_ratio'] > 0.7 else "low"
        
        # Build context
        context = f"""Current price: ${price:,.2f}
1h change: {change_1h:+.2f}%
24h change: {change_24h:+.2f}%

📊 Technical Analysis:
- Trend: {trend} ({trend_strength})
- RSI: {rsi_state}
- MACD: {macd_signal} crossover
- Bollinger Bands: {bb_state}
- Volume: {vol_state} ({row['volume_ratio']:.2f}x average)
- SMA 20: ${row['sma_20']:,.2f}
- SMA 50: ${row['sma_50']:,.2f}
- SMA 200: ${row['sma_200']:,.2f}"""
        
        # Add sentiment if available
        if sentiment:
            context += f"""

📰 Market Sentiment:
- Overall sentiment: {sentiment['avg_sentiment']:.2f} (-1 to +1 scale)
- Bullish mentions: {sentiment['bullish_pct']:.1f}%
- Bearish mentions: {sentiment['bearish_pct']:.1f}%
- Total data points: {sentiment['total_items']}"""
        
        # Add macro if available
        if macro:
            context += f"\n\n🌍 Macro Environment:"
            
            if macro.get('dxy'):
                context += f"\n- DXY (Dollar Index): ${macro['dxy']['current']:.2f} ({macro['dxy']['change_pct']:+.2f}%) - {macro['dxy']['signal']}"
            
            if macro.get('spy'):
                context += f"\n- SPY (S&P 500): ${macro['spy']['current']:.2f} ({macro['spy']['change_pct']:+.2f}%) - {macro['spy']['signal']}"
            
            if macro.get('vix'):
                context += f"\n- VIX (Fear Index): {macro['vix']['current']:.2f} - {macro['vix']['signal']}"
            
            if macro.get('btc_dominance'):
                context += f"\n- BTC Dominance: {macro['btc_dominance']['btc_dominance']:.2f}% - {macro['btc_dominance']['phase']}"
            
            if macro.get('correlations'):
                btc_spy = macro['correlations'].get('btc_spy')
                if btc_spy is not None:
                    context += f"\n- BTC-SPY Correlation: {btc_spy:.2f}"
        
        # Add on-chain if available
        if onchain and onchain.get('fear_greed'):
            fg = onchain['fear_greed']
            context += f"""

⛓️  On-Chain Metrics:
- Fear & Greed Index: {fg['value']}/100 ({fg['classification']})
- Signal: {fg['signal']} (contrarian indicator)"""
        
        # Add strategy insights if available
        if strategy and strategy.get('best_strategy'):
            bs = strategy['best_strategy']
            context += f"""

📈 Historical Strategy Performance on {pair}:
- Best strategy: {bs['name']}
- Win rate: {bs['metrics'].get('win_rate', 0):.1f}%
- Avg P&L: {bs['metrics'].get('avg_pnl', 0):.2f}%
- Total trades: {bs['metrics'].get('total_trades', 0)}"""
        
        return context
    
    def get_future_outcome(self, df: pd.DataFrame, idx: int,
                           horizon: int = 240,
                           sl_atr_mult: float = 1.5,
                           tp_atr_mult: float = 3.0) -> Optional[Dict]:
        """Simulate ATR-based SL/TP exit on forward bars — matching backtest logic.

        Labels are determined by which level is hit first within `horizon` bars:
          - TP hit before SL  → final_return positive  (BUY label)
          - SL hit before TP  → final_return negative  (SELL/HOLD label)
          - Neither within horizon → exit at close (HOLD label)

        This keeps training labels consistent with the ATR-exit backtester so the
        model learns to predict outcomes under the same evaluation methodology.
        """
        if idx + 1 >= len(df):
            return None

        row = df.iloc[idx]
        current_price = float(row['close'])
        atr = float(row.get('atr', current_price * 0.02))

        # Clamp stop distance: 1%–3% of price (same as generate_expert_response)
        stop_dist = max(min(atr / current_price, 0.03), 0.01) * current_price

        long_sl = current_price - sl_atr_mult * stop_dist
        long_tp = current_price + tp_atr_mult * stop_dist

        n = len(df)
        max_bars = min(horizon, n - idx - 1)

        long_exit_price  = None
        long_outcome     = "timeout"

        for i in range(idx + 1, idx + 1 + max_bars):
            high  = float(df.iloc[i]['high'])
            low   = float(df.iloc[i]['low'])

            if low <= long_sl:
                long_exit_price = long_sl
                long_outcome    = "stopped_out"
                break
            if high >= long_tp:
                long_exit_price = long_tp
                long_outcome    = "took_profit"
                break

        if long_exit_price is None:
            long_exit_price = float(df.iloc[idx + max_bars]['close'])

        final_return = (long_exit_price - current_price) / current_price * 100

        return {
            'max_gain':     (long_tp - current_price) / current_price * 100,
            'max_loss':     (long_sl - current_price) / current_price * 100,
            'final_return': final_return,
            'current_price': current_price,
            'future_price':  long_exit_price,
            'profitable':    final_return > 0,
            'exit_outcome':  long_outcome,
        }
    
    def generate_expert_response(
        self,
        context: str,
        outcome: Dict,
        pair: str,
        sentiment: Optional[Dict] = None,
        macro: Optional[Dict] = None,
        indicators: Optional[Dict] = None
    ) -> str:
        """Generate expert trading response with dynamic reasoning based on actual indicator values."""

        max_gain = outcome['max_gain']
        max_loss = outcome['max_loss']
        final_return = outcome['final_return']
        current = outcome['current_price']

        # Unpack indicator values for conditional reasoning (fall back to neutral defaults)
        ind = indicators or {}
        rsi = float(ind.get('rsi', 50))
        macd_bull = bool(ind.get('macd_bull', True))   # True = macd > signal
        vol_ratio = float(ind.get('volume_ratio', 1.0))
        sma_20 = float(ind.get('sma_20', current))
        sma_50 = float(ind.get('sma_50', current))
        sma_200 = float(ind.get('sma_200', current))
        bb_position = float(ind.get('bb_position', 50))  # 0-100 within bands
        change_24h = float(ind.get('change_24h', 0))
        trend = str(ind.get('trend', 'sideways'))
        atr = float(ind.get('atr', current * 0.02))     # fallback 2% of price

        # Use ATR for tighter, dynamic stop distance (capped at 3%)
        atr_stop_pct = min(atr / current, 0.03) if current > 0 else 0.02
        stop_dist_pct = max(atr_stop_pct, 0.01)  # never less than 1%

        # --- Helper: build dynamic reasoning bullets ---
        def _rsi_comment(direction: str) -> str:
            if direction == "BUY":
                if rsi < 35:
                    return f"RSI at {rsi:.1f} is oversold — historically a high-probability reversal zone"
                elif rsi < 50:
                    return f"RSI at {rsi:.1f} has room to expand before reaching overbought territory"
                elif rsi < 65:
                    return f"RSI at {rsi:.1f} is neutral but rising — momentum building without being overextended"
                else:
                    return f"RSI at {rsi:.1f} is elevated; risk of short-term pullback before resuming"
            else:  # SELL
                if rsi > 65:
                    return f"RSI at {rsi:.1f} is overbought — price extended and vulnerable to mean reversion"
                elif rsi > 50:
                    return f"RSI at {rsi:.1f} is rolling over from elevated levels — bearish divergence risk"
                else:
                    return f"RSI at {rsi:.1f} is already subdued, confirming weak buying pressure"

        def _macd_comment(direction: str) -> str:
            if direction == "BUY":
                return ("MACD crossed above signal line — bullish momentum confirmed"
                        if macd_bull else
                        "MACD is below signal line; watch for bullish crossover before adding size")
            else:
                return ("MACD crossed below signal line — bearish momentum confirmed"
                        if not macd_bull else
                        "MACD still above signal; a bearish crossover would add conviction to the short")

        def _sma_comment(direction: str) -> str:
            above_20 = current > sma_20
            above_50 = current > sma_50
            above_200 = current > sma_200
            if direction == "BUY":
                if above_20 and above_50 and above_200:
                    return f"Price (${current:,.2f}) is above all three SMAs (20/50/200) — strong structural uptrend"
                elif above_20 and above_50:
                    return f"Price is above SMA-20 (${sma_20:,.2f}) and SMA-50 (${sma_50:,.2f}), reclaiming momentum"
                elif above_20:
                    return f"Price reclaimed SMA-20 (${sma_20:,.2f}); next resistance at SMA-50 (${sma_50:,.2f})"
                else:
                    return f"Price near SMA-20 support (${sma_20:,.2f}) — watching for bounce confirmation"
            else:
                if not above_20 and not above_50 and not above_200:
                    return f"Price (${current:,.2f}) is below all three SMAs (20/50/200) — confirmed downtrend structure"
                elif not above_20 and not above_50:
                    return f"Price broke below SMA-20 (${sma_20:,.2f}) and SMA-50 (${sma_50:,.2f}) — bearish"
                elif not above_20:
                    return f"Price lost SMA-20 support (${sma_20:,.2f}); next support at SMA-50 (${sma_50:,.2f})"
                else:
                    return f"Price rejected at SMA-20 resistance (${sma_20:,.2f}) — potential breakdown"

        def _vol_comment(direction: str) -> str:
            if vol_ratio >= 2.0:
                return f"Volume at {vol_ratio:.1f}x the 20-period average — strong institutional participation {'buying' if direction=='BUY' else 'selling'}"
            elif vol_ratio >= 1.5:
                return f"Volume elevated at {vol_ratio:.1f}x average — move has above-average conviction"
            elif vol_ratio >= 0.8:
                return f"Volume near average ({vol_ratio:.1f}x) — steady participation, no unusual activity"
            else:
                return f"Volume below average ({vol_ratio:.1f}x) — {'wait for volume confirmation before adding size' if direction=='BUY' else 'low-volume selloff may lack follow-through'}"

        def _bb_comment(direction: str) -> str:
            if direction == "BUY":
                if bb_position < 20:
                    return f"Price near lower Bollinger Band ({bb_position:.0f}th percentile) — statistically oversold, snap-back likely"
                elif bb_position < 50:
                    return f"Price in lower half of Bollinger Bands ({bb_position:.0f}th percentile) — room to expand upward"
                elif bb_position > 80:
                    return f"Price near upper Bollinger Band ({bb_position:.0f}th percentile) — watch for band resistance"
                else:
                    return f"Price at mid-range Bollinger Bands ({bb_position:.0f}th percentile) — room for expansion"
            else:
                if bb_position > 80:
                    return f"Price near upper Bollinger Band ({bb_position:.0f}th percentile) — statistically overbought, reversal likely"
                elif bb_position > 50:
                    return f"Price in upper half of Bollinger Bands ({bb_position:.0f}th percentile) — room to decline"
                elif bb_position < 20:
                    return f"Price near lower Bollinger Band ({bb_position:.0f}th percentile) — watch for band support before shorting"
                else:
                    return f"Price at mid-range Bollinger Bands ({bb_position:.0f}th percentile)"

        # =====================================================================
        # Label derived from ATR-based SL/TP simulation (see get_future_outcome).
        # exit_outcome=="took_profit" means the long TP was hit before the stop
        # → genuine BUY edge under the same exit rules used by the backtester.
        # exit_outcome=="stopped_out" → long SL hit first → SELL or HOLD.
        # "timeout" → neither hit within 240 bars → HOLD (low-conviction).
        # =====================================================================
        exit_outcome = outcome.get('exit_outcome', 'timeout')
        above_200 = (current > sma_200) if sma_200 and sma_200 != current else True

        if exit_outcome == "took_profit" and above_200:
            final_label = "BUY"
        elif exit_outcome == "stopped_out" and not above_200:
            final_label = "SELL"
        else:
            final_label = "HOLD"

        # =====================================================================
        # BUY
        # =====================================================================
        if final_label == "BUY":
            decision = "BUY"
            confidence = min(85, 65 + abs(final_return) * 2)

            if sentiment and sentiment['avg_sentiment'] > 0.3:
                confidence += 5
            elif sentiment and sentiment['avg_sentiment'] < -0.3:
                confidence -= 10
            if macro:
                if macro.get('dxy', {}).get('signal') == 'bullish_crypto':
                    confidence += 5
                if macro.get('spy', {}).get('signal') == 'bullish_crypto':
                    confidence += 5
            confidence = min(95, max(55, int(confidence)))

            sl = current * (1 - stop_dist_pct)
            tp1 = current * (1 + stop_dist_pct * 1.5)
            tp2 = current * (1 + stop_dist_pct * 3.0)
            rr = round(stop_dist_pct * 3.0 / stop_dist_pct, 1)

            reasoning = (
                f"Price is in a {trend} with a {change_24h:+.2f}% move in the last 24 hours. "
                f"{_sma_comment('BUY')}. "
                f"{_rsi_comment('BUY')}. "
                f"{_macd_comment('BUY')}. "
                f"{_bb_comment('BUY')}. "
                f"{_vol_comment('BUY')}."
            )

            response = f"""**Analysis for {pair}:**

**Recommendation: BUY**
**Confidence: {confidence}%**

**Trade Setup:**
- Entry: ${current:,.2f} (current market price)
- Stop-loss: ${sl:,.2f} ({stop_dist_pct*100:.1f}% below entry, ATR-based)
- Take-profit 1: ${tp1:,.2f} ({stop_dist_pct*150:.1f}% gain - partial exit)
- Take-profit 2: ${tp2:,.2f} ({stop_dist_pct*300:.1f}% gain - remaining position)
- Position size: 2-3% of portfolio
- Risk/Reward: 1:{rr}

**Reasoning:**
{reasoning}

**Supporting Factors:**"""

            if sentiment and sentiment['bullish_pct'] > 50:
                response += f"\n- Sentiment: {sentiment['bullish_pct']:.0f}% bullish mentions — crowd positioning supportive"
            if macro and macro.get('spy', {}).get('signal') == 'bullish_crypto':
                response += "\n- Macro: risk-on environment (SPY strength) — tailwind for crypto"
            if current > sma_200:
                response += f"\n- Long-term trend: price above SMA-200 (${sma_200:,.2f}) — secular uptrend intact"

            response += f"""

**Risk Factors:**
- Stop-loss at ${sl:,.2f} protects against unexpected reversal
- {'High volume confirms move; watch for exhaustion candle' if vol_ratio >= 1.5 else 'Low volume — await volume confirmation before adding to position'}
- {'RSI approaching overbought; consider scaling into position' if rsi > 60 else 'RSI has room to run — momentum not yet stretched'}

**Exit Strategy:**
- Take 50% profit at ${tp1:,.2f}
- Trail stop on remaining 50% to lock in gains
- Full exit if price closes below ${sl:,.2f}
"""

        # =====================================================================
        # SELL
        # =====================================================================
        elif final_label == "SELL":
            decision = "SELL"
            confidence = min(85, 65 + abs(final_return) * 2)

            if sentiment and sentiment['avg_sentiment'] < -0.3:
                confidence += 5
            elif sentiment and sentiment['avg_sentiment'] > 0.3:
                confidence -= 10
            if macro:
                if macro.get('dxy', {}).get('signal') == 'bearish_crypto':
                    confidence += 5
                if macro.get('spy', {}).get('signal') == 'bearish':
                    confidence += 5
            confidence = min(95, max(55, int(confidence)))

            sl = current * (1 + stop_dist_pct)
            tp1 = current * (1 - stop_dist_pct * 1.5)
            tp2 = current * (1 - stop_dist_pct * 3.0)
            rr = round(stop_dist_pct * 3.0 / stop_dist_pct, 1)

            reasoning = (
                f"Price is in a {trend} with a {change_24h:+.2f}% move in the last 24 hours. "
                f"{_sma_comment('SELL')}. "
                f"{_rsi_comment('SELL')}. "
                f"{_macd_comment('SELL')}. "
                f"{_bb_comment('SELL')}. "
                f"{_vol_comment('SELL')}."
            )

            response = f"""**Analysis for {pair}:**

**Recommendation: SELL**
**Confidence: {confidence}%**

**Trade Setup:**
- Entry: ${current:,.2f} (current market price)
- Stop-loss: ${sl:,.2f} ({stop_dist_pct*100:.1f}% above entry, ATR-based)
- Take-profit 1: ${tp1:,.2f} ({stop_dist_pct*150:.1f}% decline - partial exit)
- Take-profit 2: ${tp2:,.2f} ({stop_dist_pct*300:.1f}% decline - remaining position)
- Position size: 2-3% of portfolio
- Risk/Reward: 1:{rr}

**Reasoning:**
{reasoning}

**Supporting Factors:**"""

            if sentiment and sentiment['bearish_pct'] > 50:
                response += f"\n- Sentiment: {sentiment['bearish_pct']:.0f}% bearish mentions — negative crowd positioning"
            if macro and macro.get('dxy', {}).get('signal') == 'bearish_crypto':
                response += "\n- Macro: strong dollar (DXY strength) — headwind for crypto assets"
            if current < sma_200:
                response += f"\n- Long-term trend: price below SMA-200 (${sma_200:,.2f}) — secular downtrend intact"

            response += f"""

**Risk Factors:**
- Stop-loss at ${sl:,.2f} limits exposure if price reverses
- {'Volume confirms distribution; watch for short-squeeze risk' if vol_ratio >= 1.5 else 'Low-volume decline may lack follow-through — use tighter sizing'}
- {'RSI already oversold; short-term bounce possible before resuming down' if rsi < 35 else 'RSI has room to fall — momentum supports continuation'}

**Exit Strategy:**
- Take 50% profit at ${tp1:,.2f}
- Trail stop on remaining 50% to lock in gains
- Full exit if price closes above ${sl:,.2f}
"""

        # =====================================================================
        # HOLD
        # =====================================================================
        else:
            decision = "HOLD"
            # Confidence varies based on how ambiguous the indicators are
            conflicting = (macd_bull and rsi > 55) or (not macd_bull and rsi < 45)
            confidence = 60 if conflicting else 70

            mixed_signals = []
            if rsi > 60:
                mixed_signals.append(f"RSI at {rsi:.1f} shows some buying pressure but not a breakout setup")
            elif rsi < 40:
                mixed_signals.append(f"RSI at {rsi:.1f} shows weakness but not yet a confirmed reversal")
            else:
                mixed_signals.append(f"RSI at {rsi:.1f} is neutral — no directional conviction from momentum")

            if macd_bull:
                mixed_signals.append("MACD above signal but margin is thin — not a high-conviction entry")
            else:
                mixed_signals.append("MACD below signal but no acceleration lower — indecisive")

            if vol_ratio < 0.8:
                mixed_signals.append(f"Volume at {vol_ratio:.1f}x average — market participants are on the sidelines")
            else:
                mixed_signals.append(f"Volume at {vol_ratio:.1f}x average — no unusual activity to signal a move")

            response = f"""**Analysis for {pair}:**

**Recommendation: HOLD**
**Confidence: {confidence}%**

**Reasoning:**
Price is in a {trend} with a {change_24h:+.2f}% move in the last 24 hours. The risk/reward ratio is not compelling enough to initiate a new position. {'. '.join(mixed_signals)}.

**Current Setup:**
- Price: ${current:,.2f}
- SMA-20: ${sma_20:,.2f} | SMA-50: ${sma_50:,.2f} | SMA-200: ${sma_200:,.2f}
- Expected movement: Sideways / low-conviction consolidation
- Volume: {vol_ratio:.1f}x average

**Strategy:**
- If holding: Maintain position with trailing stop at ${current * 0.97:,.2f}
- If not in position: Wait for clearer directional setup
- Watch for breakout above ${current * 1.02:,.2f} (bullish trigger) or breakdown below ${current * 0.98:,.2f} (bearish trigger)

**Risk Assessment:**
In low-conviction environments the best trade is often no trade. Preserve capital for high-probability setups where technicals, volume, and momentum align.
"""

        return response
    
    def build_comprehensive_dataset(
        self,
        pairs: List[str],
        timeframes: List[str],
        max_examples: int = 5000
    ) -> tuple:
        """Build dataset with all available data sources"""
        
        logger.info("🔨 Building comprehensive dataset...")

        # Pre-warm the historical data caches so any warnings fire exactly once
        self._get_hist_macro()
        self._get_hist_fg()

        all_examples = []

        for pair in pairs:
            for timeframe in timeframes:
                try:
                    # Load OHLCV
                    filename = f"{pair.replace('/', '_')}_{timeframe}.csv"
                    df = self.load_ohlcv(filename)

                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    df = df.dropna()

                    # Sentiment is excluded (no historical data available — see load_sentiment)
                    sentiment = None

                    # Load strategy research
                    strategy = self.load_strategy_research(pair, timeframe)

                    # Sample evenly
                    total_samples = min(max_examples, len(df) - 100)
                    step = max(1, len(df) // total_samples)

                    for idx in range(100, len(df) - 24, step):
                        row_date = df.index[idx]

                        # Load macro & on-chain for this specific bar's date
                        macro  = self.load_macro(date=row_date)
                        onchain = self.load_onchain(date=row_date)

                        # Generate context
                        context = self.generate_enhanced_context(
                            df, idx, pair, sentiment, macro, onchain, strategy
                        )
                        
                        # Get outcome
                        outcome = self.get_future_outcome(df, idx, horizon=24)
                        if not outcome:
                            continue
                        
                        # Build indicators dict for dynamic reasoning
                        row = df.iloc[idx]
                        bb_range = row['bb_upper'] - row['bb_lower']
                        bb_pos = float(
                            (row['close'] - row['bb_lower']) / bb_range * 100
                            if bb_range > 0 else 50
                        )
                        ind = {
                            'rsi': float(row['rsi']),
                            'macd_bull': bool(row['macd'] > row['macd_signal']),
                            'volume_ratio': float(row['volume_ratio']),
                            'sma_20': float(row['sma_20']),
                            'sma_50': float(row['sma_50']),
                            'sma_200': float(row['sma_200']),
                            'bb_position': bb_pos,
                            'change_24h': float(row['price_change_24h']),
                            'trend': (
                                'strong uptrend' if row['close'] > row['sma_20'] > row['sma_50']
                                else 'mild uptrend' if row['close'] > row['sma_20']
                                else 'strong downtrend' if row['close'] < row['sma_20'] < row['sma_50']
                                else 'mild downtrend' if row['close'] < row['sma_20']
                                else 'sideways'
                            ),
                            'atr': float(row['atr']),
                        }

                        # Generate response
                        response = self.generate_expert_response(
                            context, outcome, pair, sentiment, macro, indicators=ind
                        )
                        
                        # Create example
                        instruction = f"""Analyze the current market conditions for {pair} and provide a comprehensive trading recommendation.

{context}

What is your trading recommendation? Provide:
1. Decision (BUY/SELL/HOLD)
2. Confidence level
3. Entry/exit prices
4. Risk management plan
5. Detailed reasoning based on technical, sentiment, and macro factors"""
                        
                        example = {
                            'instruction': instruction,
                            'input': '',
                            'output': response,
                            'metadata': {
                                'pair': pair,
                                'timeframe': timeframe,
                                'timestamp': str(df.index[idx]),
                                'outcome': outcome,
                                'has_sentiment': sentiment is not None,
                                'has_macro': macro is not None,
                                'has_onchain': onchain is not None,
                                'has_strategy': strategy is not None
                            }
                        }
                        
                        all_examples.append(example)
                    
                    logger.info(f"✅ Generated examples for {pair} {timeframe}")
                    
                except FileNotFoundError:
                    logger.warning(f"⚠️  Skipping {pair} {timeframe} - file not found")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error processing {pair} {timeframe}: {e}")
                    continue
        
        # Split train/eval
        split_idx = int(len(all_examples) * 0.8)
        train_examples = all_examples[:split_idx]
        eval_examples = all_examples[split_idx:]
        
        logger.info(f"📚 Total: {len(all_examples)} | Train: {len(train_examples)} | Eval: {len(eval_examples)}")
        
        return train_examples, eval_examples
    
    def save_dataset(self, examples: List[Dict], filename: str):
        """Save dataset to JSONL"""
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            return obj
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for example in examples:
                clean = convert_numpy(example)
                f.write(json.dumps(clean) + '\n')
        
        logger.info(f"💾 Saved {len(examples)} examples to {output_path}")
    
    def build_and_save(
        self,
        pairs: List[str],
        timeframes: List[str] = ['1h'],
        max_examples: int = 5000
    ):
        """Build and save comprehensive dataset"""
        
        train, eval = self.build_comprehensive_dataset(pairs, timeframes, max_examples)
        
        self.save_dataset(train, 'enhanced_sft_train.jsonl')
        self.save_dataset(eval, 'enhanced_sft_eval.jsonl')
        
        logger.info("✅ Enhanced dataset build complete!")


def main():
    """CLI entry point"""
    import argparse

    all_pairs = [
        'BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD',
        'LINK/USD', 'BCH/USD', 'XLM/USD', 'UNI/USD',
    ]
    all_timeframes = ['1h', '6h', '1d']

    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=all_pairs)
    parser.add_argument('--timeframes', nargs='+', default=all_timeframes)
    parser.add_argument('--max-examples', type=int, default=5000,
                        help='Max examples per pair/timeframe combination (default: 5000)')

    args = parser.parse_args()

    builder = EnhancedDatasetBuilder()
    builder.build_and_save(args.pairs, args.timeframes, args.max_examples)


if __name__ == '__main__':
    main()