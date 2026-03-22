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
    
    def load_ohlcv(self, filename: str) -> pd.DataFrame:
        """Load OHLCV data"""
        filepath = self.ohlcv_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"OHLCV file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"📊 Loaded {filename}: {len(df)} candles")
        
        return df
    
    def load_sentiment(self, date: datetime) -> Optional[Dict]:
        """Load sentiment data for a given date"""
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
    
    def load_macro(self) -> Optional[Dict]:
        """Load latest macro data"""
        macro_files = list(self.macro_dir.glob('macro_*.json'))
        
        if not macro_files:
            return None
        
        latest_file = sorted(macro_files)[-1]
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def load_onchain(self) -> Optional[Dict]:
        """Load latest on-chain data"""
        onchain_files = list(self.onchain_dir.glob('onchain_*.json'))
        
        if not onchain_files:
            return None
        
        latest_file = sorted(onchain_files)[-1]
        
        with open(latest_file, 'r') as f:
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
    
    def get_future_outcome(self, df: pd.DataFrame, idx: int, horizon: int = 24) -> Optional[Dict]:
        """Get actual outcome after N candles"""
        if idx + horizon >= len(df):
            return None
        
        current_price = df.iloc[idx]['close']
        future_high = df.iloc[idx+1:idx+horizon+1]['high'].max()
        future_low = df.iloc[idx+1:idx+horizon+1]['low'].min()
        future_close = df.iloc[idx+horizon]['close']
        
        max_gain = ((future_high - current_price) / current_price) * 100
        max_loss = ((future_low - current_price) / current_price) * 100
        final_return = ((future_close - current_price) / current_price) * 100
        
        return {
            'max_gain': max_gain,
            'max_loss': max_loss,
            'final_return': final_return,
            'current_price': current_price,
            'future_price': future_close,
            'profitable': final_return > 0
        }
    
    def generate_expert_response(
        self,
        context: str,
        outcome: Dict,
        pair: str,
        sentiment: Optional[Dict] = None,
        macro: Optional[Dict] = None
    ) -> str:
        """Generate expert trading response with all context"""
        
        max_gain = outcome['max_gain']
        max_loss = outcome['max_loss']
        final_return = outcome['final_return']
        current = outcome['current_price']
        
        # Determine decision based on outcome + context
        if final_return > 2:  # Profitable long
            decision = "BUY"
            confidence = min(85, 65 + abs(final_return) * 2)
            
            # Adjust confidence based on sentiment
            if sentiment and sentiment['avg_sentiment'] > 0.3:
                confidence += 5
            elif sentiment and sentiment['avg_sentiment'] < -0.3:
                confidence -= 10
            
            # Adjust based on macro
            if macro:
                if macro.get('dxy', {}).get('signal') == 'bullish_crypto':
                    confidence += 5
                if macro.get('spy', {}).get('signal') == 'bullish_crypto':
                    confidence += 5
            
            confidence = min(95, max(55, confidence))
            
            response = f"""**Analysis for {pair}:**

**Recommendation: {decision}**
**Confidence: {int(confidence)}%**

**Trade Setup:**
- Entry: ${current:,.2f} (current market price)
- Stop-loss: ${current * 0.97:,.2f} (3% below entry)
- Take-profit 1: ${current * 1.02:,.2f} (2% gain - partial exit)
- Take-profit 2: ${current * 1.05:,.2f} (5% gain - remaining position)
- Position size: 2-3% of portfolio
- Risk/Reward: 1:{abs(final_return/3):.1f}

**Reasoning:**
The technical setup shows favorable conditions with positive momentum. Multiple indicators align for potential upward movement in the next 24 hours.

**Supporting Factors:**"""

            if sentiment and sentiment['bullish_pct'] > 50:
                response += f"\n- Strong bullish sentiment ({sentiment['bullish_pct']:.0f}% positive mentions)"
            
            if macro and macro.get('spy', {}).get('signal') == 'bullish_crypto':
                response += "\n- Risk-on macro environment (SPY strength)"
            
            response += f"""

**Risk Factors:**
- Set tight stop-loss at 3% to limit downside
- Monitor volume - confirmation needed on breakout
- Be prepared for volatility

**Exit Strategy:**
- Take 50% profit at 2% gain
- Trail stop on remaining 50% position
- Full exit if price breaks below ${current * 0.97:,.2f}
"""
        
        elif final_return < -2:  # Profitable short / bearish setup
            decision = "SELL"
            confidence = min(85, 65 + abs(final_return) * 2)

            # Adjust confidence based on sentiment
            if sentiment and sentiment['avg_sentiment'] < -0.3:
                confidence += 5
            elif sentiment and sentiment['avg_sentiment'] > 0.3:
                confidence -= 10

            # Adjust based on macro
            if macro:
                if macro.get('dxy', {}).get('signal') == 'bearish_crypto':
                    confidence += 5
                if macro.get('spy', {}).get('signal') == 'bearish':
                    confidence += 5

            confidence = min(95, max(55, confidence))

            response = f"""**Analysis for {pair}:**

**Recommendation: {decision}**
**Confidence: {int(confidence)}%**

**Trade Setup:**
- Entry: ${current:,.2f} (current market price)
- Stop-loss: ${current * 1.03:,.2f} (3% above entry)
- Take-profit 1: ${current * 0.98:,.2f} (2% decline - partial exit)
- Take-profit 2: ${current * 0.95:,.2f} (5% decline - remaining position)
- Position size: 2-3% of portfolio
- Risk/Reward: 1:{abs(final_return/3):.1f}

**Reasoning:**
The technical setup shows bearish conditions with negative momentum. Multiple indicators align for potential downward movement in the next 24 hours.

**Supporting Factors:**"""

            if sentiment and sentiment['bearish_pct'] > 50:
                response += f"\n- Negative sentiment dominates ({sentiment['bearish_pct']:.0f}% bearish mentions)"

            if macro and macro.get('dxy', {}).get('signal') == 'bearish_crypto':
                response += "\n- Strong dollar creating headwinds for crypto"

            response += f"""

**Risk Factors:**
- Set tight stop-loss at 3% above entry to limit upside risk
- Monitor volume - high volume on decline confirms bearish thesis
- Be prepared for volatility and potential short squeezes

**Exit Strategy:**
- Take 50% profit at 2% decline
- Trail stop on remaining 50% position
- Full exit if price breaks above ${current * 1.03:,.2f}
"""

        else:  # Small movement or choppy
            decision = "HOLD"
            confidence = 65
            
            response = f"""**Analysis for {pair}:**

**Recommendation: HOLD**
**Confidence: {confidence}%**

**Reasoning:**
Market is showing mixed signals with no clear directional bias. The risk/reward ratio is not compelling enough to justify a new position.

**Current Setup:**
- Price: ${current:,.2f}
- Expected movement: Low (sideways consolidation likely)
- Volume: {'Below average - lack of conviction' if 'low' in context.lower() else 'Normal'}

**Strategy:**
- If holding: Maintain current position with trailing stop at ${current * 0.97:,.2f}
- If not in position: Wait for clearer setup
- Watch for breakout above ${current * 1.02:,.2f} (bullish) or below ${current * 0.98:,.2f} (bearish)

**Risk Assessment:**
In low-conviction environments, the best trade is often no trade. Preserve capital for high-probability setups.

**Next Steps:**
- Monitor for volume increase (confirmation)
- Watch macro catalysts
- Set alerts at key levels
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
        
        # Load supplementary data
        macro = self.load_macro()
        onchain = self.load_onchain()
        
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
                    
                    # Load sentiment (if available)
                    sentiment = self.load_sentiment(df.index[-1])
                    
                    # Load strategy research
                    strategy = self.load_strategy_research(pair, timeframe)
                    
                    # Sample evenly
                    total_samples = min(max_examples, len(df) - 100)
                    step = max(1, len(df) // total_samples)
                    
                    for idx in range(100, len(df) - 24, step):
                        # Generate context
                        context = self.generate_enhanced_context(
                            df, idx, pair, sentiment, macro, onchain, strategy
                        )
                        
                        # Get outcome
                        outcome = self.get_future_outcome(df, idx, horizon=24)
                        if not outcome:
                            continue
                        
                        # Generate response
                        response = self.generate_expert_response(
                            context, outcome, pair, sentiment, macro
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=['BTC/USD', 'ETH/USD'])
    parser.add_argument('--timeframes', nargs='+', default=['1h'])
    parser.add_argument('--max-examples', type=int, default=5000)
    
    args = parser.parse_args()
    
    builder = EnhancedDatasetBuilder()
    builder.build_and_save(args.pairs, args.timeframes, args.max_examples)


if __name__ == '__main__':
    main()