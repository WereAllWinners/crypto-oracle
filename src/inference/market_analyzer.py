"""
Real-Time Market Data Collector & Analyzer
Fetches current data and prepares it for inference
"""

import ccxt
import pandas as pd
import ta
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.sentiment_collector import SentimentCollector
from data_collection.macro_collector import MacroCollector
from data_collection.onchain_collector import OnChainCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Collect and prepare real-time market data for inference"""
    
    def __init__(self, exchange_name: str = 'coinbase'):
        self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        
        # Data collectors
        self.sentiment_collector = SentimentCollector()
        self.macro_collector = MacroCollector()
        self.onchain_collector = OnChainCollector()
        
        # Cache for recent data
        self.macro_cache = None
        self.macro_cache_time = None
        self.sentiment_cache = {}
        self.strategy_cache = {}
    
    def get_current_market_data(self, pair: str, timeframe: str = '1h') -> Dict:
        """
        Fetch current market data for a pair
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            timeframe: Candle timeframe
        
        Returns:
            Dict with all market data ready for inference
        """
        logger.info(f"📊 Fetching current data for {pair}...")
        
        # 1. Get recent OHLCV data
        ohlcv_data = self._get_ohlcv(pair, timeframe)
        
        # 2. Calculate technical indicators
        technical = self._calculate_indicators(ohlcv_data)
        
        # 3. Get sentiment (cached for 1 hour)
        sentiment = self._get_sentiment(pair)
        
        # 4. Get macro data (cached for 4 hours)
        macro = self._get_macro()
        
        # 5. Get on-chain data
        onchain = self._get_onchain(pair)
        
        # 6. Get strategy performance
        strategy = self._get_strategy(pair, timeframe)
        
        # Combine everything
        market_data = {
            'pair': pair,
            'price': technical['current_price'],
            'change_1h': technical['change_1h'],
            'change_24h': technical['change_24h'],
            'technical': technical,
            'sentiment': sentiment,
            'macro': macro,
            'onchain': onchain,
            'strategy': strategy
        }
        
        logger.info(f"✅ Market data ready for {pair}")
        
        return market_data
    
    def _get_ohlcv(self, pair: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch recent OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if df.empty:
            return {}
        
        # Current price
        current = df.iloc[-1]
        price = current['close']
        
        # Price changes
        if len(df) > 1:
            change_1h = ((price - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100
        else:
            change_1h = 0
        
        if len(df) > 24:
            change_24h = ((price - df.iloc[-25]['close']) / df.iloc[-25]['close']) * 100
        else:
            change_24h = 0
        
        # Calculate indicators
        df_calc = df.copy()
        
        # Trend
        df_calc['sma_20'] = ta.trend.sma_indicator(df_calc['close'], window=20)
        df_calc['sma_50'] = ta.trend.sma_indicator(df_calc['close'], window=50)
        df_calc['sma_200'] = ta.trend.sma_indicator(df_calc['close'], window=200)
        
        # RSI
        df_calc['rsi'] = ta.momentum.rsi(df_calc['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df_calc['close'])
        df_calc['macd'] = macd.macd()
        df_calc['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_calc['close'], window=20)
        df_calc['bb_upper'] = bollinger.bollinger_hband()
        df_calc['bb_lower'] = bollinger.bollinger_lband()
        
        # Volume
        df_calc['volume_sma'] = df_calc['volume'].rolling(window=20).mean()
        
        # Get latest values
        latest = df_calc.iloc[-1]
        
        # Trend analysis
        if price > latest['sma_20'] > latest['sma_50']:
            trend = "strong uptrend"
        elif price > latest['sma_20']:
            trend = "mild uptrend"
        elif price < latest['sma_20'] < latest['sma_50']:
            trend = "strong downtrend"
        elif price < latest['sma_20']:
            trend = "mild downtrend"
        else:
            trend = "sideways"
        
        # RSI state
        rsi = latest['rsi']
        if rsi > 70:
            rsi_state = f"overbought (RSI {rsi:.1f})"
        elif rsi < 30:
            rsi_state = f"oversold (RSI {rsi:.1f})"
        else:
            rsi_state = f"neutral (RSI {rsi:.1f})"
        
        # MACD signal
        macd_signal = "bullish" if latest['macd'] > latest['macd_signal'] else "bearish"
        
        # Bollinger Bands
        bb_position = (price - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100
        if bb_position > 80:
            bb_state = "near upper band (potential reversal)"
        elif bb_position < 20:
            bb_state = "near lower band (potential bounce)"
        else:
            bb_state = "mid-range"
        
        # Volume
        volume_ratio = current['volume'] / latest['volume_sma'] if latest['volume_sma'] > 0 else 1
        volume_state = "high" if volume_ratio > 1.5 else "normal" if volume_ratio > 0.7 else "low"
        
        return {
            'current_price': price,
            'change_1h': change_1h,
            'change_24h': change_24h,
            'trend': trend,
            'rsi_state': rsi_state,
            'macd_signal': macd_signal,
            'bb_state': bb_state,
            'volume_state': volume_state,
            'sma_20': latest['sma_20'],
            'sma_50': latest['sma_50'],
            'sma_200': latest['sma_200'],
            'rsi': rsi,
            'volume_ratio': volume_ratio
        }
    
    def _get_sentiment(self, pair: str) -> Optional[Dict]:
        """Get cached or fresh sentiment data"""
        # Check cache (1 hour)
        if pair in self.sentiment_cache:
            cache_time, data = self.sentiment_cache[pair]
            if datetime.now() - cache_time < timedelta(hours=1):
                return data
        
        try:
            # Get fresh sentiment
            currency = pair.split('/')[0]
            results = self.sentiment_collector.collect_all([currency])
            
            if currency in results['aggregated']:
                data = results['aggregated'][currency]
                self.sentiment_cache[pair] = (datetime.now(), data)
                return data
        except Exception as e:
            logger.error(f"Error fetching sentiment: {e}")
        
        return None
    
    def _get_macro(self) -> Optional[Dict]:
        """Get cached or fresh macro data"""
        # Check cache (4 hours)
        if self.macro_cache_time and datetime.now() - self.macro_cache_time < timedelta(hours=4):
            return self.macro_cache
        
        try:
            macro_data = self.macro_collector.collect_all()
            
            # Flatten for template
            macro = {}
            
            if macro_data.get('dxy'):
                macro['dxy_current'] = macro_data['dxy']['current']
                macro['dxy_change'] = macro_data['dxy']['change_pct']
                macro['dxy_signal'] = macro_data['dxy']['signal']
            
            if macro_data.get('spy'):
                macro['spy_current'] = macro_data['spy']['current']
                macro['spy_change'] = macro_data['spy']['change_pct']
                macro['spy_signal'] = macro_data['spy']['signal']
            
            if macro_data.get('vix'):
                macro['vix_current'] = macro_data['vix']['current']
                macro['vix_signal'] = macro_data['vix']['crypto_signal']
            
            if macro_data.get('btc_dominance'):
                macro['btc_dominance'] = macro_data['btc_dominance']['btc_dominance']
                macro['btc_dom_phase'] = macro_data['btc_dominance']['phase']
            
            self.macro_cache = macro
            self.macro_cache_time = datetime.now()
            
            return macro
            
        except Exception as e:
            logger.error(f"Error fetching macro: {e}")
        
        return None
    
    def _get_onchain(self, pair: str) -> Optional[Dict]:
        """Get on-chain metrics"""
        try:
            onchain_data = self.onchain_collector.collect_all()
            
            if onchain_data.get('fear_greed'):
                fg = onchain_data['fear_greed']
                return {
                    'fear_greed_value': fg['value'],
                    'fear_greed_classification': fg['classification'],
                    'fear_greed_signal': fg['signal']
                }
        except Exception as e:
            logger.error(f"Error fetching on-chain: {e}")
        
        return None
    
    def _get_strategy(self, pair: str, timeframe: str) -> Optional[Dict]:
        """Load strategy research if available"""
        filename = f"{pair.replace('/', '_')}_{timeframe}_research.json"
        filepath = Path('data/strategies') / filename
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    research = json.load(f)
                    
                if research.get('best_strategy'):
                    bs = research['best_strategy']
                    return {
                        'best_strategy_name': bs['name'],
                        'win_rate': bs['metrics'].get('win_rate', 0),
                        'avg_pnl': bs['metrics'].get('avg_pnl', 0),
                        'total_trades': bs['metrics'].get('total_trades', 0)
                    }
            except Exception as e:
                logger.error(f"Error loading strategy: {e}")
        
        return None


if __name__ == '__main__':
    analyzer = MarketAnalyzer()
    
    # Test with BTC/USD
    market_data = analyzer.get_current_market_data('BTC/USD')
    
    print("\n" + "="*70)
    print("MARKET DATA")
    print("="*70)
    print(json.dumps(market_data, indent=2, default=str))