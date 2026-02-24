"""
Trading strategy research and backtesting
Analyzes successful patterns and strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyResearcher:
    """Research and document successful trading strategies"""
    
    def __init__(self, ohlcv_dir: str = 'data/ohlcv', output_dir: str = 'data/strategies'):
        self.ohlcv_dir = Path(ohlcv_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns"""
        patterns = []
        closes = df['close'].values
        
        # Detect double tops
        for i in range(20, len(df) - 20):
            window = closes[i-20:i+20]
            if closes[i] == max(window):
                for j in range(i+5, min(i+20, len(df))):
                    if abs(closes[j] - closes[i]) / closes[i] < 0.02:
                        patterns.append({
                            'pattern': 'double_top',
                            'index': i,
                            'timestamp': str(df.index[i]),
                            'price': closes[i],
                            'signal': 'bearish'
                        })
                        break
        
        # Detect double bottoms
        for i in range(20, len(df) - 20):
            window = closes[i-20:i+20]
            if closes[i] == min(window):
                for j in range(i+5, min(i+20, len(df))):
                    if abs(closes[j] - closes[i]) / closes[i] < 0.02:
                        patterns.append({
                            'pattern': 'double_bottom',
                            'index': i,
                            'timestamp': str(df.index[i]),
                            'price': closes[i],
                            'signal': 'bullish'
                        })
                        break
        
        return patterns
    
    def backtest_strategy(self, df: pd.DataFrame, strategy: str, **params) -> Dict:
        """Backtest a specific strategy"""
        results = {
            'strategy': strategy,
            'params': params,
            'trades': [],
            'metrics': {}
        }
        
        if strategy == 'rsi_mean_reversion':
            results = self._backtest_rsi(df, **params)
        elif strategy == 'macd_crossover':
            results = self._backtest_macd(df, **params)
        elif strategy == 'breakout':
            results = self._backtest_breakout(df, **params)
        
        return results
    
    def _backtest_rsi(self, df: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> Dict:
        """RSI mean reversion strategy"""
        df = df.copy()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        trades = []
        position = None
        
        for i in range(1, len(df)):
            if df['rsi'].iloc[i-1] > oversold and df['rsi'].iloc[i] <= oversold and position is None:
                position = {
                    'entry_idx': i,
                    'entry_price': df['close'].iloc[i],
                    'entry_time': str(df.index[i]),
                    'type': 'long'
                }
            elif position and (
                (df['rsi'].iloc[i-1] < overbought and df['rsi'].iloc[i] >= overbought) or
                (df['close'].iloc[i] - position['entry_price']) / position['entry_price'] > 0.05
            ):
                exit_price = df['close'].iloc[i]
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                
                trades.append({
                    **position,
                    'exit_idx': i,
                    'exit_price': exit_price,
                    'exit_time': str(df.index[i]),
                    'pnl_pct': pnl_pct,
                    'duration_candles': i - position['entry_idx']
                })
                position = None
        
        if trades:
            pnls = [t['pnl_pct'] for t in trades]
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            
            metrics = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(trades) * 100,
                'avg_pnl': float(np.mean(pnls)),
                'total_pnl': float(np.sum(pnls)),
                'max_win': float(max(pnls)),
                'max_loss': float(min(pnls)),
                'avg_duration': float(np.mean([t['duration_candles'] for t in trades]))
            }
        else:
            metrics = {'total_trades': 0}
        
        return {
            'strategy': 'rsi_mean_reversion',
            'params': {'oversold': oversold, 'overbought': overbought},
            'trades': trades,
            'metrics': metrics
        }
    
    def _backtest_macd(self, df: pd.DataFrame) -> Dict:
        """MACD crossover strategy"""
        df = df.copy()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        trades = []
        position = None
        
        for i in range(1, len(df)):
            if (df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1] and 
                df['macd'].iloc[i] > df['macd_signal'].iloc[i] and 
                position is None):
                position = {
                    'entry_idx': i,
                    'entry_price': df['close'].iloc[i],
                    'entry_time': str(df.index[i]),
                    'type': 'long'
                }
            elif position and (
                (df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1] and 
                 df['macd'].iloc[i] < df['macd_signal'].iloc[i]) or
                (df['close'].iloc[i] - position['entry_price']) / position['entry_price'] < -0.03
            ):
                exit_price = df['close'].iloc[i]
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                
                trades.append({
                    **position,
                    'exit_idx': i,
                    'exit_price': exit_price,
                    'exit_time': str(df.index[i]),
                    'pnl_pct': pnl_pct,
                    'duration_candles': i - position['entry_idx']
                })
                position = None
        
        if trades:
            pnls = [t['pnl_pct'] for t in trades]
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            
            metrics = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(trades) * 100,
                'avg_pnl': float(np.mean(pnls)),
                'total_pnl': float(np.sum(pnls))
            }
        else:
            metrics = {'total_trades': 0}
        
        return {
            'strategy': 'macd_crossover',
            'trades': trades,
            'metrics': metrics
        }
    
    def _backtest_breakout(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Breakout strategy"""
        df = df.copy()
        
        trades = []
        position = None
        
        for i in range(lookback, len(df)):
            resistance = df['high'].iloc[i-lookback:i].max()
            
            if df['close'].iloc[i] > resistance and position is None:
                position = {
                    'entry_idx': i,
                    'entry_price': df['close'].iloc[i],
                    'entry_time': str(df.index[i]),
                    'type': 'long',
                    'resistance_broken': resistance
                }
            elif position:
                pnl = (df['close'].iloc[i] - position['entry_price']) / position['entry_price']
                if pnl > 0.03 or pnl < -0.02:
                    exit_price = df['close'].iloc[i]
                    pnl_pct = pnl * 100
                    
                    trades.append({
                        **position,
                        'exit_idx': i,
                        'exit_price': exit_price,
                        'exit_time': str(df.index[i]),
                        'pnl_pct': pnl_pct,
                        'duration_candles': i - position['entry_idx']
                    })
                    position = None
        
        if trades:
            pnls = [t['pnl_pct'] for t in trades]
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            
            metrics = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
                'avg_pnl': float(np.mean(pnls)) if pnls else 0,
                'total_pnl': float(np.sum(pnls)) if pnls else 0
            }
        else:
            metrics = {'total_trades': 0}
        
        return {
            'strategy': 'breakout',
            'params': {'lookback': lookback},
            'trades': trades,
            'metrics': metrics
        }
    
    def research_pair(self, pair: str, timeframe: str) -> Dict:
        """Comprehensive research on a trading pair"""
        filename = f"{pair.replace('/', '_')}_{timeframe}.csv"
        filepath = self.ohlcv_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        logger.info(f"📊 Researching {pair} {timeframe}...")
        
        patterns = self.detect_patterns(df)
        
        strategies = {
            'rsi_mean_reversion': self.backtest_strategy(df, 'rsi_mean_reversion'),
            'macd_crossover': self.backtest_strategy(df, 'macd_crossover'),
            'breakout': self.backtest_strategy(df, 'breakout')
        }
        
        best_strategy = max(
            strategies.items(),
            key=lambda x: x[1]['metrics'].get('win_rate', 0) if x[1]['metrics'].get('total_trades', 0) > 10 else 0
        )
        
        research = {
            'pair': pair,
            'timeframe': timeframe,
            'candles': len(df),
            'date_range': {
                'start': str(df.index[0]),
                'end': str(df.index[-1])
            },
            'patterns': patterns,
            'strategies': strategies,
            'best_strategy': {
                'name': best_strategy[0],
                'metrics': best_strategy[1]['metrics']
            }
        }
        
        output_file = self.output_dir / f"{pair.replace('/', '_')}_{timeframe}_research.json"
        with open(output_file, 'w') as f:
            json.dump(research, f, indent=2)
        
        logger.info(f"💾 Saved research to {output_file}")
        
        return research


def main():
    """CLI entry point - research ALL pairs automatically"""
    researcher = StrategyResearcher()
    
    # Find all OHLCV files
    ohlcv_files = list(researcher.ohlcv_dir.glob('*_1h.csv'))
    
    pairs_researched = []
    
    for file in ohlcv_files:
        # Extract pair and timeframe from filename
        # Example: BTC_USD_1h.csv -> BTC/USD, 1h
        parts = file.stem.split('_')
        if len(parts) >= 3:
            pair = f"{parts[0]}/{parts[1]}"
            timeframe = parts[2]
            
            results = researcher.research_pair(pair, timeframe)
            
            if results:
                pairs_researched.append(pair)
                print(f"\n{'='*60}")
                print(f"STRATEGY RESEARCH: {pair}")
                print(f"{'='*60}")
                print(f"Best Strategy: {results['best_strategy']['name']}")
                print(f"Metrics: {results['best_strategy']['metrics']}")
    
    print(f"\n✅ Researched {len(pairs_researched)} pairs: {', '.join(pairs_researched)}")


if __name__ == '__main__':
    main()
