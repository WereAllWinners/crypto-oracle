"""
Macro data collector
Fetches DXY, SPY, VIX, correlations, Bitcoin dominance
"""

import requests
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroCollector:
    """Collect macro economic indicators"""
    
    def __init__(self, output_dir: str = 'data/macro'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dxy(self, days_back: int = 30) -> Dict:
        """
        Fetch US Dollar Index (DXY)
        Inverse correlation with crypto - strong dollar = weak crypto
        """
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[0]
            change = ((current - prev) / prev) * 100
            
            # Interpret
            if change > 2:
                signal = 'bearish_crypto'  # Strong dollar = crypto down
            elif change < -2:
                signal = 'bullish_crypto'  # Weak dollar = crypto up
            else:
                signal = 'neutral'
            
            return {
                'symbol': 'DXY',
                'current': float(current),
                'change_pct': float(change),
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching DXY: {e}")
            return None
    
    def get_spy(self, days_back: int = 30) -> Dict:
        """
        Fetch S&P 500 (SPY)
        Crypto has positive correlation with equities
        """
        try:
            ticker = yf.Ticker("SPY")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[0]
            change = ((current - prev) / prev) * 100
            
            # Interpret
            if change > 3:
                signal = 'bullish_crypto'  # Risk-on = crypto up
            elif change < -3:
                signal = 'bearish_crypto'  # Risk-off = crypto down
            else:
                signal = 'neutral'
            
            return {
                'symbol': 'SPY',
                'current': float(current),
                'change_pct': float(change),
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching SPY: {e}")
            return None
    
    def get_vix(self) -> Dict:
        """
        Fetch VIX (Fear Index)
        High VIX = market fear = usually bad for crypto
        """
        try:
            ticker = yf.Ticker("^VIX")
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
            
            current = hist['Close'].iloc[-1]
            
            # Interpret VIX levels
            if current > 30:
                signal = 'extreme_fear'
                crypto_signal = 'bearish'
            elif current > 20:
                signal = 'elevated_fear'
                crypto_signal = 'slightly_bearish'
            elif current < 12:
                signal = 'complacency'
                crypto_signal = 'slightly_bullish'  # Can reverse
            else:
                signal = 'normal'
                crypto_signal = 'neutral'
            
            return {
                'symbol': 'VIX',
                'current': float(current),
                'signal': signal,
                'crypto_signal': crypto_signal,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return None
    
    def get_btc_dominance(self) -> Dict:
        """
        Fetch Bitcoin dominance
        High dominance = BTC season, Low = Alt season
        """
        try:
            # CoinGecko API (free, no key needed)
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            btc_dom = data['data']['market_cap_percentage']['btc']
            
            # Interpret
            if btc_dom > 50:
                phase = 'btc_season'
                signal = 'bullish_btc_bearish_alts'
            elif btc_dom < 40:
                phase = 'alt_season'
                signal = 'bearish_btc_bullish_alts'
            else:
                phase = 'transition'
                signal = 'neutral'
            
            return {
                'btc_dominance': float(btc_dom),
                'phase': phase,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching BTC dominance: {e}")
            return None
    
    def calculate_correlation(
        self,
        crypto_symbol: str,
        macro_symbol: str,
        days: int = 30
    ) -> float:
        """Calculate correlation between crypto and macro asset"""
        try:
            # Fetch both
            crypto = yf.Ticker(f"{crypto_symbol}-USD")
            macro = yf.Ticker(macro_symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            crypto_hist = crypto.history(start=start_date, end=end_date)
            macro_hist = macro.history(start=start_date, end=end_date)
            
            if crypto_hist.empty or macro_hist.empty:
                return None
            
            # Align dates
            crypto_close = crypto_hist['Close'].pct_change()
            macro_close = macro_hist['Close'].pct_change()
            
            # Calculate correlation
            correlation = crypto_close.corr(macro_close)
            
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None
    
    def get_fed_rate(self) -> Dict:
        """
        Get Federal Reserve interest rate
        Uses FRED API (free)
        """
        try:
            # Federal Funds Rate from FRED
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'DFF',  # Federal Funds Rate
                'api_key': 'demo',  # Use demo key or get free key
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('observations'):
                    latest = data['observations'][0]
                    rate = float(latest['value'])
                    
                    # Interpret
                    if rate > 4:
                        signal = 'bearish_crypto'  # High rates = risk-off
                    elif rate < 2:
                        signal = 'bullish_crypto'  # Low rates = risk-on
                    else:
                        signal = 'neutral'
                    
                    return {
                        'fed_rate': rate,
                        'date': latest['date'],
                        'signal': signal,
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error fetching Fed rate: {e}")
        
        return None
    
    def get_total_market_cap(self) -> Dict:
        """Get total crypto market cap from CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            total_mcap = data['data']['total_market_cap']['usd']
            mcap_change_24h = data['data']['market_cap_change_percentage_24h_usd']
            
            return {
                'total_market_cap_usd': total_mcap,
                'change_24h_pct': mcap_change_24h,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching market cap: {e}")
            return None
    
    def collect_all(self) -> Dict:
        """Collect all macro indicators"""
        logger.info("🌍 Collecting macro indicators...")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'dxy': self.get_dxy(),
            'spy': self.get_spy(),
            'vix': self.get_vix(),
            'btc_dominance': self.get_btc_dominance(),
            'fed_rate': self.get_fed_rate(),
            'crypto_market': self.get_total_market_cap(),
            'correlations': {
                'btc_spy': self.calculate_correlation('BTC', 'SPY'),
                'btc_dxy': self.calculate_correlation('BTC', 'DX-Y.NYB'),
                'eth_spy': self.calculate_correlation('ETH', 'SPY')
            }
        }
        
        # Save
        output_file = self.output_dir / f"macro_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved macro data to {output_file}")
        
        return data


def main():
    """CLI entry point"""
    collector = MacroCollector()
    data = collector.collect_all()
    
    print("\n" + "="*60)
    print("MACRO INDICATORS SUMMARY")
    print("="*60)
    
    if data['dxy']:
        print(f"\nDXY: ${data['dxy']['current']:.2f} ({data['dxy']['change_pct']:+.2f}%) - {data['dxy']['signal']}")
    
    if data['spy']:
        print(f"SPY: ${data['spy']['current']:.2f} ({data['spy']['change_pct']:+.2f}%) - {data['spy']['signal']}")
    
    if data['vix']:
        print(f"VIX: {data['vix']['current']:.2f} - {data['vix']['signal']}")
    
    if data['btc_dominance']:
        print(f"\nBTC Dominance: {data['btc_dominance']['btc_dominance']:.2f}% - {data['btc_dominance']['phase']}")
    
    if data['correlations']['btc_spy']:
        print(f"\nBTC-SPY Correlation: {data['correlations']['btc_spy']:.2f}")


if __name__ == '__main__':
    main()