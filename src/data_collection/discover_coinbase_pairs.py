"""
Discover all tradeable pairs on Coinbase
Automatically finds everything you can trade
"""

import ccxt
import json
from pathlib import Path
import logging
from typing import List, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinbaseDiscovery:
    """Discover all tradeable assets on Coinbase"""
    
    def __init__(self, output_dir: str = 'config'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exchange = ccxt.coinbase({'enableRateLimit': True})
    
    def discover_all_pairs(self) -> Dict:
        """Discover all tradeable pairs on Coinbase"""
        logger.info("🔍 Discovering all Coinbase pairs...")
        
        markets = self.exchange.load_markets()
        
        # Categorize pairs
        usd_pairs = []
        usdt_pairs = []
        usdc_pairs = []
        btc_pairs = []
        eth_pairs = []
        other_pairs = []
        
        for symbol, market in markets.items():
            # Only spot markets
            if not market.get('spot', False):
                continue
            
            pair_info = {
                'symbol': symbol,
                'base': market['base'],
                'quote': market['quote'],
                'active': market.get('active', True),
                'type': market.get('type', 'spot'),
                'limits': {
                    'amount': market.get('limits', {}).get('amount', {}),
                    'cost': market.get('limits', {}).get('cost', {})
                }
            }
            
            # Categorize
            if symbol.endswith('/USD'):
                usd_pairs.append(pair_info)
            elif symbol.endswith('/USDT'):
                usdt_pairs.append(pair_info)
            elif symbol.endswith('/USDC'):
                usdc_pairs.append(pair_info)
            elif symbol.endswith('/BTC'):
                btc_pairs.append(pair_info)
            elif symbol.endswith('/ETH'):
                eth_pairs.append(pair_info)
            else:
                other_pairs.append(pair_info)
        
        discovery = {
            'timestamp': datetime.now().isoformat(),
            'exchange': 'coinbase',
            'total_pairs': len(markets),
            'spot_pairs': len(usd_pairs) + len(usdt_pairs) + len(usdc_pairs) + len(btc_pairs) + len(eth_pairs) + len(other_pairs),
            'categories': {
                'USD': {
                    'count': len(usd_pairs),
                    'pairs': usd_pairs
                },
                'USDT': {
                    'count': len(usdt_pairs),
                    'pairs': usdt_pairs
                },
                'USDC': {
                    'count': len(usdc_pairs),
                    'pairs': usdc_pairs
                },
                'BTC': {
                    'count': len(btc_pairs),
                    'pairs': btc_pairs
                },
                'ETH': {
                    'count': len(eth_pairs),
                    'pairs': eth_pairs
                },
                'OTHER': {
                    'count': len(other_pairs),
                    'pairs': other_pairs
                }
            }
        }
        
        # Save full discovery
        output_file = self.output_dir / 'coinbase_all_pairs.json'
        with open(output_file, 'w') as f:
            json.dump(discovery, f, indent=2)
        
        logger.info(f"💾 Saved {discovery['spot_pairs']} pairs to {output_file}")
        
        return discovery
    
    def get_recommended_pairs(
        self,
        categories: List[str] = ['USD', 'USDT'],
        min_volume_rank: int = 100
    ) -> List[str]:
        """
        Get recommended trading pairs based on liquidity and category
        
        Args:
            categories: Quote currencies to include (USD, USDT, USDC, BTC, ETH)
            min_volume_rank: Only include top N by volume
        """
        discovery = self.discover_all_pairs()
        
        recommended = []
        
        for category in categories:
            if category in discovery['categories']:
                pairs = discovery['categories'][category]['pairs']
                # For now, just take all active pairs
                # In production, you'd filter by volume
                active_pairs = [p['symbol'] for p in pairs if p['active']]
                recommended.extend(active_pairs[:min_volume_rank])
        
        logger.info(f"✅ Selected {len(recommended)} recommended pairs")
        
        # Save recommended list
        output_file = self.output_dir / 'recommended_pairs.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(recommended))
        
        logger.info(f"💾 Saved to {output_file}")
        
        return recommended
    
    def get_top_by_market_cap(self, limit: int = 50) -> List[str]:
        """
        Get top cryptocurrencies by market cap that are on Coinbase
        
        Args:
            limit: Number of top coins to return
        """
        import requests
        
        try:
            # Get top coins by market cap from CoinGecko
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            coins = response.json()
            
            # Get Coinbase pairs
            markets = self.exchange.load_markets()
            coinbase_bases = set([m['base'] for m in markets.values()])
            
            # Match top coins with Coinbase pairs
            top_pairs = []
            for coin in coins:
                symbol = coin['symbol'].upper()
                
                # Try to find on Coinbase
                if symbol in coinbase_bases:
                    # Prefer USD pairs
                    if f"{symbol}/USD" in markets:
                        top_pairs.append(f"{symbol}/USD")
                    elif f"{symbol}/USDT" in markets:
                        top_pairs.append(f"{symbol}/USDT")
                    elif f"{symbol}/USDC" in markets:
                        top_pairs.append(f"{symbol}/USDC")
            
            logger.info(f"✅ Found {len(top_pairs)} top market cap coins on Coinbase")
            
            return top_pairs
            
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []
    
    def generate_download_command(
        self,
        pairs: List[str],
        timeframes: List[str] = ['1h', '4h', '1d'],
        since: str = '2020-01-01'
    ) -> str:
        """Generate download command for discovered pairs"""
        
        cmd = f"""python src/data_collection/ohlcv_downloader.py \\
  --pairs {' '.join(pairs)} \\
  --timeframes {' '.join(timeframes)} \\
  --since {since} \\
  --exchange coinbase"""
        
        return cmd


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover Coinbase trading pairs')
    parser.add_argument('--mode', choices=['all', 'recommended', 'top'], default='all',
                       help='Discovery mode')
    parser.add_argument('--limit', type=int, default=50,
                       help='Limit for top coins')
    
    args = parser.parse_args()
    
    discovery = CoinbaseDiscovery()
    
    if args.mode == 'all':
        results = discovery.discover_all_pairs()
        
        print("\n" + "="*70)
        print("COINBASE PAIRS DISCOVERY")
        print("="*70)
        print(f"Total spot pairs: {results['spot_pairs']}")
        print(f"\nBy quote currency:")
        for cat, data in results['categories'].items():
            if data['count'] > 0:
                print(f"  {cat}: {data['count']} pairs")
                # Show first 10
                for pair in data['pairs'][:10]:
                    print(f"    - {pair['symbol']}")
                if data['count'] > 10:
                    print(f"    ... and {data['count'] - 10} more")
    
    elif args.mode == 'recommended':
        pairs = discovery.get_recommended_pairs()
        print(f"\n✅ Recommended {len(pairs)} pairs")
        print("Saved to: config/recommended_pairs.txt")
    
    elif args.mode == 'top':
        pairs = discovery.get_top_by_market_cap(limit=args.limit)
        print(f"\n✅ Top {len(pairs)} by market cap:")
        for pair in pairs:
            print(f"  - {pair}")
        
        # Generate download command
        cmd = discovery.generate_download_command(pairs)
        print("\n📥 To download all these pairs, run:")
        print(cmd)


if __name__ == '__main__':
    main()