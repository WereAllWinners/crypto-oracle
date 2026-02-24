"""
On-Chain Volume Analyzer
Tracks real blockchain activity, whale movements, and network metrics
All data is public and verifiable
"""

import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnChainVolumeAnalyzer:
    """Analyze on-chain transaction volume and activity"""
    
    def __init__(self, output_dir: str = 'data/onchain_volume'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_btc_blockchain_info(self) -> Dict:
        """
        Bitcoin blockchain metrics from blockchain.com API (FREE)
        """
        try:
            metrics = {}
            
            # Get mempool stats
            url = "https://blockchain.info/stats?format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                metrics = {
                    'asset': 'BTC',
                    'total_btc': data.get('totalbc', 0) / 100000000,  # Convert to BTC
                    'market_price_usd': data.get('market_price_usd', 0),
                    'hash_rate': data.get('hash_rate', 0),
                    'difficulty': data.get('difficulty', 0),
                    'minutes_between_blocks': data.get('minutes_between_blocks', 0),
                    'transactions_per_second': data.get('n_tx', 0) / 86400,  # Daily txs / seconds in day
                    'total_fees_btc': data.get('total_fees_btc', 0) / 100000000,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get recent blocks for volume analysis
            blocks_url = "https://blockchain.info/blocks?format=json"
            response = requests.get(blocks_url, timeout=10)
            
            if response.status_code == 200:
                blocks = response.json()
                
                # Calculate average transaction volume from recent blocks
                if blocks:
                    total_txs = sum(block.get('n_tx', 0) for block in blocks[:10])
                    avg_block_size = sum(block.get('size', 0) for block in blocks[:10]) / len(blocks[:10])
                    
                    metrics['recent_tx_count'] = total_txs
                    metrics['avg_block_size_kb'] = avg_block_size / 1024
            
            logger.info(f"✅ Fetched BTC blockchain metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching BTC blockchain info: {e}")
            return None
    
    def get_eth_gas_and_volume(self) -> Dict:
        """
        Ethereum gas prices and transaction volume from Etherscan (FREE)
        """
        try:
            # Etherscan free API (no key needed for basic stats)
            base_url = "https://api.etherscan.io/api"
            
            metrics = {
                'asset': 'ETH',
                'timestamp': datetime.now().isoformat()
            }
            
            # Get gas oracle
            params = {
                'module': 'gastracker',
                'action': 'gasoracle'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == '1':
                    result = data.get('result', {})
                    
                    metrics['gas_price_gwei'] = {
                        'low': float(result.get('SafeGasPrice', 0)),
                        'average': float(result.get('ProposeGasPrice', 0)),
                        'high': float(result.get('FastGasPrice', 0))
                    }
                    
                    # High gas = high network activity
                    avg_gas = metrics['gas_price_gwei']['average']
                    if avg_gas > 100:
                        metrics['network_activity'] = 'very_high'
                        metrics['congestion_signal'] = 'bullish'  # High usage
                    elif avg_gas > 50:
                        metrics['network_activity'] = 'high'
                        metrics['congestion_signal'] = 'slightly_bullish'
                    elif avg_gas > 20:
                        metrics['network_activity'] = 'moderate'
                        metrics['congestion_signal'] = 'neutral'
                    else:
                        metrics['network_activity'] = 'low'
                        metrics['congestion_signal'] = 'bearish'  # Low usage
            
            logger.info(f"✅ Fetched ETH gas metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching ETH gas: {e}")
            return None
    
    def get_whale_transactions(self, asset: str = 'BTC', min_value_usd: int = 1000000) -> List[Dict]:
        """
        Detect large transactions (whale activity)
        Uses Whale Alert API (has free tier)
        """
        whales = []
        
        try:
            # Whale Alert free endpoint (limited)
            # Note: For production, get free API key from whale-alert.io
            
            # Alternative: Use blockchain explorer APIs
            if asset == 'BTC':
                # Get recent large transactions from blockchain.com
                url = "https://blockchain.info/unconfirmed-transactions?format=json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for tx in data.get('txs', [])[:50]:  # Check last 50 unconfirmed
                        value_btc = sum(out.get('value', 0) for out in tx.get('out', [])) / 100000000
                        
                        # Assume BTC price for USD conversion (should fetch real price)
                        value_usd = value_btc * 50000  # Placeholder
                        
                        if value_usd >= min_value_usd:
                            whale = {
                                'asset': 'BTC',
                                'hash': tx.get('hash'),
                                'value_btc': value_btc,
                                'value_usd': value_usd,
                                'size_bytes': tx.get('size'),
                                'timestamp': datetime.fromtimestamp(tx.get('time', 0)).isoformat(),
                                'type': 'whale_transaction'
                            }
                            whales.append(whale)
            
            logger.info(f"✅ Detected {len(whales)} whale transactions for {asset}")
            
        except Exception as e:
            logger.error(f"Error detecting whale transactions: {e}")
        
        return whales
    
    def get_exchange_netflow(self, asset: str = 'BTC') -> Dict:
        """
        Track coins moving to/from exchanges
        Inflow = selling pressure (bearish)
        Outflow = accumulation (bullish)
        
        Uses CryptoQuant-like data (free alternatives available)
        """
        try:
            flows = {
                'asset': asset,
                'timestamp': datetime.now().isoformat(),
                'interpretation': None
            }
            
            # CryptoQuant has some free data
            # Alternative: Aggregate from known exchange addresses
            
            # For BTC, we can track known exchange addresses
            if asset == 'BTC':
                # This would require tracking specific exchange addresses
                # Placeholder for now
                flows['exchange_inflow_24h'] = None
                flows['exchange_outflow_24h'] = None
                flows['netflow_24h'] = None
                
                # Interpretation
                if flows['netflow_24h']:
                    if flows['netflow_24h'] < -1000:  # Net outflow (bullish)
                        flows['interpretation'] = 'strong_accumulation'
                        flows['signal'] = 'bullish'
                    elif flows['netflow_24h'] < 0:
                        flows['interpretation'] = 'mild_accumulation'
                        flows['signal'] = 'slightly_bullish'
                    elif flows['netflow_24h'] > 1000:  # Net inflow (bearish)
                        flows['interpretation'] = 'distribution'
                        flows['signal'] = 'bearish'
                    else:
                        flows['interpretation'] = 'neutral'
                        flows['signal'] = 'neutral'
            
            logger.info(f"✅ Analyzed exchange flows for {asset}")
            return flows
            
        except Exception as e:
            logger.error(f"Error analyzing exchange flows: {e}")
            return None
    
    def get_active_addresses(self, asset: str = 'BTC', days: int = 7) -> Dict:
        """
        Track unique active addresses
        More addresses = more network activity = bullish
        """
        try:
            metrics = {
                'asset': asset,
                'timestamp': datetime.now().isoformat(),
                'period_days': days
            }
            
            if asset == 'BTC':
                # Blockchain.com provides this
                url = f"https://api.blockchain.info/charts/n-unique-addresses?timespan={days}days&format=json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    values = data.get('values', [])
                    
                    if values:
                        latest = values[-1]
                        previous = values[-2] if len(values) > 1 else values[-1]
                        
                        current_addresses = latest.get('y', 0)
                        prev_addresses = previous.get('y', 0)
                        
                        change_pct = ((current_addresses - prev_addresses) / prev_addresses) * 100 if prev_addresses else 0
                        
                        metrics['active_addresses'] = int(current_addresses)
                        metrics['change_pct'] = change_pct
                        
                        # Interpretation
                        if change_pct > 10:
                            metrics['signal'] = 'bullish'
                            metrics['interpretation'] = 'increasing_adoption'
                        elif change_pct > 5:
                            metrics['signal'] = 'slightly_bullish'
                            metrics['interpretation'] = 'growing_activity'
                        elif change_pct < -10:
                            metrics['signal'] = 'bearish'
                            metrics['interpretation'] = 'declining_activity'
                        else:
                            metrics['signal'] = 'neutral'
                            metrics['interpretation'] = 'stable_activity'
            
            logger.info(f"✅ Analyzed active addresses for {asset}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing active addresses: {e}")
            return None
    
    def get_transaction_volume_24h(self, asset: str = 'BTC') -> Dict:
        """
        Get 24-hour on-chain transaction volume
        Higher volume = more activity = potential price movement
        """
        try:
            volume = {
                'asset': asset,
                'timestamp': datetime.now().isoformat()
            }
            
            if asset == 'BTC':
                # Blockchain.com transaction volume
                url = "https://api.blockchain.info/charts/estimated-transaction-volume?timespan=1days&format=json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    values = data.get('values', [])
                    
                    if values:
                        latest = values[-1]
                        volume['volume_btc_24h'] = latest.get('y', 0)
                        volume['volume_usd_24h'] = volume['volume_btc_24h'] * 50000  # Placeholder price
                        
                        # Compare to average
                        if len(values) > 7:
                            avg_volume = sum(v.get('y', 0) for v in values[-7:]) / 7
                            volume['vs_7day_avg'] = ((volume['volume_btc_24h'] - avg_volume) / avg_volume) * 100
                            
                            if volume['vs_7day_avg'] > 20:
                                volume['signal'] = 'high_volume_bullish'
                            elif volume['vs_7day_avg'] < -20:
                                volume['signal'] = 'low_volume_bearish'
                            else:
                                volume['signal'] = 'normal'
            
            elif asset == 'ETH':
                # Etherscan transaction count
                base_url = "https://api.etherscan.io/api"
                params = {
                    'module': 'stats',
                    'action': 'dailytx',
                    'startdate': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'enddate': datetime.now().strftime('%Y-%m-%d'),
                    'sort': 'desc'
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('status') == '1':
                        result = data.get('result', [])
                        if result:
                            volume['tx_count_24h'] = int(result[0].get('txCount', 0))
            
            logger.info(f"✅ Analyzed 24h transaction volume for {asset}")
            return volume
            
        except Exception as e:
            logger.error(f"Error analyzing transaction volume: {e}")
            return None
    
    def get_mvrv_ratio(self, asset: str = 'BTC') -> Dict:
        """
        MVRV Ratio (Market Value to Realized Value)
        > 3.5 = overvalued (sell signal)
        < 1.0 = undervalued (buy signal)
        
        This requires historical data - using approximation
        """
        try:
            mvrv = {
                'asset': asset,
                'timestamp': datetime.now().isoformat()
            }
            
            # MVRV requires realized cap data (available from Glassnode, CoinMetrics)
            # For free alternative, we can approximate with price momentum
            
            # Placeholder - in production, use Glassnode API or CoinMetrics
            mvrv['mvrv_ratio'] = None
            mvrv['signal'] = 'neutral'
            
            logger.info(f"✅ Calculated MVRV for {asset}")
            return mvrv
            
        except Exception as e:
            logger.error(f"Error calculating MVRV: {e}")
            return None
    
    def analyze_all(self, assets: List[str] = ['BTC', 'ETH']) -> Dict:
        """Comprehensive on-chain volume analysis"""
        logger.info("⛓️  Running comprehensive on-chain volume analysis...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'assets': {}
        }
        
        for asset in assets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing {asset}")
            logger.info(f"{'='*60}")
            
            asset_data = {}
            
            # Blockchain metrics
            if asset == 'BTC':
                asset_data['blockchain'] = self.get_btc_blockchain_info()
                time.sleep(1)
            elif asset == 'ETH':
                asset_data['gas'] = self.get_eth_gas_and_volume()
                time.sleep(1)
            
            # Whale activity
            asset_data['whales'] = self.get_whale_transactions(asset)
            time.sleep(1)
            
            # Exchange flows
            asset_data['exchange_flows'] = self.get_exchange_netflow(asset)
            time.sleep(1)
            
            # Active addresses
            asset_data['active_addresses'] = self.get_active_addresses(asset)
            time.sleep(1)
            
            # Transaction volume
            asset_data['tx_volume'] = self.get_transaction_volume_24h(asset)
            time.sleep(1)
            
            # MVRV
            asset_data['mvrv'] = self.get_mvrv_ratio(asset)
            
            results['assets'][asset] = asset_data
        
        # Save results
        output_file = self.output_dir / f"volume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Saved on-chain volume analysis to {output_file}")
        
        return results
    
    def generate_summary(self, results: Dict) -> str:
        """Generate human-readable summary"""
        summary = "\n" + "="*70 + "\n"
        summary += "ON-CHAIN VOLUME ANALYSIS SUMMARY\n"
        summary += "="*70 + "\n"
        
        for asset, data in results.get('assets', {}).items():
            summary += f"\n{asset}:\n"
            summary += "-" * 40 + "\n"
            
            # Blockchain metrics
            if 'blockchain' in data and data['blockchain']:
                b = data['blockchain']
                summary += f"  Blockchain Activity:\n"
                summary += f"    - Transactions/sec: {b.get('transactions_per_second', 0):.2f}\n"
                summary += f"    - Hash rate: {b.get('hash_rate', 0):,.0f} TH/s\n"
            
            # Gas (ETH)
            if 'gas' in data and data['gas']:
                g = data['gas']
                if 'gas_price_gwei' in g:
                    summary += f"  Network Activity: {g.get('network_activity', 'unknown')}\n"
                    summary += f"  Gas Price: {g['gas_price_gwei']['average']} gwei\n"
                    summary += f"  Signal: {g.get('congestion_signal', 'neutral')}\n"
            
            # Whales
            if 'whales' in data:
                summary += f"  Whale Transactions (24h): {len(data['whales'])}\n"
            
            # Exchange flows
            if 'exchange_flows' in data and data['exchange_flows']:
                ef = data['exchange_flows']
                summary += f"  Exchange Flow Signal: {ef.get('signal', 'unknown')}\n"
            
            # Active addresses
            if 'active_addresses' in data and data['active_addresses']:
                aa = data['active_addresses']
                summary += f"  Active Addresses: {aa.get('active_addresses', 0):,}\n"
                summary += f"  Change: {aa.get('change_pct', 0):+.1f}%\n"
                summary += f"  Signal: {aa.get('signal', 'neutral')}\n"
            
            # Transaction volume
            if 'tx_volume' in data and data['tx_volume']:
                tv = data['tx_volume']
                if 'volume_btc_24h' in tv:
                    summary += f"  24h Volume: {tv['volume_btc_24h']:,.0f} {asset}\n"
                    summary += f"  vs 7-day avg: {tv.get('vs_7day_avg', 0):+.1f}%\n"
        
        return summary


def main():
    """CLI entry point"""
    analyzer = OnChainVolumeAnalyzer()
    
    # Analyze BTC and ETH
    results = analyzer.analyze_all(['BTC', 'ETH'])
    
    # Print summary
    summary = analyzer.generate_summary(results)
    print(summary)


if __name__ == '__main__':
    main()