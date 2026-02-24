"""
On-chain data collector
Fetches whale movements, exchange flows, MVRV ratios
"""

import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnChainCollector:
    """Collect on-chain metrics"""
    
    def __init__(self, output_dir: str = 'data/onchain'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_glassnode_metrics(self, asset: str = 'BTC') -> Dict:
        """
        Fetch Glassnode metrics (requires API key)
        Free tier available at glassnode.com
        """
        # Placeholder - requires Glassnode API key
        logger.info(f"Fetching Glassnode metrics for {asset}")
        
        metrics = {
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'exchange_netflow': None,
            'whale_ratio': None,
            'active_addresses': None,
            'transaction_volume': None,
            'mvrv_ratio': None,
            'source': 'glassnode'
        }
        
        return metrics
    
    def get_exchange_flows(self, asset: str = 'BTC') -> Dict:
        """
        Calculate exchange inflows/outflows
        Inflow = bearish (selling pressure)
        Outflow = bullish (accumulation)
        """
        logger.info(f"Analyzing exchange flows for {asset}")
        
        flows = {
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'total_inflow_24h': None,
            'total_outflow_24h': None,
            'net_flow_24h': None,
            'top_exchanges': [],
            'interpretation': None
        }
        
        if flows['net_flow_24h'] and flows['net_flow_24h'] < 0:
            flows['interpretation'] = 'bullish_accumulation'
        elif flows['net_flow_24h'] and flows['net_flow_24h'] > 0:
            flows['interpretation'] = 'bearish_distribution'
        else:
            flows['interpretation'] = 'neutral'
        
        return flows
    
    def get_fear_greed_index(self) -> Dict:
        """
        Fetch Crypto Fear & Greed Index
        Free API from alternative.me
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['data']:
                latest = data['data'][0]
                value = int(latest['value'])
                
                # Interpret
                if value >= 75:
                    interpretation = 'extreme_greed'
                    signal = 'bearish'  # Contrarian indicator
                elif value >= 55:
                    interpretation = 'greed'
                    signal = 'slightly_bearish'
                elif value >= 45:
                    interpretation = 'neutral'
                    signal = 'neutral'
                elif value >= 25:
                    interpretation = 'fear'
                    signal = 'slightly_bullish'
                else:
                    interpretation = 'extreme_fear'
                    signal = 'bullish'  # Contrarian buy signal
                
                return {
                    'value': value,
                    'classification': latest['value_classification'],
                    'interpretation': interpretation,
                    'signal': signal,
                    'timestamp': latest['timestamp']
                }
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
        
        return None
    
    def collect_all(self, assets: List[str] = ['BTC', 'ETH']) -> Dict:
        """Collect all on-chain data"""
        logger.info("⛓️  Collecting on-chain data...")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'fear_greed': self.get_fear_greed_index(),
            'assets': {}
        }
        
        for asset in assets:
            data['assets'][asset] = {
                'glassnode': self.get_glassnode_metrics(asset),
                'exchange_flows': self.get_exchange_flows(asset)
            }
        
        # Save - FIX: Use json.dump() not json.dumps()
        output_file = self.output_dir / f"onchain_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved on-chain data to {output_file}")
        
        return data


if __name__ == '__main__':
    collector = OnChainCollector()
    data = collector.collect_all(['BTC', 'ETH', 'SOL'])
    
    print(json.dumps(data, indent=2))
