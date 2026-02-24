"""
Batch Analyzer - Analyze multiple coins simultaneously
Rank by confidence and opportunity
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate

from crypto_oracle import CryptoOracle
from market_analyzer import MarketAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchAnalyzer:
    """Analyze multiple trading pairs and rank opportunities"""
    
    def __init__(self, model_path: str = "models/crypto-oracle-qwen-32b/final_model"):
        self.oracle = CryptoOracle(model_path=model_path)
        self.market_analyzer = MarketAnalyzer()
    
    def analyze_pairs(
        self,
        pairs: List[str],
        max_workers: int = 4,
        filter_decision: str = None,
        min_confidence: int = None
    ) -> List[Dict]:
        """
        Analyze multiple pairs in parallel
        
        Args:
            pairs: List of trading pairs
            max_workers: Number of parallel workers
            filter_decision: Only return specific decisions (BUY/SELL/HOLD)
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of predictions sorted by confidence
        """
        logger.info(f"🔍 Analyzing {len(pairs)} pairs...")
        
        predictions = []
        
        # Parallel analysis
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(self._analyze_single, pair): pair
                for pair in pairs
            }
            
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        predictions.append(result)
                        logger.info(f"✅ {pair}: {result['recommendation']['decision']} ({result['recommendation']['confidence']}%)")
                except Exception as e:
                    logger.error(f"❌ Error analyzing {pair}: {e}")
        
        # Filter results
        if filter_decision:
            predictions = [p for p in predictions if p['recommendation']['decision'] == filter_decision.upper()]
        
        if min_confidence:
            predictions = [p for p in predictions if p['recommendation'].get('confidence', 0) >= min_confidence]
        
        # Sort by confidence (descending)
        predictions.sort(key=lambda x: x['recommendation'].get('confidence', 0), reverse=True)
        
        logger.info(f"✅ Analysis complete: {len(predictions)} results")
        
        return predictions
    
    def _analyze_single(self, pair: str) -> Dict:
        """Analyze a single pair"""
        try:
            # Get market data
            market_data = self.market_analyzer.get_current_market_data(pair)
            
            # Generate prediction
            prediction = self.oracle.predict(market_data, stream=False)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}")
            return None
    
    def generate_report(self, predictions: List[Dict], output_format: str = 'table') -> str:
        """
        Generate formatted report
        
        Args:
            predictions: List of predictions
            output_format: 'table', 'json', or 'html'
        
        Returns:
            Formatted report string
        """
        if output_format == 'json':
            return json.dumps(predictions, indent=2, default=str)
        
        elif output_format == 'table':
            # Create table
            headers = ['Pair', 'Decision', 'Confidence', 'Entry', 'Stop Loss', 'Take Profit', 'Price']
            rows = []
            
            for pred in predictions:
                rec = pred['recommendation']
                market = pred['market_data']
                
                rows.append([
                    pred['pair'],
                    rec['decision'],
                    f"{rec.get('confidence', 'N/A')}%",
                    f"${rec.get('entry_price', 'N/A'):,.2f}" if rec.get('entry_price') else 'N/A',
                    f"${rec.get('stop_loss', 'N/A'):,.2f}" if rec.get('stop_loss') else 'N/A',
                    f"${rec.get('take_profit', 'N/A'):,.2f}" if rec.get('take_profit') else 'N/A',
                    f"${market['price']:,.2f}"
                ])
            
            return tabulate(rows, headers=headers, tablefmt='grid')
        
        elif output_format == 'html':
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Oracle - Batch Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        h1 { color: #00ff88; }
        .timestamp { color: #888; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #2a2a2a; padding: 12px; text-align: left; border: 1px solid #444; }
        td { padding: 12px; border: 1px solid #444; }
        tr:hover { background: #2a2a2a; }
        .buy { color: #00ff88; font-weight: bold; }
        .sell { color: #ff4444; font-weight: bold; }
        .hold { color: #ffaa00; font-weight: bold; }
        .high-conf { background: #003322; }
        .med-conf { background: #332200; }
    </style>
</head>
<body>
    <h1>🔮 Crypto Oracle - Batch Analysis Report</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    <table>
        <thead>
            <tr>
                <th>Pair</th>
                <th>Decision</th>
                <th>Confidence</th>
                <th>Current Price</th>
                <th>Entry Price</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
            </tr>
        </thead>
        <tbody>
"""
            
            for pred in predictions:
                rec = pred['recommendation']
                market = pred['market_data']
                
                decision = rec['decision']
                conf = rec.get('confidence', 0)
                
                # Style classes
                decision_class = decision.lower()
                conf_class = 'high-conf' if conf >= 75 else 'med-conf' if conf >= 65 else ''
                
                html += f"""
            <tr class="{conf_class}">
                <td><strong>{pred['pair']}</strong></td>
                <td class="{decision_class}">{decision}</td>
                <td>{conf}%</td>
                <td>${market['price']:,.2f}</td>
                <td>${rec.get('entry_price', 'N/A'):,.2f if rec.get('entry_price') else 'N/A'}</td>
                <td>${rec.get('stop_loss', 'N/A'):,.2f if rec.get('stop_loss') else 'N/A'}</td>
                <td>${rec.get('take_profit', 'N/A'):,.2f if rec.get('take_profit') else 'N/A'}</td>
            </tr>
"""
            
            html += """
        </tbody>
    </table>
</body>
</html>
"""
            return html
        
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def save_report(self, predictions: List[Dict], filename: str = None, format: str = 'json'):
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = 'html' if format == 'html' else 'json' if format == 'json' else 'txt'
            filename = f"reports/batch_analysis_{timestamp}.{ext}"
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report(predictions, output_format=format)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"💾 Report saved to {output_path}")
        
        return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch analyze crypto pairs')
    parser.add_argument('--pairs', nargs='+', help='Trading pairs to analyze')
    parser.add_argument('--top', type=int, help='Analyze top N coins by market cap')
    parser.add_argument('--filter', choices=['BUY', 'SELL', 'HOLD'], help='Filter by decision')
    parser.add_argument('--min-confidence', type=int, help='Minimum confidence threshold')
    parser.add_argument('--format', choices=['table', 'json', 'html'], default='table', help='Output format')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    parser.add_argument('--model', default='models/crypto-oracle-qwen-32b/final_model', help='Model path')
    
    args = parser.parse_args()
    
    # Determine pairs
    if args.pairs:
        pairs = args.pairs
    elif args.top:
        # Top coins by market cap
        pairs = [
            'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD',
            'ADA/USD', 'LINK/USD', 'AVAX/USD', 'DOT/USD', 'UNI/USD'
        ][:args.top]
    else:
        # Default top 5
        pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD']
    
    # Analyze
    analyzer = BatchAnalyzer(model_path=args.model)
    predictions = analyzer.analyze_pairs(
        pairs,
        filter_decision=args.filter,
        min_confidence=args.min_confidence
    )
    
    # Generate report
    report = analyzer.generate_report(predictions, output_format=args.format)
    
    print("\n" + "="*70)
    print("BATCH ANALYSIS REPORT")
    print("="*70)
    print(report)
    
    # Save if requested
    if args.save:
        analyzer.save_report(predictions, format=args.format)