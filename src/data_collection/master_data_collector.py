"""
Master orchestrator for comprehensive data collection
Manages OHLCV, sentiment, macro, on-chain, and strategy research
"""

import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MasterCollector:
    """Orchestrate all data collection"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        
    def collect_ohlcv_top_coins(self, limit: int = 31):
        """Download top coins by market cap"""
        logger.info(f"📥 Starting OHLCV download for top {limit} coins...")
        
        # Top 31 pairs from your discovery
        pairs = [
            'BTC/USD', 'ETH/USD', 'XRP/USD', 'BNB/USD', 'SOL/USD',
            'DOGE/USD', 'BCH/USD', 'ADA/USD', 'LINK/USD', 'XLM/USD',
            'ZEC/USD', 'HBAR/USD', 'LTC/USD', 'AVAX/USD', 'SHIB/USD',
            'SUI/USD', 'TON/USD', 'DOT/USD', 'UNI/USD', 'AAVE/USD',
            'PEPE/USD', 'TAO/USD'
        ][:limit]
        
        # Download in batches to avoid overwhelming the system
        batch_size = 5
        timeframes = ['1h', '4h', '1d']
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            logger.info(f"📦 Batch {i//batch_size + 1}: {batch}")
            
            cmd = [
                'python', 'src/data_collection/ohlcv_downloader.py',
                '--pairs'] + batch + [
                '--timeframes'] + timeframes + [
                '--since', '2020-01-01',
                '--exchange', 'coinbase'
            ]
            
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"✅ Batch {i//batch_size + 1} complete")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Batch {i//batch_size + 1} failed: {e}")
            
            # Rate limiting
            time.sleep(5)
        
        logger.info("✅ OHLCV collection complete")
    
    def collect_sentiment(self):
        """Collect sentiment data"""
        logger.info("📰 Collecting sentiment data...")
        
        try:
            subprocess.run([
                'python', 'src/data_collection/sentiment_collector.py'
            ], check=True)
            logger.info("✅ Sentiment collection complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"⚠️  Sentiment collection failed: {e}")
            logger.info("Continuing without sentiment data...")
    
    def collect_macro(self):
        """Collect macro indicators"""
        logger.info("🌍 Collecting macro indicators...")
        
        try:
            subprocess.run([
                'python', 'src/data_collection/macro_collector.py'
            ], check=True)
            logger.info("✅ Macro collection complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Macro collection failed: {e}")
    
    def collect_onchain(self):
        """Collect on-chain data"""
        logger.info("⛓️  Collecting on-chain data...")
        
        try:
            subprocess.run([
                'python', 'src/data_collection/onchain_collector.py'
            ], check=True)
            logger.info("✅ On-chain collection complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"⚠️  On-chain collection failed: {e}")
            logger.info("Continuing without on-chain data...")
    
    def research_strategies(self):
        """Run strategy research on downloaded pairs"""
        logger.info("📊 Running strategy research...")
        
        try:
            subprocess.run([
                'python', 'src/data_collection/strategy_researcher.py'
            ], check=True)
            logger.info("✅ Strategy research complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"⚠️  Strategy research failed: {e}")
    
    def build_enhanced_dataset(self, max_examples: int = 10000):
        """Build enhanced training dataset"""
        logger.info("🔨 Building enhanced dataset...")
        
        # Get all downloaded pairs
        ohlcv_dir = Path('data/ohlcv')
        pairs = list(set([
            f.stem.replace('_1h', '').replace('_4h', '').replace('_1d', '').replace('_', '/')
            for f in ohlcv_dir.glob('*.csv')
        ]))
        
        logger.info(f"Found {len(pairs)} pairs: {pairs[:10]}...")
        
        try:
            subprocess.run([
                'python', 'src/data_collection/enhanced_dataset_builder.py',
                '--pairs'] + pairs + [
                '--timeframes', '1h',
                '--max-examples', str(max_examples)
            ], check=True)
            logger.info("✅ Dataset build complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Dataset build failed: {e}")
    
    def run_full_pipeline(self, skip_ohlcv: bool = False):
        """Run complete data collection pipeline"""
        start_time = datetime.now()
        logger.info("="*70)
        logger.info("🚀 STARTING FULL DATA COLLECTION PIPELINE")
        logger.info("="*70)
        
        steps = [
            ("OHLCV Data", self.collect_ohlcv_top_coins if not skip_ohlcv else lambda: logger.info("⏭️  Skipping OHLCV")),
            ("Macro Indicators", self.collect_macro),
            ("Sentiment Analysis", self.collect_sentiment),
            ("On-Chain Metrics", self.collect_onchain),
            ("Strategy Research", self.research_strategies),
            ("Enhanced Dataset", self.build_enhanced_dataset)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*70}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*70}")
            
            try:
                step_func()
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                logger.info("Continuing to next step...")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info(f"Duration: {duration}")
        logger.info("="*70)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate collection summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ohlcv_files': len(list(Path('data/ohlcv').glob('*.csv'))),
            'sentiment_files': len(list(Path('data/sentiment').glob('*.jsonl'))),
            'macro_files': len(list(Path('data/macro').glob('*.json'))),
            'onchain_files': len(list(Path('data/onchain').glob('*.json'))),
            'strategy_files': len(list(Path('data/strategies').glob('*.json'))),
            'dataset_files': {
                'train': Path('datasets/enhanced_sft_train.jsonl').exists(),
                'eval': Path('datasets/enhanced_sft_eval.jsonl').exists()
            }
        }
        
        # Count training examples
        if summary['dataset_files']['train']:
            with open('datasets/enhanced_sft_train.jsonl', 'r') as f:
                summary['training_examples'] = sum(1 for _ in f)
        
        if summary['dataset_files']['eval']:
            with open('datasets/enhanced_sft_eval.jsonl', 'r') as f:
                summary['eval_examples'] = sum(1 for _ in f)
        
        # Save summary
        with open('logs/collection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n📊 COLLECTION SUMMARY:")
        logger.info(json.dumps(summary, indent=2))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Master data collection orchestrator')
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--skip-ohlcv', action='store_true', help='Skip OHLCV download')
    parser.add_argument('--ohlcv-only', action='store_true', help='Only download OHLCV')
    parser.add_argument('--macro-only', action='store_true', help='Only collect macro')
    parser.add_argument('--build-dataset', action='store_true', help='Only build dataset')
    
    args = parser.parse_args()
    
    collector = MasterCollector()
    
    if args.full:
        collector.run_full_pipeline(skip_ohlcv=args.skip_ohlcv)
    elif args.ohlcv_only:
        collector.collect_ohlcv_top_coins()
    elif args.macro_only:
        collector.collect_macro()
    elif args.build_dataset:
        collector.build_enhanced_dataset()
    else:
        # Default: show help
        parser.print_help()


if __name__ == '__main__':
    main()