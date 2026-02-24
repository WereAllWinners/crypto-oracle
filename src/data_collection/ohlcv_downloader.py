"""
OHLCV Data Downloader - FIXED VERSION
Handles "reaching now" gracefully
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OHLCVDownloader:
    """Download OHLCV data from crypto exchanges"""
    
    def __init__(self, exchange_name: str = 'coinbase', output_dir: str = 'data/ohlcv'):
        self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / '.download_progress.json'
        self.progress = self._load_progress()
    
    def _load_progress(self) -> dict:
        """Load download progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_progress(self):
        """Save download progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def download_pair(
        self,
        pair: str,
        timeframe: str = '1h',
        since: str = '2020-01-01',
        until: str = None
    ):
        """Download OHLCV data for a trading pair"""
        
        progress_key = f"{pair}_{timeframe}"
        
        # Parse dates
        since_dt = datetime.strptime(since, '%Y-%m-%d')
        since_ts = int(since_dt.timestamp() * 1000)
        
        # Set until timestamp (default to now - 1 hour to avoid hitting "now")
        if until:
            until_dt = datetime.strptime(until, '%Y-%m-%d')
            until_ts = int(until_dt.timestamp() * 1000)
        else:
            # Default: stop 1 hour ago to avoid the "now" problem
            until_ts = int((datetime.now() - timedelta(hours=1)).timestamp() * 1000)
        
        # Resume from progress or start fresh
        current_ts = self.progress.get(progress_key, since_ts)
        
        # Output file
        filename = f"{pair.replace('/', '_')}_{timeframe}.csv"
        filepath = self.output_dir / filename
        
        # Load existing data if resuming
        if filepath.exists():
            existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            all_candles = existing_df.to_dict('records')
        else:
            all_candles = []
        
        logger.info(f"Downloading {pair} {timeframe} from {datetime.fromtimestamp(current_ts/1000)}")
        
        # Calculate total days for progress bar
        total_days = (until_ts - since_ts) / (1000 * 60 * 60 * 24)
        pbar = tqdm(total=int(total_days), desc=f"{pair} {timeframe}", unit="days")
        
        consecutive_empty = 0
        max_consecutive_empty = 10  # Stop after 10 empty responses
        
        while current_ts < until_ts:
            try:
                # Fetch data
                ohlcv = self.exchange.fetch_ohlcv(
                    pair,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not ohlcv:
                    consecutive_empty += 1
                    logger.warning(f"No data for {pair} at {datetime.fromtimestamp(current_ts/1000)}")
                    
                    # If we get too many empty responses, this pair probably doesn't exist yet
                    if consecutive_empty >= max_consecutive_empty:
                        logger.warning(f"Too many empty responses for {pair} - skipping")
                        break
                    
                    # Move forward by 1 day
                    current_ts += 86400000
                    time.sleep(1)
                    continue
                
                # Reset empty counter on successful fetch
                consecutive_empty = 0
                
                # Process candles
                for candle in ohlcv:
                    timestamp = candle[0]
                    
                    # Stop if we've reached the until timestamp
                    if timestamp >= until_ts:
                        logger.info(f"Reached end time for {pair}")
                        current_ts = until_ts
                        break
                    
                    all_candles.append({
                        'timestamp': datetime.fromtimestamp(timestamp / 1000),
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                    
                    current_ts = timestamp + 1
                
                # Update progress
                days_done = (current_ts - since_ts) / (1000 * 60 * 60 * 24)
                pbar.update(int(days_done) - pbar.n)
                
                # Save progress
                self.progress[progress_key] = current_ts
                self._save_progress()
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
            except ccxt.NetworkError as e:
                logger.error(f"Network error: {e}")
                time.sleep(5)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(5)
        
        pbar.close()
        
        # Save to CSV
        if all_candles:
            df = pd.DataFrame(all_candles)
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='last')]  # Remove duplicates
            df.sort_index(inplace=True)
            df.to_csv(filepath)
            
            logger.info(f"✅ Saved {len(df)} candles to {filepath}")
        else:
            logger.warning(f"No data collected for {pair} {timeframe}")


def main():
    parser = argparse.ArgumentParser(description='Download OHLCV data')
    parser.add_argument('--pairs', nargs='+', required=True)
    parser.add_argument('--timeframes', nargs='+', default=['1h'])
    parser.add_argument('--since', default='2020-01-01')
    parser.add_argument('--until', default=None)
    parser.add_argument('--exchange', default='coinbase')
    
    args = parser.parse_args()
    
    downloader = OHLCVDownloader(exchange_name=args.exchange)
    
    for pair in args.pairs:
        for timeframe in args.timeframes:
            downloader.download_pair(pair, timeframe, args.since, args.until)
    
    logger.info("✅ Download complete!")


if __name__ == '__main__':
    main()
