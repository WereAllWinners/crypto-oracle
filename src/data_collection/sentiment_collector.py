"""
Multi-source sentiment collector - 100% FREE
No API keys required (all optional)
"""

import requests
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import List, Dict
import time
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentCollector:
    """Collect sentiment from FREE sources"""
    
    def __init__(self, output_dir: str = 'data/sentiment'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_coingecko_sentiment(self, currencies: List[str]) -> List[Dict]:
        """
        CoinGecko - 100% FREE, no API key
        Market sentiment + community voting
        """
        articles = []
        
        coin_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
            'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin',
            'LINK': 'chainlink', 'AVAX': 'avalanche-2', 'DOT': 'polkadot',
            'UNI': 'uniswap', 'LTC': 'litecoin', 'BCH': 'bitcoin-cash',
            'MATIC': 'matic-network', 'ATOM': 'cosmos', 'NEAR': 'near',
            'AAVE': 'aave', 'SHIB': 'shiba-inu', 'PEPE': 'pepe'
        }
        
        for currency in currencies:
            coin_id = coin_map.get(currency)
            if not coin_id:
                continue
            
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Community sentiment voting
                    sentiment_up = data.get('sentiment_votes_up_percentage', 50)
                    sentiment_score = (sentiment_up - 50) / 50  # -1 to +1
                    
                    # Price change sentiment
                    market_data = data.get('market_data', {})
                    price_change_24h = market_data.get('price_change_percentage_24h', 0)
                    
                    article = {
                        'timestamp': datetime.now().isoformat(),
                        'title': f"{currency} Market Sentiment",
                        'currencies': [currency],
                        'sentiment_score': sentiment_score,
                        'sentiment_label': self._sentiment_label(sentiment_score),
                        'price_change_24h': price_change_24h,
                        'community_votes_up': sentiment_up,
                        'kind': 'coingecko',
                        'source': 'CoinGecko Community'
                    }
                    articles.append(article)
                    
                time.sleep(1.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching CoinGecko for {currency}: {e}")
        
        logger.info(f"✅ Fetched {len(articles)} CoinGecko sentiment points")
        return articles
    
    def get_cryptocompare_news(self) -> List[Dict]:
        """
        CryptoCompare - FREE (50 calls/hour)
        Latest crypto news with sentiment
        """
        articles = []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {'lang': 'EN'}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('Data', [])[:30]:  # Latest 30 articles
                    text = item.get('title', '') + ' ' + item.get('body', '')
                    sentiment_score = self._analyze_sentiment(text[:500])
                    
                    # Extract mentioned coins from categories
                    categories = item.get('categories', '').split('|')
                    
                    article = {
                        'timestamp': datetime.fromtimestamp(item.get('published_on')).isoformat(),
                        'title': item.get('title'),
                        'url': item.get('url'),
                        'source': item.get('source'),
                        'categories': categories,
                        'sentiment_score': sentiment_score,
                        'sentiment_label': self._sentiment_label(sentiment_score),
                        'kind': 'news'
                    }
                    articles.append(article)
                
                logger.info(f"✅ Fetched {len(articles)} CryptoCompare news articles")
        
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare: {e}")
        
        return articles
    
    def get_coinmarketcap_sentiment(self, currencies: List[str]) -> List[Dict]:
        """
        CoinMarketCap trending - FREE
        Social mentions and trends
        """
        articles = []
        
        try:
            # CMC trending coins (no API key needed for trending)
            url = "https://api.coinmarketcap.com/data-api/v3/topsearch/rank"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', {}).get('cryptoTopSearchRanks', [])[:20]:
                    symbol = item.get('symbol', '').upper()
                    
                    if symbol in currencies:
                        # Rank as sentiment (lower rank = more interest = bullish)
                        rank = item.get('rank', 100)
                        sentiment_score = max(-1, min(1, (50 - rank) / 50))
                        
                        article = {
                            'timestamp': datetime.now().isoformat(),
                            'title': f"{symbol} Trending #{rank}",
                            'currencies': [symbol],
                            'sentiment_score': sentiment_score,
                            'sentiment_label': self._sentiment_label(sentiment_score),
                            'trending_rank': rank,
                            'kind': 'trending',
                            'source': 'CoinMarketCap'
                        }
                        articles.append(article)
                
                logger.info(f"✅ Fetched {len(articles)} CMC trending signals")
        
        except Exception as e:
            logger.error(f"Error fetching CoinMarketCap: {e}")
        
        return articles
    
    def get_alternative_me_sentiment(self) -> List[Dict]:
        """
        Alternative.me - FREE
        Crypto Fear & Greed Index + news
        """
        articles = []
        
        try:
            # Fear & Greed Index
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', [])[:7]:  # Last 7 days
                    value = int(item['value'])
                    
                    # Convert to sentiment score
                    sentiment_score = (value - 50) / 50  # -1 to +1
                    
                    article = {
                        'timestamp': datetime.fromtimestamp(int(item['timestamp'])).isoformat(),
                        'title': f"Market {item['value_classification']}",
                        'currencies': ['BTC'],  # Fear & Greed is BTC-correlated
                        'sentiment_score': sentiment_score,
                        'sentiment_label': self._sentiment_label(sentiment_score),
                        'fear_greed_value': value,
                        'classification': item['value_classification'],
                        'kind': 'fear_greed',
                        'source': 'Alternative.me'
                    }
                    articles.append(article)
                
                logger.info(f"✅ Fetched {len(articles)} Fear & Greed datapoints")
        
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
        
        return articles
    
    def get_lunarcrush_sentiment(self, currencies: List[str]) -> List[Dict]:
        """
        LunarCrush - FREE tier
        Social media aggregator (Twitter, Reddit, etc.)
        """
        articles = []
        
        # LunarCrush has a free public feed
        try:
            url = "https://lunarcrush.com/api3/coins"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for coin in data.get('data', [])[:50]:
                    symbol = coin.get('s', '').upper()
                    
                    if symbol in currencies:
                        # Galaxy score (0-100) to sentiment
                        galaxy_score = coin.get('gs', 50)
                        sentiment_score = (galaxy_score - 50) / 50
                        
                        article = {
                            'timestamp': datetime.now().isoformat(),
                            'title': f"{symbol} Social Sentiment",
                            'currencies': [symbol],
                            'sentiment_score': sentiment_score,
                            'sentiment_label': self._sentiment_label(sentiment_score),
                            'galaxy_score': galaxy_score,
                            'social_volume': coin.get('sv', 0),
                            'kind': 'social',
                            'source': 'LunarCrush'
                        }
                        articles.append(article)
                
                logger.info(f"✅ Fetched {len(articles)} LunarCrush social signals")
        
        except Exception as e:
            logger.error(f"Error fetching LunarCrush: {e}")
        
        return articles
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.3:
            return 'bullish'
        elif score < -0.3:
            return 'bearish'
        else:
            return 'neutral'
    
    def aggregate_sentiment(self, currency: str, articles: List[Dict]) -> Dict:
        """Aggregate sentiment for a currency"""
        relevant = []
        
        for a in articles:
            if currency in a.get('currencies', []):
                relevant.append(a)
            elif currency.lower() in a.get('title', '').lower():
                relevant.append(a)
            elif currency in a.get('categories', []):
                relevant.append(a)
        
        if not relevant:
            return {
                'currency': currency,
                'article_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_label': 'neutral'
            }
        
        avg_sentiment = sum(a['sentiment_score'] for a in relevant) / len(relevant)
        
        return {
            'currency': currency,
            'article_count': len(relevant),
            'avg_sentiment': avg_sentiment,
            'sentiment_label': self._sentiment_label(avg_sentiment),
            'bullish_count': sum(1 for a in relevant if a['sentiment_score'] > 0.3),
            'bearish_count': sum(1 for a in relevant if a['sentiment_score'] < -0.3),
            'neutral_count': sum(1 for a in relevant if -0.3 <= a['sentiment_score'] <= 0.3)
        }
    
    def collect_all(self, currencies: List[str] = ['BTC', 'ETH', 'SOL']) -> Dict:
        """Collect from ALL free sources (no API keys needed!)"""
        logger.info("📰 Collecting sentiment from free sources...")
        
        all_data = []
        
        # 1. CoinGecko (community sentiment)
        coingecko = self.get_coingecko_sentiment(currencies)
        all_data.extend(coingecko)
        
        # 2. CryptoCompare (news)
        cryptocompare = self.get_cryptocompare_news()
        all_data.extend(cryptocompare)
        
        # 3. CoinMarketCap (trending)
        cmc = self.get_coinmarketcap_sentiment(currencies)
        all_data.extend(cmc)
        
        # 4. Fear & Greed Index
        fear_greed = self.get_alternative_me_sentiment()
        all_data.extend(fear_greed)
        
        # 5. LunarCrush (social aggregator)
        try:
            lunarcrush = self.get_lunarcrush_sentiment(currencies)
            all_data.extend(lunarcrush)
        except:
            logger.warning("LunarCrush not available")
        
        # Aggregate by currency
        aggregated = {}
        for currency in currencies:
            aggregated[currency] = self.aggregate_sentiment(currency, all_data)
        
        # Save raw data
        output_file = self.output_dir / f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w') as f:
            for item in all_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"💾 Saved {len(all_data)} sentiment items to {output_file}")
        
        return {
            'raw_data': all_data,
            'aggregated': aggregated,
            'summary': {
                'total_items': len(all_data),
                'coingecko': len([d for d in all_data if d['kind'] == 'coingecko']),
                'news': len([d for d in all_data if d['kind'] == 'news']),
                'trending': len([d for d in all_data if d['kind'] == 'trending']),
                'fear_greed': len([d for d in all_data if d['kind'] == 'fear_greed']),
                'social': len([d for d in all_data if d['kind'] == 'social'])
            }
        }


def main():
    """CLI entry point"""
    collector = SentimentCollector()
    results = collector.collect_all(['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE'])
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    for currency, data in results['aggregated'].items():
        print(f"\n{currency}:")
        print(f"  Data points: {data['article_count']}")
        print(f"  Sentiment: {data['sentiment_label']} ({data['avg_sentiment']:.2f})")
        print(f"  Bullish: {data['bullish_count']}, Bearish: {data['bearish_count']}, Neutral: {data['neutral_count']}")
    
    print(f"\n{results['summary']}")


if __name__ == '__main__':
    main()