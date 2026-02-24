"""
Crypto Oracle Inference Engine
Loads fine-tuned model and generates trading recommendations
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from unsloth import FastLanguageModel
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoOracle:
    """Fine-tuned LLM for crypto trading recommendations"""
    
    def __init__(
        self,
        model_path: str = "models/crypto-oracle-qwen-32b/final_model",
        use_unsloth: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize the Crypto Oracle
        
        Args:
            model_path: Path to fine-tuned model
            use_unsloth: Use Unsloth for faster inference
            device: Device to run on (cuda/cpu)
        """
        self.model_path = Path(model_path)
        self.use_unsloth = use_unsloth
        self.device = device
        
        logger.info(f"🤖 Loading Crypto Oracle from {model_path}...")
        
        if use_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True
            )
        
        logger.info("✅ Model loaded successfully")
    
    def _format_prompt(self, market_data: Dict) -> str:
        """
        Format market data into instruction prompt
        
        Args:
            market_data: Dict containing technical, sentiment, macro, on-chain data
        
        Returns:
            Formatted instruction string
        """
        pair = market_data.get('pair', 'UNKNOWN')
        price = market_data.get('price', 0)
        change_1h = market_data.get('change_1h', 0)
        change_24h = market_data.get('change_24h', 0)
        
        # Technical indicators
        tech = market_data.get('technical', {})
        
        instruction = f"""Analyze the current market conditions for {pair} and provide a comprehensive trading recommendation.

Current price: ${price:,.2f}
1h change: {change_1h:+.2f}%
24h change: {change_24h:+.2f}%

📊 Technical Analysis:
- Trend: {tech.get('trend', 'unknown')}
- RSI: {tech.get('rsi_state', 'unknown')}
- MACD: {tech.get('macd_signal', 'unknown')}
- Bollinger Bands: {tech.get('bb_state', 'unknown')}
- Volume: {tech.get('volume_state', 'unknown')}
- SMA 20: ${tech.get('sma_20', 0):,.2f}
- SMA 50: ${tech.get('sma_50', 0):,.2f}
- SMA 200: ${tech.get('sma_200', 0):,.2f}"""
        
        # Add sentiment if available
        if 'sentiment' in market_data:
            sent = market_data['sentiment']
            instruction += f"""

📰 Market Sentiment:
- Overall sentiment: {sent.get('avg_sentiment', 0):.2f} (-1 to +1 scale)
- Bullish mentions: {sent.get('bullish_pct', 0):.1f}%
- Bearish mentions: {sent.get('bearish_pct', 0):.1f}%
- Total data points: {sent.get('total_items', 0)}"""
        
        # Add macro if available
        if 'macro' in market_data:
            macro = market_data['macro']
            instruction += f"""

🌍 Macro Environment:
- DXY (Dollar Index): ${macro.get('dxy_current', 0):.2f} ({macro.get('dxy_change', 0):+.2f}%) - {macro.get('dxy_signal', 'unknown')}
- SPY (S&P 500): ${macro.get('spy_current', 0):.2f} ({macro.get('spy_change', 0):+.2f}%) - {macro.get('spy_signal', 'unknown')}
- VIX (Fear Index): {macro.get('vix_current', 0):.2f} - {macro.get('vix_signal', 'unknown')}
- BTC Dominance: {macro.get('btc_dominance', 0):.2f}% - {macro.get('btc_dom_phase', 'unknown')}"""
        
        # Add on-chain if available
        if 'onchain' in market_data:
            onchain = market_data['onchain']
            instruction += f"""

⛓️  On-Chain Metrics:
- Fear & Greed Index: {onchain.get('fear_greed_value', 0)}/100 ({onchain.get('fear_greed_classification', 'Unknown')})
- Signal: {onchain.get('fear_greed_signal', 'unknown')} (contrarian indicator)"""
        
        # Add strategy performance if available
        if 'strategy' in market_data:
            strat = market_data['strategy']
            instruction += f"""

📈 Historical Strategy Performance on {pair}:
- Best strategy: {strat.get('best_strategy_name', 'unknown')}
- Win rate: {strat.get('win_rate', 0):.1f}%
- Avg P&L: {strat.get('avg_pnl', 0):.2f}%
- Total trades: {strat.get('total_trades', 0)}"""
        
        instruction += """

What is your trading recommendation? Provide:
1. Decision (BUY/SELL/HOLD)
2. Confidence level
3. Entry/exit prices
4. Risk management plan
5. Detailed reasoning based on technical, sentiment, and macro factors"""
        
        return instruction
    
    def predict(
        self,
        market_data: Dict,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict:
        """
        Generate trading recommendation
        
        Args:
            market_data: Market data dictionary
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0-1.0)
            stream: Stream output token-by-token
        
        Returns:
            Dict with recommendation and metadata
        """
        instruction = self._format_prompt(market_data)
        
        # Format with chat template
        chat_prompt = f"""<|im_start|>system
You are a professional cryptocurrency trading advisor with expertise in technical analysis, market sentiment, on-chain metrics, and risk management. Provide clear, actionable trading recommendations based on comprehensive market data.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                use_cache=True
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                use_cache=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = response.split("<|im_start|>assistant")[-1].strip()
        
        # Parse recommendation
        recommendation = self._parse_recommendation(response)
        
        return {
            'pair': market_data.get('pair', 'UNKNOWN'),
            'timestamp': datetime.now().isoformat(),
            'recommendation': recommendation,
            'full_response': response,
            'market_data': market_data
        }
    
    def _parse_recommendation(self, response: str) -> Dict:
        """Extract structured data from response"""
        import re
        
        # Extract decision (BUY/SELL/HOLD)
        decision = 'UNKNOWN'
        if 'BUY' in response.upper() and 'NO' not in response[:200].upper():
            decision = 'BUY'
        elif 'SELL' in response.upper():
            decision = 'SELL'
        elif 'HOLD' in response.upper() or 'NO TRADE' in response.upper():
            decision = 'HOLD'
        
        # Extract confidence
        confidence = None
        conf_match = re.search(r'[Cc]onfidence:?\s*(\d+)%', response)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        # Extract entry price
        entry_price = None
        entry_match = re.search(r'[Ee]ntry:?\s*\$?([\d,]+\.?\d*)', response)
        if entry_match:
            entry_price = float(entry_match.group(1).replace(',', ''))
        
        # Extract stop loss
        stop_loss = None
        stop_match = re.search(r'[Ss]top[-\s]?[Ll]oss:?\s*\$?([\d,]+\.?\d*)', response)
        if stop_match:
            stop_loss = float(stop_match.group(1).replace(',', ''))
        
        # Extract take profit
        take_profit = None
        tp_match = re.search(r'[Tt]ake[-\s]?[Pp]rofit:?\s*\$?([\d,]+\.?\d*)', response)
        if tp_match:
            take_profit = float(tp_match.group(1).replace(',', ''))
        
        return {
            'decision': decision,
            'confidence': confidence,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def batch_predict(
        self,
        market_data_list: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """
        Generate predictions for multiple assets
        
        Args:
            market_data_list: List of market data dicts
            **kwargs: Arguments passed to predict()
        
        Returns:
            List of prediction dicts
        """
        predictions = []
        
        for market_data in market_data_list:
            try:
                pred = self.predict(market_data, **kwargs)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting {market_data.get('pair')}: {e}")
                predictions.append({
                    'pair': market_data.get('pair', 'UNKNOWN'),
                    'error': str(e)
                })
        
        return predictions


if __name__ == '__main__':
    # Example usage
    oracle = CryptoOracle()
    
    # Example market data
    market_data = {
        'pair': 'BTC/USD',
        'price': 96500,
        'change_1h': 0.5,
        'change_24h': -1.2,
        'technical': {
            'trend': 'mild uptrend',
            'rsi_state': 'neutral (RSI 52.3)',
            'macd_signal': 'bullish',
            'bb_state': 'mid-range',
            'volume_state': 'normal',
            'sma_20': 96200,
            'sma_50': 95800,
            'sma_200': 89500
        }
    }
    
    prediction = oracle.predict(market_data, stream=True)
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print(json.dumps(prediction['recommendation'], indent=2))