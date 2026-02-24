"""
Crypto Oracle REST API
FastAPI server for inference and analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.crypto_oracle import CryptoOracle
from inference.market_analyzer import MarketAnalyzer
from inference.batch_analyzer import BatchAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Crypto Oracle API",
    description="AI-powered cryptocurrency trading recommendations",
    version="1.0.0"
)

# Global instances (loaded on startup)
oracle = None
market_analyzer = None
batch_analyzer = None


# ============================================================================
# MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    pair: str = Field(..., example="BTC/USD")
    include_sentiment: bool = Field(default=True)
    include_macro: bool = Field(default=True)
    include_onchain: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    pairs: List[str] = Field(..., example=["BTC/USD", "ETH/USD", "SOL/USD"])
    filter_decision: Optional[str] = Field(None, example="BUY")
    min_confidence: Optional[int] = Field(None, ge=0, le=100, example=70)


class CustomAnalysisRequest(BaseModel):
    pair: str
    price: float
    change_1h: float
    change_24h: float
    technical: Dict
    sentiment: Optional[Dict] = None
    macro: Optional[Dict] = None
    onchain: Optional[Dict] = None


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global oracle, market_analyzer, batch_analyzer
    
    logger.info("🚀 Starting Crypto Oracle API...")
    
    model_path = "models/crypto-oracle-qwen-32b/final_model"
    
    try:
        oracle = CryptoOracle(model_path=model_path)
        market_analyzer = MarketAnalyzer()
        batch_analyzer = BatchAnalyzer(model_path=model_path)
        
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Oracle API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #1a1a1a; color: #fff; }
            h1 { color: #00ff88; }
            .endpoint { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #00ff88; }
            .method { color: #00ff88; font-weight: bold; }
            code { background: #000; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🔮 Crypto Oracle API</h1>
        <p>AI-powered cryptocurrency trading recommendations</p>
        
        <h2>Endpoints</h2>
        
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/health</code></p>
            <p>Check API health status</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <code>/predict</code></p>
            <p>Get trading recommendation for a single pair</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <code>/batch</code></p>
            <p>Analyze multiple pairs simultaneously</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <code>/analyze/custom</code></p>
            <p>Analyze with custom market data</p>
        </div>
        
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/pairs</code></p>
            <p>List supported trading pairs</p>
        </div>
        
        <p><a href="/docs" style="color: #00ff88;">Interactive API Docs →</a></p>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": oracle is not None
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Generate trading recommendation for a single pair
    
    Example:
```
    {
        "pair": "BTC/USD",
        "temperature": 0.7
    }
```
    """
    try:
        logger.info(f"📊 Prediction request for {request.pair}")
        
        # Get market data
        market_data = market_analyzer.get_current_market_data(request.pair)
        
        # Filter data based on request
        if not request.include_sentiment:
            market_data.pop('sentiment', None)
        if not request.include_macro:
            market_data.pop('macro', None)
        if not request.include_onchain:
            market_data.pop('onchain', None)
        
        # Generate prediction
        prediction = oracle.predict(
            market_data,
            temperature=request.temperature
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_analyze(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Analyze multiple pairs simultaneously
    
    Example:
```
    {
        "pairs": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "filter_decision": "BUY",
        "min_confidence": 70
    }
```
    """
    try:
        logger.info(f"📊 Batch analysis request for {len(request.pairs)} pairs")
        
        # Analyze pairs
        predictions = batch_analyzer.analyze_pairs(
            request.pairs,
            filter_decision=request.filter_decision,
            min_confidence=request.min_confidence
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_analyzed": len(request.pairs),
            "results_returned": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error in batch analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/custom")
async def analyze_custom(request: CustomAnalysisRequest):
    """
    Analyze with custom market data
    
    Useful for testing or if you have your own data sources
    """
    try:
        market_data = request.dict()
        
        prediction = oracle.predict(market_data)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in custom analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pairs")
async def list_pairs():
    """List supported trading pairs"""
    pairs = [
        "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD",
        "ADA/USD", "LINK/USD", "AVAX/USD", "DOT/USD", "LTC/USD",
        "UNI/USD", "AAVE/USD"
    ]
    
    return {
        "supported_pairs": pairs,
        "count": len(pairs)
    }


@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    # In production, track these in a database
    return {
        "uptime": "API just started",
        "total_predictions": 0,
        "models_loaded": oracle is not None
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )