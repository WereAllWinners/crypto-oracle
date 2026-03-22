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
from trading.risk_manager import RiskManager, Portfolio
from trading.live_trader import is_paused, pause_trading, resume_trading, KILL_SWITCH_FILE
from trading.notifier import Notifier
from trading.trade_logger import (
    log_approved_trade, log_rejection, get_open_trades,
    get_closed_trades, get_performance_summary, get_daily_pnl,
)

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
risk_manager = None


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


class TradeRequest(BaseModel):
    """
    Request to run the full pipeline: market data → fine-tuned LLM → rule layer.
    The caller must supply their current portfolio state so the rule layer can
    apply position sizing and circuit breakers correctly.
    """
    pair: str = Field(..., example="BTC/USD")
    include_sentiment: bool = Field(default=True)
    include_macro: bool = Field(default=True)
    include_onchain: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)

    # Portfolio state — required for position sizing and circuit breakers
    total_equity: float = Field(..., example=100000.0, description="Total portfolio value USD")
    available_cash: float = Field(..., example=80000.0, description="Deployable USD")
    open_positions: List[Dict] = Field(
        default=[],
        example=[{"pair": "ETH/USD", "direction": "BUY", "size_usd": 5000, "entry_price": 3200}],
    )
    daily_pnl_usd: float = Field(default=0.0, example=-500.0)
    high_water_mark_equity: float = Field(
        default=0.0,
        description="Peak equity for drawdown calc. Defaults to total_equity if 0.",
    )


class CloseTradeRequest(BaseModel):
    trade_id: str
    exit_price: float
    outcome: str = Field(..., example="took_profit",
                         description="win | loss | breakeven | stopped_out | took_profit")


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global oracle, market_analyzer, batch_analyzer, risk_manager

    logger.info("Starting Crypto Oracle API...")

    # Fine-tuned LoRA adapter — this IS the trained model, not the base model
    model_path = "models/crypto-oracle-qwen-32b/final_model"

    try:
        oracle = CryptoOracle(model_path=model_path)
        market_analyzer = MarketAnalyzer()
        batch_analyzer = BatchAnalyzer(model_path=model_path)
        risk_manager = RiskManager()

        logger.info("Models loaded (fine-tuned Qwen2.5-32B LoRA adapter)")
        logger.info("Risk manager initialised")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
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
    perf = get_performance_summary()
    return {
        "models_loaded": oracle is not None,
        "risk_manager_active": risk_manager is not None,
        "open_positions": len(get_open_trades()),
        "daily_pnl_usd": get_daily_pnl(),
        "lifetime_performance": perf,
    }


# ============================================================================
# TRADE ENDPOINTS (LLM + Rule Layer)
# ============================================================================

@app.post("/trade")
async def trade(request: TradeRequest):
    """
    Full pipeline: collect live market data → fine-tuned LLM recommendation
    → deterministic rule layer → approved/rejected trade decision.

    This is the primary endpoint for the trading agent.
    Raw /predict is still available for research/monitoring.
    """
    try:
        logger.info(f"Trade request: {request.pair}")

        # 1. Collect live market data
        market_data = market_analyzer.get_current_market_data(request.pair)
        if not request.include_sentiment:
            market_data.pop("sentiment", None)
        if not request.include_macro:
            market_data.pop("macro", None)
        if not request.include_onchain:
            market_data.pop("onchain", None)

        # 2. Fine-tuned LLM generates recommendation
        prediction = oracle.predict(market_data, temperature=request.temperature)
        recommendation = prediction["recommendation"]

        # 3. Deterministic rule layer evaluates the signal
        portfolio = Portfolio(
            total_equity=request.total_equity,
            available_cash=request.available_cash,
            open_positions=request.open_positions,
            daily_pnl_usd=request.daily_pnl_usd,
            high_water_mark_equity=request.high_water_mark_equity or request.total_equity,
        )
        decision = risk_manager.evaluate(recommendation, market_data, portfolio)
        decision_dict = decision.to_dict()

        # 4. Log everything
        if decision.approved:
            import uuid
            trade_id = str(uuid.uuid4())
            log_approved_trade(
                trade_id=trade_id,
                decision=decision_dict,
                recommendation=recommendation,
                market_data=market_data,
                model_response=prediction.get("full_response", ""),
            )
            decision_dict["trade_id"] = trade_id
        else:
            log_rejection(
                pair=request.pair,
                direction=recommendation.get("decision", "UNKNOWN"),
                confidence=recommendation.get("confidence"),
                reasons=decision.rejection_reasons,
                market_data=market_data,
            )

        return {
            "trade_decision": decision_dict,
            "llm_recommendation": recommendation,
            "llm_full_response": prediction.get("full_response", ""),
            "market_data": market_data,
        }

    except Exception as e:
        logger.error(f"Error in /trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trade/close")
async def close_trade_endpoint(request: CloseTradeRequest):
    """
    Record the outcome of a closed trade.
    Call this when your position is actually closed (SL hit, TP hit, manual close).
    Outcome is stored and feeds the continual learning pipeline.
    """
    from trading.trade_logger import close_trade
    try:
        close_trade(
            trade_id=request.trade_id,
            exit_price=request.exit_price,
            outcome=request.outcome,
        )
        return {"status": "ok", "trade_id": request.trade_id, "outcome": request.outcome}
    except Exception as e:
        logger.error(f"Error closing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trade/open")
async def list_open_trades():
    """List all currently open (not yet closed) trades."""
    return {"open_trades": get_open_trades()}


@app.get("/trade/history")
async def trade_history(limit: int = 100):
    """Recent closed trade history with outcomes."""
    trades = get_closed_trades(limit=limit)
    summary = get_performance_summary()
    return {"summary": summary, "trades": trades}


@app.post("/learn")
async def run_learning_cycle(dry_run: bool = False):
    """
    Manually trigger a continual learning cycle:
    analyse closed trades → generate training examples → optionally retrain.
    Set dry_run=true to see what would happen without actually retraining.
    """
    try:
        from trading.continual_learner import run_learning_cycle as _run
        result = _run(dry_run=dry_run)
        return result
    except Exception as e:
        logger.error(f"Error in learning cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PAPER TRADING ENDPOINTS
# ============================================================================

@app.post("/paper/trade")
async def paper_trade(request: TradeRequest):
    """
    Same as /trade but executes against a paper (simulated) portfolio.
    Safe to call at any time — no real money involved.
    Use this for at least 2-4 weeks of validation before enabling /trade.
    """
    try:
        from testing.paper_trader import load_state, save_state, portfolio_equity, fetch_prices, run_paper_cycle

        state = load_state(initial_equity=request.total_equity)

        # Single-pair cycle using the loaded oracle + market_analyzer
        cycle_log = run_paper_cycle(
            pairs=[request.pair],
            state=state,
            risk_manager=risk_manager,
            oracle=oracle,
            market_analyzer=market_analyzer,
        )

        prices = fetch_prices([p["pair"] for p in state["open_positions"]] or [request.pair])
        equity = portfolio_equity(state, prices)

        return {
            "cycle": cycle_log,
            "paper_portfolio": {
                "equity":         round(equity, 2),
                "cash":           round(state["cash"], 2),
                "open_positions": state["open_positions"],
                "total_trades":   state["total_trades"],
            },
        }
    except Exception as e:
        logger.error(f"Error in /paper/trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/status")
async def paper_status():
    """Current paper portfolio equity, positions, and performance."""
    try:
        from testing.paper_trader import load_state, portfolio_equity, fetch_prices
        from trading.trade_logger import get_performance_summary, get_daily_pnl

        state  = load_state()
        prices = fetch_prices([p["pair"] for p in state["open_positions"]] or ["BTC/USD"])
        equity = portfolio_equity(state, prices)

        return {
            "equity":          round(equity, 2),
            "cash":            round(state["cash"], 2),
            "high_water_mark": round(state["high_water_mark"], 2),
            "total_return_pct": round((equity - state["initial_equity"]) / state["initial_equity"] * 100, 2),
            "daily_pnl_usd":   get_daily_pnl(),
            "open_positions":  state["open_positions"],
            "performance":     get_performance_summary(),
        }
    except Exception as e:
        logger.error(f"Error in /paper/status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ADMIN — KILL-SWITCH & HEALTH
# ============================================================================

@app.get("/admin/status")
async def admin_status():
    """Return current kill-switch state and system health."""
    paused = is_paused()
    reason = ""
    if paused and KILL_SWITCH_FILE.exists():
        reason = KILL_SWITCH_FILE.read_text().strip()
    return {
        "trading_paused": paused,
        "pause_reason":   reason,
        "model_loaded":   oracle is not None,
        "open_positions": len(get_open_trades()),
        "daily_pnl_usd":  get_daily_pnl(),
    }


@app.post("/admin/pause")
async def admin_pause(reason: str = "paused via API"):
    """
    Activate the kill-switch. New entry orders will be refused until resumed.
    Existing open positions continue to be monitored for SL/TP.
    """
    pause_trading(reason)
    notifier = Notifier()
    notifier.paused(reason)
    logger.warning(f"[API] Kill-switch activated: {reason}")
    return {"status": "paused", "reason": reason}


@app.post("/admin/resume")
async def admin_resume():
    """Deactivate the kill-switch and allow new entry orders."""
    resume_trading()
    notifier = Notifier()
    notifier.resumed()
    logger.info("[API] Kill-switch cleared -- trading resumed")
    return {"status": "active"}


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