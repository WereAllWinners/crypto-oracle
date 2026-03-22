"""
Deterministic Trading Rule Layer
Sits between LLM recommendation and order execution.
No trade is approved without passing all gates.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Portfolio:
    """Current portfolio state — caller must supply and keep up to date."""
    total_equity: float              # Total USD value (cash + positions)
    available_cash: float            # USD available to deploy
    open_positions: list             # [{"pair": str, "direction": str, "size_usd": float, "entry_price": float}]
    daily_pnl_usd: float = 0.0       # Today's realised + unrealised PnL
    high_water_mark_equity: float = 0.0  # Peak equity (set to total_equity if unknown)

    def __post_init__(self):
        if self.high_water_mark_equity == 0.0:
            self.high_water_mark_equity = self.total_equity


@dataclass
class TradeDecision:
    """Output of the rule layer."""
    approved: bool
    pair: str
    direction: str                   # BUY / SELL / HOLD
    rejection_reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    applied_rules: list = field(default_factory=list)

    # Populated only when approved=True
    position_size_usd: float = 0.0
    position_size_asset: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    reward_risk_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "pair": self.pair,
            "direction": self.direction,
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "applied_rules": self.applied_rules,
            "position_size_usd": round(self.position_size_usd, 2),
            "position_size_asset": self.position_size_asset,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_usd": round(self.risk_usd, 2),
            "risk_pct": round(self.risk_pct * 100, 3),
            "reward_risk_ratio": round(self.reward_risk_ratio, 2),
        }


# ============================================================================
# RULE ENGINE
# ============================================================================

class RiskManager:
    """
    Deterministic rule layer. Validates LLM recommendations and computes
    position sizes before any order is placed.

    Usage:
        rm = RiskManager()
        decision = rm.evaluate(recommendation, market_data, portfolio)
        if decision.approved:
            # place order using decision.position_size_usd etc.
    """

    def __init__(
        self,
        min_confidence: int = 70,          # % — reject below this
        min_rr: float = 1.5,               # minimum reward/risk ratio
        max_stop_pct: float = 0.10,        # max stop distance as % of entry
        max_open_positions: int = 5,
        daily_loss_limit_pct: float = 0.05,  # -5% equity triggers circuit breaker
        max_drawdown_pct: float = 0.15,      # -15% from HWM triggers circuit breaker
        risk_pct_per_trade: float = 0.01,    # 1% of equity at risk per trade
        max_position_pct: float = 0.20,      # 20% of equity max per position
        min_trade_size_usd: float = 100.0,
        vix_long_block: float = 40.0,        # block BUY above this VIX level
        fear_greed_extreme: tuple = (20, 80),  # (low, high) — halve size outside range
    ):
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self.max_stop_pct = max_stop_pct
        self.max_open_positions = max_open_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_pct_per_trade = risk_pct_per_trade
        self.max_position_pct = max_position_pct
        self.min_trade_size_usd = min_trade_size_usd
        self.vix_long_block = vix_long_block
        self.fear_greed_extreme = fear_greed_extreme

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        recommendation: dict,
        market_data: dict,
        portfolio: Portfolio,
    ) -> TradeDecision:
        """
        Run all rule gates and return a TradeDecision.

        Args:
            recommendation: output of crypto_oracle._parse_recommendation()
                {decision, confidence, entry_price, stop_loss, take_profit}
            market_data: full market data dict (includes macro, onchain)
            portfolio: current Portfolio state
        """
        pair = market_data.get("pair", "UNKNOWN")
        direction = (recommendation.get("decision") or "UNKNOWN").upper()

        decision = TradeDecision(approved=False, pair=pair, direction=direction)

        # Gate 0: HOLD is not a trade
        if direction in ("HOLD", "NO TRADE", "UNKNOWN"):
            decision.rejection_reasons.append("HOLD signal — no trade to execute")
            decision.applied_rules.append("gate0_hold_passthrough")
            return decision

        # Unpack recommendation fields
        confidence = recommendation.get("confidence")
        entry = recommendation.get("entry_price")
        stop = recommendation.get("stop_loss")
        tp = recommendation.get("take_profit")
        current_price = market_data.get("price", 0.0)

        # Use current price as entry fallback if model didn't specify one
        if not entry and current_price:
            entry = current_price
            decision.warnings.append("Entry price not specified by model — using current market price")

        # ---- Gate 1: Signal quality ----
        if not self._gate_signal_quality(decision, direction, confidence, entry, stop, tp):
            return decision

        # ---- Gate 2: Portfolio limits ----
        if not self._gate_portfolio_limits(decision, pair, portfolio):
            return decision

        # ---- Gate 3: Macro overrides ----
        size_multiplier = self._gate_macro_overrides(decision, direction, market_data)
        # A hard block returns None
        if size_multiplier is None:
            return decision

        # ---- Gate 4: Position sizing (fixed fractional) ----
        sized = self._compute_position_size(
            decision, entry, stop, tp, portfolio, size_multiplier
        )
        if not sized:
            return decision

        decision.approved = True
        logger.info(
            f"[RiskManager] APPROVED {direction} {pair} | "
            f"size=${decision.position_size_usd:.0f} | "
            f"risk={decision.risk_pct*100:.2f}% | "
            f"R/R={decision.reward_risk_ratio:.2f}"
        )
        return decision

    # -------------------------------------------------------------------------
    # GATES
    # -------------------------------------------------------------------------

    def _gate_signal_quality(
        self,
        decision: TradeDecision,
        direction: str,
        confidence: Optional[int],
        entry: Optional[float],
        stop: Optional[float],
        tp: Optional[float],
    ) -> bool:
        """Gate 1: Validate the LLM signal before doing anything else."""
        decision.applied_rules.append("gate1_signal_quality")
        ok = True

        # Confidence threshold
        if confidence is None:
            decision.rejection_reasons.append("No confidence value extracted from model output")
            ok = False
        elif confidence < self.min_confidence:
            decision.rejection_reasons.append(
                f"Confidence {confidence}% below minimum {self.min_confidence}%"
            )
            ok = False

        # Entry price required
        if not entry or entry <= 0:
            decision.rejection_reasons.append("No valid entry price from model or market data")
            ok = False

        # Stop loss required and directionally correct
        if stop is None:
            decision.rejection_reasons.append("No stop loss specified — required for all trades")
            ok = False
        elif entry and direction == "BUY" and stop >= entry:
            decision.rejection_reasons.append(
                f"Invalid stop loss {stop} >= entry {entry} for BUY"
            )
            ok = False
        elif entry and direction == "SELL" and stop <= entry:
            decision.rejection_reasons.append(
                f"Invalid stop loss {stop} <= entry {entry} for SELL"
            )
            ok = False

        # Take profit required and directionally correct
        if tp is None:
            decision.rejection_reasons.append("No take profit specified — required for all trades")
            ok = False
        elif entry and direction == "BUY" and tp <= entry:
            decision.rejection_reasons.append(
                f"Invalid take profit {tp} <= entry {entry} for BUY"
            )
            ok = False
        elif entry and direction == "SELL" and tp >= entry:
            decision.rejection_reasons.append(
                f"Invalid take profit {tp} >= entry {entry} for SELL"
            )
            ok = False

        if not ok or entry is None or stop is None or tp is None:
            return False

        # Stop distance cap (catches hallucinated stops)
        stop_dist = abs(entry - stop)
        if stop_dist / entry > self.max_stop_pct:
            decision.rejection_reasons.append(
                f"Stop distance {stop_dist/entry*100:.1f}% exceeds max {self.max_stop_pct*100:.0f}%"
            )
            return False

        # Reward / risk ratio
        tp_dist = abs(tp - entry)
        rr = tp_dist / stop_dist if stop_dist > 0 else 0
        if rr < self.min_rr:
            decision.rejection_reasons.append(
                f"Reward/risk ratio {rr:.2f} below minimum {self.min_rr}"
            )
            return False

        decision.reward_risk_ratio = rr
        decision.entry_price = entry
        decision.stop_loss = stop
        decision.take_profit = tp
        return True

    def _gate_portfolio_limits(
        self,
        decision: TradeDecision,
        pair: str,
        portfolio: Portfolio,
    ) -> bool:
        """Gate 2: Portfolio-level risk limits and circuit breakers."""
        decision.applied_rules.append("gate2_portfolio_limits")

        # Already have a position in this pair
        existing = [p for p in portfolio.open_positions if p.get("pair") == pair]
        if existing:
            decision.rejection_reasons.append(
                f"Already have an open position in {pair} — no doubling down"
            )
            return False

        # Max concurrent positions
        if len(portfolio.open_positions) >= self.max_open_positions:
            decision.rejection_reasons.append(
                f"Max open positions ({self.max_open_positions}) reached"
            )
            return False

        # Available cash check
        if portfolio.available_cash < self.min_trade_size_usd:
            decision.rejection_reasons.append(
                f"Available cash ${portfolio.available_cash:.0f} below minimum trade size ${self.min_trade_size_usd:.0f}"
            )
            return False

        # Daily loss circuit breaker
        daily_loss_pct = portfolio.daily_pnl_usd / portfolio.total_equity
        if daily_loss_pct <= -self.daily_loss_limit_pct:
            decision.rejection_reasons.append(
                f"Daily loss circuit breaker: down {daily_loss_pct*100:.1f}% today (limit: -{self.daily_loss_limit_pct*100:.0f}%)"
            )
            return False

        # Drawdown circuit breaker
        if portfolio.high_water_mark_equity > 0:
            drawdown = (portfolio.total_equity - portfolio.high_water_mark_equity) / portfolio.high_water_mark_equity
            if drawdown <= -self.max_drawdown_pct:
                decision.rejection_reasons.append(
                    f"Max drawdown circuit breaker: {drawdown*100:.1f}% from peak (limit: -{self.max_drawdown_pct*100:.0f}%)"
                )
                return False

        return True

    def _gate_macro_overrides(
        self,
        decision: TradeDecision,
        direction: str,
        market_data: dict,
    ) -> Optional[float]:
        """
        Gate 3: Macro environment overrides.
        Returns a size multiplier (1.0 = normal, 0.5 = half size).
        Returns None if trade should be hard-blocked.
        """
        decision.applied_rules.append("gate3_macro_overrides")
        size_mult = 1.0

        macro = market_data.get("macro", {})
        onchain = market_data.get("onchain", {})

        vix = macro.get("vix_current", 0.0)
        fg = onchain.get("fear_greed_value")

        # Hard block: extreme fear (VIX) for longs
        if direction == "BUY" and vix >= self.vix_long_block:
            decision.rejection_reasons.append(
                f"VIX at {vix:.1f} (>= {self.vix_long_block}) — blocking new long in extreme fear"
            )
            return None

        # Soft: extreme fear/greed → half position
        if fg is not None:
            fg_low, fg_high = self.fear_greed_extreme
            if fg <= fg_low:
                size_mult *= 0.5
                decision.warnings.append(
                    f"Extreme Fear & Greed ({fg}/100) — position size halved"
                )
            elif fg >= fg_high:
                size_mult *= 0.5
                decision.warnings.append(
                    f"Extreme Greed Fear & Greed ({fg}/100) — position size halved"
                )

        return size_mult

    def _compute_position_size(
        self,
        decision: TradeDecision,
        entry: float,
        stop: float,
        tp: float,
        portfolio: Portfolio,
        size_multiplier: float,
    ) -> bool:
        """Gate 4: Fixed fractional position sizing."""
        decision.applied_rules.append("gate4_position_sizing")

        stop_dist = abs(entry - stop)
        risk_usd = portfolio.total_equity * self.risk_pct_per_trade * size_multiplier

        # Position size so that hitting stop = losing exactly risk_usd
        raw_size_usd = (risk_usd / stop_dist) * entry

        # Apply caps
        max_by_pct = portfolio.total_equity * self.max_position_pct
        size_usd = min(raw_size_usd, max_by_pct, portfolio.available_cash)

        if size_usd < self.min_trade_size_usd:
            decision.rejection_reasons.append(
                f"Computed position size ${size_usd:.0f} below minimum ${self.min_trade_size_usd:.0f}"
            )
            return False

        decision.position_size_usd = size_usd
        decision.position_size_asset = size_usd / entry
        decision.risk_usd = risk_usd
        decision.risk_pct = risk_usd / portfolio.total_equity

        tp_dist = abs(tp - entry)
        decision.reward_risk_ratio = tp_dist / stop_dist if stop_dist > 0 else 0

        return True


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    rm = RiskManager()

    portfolio = Portfolio(
        total_equity=100_000,
        available_cash=80_000,
        open_positions=[],
        daily_pnl_usd=0,
        high_water_mark_equity=100_000,
    )

    print("=" * 60)
    print("SMOKE TEST: RiskManager")
    print("=" * 60)

    # --- Test 1: Clean BUY should be approved ---
    rec = {
        "decision": "BUY",
        "confidence": 82,
        "entry_price": 95_000,
        "stop_loss": 92_000,
        "take_profit": 104_000,
    }
    market = {"pair": "BTC/USD", "price": 95_000, "macro": {"vix_current": 18.0}, "onchain": {"fear_greed_value": 55}}
    d = rm.evaluate(rec, market, portfolio)
    print(f"\n[1] Clean BUY => approved={d.approved}")
    assert d.approved, d.rejection_reasons
    print(f"    size=${d.position_size_usd:.0f}  risk={d.risk_pct*100:.2f}%  R/R={d.reward_risk_ratio:.2f}")

    # --- Test 2: Low confidence should be rejected ---
    rec2 = {**rec, "confidence": 55}
    d2 = rm.evaluate(rec2, market, portfolio)
    print(f"\n[2] Low confidence (55%) => approved={d2.approved}  reason={d2.rejection_reasons}")
    assert not d2.approved

    # --- Test 3: Wrong-side stop loss ---
    rec3 = {**rec, "stop_loss": 98_000}  # stop above entry for BUY = wrong
    d3 = rm.evaluate(rec3, market, portfolio)
    print(f"\n[3] Stop above entry => approved={d3.approved}  reason={d3.rejection_reasons}")
    assert not d3.approved

    # --- Test 4: Daily loss circuit breaker ---
    portfolio_down = Portfolio(
        total_equity=100_000,
        available_cash=80_000,
        open_positions=[],
        daily_pnl_usd=-6_000,  # -6% → triggers -5% limit
        high_water_mark_equity=100_000,
    )
    d4 = rm.evaluate(rec, market, portfolio_down)
    print(f"\n[4] Daily loss CB => approved={d4.approved}  reason={d4.rejection_reasons}")
    assert not d4.approved

    # --- Test 5: VIX hard block ---
    market_vix = {**market, "macro": {"vix_current": 45.0}}
    d5 = rm.evaluate(rec, market_vix, portfolio)
    print(f"\n[5] VIX=45 BUY block => approved={d5.approved}  reason={d5.rejection_reasons}")
    assert not d5.approved

    # --- Test 6: Fear & Greed extreme → half size ---
    market_fg = {**market, "onchain": {"fear_greed_value": 12}}
    d6 = rm.evaluate(rec, market_fg, portfolio)
    print(f"\n[6] Extreme fear (FG=12) => approved={d6.approved}  warnings={d6.warnings}")
    print(f"    size=${d6.position_size_usd:.0f} (expect ~half of normal)")
    assert d6.approved and d6.warnings

    # --- Test 7: HOLD passthrough ---
    rec7 = {**rec, "decision": "HOLD"}
    d7 = rm.evaluate(rec7, market, portfolio)
    print(f"\n[7] HOLD passthrough => approved={d7.approved}  reason={d7.rejection_reasons}")
    assert not d7.approved

    # --- Test 8: Drawdown circuit breaker ---
    portfolio_dd = Portfolio(
        total_equity=83_000,
        available_cash=70_000,
        open_positions=[],
        daily_pnl_usd=0,
        high_water_mark_equity=100_000,  # -17% drawdown
    )
    d8 = rm.evaluate(rec, market, portfolio_dd)
    print(f"\n[8] Drawdown CB => approved={d8.approved}  reason={d8.rejection_reasons}")
    assert not d8.approved

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
