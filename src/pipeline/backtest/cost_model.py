"""Cost model, fill simulation, and position sizing (Section 4.4)."""

from __future__ import annotations

import math
from typing import Any


class CostModel:
    """Fee and slippage model."""

    def __init__(self, commission_per_share: float = 0.0, slippage_bps: int = 5) -> None:
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

    def commission(self, shares: int, price: float) -> float:
        return self.commission_per_share * shares

    def apply_slippage(self, price: float, side: str = "buy") -> float:
        """Apply slippage to execution price."""
        slippage = price * (self.slippage_bps / 10_000)
        if side == "buy":
            return price + slippage
        else:
            return price - slippage


class FillSimulator:
    """Simulate fill rules per Section 4.4."""

    def check_stop_loss(
        self, stop_price: float, open_price: float, low_price: float
    ) -> dict[str, Any]:
        """Check stop-loss trigger on T+1 bar."""
        if open_price < stop_price:
            return {"triggered": True, "fill_price": open_price}
        if low_price <= stop_price:
            return {"triggered": True, "fill_price": stop_price}
        return {"triggered": False, "fill_price": None}

    def check_take_profit(
        self, target_price: float, open_price: float, high_price: float
    ) -> dict[str, Any]:
        """Check take-profit trigger on T+1 bar."""
        if open_price > target_price:
            return {"triggered": True, "fill_price": open_price}
        if high_price >= target_price:
            return {"triggered": True, "fill_price": target_price}
        return {"triggered": False, "fill_price": None}

    def check_trailing_stop(
        self,
        trailing_stop_price: float,
        open_price: float,
        low_price: float,
    ) -> dict[str, Any]:
        """Check trailing stop trigger."""
        if open_price < trailing_stop_price:
            return {"triggered": True, "fill_price": open_price}
        if low_price <= trailing_stop_price:
            return {"triggered": True, "fill_price": trailing_stop_price}
        return {"triggered": False, "fill_price": None}

    def resolve_exit(
        self,
        stop_price: float,
        target_price: float,
        open_price: float,
        high_price: float,
        low_price: float,
        trailing_stop_price: float | None = None,
    ) -> dict[str, Any]:
        """Resolve exit priority: stop_loss > take_profit > trailing_stop."""
        # Check stop loss first
        sl = self.check_stop_loss(stop_price, open_price, low_price)
        if sl["triggered"]:
            return {"exit_reason": "stop_loss", "fill_price": sl["fill_price"]}

        # Check take profit
        tp = self.check_take_profit(target_price, open_price, high_price)
        if tp["triggered"]:
            return {"exit_reason": "take_profit", "fill_price": tp["fill_price"]}

        # Check trailing stop
        if trailing_stop_price is not None:
            ts = self.check_trailing_stop(trailing_stop_price, open_price, low_price)
            if ts["triggered"]:
                return {"exit_reason": "trailing_stop", "fill_price": ts["fill_price"]}

        return {"exit_reason": None, "fill_price": None}


class TrailingStopCalculator:
    """Compute trailing stop as tighter of 21-EMA and 10-day low."""

    def compute(self, ema_21: float, low_10: float) -> float:
        """Return the tighter (higher) of the two trailing stop levels."""
        return max(ema_21, low_10)


class PositionSizer:
    """Fixed-risk position sizing."""

    def __init__(
        self,
        portfolio_value: float,
        risk_per_trade: float = 0.01,
        max_positions: int = 5,
    ) -> None:
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions

    def calculate(self, entry_price: float, stop_price: float) -> int:
        """Calculate position size in shares.

        position_size = (portfolio * risk_per_trade) / (entry - stop)
        Capped by max position value = portfolio / max_positions.
        """
        risk_amount = self.portfolio_value * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share == 0:
            return 0

        shares_by_risk = math.floor(risk_amount / risk_per_share)

        # Cap by max position value
        max_position_value = self.portfolio_value / self.max_positions
        max_shares_by_position = math.floor(max_position_value / entry_price)

        return min(shares_by_risk, max_shares_by_position)
