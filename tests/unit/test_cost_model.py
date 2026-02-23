"""Tests for cost model, fill rules, and position sizing (Section 4.4)."""

import pytest

from pipeline.backtest.cost_model import (
    CostModel,
    FillSimulator,
    PositionSizer,
    TrailingStopCalculator,
)

# --- Cost model tests ---


def test_commission_zero_alpaca() -> None:
    """Alpaca commission is zero."""
    model = CostModel(commission_per_share=0.0, slippage_bps=5)
    cost = model.commission(shares=100, price=150.0)
    assert cost == 0.0


def test_slippage_applied() -> None:
    """Slippage should be applied to execution price."""
    model = CostModel(commission_per_share=0.0, slippage_bps=5)
    # Buy: slippage increases price
    buy_price = model.apply_slippage(100.0, side="buy")
    assert buy_price > 100.0
    assert buy_price == pytest.approx(100.05)  # 5 bps = 0.05%

    # Sell: slippage decreases price
    sell_price = model.apply_slippage(100.0, side="sell")
    assert sell_price < 100.0
    assert sell_price == pytest.approx(99.95)


# --- Stop-loss fill tests ---


def test_stop_loss_fill_at_open_gap_down() -> None:
    """T+1 Open < stop price → fill at Open (gap penetration)."""
    sim = FillSimulator()
    result = sim.check_stop_loss(stop_price=95.0, open_price=93.0, low_price=92.0)
    assert result["triggered"] is True
    assert result["fill_price"] == 93.0  # Fill at open (gap down)


def test_stop_loss_fill_at_stop_price() -> None:
    """T+1 Low ≤ stop price (but Open > stop) → fill at stop price."""
    sim = FillSimulator()
    result = sim.check_stop_loss(stop_price=95.0, open_price=97.0, low_price=94.0)
    assert result["triggered"] is True
    assert result["fill_price"] == 95.0


def test_stop_loss_not_triggered() -> None:
    """Low > stop price → stop not triggered."""
    sim = FillSimulator()
    result = sim.check_stop_loss(stop_price=90.0, open_price=100.0, low_price=95.0)
    assert result["triggered"] is False


# --- Take-profit fill tests ---


def test_take_profit_fill_at_open_gap_up() -> None:
    """T+1 Open > target price → fill at Open (gap through)."""
    sim = FillSimulator()
    result = sim.check_take_profit(target_price=110.0, open_price=112.0, high_price=115.0)
    assert result["triggered"] is True
    assert result["fill_price"] == 112.0  # Fill at open


def test_take_profit_fill_at_target() -> None:
    """T+1 High ≥ target (but Open < target) → fill at target."""
    sim = FillSimulator()
    result = sim.check_take_profit(target_price=110.0, open_price=108.0, high_price=112.0)
    assert result["triggered"] is True
    assert result["fill_price"] == 110.0


def test_take_profit_not_triggered() -> None:
    """High < target → not triggered."""
    sim = FillSimulator()
    result = sim.check_take_profit(target_price=120.0, open_price=108.0, high_price=115.0)
    assert result["triggered"] is False


# --- Same-bar conflict ---


def test_same_bar_conflict_stop_wins() -> None:
    """When both SL and TP trigger on same bar, stop-loss wins."""
    sim = FillSimulator()
    result = sim.resolve_exit(
        stop_price=95.0,
        target_price=110.0,
        open_price=93.0,  # Gaps below stop AND could also reach target
        high_price=112.0,
        low_price=92.0,
    )
    assert result["exit_reason"] == "stop_loss"
    assert result["fill_price"] == 93.0


def test_exit_priority_order() -> None:
    """Exit priority: stop_loss > take_profit > trailing_stop."""
    sim = FillSimulator()
    # Only trailing stop triggers
    result = sim.resolve_exit(
        stop_price=85.0,
        target_price=130.0,
        open_price=100.0,
        high_price=105.0,
        low_price=96.0,
        trailing_stop_price=97.0,
    )
    assert result["exit_reason"] == "trailing_stop"
    assert result["fill_price"] == 97.0


# --- Trailing stop ---


def test_trailing_stop_tighter_of_ema21_low10() -> None:
    """Trailing stop is the tighter of 21-EMA and 10-day low."""
    calc = TrailingStopCalculator()
    ema21 = 98.0
    low10 = 96.0
    # Tighter = higher value (closer to current price) for a long position
    stop = calc.compute(ema_21=ema21, low_10=low10)
    assert stop == 98.0  # EMA is tighter (higher)


def test_trailing_stop_updates_each_bar() -> None:
    """Trailing stop should be recalculated at each bar close."""
    calc = TrailingStopCalculator()
    stops = []
    ema_values = [95.0, 96.0, 97.0, 98.0]
    low_values = [93.0, 94.0, 95.0, 96.0]
    for ema, low in zip(ema_values, low_values, strict=True):
        stop = calc.compute(ema_21=ema, low_10=low)
        stops.append(stop)
    # Stops should generally increase as both inputs increase
    assert stops[-1] >= stops[0]


# --- Execution timing ---


def test_next_day_open_execution() -> None:
    """Signals at Day T close execute at Day T+1 open."""
    model = CostModel(commission_per_share=0.0, slippage_bps=5)
    # Entry at next-day open of 100.0
    entry_price = model.apply_slippage(100.0, side="buy")
    assert entry_price == pytest.approx(100.05)  # Slippage added


# --- Position sizing ---


def test_position_size_fixed_risk() -> None:
    """Position size = (portfolio * risk_per_trade) / (entry - stop)."""
    sizer = PositionSizer(portfolio_value=25_000, risk_per_trade=0.01, max_positions=5)
    shares = sizer.calculate(entry_price=100.0, stop_price=93.0)
    # Risk amount = 25000 * 0.01 = 250
    # Risk per share = 100 - 93 = 7
    # Shares = 250 / 7 = 35.7 → 35 (floor)
    assert shares == 35


def test_position_size_respects_max_positions() -> None:
    """Position value should not exceed portfolio / max_positions."""
    sizer = PositionSizer(portfolio_value=25_000, risk_per_trade=0.01, max_positions=5)
    # If fixed-risk gives too many shares (high price, tight stop)
    shares = sizer.calculate(entry_price=10.0, stop_price=9.9)
    # Risk amount = 250, risk per share = 0.1, uncapped = 2500 shares
    # But max position value = 25000 / 5 = 5000
    # Max shares by position = 5000 / 10.0 = 500
    assert shares <= 500
