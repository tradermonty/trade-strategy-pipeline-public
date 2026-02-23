"""Tests for performance metrics calculation."""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation.metrics import (
    compute_max_drawdown,
    compute_metrics,
    compute_profit_factor,
    compute_sharpe,
    compute_trade_count,
    compute_win_rate,
)


def test_sharpe_ratio_calculation() -> None:
    """Sharpe ratio should be annualized return / annualized volatility."""
    # 252 days of 0.1% daily return with 1% daily vol
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.01, 252))
    sharpe = compute_sharpe(returns)
    # Expected: ~0.001/0.01 * sqrt(252) ≈ 1.59
    assert 0.5 < sharpe < 3.0


def test_sharpe_ratio_zero_std() -> None:
    """Sharpe with zero volatility should return 0 or inf gracefully."""
    returns = pd.Series([0.001] * 252)
    sharpe = compute_sharpe(returns)
    # All same returns = zero std, should handle gracefully
    assert sharpe == 0.0 or np.isinf(sharpe) or sharpe > 100


def test_profit_factor_calculation() -> None:
    """Profit factor = gross profit / gross loss."""
    pnls = [100.0, 200.0, -50.0, -100.0, 150.0]
    pf = compute_profit_factor(pnls)
    # Gross profit = 450, gross loss = 150
    assert pf == pytest.approx(3.0)


def test_profit_factor_no_losses() -> None:
    """Profit factor with no losses should be inf."""
    pnls = [100.0, 200.0, 300.0]
    pf = compute_profit_factor(pnls)
    assert np.isinf(pf) or pf > 100


def test_max_drawdown_calculation() -> None:
    """Max drawdown should measure peak-to-trough decline."""
    equity = pd.Series([100, 110, 105, 90, 95, 100])
    dd = compute_max_drawdown(equity)
    # Peak = 110, trough = 90, drawdown = 20/110 ≈ 18.18%
    assert dd == pytest.approx(20 / 110, rel=0.01)


def test_trade_count() -> None:
    """Trade count should return number of completed trades."""
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}]
    assert compute_trade_count(trades) == 3


def test_win_rate() -> None:
    """Win rate = winning trades / total trades."""
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -10}]
    wr = compute_win_rate(trades)
    assert wr == pytest.approx(0.50)


def test_compute_all_metrics() -> None:
    """compute_metrics should return dict with all standard metrics."""
    equity = pd.Series(np.linspace(25000, 30000, 252))
    trades = [
        {"pnl": 500},
        {"pnl": -200},
        {"pnl": 300},
        {"pnl": 400},
        {"pnl": -100},
    ]
    metrics = compute_metrics(equity, trades)
    assert "sharpe" in metrics
    assert "profit_factor" in metrics
    assert "max_drawdown" in metrics
    assert "trade_count" in metrics
    assert "win_rate" in metrics
