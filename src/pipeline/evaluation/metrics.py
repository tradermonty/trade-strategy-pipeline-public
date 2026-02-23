"""Performance metrics calculation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Compute annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def compute_profit_factor(pnls: list[float] | list[dict[str, Any]]) -> float:
    """Compute profit factor = gross profit / gross loss."""
    values: list[float]
    if pnls and isinstance(pnls[0], dict):
        dict_pnls = cast(list[dict[str, Any]], pnls)
        values = [t["pnl"] for t in dict_pnls]
    else:
        values = cast(list[float], list(pnls))

    gross_profit = sum(v for v in values if v > 0)
    gross_loss = abs(sum(v for v in values if v < 0))

    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


def compute_max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown as a fraction."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(abs(drawdown.min()))


def compute_trade_count(trades: list[dict[str, Any]]) -> int:
    """Count number of completed trades."""
    return len(trades)


def compute_win_rate(trades: list[dict[str, Any]]) -> float:
    """Compute win rate = winning trades / total trades."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t["pnl"] > 0)
    return winners / len(trades)


def compute_metrics(equity: pd.Series, trades: list[dict[str, Any]]) -> dict[str, float]:
    """Compute all standard performance metrics."""
    returns = equity.pct_change().dropna()
    pnls = [t["pnl"] for t in trades] if trades else [0.0]

    return {
        "sharpe": compute_sharpe(returns),
        "profit_factor": compute_profit_factor(pnls) if trades else 0.0,
        "max_drawdown": compute_max_drawdown(equity),
        "trade_count": compute_trade_count(trades),
        "win_rate": compute_win_rate(trades),
    }
