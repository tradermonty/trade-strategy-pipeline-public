"""Tests for chart generation."""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.reporting.charts import (
    generate_drawdown_chart,
    generate_equity_chart,
    generate_trade_distribution,
)


def test_generate_equity_chart(tmp_path: Path) -> None:
    """Equity chart should produce a non-zero PNG file."""
    equity = pd.Series(np.linspace(25000, 30000, 252))
    output = tmp_path / "equity.png"
    generate_equity_chart(equity, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_generate_drawdown_chart(tmp_path: Path) -> None:
    """Drawdown chart should produce a non-zero PNG file."""
    equity = pd.Series(np.linspace(25000, 30000, 252))
    output = tmp_path / "drawdown.png"
    generate_drawdown_chart(equity, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_generate_trade_distribution_chart(tmp_path: Path) -> None:
    """Trade distribution chart should produce a non-zero PNG file."""
    trades = [
        {"pnl": 100},
        {"pnl": -50},
        {"pnl": 200},
        {"pnl": -30},
        {"pnl": 150},
        {"pnl": -80},
    ]
    output = tmp_path / "trade_dist.png"
    generate_trade_distribution(trades, output)
    assert output.exists()
    assert output.stat().st_size > 0
