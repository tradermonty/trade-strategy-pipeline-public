"""Chart generation for backtest reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless rendering

import matplotlib.pyplot as plt
import pandas as pd


def generate_equity_chart(equity: pd.Series, output_path: Path) -> None:
    """Generate equity curve chart as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.tolist(), linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def generate_drawdown_chart(equity: pd.Series, output_path: Path) -> None:
    """Generate drawdown chart as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(drawdown)), drawdown.tolist(), 0, alpha=0.4, color="red")
    ax.plot(drawdown.tolist(), color="red", linewidth=1)
    ax.set_title("Drawdown")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def generate_trade_distribution(trades: list[dict[str, Any]], output_path: Path) -> None:
    """Generate trade PnL distribution histogram as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pnls = [t["pnl"] for t in trades]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pnls, bins=max(10, len(pnls) // 5), edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Trade PnL Distribution")
    ax.set_xlabel("PnL ($)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
