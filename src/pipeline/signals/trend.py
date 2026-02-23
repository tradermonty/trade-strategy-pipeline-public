"""Trend filters — moving average alignment checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """Compute simple moving average."""
    return prices.rolling(window=window, min_periods=window).mean()


def trend_filter(
    prices: pd.Series,
    sma_short: int = 50,
    sma_long: int = 200,
) -> pd.Series:
    """Apply Minervini Stage 2 trend filter (legacy 3-condition version).

    Returns boolean Series: True where:
    - price > SMA(sma_long)
    - price > SMA(sma_short)
    - SMA(sma_short) > SMA(sma_long)
    """
    ma_short = compute_sma(prices, sma_short)
    ma_long = compute_sma(prices, sma_long)

    return (prices > ma_long) & (prices > ma_short) & (ma_short > ma_long)


def trend_template(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rs_rank: pd.Series | None = None,
    min_criteria: int = 6,
) -> pd.Series:
    """Minervini 7-Point Trend Template.

    Criteria:
      C1: Price > SMA150 AND Price > SMA200
      C2: SMA150 > SMA200
      C3: SMA200 trending up for 22 trading days
      C4: Price > SMA50
      C5: Price >= 25% above 52-week low
      C6: Price within 25% of 52-week high (i.e. >= 75% of 52w high)
      C7: RS rank > 70 (if provided; otherwise counts as met)

    Returns a boolean Series where True means the bar passes the template
    with at least *min_criteria* criteria met.
    """
    sma50 = compute_sma(close, 50)
    sma150 = compute_sma(close, 150)
    sma200 = compute_sma(close, 200)

    c1 = (close > sma150) & (close > sma200)
    c2 = sma150 > sma200
    c3 = sma200 > sma200.shift(22)
    c4 = close > sma50

    high_52w = high.rolling(252, min_periods=200).max()
    low_52w = low.rolling(252, min_periods=200).min()
    c5 = close >= low_52w * 1.25
    c6 = close >= high_52w * 0.75

    c7 = rs_rank > 70 if rs_rank is not None else pd.Series(True, index=close.index)

    criteria = [c1, c2, c3, c4, c5, c6, c7]
    score: pd.Series = sum(c.fillna(False).astype(int) for c in criteria)  # type: ignore[assignment]
    result: pd.Series = score >= min_criteria
    return result


def compute_rs_ranks(ohlcv: pd.DataFrame, period: int = 126) -> dict[str, pd.Series]:
    """Compute rolling cross-sectional RS percentile rank (0-99).

    At each date, computes each symbol's trailing return over ``period`` days,
    then ranks across all symbols that have data on that date.
    Returns ``{symbol: pd.Series(rank, index=dates)}``.
    """
    symbols = ohlcv["symbol"].unique()

    # Build a wide close-price DataFrame (date x symbol)
    close_wide = ohlcv.pivot_table(index="timestamp", columns="symbol", values="close")
    close_wide = close_wide.sort_index()

    # Rolling return for each symbol
    rolling_ret = close_wide / close_wide.shift(period) - 1.0

    # Cross-sectional percentile rank at each date (0-99 scale)
    ranked = rolling_ret.rank(axis=1, pct=True, na_option="keep") * 99.0
    ranked = ranked.apply(np.floor)

    result: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol in ranked.columns:
            result[symbol] = ranked[symbol]
        else:
            result[symbol] = pd.Series(dtype=float)

    return result
