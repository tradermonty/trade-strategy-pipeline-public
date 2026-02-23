"""Tests for trend filters and RS rank computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.signals.trend import compute_rs_ranks


def _make_ohlcv(n_days: int = 200, n_symbols: int = 5) -> pd.DataFrame:
    """Create synthetic multi-symbol OHLCV data."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for i in range(n_symbols):
        symbol = f"SYM_{i}"
        base = 100.0 + i * 10
        close = base + np.cumsum(rng.normal(0.1 * (i + 1), 1.0, n_days))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": rng.integers(100_000, 1_000_000, n_days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_rs_ranks_returns_series() -> None:
    """compute_rs_ranks should return dict[str, pd.Series]."""
    ohlcv = _make_ohlcv(n_days=200, n_symbols=3)
    result = compute_rs_ranks(ohlcv, period=126)
    assert isinstance(result, dict)
    assert len(result) == 3
    for symbol, series in result.items():
        assert isinstance(series, pd.Series), f"{symbol} should be pd.Series"


def test_rs_ranks_no_lookahead() -> None:
    """RS rank at date T should not depend on data after T."""
    ohlcv = _make_ohlcv(n_days=200, n_symbols=3)
    full_ranks = compute_rs_ranks(ohlcv, period=126)

    # Truncate data to first 180 days
    dates = sorted(ohlcv["timestamp"].unique())
    cutoff = dates[179]
    truncated = ohlcv[ohlcv["timestamp"] <= cutoff]
    trunc_ranks = compute_rs_ranks(truncated, period=126)

    # Ranks at dates before cutoff should be identical
    for symbol in full_ranks:
        full_s = full_ranks[symbol].dropna()
        trunc_s = trunc_ranks[symbol].dropna()
        # Only compare dates that exist in both
        common = full_s.index.intersection(trunc_s.index)
        if len(common) > 0:
            pd.testing.assert_series_equal(
                full_s.loc[common],
                trunc_s.loc[common],
                check_names=False,
            )


def test_rs_ranks_period_insufficient() -> None:
    """Symbols with fewer than `period` bars should have NaN ranks."""
    ohlcv = _make_ohlcv(n_days=50, n_symbols=2)
    result = compute_rs_ranks(ohlcv, period=126)
    for series in result.values():
        # All values should be NaN because 50 < 126
        assert series.dropna().empty, "Expected all NaN for insufficient data"
