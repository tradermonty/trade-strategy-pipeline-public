"""Tests for weekly trailing stop module."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.backtest.weekly_trailing import (
    aggregate_to_weekly,
    check_weekly_trailing_stop,
    compute_weekly_ema,
    compute_weekly_nweek_low,
    count_completed_weeks_since,
    is_week_end,
)


def _make_daily_bars(n_weeks: int = 10) -> pd.DataFrame:
    """Create synthetic daily bars spanning n_weeks (5 trading days/week)."""
    dates = pd.bdate_range("2024-01-02", periods=n_weeks * 5)
    n = len(dates)
    # Simple uptrend: close rises from 100 to ~100 + n*0.5
    closes = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [c - 0.3 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        }
    )


class TestAggregateToWeekly:
    def test_correct_number_of_weeks(self) -> None:
        """Should produce approximately n_weeks weekly bars."""
        df = _make_daily_bars(10)
        weekly = aggregate_to_weekly(df)
        # bdate_range may cross ISO week boundaries slightly differently
        assert len(weekly) >= 9
        assert len(weekly) <= 12

    def test_weekly_high_is_max(self) -> None:
        """Weekly high should be max of daily highs."""
        df = _make_daily_bars(4)
        weekly = aggregate_to_weekly(df)
        # Check first week
        first_week_end = weekly.iloc[0]["week_end"]
        first_week_start = weekly.iloc[0]["week_start"]
        daily_in_week = df[
            (pd.to_datetime(df["timestamp"]) >= first_week_start)
            & (pd.to_datetime(df["timestamp"]) <= first_week_end)
        ]
        assert weekly.iloc[0]["high"] == pytest.approx(daily_in_week["high"].max())

    def test_weekly_low_is_min(self) -> None:
        """Weekly low should be min of daily lows."""
        df = _make_daily_bars(4)
        weekly = aggregate_to_weekly(df)
        first_week_end = weekly.iloc[0]["week_end"]
        first_week_start = weekly.iloc[0]["week_start"]
        daily_in_week = df[
            (pd.to_datetime(df["timestamp"]) >= first_week_start)
            & (pd.to_datetime(df["timestamp"]) <= first_week_end)
        ]
        assert weekly.iloc[0]["low"] == pytest.approx(daily_in_week["low"].min())

    def test_weekly_close_is_last(self) -> None:
        """Weekly close should be the last day's close."""
        df = _make_daily_bars(4)
        weekly = aggregate_to_weekly(df)
        first_week_end = weekly.iloc[0]["week_end"]
        first_week_start = weekly.iloc[0]["week_start"]
        daily_in_week = df[
            (pd.to_datetime(df["timestamp"]) >= first_week_start)
            & (pd.to_datetime(df["timestamp"]) <= first_week_end)
        ]
        assert weekly.iloc[0]["close"] == pytest.approx(daily_in_week["close"].iloc[-1])


class TestComputeWeeklyEMA:
    def test_ema_length(self) -> None:
        """EMA series should be same length as input."""
        closes = pd.Series([100.0 + i for i in range(20)])
        ema = compute_weekly_ema(closes, period=10)
        assert len(ema) == len(closes)

    def test_ema_nan_before_period(self) -> None:
        """EMA should be NaN before min_periods is reached."""
        closes = pd.Series([100.0 + i for i in range(20)])
        ema = compute_weekly_ema(closes, period=10)
        assert pd.isna(ema.iloc[5])
        assert not pd.isna(ema.iloc[9])

    def test_ema_follows_trend(self) -> None:
        """In an uptrend, EMA should be below latest close."""
        closes = pd.Series([100.0 + i * 2.0 for i in range(20)])
        ema = compute_weekly_ema(closes, period=5)
        assert ema.iloc[-1] < closes.iloc[-1]


class TestComputeWeeklyNweekLow:
    def test_excludes_current_week(self) -> None:
        """N-week low should not include the current week's data."""
        lows = pd.Series([100.0, 98.0, 97.0, 96.0, 50.0])
        nweek_low = compute_weekly_nweek_low(lows, period=4)
        # At index 4, should look at shifted values [NaN, 100, 98, 97, 96]
        # Rolling min of [97, 96] (last 4 of shifted[1:5]) = 96.0
        assert nweek_low.iloc[4] == pytest.approx(96.0)

    def test_length_matches_input(self) -> None:
        """Output should match input length."""
        lows = pd.Series([100.0 - i for i in range(15)])
        result = compute_weekly_nweek_low(lows, period=4)
        assert len(result) == len(lows)


class TestIsWeekEnd:
    def test_friday_is_week_end(self) -> None:
        """Friday should be week end when next bar is Monday."""
        # 2024-01-02 is Tuesday, 2024-01-05 is Friday
        dates = pd.bdate_range("2024-01-02", periods=10)
        ts = pd.Series(dates)
        # Index 3 = Friday 2024-01-05, Index 4 = Monday 2024-01-08
        assert is_week_end(ts, 3) is True
        # Index 2 = Thursday, not week end
        assert is_week_end(ts, 2) is False

    def test_last_bar_is_week_end(self) -> None:
        """Last bar in series should always be week end."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        ts = pd.Series(dates)
        assert is_week_end(ts, len(ts) - 1) is True


class TestCountCompletedWeeks:
    def test_counts_correctly(self) -> None:
        """Should count only weeks whose week_end is between entry and current."""
        df = _make_daily_bars(10)
        weekly = aggregate_to_weekly(df)
        entry = pd.Timestamp("2024-01-02")
        # 4 weeks later
        current = pd.Timestamp("2024-02-02")
        count = count_completed_weeks_since(weekly, entry, current)
        assert count >= 3
        assert count <= 5


class TestCheckWeeklyTrailingStop:
    def test_no_trigger_during_transition(self) -> None:
        """Should not trigger during transition period."""
        df = _make_daily_bars(10)
        weekly = aggregate_to_weekly(df)
        entry = pd.Timestamp("2024-01-02")
        # Only 1 week later, transition_weeks=3
        current = pd.Timestamp("2024-01-09")
        result = check_weekly_trailing_stop(
            close_price=50.0,  # well below any EMA
            weekly_df=weekly,
            entry_date=entry,
            current_date=current,
            transition_weeks=3,
        )
        assert result is False

    def test_trigger_after_transition_price_below_ema(self) -> None:
        """Should trigger when close < weekly EMA after transition."""
        df = _make_daily_bars(20)
        weekly = aggregate_to_weekly(df)
        entry = pd.Timestamp("2024-01-02")
        current = pd.Timestamp("2024-03-15")
        result = check_weekly_trailing_stop(
            close_price=50.0,  # well below EMA
            weekly_df=weekly,
            entry_date=entry,
            current_date=current,
            ema_period=10,
            transition_weeks=3,
        )
        assert result is True

    def test_no_trigger_price_above_ema(self) -> None:
        """Should not trigger when close > weekly EMA."""
        df = _make_daily_bars(20)
        weekly = aggregate_to_weekly(df)
        entry = pd.Timestamp("2024-01-02")
        current = pd.Timestamp("2024-03-15")
        result = check_weekly_trailing_stop(
            close_price=500.0,  # well above EMA
            weekly_df=weekly,
            entry_date=entry,
            current_date=current,
            ema_period=10,
            transition_weeks=3,
        )
        assert result is False

    def test_nweek_low_mode(self) -> None:
        """Should work with weekly_nweek_low mode."""
        df = _make_daily_bars(20)
        weekly = aggregate_to_weekly(df)
        entry = pd.Timestamp("2024-01-02")
        current = pd.Timestamp("2024-03-15")
        result = check_weekly_trailing_stop(
            close_price=50.0,  # well below N-week low
            weekly_df=weekly,
            entry_date=entry,
            current_date=current,
            trailing_type="weekly_nweek_low",
            nweek_low_period=4,
            transition_weeks=3,
        )
        assert result is True
